#!/usr/bin/env python3
"""Convert h0 (time, lat, lon) forcing into (time, gridcell) files for fast extraction.

Default mode loads each source variable once into memory, gathers land cells with
numpy advanced indexing, and writes a compact (time, gridcell) NetCDF. This is
much faster than per-latitude slice reads from the compressed 3D source file.

Example:
  python scripts/preprocess_h0_forcing_gridcell.py \
    --config-input config/CNP_dataInput_h0_forcing.txt

  python scripts/preprocess_h0_forcing_gridcell.py --overwrite --mode memory
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config

H0_SOURCE_VARS = {
    "FLDS": "FLDS",
    "PSRF": "PBOT",
    "FSDS": "FSDS",
    "QBOT": "QBOT",
    "PRECTmms": "PRECmms",
    "TBOT": "TBOT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-input", type=str, default=None)
    parser.add_argument("--forcing-nc", type=str, default=None)
    parser.add_argument("--index-master", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--vars",
        nargs="+",
        default=list(H0_SOURCE_VARS.keys()),
        help="Training variable names to preprocess.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "memory", "stream"),
        default="auto",
        help="auto: try memory gather, fall back to stream; memory: one full read per var; "
        "stream: per-latitude reads with incremental writes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def default_paths(cfg_path: str | None) -> tuple[str, str, str]:
    if cfg_path:
        config.load_config(cfg_path)
    forcing_nc = config.FILE_PATHS["ds4"]
    index_master = os.path.join(
        config.ARTIFACT_ROOT, "A_index_core", "index_master.pkl"
    )
    output_dir = os.path.join(config.BASE_OUTPUT_ROOT, "forcing_gridcell_preprocessed")
    return forcing_nc, index_master, output_dir


def _lat_lon_indices(index_df: pd.DataFrame, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    if "lat_idx" in index_df.columns and "lon_idx" in index_df.columns:
        lat_idx = index_df["lat_idx"].to_numpy(dtype=np.int64)
        lon_idx = index_df["lon_idx"].to_numpy(dtype=np.int64)
    else:
        flat = index_df["nearest_forcing_index"].to_numpy(dtype=np.int64)
        lat_idx = (flat // n_lon).astype(np.int64)
        lon_idx = (flat % n_lon).astype(np.int64)
    return lat_idx, lon_idx


def _positions_by_lat(lat_idx: np.ndarray, lon_idx: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    positions_by_lat: dict[int, np.ndarray] = {}
    lons_by_lat: dict[int, np.ndarray] = {}
    for lat in np.unique(lat_idx):
        pos = np.where(lat_idx == lat)[0]
        positions_by_lat[int(lat)] = pos.astype(np.int64)
        lons_by_lat[int(lat)] = lon_idx[pos]
    return positions_by_lat, lons_by_lat


def _create_output_dataset(
    tmp_path: str,
    src: nc.Dataset,
    index_df: pd.DataFrame,
    train_var: str,
    source_var: nc.Variable,
    n_time: int,
    n_cells: int,
) -> tuple[nc.Dataset, nc.Variable]:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    dst = nc.Dataset(tmp_path, "w", format="NETCDF4")
    dst.createDimension("time", n_time)
    dst.createDimension("gridcell", n_cells)

    if "time" in src.variables:
        tcoord = dst.createVariable("time", src.variables["time"].datatype, ("time",))
        tcoord[:] = src.variables["time"][:]
        for attr in src.variables["time"].ncattrs():
            if attr != "_FillValue":
                tcoord.setncattr(attr, src.variables["time"].getncattr(attr))

    lat_out = dst.createVariable("LATIXY", "f4", ("gridcell",), zlib=True, complevel=1)
    lon_out = dst.createVariable("LONGXY", "f4", ("gridcell",), zlib=True, complevel=1)
    lat_out[:] = index_df["Latitude"].to_numpy(dtype=np.float32)
    lon_out[:] = index_df["Longitude"].to_numpy(dtype=np.float32)

    chunk_time = min(n_time, 240)
    chunk_cells = min(n_cells, 8192)
    out_var = dst.createVariable(
        train_var,
        "f4",
        ("time", "gridcell"),
        zlib=True,
        complevel=1,
        chunksizes=(chunk_time, chunk_cells),
    )
    for attr in source_var.ncattrs():
        if attr != "_FillValue":
            out_var.setncattr(attr, source_var.getncattr(attr))
    if train_var != source_var.name:
        out_var.source_variable = source_var.name

    dst.history = (
        "Preprocessed from h0 forcing by preprocess_h0_forcing_gridcell.py; "
        "gridcell axis follows index_master row order."
    )
    return dst, out_var


def _read_full_variable(src_var: nc.Variable) -> np.ndarray:
    log(f"  loading full array {src_var.name} {src_var.shape} ...")
    t0 = time.time()
    data = np.asarray(src_var[:], dtype=np.float32)
    log(f"  loaded {src_var.name} in {time.time() - t0:.1f}s ({data.nbytes / 1e9:.2f} GB)")
    return data


def _gather_land_cells(full: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    t0 = time.time()
    gathered = full[:, lat_idx, lon_idx]
    log(f"  gathered land cells {gathered.shape} in {time.time() - t0:.1f}s")
    return np.asarray(gathered, dtype=np.float32)


def _write_output(dst: nc.Dataset, out_var: nc.Variable, data: np.ndarray) -> None:
    t0 = time.time()
    out_var[:] = data
    dst.sync()
    log(f"  wrote output in {time.time() - t0:.1f}s")


def preprocess_variable_memory(
    forcing_nc: str,
    output_nc: str,
    index_df: pd.DataFrame,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    train_var: str,
    source_var: str,
) -> None:
    with nc.Dataset(forcing_nc, "r") as src:
        if source_var not in src.variables:
            raise KeyError(f"{source_var} missing in {forcing_nc}")
        src_var = src.variables[source_var]
        if src_var.ndim != 3 or tuple(d.lower() for d in src_var.dimensions) != ("time", "lat", "lon"):
            raise ValueError(
                f"Expected (time, lat, lon) for {source_var}, got {src_var.dimensions}"
            )

        n_time = int(src_var.shape[0])
        n_cells = len(index_df)
        tmp_path = output_nc + ".tmp"
        dst, out_var = _create_output_dataset(
            tmp_path, src, index_df, train_var, src_var, n_time, n_cells
        )
        try:
            full = _read_full_variable(src_var)
            gathered = _gather_land_cells(full, lat_idx, lon_idx)
            del full
            _write_output(dst, out_var, gathered)
        finally:
            dst.close()

    os.replace(tmp_path, output_nc)


def preprocess_variable_stream(
    forcing_nc: str,
    output_nc: str,
    index_df: pd.DataFrame,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    train_var: str,
    source_var: str,
) -> None:
    positions_by_lat, lons_by_lat = _positions_by_lat(lat_idx, lon_idx)
    tmp_path = output_nc + ".tmp"

    with nc.Dataset(forcing_nc, "r") as src:
        if source_var not in src.variables:
            raise KeyError(f"{source_var} missing in {forcing_nc}")
        src_var = src.variables[source_var]
        n_time = int(src_var.shape[0])
        n_cells = len(index_df)
        n_lat = int(src_var.shape[1])

        dst, out_var = _create_output_dataset(
            tmp_path, src, index_df, train_var, src_var, n_time, n_cells
        )
        try:
            t0 = time.time()
            active_lats = sorted(positions_by_lat)
            for i, lat in enumerate(active_lats, start=1):
                pos = positions_by_lat[lat]
                lons = lons_by_lat[lat]
                slab = np.asarray(src_var[:, lat, lons], dtype=np.float32)
                out_var[:, pos] = slab
                if i == 1 or i == len(active_lats) or i % 10 == 0:
                    elapsed = time.time() - t0
                    log(
                        f"  [{train_var}] stream lat {lat}/{n_lat - 1} "
                        f"({i}/{len(active_lats)} active rows, {elapsed:.1f}s elapsed)"
                    )
            dst.sync()
        finally:
            dst.close()

    os.replace(tmp_path, output_nc)


def preprocess_variable(
    forcing_nc: str,
    output_nc: str,
    index_df: pd.DataFrame,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    train_var: str,
    source_var: str,
    *,
    mode: str,
    overwrite: bool,
) -> None:
    if os.path.exists(output_nc) and not overwrite:
        log(f"[skip] {output_nc} exists")
        return

    os.makedirs(os.path.dirname(output_nc), exist_ok=True)
    tmp_path = output_nc + ".tmp"
    if overwrite and os.path.exists(tmp_path):
        os.remove(tmp_path)

    t0 = time.time()
    chosen_mode = mode
    if mode == "auto":
        chosen_mode = "memory"

    log(f"[preprocess] {train_var} (source={source_var}, mode={chosen_mode})")
    try:
        if chosen_mode == "memory":
            preprocess_variable_memory(
                forcing_nc, output_nc, index_df, lat_idx, lon_idx, train_var, source_var
            )
        else:
            preprocess_variable_stream(
                forcing_nc, output_nc, index_df, lat_idx, lon_idx, train_var, source_var
            )
    except MemoryError:
        if mode == "auto":
            log(f"[warn] memory mode failed for {train_var}; falling back to stream mode")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            preprocess_variable_stream(
                forcing_nc, output_nc, index_df, lat_idx, lon_idx, train_var, source_var
            )
        else:
            raise

    log(f"[done] {train_var} -> {output_nc} ({time.time() - t0:.1f}s total)")


def main() -> None:
    args = parse_args()
    forcing_nc, index_master, output_dir = default_paths(args.config_input)
    if args.forcing_nc:
        forcing_nc = args.forcing_nc
    if args.index_master:
        index_master = args.index_master
    if args.output_dir:
        output_dir = args.output_dir

    if not os.path.exists(forcing_nc):
        raise FileNotFoundError(forcing_nc)
    if not os.path.exists(index_master):
        raise FileNotFoundError(index_master)

    index_df = pd.read_pickle(index_master).sort_values("__row_id").reset_index(drop=True)
    lat_idx, lon_idx = _lat_lon_indices(index_df, n_lon=1440)

    log(f"forcing_nc: {forcing_nc}")
    log(f"index_master: {index_master} ({len(index_df)} cells)")
    log(f"output_dir: {output_dir}")
    log(f"mode: {args.mode}")

    total_t0 = time.time()
    for train_var in args.vars:
        source_var = H0_SOURCE_VARS.get(train_var, train_var)
        output_nc = os.path.join(output_dir, f"{train_var}_gridcell.nc")
        preprocess_variable(
            forcing_nc,
            output_nc,
            index_df,
            lat_idx,
            lon_idx,
            train_var,
            source_var,
            mode=args.mode,
            overwrite=args.overwrite,
        )

    log(f"All variables preprocessed in {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
