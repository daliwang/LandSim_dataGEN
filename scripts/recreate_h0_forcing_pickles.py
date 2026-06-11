#!/usr/bin/env python3
"""Create new training pickle files by replacing only atmospheric forcing.

This script reuses an existing final training dataset as the template for all
non-forcing columns. For each `training_data_batch_*.pkl`, it reads row
Latitude/Longitude, extracts the nearest 240-month h0-derived forcing series,
replaces the forcing columns, and writes a new pickle file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OLD_FINAL_DIR = DEFAULT_PROJECT_ROOT / "output_v1_with_trigridinterepartion" / "final_dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "output_h0_forcing_0001_0020" / "final_dataset"
DEFAULT_RECORD_DIR = DEFAULT_PROJECT_ROOT / "records"
DEFAULT_FORCING_FILE = Path(
    "/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/"
    "forcing_TBOT_PBOT_QBOT_FLDS_FSDS_RAIN_SNOW_PRECmms_0001-0020.nc"
)

FORCING_MAP = {
    "FLDS": "FLDS",
    "PSRF": "PBOT",
    "FSDS": "FSDS",
    "QBOT": "QBOT",
    "PRECTmms": "PRECmms",
    "TBOT": "TBOT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace forcing columns in existing training_data_batch_*.pkl files."
    )
    parser.add_argument("--old-final-dir", type=Path, default=DEFAULT_OLD_FINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forcing-file", type=Path, default=DEFAULT_FORCING_FILE)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument(
        "--pattern",
        default="training_data_batch_*.pkl",
        help="Input pickle filename pattern under --old-final-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output pickle files.",
    )
    return parser.parse_args()


def nearest_indices(axis_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    """Return nearest index on a monotonic 1D coordinate axis."""
    axis_values = np.asarray(axis_values, dtype=float)
    query_values = np.asarray(query_values, dtype=float)

    order = np.argsort(axis_values)
    sorted_values = axis_values[order]
    positions = np.searchsorted(sorted_values, query_values, side="left")
    positions = np.clip(positions, 0, len(sorted_values) - 1)

    left_positions = np.clip(positions - 1, 0, len(sorted_values) - 1)
    right_positions = positions
    use_left = np.abs(query_values - sorted_values[left_positions]) <= np.abs(
        query_values - sorted_values[right_positions]
    )
    nearest_sorted_positions = np.where(use_left, left_positions, right_positions)
    return order[nearest_sorted_positions].astype(np.int64)


def normalize_longitudes(query_lons: np.ndarray, lon_axis: np.ndarray) -> np.ndarray:
    """Normalize query longitudes to match either 0..360 or -180..180 axes."""
    query_lons = np.asarray(query_lons, dtype=float).copy()
    lon_min = float(np.nanmin(lon_axis))
    lon_max = float(np.nanmax(lon_axis))

    if lon_min >= 0.0 and np.nanmin(query_lons) < 0.0:
        query_lons = query_lons % 360.0
    elif lon_max <= 180.0 and np.nanmax(query_lons) > 180.0:
        query_lons = ((query_lons + 180.0) % 360.0) - 180.0

    return query_lons


def extract_series(var, lat_indices: np.ndarray, lon_indices: np.ndarray) -> list[list[float]]:
    """Extract time series for all requested row points from a time/lat/lon variable."""
    if var.dimensions != ("time", "lat", "lon"):
        raise ValueError(f"{var.name} must have dimensions ('time', 'lat', 'lon'), got {var.dimensions}")

    series_by_row = []
    for lat_idx, lon_idx in zip(lat_indices, lon_indices):
        values = np.asarray(var[:, int(lat_idx), int(lon_idx)], dtype=np.float32)
        series_by_row.append(values.reshape(-1).astype(float).tolist())
    return series_by_row


def replace_forcing_in_batch(df: pd.DataFrame, ds: nc.Dataset) -> pd.DataFrame:
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise KeyError("Input pickle must include Latitude and Longitude columns.")

    lat_axis = np.asarray(ds.variables["lat"][:], dtype=float)
    lon_axis = np.asarray(ds.variables["lon"][:], dtype=float)

    lat_indices = nearest_indices(lat_axis, df["Latitude"].to_numpy(dtype=float))
    query_lons = normalize_longitudes(df["Longitude"].to_numpy(dtype=float), lon_axis)
    lon_indices = nearest_indices(lon_axis, query_lons)

    out = df.copy()
    for output_name, source_name in FORCING_MAP.items():
        if source_name not in ds.variables:
            raise KeyError(f"Source forcing variable {source_name} not found in {ds.filepath()}")
        out[output_name] = extract_series(ds.variables[source_name], lat_indices, lon_indices)

    return out


def write_manifest(args: argparse.Namespace, input_files: list[Path], output_files: list[Path]) -> None:
    args.record_dir.mkdir(parents=True, exist_ok=True)
    forcing_stat = args.forcing_file.stat()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "old_final_dir": str(args.old_final_dir),
        "output_dir": str(args.output_dir),
        "forcing_file": str(args.forcing_file),
        "forcing_file_size": forcing_stat.st_size,
        "forcing_file_mtime": forcing_stat.st_mtime,
        "variable_mapping": FORCING_MAP,
        "input_pickles": [str(path) for path in input_files],
        "output_pickles": [str(path) for path in output_files],
    }
    manifest_path = args.record_dir / "h0_forcing_pickle_recreation_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Record] wrote {manifest_path}")


def main() -> None:
    args = parse_args()
    input_files = sorted(args.old_final_dir.glob(args.pattern))
    if not input_files:
        raise FileNotFoundError(f"No input pickle files found: {args.old_final_dir / args.pattern}")
    if not args.forcing_file.exists():
        raise FileNotFoundError(f"Forcing file not found: {args.forcing_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    print(f"[Input] old final pickles: {args.old_final_dir}")
    print(f"[Input] h0 forcing: {args.forcing_file}")
    print(f"[Output] new final pickles: {args.output_dir}")

    with nc.Dataset(args.forcing_file) as ds:
        for input_path in input_files:
            output_path = args.output_dir / input_path.name
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")

            print(f"[Batch] {input_path.name}")
            df = pd.read_pickle(input_path)
            updated = replace_forcing_in_batch(df, ds)
            updated.to_pickle(output_path)
            output_files.append(output_path)
            print(f"[Batch] wrote {output_path}")

    write_manifest(args, input_files, output_files)
    print("[Done] recreated forcing columns in final pickle files.")


if __name__ == "__main__":
    main()
