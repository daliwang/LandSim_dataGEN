#!/usr/bin/env python3
import argparse
import datetime
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config

try:
    import xarray as xr
except Exception:
    xr = None

try:
    import cftime
except Exception:
    cftime = None


def ensure_dirs() -> None:
    os.makedirs(config.ARTIFACT_ROOT, exist_ok=True)
    os.makedirs(config.MANIFEST_ROOT, exist_ok=True)
    os.makedirs(config.FINAL_OUTPUT_DIR, exist_ok=True)
    for module in config.ALL_MODULES:
        os.makedirs(module_dir(module), exist_ok=True)


def module_dir(module_name: str) -> str:
    return os.path.join(config.ARTIFACT_ROOT, module_name)


def module_manifest_path(module_name: str) -> str:
    return os.path.join(config.MANIFEST_ROOT, f"{module_name}.json")


def save_manifest(module_name: str, payload: Dict) -> None:
    with open(module_manifest_path(module_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def source_meta(path: str) -> Dict:
    st = os.stat(path)
    return {"path": path, "mtime": st.st_mtime, "size": st.st_size}


def _read_forcing_coords_2d(ds_forcing: nc.Dataset) -> np.ndarray:
    """
    Read forcing coordinates in a memory-safe way.
    Some forcing files store LATIXY/LONGXY with a time dimension; we only need
    one spatial slice for nearest-neighbor mapping in A_index_core.
    """
    if "LATIXY" not in ds_forcing.variables or "LONGXY" not in ds_forcing.variables:
        raise KeyError("Forcing file must include LATIXY and LONGXY.")

    lat_var = ds_forcing.variables["LATIXY"]
    lon_var = ds_forcing.variables["LONGXY"]

    def _slice_without_time(var):
        # Build index tuple dynamically so we do not load full time stacks.
        idx = []
        for dim_name in var.dimensions:
            if dim_name.lower() == "time":
                idx.append(0)
            else:
                idx.append(slice(None))
        return np.asarray(var[tuple(idx)], dtype=float)

    lat_data = _slice_without_time(lat_var)
    lon_data = _slice_without_time(lon_var)

    if lat_data.ndim == 1 and lon_data.ndim == 1:
        return np.column_stack((lat_data, lon_data))
    if lat_data.ndim == 2 and lon_data.ndim == 2:
        return np.column_stack((lat_data.reshape(-1), lon_data.reshape(-1)))
    raise ValueError(f"Unsupported forcing coord shapes: {lat_data.shape}, {lon_data.shape}")


def _build_coord_key_map(coords: np.ndarray, decimals: int = 6) -> Dict[tuple, int]:
    """
    Build a coordinate->index map using rounded (lat, lon) keys.
    Duplicate rounded keys are rejected because they make direct mapping ambiguous.
    """
    coord_map: Dict[tuple, int] = {}
    for idx, (lat_val, lon_val) in enumerate(np.asarray(coords, dtype=float)):
        key = (round(float(lat_val), decimals), round(float(lon_val), decimals))
        if key in coord_map:
            raise ValueError(
                f"Duplicate rounded coordinate key detected at decimals={decimals}: {key}. "
                "Cannot build unambiguous direct same-mesh mapping."
            )
        coord_map[key] = int(idx)
    return coord_map


def _direct_same_mesh_indices(
    query_coords: np.ndarray,
    ds_restart: nc.Dataset,
    ds_forcing: nc.Dataset,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Directly map query coordinates to restart/forcing indices when datasets share
    the same mesh (DATM assumption). This avoids KD-tree nearest-neighbor search
    and enforces coordinate consistency.
    """
    restart_coords = np.vstack((ds_restart.variables["grid1d_lat"][:], ds_restart.variables["grid1d_lon"][:])).T
    forcing_coords = _read_forcing_coords_2d(ds_forcing)

    restart_map = _build_coord_key_map(restart_coords, decimals=decimals)
    forcing_map = _build_coord_key_map(forcing_coords, decimals=decimals)

    restart_indices = np.empty(len(query_coords), dtype=np.int64)
    forcing_indices = np.empty(len(query_coords), dtype=np.int64)
    tolerance = 10 ** (-decimals)

    for row_id, (lat_val, lon_val) in enumerate(np.asarray(query_coords, dtype=float)):
        key = (round(float(lat_val), decimals), round(float(lon_val), decimals))
        if key not in restart_map:
            raise ValueError(
                f"DATM same-mesh mapping failed: query point {key} not found in restart grid."
            )
        if key not in forcing_map:
            raise ValueError(
                f"DATM same-mesh mapping failed: query point {key} not found in forcing grid."
            )

        r_idx = int(restart_map[key])
        f_idx = int(forcing_map[key])
        restart_indices[row_id] = r_idx
        forcing_indices[row_id] = f_idx

        # Validate the mapped restart and forcing coordinates align.
        if (
            abs(float(restart_coords[r_idx, 0]) - float(forcing_coords[f_idx, 0])) > tolerance
            or abs(float(restart_coords[r_idx, 1]) - float(forcing_coords[f_idx, 1])) > tolerance
        ):
            raise ValueError(
                "DATM same-mesh mapping mismatch: restart and forcing coordinates differ "
                f"for key {key} (restart_idx={r_idx}, forcing_idx={f_idx})."
            )

    return restart_indices, forcing_indices


def load_index_master() -> pd.DataFrame:
    index_path = os.path.join(module_dir("A_index_core"), "index_master.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError("A_index_core/index_master.pkl not found. Build A_index_core first.")
    return pd.read_pickle(index_path)


def get_batch_ids(index_df: pd.DataFrame) -> List[int]:
    return sorted(index_df["batch_id"].unique().tolist())


def build_index_core() -> None:
    print("[A_index_core] building...")
    ds2 = nc.Dataset(config.FILE_PATHS["ds2"])
    ds10 = nc.Dataset(config.FILE_PATHS["ds10"])
    ds4 = nc.Dataset(config.FILE_PATHS["ds4"])
    try:
        lats = ds2.variables["lat"][:]
        lons = ds2.variables["lon"][:]
        landmask = ds2.variables["landmask"][:]

        lat_indices = np.where((lats >= config.LAT2) & (lats <= config.LAT1))[0]
        lon_indices = np.where((lons >= config.LON1) & (lons <= config.LON2))[0]

        cell_indices = None

        # Support both structured 2D (lat, lon) and unstructured 1D meshes.
        if landmask.ndim == 2:
            filtered_coordinates = [(i, j) for i in lat_indices for j in lon_indices if landmask[i, j] == 1]
            query_coords = np.array([(lats[i], lons[j]) for i, j in filtered_coordinates], dtype=float)
        elif landmask.ndim == 1:
            # Unstructured grid: lat/lon/landmask index the same gridcell axis.
            if lats.ndim != 1 or lons.ndim != 1 or lats.shape[0] != lons.shape[0] or landmask.shape[0] != lats.shape[0]:
                raise ValueError(
                    "Unsupported 1D mesh definition: expected lat/lon/landmask to be 1D and aligned."
                )
            cell_indices = np.where(
                (lats >= config.LAT2)
                & (lats <= config.LAT1)
                & (lons >= config.LON1)
                & (lons <= config.LON2)
                & (landmask == 1)
            )[0]
            filtered_coordinates = [(int(idx), int(idx)) for idx in cell_indices]
            query_coords = np.column_stack((lats[cell_indices], lons[cell_indices])).astype(float)
        else:
            raise ValueError(f"Unsupported landmask shape: {landmask.shape}")

        print(f"[A_index_core] selected cells after filtering: {len(filtered_coordinates)}")

        if config.FORCING_MODE == "datm":
            if landmask.ndim == 1 and cell_indices is not None:
                print("[A_index_core] DATM mode: using direct same-mesh index mapping (no KD-tree).")
                restart_coords = np.vstack((ds10.variables["grid1d_lat"][:], ds10.variables["grid1d_lon"][:])).T
                forcing_coords = _read_forcing_coords_2d(ds4)
                n_cells = int(lats.shape[0])
                if restart_coords.shape[0] != n_cells or forcing_coords.shape[0] != n_cells:
                    raise ValueError(
                        "DATM same-mesh index mapping requires ds2/ds10/ds4 to have aligned 1D grid size. "
                        f"Got ds2={n_cells}, ds10={restart_coords.shape[0]}, ds4={forcing_coords.shape[0]}."
                    )

                sel = np.asarray(cell_indices, dtype=np.int64)
                tol = 1e-6
                if sel.size > 0:
                    lat_diff_ds10 = np.abs(np.asarray(lats[sel], dtype=float) - np.asarray(restart_coords[sel, 0], dtype=float))
                    lon_diff_ds10 = np.abs(np.asarray(lons[sel], dtype=float) - np.asarray(restart_coords[sel, 1], dtype=float))
                    if np.max(lat_diff_ds10) > tol or np.max(lon_diff_ds10) > tol:
                        raise ValueError(
                            "DATM same-mesh validation failed between ds2 and ds10 coordinates: "
                            f"max_lat_diff={float(np.max(lat_diff_ds10)):.6g}, "
                            f"max_lon_diff={float(np.max(lon_diff_ds10)):.6g}"
                        )

                    lat_diff_ds4 = np.abs(np.asarray(lats[sel], dtype=float) - np.asarray(forcing_coords[sel, 0], dtype=float))
                    lon_diff_ds4 = np.abs(np.asarray(lons[sel], dtype=float) - np.asarray(forcing_coords[sel, 1], dtype=float))
                    if np.max(lat_diff_ds4) > tol or np.max(lon_diff_ds4) > tol:
                        print(
                            "[A_index_core] warning: ds4 coordinates do not match ds2 in DATM 1D mode: "
                            f"max_lat_diff={float(np.max(lat_diff_ds4)):.6g}, "
                            f"max_lon_diff={float(np.max(lon_diff_ds4)):.6g}. "
                            "Proceeding with direct same-index mapping; consider rebuilding forcing "
                            "intermediates to embed ds2 mesh coordinates."
                        )

                restart_indices = sel.copy()
                forcing_indices = sel.copy()
            else:
                print("[A_index_core] DATM mode: using direct same-mesh coordinate mapping (no KD-tree).")
                restart_indices, forcing_indices = _direct_same_mesh_indices(query_coords, ds10, ds4, decimals=6)
        else:
            restart_tree = cKDTree(np.vstack((ds10.variables["grid1d_lat"][:], ds10.variables["grid1d_lon"][:])).T)
            _, restart_indices = restart_tree.query(query_coords, k=1)

            forcing_coords = _read_forcing_coords_2d(ds4)
            forcing_tree = cKDTree(forcing_coords)
            _, forcing_indices = forcing_tree.query(query_coords, k=1)

        rows = []
        for row_id, (i, j) in enumerate(filtered_coordinates):
            rows.append(
                {
                    "__row_id": row_id,
                    "batch_id": (row_id // config.BATCH_SIZE) + 1,
                    "lat_idx": int(i),
                    "lon_idx": int(j),
                    "Latitude": float(lats[i]),
                    "Longitude": float(lons[j]),
                    "nearest_restart_index": int(restart_indices[row_id]),
                    "nearest_forcing_index": int(forcing_indices[row_id]),
                    "gridcell_id": int(restart_indices[row_id]) + 1,
                }
            )
        index_df = pd.DataFrame(rows)
        out_dir = module_dir("A_index_core")
        index_df.to_pickle(os.path.join(out_dir, "index_master.pkl"))

        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            batch_df.to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest(
            "A_index_core",
            {
                "module": "A_index_core",
                "sources": [
                    source_meta(config.FILE_PATHS["ds2"]),
                    source_meta(config.FILE_PATHS["ds10"]),
                    source_meta(config.FILE_PATHS["ds4"]),
                ],
                "rows": int(index_df.shape[0]),
                "batches": int(index_df["batch_id"].max()),
            },
        )
    finally:
        ds2.close()
        ds10.close()
        ds4.close()
    print("[A_index_core] done.")


def build_ds1_surface() -> None:
    print("[A_ds1_surface] building...")
    index_df = load_index_master()
    ds1 = nc.Dataset(config.FILE_PATHS["ds1"])
    try:
        out_dir = module_dir("A_ds1_surface")
        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            rows = {"__row_id": batch_df["__row_id"].tolist()}
            for var in config.STATIC_SURFACE_VARS_2D:
                arr = ds1.variables[var]
                if arr.ndim == 1:
                    rows[var] = [float(arr[i]) for i in batch_df["lat_idx"]]
                elif arr.ndim == 2:
                    rows[var] = [float(arr[i, j]) for i, j in zip(batch_df["lat_idx"], batch_df["lon_idx"])]
                else:
                    raise ValueError(f"Unsupported surface var shape for {var}: {arr.shape}")
            for var in config.STATIC_SURFACE_VARS_3D:
                arr = ds1.variables[var]
                if arr.ndim == 2:
                    rows[var] = [np.asarray(arr[:, i], dtype=float).tolist() for i in batch_df["lat_idx"]]
                elif arr.ndim == 3:
                    rows[var] = [
                        np.asarray(arr[:, i, j], dtype=float).tolist()
                        for i, j in zip(batch_df["lat_idx"], batch_df["lon_idx"])
                    ]
                else:
                    raise ValueError(f"Unsupported layered surface var shape for {var}: {arr.shape}")
            pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest("A_ds1_surface", {"module": "A_ds1_surface", "sources": [source_meta(config.FILE_PATHS["ds1"])]})
    finally:
        ds1.close()
    print("[A_ds1_surface] done.")


def build_ds2_history_x() -> None:
    print("[A_ds2_history_x] building...")
    index_df = load_index_master()
    ds2 = nc.Dataset(config.FILE_PATHS["ds2"])
    try:
        out_dir = module_dir("A_ds2_history_x")
        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            rows = {"__row_id": batch_df["__row_id"].tolist()}
            landfrac = ds2.variables["landfrac"]
            if landfrac.ndim == 1:
                rows["landfrac"] = [float(landfrac[i]) for i in batch_df["lat_idx"]]
            elif landfrac.ndim == 2:
                rows["landfrac"] = [float(landfrac[i, j]) for i, j in zip(batch_df["lat_idx"], batch_df["lon_idx"])]
            else:
                raise ValueError(f"Unsupported landfrac shape: {landfrac.shape}")
            for var in config.HISTORY_GRID_VARS_2D:
                arr = ds2.variables[var]
                if arr.ndim == 2:
                    rows[var] = [float(arr[0, i]) for i in batch_df["lat_idx"]]
                elif arr.ndim == 3:
                    rows[var] = [float(arr[0, i, j]) for i, j in zip(batch_df["lat_idx"], batch_df["lon_idx"])]
                else:
                    raise ValueError(f"Unsupported history var shape for {var}: {arr.shape}")
            pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest("A_ds2_history_x", {"module": "A_ds2_history_x", "sources": [source_meta(config.FILE_PATHS["ds2"])]})
    finally:
        ds2.close()
    print("[A_ds2_history_x] done.")


def build_h0_list_y() -> None:
    print("[A_h0_list_y] building...")
    index_df = load_index_master()
    h0_paths = config.FILE_PATHS["h0_list"]
    if not h0_paths:
        raise ValueError("FILE_PATHS['h0_list'] is empty. Please provide one h0 file path.")
    ds_h0 = nc.Dataset(h0_paths[0])
    try:
        out_dir = module_dir("A_h0_list_y")
        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            rows = {"__row_id": batch_df["__row_id"].tolist()}
            for var in config.HISTORY_GRID_VARS_2D:
                arr = ds_h0.variables[var]
                if arr.ndim == 2:
                    rows[f"Y_{var}"] = [float(arr[0, i]) for i in batch_df["lat_idx"]]
                elif arr.ndim == 3:
                    rows[f"Y_{var}"] = [float(arr[0, i, j]) for i, j in zip(batch_df["lat_idx"], batch_df["lon_idx"])]
                else:
                    raise ValueError(f"Unsupported target history var shape for {var}: {arr.shape}")
            pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest(
            "A_h0_list_y",
            {"module": "A_h0_list_y", "sources": [source_meta(h0_paths[0])]},
        )
    finally:
        ds_h0.close()
    print("[A_h0_list_y] done.")


def _resolve_datm_files(var_name: str) -> List[str]:
    direct_path = config.DATM_FORCING_PATHS.get(var_name, "").strip()
    if direct_path:
        if not os.path.exists(direct_path):
            raise FileNotFoundError(f"DATM forcing file not found for {var_name}: {direct_path}")
        return [direct_path]

    datm_root = (config.DATM_ROOT or "").strip()
    if not datm_root:
        raise ValueError(f"DATM_ROOT is empty and DATM_{var_name}_PATH is not set.")
    if not os.path.isdir(datm_root):
        raise FileNotFoundError(f"DATM_ROOT does not exist: {datm_root}")

    token = config.DATM_TOKEN_MAP.get(var_name, "").strip()
    if not token:
        raise ValueError(f"No DATM token configured for forcing variable: {var_name}")

    pattern = re.compile(rf"(?:.*_)?cl[i]?mforc\..*\.{re.escape(token)}\.(\d{{4}})-(\d{{2}})\.nc$")
    files_by_month: Dict[tuple, List[str]] = {}
    for root, _, names in os.walk(datm_root, followlinks=True):
        for name in names:
            match = pattern.match(name)
            if not match:
                continue
            year = int(match.group(1))
            month = int(match.group(2))
            if not (config.DATM_START_YEAR <= year <= config.DATM_END_YEAR):
                continue
            key = (year, month)
            files_by_month.setdefault(key, []).append(os.path.join(root, name))

    files: List[str] = []
    for year in range(config.DATM_START_YEAR, config.DATM_END_YEAR + 1):
        for month in range(1, 13):
            key = (year, month)
            candidates = sorted(files_by_month.get(key, []))
            if not candidates:
                print(f"[ForcingPrep] warning: missing DATM file for {var_name} {year}-{month:02d}")
                continue
            if len(candidates) > 1:
                print(f"[ForcingPrep] warning: multiple DATM files for {var_name} {year}-{month:02d}; using {candidates[0]}")
            files.append(candidates[0])

    if not files:
        raise FileNotFoundError(f"No DATM files found for {var_name} in {datm_root}")
    return files


def _preprocessed_forcing_output_dir() -> str:
    year_tag = f"{config.DATM_START_YEAR}_{config.DATM_END_YEAR}"
    return os.path.join(config.BASE_OUTPUT_ROOT, f"forcing_netcdf_datm_{year_tag}")


def _preprocessed_forcing_path(var_name: str) -> str:
    year_tag = f"{config.DATM_START_YEAR}-{config.DATM_END_YEAR}"
    return os.path.join(_preprocessed_forcing_output_dir(), f"{var_name}_{year_tag}.nc")


def _flatten_spatial_coords(ds: "xr.Dataset"):
    lat_name = "LATIXY" if "LATIXY" in ds else ("lat" if "lat" in ds else None)
    lon_name = "LONGXY" if "LONGXY" in ds else ("lon" if "lon" in ds else None)
    if lat_name is None or lon_name is None:
        raise KeyError("DATM forcing dataset must include LATIXY/LONGXY or lat/lon.")

    lat_data = np.asarray(ds[lat_name].values, dtype=float)
    lon_data = np.asarray(ds[lon_name].values, dtype=float)

    if lat_data.ndim == 3 and lon_data.ndim == 3:
        lat_data = lat_data[0, :, :]
        lon_data = lon_data[0, :, :]
    elif lat_data.ndim == 2 and lon_data.ndim == 2:
        pass
    elif lat_data.ndim == 1 and lon_data.ndim == 1:
        pass
    else:
        raise ValueError(f"Unsupported DATM coordinate shapes: {lat_data.shape}, {lon_data.shape}")

    return lat_data.reshape(-1), lon_data.reshape(-1)


def _monthly_mean_series_from_datm_files(var_name: str, datm_files: List[str]):
    if xr is None:
        raise ImportError("xarray is required to preprocess DATM forcing files.")

    monthly_values = []
    lat_flat = None
    lon_flat = None

    for fp in datm_files:
        with xr.open_dataset(fp, decode_times=False) as ds:
            if var_name not in ds:
                raise KeyError(f"Variable {var_name} not found in DATM file: {fp}")
            if lat_flat is None or lon_flat is None:
                lat_flat, lon_flat = _flatten_spatial_coords(ds)

            da = ds[var_name]
            if "time" not in da.dims:
                raise ValueError(f"{var_name} has no time dimension in file: {fp}")
            mean_da = da.mean(dim="time", skipna=True)
            monthly_values.append(np.asarray(mean_da.values, dtype=float).reshape(-1))

    if not monthly_values:
        raise ValueError(f"No monthly DATM files collected for {var_name}")

    series = np.stack(monthly_values, axis=0)  # (n_months, n_grid)
    if lat_flat is None or lon_flat is None:
        raise ValueError("Failed to read DATM spatial coordinates.")
    if series.shape[1] != lat_flat.shape[0] or series.shape[1] != lon_flat.shape[0]:
        raise ValueError("Spatial size mismatch between forcing data and coordinates.")
    return series, lat_flat, lon_flat


def _reference_mesh_coords_from_ds2() -> tuple[np.ndarray, np.ndarray]:
    """
    Read authoritative mesh coordinates from ds2 when running DATM same-mesh workflow.
    """
    with nc.Dataset(config.FILE_PATHS["ds2"]) as ds2:
        if "lat" not in ds2.variables or "lon" not in ds2.variables:
            raise KeyError("ds2 must include lat/lon variables for same-mesh DATM workflow.")
        lat = np.asarray(ds2.variables["lat"][:], dtype=float).reshape(-1)
        lon = np.asarray(ds2.variables["lon"][:], dtype=float).reshape(-1)
    return lat, lon


def prepare_forcing_inputs_from_datm(force_rebuild: bool = False, required_vars: List[str] = None) -> None:
    """
    Build consolidated forcing NetCDF files (DS4~DS9) from DATM monthly files.
    This mirrors the Zhuowei workflow so downstream modules can reuse stable
    intermediate forcing artifacts.
    
    Args:
        force_rebuild: If True, rebuild even if files exist
        required_vars: List of variable names to process (e.g., ["FLDS", "QBOT"]).
                       If None, process all variables.
    """
    if config.FORCING_MODE != "datm":
        return
    if xr is None:
        raise ImportError("xarray is required to preprocess DATM forcing files.")

    var_to_ds_key = {
        "FLDS": "ds4",
        "PSRF": "ds5",
        "FSDS": "ds6",
        "QBOT": "ds7",
        "PRECTmms": "ds8",
        "TBOT": "ds9",
    }
    
    # Filter to only required variables if specified
    if required_vars is not None:
        var_to_ds_key = {var: key for var, key in var_to_ds_key.items() if var in required_vars}
    
    out_dir = _preprocessed_forcing_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    for var_name, ds_key in var_to_ds_key.items():
        target_path = _preprocessed_forcing_path(var_name)

        if not force_rebuild and os.path.exists(target_path):
            config.FILE_PATHS[ds_key] = target_path
            print(f"[ForcingPrep] reuse existing {var_name}: {target_path}")
            continue

        datm_files = _resolve_datm_files(var_name)
        print(f"[ForcingPrep] building {var_name} -> {target_path}")

        series, lat_flat, lon_flat = _monthly_mean_series_from_datm_files(var_name, datm_files)
        try:
            ref_lat, ref_lon = _reference_mesh_coords_from_ds2()
            if ref_lat.shape[0] == series.shape[1] and ref_lon.shape[0] == series.shape[1]:
                lat_flat = ref_lat
                lon_flat = ref_lon
                print(f"[ForcingPrep] using ds2 mesh coordinates for {var_name}")
            else:
                print(
                    f"[ForcingPrep] warning: ds2 mesh size ({ref_lat.shape[0]}) does not match "
                    f"{var_name} grid size ({series.shape[1]}). Keeping DATM file coordinates."
                )
        except Exception as exc:
            print(f"[ForcingPrep] warning: could not use ds2 mesh coordinates for {var_name}: {exc}")

        n_months, n_grid = series.shape
        time_index = np.arange(n_months, dtype=np.int32)
        grid_ids = np.arange(n_grid, dtype=np.int32)

        ds_out = xr.Dataset(
            data_vars={
                var_name: (("time", "gridcell"), series.astype(np.float32)),
                "LATIXY": (("gridcell",), lat_flat.astype(np.float32)),
                "LONGXY": (("gridcell",), lon_flat.astype(np.float32)),
                "gridID": (("gridcell",), grid_ids),
            },
            coords={
                "time": time_index,
                "gridcell": grid_ids,
            },
            attrs={
                "description": "Monthly-mean forcing generated from DATM monthly files",
                "start_year": int(config.DATM_START_YEAR),
                "end_year": int(config.DATM_END_YEAR),
            },
        )

        encoding = {
            var_name: {"zlib": True, "complevel": 4, "dtype": "float32"},
            "LATIXY": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan},
            "LONGXY": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan},
            "gridID": {"dtype": "int32"},
            "time": {"dtype": "int32"},
        }
        ds_out.to_netcdf(target_path, encoding=encoding)

        config.FILE_PATHS[ds_key] = target_path
        print(f"[ForcingPrep] done {var_name}")


def _build_datm_spatial_mapping(ds: "xr.Dataset", index_df: pd.DataFrame):
    lat_name = "LATIXY" if "LATIXY" in ds else "lat"
    lon_name = "LONGXY" if "LONGXY" in ds else "lon"
    if lat_name not in ds or lon_name not in ds:
        raise KeyError("DATM forcing dataset must include LATIXY/LONGXY or lat/lon.")

    lat_data = np.asarray(ds[lat_name].values, dtype=float)
    lon_data = np.asarray(ds[lon_name].values, dtype=float)

    if lat_data.ndim == 1 and lon_data.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
        lat_data = lat_grid
        lon_data = lon_grid
    elif lat_data.ndim == 3 and lon_data.ndim == 3:
        lat_data = lat_data[0, :, :]
        lon_data = lon_data[0, :, :]
    elif lat_data.ndim != 2 or lon_data.ndim != 2:
        raise ValueError(f"Unsupported DATM coordinate shapes: {lat_data.shape}, {lon_data.shape}")

    forcing_coords = np.column_stack((lat_data.reshape(-1), lon_data.reshape(-1)))
    forcing_tree = cKDTree(forcing_coords)
    query_coords = index_df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    _, nearest_indices = forcing_tree.query(query_coords, k=1)
    return nearest_indices, lat_data.shape


def _extract_datm_series_for_batch(var_da, flat_index: int, spatial_shape) -> List[float]:
    if var_da.ndim == 2:
        dims = list(var_da.dims)
        if "time" in dims:
            other_dim = next(dim for dim in dims if dim != "time")
            values = var_da.isel({other_dim: int(flat_index)}).values
        else:
            # fallback: assume first dimension is time-like
            values = var_da.isel({dims[1]: int(flat_index)}).values
        return np.asarray(values, dtype=float).reshape(-1).tolist()

    if var_da.ndim == 3:
        dims = list(var_da.dims)
        non_time_dims = [d for d in dims if d != "time"]
        if len(non_time_dims) != 2:
            raise ValueError(f"Unsupported DATM variable dims: {dims}")
        i, j = np.unravel_index(int(flat_index), spatial_shape)
        values = var_da.isel({non_time_dims[0]: int(i), non_time_dims[1]: int(j)}).values
        return np.asarray(values, dtype=float).reshape(-1).tolist()

    raise ValueError(f"Unsupported DATM forcing dimensions: {var_da.ndim}")


def build_forcing_module(module_name: str, ds_key: str, var_name: str) -> None:
    print(f"[{module_name}] building...")
    index_df = load_index_master()
    out_dir = module_dir(module_name)

    if config.FORCING_MODE == "datm":
        ds_path = config.FILE_PATHS[ds_key]
        if not os.path.exists(ds_path):
            # Usually run_extraction() has already prepared these files once.
            # Keep this fallback to support direct module invocation.
            print(f"[ForcingPrep] missing preprocessed file for {var_name}, preparing now...")
            prepare_forcing_inputs_from_datm(force_rebuild=False, required_vars=[var_name])
            ds_path = config.FILE_PATHS[ds_key]
        print(f"[{module_name}] using DATM preprocessed forcing: {ds_path}")
        ds = nc.Dataset(ds_path)
        sources = [source_meta(ds_path), {"mode": "datm_preprocessed_monthly"}]
        try:
            if var_name not in ds.variables:
                raise KeyError(f"Variable {var_name} not found in preprocessed forcing file: {ds_path}")
            arr = ds.variables[var_name]
            if arr.ndim != 2:
                raise ValueError(f"Preprocessed forcing var {var_name} must be 2D, got ndim={arr.ndim}")

            dims = list(arr.dimensions)
            if "gridcell" in dims:
                grid_axis = dims.index("gridcell")
            else:
                # Fallback: assume non-time axis is the grid axis.
                grid_axis = 1 if dims and dims[0] == "time" else 0

            batch_ids = sorted(index_df["batch_id"].unique())
            for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
                print(f"[{module_name}] Processing batch {batch_id}/{batch_ids[-1]}...")
                rows = {"__row_id": batch_df["__row_id"].tolist(), var_name: []}
                for idx in batch_df["nearest_forcing_index"]:
                    idx = int(idx)
                    if grid_axis == 1:
                        series = np.asarray(arr[:, idx], dtype=float).tolist()
                    else:
                        series = np.asarray(arr[idx, :], dtype=float).tolist()
                    rows[var_name].append(series)
                pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))
                print(f"[{module_name}] Saved batch {batch_id}")
        finally:
            ds.close()
    else:
        ds_path = config.FILE_PATHS[ds_key]
        ds = nc.Dataset(ds_path)
        sources = [source_meta(ds_path), {"mode": "legacy"}]
        try:
            batch_ids = sorted(index_df["batch_id"].unique())
            for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
                print(f"[{module_name}] Processing batch {batch_id}/{batch_ids[-1]}...")
                rows = {
                    "__row_id": batch_df["__row_id"].tolist(),
                    var_name: [
                        np.asarray(ds.variables[var_name][idx, :], dtype=float).tolist()
                        for idx in batch_df["nearest_forcing_index"]
                    ],
                }
                pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))
                print(f"[{module_name}] Saved batch {batch_id}")
        finally:
            ds.close()

    save_manifest(module_name, {"module": module_name, "sources": sources})
    print(f"[{module_name}] done.")


def _build_grid_maps(ds10: nc.Dataset) -> Dict[str, Dict[int, np.ndarray]]:
    pft_gridcell_index = ds10.variables["pfts1d_gridcell_index"][:]
    col_gridcell_index = ds10.variables["cols1d_gridcell_index"][:]
    unique_ids = np.unique(pft_gridcell_index)
    pft_map = {int(grid_id): np.where(pft_gridcell_index == grid_id)[0] for grid_id in unique_ids}
    col_map = {int(grid_id): np.where(col_gridcell_index == grid_id)[0] for grid_id in unique_ids}
    return {"pft_map": pft_map, "col_map": col_map}


def build_ds10_restart_x() -> None:
    print("[A_ds10_restart_x] building...")
    index_df = load_index_master()
    ds10 = nc.Dataset(config.FILE_PATHS["ds10"])
    try:
        maps = _build_grid_maps(ds10)
        pft_map = maps["pft_map"]
        col_map = maps["col_map"]
        out_dir = module_dir("A_ds10_restart_x")

        all_vars = config.RESTART_PFT_VARS + config.RESTART_COL_1D_VARS + config.RESTART_COL_2D_VARS
        x_values = {var: ds10.variables[var][:] for var in all_vars}

        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            rows: Dict[str, List] = {"__row_id": batch_df["__row_id"].tolist()}
            for var in all_vars:
                rows[var] = []
            for gridcell_id in batch_df["gridcell_id"]:
                pft_idx = pft_map.get(int(gridcell_id), np.array([]))
                col_idx = col_map.get(int(gridcell_id), np.array([]))

                for var in config.RESTART_PFT_VARS:
                    rows[var].append(None if pft_idx.size == 0 else np.asarray(x_values[var][pft_idx], dtype=float).tolist())
                for var in config.RESTART_COL_1D_VARS:
                    rows[var].append(None if col_idx.size == 0 else np.asarray(x_values[var][col_idx], dtype=float).tolist())
                for var in config.RESTART_COL_2D_VARS:
                    rows[var].append(None if col_idx.size == 0 else np.asarray(x_values[var][col_idx, :], dtype=float).tolist())

            pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest("A_ds10_restart_x", {"module": "A_ds10_restart_x", "sources": [source_meta(config.FILE_PATHS["ds10"])]})
    finally:
        ds10.close()
    print("[A_ds10_restart_x] done.")


def build_r_list_y() -> None:
    print("[A_r_list_y] building...")
    index_df = load_index_master()
    ds10 = nc.Dataset(config.FILE_PATHS["ds10"])
    r_paths = config.FILE_PATHS["r_list"]
    if not r_paths:
        raise ValueError("FILE_PATHS['r_list'] is empty. Please provide one r file path.")
    ds_r = nc.Dataset(r_paths[0])
    try:
        maps = _build_grid_maps(ds10)
        pft_map = maps["pft_map"]
        col_map = maps["col_map"]
        out_dir = module_dir("A_r_list_y")

        all_vars = config.RESTART_PFT_VARS + config.RESTART_COL_1D_VARS + config.RESTART_COL_2D_VARS
        y_values = {var: ds_r.variables[var][:] for var in all_vars}

        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            rows: Dict[str, List] = {"__row_id": batch_df["__row_id"].tolist()}
            for var in all_vars:
                rows[f"Y_{var}"] = []

            for gridcell_id in batch_df["gridcell_id"]:
                pft_idx = pft_map.get(int(gridcell_id), np.array([]))
                col_idx = col_map.get(int(gridcell_id), np.array([]))

                for var in config.RESTART_PFT_VARS:
                    if pft_idx.size == 0:
                        rows[f"Y_{var}"].append(None)
                    else:
                        rows[f"Y_{var}"].append(np.asarray(y_values[var][pft_idx], dtype=float).tolist())
                for var in config.RESTART_COL_1D_VARS:
                    if col_idx.size == 0:
                        rows[f"Y_{var}"].append(None)
                    else:
                        rows[f"Y_{var}"].append(np.asarray(y_values[var][col_idx], dtype=float).tolist())
                for var in config.RESTART_COL_2D_VARS:
                    if col_idx.size == 0:
                        rows[f"Y_{var}"].append(None)
                    else:
                        rows[f"Y_{var}"].append(np.asarray(y_values[var][col_idx, :], dtype=float).tolist())

            pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest("A_r_list_y", {"module": "A_r_list_y", "sources": [source_meta(r_paths[0])]})
    finally:
        ds10.close()
        ds_r.close()
    print("[A_r_list_y] done.")


def build_clm_params_pft() -> None:
    print("[A_clm_params_pft] building...")
    index_df = load_index_master()
    ds = nc.Dataset(config.FILE_PATHS["clm_params"])
    try:
        broadcast_feature_dict = {}
        for var in config.PFT_TARGET_VARS:
            if var in ds.variables:
                vals = ds.variables[var][:17]
                if not (np.any(np.isnan(vals)) or np.ma.is_masked(vals)):
                    broadcast_feature_dict[var] = np.asarray(vals, dtype=float).tolist()

        out_dir = module_dir("A_clm_params_pft")
        for batch_id, batch_df in index_df.groupby("batch_id", sort=True):
            rows = {"__row_id": batch_df["__row_id"].tolist()}
            for var, val_list in broadcast_feature_dict.items():
                rows[f"pft_{var}"] = [val_list] * len(batch_df)
            pd.DataFrame(rows).to_pickle(os.path.join(out_dir, f"batch_{int(batch_id):02d}.pkl"))

        save_manifest("A_clm_params_pft", {"module": "A_clm_params_pft", "sources": [source_meta(config.FILE_PATHS["clm_params"])]})
    finally:
        ds.close()
    print("[A_clm_params_pft] done.")


def calculate_monthly_avg(time_series):
    if not isinstance(time_series, (list, np.ndarray)):
        return []
    if len(time_series) == 0:
        return []
    # Already monthly sequence (e.g. DATM preprocessed forcing): keep as-is.
    if len(time_series) % 12 == 0 and len(time_series) != config.TIME_SERIES_LENGTH:
        return [float(x) for x in np.asarray(time_series, dtype=float).reshape(-1).tolist()]
    if len(time_series) != config.TIME_SERIES_LENGTH:
        return []
    monthly_averages = []
    start_idx = 0
    for _ in range(config.YEARS_IN_DATA):
        for month_days in config.DAYS_PER_MONTH:
            end_idx = start_idx + month_days * config.STEPS_PER_DAY
            monthly_averages.append(float(np.mean(time_series[start_idx:end_idx])))
            start_idx = end_idx
    return monthly_averages


def read_module_batch(module_name: str, batch_id: int) -> pd.DataFrame:
    path = os.path.join(module_dir(module_name), f"batch_{batch_id:02d}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_pickle(path)


def module_batch_exists(module_name: str, batch_id: int) -> bool:
    """Check if a module batch file exists."""
    path = os.path.join(module_dir(module_name), f"batch_{batch_id:02d}.pkl")
    return os.path.exists(path)


def read_module_batch_optional(module_name: str, batch_id: int) -> pd.DataFrame:
    """Read module batch file if it exists, return None otherwise."""
    path = os.path.join(module_dir(module_name), f"batch_{batch_id:02d}.pkl")
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def assemble_final_dataset() -> None:
    print("[Assembler] assembling final dataset...")
    index_df = load_index_master()
    batch_ids = get_batch_ids(index_df)
    forcing_modules = list(config.FORCING_MODULE_MAP.keys())
    mandatory_modules = [
        "A_ds1_surface",
        "A_ds2_history_x",
        "A_ds10_restart_x",
        "A_h0_list_y",
        "A_r_list_y",
        "A_clm_params_pft",
    ] + forcing_modules

    # Track missing modules
    missing_modules_summary = {batch_id: [] for batch_id in batch_ids}
    
    for batch_id in batch_ids:
        print(f"[Assembler] Processing batch {batch_id}...")
        # A_index_core is required, raise error if missing
        if not module_batch_exists("A_index_core", batch_id):
            raise FileNotFoundError(f"Missing required module A_index_core for batch {batch_id}")
        merged = read_module_batch("A_index_core", batch_id)[["__row_id", "Latitude", "Longitude"]].copy()
        print(f"[Assembler] Loaded A_index_core for batch {batch_id}")

        # Process non-time-series modules first
        non_forcing_modules = [m for m in mandatory_modules if not m.startswith("A_forcing_")]
        for module in non_forcing_modules:
            if not module_batch_exists(module, batch_id):
                print(f"[Assembler] WARNING: Skipping missing module {module} for batch {batch_id}")
                missing_modules_summary[batch_id].append(module)
                continue
            print(f"[Assembler] Loading {module} for batch {batch_id}...")
            df_mod = read_module_batch(module, batch_id)
            merged = merged.merge(df_mod, on="__row_id", how="left")
            del df_mod  # Free memory
            gc.collect()  # Force garbage collection
            print(f"[Assembler] Merged {module} for batch {batch_id}")

        # Process time-series modules, convert to monthly averages immediately to save memory
        print(f"[Assembler] Processing forcing modules (with immediate monthly conversion) for batch {batch_id}...")
        time_series_columns = ["FLDS", "PSRF", "FSDS", "QBOT", "PRECTmms", "TBOT"]
        forcing_modules_sorted = [m for m in mandatory_modules if m.startswith("A_forcing_")]
        
        for module in forcing_modules_sorted:
            if not module_batch_exists(module, batch_id):
                print(f"[Assembler] WARNING: Skipping missing module {module} for batch {batch_id}")
                missing_modules_summary[batch_id].append(module)
                continue
            print(f"[Assembler] Loading and converting {module} for batch {batch_id}...")
            df_mod = read_module_batch(module, batch_id)
            # Find corresponding time-series column name
            var_name = None
            for ts_col in time_series_columns:
                if ts_col in df_mod.columns:
                    var_name = ts_col
                    break
            
            if var_name:
                # Convert to monthly averages immediately to reduce memory usage
                df_mod[var_name] = df_mod[var_name].apply(calculate_monthly_avg)
            
            merged = merged.merge(df_mod, on="__row_id", how="left")
            del df_mod  # Free memory
            gc.collect()  # Force garbage collection
            print(f"[Assembler] Merged {module} for batch {batch_id}")
        
        print(f"[Assembler] Completed all merges for batch {batch_id}")

        single_value_columns = [
            "landfrac", "LANDFRAC_PFT", "PCT_NATVEG", "AREA", "SOIL_COLOR", "SOIL_ORDER",
            "GPP", "Y_GPP", "HR", "AR", "NPP", "Y_HR", "Y_AR", "Y_NPP",
            "OCCLUDED_P", "SECONDARY_P", "LABILE_P", "APATITE_P",
        ]
        for col in single_value_columns:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col].astype(str).str.strip(), errors="coerce")

        for col in ["PCT_NAT_PFT", "PCT_SAND", "PCT_CLAY"]:
            if col in merged.columns:
                expanded_cols = merged[col].apply(pd.Series).fillna(0).add_prefix(f"{col}_")
                merged = merged.drop(columns=[col]).join(expanded_cols)

        y_columns = [col for col in merged.columns if col.startswith("Y_")]
        other_columns = [col for col in merged.columns if not col.startswith("Y_")]
        merged = merged[other_columns + y_columns]
        # Internal join key only; keep output schema aligned with legacy datasets.
        if "__row_id" in merged.columns:
            merged = merged.drop(columns=["__row_id"])

        out_path = os.path.join(config.FINAL_OUTPUT_DIR, f"training_data_batch_{batch_id:02d}.pkl")
        merged.to_pickle(out_path)
        print(f"[Assembler] saved: {out_path}")

    # Print summary of missing modules
    all_missing = {bid: mods for bid, mods in missing_modules_summary.items() if mods}
    if all_missing:
        print("\n[Assembler] WARNING: Some modules were missing during assembly:")
        for batch_id, modules in all_missing.items():
            print(f"  Batch {batch_id}: {', '.join(modules)}")
    else:
        print("\n[Assembler] All modules were successfully assembled.")
    
    print("[Assembler] done.")


def build_module(module_name: str) -> None:
    if module_name == "A_index_core":
        build_index_core()
    elif module_name == "A_ds1_surface":
        build_ds1_surface()
    elif module_name == "A_ds2_history_x":
        build_ds2_history_x()
    elif module_name == "A_ds10_restart_x":
        build_ds10_restart_x()
    elif module_name == "A_h0_list_y":
        build_h0_list_y()
    elif module_name == "A_r_list_y":
        build_r_list_y()
    elif module_name == "A_clm_params_pft":
        build_clm_params_pft()
    elif module_name in config.FORCING_MODULE_MAP:
        ds_key, var_name = config.FORCING_MODULE_MAP[module_name]
        build_forcing_module(module_name, ds_key, var_name)
    else:
        raise ValueError(f"Unknown module: {module_name}")


def run_extraction(to_build: List[str]) -> None:
    modules = to_build
    if "all" in modules:
        modules = config.ALL_MODULES.copy()
    
    forcing_modules = [m for m in modules if m.startswith("A_forcing_")]
    if forcing_modules and config.FORCING_MODE == "datm":
        # Determine which variables are needed based on modules to build
        required_vars = []
        for module_name in forcing_modules:
            if module_name in config.FORCING_MODULE_MAP:
                _, var_name = config.FORCING_MODULE_MAP[module_name]
                required_vars.append(var_name)
        prepare_forcing_inputs_from_datm(force_rebuild=False, required_vars=required_vars if required_vars else None)
    
    if modules:
        if "A_index_core" not in modules:
            if not os.path.exists(os.path.join(module_dir("A_index_core"), "index_master.pkl")):
                modules = ["A_index_core"] + modules
        for module_name in modules:
            build_module(module_name)


def run_assembly() -> None:
    assemble_final_dataset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular dataset construction by input file")
    parser.add_argument(
        "--config-input",
        type=str,
        default=None,
        help="Path to CNP_dataInput-style config file (default: config/CNP_dataInput.txt)",
    )
    parser.add_argument(
        "--build",
        nargs="+",
        default=[],
        help="Modules to build. Use 'all' or explicit names like A_forcing_ds7_qbot",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Assemble final dataset from existing artifacts",
    )
    parser.add_argument(
        "--forcing-mode",
        choices=["legacy", "datm"],
        default=None,
        help="Override forcing extraction mode from config",
    )
    parser.add_argument(
        "--prepare-forcing",
        action="store_true",
        help="Preprocess DATM monthly forcing files into consolidated DS4~DS9 NetCDF files.",
    )
    parser.add_argument(
        "--prepare-forcing-only",
        action="store_true",
        help="Only preprocess DATM monthly forcing files and exit.",
    )
    parser.add_argument(
        "--force-rebuild-forcing",
        action="store_true",
        help="Rebuild consolidated forcing files even if output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_input:
        config.load_config(args.config_input)
    ensure_dirs()

    if args.forcing_mode:
        config.FORCING_MODE = args.forcing_mode

    if args.prepare_forcing or args.prepare_forcing_only:
        prepare_forcing_inputs_from_datm(force_rebuild=args.force_rebuild_forcing)

    if args.prepare_forcing_only:
        return

    to_build = args.build
    if to_build:
        run_extraction(to_build)

    if args.assemble:
        run_assembly()

    if not to_build and not args.assemble:
        print("No action. Use --build all --assemble, or --build <module>, or --assemble.")


if __name__ == "__main__":
    main()

