"""Planning helpers for MPI merge_datm: constants, sharding, rank-0 plan, timing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config

VAR_TO_DS_KEY = {
    "FLDS": "ds4",
    "PSRF": "ds5",
    "FSDS": "ds6",
    "QBOT": "ds7",
    "PRECTmms": "ds8",
    "TBOT": "ds9",
}


def shard_files(unique_files: List[str], rank: int, size: int) -> List[str]:
    """Assign disjoint contiguous chunks of source files to each rank."""
    n = len(unique_files)
    if n == 0:
        return []
    chunk = (n + size - 1) // size
    start = rank * chunk
    if start >= n:
        return []
    end = min(n, start + chunk)
    return unique_files[start:end]


def select_var_to_ds_key(selected_vars: Optional[Sequence[str]]) -> Dict[str, str]:
    var_to_ds_key = dict(VAR_TO_DS_KEY)
    if selected_vars is None:
        return var_to_ds_key
    wanted = set(selected_vars)
    unknown = wanted - set(var_to_ds_key)
    if unknown:
        raise ValueError(f"Unknown forcing variable(s): {sorted(unknown)}")
    return {v: k for v, k in var_to_ds_key.items() if v in wanted}


def target_path(var_name: str, smoke_out_dir: Optional[str], *, preprocessed_path_fn) -> str:
    year_tag = f"{config.DATM_START_YEAR}-{config.DATM_END_YEAR}"
    if smoke_out_dir:
        return os.path.join(smoke_out_dir, f"{var_name}_{year_tag}.nc")
    return preprocessed_path_fn(var_name)


def plan_on_root(
    var_to_ds_key: Dict[str, str],
    force_rebuild: bool,
    *,
    merge_datm_monthly,
    preprocessed_output_dir_fn,
    preprocessed_path_fn,
    smoke_max_files: int = 0,
    smoke_out_dir: Optional[str] = None,
) -> Optional[dict]:
    mdm = merge_datm_monthly
    if config.FORCING_MODE != "datm":
        raise RuntimeError(f"FORCING_MODE must be datm (got {config.FORCING_MODE!r})")

    out_dir = smoke_out_dir if smoke_out_dir else preprocessed_output_dir_fn()
    os.makedirs(out_dir, exist_ok=True)

    work_keys = dict(var_to_ds_key)
    if smoke_max_files > 0:
        force_rebuild = True
    if not force_rebuild:
        for var_name in list(work_keys.keys()):
            target = target_path(var_name, smoke_out_dir, preprocessed_path_fn=preprocessed_path_fn)
            if os.path.exists(target):
                config.FILE_PATHS[work_keys[var_name]] = target
                del work_keys[var_name]
    if not work_keys:
        return None

    components_per_var = {v: mdm._component_names_for(v) for v in work_keys}
    unique_files, file_to_vars, _ = mdm._build_file_to_vars(work_keys)
    if not unique_files:
        raise FileNotFoundError("No DATM source files resolved.")

    if smoke_max_files > 0:
        unique_files = unique_files[:smoke_max_files]
        file_to_vars = {fp: file_to_vars[fp] for fp in unique_files}

    ordered_ym = sorted({mdm._year_month_from_name(fp) for fp in unique_files})
    lat_flat, lon_flat = mdm._read_grid_coords_once(unique_files[0])

    return {
        "var_names": list(work_keys.keys()),
        "var_to_ds_key": work_keys,
        "components_per_var": components_per_var,
        "unique_files": unique_files,
        "file_to_vars": file_to_vars,
        "ordered_ym": ordered_ym,
        "lat_flat": lat_flat,
        "lon_flat": lon_flat,
        "smoke_out_dir": smoke_out_dir,
        "smoke_mode": smoke_max_files > 0,
    }


def write_timing_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
