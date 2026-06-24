"""MainPreload, parallel NetCDF reads, and fork-based intra-node worker pool."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd

import config
from parallel_pipeline.direct.core import compute_local_restart_window

# Populated by the parent before fork(); inherited by workers via COW.
DIRECT_PRELOAD: Dict[str, object] = {}
DIRECT_INDEX_DF_FORK: Optional[pd.DataFrame] = None
_DIRECT_W: Dict[str, object] = {}
_COMPUTE_ONE_BATCH = None


# --- parallel NetCDF reads (spawn pool) --------------------------------------

def read_vars_worker_mp(
    shard: List[Tuple[str, str, str, Optional[Tuple[int, int]]]],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    by_file: Dict[str, List[Tuple[str, str, Optional[Tuple[int, int]]]]] = {}
    for bucket, fp, var_name, slab in shard:
        by_file.setdefault(fp, []).append((bucket, var_name, slab))
    for fp, items in by_file.items():
        ds = nc.Dataset(fp)
        try:
            for bucket, var_name, slab in items:
                if slab is None:
                    out.setdefault(bucket, {})[var_name] = ds.variables[var_name][:]
                else:
                    start, end = slab
                    out.setdefault(bucket, {})[var_name] = ds.variables[var_name][start:end]
        finally:
            ds.close()
    return out


def read_vars_parallel(
    tasks: List[Tuple[str, str, str, Optional[Tuple[int, int]]]],
    workers: int,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for bucket, _fp, _var, _slab in tasks:
        out.setdefault(bucket, {})

    if workers <= 1 or len(tasks) <= 1:
        per_file: Dict[str, List[Tuple[str, str, Optional[Tuple[int, int]]]]] = {}
        for bucket, fp, var_name, slab in tasks:
            per_file.setdefault(fp, []).append((bucket, var_name, slab))
        for fp, items in per_file.items():
            ds = nc.Dataset(fp)
            try:
                for bucket, var_name, slab in items:
                    if slab is None:
                        out[bucket][var_name] = ds.variables[var_name][:]
                    else:
                        start, end = slab
                        out[bucket][var_name] = ds.variables[var_name][start:end]
            finally:
                ds.close()
        return out

    n_workers = min(int(workers), len(tasks))
    shards: List[List[Tuple[str, str, str, Optional[Tuple[int, int]]]]] = [
        [] for _ in range(n_workers)
    ]
    for i, task in enumerate(tasks):
        shards[i % n_workers].append(task)

    started = time.time()
    completed_vars = 0
    total = len(tasks)
    ctx = mp.get_context("spawn")
    pool: Optional[mp.pool.Pool] = None
    try:
        pool = ctx.Pool(processes=n_workers)
        for shard_result in pool.imap_unordered(read_vars_worker_mp, shards):
            n_in_shard = sum(len(d) for d in shard_result.values())
            for bucket, var_dict in shard_result.items():
                out[bucket].update(var_dict)
            completed_vars += n_in_shard
            elapsed = time.time() - started
            rate = completed_vars / elapsed if elapsed > 0 else 0.0
            eta = (total - completed_vars) / rate if rate > 0 else float("inf")
            print(
                f"[MainPreload]   shard done: {completed_vars}/{total} vars "
                f"({rate:.2f} var/s, elapsed {elapsed:.1f}s, ETA {eta:.1f}s)",
                flush=True,
            )
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    return out


# --- MainPreload -------------------------------------------------------------

def direct_main_preload(
    index_df: pd.DataFrame,
    preload_workers: int,
    local_batch_ids: Optional[List[int]],
    *,
    build_grid_maps,
    build_forcing_index_mapping,
    load_ds1_surface_arrays,
) -> Dict[str, object]:
    preload: Dict[str, object] = {}

    ds10_path = config.FILE_PATHS["ds10"]
    r_paths = config.FILE_PATHS.get("r_list", []) or []
    if not r_paths or not r_paths[0] or not os.path.exists(r_paths[0]):
        raise FileNotFoundError("Direct mode requires r_list[0] to exist (restart-Y file).")
    ds_r_path = r_paths[0]
    print(f"[MainPreload] ds10 path: {ds10_path}", flush=True)
    print(f"[MainPreload] r_list[0] path: {ds_r_path}", flush=True)

    t = time.time()
    ds10 = nc.Dataset(ds10_path)
    try:
        maps = build_grid_maps(ds10)
    finally:
        ds10.close()
    full_pft_map = maps["pft_map"]
    full_col_map = maps["col_map"]
    print(
        f"[MainPreload] grid maps built ({time.time() - t:.1f}s; "
        f"|pft_map|={len(full_pft_map)}, |col_map|={len(full_col_map)})",
        flush=True,
    )

    pft_slab: Optional[Tuple[int, int]] = None
    col_slab: Optional[Tuple[int, int]] = None
    if local_batch_ids is not None and len(local_batch_ids) > 0:
        win = compute_local_restart_window(
            local_batch_ids, index_df, full_pft_map, full_col_map
        )
        pft_slab = win["pft_window"]  # type: ignore[assignment]
        col_slab = win["col_window"]  # type: ignore[assignment]
        preload["pft_map"] = win["pft_map_local"]
        preload["col_map"] = win["col_map_local"]
        n_local = win["n_local_grids"]
        full_pft_n = sum(int(v.size) for v in full_pft_map.values()) if full_pft_map else 0
        full_col_n = sum(int(v.size) for v in full_col_map.values()) if full_col_map else 0
        pft_span = pft_slab[1] - pft_slab[0]
        col_span = col_slab[1] - col_slab[0]
        pft_pct = (100.0 * pft_span / full_pft_n) if full_pft_n else 0.0
        col_pct = (100.0 * col_span / full_col_n) if full_col_n else 0.0
        print(
            f"[MainPreload] local cell window: {n_local} gridcells from "
            f"{len(local_batch_ids)} batch(es); "
            f"PFT slab [{pft_slab[0]}:{pft_slab[1]}) = {pft_span} rows ({pft_pct:.1f}% of full {full_pft_n}); "
            f"COL slab [{col_slab[0]}:{col_slab[1]}) = {col_span} rows ({col_pct:.1f}% of full {full_col_n})",
            flush=True,
        )
    else:
        preload["pft_map"] = full_pft_map
        preload["col_map"] = full_col_map
        print(
            "[MainPreload] no local_batch_ids supplied; preloading FULL restart vars "
            "(no per-task slabbing).",
            flush=True,
        )

    all_pft_vars = list(config.RESTART_PFT_VARS)
    all_col_vars = list(config.RESTART_COL_1D_VARS) + list(config.RESTART_COL_2D_VARS)
    all_restart_vars = all_pft_vars + all_col_vars
    n_vars = len(all_restart_vars)
    workers_n = max(1, int(preload_workers))
    t = time.time()
    print(
        f"[MainPreload] preloading {n_vars} x_values + {n_vars} y_values "
        f"({2 * n_vars} reads) using {workers_n} subprocess worker(s)...",
        flush=True,
    )
    tasks: List[Tuple[str, str, str, Optional[Tuple[int, int]]]] = []
    for var in all_pft_vars:
        tasks.append(("x", ds10_path, var, pft_slab))
        tasks.append(("y", ds_r_path, var, pft_slab))
    for var in all_col_vars:
        tasks.append(("x", ds10_path, var, col_slab))
        tasks.append(("y", ds_r_path, var, col_slab))
    arrays = read_vars_parallel(tasks, workers=workers_n)
    preload["x_values"] = arrays["x"]
    preload["y_values"] = arrays["y"]
    elapsed = time.time() - t
    rate = (2 * n_vars) / elapsed if elapsed > 0 else 0.0
    print(
        f"[MainPreload] x_values + y_values preloaded "
        f"({2 * n_vars} vars, {elapsed:.1f}s, {rate:.2f} var/s)",
        flush=True,
    )

    preload["forcing_layout"] = {}
    preload["forcing_index_map"] = {}
    preload["forcing_paths"] = {}
    preload["forcing_full"] = None  # type: ignore[assignment]
    if config.FORCING_MODE == "datm":
        forcing_full: Dict[str, np.ndarray] = {}
        for _m, (ds_key, var_name) in config.FORCING_MODULE_MAP.items():
            ds_path = config.FILE_PATHS[ds_key]
            if not os.path.exists(ds_path):
                raise FileNotFoundError(
                    f"Direct mode: DATM-mode forcing file not found for {var_name}: {ds_path}"
                )
            t = time.time()
            ds_f = nc.Dataset(ds_path)
            arr = ds_f.variables[var_name]
            if arr.ndim != 2:
                ds_f.close()
                raise ValueError(
                    f"Preprocessed forcing var {var_name} must be 2D, got ndim={arr.ndim}"
                )
            dims = list(arr.dimensions)
            grid_axis = dims.index("gridcell") if "gridcell" in dims else (1 if dims and dims[0] == "time" else 0)
            preload["forcing_layout"][var_name] = grid_axis
            try:
                mapped = build_forcing_index_mapping(index_df, ds_f).astype(np.int64)
            except Exception as exc:
                print(
                    f"[MainPreload] forcing-coord mapping failed for {var_name} ({exc}); "
                    "fallback to nearest_forcing_index from index_master.",
                    flush=True,
                )
                mapped = index_df["nearest_forcing_index"].to_numpy(dtype=np.int64)
            preload["forcing_index_map"][var_name] = mapped
            preload["forcing_paths"][var_name] = ds_path
            t_read = time.time()
            forcing_full[var_name] = np.asarray(arr[:], dtype=float)
            ds_f.close()
            print(
                f"[MainPreload] forcing {var_name}: layout+map+RAM {forcing_full[var_name].shape} "
                f"({time.time() - t:.1f}s total, read {time.time() - t_read:.1f}s)",
                flush=True,
            )
        preload["forcing_full"] = forcing_full
    else:
        for _m, (ds_key, var_name) in config.FORCING_MODULE_MAP.items():
            preload["forcing_paths"][var_name] = config.FILE_PATHS[ds_key]

    t = time.time()
    print(
        "[MainPreload] loading ds1 static surface arrays into RAM "
        "(one slab read per var; eliminates per-batch HDF5 random I/O)...",
        flush=True,
    )
    preload["ds1_surface_arrays"] = load_ds1_surface_arrays(config.FILE_PATHS["ds1"])
    print(f"[MainPreload] ds1 surface arrays loaded ({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    ds_clm = nc.Dataset(config.FILE_PATHS["clm_params"])
    broadcast_feature_dict: Dict[str, List[float]] = {}
    for var in config.PFT_TARGET_VARS:
        if var in ds_clm.variables:
            vals = ds_clm.variables[var][:17]
            if not (np.any(np.isnan(vals)) or np.ma.is_masked(vals)):
                broadcast_feature_dict[var] = np.asarray(vals, dtype=float).tolist()
    ds_clm.close()
    preload["broadcast_feature_dict"] = broadcast_feature_dict
    print(f"[MainPreload] broadcast features extracted ({time.time() - t:.1f}s)", flush=True)

    return preload


# --- fork worker pool --------------------------------------------------------

def process_one_batch_entry(batch_id: int) -> Tuple[int, str, int, int, float]:
    if _COMPUTE_ONE_BATCH is None:
        raise RuntimeError("workers._COMPUTE_ONE_BATCH was not set before starting the pool")
    return process_one_batch(batch_id, compute_one_batch=_COMPUTE_ONE_BATCH)


def worker_init_fork() -> None:
    pid = os.getpid()
    t0 = time.time()
    print(f"[Worker {pid}] init: opening per-batch NetCDF handles...", flush=True)

    resources: Dict[str, object] = {}
    handles_to_close: list[object] = []
    preload = DIRECT_PRELOAD

    ds1_ram = preload.get("ds1_surface_arrays")
    if ds1_ram is not None:
        resources["ds1_surface_arrays"] = ds1_ram
        resources["ds1"] = None
    else:
        resources["ds1_surface_arrays"] = None
        resources["ds1"] = nc.Dataset(config.FILE_PATHS["ds1"])
        handles_to_close.append(resources["ds1"])

    ds2_path = (config.FILE_PATHS.get("ds2") or "").strip()
    if ds2_path and os.path.exists(ds2_path):
        resources["ds2"] = nc.Dataset(ds2_path)
        handles_to_close.append(resources["ds2"])
    else:
        resources["ds2"] = None

    h0_paths = config.FILE_PATHS.get("h0_list", []) or []
    h0_path = h0_paths[0] if h0_paths and h0_paths[0] else ""
    if h0_path and os.path.exists(h0_path):
        resources["ds_h0"] = nc.Dataset(h0_path)
        handles_to_close.append(resources["ds_h0"])
    else:
        resources["ds_h0"] = None

    resources["forcing_open"] = {}
    resources["legacy_forcing_open"] = {}
    forcing_paths: Dict[str, str] = preload["forcing_paths"]  # type: ignore[assignment]
    forcing_ram = preload.get("forcing_full")
    resources["forcing_full"] = forcing_ram
    if config.FORCING_MODE == "datm":
        if forcing_ram is None:
            for var_name, ds_path in forcing_paths.items():
                ds_f = nc.Dataset(ds_path)
                resources["forcing_open"][var_name] = ds_f
                handles_to_close.append(ds_f)
    elif forcing_ram is None:
        for var_name, ds_path in forcing_paths.items():
            ds_f = nc.Dataset(ds_path)
            resources["legacy_forcing_open"][var_name] = ds_f
            handles_to_close.append(ds_f)

    resources["pft_map"] = DIRECT_PRELOAD["pft_map"]
    resources["col_map"] = DIRECT_PRELOAD["col_map"]
    resources["x_values"] = DIRECT_PRELOAD["x_values"]
    resources["y_values"] = DIRECT_PRELOAD["y_values"]
    resources["forcing_layout"] = DIRECT_PRELOAD["forcing_layout"]
    resources["forcing_index_map"] = DIRECT_PRELOAD["forcing_index_map"]
    resources["broadcast_feature_dict"] = DIRECT_PRELOAD["broadcast_feature_dict"]
    resources["_handles_to_close"] = handles_to_close

    _DIRECT_W["resources"] = resources
    _DIRECT_W["index_df"] = DIRECT_INDEX_DF_FORK
    print(f"[Worker {pid}] init: ready ({time.time() - t0:.1f}s)", flush=True)


def process_one_batch(
    batch_id: int,
    *,
    compute_one_batch,
) -> Tuple[int, str, int, int, float]:
    t_b = time.time()
    index_df: pd.DataFrame = _DIRECT_W["index_df"]  # type: ignore[assignment]
    resources: Dict[str, object] = _DIRECT_W["resources"]  # type: ignore[assignment]
    if not _DIRECT_W.get("_announced_first_task", False):
        print(
            f"[Worker {os.getpid()}] picked up first batch (batch_id={batch_id})",
            flush=True,
        )
        _DIRECT_W["_announced_first_task"] = True
    batch_df = index_df[index_df["batch_id"] == batch_id].reset_index(drop=True)
    merged = compute_one_batch(batch_df, resources)
    out_path = os.path.join(
        config.FINAL_OUTPUT_DIR, f"training_data_batch_{int(batch_id):02d}.pkl"
    )
    merged.to_pickle(out_path)
    n_rows = int(len(merged))
    n_cols = int(len(merged.columns))
    del merged
    gc.collect()
    return int(batch_id), out_path, n_rows, n_cols, time.time() - t_b
