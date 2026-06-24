"""Batch partitioning, PFT-stripe load balance, and multi-node SLURM planning."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd

import config

# --- batch partition ---------------------------------------------------------

_BATCH_PARTITION_ALIASES = {
    "geographic": "geographic",
    "geo": "geographic",
    "latlon": "geographic",
    "restart_io": "restart_io",
    "restart": "restart_io",
    "io": "restart_io",
    "gridcell": "restart_io",
}


def partition_choices() -> tuple[str, ...]:
    return tuple(sorted(set(_BATCH_PARTITION_ALIASES.keys())))


def normalize_batch_partition(raw: str | None) -> str:
    if raw is None or not str(raw).strip():
        return "geographic"
    key = str(raw).strip().lower()
    canonical = _BATCH_PARTITION_ALIASES.get(key)
    if canonical is None:
        allowed = ", ".join(partition_choices())
        raise ValueError(f"Unknown batch partition {raw!r}; expected one of: {allowed}")
    return canonical


def assign_batch_ids_geographic(n_rows: int, batch_size: int) -> np.ndarray:
    return (np.arange(n_rows, dtype=np.int64) // batch_size) + 1


def assign_batch_ids_restart_io(gridcell_ids: np.ndarray, batch_size: int) -> np.ndarray:
    """Assign batch_id so each batch holds gridcells contiguous on restart axis-0."""
    gc = np.asarray(gridcell_ids, dtype=np.int64)
    if gc.size == 0:
        return gc
    order = np.argsort(gc, kind="stable")
    ranks = np.empty(gc.size, dtype=np.int64)
    ranks[order] = np.arange(gc.size, dtype=np.int64)
    return (ranks // batch_size) + 1


def assign_batch_ids(
    gridcell_ids: np.ndarray,
    *,
    partition: str,
    batch_size: int,
) -> np.ndarray:
    mode = normalize_batch_partition(partition)
    if mode == "restart_io":
        return assign_batch_ids_restart_io(gridcell_ids, batch_size)
    return assign_batch_ids_geographic(len(gridcell_ids), batch_size)


def _module_manifest_path(module_name: str) -> str:
    return os.path.join(config.MANIFEST_ROOT, f"{module_name}.json")


def load_index_manifest() -> Dict:
    path = _module_manifest_path("A_index_core")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_index_batch_partition() -> None:
    stored = load_index_manifest().get("batch_partition")
    if not stored:
        return
    current = normalize_batch_partition(config.BATCH_PARTITION)
    if normalize_batch_partition(stored) != current:
        raise SystemExit(
            "index_master.pkl batch partition does not match current settings: "
            f"manifest has {stored!r}, config/CLI has {current!r}. "
            "Remove artifacts/A_index_core (and downstream module artifacts) or pass --rebuild-index."
        )


def log_restart_io_batch_stats(
    index_df: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    """Log median PFT-axis span per batch (restart_io sanity check)."""
    if stats.empty:
        return
    try:
        pft_spans = [int(s["pft_max"] - s["pft_min"] + 1) for _, s in stats.iterrows()]
        print(
            "[A_index_core] restart_io batch PFT-axis span (rows): "
            f"median={int(np.median(pft_spans))}, "
            f"p95={int(np.percentile(pft_spans, 95))}, "
            f"max={max(pft_spans)}"
        )
    except Exception as exc:
        print(f"[A_index_core] warning: could not log restart_io batch stats: {exc}")


# --- PFT-stripe load balance -------------------------------------------------

PFT_STRIPE_A_SLAB = 7.5e-6
PFT_STRIPE_B_COST = 5.5e-6
PFT_STRIPE_INTERCEPT = 6.0


def compute_batch_axis_stats(
    index_df: pd.DataFrame,
    pft_map: Dict[int, np.ndarray],
    col_map: Dict[int, np.ndarray],
) -> pd.DataFrame:
    rows: List[Tuple[int, int, int, int, int, int, int, int]] = []
    for batch_id, sub in index_df.groupby("batch_id", sort=True):
        cells = np.unique(sub["gridcell_id"].to_numpy(dtype=np.int64))
        pft_buckets: List[np.ndarray] = []
        col_buckets: List[np.ndarray] = []
        for g in cells:
            pi = pft_map.get(int(g))
            if pi is not None and pi.size > 0:
                pft_buckets.append(pi)
            ci = col_map.get(int(g))
            if ci is not None and ci.size > 0:
                col_buckets.append(ci)
        if pft_buckets:
            pa = np.concatenate(pft_buckets)
            pft_min = int(pa.min())
            pft_max = int(pa.max()) + 1
            n_pft = int(pa.size)
        else:
            pft_min, pft_max, n_pft = 0, 0, 0
        if col_buckets:
            ca = np.concatenate(col_buckets)
            col_min = int(ca.min())
            col_max = int(ca.max()) + 1
            n_col = int(ca.size)
        else:
            col_min, col_max, n_col = 0, 0, 0
        rows.append(
            (
                int(batch_id),
                pft_min,
                pft_max,
                col_min,
                col_max,
                n_pft,
                n_col,
                int(cells.size),
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "batch_id",
            "pft_min",
            "pft_max",
            "col_min",
            "col_max",
            "n_pft",
            "n_col",
            "n_cells",
        ],
    )


def build_pft_stripe_assignment(
    stats: pd.DataFrame,
    n_tasks: int,
    slab_weight: float = PFT_STRIPE_A_SLAB,
    cost_weight: float = PFT_STRIPE_B_COST,
    intercept: float = PFT_STRIPE_INTERCEPT,
) -> List[List[int]]:
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if len(stats) == 0:
        return [[] for _ in range(n_tasks)]

    ordered = stats.sort_values("pft_min", kind="mergesort").reset_index(drop=True)
    bids = ordered["batch_id"].to_numpy(dtype=np.int64)
    batch_count = int(len(ordered))

    if n_tasks == 1:
        return [bids.astype(int).tolist()]
    if batch_count <= n_tasks:
        out: List[List[int]] = [[] for _ in range(n_tasks)]
        for i, bid in enumerate(bids.tolist()):
            out[i] = [int(bid)]
        return out

    pft_min = ordered["pft_min"].to_numpy(dtype=np.int64)
    pft_max = ordered["pft_max"].to_numpy(dtype=np.int64)
    col_min = ordered["col_min"].to_numpy(dtype=np.int64)
    col_max = ordered["col_max"].to_numpy(dtype=np.int64)
    cost_arr = (
        ordered["n_pft"].to_numpy(dtype=np.int64)
        + ordered["n_col"].to_numpy(dtype=np.int64)
    )

    inf = float("inf")
    dp = [[inf] * (batch_count + 1) for _ in range(n_tasks + 1)]
    parent = [[-1] * (batch_count + 1) for _ in range(n_tasks + 1)]
    dp[0][0] = 0.0

    pmin = pft_min.tolist()
    pmax = pft_max.tolist()
    cmin = col_min.tolist()
    cmax = col_max.tolist()
    cost = cost_arr.tolist()
    sw = float(slab_weight)
    cw = float(cost_weight)
    ic = float(intercept)

    for k in range(1, n_tasks + 1):
        dp_prev = dp[k - 1]
        dp_k = dp[k]
        par_k = parent[k]
        for i in range(k, batch_count + 1):
            cur_pft_max = -1
            cur_col_min = (1 << 62)
            cur_col_max = -1
            cur_cost = 0
            best = inf
            best_j = -1
            for j in range(i - 1, k - 2, -1):
                pm = pmax[j]
                if pm > cur_pft_max:
                    cur_pft_max = pm
                cm = cmin[j]
                if cm < cur_col_min:
                    cur_col_min = cm
                cM = cmax[j]
                if cM > cur_col_max:
                    cur_col_max = cM
                cur_cost += cost[j]
                prev = dp_prev[j]
                if prev == inf:
                    continue
                slab = (cur_pft_max - pmin[j]) + (cur_col_max - cur_col_min)
                t = sw * slab + cw * cur_cost + ic
                val = prev if prev > t else t
                if val < best:
                    best = val
                    best_j = j
            dp_k[i] = best
            par_k[i] = best_j

    cuts: List[Tuple[int, int]] = []
    i = batch_count
    for k in range(n_tasks, 0, -1):
        j = parent[k][i]
        cuts.append((j, i))
        i = j
    cuts.reverse()
    tasks: List[List[int]] = []
    for lo, hi in cuts:
        tasks.append([int(x) for x in bids[lo:hi]])
    while len(tasks) < n_tasks:
        tasks.append([])
    return tasks


def compute_local_restart_window(
    local_batch_ids: List[int],
    index_df: pd.DataFrame,
    pft_map: Dict[int, np.ndarray],
    col_map: Dict[int, np.ndarray],
) -> Dict[str, object]:
    local_grids_arr = (
        index_df.loc[index_df["batch_id"].isin(local_batch_ids), "gridcell_id"]
        .to_numpy(dtype=np.int64)
    )
    local_grids = np.unique(local_grids_arr)

    pft_buckets: List[np.ndarray] = []
    col_buckets: List[np.ndarray] = []
    for g in local_grids:
        pi = pft_map.get(int(g))
        if pi is not None and pi.size > 0:
            pft_buckets.append(pi)
        ci = col_map.get(int(g))
        if ci is not None and ci.size > 0:
            col_buckets.append(ci)

    if pft_buckets:
        pft_all = np.concatenate(pft_buckets)
        pft_start = int(pft_all.min())
        pft_end = int(pft_all.max()) + 1
    else:
        pft_start, pft_end = 0, 0

    if col_buckets:
        col_all = np.concatenate(col_buckets)
        col_start = int(col_all.min())
        col_end = int(col_all.max()) + 1
    else:
        col_start, col_end = 0, 0

    pft_map_local: Dict[int, np.ndarray] = {}
    col_map_local: Dict[int, np.ndarray] = {}
    for g in local_grids:
        gi = int(g)
        pi = pft_map.get(gi)
        if pi is not None and pi.size > 0:
            pft_map_local[gi] = (pi - pft_start).astype(np.int64, copy=False)
        ci = col_map.get(gi)
        if ci is not None and ci.size > 0:
            col_map_local[gi] = (ci - col_start).astype(np.int64, copy=False)

    return {
        "pft_window": (pft_start, pft_end),
        "col_window": (col_start, col_end),
        "pft_map_local": pft_map_local,
        "col_map_local": col_map_local,
        "n_local_grids": int(local_grids.size),
    }


# --- multi-node SLURM plan ---------------------------------------------------

def detect_slurm_layout() -> Tuple[int, int, int]:
    """Return (SLURM_PROCID, SLURM_NTASKS, SLURM_NNODES)."""
    try:
        procid = int(os.environ.get("SLURM_PROCID", "0"))
        ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    except ValueError:
        procid, ntasks, nnodes = 0, 1, 1
    return procid, ntasks, nnodes


def build_multi_node_batch_plan(
    index_df: pd.DataFrame,
    ds10_path: str,
    n_tasks: int,
    build_grid_maps: Callable[[nc.Dataset], Dict[str, Dict[int, np.ndarray]]],
) -> Tuple[List[List[int]], pd.DataFrame, float]:
    if not ds10_path or not os.path.exists(ds10_path):
        raise FileNotFoundError(
            "Multi-node PFT-stripe assignment requires FILE_PATHS['ds10']."
        )
    t_plan = time.time()
    ds10 = nc.Dataset(ds10_path)
    try:
        maps = build_grid_maps(ds10)
    finally:
        ds10.close()
    stats = compute_batch_axis_stats(index_df, maps["pft_map"], maps["col_map"])
    assignment = build_pft_stripe_assignment(stats, n_tasks)
    plan_s = time.time() - t_plan
    return assignment, stats, plan_s


def log_pft_stripe_plan(
    assignment: List[List[int]],
    stats: pd.DataFrame,
    procid: int,
    ntasks: int,
    nnodes: int,
    plan_s: float,
) -> None:
    print(
        f"[Direct] PFT-stripe plan built ({plan_s:.1f}s); "
        f"per-task summary (cost = sum(n_pft+n_col); "
        f"slab = pft_span+col_span; "
        f"t_pred = {PFT_STRIPE_A_SLAB:.1e}*slab + {PFT_STRIPE_B_COST:.1e}*cost + {PFT_STRIPE_INTERCEPT:.1f}):",
        flush=True,
    )
    for k, ids in enumerate(assignment):
        sub = stats[stats["batch_id"].isin(ids)]
        if len(sub) > 0:
            pft_lo = int(sub["pft_min"].min())
            pft_hi = int(sub["pft_max"].max())
            col_lo = int(sub["col_min"].min())
            col_hi = int(sub["col_max"].max())
            cost_k = int(sub["n_pft"].sum() + sub["n_col"].sum())
            slab_k = (pft_hi - pft_lo) + (col_hi - col_lo)
            t_pred = (
                PFT_STRIPE_A_SLAB * slab_k
                + PFT_STRIPE_B_COST * cost_k
                + PFT_STRIPE_INTERCEPT
            )
        else:
            pft_lo = pft_hi = col_lo = col_hi = 0
            cost_k = slab_k = 0
            t_pred = PFT_STRIPE_INTERCEPT
        marker = " <-- this task" if k == procid else ""
        print(
            f"[Direct]   task {k}: n_batches={len(ids):3d}  "
            f"slab={slab_k:>9d}  cost={cost_k:>10d}  t_pred={t_pred:5.1f}s  "
            f"PFT [{pft_lo:>9}:{pft_hi:>9})  COL [{col_lo:>8}:{col_hi:>8})"
            f"{marker}",
            flush=True,
        )


def task_batch_ids(
    assignment: List[List[int]],
    procid: int,
    ntasks: int,
    nnodes: int,
) -> List[int]:
    batch_ids = assignment[procid]
    print(
        f"[Direct] SLURM_PROCID={procid}/{ntasks} "
        f"(SLURM_NNODES={nnodes}); this task handles {len(batch_ids)} batches "
        f"(ids {batch_ids[0] if batch_ids else '-'}..{batch_ids[-1] if batch_ids else '-'})",
        flush=True,
    )
    return batch_ids
