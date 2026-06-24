"""MPI map-reduce orchestration for DATM forcing merge."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from mpi4py import MPI

import config
from parallel_pipeline.merge_datm.core import (
    plan_on_root,
    select_var_to_ds_key,
    shard_files,
    target_path,
    write_timing_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPI map-reduce DATM monthly merge (file-sharded map, Allreduce, rank-0 write)."
    )
    parser.add_argument("--config-input", type=str, default=None)
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help="Subset of forcing variables (default: all six).",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Log progress every N local files per rank (default 50).",
    )
    parser.add_argument(
        "--smoke-max-files",
        type=int,
        default=0,
        help="Smoke test: only process the first N source files (0=disabled).",
    )
    parser.add_argument(
        "--smoke-out-dir",
        type=str,
        default=None,
        help="Smoke test: write outputs here instead of BASE_OUTPUT_ROOT path.",
    )
    parser.add_argument(
        "--no-write-zlib",
        action="store_true",
        help="Disable zlib compression on rank-0 consolidated NetCDF writes.",
    )
    return parser.parse_args()


def _log(msg: str, *, rank: int = 0, comm: MPI.Intracomm | None = None, root_only: bool = True) -> None:
    if root_only and comm is not None and rank != 0:
        return
    if comm is not None:
        host = MPI.Get_processor_name()
        if isinstance(host, tuple):
            host = host[0]
    else:
        host = "local"
    print(f"[MergeDATM-MPI r{rank}@{host}] {msg}", flush=True)


def _wtime() -> float:
    return MPI.Wtime()


def merge_datm_mpi(
    selected_vars: Optional[Sequence[str]] = None,
    force_rebuild: bool = False,
    progress_every: int = 50,
    smoke_max_files: int = 0,
    smoke_out_dir: Optional[str] = None,
    write_zlib: bool = True,
) -> int:
    import merge_datm_monthly as mdm
    import run_pipeline as rp

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t_mpi_entry = _wtime()
    entry_times = np.array([t_mpi_entry], dtype=np.float64)
    entry_min = np.empty(1, dtype=np.float64)
    entry_max = np.empty(1, dtype=np.float64)
    comm.Allreduce(entry_times, entry_min, op=MPI.MIN)
    comm.Allreduce(entry_times, entry_max, op=MPI.MAX)

    timing: Dict[str, object] = {
        "mpi_size": size,
        "rank": rank,
        "rank_entry_wtime_spread_s": float(entry_max[0] - entry_min[0]),
    }

    plan: Optional[dict] = None
    plan_wall_s = 0.0
    if rank == 0:
        t_plan = _wtime()
        try:
            var_to_ds_key = select_var_to_ds_key(selected_vars)
            plan = plan_on_root(
                var_to_ds_key,
                force_rebuild,
                merge_datm_monthly=mdm,
                preprocessed_output_dir_fn=rp._preprocessed_forcing_output_dir,
                preprocessed_path_fn=rp._preprocessed_forcing_path,
                smoke_max_files=smoke_max_files,
                smoke_out_dir=smoke_out_dir,
            )
            plan_wall_s = _wtime() - t_plan
            if plan is None:
                _log("all outputs exist; nothing to do", rank=rank, comm=comm)
        except Exception as exc:
            plan = {"__error__": repr(exc)}
            plan_wall_s = _wtime() - t_plan
    plan = comm.bcast(plan, root=0)
    t_post_plan = _wtime()
    timing["plan_wall_s_rank0"] = plan_wall_s
    timing["post_plan_bcast_s"] = float(t_post_plan - t_mpi_entry)

    if isinstance(plan, dict) and plan.get("__error__"):
        raise RuntimeError(f"planning failed on rank 0: {plan['__error__']}")
    if plan is None:
        return 0

    var_names: List[str] = plan["var_names"]
    var_to_ds_key: Dict[str, str] = plan["var_to_ds_key"]
    components_per_var: Dict[str, List[str]] = plan["components_per_var"]
    unique_files: List[str] = plan["unique_files"]
    file_to_vars: Dict[str, List[str]] = plan["file_to_vars"]
    ordered_ym = plan["ordered_ym"]
    lat_flat: np.ndarray = plan["lat_flat"]
    lon_flat: np.ndarray = plan["lon_flat"]
    smoke_mode: bool = bool(plan.get("smoke_mode"))
    plan_smoke_out_dir: Optional[str] = plan.get("smoke_out_dir")

    n_files = len(unique_files)
    n_months = len(ordered_ym)
    n_grid = int(lat_flat.shape[0])
    n_vars = len(var_names)
    ym_to_idx = {ym: i for i, ym in enumerate(ordered_ym)}
    var_to_vi = {v: i for i, v in enumerate(var_names)}

    my_files = shard_files(unique_files, rank, size)
    local_n_files = len(my_files)
    if rank == 0:
        _log(
            f"MPI size={size} | source files={n_files} | months={n_months} | "
            f"grid={n_grid} | vars={var_names}"
            + (f" | SMOKE max_files={smoke_max_files}" if smoke_mode else ""),
            rank=rank,
            comm=comm,
        )

    t_pre_alloc = _wtime()
    partial_sum = np.zeros((n_vars, n_months, n_grid), dtype=np.float64)
    partial_count = np.zeros((n_vars, n_months), dtype=np.int32)
    t_pre_map = _wtime()
    timing["alloc_partial_arrays_s"] = float(t_pre_map - t_pre_alloc)

    for local_idx, fp in enumerate(my_files, start=1):
        vars_needed = list(file_to_vars[fp])
        ym, var_means = mdm._process_one_file_worker((fp, vars_needed, components_per_var))
        mi = ym_to_idx[ym]
        for vname, vec in var_means.items():
            vi = var_to_vi[vname]
            if vec.shape[0] != n_grid:
                raise ValueError(
                    f"grid size mismatch in {fp} for {vname}: {vec.shape[0]} vs {n_grid}"
                )
            partial_sum[vi, mi] += vec
            partial_count[vi, mi] += 1

        if progress_every > 0 and (
            local_idx == 1
            or local_idx % progress_every == 0
            or local_idx == len(my_files)
        ):
            _log(
                f"map progress {local_idx}/{len(my_files)} local files",
                rank=rank,
                comm=comm,
                root_only=False,
            )

    map_local_s = _wtime() - t_pre_map
    t_map_local_done = _wtime()
    comm.Barrier()
    t_map_sync = _wtime()
    map_sync_wait_s = float(t_map_sync - t_map_local_done)

    all_map_local = comm.gather(map_local_s, root=0)
    if rank == 0:
        map_max_rank = int(np.argmax(all_map_local))
        map_max_s = float(all_map_local[map_max_rank])
        map_min_s = float(min(all_map_local))
        map_mean_s = float(sum(all_map_local) / len(all_map_local))
    else:
        map_max_rank = -1
        map_max_s = 0.0
        map_min_s = 0.0
        map_mean_s = 0.0
    map_max_s = comm.bcast(map_max_s, root=0)
    map_min_s = comm.bcast(map_min_s, root=0)
    map_mean_s = comm.bcast(map_mean_s, root=0)
    map_max_rank = comm.bcast(map_max_rank, root=0)

    all_sync_wait = comm.gather(map_sync_wait_s, root=0)
    if rank == 0:
        sync_wait_max_s = float(max(all_sync_wait))
        sync_wait_mean_s = float(sum(all_sync_wait) / len(all_sync_wait))
    else:
        sync_wait_max_s = 0.0
        sync_wait_mean_s = 0.0
    sync_wait_max_s = comm.bcast(sync_wait_max_s, root=0)
    sync_wait_mean_s = comm.bcast(sync_wait_mean_s, root=0)

    local_done = np.array([local_n_files], dtype=np.int64)
    total_done = np.zeros(1, dtype=np.int64)
    comm.Allreduce(local_done, total_done, op=MPI.SUM)
    if rank == 0:
        if int(total_done[0]) != n_files:
            raise RuntimeError(
                f"file shard coverage mismatch: mapped {int(total_done[0])} "
                f"!= expected {n_files}"
            )
        _log(
            f"map done: {int(total_done[0])}/{n_files} files | "
            f"local={map_local_s:.3f}s max={map_max_s:.3f}s@r{map_max_rank} "
            f"min={map_min_s:.3f}s mean={map_mean_s:.3f}s",
            rank=rank,
            comm=comm,
        )

    t_pre_reduce = _wtime()
    map_phase_wall_s = float(t_pre_reduce - t_pre_map)
    timing.update(
        {
            "local_files": local_n_files,
            "map_local_s": float(map_local_s),
            "map_max_s": float(map_max_s),
            "map_max_rank": int(map_max_rank),
            "map_min_s": float(map_min_s),
            "map_mean_s": float(map_mean_s),
            "map_phase_wall_s": map_phase_wall_s,
            "map_sync_wait_s_local": map_sync_wait_s,
            "map_sync_wait_max_s": sync_wait_max_s,
            "map_sync_wait_mean_s": sync_wait_mean_s,
        }
    )

    global_sum = np.zeros_like(partial_sum)
    global_count = np.zeros_like(partial_count)
    t_reduce_sum = _wtime()
    comm.Allreduce(partial_sum, global_sum, op=MPI.SUM)
    reduce_sum_s = _wtime() - t_reduce_sum
    t_reduce_count = _wtime()
    comm.Allreduce(partial_count, global_count, op=MPI.SUM)
    reduce_count_s = _wtime() - t_reduce_count
    reduce_total_s = reduce_sum_s + reduce_count_s
    if rank == 0:
        _log(
            f"reduce done in {reduce_total_s:.3f}s "
            f"(sum={reduce_sum_s:.3f}s count={reduce_count_s:.3f}s)",
            rank=rank,
            comm=comm,
        )

    write_total_s = 0.0
    write_per_var_s: Dict[str, float] = {}
    if rank == 0:
        t_write = _wtime()
        try:
            for vi, var_name in enumerate(var_names):
                t_var = _wtime()
                counts = global_count[vi]
                if not np.any(counts > 0):
                    raise RuntimeError(f"No samples accumulated for {var_name}")
                if smoke_mode:
                    valid_idx = [i for i, c in enumerate(counts) if c > 0]
                    series = np.stack(
                        [global_sum[vi, i] / float(counts[i]) for i in valid_idx],
                        axis=0,
                    )
                else:
                    if np.any(counts == 0):
                        missing = [ordered_ym[i] for i, c in enumerate(counts) if c == 0]
                        raise RuntimeError(
                            f"Missing months for {var_name} after reduce: {missing[:5]}"
                            f"{'...' if len(missing) > 5 else ''}"
                        )
                    series = (global_sum[vi] / counts[:, np.newaxis]).astype(np.float64)
                n_g = series.shape[1]
                var_lat, var_lon = mdm._maybe_override_coords_with_ds2(
                    var_name, lat_flat, lon_flat, n_g
                )
                out_path = target_path(
                    var_name,
                    plan_smoke_out_dir,
                    preprocessed_path_fn=rp._preprocessed_forcing_path,
                )
                mdm._write_consolidated(
                    var_name,
                    out_path,
                    series,
                    var_lat,
                    var_lon,
                    description=(
                        "Monthly-mean forcing built by merge_datm_mpi.py "
                        f"(MPI map-reduce, {size} ranks, file-sharded read)."
                    ),
                    use_zlib=write_zlib,
                )
                config.FILE_PATHS[var_to_ds_key[var_name]] = out_path
                var_secs = _wtime() - t_var
                write_per_var_s[var_name] = float(var_secs)
                _log(
                    f"wrote {var_name}: months={series.shape[0]} grid={n_g} "
                    f"in {var_secs:.3f}s -> {out_path}",
                    rank=rank,
                    comm=comm,
                )
            write_total_s = _wtime() - t_write
            _log(f"write done in {write_total_s:.3f}s", rank=rank, comm=comm)
        except Exception:
            comm.Abort(1)
            raise

    t_pre_barrier = _wtime()
    comm.Barrier()
    t_end = _wtime()
    barrier_s = float(t_end - t_pre_barrier)

    compute_total_s = float(t_end - t_mpi_entry)
    timing.update(
        {
            "reduce_sum_s": float(reduce_sum_s),
            "reduce_count_s": float(reduce_count_s),
            "reduce_total_s": float(reduce_total_s),
            "write_total_s_rank0": float(write_total_s),
            "write_per_var_s_rank0": write_per_var_s,
            "final_barrier_s": barrier_s,
            "compute_total_s": compute_total_s,
        }
    )

    if rank == 0:
        _log(
            "[TIMING] "
            f"rank_entry_spread={timing['rank_entry_wtime_spread_s']:.3f}s "
            f"plan={plan_wall_s:.3f}s alloc={timing['alloc_partial_arrays_s']:.3f}s "
            f"map_max={map_max_s:.3f}s@r{map_max_rank} straggler_wait={sync_wait_max_s:.3f}s "
            f"reduce={reduce_total_s:.3f}s write={write_total_s:.3f}s "
            f"barrier={barrier_s:.3f}s compute_total={compute_total_s:.3f}s",
            rank=rank,
            comm=comm,
        )

    per_rank = np.array([map_local_s, float(local_n_files), float(rank)], dtype=np.float64)
    gathered = comm.gather(per_rank, root=0)
    if rank == 0:
        report = {
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("SLURM_BATCH_JOB_ID", "")),
            "n_nodes": os.environ.get("SLURM_NNODES", ""),
            "ntasks": size,
            "source_files": n_files,
            "summary": {
                k: timing[k]
                for k in timing
                if k not in ("rank", "local_files", "map_local_s")
            },
            "per_rank_map": [
                {
                    "rank": int(row[2]),
                    "local_files": int(row[1]),
                    "map_local_s": float(row[0]),
                }
                for row in gathered
            ],
        }
        log_dir = os.environ.get("RUN_LOG_DIR", "").strip()
        if log_dir:
            out = Path(log_dir) / "timing.json"
            write_timing_report(out, report)
            _log(f"timing report -> {out}", rank=rank, comm=comm)

    return 0


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    t_main = _wtime()
    args = _parse_args()
    config_load_s = 0.0
    if args.config_input:
        t_cfg = _wtime()
        config.load_config(args.config_input)
        config_load_s = _wtime() - t_cfg
    if rank == 0 and config_load_s > 0:
        _log(f"config loaded in {config_load_s:.3f}s", rank=rank, comm=comm)
    try:
        rc = merge_datm_mpi(
            selected_vars=args.vars,
            force_rebuild=args.force_rebuild,
            progress_every=args.progress_every,
            smoke_max_files=max(0, int(args.smoke_max_files or 0)),
            smoke_out_dir=args.smoke_out_dir,
            write_zlib=not args.no_write_zlib,
        )
        if rank == 0:
            _log(
                f"main() wall {(_wtime() - t_main):.3f}s (includes config load)",
                rank=rank,
                comm=comm,
            )
        return rc
    except Exception as exc:
        print(f"[MergeDATM-MPI r{rank}] FATAL: {exc}", flush=True)
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
