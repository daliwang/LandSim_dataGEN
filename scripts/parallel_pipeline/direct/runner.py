"""Direct pipeline orchestration (multi-node + fork worker pool)."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import time
from typing import Optional

import config
from parallel_pipeline.direct import core
from parallel_pipeline.direct import workers
from parallel_pipeline.direct.core import (
    build_multi_node_batch_plan,
    detect_slurm_layout,
    log_pft_stripe_plan,
    task_batch_ids,
)
from parallel_pipeline.direct.workers import direct_main_preload


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m{sec:04.1f}s"
    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m{sec:04.1f}s"


def run_direct(
    config_input_path: Optional[str] = None,
    workers: int = 1,
    preload_workers: int = 1,
    *,
    rebuild_index: bool = False,
) -> None:
    """
    Artifact-free direct pipeline with optional multi-node and multi-process
    parallelism. See ``parallel_pipeline/README.md`` for SLURM launch patterns.
    """
    import run_pipeline as rp

    t_total = time.time()
    workers = max(1, int(workers))
    print(f"[Direct] starting direct (artifact-free) pipeline (workers={workers})...")

    if config.FORCING_MODE == "datm":
        rp._run_with_timing(
            "prepare DATM forcing inputs",
            rp.prepare_forcing_inputs_from_datm,
            force_rebuild=False,
        )

    rp._ensure_index_master(rebuild_index=rebuild_index)

    index_df = rp.load_index_master()
    print(
        f"[Direct] batch_partition={core.normalize_batch_partition(config.BATCH_PARTITION)}",
        flush=True,
    )
    all_batch_ids = sorted(index_df["batch_id"].unique().tolist())

    slurm_procid, slurm_ntasks, slurm_nnodes = detect_slurm_layout()

    if slurm_ntasks > 1:
        total = len(all_batch_ids)
        print(
            f"[Direct] index loaded: rows={len(index_df)} total_batches={total}",
            flush=True,
        )
        ds10_path = (config.FILE_PATHS.get("ds10") or "").strip()
        assignment, stats, plan_s = build_multi_node_batch_plan(
            index_df,
            ds10_path,
            slurm_ntasks,
            rp._build_grid_maps,
        )
        log_pft_stripe_plan(assignment, stats, slurm_procid, slurm_ntasks, slurm_nnodes, plan_s)
        batch_ids = task_batch_ids(assignment, slurm_procid, slurm_ntasks, slurm_nnodes)
    else:
        batch_ids = all_batch_ids
        print(
            f"[Direct] index loaded: rows={len(index_df)} batches={len(batch_ids)}",
            flush=True,
        )

    if len(batch_ids) == 0:
        print(f"[Direct] no batches assigned to this task (procid={slurm_procid}); exiting.")
        return

    os.makedirs(config.FINAL_OUTPUT_DIR, exist_ok=True)

    if workers <= 1:
        resources = rp._direct_open_resources(index_df)
        try:
            for batch_pos, batch_id in enumerate(batch_ids, start=1):
                t_b = time.time()
                batch_df = index_df[index_df["batch_id"] == batch_id].reset_index(drop=True)
                merged = rp._direct_compute_one_batch(batch_df, resources)
                out_path = os.path.join(
                    config.FINAL_OUTPUT_DIR, f"training_data_batch_{int(batch_id):02d}.pkl"
                )
                merged.to_pickle(out_path)
                print(
                    f"[Direct] batch {batch_id} ({batch_pos}/{len(batch_ids)}): "
                    f"rows={len(merged)} cols={len(merged.columns)} "
                    f"-> {out_path} ({time.time() - t_b:.1f}s)"
                )
                del merged
                gc.collect()
        finally:
            rp._direct_close_resources(resources)
        print(f"[Direct] done (elapsed {_format_elapsed(time.time() - t_total)})")
        return

    n_workers = min(workers, len(batch_ids))
    print(
        f"[Direct] parallel mode: {n_workers} worker process(es) over {len(batch_ids)} batch(es) "
        f"(fork). Main preloads heavy shared state once; workers inherit via COW.",
        flush=True,
    )

    t_pre = time.time()
    preload_local_batch_ids = batch_ids if slurm_ntasks > 1 else None
    workers.DIRECT_PRELOAD = direct_main_preload(
        index_df,
        preload_workers,
        preload_local_batch_ids,
        build_grid_maps=rp._build_grid_maps,
        build_forcing_index_mapping=rp._build_forcing_index_mapping,
        load_ds1_surface_arrays=rp._direct_load_ds1_surface_arrays,
    )
    workers.DIRECT_INDEX_DF_FORK = index_df
    print(
        f"[Direct] main preload done ({time.time() - t_pre:.1f}s); forking workers...",
        flush=True,
    )

    ctx = mp.get_context("fork")
    completed = 0
    started = time.time()
    pool: Optional[mp.pool.Pool] = None
    try:
        pool = ctx.Pool(processes=n_workers, initializer=workers.worker_init_fork)
        print(
            f"[Direct] pool created (t={time.time() - started:.1f}s); "
            "submitting batches and waiting for first completion...",
            flush=True,
        )
        workers._COMPUTE_ONE_BATCH = rp._direct_compute_one_batch
        for result in pool.imap_unordered(
            workers.process_one_batch_entry,
            batch_ids,
            chunksize=1,
        ):
            batch_id, out_path, n_rows, n_cols, elapsed_b = result
            completed += 1
            wall = time.time() - started
            rate = completed / wall if wall > 0 else 0.0
            eta = (len(batch_ids) - completed) / rate if rate > 0 else float("inf")
            print(
                f"[Direct] batch {batch_id} ({completed}/{len(batch_ids)}): "
                f"rows={n_rows} cols={n_cols} -> {out_path} "
                f"(batch={elapsed_b:.1f}s, wall={wall/60.0:.1f}min, ETA={eta/60.0:.1f}min)",
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

    print(f"[Direct] done (elapsed {_format_elapsed(time.time() - t_total)})")
