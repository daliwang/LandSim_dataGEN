# Parallel Direct Pipeline (025deg)

Multi-node, artifact-free pipeline that reads original NetCDF inputs and writes
`training_data_batch_XX.pkl` directly. Invoked by:

```bash
python scripts/run_pipeline.py --direct ...
```

Parallel orchestration lives here; dataset extraction logic remains in
`scripts/run_pipeline.py`.

---

## Why a separate package?

The direct path replaces modular extract → assemble with a single in-memory pass.
At 16 Perlmutter nodes the bottlenecks are **restart I/O slab width** and
**batch-level load imbalance**, not raw CPU count. This package implements three
independent parallel layers (index partition, multi-node plan, intra-node fork pool).

---

## Directory layout

```
parallel_pipeline/direct/
├── core.py       ← batch partition + PFT-stripe DP + SLURM plan
├── workers.py    ← MainPreload + parallel reads + fork pool
├── runner.py     ← run_direct() orchestration
└── scripts/
    ├── direct_run_log_setup.sh
    ├── direct_run_8node.sh
    └── direct_run_16node.sh    ← recommended
```

---

## Architecture (three layers)

### Layer 1 — Index batch partition (`core.py`)

`A_index_core` assigns each gridcell a `batch_id`. Two strategies:

| Mode | Assignment | Restart I/O effect |
|------|------------|------------------|
| `geographic` | Lat×lon scan order | Wide PFT-axis spans per batch |
| **`restart_io` (recommended)** | Sort by `gridcell_id`, 1000 cells/batch | Narrow slabs; −18% wall at 16 nodes |

CLI / config:

```bash
python run_pipeline.py --direct --batch-partition restart_io ...
# or in config:  BATCH_PARTITION: restart_io
```

After switching mode, rebuild the index once:

```bash
REBUILD_INDEX=1 sbatch parallel_pipeline/direct/scripts/direct_run_16node.sh
```

### Layer 2 — Multi-node load balance (`core.py`)

With `srun --ntasks=N` (one task per node), each task owns a subset of batches.
Naïve contiguous splits cause stragglers because restart preload time scales with
**PFT slab width**, not row count.

**PFT-stripe + v2 DP:**

1. Per-batch stats: PFT/COL axis spans, fancy-index row counts.
2. Sort batches by `pft_min`.
3. DP-partition into `N` segments minimizing predicted max task time:

   ```
   t_pred = 7.5e-6 * slab_rows + 5.5e-6 * cost_rows + 6.0
   ```

All tasks run the same deterministic algorithm on `index_master.pkl` — no plan
file, no inter-node communication.

**Per-task restart slabbing** (`compute_local_restart_window`): each node preloads
only the axis-0 window covering its batches (~1/N of restart data).

### Layer 3 — Intra-node parallelism (`workers.py`)

| Flag | Default | Role |
|------|---------|------|
| `--preload-workers` | 16 | Spawn pool for parallel restart NetCDF reads (HDF5 not thread-safe on NERSC) |
| `--direct-workers` | 64 | Fork pool; workers inherit preloaded RAM via copy-on-write |

Per SLURM task flow:

1. Main process preloads restart slabs, ds1 surface, DATM forcing.
2. Close all NetCDF handles (HDF5 + fork is unsafe).
3. Fork workers; each writes assigned batch PKLs.

---

## Measured results (Perlmutter, striped `$SCRATCH` inputs)

| Configuration | Nodes | Job wall (max task) |
|---------------|-------|---------------------|
| Geographic + v1 cost split | 8 | 48.7 s |
| Geographic + v2 DP | 8 | 38.8 s |
| restart_io + v2 DP | 8 | 39.0 s |
| **restart_io + v2 DP** | **16** | **29.2 s** |

Reports: [`025deg/results/pft_stripe_load_balance_2026-05-17/`](../../results/pft_stripe_load_balance_2026-05-17/),
[`025deg/results/restart_io_load_balance_2026-05-17/`](../../results/restart_io_load_balance_2026-05-17/).

---

## How to run

### Prerequisites

1. Striped inputs on `$SCRATCH` (restart, ds1, forcing, grid maps).
2. Conda env `landsim`.
3. `artifacts/A_index_core/index_master.pkl` matching `BATCH_PARTITION`.

### Recommended SLURM (16 nodes)

```bash
cd 025deg
REBUILD_INDEX=1 sbatch parallel_pipeline/direct/scripts/direct_run_16node.sh   # first time only
sbatch parallel_pipeline/direct/scripts/direct_run_16node.sh
```

Logs: `scripts/logs/job_<JOBID>/legacy_<JOBID>_n<task>.out`

### Custom SLURM template

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1       # SLURM_NTASKS = node count
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --account=YOUR_ACCOUNT

module load conda/Miniforge3-25.11.0-1
conda activate landsim
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd /path/to/LandSim_dataGEN/025deg/scripts
srun python -u run_pipeline.py \
    --config-input ../config/CNP_dataInput_from_LandSim_dataGEN.txt \
    --direct \
    --batch-partition restart_io \
    --direct-workers 64 \
    --preload-workers 16
```

| Setting | Recommended | Why |
|---------|-------------|-----|
| `SLURM_NTASKS` | = node count | One main process per node; DP uses this as `n_tasks` |
| `--direct-workers` | 64 on 128-core nodes | Fork pool; leave headroom for preload subprocesses |
| `--preload-workers` | 16 | Parallel restart variable reads |
| `BATCH_PARTITION` | `restart_io` | Better balance at 16+ nodes |

### Single-node debug

```bash
cd 025deg/scripts
python run_pipeline.py --direct --batch-partition restart_io \
    --direct-workers 8 --preload-workers 4
```

With `SLURM_NTASKS=1`, multi-node DP is skipped.

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--direct` | off | Enable artifact-free direct pipeline |
| `--batch-partition` | from config | `geographic` or `restart_io` |
| `--rebuild-index` | off | Force rebuild `index_master.pkl` |
| `--direct-workers` | min(32, cpu_count) | Intra-node fork pool size |
| `--preload-workers` | min(16, cpu_count) | MainPreload spawn pool size |
| `--config-input` | `config/CNP_dataInput.txt` | Dataset paths and variables |

---

## Monitoring

Look for these lines in `legacy_*_n*.out`:

```
[Direct] batch_partition=restart_io
[Direct] PFT-stripe plan built (...s)
[Direct]   task k: n_batches= ... slab= ... t_pred= ...
[MainPreload] x_values + y_values preloaded (... var/s)
[Direct] main preload done (...s); forking workers...
[Direct] done (elapsed ...)
```

Healthy multi-node runs: `t_pred` within ~±15% across tasks; task wall ratio
&lt; 1.4 (restart_io at 16n ≈ 1.36).

---

## Python API

```python
from parallel_pipeline.direct import run_direct, build_pft_stripe_assignment

# run_direct(config, ...) mirrors the --direct CLI path
```

Requires `025deg` on `sys.path` (same as `run_pipeline.py`).

---

## Known limits

1. Each node still reads full ds1 + forcing (duplicate I/O by design).
2. Index build (`build_index_core`) remains single-threaded.
3. Experimental PnetCDF driver: `scripts/pnetcdf/` (not yet production default).

---

## Related

| Path | Content |
|------|---------|
| [`parallel_pipeline/README.md`](../README.md) | Top-level index + quick start |
| [`scripts/run_pipeline.py`](../../scripts/run_pipeline.py) | CLI + extraction logic |
| [`parallel_pipeline/merge_datm/`](../merge_datm/) | DATM forcing preprocess (run before direct if needed) |
