# 025deg Parallel Pipeline

Parallel execution cores for the **025deg LandSim dataset construction** workflow.
This directory was extracted from monolithic scripts so collaborators can see
**what is parallelized**, **how to launch jobs**, and **where to tune**.

Designed and benchmarked on **Perlmutter (NERSC)** with striped inputs on
`$SCRATCH`. Other HPC sites should work with the same patterns after adjusting
config paths and `#SBATCH --account`.

---

## What this PR adds

| Before | After |
|--------|-------|
| Multi-node logic embedded in `run_pipeline.py` (~900 lines) | Reusable package under `parallel_pipeline/direct/` |
| MPI merge logic only in `merge_datm_mpi.py` | Reusable package under `parallel_pipeline/merge_datm/` |
| SLURM examples scattered under `scripts/` | Canonical examples in each `*/scripts/` subdirectory |

**Entry points are unchanged** — thin wrappers remain in `025deg/scripts/`:

- `run_pipeline.py --direct` → `parallel_pipeline.direct.runner.run_direct`
- `merge_datm_mpi.py` → `parallel_pipeline.merge_datm.runner`

Legacy launchers (`scripts/direct_run_*.sh`, `scripts/merge_datm/`) still work.

---

## Two pipelines

| Package | Purpose | Entry point | Best measured (Perlmutter) |
|---------|---------|-------------|----------------------------|
| [`direct/`](direct/README.md) | Build `training_data_batch_XX.pkl` from original NetCDF (no artifacts) | `scripts/run_pipeline.py --direct` | **29.2 s** job wall, 16 nodes |
| [`merge_datm/`](merge_datm/README.md) | Consolidate raw DATM `ha2x3h` → six monthly forcing NetCDFs (DS4–DS9) | `scripts/merge_datm_mpi.py` | **21.5 s** compute, 8 nodes / 1024 MPI ranks |

Typical **production order**:

```
1. merge_datm (MPI)     →  forcing_netcdf_datm_*/FLDS_*.nc … TBOT_*.nc
2. direct pipeline      →  training_data_batch_XX.pkl
```

Set `FORCING_MODE: datm` and point `CNP_dataInput.txt` at the consolidated
forcing directory before step 2.

---

## Directory layout

```
025deg/parallel_pipeline/
├── README.md                 ← this file
├── direct/
│   ├── README.md
│   ├── core.py               ← partition + load balance + SLURM plan
│   ├── workers.py            ← preload + parallel I/O + fork pool
│   ├── runner.py
│   └── scripts/
└── merge_datm/
    ├── README.md
    ├── core.py               ← constants + sharding + plan + timing
    ├── runner.py
    └── scripts/
```

Dataset-specific logic **stays outside** this package:

| Logic | Location |
|-------|----------|
| Direct batch extraction (`_direct_compute_one_batch`, index build) | `scripts/run_pipeline.py` |
| Per-file DATM read / NetCDF write | `scripts/merge_datm_monthly.py` |
| CNP config loader | `src/config.py` |

---

## Prerequisites

1. **Conda env `landsim`** (same as existing 025deg workflow).
2. **Inputs on fast storage** — stage striped copies to `$SCRATCH` before large runs.
3. **Config files** — edit paths in `025deg/config/*.txt` for your site (checked-in
   examples use NERSC scratch paths as templates).
4. **SLURM account** — change `#SBATCH --account=...` in example scripts.
5. **merge_datm only:** Cray MPICH + `mpi4py` (loaded via `scripts/merge_datm/env_cray_mpi.sh`).

---

## Quick start

Submit from the **`025deg`** directory so log paths resolve correctly.

**Step 1 — DATM forcing preprocess (optional if consolidated forcing already exists):**

```bash
cd 025deg
# Edit config/CNP_merge_datm_x5_phys_test.txt (DATM_ROOT, etc.) for your paths
sbatch parallel_pipeline/merge_datm/scripts/merge_datm_x5_phys_8node.sh
```

**Step 2 — Direct training-data build:**

```bash
cd 025deg
# First run after switching batch partition mode:
REBUILD_INDEX=1 sbatch parallel_pipeline/direct/scripts/direct_run_16node.sh
sbatch parallel_pipeline/direct/scripts/direct_run_16node.sh
```

---

## Configuration cheat sheet

### Direct pipeline (environment overrides)

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONFIG_INPUT` | `config/CNP_dataInput_from_LandSim_dataGEN.txt` | Main dataset config |
| `BATCH_PARTITION` | `restart_io` | `geographic` or `restart_io` (use `restart_io` at 16+ nodes) |
| `REBUILD_INDEX` | `0` | Set `1` once after changing `BATCH_PARTITION` |
| `DIRECT_WORKERS` | `64` | Fork pool size per node |
| `PRELOAD_WORKERS` | `16` | Spawn pool for parallel restart reads |

### merge_datm (environment overrides)

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONFIG_INPUT` | `config/CNP_merge_datm_x5_phys_test.txt` | Merge config (`DATM_ROOT`, year range) |
| `FORCING_OUT_DIR` | `$SCRATCH/.../forcing_netcdf_datm_x5_phys_1901_2023` | Output directory for six `.nc` files |
| `FORCE_REBUILD` | `1` in x5_phys example | Skip vars whose output already exists when `0` |
| `PROGRESS_EVERY` | `200` | Map-phase progress log interval per rank |
| `MERGE_DATM_MPI_EXTRA_ARGS` | — | Extra CLI, e.g. `--smoke-max-files 100` |

---

## Logs and benchmarks

| Pipeline | Log location |
|----------|--------------|
| Direct | `scripts/logs/job_<JOBID>/` |
| merge_datm | `scripts/logs/merge_datm_job_<JOBID>/timing.json` |

Reproducible benchmark reports: [`025deg/results/`](../results/README.md).

Smoke test for merge_datm MPI:

```bash
cd 025deg && sbatch scripts/merge_datm/validate_merge_datm_mpi.sh
```

---

## For reviewers

- **Scope:** parallel orchestration only; no change to scientific extraction math.
- **Backward compatible:** existing `scripts/` entry points and legacy SLURM scripts unchanged.
- **Tests:** import smoke (`landsim` env); full runs require Perlmutter + staged inputs.
- **Docs:** each subdirectory has a standalone README with architecture, CLI, and SLURM patterns.

See [`direct/README.md`](direct/README.md) and [`merge_datm/README.md`](merge_datm/README.md) for full detail.
