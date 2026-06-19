# Parallel DATM Forcing Preprocess (merge_datm MPI)

This package contains the **parallelization core** for consolidating raw DATM
`ha2x3h` files into six monthly forcing NetCDFs (DS4–DS9). It lives under
`025deg/parallel_pipeline/merge_datm/` and is invoked by
`025deg/scripts/merge_datm_mpi.py` under `srun`.

The serial alternative is `025deg/scripts/merge_datm_monthly.py` (single-pass
multiprocessing on one node). The MPI path is preferred for full production
datasets (tens of thousands of source files).

---

## Directory layout

```
025deg/parallel_pipeline/merge_datm/
├── README.md
├── core.py                   ← constants, sharding, rank-0 plan, timing
├── runner.py                 ← merge_datm_mpi() + CLI main()
└── scripts/
```

Per-file read logic (`_process_one_file_worker`, `_write_consolidated`, etc.)
remains in `025deg/scripts/merge_datm_monthly.py`.

---

## What was parallelized

### Problem

Legacy `prepare_forcing_inputs_from_datm` in `run_pipeline.py` walks the DATM
tree **once per variable** (6× redundant I/O over ~38k files for x5_phys).

`merge_datm_monthly.py` fixed that with a **single-pass** read per file on one
node, but full production still took ~96s compute on 128 ranks — map phase bound.

### Solution: MPI map-reduce (`merge_datm_mpi`)

Three phases, one job:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Plan (r0)   │────▶│ Map (all ranks)  │────▶│ Reduce + Write  │
│ file list   │     │ read local shard │     │ Allreduce (SUM) │
│ grid coords │     │ partial sums     │     │ rank 0 → 6 nc   │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

| Phase | Parallelism | Details |
|-------|-------------|---------|
| **Plan** | rank 0 only | Resolve unique source files, month list, lat/lon grid; skip vars whose outputs already exist (unless `--force-rebuild`) |
| **Map** | **file-sharded** | Each rank reads `ceil(N/size)` files; accumulates `(var, month, grid)` partial sums + counts |
| **Reduce** | `MPI_Allreduce` | Global sum of partial sums and sample counts |
| **Write** | rank 0 only | Divide sum/count → monthly means; write 6 consolidated NetCDFs |

**File sharding** (`core.shard_files`): contiguous chunks of the sorted file list.
With 38,325 files and 1024 ranks → ~38 files/rank, highly balanced (map spread
&lt; 4.3s in job 53772332).

**Why MPI over multiprocessing only?**

- Scales to **1024+ ranks** across 8 Perlmutter nodes
- Map phase is embarrassingly parallel; Allreduce on `(6 vars × 1260 months × grid)`
  is ~8s at 1024 ranks — small vs map I/O

---

## Measured results (x5_phys, 38,325 source files, Perlmutter)

| Config | Job (approx.) | MPI ranks | `compute_total_s` | `map_phase_wall_s` | `write` (rank0) | `shell srun_wall_s` |
|--------|---------------|-----------|-------------------|--------------------|-----------------|---------------------|
| 1n × 128 | 53688016 | 128 | 96.2 | 72.3 | 14.9 | ~150 |
| 4n × 512 | 53688018 | 512 | 50.1 | 7.5 | 15.5 | ~101 |
| 8n × 1024 | 53688019 | 1024 | 46.9 | 3.8 | 15.9 | ~94 |
| **8n × 1024 (best)** | **53772332** | **1024** | **21.5** | **3.8** | **2.2** | **~79** |

**Current best: 8 nodes / 1024 ranks → `compute_total ≈ 21.5s`** (~4.5× vs 1-node).

Logs: `025deg/scripts/logs/merge_datm_job_<JOBID>/timing.json` and
`shell_timing.json`.

Smoke test (7,300 files): 4 ranks × 256 ≈ 8.2s compute — diminishing returns
beyond ~512 ranks for small inputs.

---

## How to run

### Prerequisites

1. **Striped DATM inputs** on `$SCRATCH` (e.g. `datm_3hrly_drives_x5_phys`).
2. **Config** pointing at `DATM_ROOT` — e.g.
   `025deg/config/CNP_merge_datm_x5_phys_test.txt`.
3. **Conda `landsim`** + Cray MPICH (via `scripts/merge_datm/env_cray_mpi.sh`).

### Option A — Recommended SLURM script (8 nodes)

```bash
cd /path/to/LandSim_dataGEN/025deg
sbatch parallel_pipeline/merge_datm/scripts/merge_datm_x5_phys_8node.sh
```

Outputs (default pattern — override `FORCING_OUT_DIR` for your site):

```
${SCRATCH}/landsim_inputs/striped/forcing_netcdf_datm_x5_phys_1901_2023/
  FLDS_1901-2023.nc  …  TBOT_1901-2023.nc
```

**Important:** `DATM_ROOT` and other input paths are read from `CONFIG_INPUT`
(e.g. `config/CNP_merge_datm_x5_phys_test.txt`). Copy that file and edit paths
before running on a new machine.

### Option B — Legacy launchers (still valid)

```bash
sbatch scripts/merge_datm/merge_datm_x5_phys_8node.sh
```

Both call the same `scripts/merge_datm_mpi.py` entry point.

### Option C — Custom SLURM script

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=1024
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
# ...

source parallel_pipeline/merge_datm/scripts/merge_datm_paths.sh
export CONFIG_INPUT=/path/to/CNP_merge_datm_x5_phys_test.txt
export FORCING_OUT_DIR=/path/to/forcing_output
export FORCE_REBUILD=1
export N_NODES=8
source "${SCRIPT_DIR}/merge_datm_common.sh"
```

**Layout rules**

| Setting | Recommended | Why |
|---------|-------------|-----|
| `SLURM_NTASKS` | 128 × nodes | One rank per CPU; map scales linearly until I/O saturated |
| Files/rank | ~30–300 | x5_phys full set: 38325/1024 ≈ 37 |
| `PROGRESS_EVERY` | 200 (x5_phys) | Log noise vs visibility tradeoff |
| `FORCE_REBUILD` | `1` first run, `0` incremental | Skip vars whose `.nc` already exist |

### Option D — Smoke / subset test

```bash
cd 025deg
export MERGE_DATM_MPI_EXTRA_ARGS="--smoke-max-files 100 --smoke-out-dir /tmp/merge_smoke"
sbatch parallel_pipeline/merge_datm/scripts/merge_datm_x5_phys_1node.sh
```

Or use the dedicated validation script (4 MPI ranks, debug QoS):

```bash
cd 025deg && sbatch scripts/merge_datm/validate_merge_datm_mpi.sh
```

---

## CLI reference (`merge_datm_mpi.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--config-input` | required | CNP merge config with `DATM_ROOT`, year range |
| `--force-rebuild` | off | Overwrite existing consolidated outputs |
| `--vars FLDS …` | all six | Restrict output variables |
| `--progress-every N` | 50 | Rank-local map progress interval |
| `--smoke-max-files N` | 0 | Process only first N source files |
| `--smoke-out-dir PATH` | — | Alternate output directory (smoke) |
| `--no-write-zlib` | off | Disable NetCDF zlib compression on write |

---

## Monitoring

Batch log: `scripts/logs/merge_datm_x5_phys_sbatch_<JOBID>.out`

Per-rank MPI logs: `scripts/logs/merge_datm_job_<JOBID>/mpi_<jobid>_r<rank>.out`

Rank 0 summary line:

```
[MergeDATM-MPI r0@...] [TIMING] rank_entry_spread=... plan=... map_max=...@r639 ...
  reduce=... write=... compute_total=21.482s
```

Structured timing: `merge_datm_job_<JOBID>/timing.json`

Healthy 1024-rank runs show:

- `map_max_s` / `map_mean_s` ratio close to 1.05–1.10
- `reduce_total_s` ~8–9s
- `write_total_s_rank0` ~2–16s depending on zlib/settings

---

## Python API

```python
from parallel_pipeline.merge_datm import merge_datm_mpi

# Must run under mpirun/srun; config loaded beforehand
import config
config.load_config("path/to/CNP_merge_datm_x5_phys_test.txt")
merge_datm_mpi(force_rebuild=True, progress_every=200)
```

---

## Integration with direct pipeline

After merge completes, point `CNP_dataInput.txt` (or your run config) at the
consolidated forcing directory, set `FORCING_MODE: datm`, and run:

```bash
python run_pipeline.py --direct --batch-partition restart_io ...
```

The direct pipeline will skip DATM prep when consolidated files already exist.

---

## Related paths

| Path | Content |
|------|---------|
| `025deg/scripts/merge_datm_monthly.py` | Serial / mp.Pool single-node merge |
| `025deg/scripts/merge_datm/` | Original SLURM launchers |
| `025deg/parallel_pipeline/direct/` | Parallel training-data direct pipeline |
| `025deg/scripts/merge_datm/validate_merge_datm_mpi.sh` | Smoke validation helper |
