# LandSim_dataGEN 0.25° Training Dataset Generation Guide

> How to generate the 0.25° E3SMV3 training dataset with LandSim_dataGEN.

---

## 1. What This Project Does

It turns raw land-model (ELM/E3SM) NetCDF outputs into a dataset that AI models can train on directly.

- **Input**: 0.25°-resolution raw `.nc` files — surfdata, restart, DATM climate forcing, etc.
- **Output**: a set of `training_data_batch_XX.pkl` files; each pkl holds a number of grid-cell
  samples with X features, Y labels, and time-series forcing all aligned.
- **Design idea**: modular — each input source is extracted into its own artifact, only changed
  modules are rebuilt, and everything is assembled once at the end.

---

## 2. Role of Each Code File

| File | Role | Key points |
|---|---|---|
| **`scripts/run_pipeline.py`** | The engine of the whole pipeline | Contains the 13 `build_*` module functions, DATM forcing preprocessing `prepare_forcing_inputs_from_datm()`, and assembly `assemble_final_dataset()`. `run_extraction`/`run_assembly` just call into it. |
| `scripts/run_extraction.py` | Extraction only | Parse args → load config → (optional) preprocess forcing → build module artifacts. No assembly. |
| `scripts/run_assembly.py` | Assembly only | Merge existing module artifacts by `__row_id` into the final dataset. |
| `scripts/validate_final_dataset.py` | Quality check | Per-batch checks: row counts, coordinate NaNs, whether forcing columns are arrays of the correct length, etc. |
| `src/config.py` | Config hub | Sets default values for all parameters first; `load_config()` then applies overrides from the config file. `ALL_MODULES` lists all 13 modules. |
| `src/cnp_data_input_parse.py` | Text parser | Parses `CNP_dataInput.txt`: supports comments, variable groups, `alias` keys, and `${DATA_ROOT}` variable expansion. |
| `config/CNP_dataInput*.txt` | Config files | Specify input file paths, output directory, forcing mode, batch size, variable lists. |

---

## 3. Pipeline Flow (4 Stages + 13 Modules)

What `run_pipeline.py` does can be seen as 4 layers:

```
① Config loading      Parse CNP_dataInput.txt; resolve input paths, output dir, variable lists
      ↓
② Forcing preprocessing  Aggregate raw 3-hourly DATM drives into monthly-mean NetCDF (6 climate variables)
      ↓
③ Module build        13 modules each extract from their input source → artifacts/<module>/batch_XX.pkl
      ↓
④ Assembly            Using A_index_core as the base, merge all modules by __row_id
                      → final_dataset/training_data_batch_XX.pkl
```

### What each of the 13 modules does

| Module | Purpose | Output type |
|---|---|---|
| **`A_index_core`** | **The core.** Selects land grid cells, assigns the unique row key `__row_id`, partitions batches, and uses a KD-tree to build mappings to the restart/forcing grids. Every other module depends on it. | Index |
| `A_ds1_surface` | Extract static surface attributes from surfdata (soil, vegetation fractions, etc.) | X feature |
| `A_ds2_history_x` | Extract history grid state from the history file (optional; skipped if no DS2) | X feature |
| `A_ds10_restart_x` | Extract PFT/COL structural state from the restart file (C/N/P pools, etc.) | X feature |
| `A_h0_list_y` | Extract target grid variables from the target h0 file (optional; skipped if no H0) | Y label |
| `A_r_list_y` | Extract target structural state from the target restart file | Y label |
| `A_forcing_ds4_flds` | FLDS longwave radiation time series | Forcing |
| `A_forcing_ds5_psrf` | PSRF surface pressure time series | Forcing |
| `A_forcing_ds6_fsds` | FSDS shortwave radiation time series | Forcing |
| `A_forcing_ds7_qbot` | QBOT specific humidity time series | Forcing |
| `A_forcing_ds8_prectmms` | PRECTmms precipitation time series | Forcing |
| `A_forcing_ds9_tbot` | TBOT air temperature time series | Forcing |
| `A_clm_params_pft` | Extract PFT physical parameters from clm_params | X feature |

---

## 4. How to Read the Config File (using the 0.25° config)

File: `config/CNP_dataInput_from_LandSim_dataGEN.txt`

| Config key | Meaning | Value for the 0.25° dataset |
|---|---|---|
| `DATA_ROOT` | Raw data root directory | `.../9_0_25_degree/025E3SMV3` |
| `BASE_OUTPUT_ROOT` | Artifact output root directory | `.../LandSim_dataGEN/output2` |
| `DS1_PATH` | surfdata file | `surfdata_0.25x0.25_simyr1850_...nc` |
| `DS10_PATH` | Initial restart file | `...adsp.elm.r.0021-...nc` |
| `R_LIST_PATHS` | Target restart file | `...fnsp.elm.r.0251-...nc` |
| `CLM_PARAMS_PATH` | PFT parameter file | `clm_params_c211124.nc` |
| `DS2_PATH` / `H0_LIST_PATHS` | Optional inputs | Left empty → corresponding module auto-skips |
| `FORCING_MODE` | Forcing mode | `datm` |
| `DATM_ROOT` | Raw 3-hourly drives directory | `.../3hrly_drives` |
| `BATCH_SIZE` | Samples per batch | `1000` |
| `LAT1/LAT2/LON1/LON2` | Spatial extent | Global `90/-90/-180/180` |

**To switch datasets you only edit this file — no code changes.** The config supports
`${DATA_ROOT}` variable expansion and alias keys (see `cnp_data_input_parse.py`).

---

## 5. How to Run

```bash
mkdir -p logs && PYTHONUNBUFFERED=1 python scripts/run_pipeline.py \
  --config-input config/CNP_dataInput_from_LandSim_dataGEN.txt \
  --prepare-forcing --force-rebuild-forcing \
  --build all --assemble --forcing-mode datm \
  2>&1 | tee "logs/run_$(date +%Y%m%d_%H%M%S).log"
```

Meaning of the arguments:

| Argument | Purpose |
|---|---|
| `--config-input` | Which config file to use |
| `--prepare-forcing` | First aggregate raw DATM drives into monthly-mean nc |
| `--force-rebuild-forcing` | Force rebuilding forcing, ignore existing files (add this on first run or when changing data) |
| `--build all` | Build all 13 modules |
| `--assemble` | Run assembly after the build |
| `--forcing-mode datm` | Use the DATM branch |

---

## 6. Where the Results Are

All outputs go under `output2/`, the directory given by `BASE_OUTPUT_ROOT` in the config:

```
output2/
├── forcing_netcdf_datm_1901_2023/      Forcing intermediate files (6 .nc)
├── modular_by_input_v1/
│   ├── artifacts/<module>/batch_XX.pkl  Per-module intermediate artifacts (13 modules)
│   └── manifests/<module>.json          Per-module manifest (source files, row count, batch count)
└── final_dataset/                       ★ The final training dataset
    ├── training_data_batch_01.pkl
    ├── …
    └── training_data_batch_396.pkl       (396 files, ~57 GB total)
```

**`output2/final_dataset/training_data_batch_*.pkl` is the deliverable.**

---

## 7. Validating the Result

```bash
python scripts/validate_final_dataset.py \
  --config-input config/CNP_dataInput_from_LandSim_dataGEN.txt
```

It checks per batch: whether row counts match `A_index_core`, whether coordinates contain NaNs,
whether the 6 forcing columns are arrays of the correct monthly length, etc.
For a quick smoke test, add `--max-batches 3` to check only the first 3 batches.

---

## 8. Common Operations

**If you changed only one input file (e.g. the QBOT drive), no need to rerun everything:**

```bash
# 1) Edit the corresponding path in the config
# 2) Rebuild only that module
python scripts/run_extraction.py \
  --config-input config/CNP_dataInput_from_LandSim_dataGEN.txt \
  --build A_forcing_ds7_qbot
# 3) Re-assemble
python scripts/run_assembly.py
```

**To preprocess forcing only, without building modules yet:** add `--prepare-forcing-only`.

**To run in stages:** `run_extraction.py` (extraction only) → `run_assembly.py` (assembly only).
