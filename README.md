# Modular Final Dataset Construction

This folder contains a modular pipeline for dataset generation.

The design goal is:

- one artifact per input source file,
- rebuild only the changed module,
- reuse all unchanged modules,
- assemble once to produce the final dataset.

## Output Policy

There is only one final output directory:

- `/mnt/DATA/0_oak_data/3_dataset_based_construction/final_dataset/`

Final file naming stays consistent:

- `training_data_batch_01.pkl`
- `training_data_batch_02.pkl`
- ...

## Core Files

- `config.py`: all input paths, variables, and path settings.
- `run_pipeline.py`: module build and final assembly entrypoint.


## Required Modules

- `A_index_core`
- `A_ds1_surface`
- `A_ds2_history_x`
- `A_ds10_restart_x`
- `A_h0_list_y`
- `A_r_list_y`
- `A_forcing_ds4_flds`
- `A_forcing_ds5_psrf`
- `A_forcing_ds6_fsds`
- `A_forcing_ds7_qbot`
- `A_forcing_ds8_prectmms`
- `A_forcing_ds9_tbot`
- `A_clm_params_pft` (required)

## Commands

### 1) Full build + assemble

```bash
python3 /home/UNT/dg0997/all_gdw/0_oak_weather/16_add_4_surf_input_output/5_final_dataset_construction/run_pipeline.py --build all --assemble
```

### 2) Rebuild one module only (example: QBOT)

```bash
python3 /home/UNT/dg0997/all_gdw/0_oak_weather/16_add_4_surf_input_output/5_final_dataset_construction/run_pipeline.py --build A_forcing_ds7_qbot
```

### 3) Assemble only (reuse existing artifacts)

```bash
python3 /home/UNT/dg0997/all_gdw/0_oak_weather/16_add_4_surf_input_output/5_final_dataset_construction/run_pipeline.py --assemble
```

### 4) Rebuild one module and assemble (recommended for updates)

```bash
python3 /home/UNT/dg0997/all_gdw/0_oak_weather/16_add_4_surf_input_output/5_final_dataset_construction/run_pipeline.py --build A_forcing_ds7_qbot --assemble
```

## How to Change Input File Paths

Edit `FILE_PATHS` in `config.py`.

Examples:

- replace QBOT source:
  - `FILE_PATHS["ds7"] = "/abs/path/to/new_QBOT.nc"`
- replace Y inputs:
  - `FILE_PATHS["h0_list"] = ["/abs/path/to/new_h0.nc"]`
  - `FILE_PATHS["r_list"] = ["/abs/path/to/new_r.nc"]`
- replace PFT parameter file:
  - `FILE_PATHS["clm_params"] = "/abs/path/to/new_clm_params.nc"`

After updating paths, rebuild the related module(s), then run assembly.

Notes:

- `A_h0_list_y` now uses one h0 file directly (no averaging).
- `A_r_list_y` now uses one r file directly (no averaging).

## Typical Update Workflow

If only one source file changes (for example `ds7` QBOT):

1. Update `FILE_PATHS["ds7"]` in `config.py`.
2. Rebuild only `A_forcing_ds7_qbot`.
3. Run final assembly.

This avoids full recomputation and keeps all unchanged modules reusable.

