# h0-Derived Forcing Rebuild Workflow

This workflow replaces the previous interpolated atmospheric forcing with:

`/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/forcing_TBOT_PBOT_QBOT_FLDS_FSDS_RAIN_SNOW_PRECmms_0001-0020.nc`

The local 025E3SMV3 non-forcing source files recorded for repeatability are:

- `surfdata_0.25x0.25_simyr1850_c240125_TOP.nc`
- `clm_params_c211124.nc`
- `20240214.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.adsp.elm.r.0021-01-01-00000.nc`
- `20240223.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.fnsp.elm.r.0251-01-01-00000.nc`

The non-forcing columns are reused from previous final pickle files under:

`/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN/output_v1_with_trigridinterepartion`

## Files Added For Repeatability

- `config/CNP_dataInput_h0_forcing.txt`
  - Dedicated config for the h0-derived forcing run.
  - Points `DS1_PATH`, `DS2_PATH`, `DS10_PATH`, `R_LIST_PATHS`, and `CLM_PARAMS_PATH` to local 025E3SMV3 files.
  - Points `DS4_PATH` through `DS9_PATH` to the same h0 forcing NetCDF.
  - Uses `BASE_OUTPUT_ROOT: output_h0_forcing_0001_0020`.

- `scripts/rebuild_h0_forcing_training.sh`
  - Timestamped driver for the forcing rebuild.
  - Calls `scripts/recreate_h0_forcing_pickles.py`.
  - Writes run records to `records/<RUN_ID>/run.log`.
  - Records config contents, forcing metadata, git status, commands, and pickle recreation logs.

- `scripts/recreate_h0_forcing_pickles.py`
  - Reads previous `training_data_batch_*.pkl` files.
  - Keeps all non-forcing columns unchanged.
  - Replaces only `FLDS`, `PSRF`, `FSDS`, `QBOT`, `PRECTmms`, and `TBOT`.

## Variable Mapping

The pipeline training schema keeps the previous forcing column names:

- `FLDS` -> `FLDS`
- `PBOT` -> `PSRF`
- `FSDS` -> `FSDS`
- `QBOT` -> `QBOT`
- `PRECmms` -> `PRECTmms`
- `TBOT` -> `TBOT`

`PRECmms` was created upstream as `RAIN + SNOW`.

## Run Command

From the `LandSim_dataGEN` directory:

```bash
bash scripts/rebuild_h0_forcing_training.sh
```

Or with an explicit run id:

```bash
RUN_ID=manual_h0_forcing_0001_0020 bash scripts/rebuild_h0_forcing_training.sh
```

The script writes new pickle files to:

`output_h0_forcing_0001_0020/final_dataset/`

## Reusing Previous Pickle Files

The driver reads old final pickles from:

`output_v1_with_trigridinterepartion/final_dataset/`

It preserves every column except the six forcing columns. This avoids rebuilding PFT,
soil, parameter, restart, and history data.

If the old final pickle files are in a different location:

```bash
OLD_FINAL_DIR=/path/to/old/final_dataset bash scripts/rebuild_h0_forcing_training.sh
```

The script also writes `h0_forcing_pickle_recreation_manifest.json` into the run record directory.
