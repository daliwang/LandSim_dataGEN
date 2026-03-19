# Inference artifacts contents report

## Where this report comes from

This report summarizes the current contents of:

- `output/TESNORTNERA5inference/modular_by_input_v1/artifacts/`

It is based on inspection of the artifact pickle files currently present on disk.

## Artifact modules found

These artifact subdirectories exist under `artifacts/`:

- `A_index_core`
- `A_ds1_surface`
- `A_ds2_history_x`
- `A_ds10_restart_x`
- `A_forcing_ds4_flds`
- `A_forcing_ds5_psrf`
- `A_forcing_ds6_fsds`
- `A_forcing_ds7_qbot`
- `A_forcing_ds8_prectmms`
- `A_forcing_ds9_tbot`
- `A_h0_list_y`
- `A_r_list_y`
- `A_clm_params_pft`

## Which modules have data right now

Counting `batch_*.pkl` files per module:

- `A_index_core`: `130` batches (uses `index_master.pkl` + implicit batching)
- `A_ds1_surface`: `130` batches (`batch_01.pkl` .. `batch_130.pkl`)
- `A_ds2_history_x`: `130` batches
- `A_ds10_restart_x`: `130` batches
- `A_clm_params_pft`: `0` batches
- `A_forcing_ds4_flds` .. `A_forcing_ds9_tbot`: `0` batches
- `A_h0_list_y`: `0` batches
- `A_r_list_y`: `0` batches

So this inference output currently contains the “inputs-only” parts (index + surface + history + restart), but not the forcing series nor the target/label modules.

## Detailed structure (representative files)

### `A_index_core/index_master.pkl`

- Rows: `259535`
- Columns (9):
  - `__row_id`, `batch_id`
  - `lat_idx`, `lon_idx`
  - `Latitude`, `Longitude`
  - `nearest_restart_index`, `nearest_forcing_index`
  - `gridcell_id`

Representative sample (first row) shows:
- numeric coordinate fields (`Latitude`, `Longitude`)
- integer mapping/index fields used as the join backbone for later modules

### `A_ds1_surface/batch_01.pkl`

- Shape: `(2000, 13)`
- Columns:
  - `__row_id` plus surface properties
  - Includes `LANDFRAC_PFT`, `PCT_NATVEG`, `AREA`, `SOIL_COLOR`, `SOIL_ORDER`, `OCCLUDED_P`, `SECONDARY_P`, `LABILE_P`, `APATITE_P`
  - Includes `PCT_SAND`, `PCT_CLAY`, `PCT_NAT_PFT`

Value type pattern:
- all columns inspected are scalar floats for each row

### `A_ds2_history_x/batch_01.pkl`

- Shape: `(2000, 6)`
- Columns:
  - `__row_id`, `landfrac`, and climate scalar fields: `GPP`, `HR`, `AR`, `NPP`

Value type pattern:
- scalar floats per row

### `A_ds10_restart_x/batch_01.pkl`

- Shape: `(2000, 94)`
- Columns:
  - `__row_id` plus many restart variables
  - Many restart variables are vertical/profile-like values stored as `list` per row (example variables observed: `totvegc`, `leafc`, `deadstemn`, etc.)

Value type pattern:
- profile-like restart variables appear as Python lists in the dataframe cells

## Why forcing and labels are currently missing

In this output folder, the forcing modules (`A_forcing_ds4_flds` .. `A_forcing_ds9_tbot`) and the label/target modules (`A_h0_list_y`, `A_r_list_y`) have `0` `batch_*.pkl` files.

That means:
- `training_data_batch_XX.pkl` (if produced) may not have monthly forcing-series columns (`FLDS`, `PSRF`, ...) and may not have `Y_*` targets.

## Quick local inspection commands

You can re-run similar checks with:

```bash
python3 - <<'PY'
import os, pandas as pd

base="output/TESNORTNERA5inference/modular_by_input_v1/artifacts"
print("Modules:", sorted(os.listdir(base)))

df=pd.read_pickle(os.path.join(base,"A_index_core","index_master.pkl"))
print("A_index_core rows:", len(df))
print("A_index_core columns:", list(df.columns))

for mod in ["A_ds1_surface","A_ds2_history_x","A_ds10_restart_x"]:
    p=os.path.join(base,mod,"batch_01.pkl")
    d=pd.read_pickle(p)
    print(mod,"shape",d.shape,"columns",len(d.columns))
PY
```

