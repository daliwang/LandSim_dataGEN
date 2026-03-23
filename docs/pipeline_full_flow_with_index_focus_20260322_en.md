# LandSim_dataGEN_Brench1 Full Pipeline Guide (Detailed, with A_index_core Focus)

## 1. Purpose

This document focuses on two questions:

1. What the pipeline does from raw inputs to final training datasets.
2. Why `A_index_core` is the central module, how it works internally, and how its fields drive all downstream modules.

---

## 2. End-to-End Pipeline Overview

`scripts/run_pipeline.py` can be viewed as four layers:

1. **Config loading layer**: parse `CNP_dataInput` and build runtime config.
2. **Preparation layer**: create directories and optionally preprocess forcing.
3. **Module build layer**: generate module artifacts `batch_XX.pkl`.
4. **Assembly layer**: merge all modules by primary key `__row_id` and export training data.

Typical execution chain:

1. `--prepare-forcing` (optional)  
2. `--build ...` (at least includes `A_index_core`)  
3. `--assemble`

---

## 3. Inputs, Outputs, and Directory Layout

### 3.1 Main Input Sources

- `DS1`: surface data (static land/surface attributes)
- `DS2`: history grid data (optional; can fall back if missing)
- `DS10`: restart state data (required)
- `H0_LIST`: target grid files for `A_h0_list_y` (can be missing; module will skip)
- `R_LIST`: target restart files for `A_r_list_y` (required if that module is executed)
- `DATM_ROOT`: raw forcing directory (DATM mode)

### 3.2 Intermediate Artifacts

Module artifacts are written to:

- `BASE_OUTPUT_ROOT/modular_by_input_v1/artifacts/<module>/batch_XX.pkl`

### 3.3 Final Outputs

- `BASE_OUTPUT_ROOT/final_dataset/training_data_batch_XX.pkl`

---

## 4. Runtime Arguments (Key Options)

- `--config-input`: path to config file
- `--build all` or `--build A_xxx ...`: modules to build
- `--assemble`: run final merge
- `--prepare-forcing`: preprocess DATM forcing
- `--force-rebuild-forcing`: force rebuilding forcing intermediate nc files
- `--forcing-mode datm|legacy`: forcing branch selection

---

## 5. Why A_index_core Is the Pipeline Core

`A_index_core` does not extract model features directly.  
It defines the unified indexing contract for the entire pipeline.

All downstream modules depend on this contract to avoid cross-source misalignment.

It defines three critical components:

1. **Sample set** (which grid cells are included)
2. **Sample identity** (unique row key and batch assignment)
3. **Cross-source mapping** (how samples map to restart/forcing grids)

---

## 6. A_index_core Internal Workflow (Detailed)

### 6.1 Read Paths and Validate Required Inputs

The module reads:

- `ds2_path`
- `ds10_path`
- `ds1_path`

`ds10` must exist, otherwise the module fails immediately (restart mapping and `gridcell_id` depend on it).

### 6.2 Build Spatial Candidate Grid (Prefer ds2, fallback to ds1)

Two cases:

1. If `ds2` exists: use `ds2` `lat/lon/landmask`.
2. If `ds2` is missing: fall back to `ds1` (`LATIXY/LONGXY`, mask from `LANDFRAC_PFT` when available).

Filtering conditions:

- `landmask == 1`
- `LAT2 <= lat <= LAT1`
- `LON1 <= lon <= LON2`

Outputs:

- `filtered_coordinates`: per-sample grid index tuple `(i,j)` (or `(idx,idx)` placeholder for 1D mesh)
- `query_coords`: per-sample geographic coordinate `(lat, lon)`

### 6.3 Compute Restart Mapping (Nearest Neighbor)

Build `cKDTree` from `ds10.grid1d_lat/lon`, query with `query_coords`, and obtain:

- `nearest_restart_index`

This defines sample-to-restart 1D indexing.

### 6.4 Compute Forcing Mapping (DATM Branch)

Current DATM logic uses a “strict + safe fallback” design:

1. same-mesh exact match (coordinate key matching with longitude normalization)
2. if that fails, fallback to forcing-grid `cKDTree` nearest neighbor

If `ds4` is missing in DATM mode, current code raises an error and requires forcing preprocessing first.

### 6.5 Build `index_master` Rows

For each selected sample, generate one row with:

- `__row_id`
- `batch_id`
- `lat_idx`, `lon_idx`
- `Latitude`, `Longitude`
- `nearest_restart_index`, `nearest_forcing_index`
- `gridcell_id = nearest_restart_index + 1`

### 6.6 Persist Full Index and Batch Slices

- Write full table: `A_index_core/index_master.pkl`
- Write per-batch slices: `A_index_core/batch_XX.pkl`
- Write manifest: source files, row count, batch count

---

## 7. Engineering Meaning of `index_master.pkl` Fields

### 7.1 `__row_id`

- Global unique primary key
- Merge key during assembly
- Defines “same sample row” across all modules

### 7.2 `batch_id`

- Derived from `BATCH_SIZE`
- Enables chunked processing and lower memory peaks
- Useful for parallelization and retry

### 7.3 `lat_idx`, `lon_idx`

- Direct index location for surface/history/h0 grid variables
- Primary indexing fields for static/grid-based modules

### 7.4 `Latitude`, `Longitude`

- Physical sample coordinates
- Query points for forcing spatial remapping
- Interpretable spatial identifiers in final data

### 7.5 `nearest_restart_index`

- 1D nearest mapping to restart grid
- Drives `gridcell_id`
- Entry index for restart-related extraction

### 7.6 `nearest_forcing_index`

- 1D mapping to forcing grid
- In DATM, forcing modules may further remap by coordinates
- Still useful for debugging, validation, and compatibility paths

### 7.7 `gridcell_id`

- `nearest_restart_index + 1`
- Matches restart file gridcell ID conventions
- Used by `A_ds10_restart_x` / `A_r_list_y` for PFT/COL extraction

---

## 8. Dependency of Downstream Modules on A_index_core

### 8.1 `A_ds1_surface`

- Depends on: `lat_idx/lon_idx`
- Purpose: static surface input features

### 8.2 `A_ds2_history_x`

- Depends on: `lat_idx/lon_idx`
- Purpose: history grid state features (if `ds2` exists)

### 8.3 `A_h0_list_y`

- Depends on: `lat_idx/lon_idx`
- Purpose: target grid variables (`Y_*`)

### 8.4 `A_ds10_restart_x`

- Depends on: `gridcell_id`
- Purpose: structured restart state (PFT/COL)

### 8.5 `A_r_list_y`

- Depends on: `gridcell_id`
- Purpose: target restart state (`Y_*`)

### 8.6 `A_forcing_*`

- Depends on: `Latitude/Longitude` (core)
- Purpose: forcing time-series extraction on forcing grid mapping

---

## 9. DATM Forcing Preprocessing and Its Relationship to A_index_core

`prepare_forcing_inputs_from_datm()` standardizes forcing format:

- Aggregate many raw DATM files into consistent intermediate forcing nc files
- Provide stable forcing inputs for `A_forcing_*`
- Ensure forcing grid coordinates are available for mapping

Recommended order:

1. prepare forcing
2. build index/core modules
3. assemble

---

## 10. How Assembly Uses A_index_core

For each batch, assembly starts with `A_index_core` and merges all modules by `__row_id`:

1. `A_index_core` provides row key and spatial columns
2. Merge non-forcing modules
3. Merge forcing modules (with time-series handling)
4. Write final batch training file

As long as each module preserves `__row_id`, cross-module alignment remains stable.

---

## 11. Common Issues and Debugging Checklist (A_index_core-Centric)

### 11.1 Abnormal `selected cells`

Check:

- domain bounds (`LAT1/LAT2/LON1/LON2`)
- landmask construction
- expected input grid shape/convention

### 11.2 same-mesh mapping failure

This is recoverable by forcing KDTree fallback.  
Use logged mean/p95/max distance to assess mapping quality.

### 11.3 Row misalignment after assembly

Check:

- whether all modules preserve `__row_id`
- whether `batch_id` is continuous
- whether all modules use the same `index_master.pkl`

---

## 12. Recommended Stable Command

```bash
python scripts/run_pipeline.py \
  --config-input config/CNP_dataInput_from_LandSim_dataGEN.txt \
  --prepare-forcing --force-rebuild-forcing \
  --build all --assemble --forcing-mode datm
```

Notes:

- First two flags ensure complete forcing intermediates
- `build all` keeps index and modules consistent
- `assemble` merges under one consistent artifact set

---

## 13. Conclusion

`A_index_core` is the alignment protocol generator of the full pipeline.  
It fixes sample definition, batch partition, and cross-source mappings once, so all downstream modules can be merged under a single, stable sample coordinate system to produce trainable datasets.

