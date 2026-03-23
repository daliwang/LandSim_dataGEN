# LandSim_dataGEN_Brench1 Vectorization Guide (Vectorization-Only)

## 1. What Vectorization Means

Vectorization means changing computation from “processing one record at a time in loops” to “processing whole arrays in batches at once”.

In Python data workflows, this usually means:

- reducing Python-level `for` loops,
- using `numpy` / `xarray` / lower-level array interfaces for bulk operations,
- pushing most heavy computation into efficient low-level implementations.

---

## 2. Vectorization Goals in This Project

The core goals are:

1. use batch indexing in module processing to reduce per-sample access;
2. use array-column mapping instead of row-wise dictionary lookup;
3. keep output format unchanged (`batch_XX.pkl`) while optimizing execution paths.

---

## 3. Implemented Vectorization Areas

### 3.1 Forcing Extraction Modules (`A_forcing_*`)

Modules covered:

- `A_forcing_ds4_flds`
- `A_forcing_ds5_psrf`
- `A_forcing_ds6_fsds`
- `A_forcing_ds7_qbot`
- `A_forcing_ds8_prectmms`
- `A_forcing_ds9_tbot`

Implementation pattern:

1. extract a forcing index vector `idx_arr` for each batch;
2. perform one bulk gather on forcing arrays;
3. directly build a 2D result matrix and write into DataFrame.

Key shape conventions:

- when `grid_axis == 1`, read as `(time, grid)`, gather in batch, then transpose to `(sample, time)`;
- when `grid_axis == 0`, read as `(grid, time)`, gather directly to `(sample, time)`.

---

### 3.2 Forcing Index Mapping as a DataFrame Column

Implementation pattern:

- store sample-to-forcing mapping in one full column `_forcing_idx`;
- after batching, read this column as a numpy array for bulk indexing;
- avoid row-wise dictionary lookups.

---

### 3.3 Safe Vectorization in Surface / History Modules

Modules involved:

- `A_ds1_surface`
- `A_ds2_history_x`
- `A_h0_list_y`

Vectorized parts:

- 1D indexing paths are batch-read (e.g., `arr[lat_idx]`, `arr[0, lat_idx]`).

Kept as point-wise reads:

- accesses using paired 2D coordinates `(lat_idx, lon_idx)` remain point-wise to ensure compatibility and stable behavior.

---

## 4. Dataflow After Vectorization

Typical vectorized batch flow:

1. read current batch index arrays from `index_master`;
2. bulk-gather all sample values from source variables;
3. form a 2D matrix (`sample x feature` or `sample x time`);
4. write to `batch_XX.pkl`.

The key idea is: vectorize indices first, then bulk read, then write in unified batches.

---

## 5. Current Vectorization Boundary

Current version implements “main-path vectorization” with compatibility safeguards:

- fully vectorized: forcing module batch time-series extraction;
- partially vectorized: 1D indexing in surface/history modules;
- pending further vectorization: restart paths with variable-length mappings.

---

## 6. Next Vectorization Directions

Potential next steps:

1. grouped gather vectorization for `A_ds10_restart_x`;
2. grouped gather vectorization for `A_r_list_y`;
3. “block read + numpy advanced indexing” for 2D coordinate access;
4. combine with batch-based job splitting to amplify throughput gains.

---

## 7. Run and Validation Suggestions

Without changing existing commands, monitor these in logs:

- total module runtime,
- whether per-batch progress advances linearly,
- whether forcing module batch outputs are continuous,
- whether final assembly output structure remains consistent.

---

## 8. Summary

The project now follows a consistent vectorization pattern:

- index vectors as the control core,
- bulk array gather as the execution core,
- batch files as stable output boundaries.

This pattern can be extended to more modules for higher end-to-end throughput.

