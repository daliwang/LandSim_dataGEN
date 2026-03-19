# `A_index_core` (`build_index_core`) Performance Report

## Scope

This report explains how `build_index_core` in `scripts/run_pipeline.py` works and why it can be slow for large domains.

## What the function does

1. Opens three NetCDF inputs (`ds2`, `ds10`, `ds4`).
2. Reads target-region lat/lon ranges from `ds2` and filters by `landmask == 1`.
3. Builds query coordinates `(lat, lon)` for each selected land grid point.
4. Builds a KD-tree on restart coordinates from `ds10` and queries nearest restart index.
5. Builds a KD-tree on forcing coordinates from `ds4` and queries nearest forcing index.
6. Constructs `index_master` rows with metadata:
   - row id, batch id
   - source indices (`lat_idx`, `lon_idx`)
   - coordinates (`Latitude`, `Longitude`)
   - mapped indices (`nearest_restart_index`, `nearest_forcing_index`)
   - `gridcell_id`
7. Writes one master pickle and per-batch pickles.

## Why it is slow

### 1) Large point count drives total runtime

Runtime scales with the number of filtered land points `N`.
If the lat/lon window is large, `N` can be very large, and all downstream work scales with it.

### 2) Python loop + list-of-dicts construction is expensive

The function builds rows one-by-one in Python and appends dictionaries to a list before creating a DataFrame.
This creates heavy Python object overhead and often dominates runtime at scale.

### 3) DataFrame materialization cost

`pd.DataFrame(rows)` on a huge list-of-dicts incurs dtype inference and large memory allocation.
If memory pressure occurs, performance drops further due to garbage collection and potential paging.

### 4) Two nearest-neighbor queries per point

Each selected point runs:
- one nearest-neighbor query against restart grid
- one nearest-neighbor query against forcing grid

`cKDTree` is efficient, but the total cost is still substantial when `N` is large.

### 5) I/O is not usually the main bottleneck here

The helper `_read_forcing_coords_2d` avoids reading full time stacks and only loads one spatial slice as needed.
That keeps I/O relatively controlled; CPU/object-construction work is usually the bigger cost.

## Practical optimization directions

1. Replace row-wise dict appends with vectorized array-based DataFrame construction.
2. Avoid Python loops where possible when assigning columns.
3. Keep arrays in NumPy dtypes and build DataFrame in one shot.
4. Profile with different domain sizes (`N`) to confirm scaling and bottleneck shifts.

## Expected impact

For large regions, vectorizing row construction generally provides the biggest speedup.
The nearest-neighbor searches often become a smaller fraction once Python object overhead is reduced.
