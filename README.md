# Modular Final Dataset Construction

This folder builds training datasets in a modular way:

- one artifact per input source,
- rebuild only changed modules,
- reuse unchanged artifacts,
- assemble once into final `training_data_batch_XX.pkl`.

## New configuration entrypoint

Use one text file as the main config source:

- `config/CNP_dataInput.txt`

It contains:

- variable groups (similar to `CNP_IO_updated9_dev.txt`),
- input paths and file names,
- forcing mode switch (`legacy` / `datm`).



## Scripts

- `scripts/run_extraction.py`: extraction only (build artifacts/manifests)
- `scripts/run_assembly.py`: assembly only (build final dataset from artifacts)
- `scripts/run_pipeline.py`: legacy combined entrypoint (still supported)

## Docs

- `docs/build_index_core_performance_report.md`: short note on how `build_index_core` works and why it can be slow

## Forcing extraction mode

- `FORCING_MODE: legacy`
  - current behavior: read consolidated forcing NetCDF files (`ds4`~`ds9`)
- `FORCING_MODE: datm`
  - DATM/new uELM forcing mode
  - supports direct per-variable path (`DATM_*_PATH`) or monthly scan from `DATM_ROOT` + token map
  - DS4~DS9 coordinates are normalized to `ds2` mesh during preprocessing.

## Commands

### 1) Full extraction + assembly (recommended)

**Legacy mode** (`FORCING_MODE: legacy`):

```bash
python3 scripts/run_extraction.py --build all
python3 scripts/run_assembly.py
```
Or use the one-step approach:
```bash
python3 scripts/run_pipeline.py --build all --assemble

**DATM mode** (`FORCING_MODE: datm`):

```bash
python3 scripts/run_extraction.py --build all --forcing-mode datm
python3 scripts/run_assembly.py
```
Or use the one-step approach:
```bash
python3 scripts/run_pipeline.py --build all --assemble --forcing-mode datm
```


### 1b) Inference (inputs-only)

For inference configs where `H0_LIST_PATHS` / `R_LIST_PATHS` are intentionally omitted, run:

```bash
python3 scripts/run_pipeline.py \
  --config-input config/CNP_TESNORTH_inference.txt \
  --output-name TESNorthERA510PCT \
  --build all --assemble --inference
```

To validate the produced batches:

```bash
python3 scripts/validate_final_dataset.py \
  --config-input config/CNP_TESNORTH_inference.txt \
  --output-name TESNorthERA510PCT \
  --inference \
  --max-batches 1
```

### 2) Rebuild one module only (example: QBOT forcing)

**Legacy mode** (`FORCING_MODE: legacy`):

```bash
python3 scripts/run_extraction.py --build A_forcing_ds7_qbot
```

**DATM mode** (`FORCING_MODE: datm`):

```bash
python3 scripts/run_extraction.py --build A_forcing_ds7_qbot --forcing-mode datm

If only one input file changes (for example QBOT):
1. Update path in `config/CNP_dataInput.txt` (for example `DS7_PATH` or DATM path).
2. Rebuild only the affected module (e.g., `A_forcing_ds7_qbot`).
   - Legacy mode: `python3 scripts/run_extraction.py --build A_forcing_ds7_qbot`
   - DATM mode: `python3 scripts/run_extraction.py --build A_forcing_ds7_qbot --forcing-mode datm`
3. Run assembly: `python3 scripts/run_assembly.py`

```

### 3) Build forcing intermediates (DS4~DS9) from DATM monthly files

This step is **only needed for DATM mode** to preprocess monthly DATM files into consolidated NetCDF files. The intermediates are written to:

- `BASE_OUTPUT_ROOT/forcing_netcdf_datm_<DATM_START_YEAR>_<DATM_END_YEAR>/`

**Build intermediates**:

```bash
python3 scripts/run_extraction.py \
  --forcing-mode datm \
  --prepare-forcing-only
```

**Force rebuild** (if intermediates already exist):

```bash
python3 scripts/run_extraction.py \
  --forcing-mode datm \
  --prepare-forcing-only \
  --force-rebuild-forcing
```

**Note**: This step is usually done automatically when running extraction with `--forcing-mode datm`. Only run this separately if you want to preprocess forcing files before running the full extraction.


