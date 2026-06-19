#!/usr/bin/env bash
# Fast path for remaining h0 forcing modules (ds5-ds9) + assembly.
#
# Why: legacy (time, lat, lon) reads take ~13h per variable. Preprocessing to
# (time, gridcell) once, then vectorized bulk extraction, typically finishes all
# remaining forcing modules in under an hour on a compute node.
#
# Prerequisites:
#   - A_index_core through A_forcing_ds4_flds already complete (or use full resume)
#
# Usage on compute node:
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/run_h0_forcing_remaining_fast.sh
#
# Optional:
#   SKIP_PREPROCESS=1 bash scripts/run_h0_forcing_remaining_fast.sh
#   VALIDATE=1 bash scripts/run_h0_forcing_remaining_fast.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_CONFIG="${PROJECT_ROOT}/config/CNP_dataInput_h0_forcing.txt"
GRIDCELL_CONFIG="${PROJECT_ROOT}/config/CNP_dataInput_h0_forcing_gridcell.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"
VALIDATE="${VALIDATE:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

cd "${PROJECT_ROOT}"

if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  echo "[1/3] Preprocess h0 forcing to (time, gridcell)..."
  "${PYTHON_BIN}" scripts/preprocess_h0_forcing_gridcell.py \
    --config-input "${BASE_CONFIG}" \
    --mode "${PREPROCESS_MODE:-memory}" \
    --overwrite
else
  echo "[1/3] Skipping preprocess (SKIP_PREPROCESS=1)"
fi

echo "[2/3] Build remaining forcing modules with vectorized legacy path..."
"${PYTHON_BIN}" scripts/run_pipeline.py \
  --config-input "${GRIDCELL_CONFIG}" \
  --build \
    A_forcing_ds5_psrf \
    A_forcing_ds6_fsds \
    A_forcing_ds7_qbot \
    A_forcing_ds8_prectmms \
    A_forcing_ds9_tbot \
  --forcing-mode legacy

echo "[3/3] Assemble final dataset..."
"${PYTHON_BIN}" scripts/run_pipeline.py \
  --config-input "${GRIDCELL_CONFIG}" \
  --assemble

if [[ "${VALIDATE}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/validate_final_dataset.py \
    --config-input "${GRIDCELL_CONFIG}"
fi

echo "Done."
