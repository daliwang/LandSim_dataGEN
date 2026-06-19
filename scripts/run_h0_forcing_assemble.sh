#!/usr/bin/env bash
# Re-assemble final_dataset from existing modular artifacts (no rebuild).
#
# Use after fixing assembly logic or when forcing/surface modules are already built.
#
# Usage on compute node:
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/run_h0_forcing_assemble.sh
#
# Optional:
#   VALIDATE=1 bash scripts/run_h0_forcing_assemble.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRIDCELL_CONFIG="${PROJECT_ROOT}/config/CNP_dataInput_h0_forcing_gridcell.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"
VALIDATE="${VALIDATE:-0}"

cd "${PROJECT_ROOT}"

echo "[1/2] Assemble final dataset from existing artifacts..."
"${PYTHON_BIN}" scripts/run_pipeline.py \
  --config-input "${GRIDCELL_CONFIG}" \
  --assemble

if [[ "${VALIDATE}" == "1" ]]; then
  echo "[2/2] Validate assembled final dataset..."
  "${PYTHON_BIN}" scripts/validate_final_dataset.py \
    --config-input "${GRIDCELL_CONFIG}"
else
  echo "[2/2] Skipping validation (set VALIDATE=1 to enable)"
fi

echo "Done."
