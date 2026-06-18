#!/usr/bin/env bash
# Build the 0.25 degree h0-forcing training dataset (vectorized pipeline).
#
# Recommended: run on a compute node, not the login node.
#
# Quick start (interactive compute node):
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/run_h0_forcing_build.sh
#
# Submit to SLURM (recommended):
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/submit_h0_forcing_build.sh
#
# Detached on a compute node:
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   nohup bash scripts/run_h0_forcing_build.sh > logs/nohup.out 2>&1 &
#   echo $! > logs/build.pid
#
# Resume an interrupted build (default; skips existing artifact batches):
#   BUILD_MODE=resume nohup bash scripts/run_h0_forcing_build.sh > logs/nohup_resume.out 2>&1 &
#
# Full rebuild from scratch (rebuilds all modules; still skips per-batch files unless
# you delete output_h0_forcing_0001_0020/modular_by_input_v1/artifacts first):
#   BUILD_MODE=full nohup bash scripts/run_h0_forcing_build.sh > logs/nohup_full.out 2>&1 &
#
# Optional environment overrides:
#   PYTHON_BIN=python3
#   RUN_ID=manual_h0_forcing_0001_0020
#   VALIDATE=1                 # run validate_final_dataset.py after assembly
#   MODULE_INIT='module load python/3.11'   # site-specific module setup
#   CONDA_ENV=myenv            # conda activate before build
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/config/CNP_dataInput_h0_forcing.txt}"
BUILD_MODE="${BUILD_MODE:-resume}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_h0_forcing_0001_0020}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VALIDATE="${VALIDATE:-0}"

LOG_DIR="${PROJECT_ROOT}/logs"
RECORD_DIR="${PROJECT_ROOT}/records/${RUN_ID}"
LOG_FILE="${RECORD_DIR}/run.log"
ARTIFACT_ROOT="${PROJECT_ROOT}/output_h0_forcing_0001_0020/modular_by_input_v1/artifacts"
FINAL_DIR="${PROJECT_ROOT}/output_h0_forcing_0001_0020/final_dataset"

SOURCE_FORCING="/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/forcing_TBOT_PBOT_QBOT_FLDS_FSDS_RAIN_SNOW_PRECmms_0001-0020.nc"
DATA_ROOT="/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/data_from_xiaoying"

RESUME_MODULES=(
  A_ds10_restart_x
  A_r_list_y
  A_clm_params_pft
  A_forcing_ds4_flds
  A_forcing_ds5_psrf
  A_forcing_ds6_fsds
  A_forcing_ds7_qbot
  A_forcing_ds8_prectmms
  A_forcing_ds9_tbot
)

FULL_MODULES=(
  A_index_core
  A_ds1_surface
  A_ds2_history_x
  A_ds10_restart_x
  A_r_list_y
  A_clm_params_pft
  A_forcing_ds4_flds
  A_forcing_ds5_psrf
  A_forcing_ds6_fsds
  A_forcing_ds7_qbot
  A_forcing_ds8_prectmms
  A_forcing_ds9_tbot
)

log() {
  echo "$*" | tee -a "${LOG_FILE}"
}

artifact_progress() {
  local label="$1"
  log ""
  log "[artifact progress: ${label}]"
  if [[ ! -d "${ARTIFACT_ROOT}" ]]; then
    log "  (no artifact directory yet)"
    return
  fi
  for module_dir in "${ARTIFACT_ROOT}"/*/; do
    [[ -d "${module_dir}" ]] || continue
    local module_name
    module_name="$(basename "${module_dir}")"
    local count
    count="$(find "${module_dir}" -maxdepth 1 -name 'batch_*.pkl' | wc -l | tr -d ' ')"
    log "  ${module_name}: ${count}/396"
  done
  local final_count
  final_count="$(find "${FINAL_DIR}" -maxdepth 1 -name 'training_data_batch_*.pkl' 2>/dev/null | wc -l | tr -d ' ')"
  log "  final_dataset: ${final_count}/396"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input file: ${path}" >&2
    exit 1
  fi
}

activate_runtime() {
  if [[ -n "${MODULE_INIT:-}" ]]; then
    # shellcheck disable=SC1090
    eval "${MODULE_INIT}"
  fi
  if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
}

check_python_deps() {
  "${PYTHON_BIN}" - <<'PY'
import importlib

required = ("numpy", "pandas", "scipy", "netCDF4", "xarray")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("Python dependencies OK")
PY
}

select_modules() {
  case "${BUILD_MODE}" in
    resume)
      MODULES=("${RESUME_MODULES[@]}")
      ;;
    full)
      MODULES=("${FULL_MODULES[@]}")
      ;;
    *)
      echo "Unknown BUILD_MODE=${BUILD_MODE}. Use 'resume' or 'full'." >&2
      exit 1
      ;;
  esac
}

mkdir -p "${LOG_DIR}" "${RECORD_DIR}" "${FINAL_DIR}"
: > "${LOG_FILE}"

log "run_id: ${RUN_ID}"
log "utc_start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "hostname: $(hostname)"
log "project_root: ${PROJECT_ROOT}"
log "config_file: ${CONFIG_FILE}"
log "build_mode: ${BUILD_MODE}"
log "python_bin: ${PYTHON_BIN}"
log "source_forcing: ${SOURCE_FORCING}"
log "artifact_root: ${ARTIFACT_ROOT}"
log "final_dir: ${FINAL_DIR}"
log ""

require_file "${CONFIG_FILE}"
require_file "${SOURCE_FORCING}"
require_file "${DATA_ROOT}/surfdata_0.25x0.25_simyr1850_c240125_TOP.nc"
require_file "${DATA_ROOT}/clm_params_c211124.nc"
require_file "${DATA_ROOT}/20240214.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.adsp.elm.r.0021-01-01-00000.nc"
require_file "${DATA_ROOT}/20240223.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.fnsp.elm.r.0251-01-01-00000.nc"

activate_runtime
check_python_deps | tee -a "${LOG_FILE}"

select_modules
log "modules: ${MODULES[*]}"
artifact_progress "before build"

export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

log ""
log "[command]"
log "${PYTHON_BIN} scripts/run_pipeline.py \\"
log "  --config-input ${CONFIG_FILE} \\"
log "  --build ${MODULES[*]} \\"
log "  --assemble --forcing-mode legacy"
log ""

"${PYTHON_BIN}" scripts/run_pipeline.py \
  --config-input "${CONFIG_FILE}" \
  --build "${MODULES[@]}" \
  --assemble \
  --forcing-mode legacy 2>&1 | tee -a "${LOG_FILE}"

artifact_progress "after build"

if [[ "${VALIDATE}" == "1" ]]; then
  log ""
  log "[validation]"
  "${PYTHON_BIN}" scripts/validate_final_dataset.py \
    --config-input "${CONFIG_FILE}" 2>&1 | tee -a "${LOG_FILE}"
fi

log ""
log "utc_end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "record_dir: ${RECORD_DIR}"
log "log_file: ${LOG_FILE}"
log "done."
