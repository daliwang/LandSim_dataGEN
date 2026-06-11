#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${LANDSIM_CONFIG_FILE:-${PROJECT_ROOT}/config/CNP_dataInput_h0_forcing.txt}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_h0_forcing_0001_0020}"
RECORD_DIR="${PROJECT_ROOT}/records/${RUN_ID}"
LOG_FILE="${RECORD_DIR}/run.log"
SOURCE_FORCING="/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/forcing_TBOT_PBOT_QBOT_FLDS_FSDS_RAIN_SNOW_PRECmms_0001-0020.nc"
REUSE_OUTPUT_ROOT="${REUSE_OUTPUT_ROOT:-${PROJECT_ROOT}/output_v1_with_trigridinterepartion}"
NEW_OUTPUT_ROOT="${NEW_OUTPUT_ROOT:-${PROJECT_ROOT}/output_h0_forcing_0001_0020}"

mkdir -p "${RECORD_DIR}"

{
  echo "run_id: ${RUN_ID}"
  echo "utc_start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root: ${PROJECT_ROOT}"
  echo "config_file: ${CONFIG_FILE}"
  echo "source_forcing: ${SOURCE_FORCING}"
  echo "reuse_output_root: ${REUSE_OUTPUT_ROOT}"
  echo "new_output_root: ${NEW_OUTPUT_ROOT}"
  echo
  echo "[config]"
  sed -n '1,220p' "${CONFIG_FILE}"
  echo
  echo "[source forcing]"
  python3 - <<'PY'
from pathlib import Path
from netCDF4 import Dataset

paths = [
    Path("/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/forcing_TBOT_PBOT_QBOT_FLDS_FSDS_RAIN_SNOW_PRECmms_0001-0020.nc"),
    Path("/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/surfdata_0.25x0.25_simyr1850_c240125_TOP.nc"),
    Path("/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/clm_params_c211124.nc"),
    Path("/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/20240214.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.adsp.elm.r.0021-01-01-00000.nc"),
    Path("/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/025E3SMV3/20240223.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.fnsp.elm.r.0251-01-01-00000.nc"),
]

for path in paths:
    stat = path.stat()
    print(f"path: {path}")
    print(f"size_bytes: {stat.st_size}")
    print(f"mtime: {stat.st_mtime}")
    with Dataset(path) as ds:
        print("dimensions:")
        for name, dim in list(ds.dimensions.items())[:10]:
            print(f"  {name}: {len(dim)}")
        if path.name.startswith("forcing_"):
            print("variables:")
            for name in ("TBOT", "PBOT", "QBOT", "FLDS", "FSDS", "RAIN", "SNOW", "PRECmms"):
                var = ds.variables[name]
                print(f"  {name}: dims={var.dimensions}, shape={var.shape}, units={getattr(var, 'units', '')}")
    print()
PY
  echo
  echo "[git]"
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  echo
  echo "[commands]"
  echo "LANDSIM_CONFIG_FILE=${CONFIG_FILE} python3 scripts/run_extraction.py --build A_index_core A_ds1_surface A_ds2_history_x A_ds10_restart_x A_r_list_y A_clm_params_pft A_forcing_ds4_flds A_forcing_ds5_psrf A_forcing_ds6_fsds A_forcing_ds7_qbot A_forcing_ds8_prectmms A_forcing_ds9_tbot"
  echo "LANDSIM_CONFIG_FILE=${CONFIG_FILE} python3 scripts/run_assembly.py"
  echo
} | tee "${LOG_FILE}"

export LANDSIM_CONFIG_FILE="${CONFIG_FILE}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

python3 scripts/run_extraction.py --build \
  A_index_core \
  A_ds1_surface \
  A_ds2_history_x \
  A_ds10_restart_x \
  A_r_list_y \
  A_clm_params_pft \
  A_forcing_ds4_flds \
  A_forcing_ds5_psrf \
  A_forcing_ds6_fsds \
  A_forcing_ds7_qbot \
  A_forcing_ds8_prectmms \
  A_forcing_ds9_tbot 2>&1 | tee -a "${LOG_FILE}"

python3 scripts/run_assembly.py 2>&1 | tee -a "${LOG_FILE}"

{
  echo
  echo "utc_end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "record_dir: ${RECORD_DIR}"
} | tee -a "${LOG_FILE}"
