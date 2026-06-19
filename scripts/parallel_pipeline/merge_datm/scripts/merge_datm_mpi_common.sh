# shellcheck shell=bash
# MPI map-reduce body for merge_datm SLURM launchers under parallel_pipeline/merge_datm/scripts/
#
# Strategy: srun launches one MPI rank per CPU task. Ranks shard source files
# evenly; partial monthly sums are Allreduce'd; rank 0 writes 6 NetCDF files.
#
# Logs:
#   025deg/scripts/logs/merge_datm_x5_phys_sbatch_<JOBID>.{out,err}
#   025deg/scripts/logs/merge_datm_job_<JOBID>/mpi_<jobid>_r<rank>.out|err
#   025deg/scripts/logs/merge_datm_job_<JOBID>/timing.json
#   025deg/scripts/logs/merge_datm_job_<JOBID>/shell_timing.json

set -euo pipefail
trap 'echo "[merge_datm_mpi] ERROR: failed at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

SCRIPT_DIR="${SCRIPT_DIR:?SCRIPT_DIR not set — source merge_datm_paths.sh first}"
: "${MERGE_DATM_PIPELINE_SCRIPTS:?MERGE_DATM_PIPELINE_SCRIPTS not set}"
: "${MERGE_DATM_LEGACY_SCRIPT_DIR:?MERGE_DATM_LEGACY_SCRIPT_DIR not set}"
N_NODES="${N_NODES:?N_NODES not set by launcher}"

JOB_ID="${SLURM_JOB_ID:-${SLURM_BATCH_JOB_ID:-unknown}}"
LOG_ROOT="${MERGE_DATM_PROJECT_ROOT}/scripts/logs"
RUN_LOG_DIR="${LOG_ROOT}/merge_datm_job_${JOB_ID}"
mkdir -p "${RUN_LOG_DIR}" "${LOG_ROOT}"

CONFIG_INPUT="${CONFIG_INPUT:-${MERGE_DATM_PROJECT_ROOT}/config/CNP_merge_datm_to_striped.txt}"
FORCING_OUT_DIR="${FORCING_OUT_DIR:-${SCRATCH:-${HOME}}/landsim_inputs/striped/forcing_netcdf_datm_1901_2023}"
FORCE_REBUILD="${FORCE_REBUILD:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

_ts() { date +%s.%N 2>/dev/null || date +%s; }
T_SCRIPT_START="$(_ts)"

echo "[merge_datm_mpi] start $(date -Is 2>/dev/null || date) job=${JOB_ID} nodes=${N_NODES}"
echo "[merge_datm_mpi] CONFIG_INPUT=${CONFIG_INPUT}"
echo "[merge_datm_mpi] FORCING_OUT_DIR=${FORCING_OUT_DIR}"
echo "[merge_datm_mpi] FORCE_REBUILD=${FORCE_REBUILD} ntasks=${SLURM_NTASKS:-?}"
echo "[merge_datm_mpi] RUN_LOG_DIR=${RUN_LOG_DIR}"
export RUN_LOG_DIR

set +e
# shellcheck source=/dev/null
source "${MERGE_DATM_LEGACY_SCRIPT_DIR}/env_cray_mpi.sh"
_env_rc=$?
set -e
T_AFTER_ENV="$(_ts)"
if [[ "${_env_rc}" -ne 0 ]]; then
    echo "[merge_datm_mpi] FATAL: env_cray_mpi.sh returned ${_env_rc}" >&2
    exit 1
fi

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[merge_datm_mpi] FATAL: ${PYTHON_BIN} not found" >&2
    exit 1
fi
echo "[merge_datm_mpi] python=${PYTHON_BIN}"
"${PYTHON_BIN}" -V
T_IMPORT_START="$(_ts)"
"${PYTHON_BIN}" -c "from mpi4py import MPI; import netCDF4, xarray; print('[merge_datm_mpi] import ok', MPI.Get_library_version()[:60])" \
    || { echo "[merge_datm_mpi] FATAL: mpi4py/netCDF4/xarray import failed on batch node" >&2; exit 1; }
T_AFTER_IMPORT="$(_ts)"

if [[ ! -f "${CONFIG_INPUT}" ]]; then
    echo "[merge_datm_mpi] FATAL: config not found: ${CONFIG_INPUT}" >&2
    exit 1
fi

mkdir -p "${FORCING_OUT_DIR}"

rebuild_flag=()
if [[ "${FORCE_REBUILD}" == "1" ]]; then
    rebuild_flag=(--force-rebuild)
fi

extra_args=()
if [[ -n "${MERGE_DATM_MPI_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${MERGE_DATM_MPI_EXTRA_ARGS})
fi

smoke_out_args=()
if [[ -n "${MERGE_DATM_SMOKE_OUT_DIR:-}" ]]; then
    smoke_out_args=(--smoke-out-dir "${MERGE_DATM_SMOKE_OUT_DIR}")
fi

cd "${MERGE_DATM_PIPELINE_SCRIPTS}"
T_SRUN_START="$(_ts)"
echo "[merge_datm_mpi] launching srun merge_datm_mpi.py (ntasks=${SLURM_NTASKS:-?}) ..."
_srun_rc=0
srun --export=ALL \
    --output="${RUN_LOG_DIR}/mpi_%j_r%t.out" \
    --error="${RUN_LOG_DIR}/mpi_%j_r%t.err" \
    "${PYTHON_BIN}" -u merge_datm_mpi.py \
        --config-input "${CONFIG_INPUT}" \
        --progress-every "${PROGRESS_EVERY}" \
        "${rebuild_flag[@]}" \
        "${smoke_out_args[@]}" \
        "${extra_args[@]}" \
    || _srun_rc=$?

T_SRUN_END="$(_ts)"
echo "[merge_datm_mpi] srun finished rc=${_srun_rc}"

python3 - <<'PY' "${T_SCRIPT_START}" "${T_AFTER_ENV}" "${T_IMPORT_START}" "${T_AFTER_IMPORT}" "${T_SRUN_START}" "${T_SRUN_END}" "${RUN_LOG_DIR}/shell_timing.json"
import json, sys
from pathlib import Path

def f(x):
    return float(x)

t0, t_env, t_imp0, t_imp1, t_sr0, t_sr1 = map(f, sys.argv[1:7])
out = Path(sys.argv[7])
report = {
    "shell_total_s": t_sr1 - t0,
    "env_setup_s": t_env - t0,
    "import_probe_s": t_imp1 - t_imp0,
    "pre_srun_s": t_sr0 - t_imp1,
    "srun_wall_s": t_sr1 - t_sr0,
    "srun_minus_import_probe_s": (t_sr1 - t_sr0) - (t_imp1 - t_imp0),
}
out.write_text(json.dumps(report, indent=2) + "\n")
print("[merge_datm_mpi] [TIMING] shell breakdown (seconds):")
for k, v in report.items():
    print(f"[merge_datm_mpi] [TIMING]   {k}: {v:.3f}")
PY

echo "[merge_datm_mpi] output files in ${FORCING_OUT_DIR}:"
ls -lh "${FORCING_OUT_DIR}"/*.nc 2>/dev/null || echo "[merge_datm_mpi] (no .nc files)"

exit "${_srun_rc}"
