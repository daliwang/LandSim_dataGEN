#!/usr/bin/env bash
# Submit the h0-forcing dataset build to SLURM (hpcl-cli185 / parallel / normal).
#
# Usage:
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/submit_h0_forcing_build.sh
#
# Examples:
#   BUILD_MODE=resume bash scripts/submit_h0_forcing_build.sh
#   BUILD_MODE=full VALIDATE=1 bash scripts/submit_h0_forcing_build.sh
#   SLURM_CPUS=32 SLURM_MEM_PER_CPU=4gb bash scripts/submit_h0_forcing_build.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/submit_h0_forcing_build.slurm"
LOG_DIR="${PROJECT_ROOT}/logs"

BUILD_MODE="${BUILD_MODE:-resume}"
VALIDATE="${VALIDATE:-0}"
RUN_ID="${RUN_ID:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-h0_forcing_build}"
SLURM_PARTITION="${SLURM_PARTITION:-parallel}"
SLURM_QOS="${SLURM_QOS:-normal}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpcl-cli185}"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_NTASKS="${SLURM_NTASKS:-1}"
SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_MEM_PER_CPU="${SLURM_MEM_PER_CPU:-4gb}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Missing SLURM script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

EXPORT_VARS="ALL,BUILD_MODE=${BUILD_MODE},VALIDATE=${VALIDATE},PYTHON_BIN=${PYTHON_BIN}"
if [[ -n "${RUN_ID}" ]]; then
  EXPORT_VARS="${EXPORT_VARS},RUN_ID=${RUN_ID}"
fi
if [[ -n "${MODULE_INIT:-}" ]]; then
  EXPORT_VARS="${EXPORT_VARS},MODULE_INIT=${MODULE_INIT}"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
  EXPORT_VARS="${EXPORT_VARS},CONDA_ENV=${CONDA_ENV}"
fi

echo "Submitting h0-forcing build to SLURM"
echo "  project_root:       ${PROJECT_ROOT}"
echo "  build_mode:         ${BUILD_MODE}"
echo "  validate:           ${VALIDATE}"
echo "  account:            ${SLURM_ACCOUNT}"
echo "  partition:          ${SLURM_PARTITION}"
echo "  qos:                ${SLURM_QOS}"
echo "  nodes:              ${SLURM_NODES}"
echo "  ntasks:             ${SLURM_NTASKS}"
echo "  ntasks_per_node:    ${SLURM_NTASKS_PER_NODE}"
echo "  cpus_per_task:      ${SLURM_CPUS}"
echo "  mem_per_cpu:        ${SLURM_MEM_PER_CPU}"
echo "  time_limit:         ${SLURM_TIME}"
echo

JOB_ID="$(
  sbatch \
    --job-name="${SLURM_JOB_NAME}" \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --qos="${SLURM_QOS}" \
    --nodes="${SLURM_NODES}" \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node="${SLURM_NTASKS_PER_NODE}" \
    --cpus-per-task="${SLURM_CPUS}" \
    --mem-per-cpu="${SLURM_MEM_PER_CPU}" \
    --time="${SLURM_TIME}" \
    --output="${LOG_DIR}/run.${SLURM_JOB_NAME}.%j" \
    --export="${EXPORT_VARS}" \
    "${SLURM_SCRIPT}" \
    | awk '{print $4}'
)"

echo "Submitted job ${JOB_ID}"
echo "log: ${LOG_DIR}/run.${SLURM_JOB_NAME}.${JOB_ID}"
echo "${JOB_ID}" > "${LOG_DIR}/slurm.latest.jobid"
echo
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/run.${SLURM_JOB_NAME}.${JOB_ID}"
