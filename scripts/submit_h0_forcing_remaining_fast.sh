#!/usr/bin/env bash
# Submit fast h0 forcing path (preprocess + ds5-ds9 + assembly) to SLURM.
#
# Run from the LOGIN node:
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/submit_h0_forcing_remaining_fast.sh
#
# Examples:
#   VALIDATE=1 bash scripts/submit_h0_forcing_remaining_fast.sh
#   SKIP_PREPROCESS=1 bash scripts/submit_h0_forcing_remaining_fast.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/submit_h0_forcing_remaining_fast.slurm"
LOG_DIR="${PROJECT_ROOT}/logs"

VALIDATE="${VALIDATE:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_MEM_PER_CPU="${SLURM_MEM_PER_CPU:-4gb}"

mkdir -p "${LOG_DIR}"

EXPORT_VARS="ALL,VALIDATE=${VALIDATE},SKIP_PREPROCESS=${SKIP_PREPROCESS},PYTHON_BIN=${PYTHON_BIN}"

echo "Submitting h0 forcing FAST path to SLURM"
echo "  preprocess: $([[ ${SKIP_PREPROCESS} == 1 ]] && echo skip || echo yes)"
echo "  build: ds5-ds9 + assembly"
echo "  cpus: ${SLURM_CPUS}, mem/cpu: ${SLURM_MEM_PER_CPU}, time: ${SLURM_TIME}"
echo

JOB_ID="$(
  sbatch \
    --account=hpcl-cli185 \
    --qos=normal \
    --partition=parallel \
    --time="${SLURM_TIME}" \
    --cpus-per-task="${SLURM_CPUS}" \
    --mem-per-cpu="${SLURM_MEM_PER_CPU}" \
    --export="${EXPORT_VARS}" \
    "${SLURM_SCRIPT}" \
    | awk '{print $4}'
)"

echo "Submitted job ${JOB_ID}"
echo "log: ${LOG_DIR}/run.h0_forcing_fast.${JOB_ID}"
echo "${JOB_ID}" > "${LOG_DIR}/slurm.fast.latest.jobid"
echo
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/run.h0_forcing_fast.${JOB_ID}"
