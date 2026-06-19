#!/usr/bin/env bash
# Submit assembly-only (+ optional validation) for h0 forcing final_dataset.
#
# Run from the LOGIN node:
#   cd /projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized
#   bash scripts/submit_h0_forcing_assemble.sh
#
# Examples:
#   VALIDATE=0 bash scripts/submit_h0_forcing_assemble.sh
#   SLURM_MEM_PER_CPU=8gb SLURM_TIME=6:00:00 bash scripts/submit_h0_forcing_assemble.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/submit_h0_forcing_assemble.slurm"
LOG_DIR="${PROJECT_ROOT}/logs"

VALIDATE="${VALIDATE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_MEM_PER_CPU="${SLURM_MEM_PER_CPU:-8gb}"

mkdir -p "${LOG_DIR}"

EXPORT_VARS="ALL,VALIDATE=${VALIDATE},PYTHON_BIN=${PYTHON_BIN}"

echo "Submitting h0 forcing ASSEMBLY to SLURM"
echo "  assemble: all batches from existing artifacts"
echo "  validate: ${VALIDATE}"
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
echo "log: ${LOG_DIR}/run.h0_forcing_assemble.${JOB_ID}"
echo "${JOB_ID}" > "${LOG_DIR}/slurm.assemble.latest.jobid"
echo
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/run.h0_forcing_assemble.${JOB_ID}"
