#!/bin/bash
#SBATCH  --job-name=h0_forcing_fast
#SBATCH  -A hpcl-cli185
#SBATCH  -p parallel
#SBATCH  -q normal
#SBATCH  -N 1
#SBATCH  -n 1
#SBATCH  -c 16
#SBATCH  --ntasks-per-node=1
#SBATCH  --mem-per-cpu=4gb
#SBATCH  -t 12:00:00
#SBATCH  --output=/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized/logs/run.h0_forcing_fast.%j

set -euo pipefail

PROJECT_ROOT="/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export VALIDATE="${VALIDATE:-0}"
export SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
export PREPROCESS_MODE="${PREPROCESS_MODE:-memory}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${PROJECT_ROOT}/logs"

echo "slurm_job_id: ${SLURM_JOB_ID}"
echo "slurm_node: ${SLURMD_NODENAME}"
echo "utc_start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "skip_preprocess: ${SKIP_PREPROCESS}"
echo "validate: ${VALIDATE}"
echo

cd "${PROJECT_ROOT}"
bash scripts/run_h0_forcing_remaining_fast.sh

echo
echo "utc_end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
