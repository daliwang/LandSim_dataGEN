#!/bin/bash
#SBATCH  --job-name=h0_forcing_build
#SBATCH  -A hpcl-cli185
#SBATCH  -p parallel
#SBATCH  -q normal
#SBATCH  -N 1
#SBATCH  -n 1
#SBATCH  -c 16
#SBATCH  --ntasks-per-node=1
#SBATCH  --mem-per-cpu=4gb
#SBATCH  -t 24:00:00
#SBATCH  --output=/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized/logs/run.h0_forcing_build.%j

# Build the 0.25 degree h0-forcing training dataset on a compute node.
#
# Submit from the login node:
#   bash scripts/submit_h0_forcing_build.sh
#
# Optional overrides (passed through by the submit wrapper):
#   BUILD_MODE=resume|full
#   VALIDATE=0|1
#   RUN_ID=custom_run_id
#   PYTHON_BIN=python
#   MODULE_INIT='module load ...'
#   CONDA_ENV=myenv

set -euo pipefail

PROJECT_ROOT="/projects/hpcl-cli185/proj-shared/wangd/AI4ELM/AI_data/LandSim_dataGEN_h0_vectorized"

export BUILD_MODE="${BUILD_MODE:-resume}"
export VALIDATE="${VALIDATE:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

mkdir -p "${PROJECT_ROOT}/logs"

echo "slurm_job_id: ${SLURM_JOB_ID:-local}"
echo "slurm_node: ${SLURMD_NODENAME:-$(hostname)}"
echo "utc_start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "build_mode: ${BUILD_MODE}"
echo "validate: ${VALIDATE}"
echo

cd "${PROJECT_ROOT}"
bash scripts/run_h0_forcing_build.sh

echo
echo "utc_end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
