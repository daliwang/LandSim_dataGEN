#!/bin/bash
#SBATCH --job-name=LandSim_direct_16n
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:15:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=m4814    # ← change to your NERSC (or site) account

# Recommended multi-node layout (best measured wall time ~29s on Perlmutter):
#   - 1 SLURM task per node (SLURM_NTASKS = node count)
#   - Each task fork()s 64 worker processes (--direct-workers 64)
#   - batch_partition=restart_io + automatic PFT-stripe DP load balance
#
# Submit from anywhere:
#   sbatch 025deg/parallel_pipeline/direct/scripts/direct_run_16node.sh
#
# First run after switching BATCH_PARTITION:
#   REBUILD_INDEX=1 sbatch ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../../../scripts" && pwd)"
# shellcheck source=direct_run_log_setup.sh
source "${SCRIPT_DIR}/direct_run_log_setup.sh"
direct_run_init_logs "${PIPELINE_SCRIPT_DIR}"

module load conda/Miniforge3-25.11.0-1
conda activate landsim

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CONFIG_INPUT="${CONFIG_INPUT:-${PIPELINE_SCRIPT_DIR}/../config/CNP_dataInput_from_LandSim_dataGEN.txt}"
BATCH_PARTITION="${BATCH_PARTITION:-restart_io}"
REBUILD_INDEX_ARGS=()
[[ "${REBUILD_INDEX:-0}" == "1" ]] && REBUILD_INDEX_ARGS=(--rebuild-index)

cd "${PIPELINE_SCRIPT_DIR}"

srun --output="${RUN_LOG_DIR}/legacy_%j_n%t.out" --error="${RUN_LOG_DIR}/legacy_%j_n%t.err" \
    python -u run_pipeline.py \
        --config-input "${CONFIG_INPUT}" \
        --direct \
        --batch-partition "${BATCH_PARTITION}" \
        "${REBUILD_INDEX_ARGS[@]}" \
        --direct-workers "${DIRECT_WORKERS:-64}" \
        --preload-workers "${PRELOAD_WORKERS:-16}"
