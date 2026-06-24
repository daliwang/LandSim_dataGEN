#!/bin/bash
#SBATCH --job-name=merge_datm_x5_8n
# Submit from the 025deg directory (SLURM output paths are relative to submit cwd):
#   cd 025deg && sbatch parallel_pipeline/merge_datm/scripts/merge_datm_x5_phys_8node.sh
#SBATCH --output=scripts/logs/merge_datm_x5_phys_sbatch_%j.out
#SBATCH --error=scripts/logs/merge_datm_x5_phys_sbatch_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=1024
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=m4814    # ← change to your NERSC (or site) account

set -euo pipefail

# Best measured layout for full x5_phys (38325 ha2x3h files):
#   8 nodes × 128 ranks = 1024 MPI tasks (~37 files/rank)
#   compute_total ≈ 21.5s (job 53772332)
#
_launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=merge_datm_paths.sh
source "${_launcher_dir}/merge_datm_paths.sh"
# shellcheck source=merge_datm_x5_phys_env.sh
source "${SCRIPT_DIR}/merge_datm_x5_phys_env.sh"
export N_NODES=8
# shellcheck source=merge_datm_common.sh
source "${SCRIPT_DIR}/merge_datm_common.sh"
