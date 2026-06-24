#!/bin/bash
#SBATCH --job-name=merge_datm_x5_1n
# Submit from the 025deg directory:
#   cd 025deg && sbatch parallel_pipeline/merge_datm/scripts/merge_datm_x5_phys_1node.sh
#SBATCH --output=scripts/logs/merge_datm_x5_phys_sbatch_%j.out
#SBATCH --error=scripts/logs/merge_datm_x5_phys_sbatch_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=m4814    # ← change to your NERSC (or site) account

set -euo pipefail

# 1 node, 128 ranks (~299 files/rank on full x5_phys dataset).
_launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=merge_datm_paths.sh
source "${_launcher_dir}/merge_datm_paths.sh"
# shellcheck source=merge_datm_x5_phys_env.sh
source "${SCRIPT_DIR}/merge_datm_x5_phys_env.sh"
export N_NODES=1
# shellcheck source=merge_datm_common.sh
source "${SCRIPT_DIR}/merge_datm_common.sh"
