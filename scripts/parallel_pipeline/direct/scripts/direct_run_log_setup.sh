# Shared SLURM log layout for direct_run*.sh
# Each job writes under:  <pipeline_scripts>/logs/job_<SLURM_JOB_ID>/
#
# Usage (after #SBATCH directives):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/direct_run_log_setup.sh"
#   direct_run_init_logs "${PIPELINE_SCRIPT_DIR}"

direct_run_init_logs() {
    local pipeline_script_dir="$1"
    if [[ -z "${pipeline_script_dir}" ]]; then
        echo "direct_run_init_logs: pipeline_script_dir required" >&2
        return 1
    fi

    export PIPELINE_SCRIPT_DIR="${pipeline_script_dir}"
    export RUN_LOG_DIR="${PIPELINE_SCRIPT_DIR}/logs/job_${SLURM_JOB_ID:?SLURM_JOB_ID not set}"
    mkdir -p "${RUN_LOG_DIR}"

    exec > >(tee -a "${RUN_LOG_DIR}/slurm_batch.out") 2> >(tee -a "${RUN_LOG_DIR}/slurm_batch.err" >&2)

    {
        echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
        echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
        echo "SLURM_NNODES=${SLURM_NNODES:-}"
        echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
        echo "SLURM_PROCID=${SLURM_PROCID:-}"
        echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
        echo "SUBMIT_HOST=$(hostname -f 2>/dev/null || hostname)"
        echo "START_TIME=$(date -Is 2>/dev/null || date)"
        echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
    } > "${RUN_LOG_DIR}/job_info.txt"

    echo "[direct_run] logs -> ${RUN_LOG_DIR}"
}
