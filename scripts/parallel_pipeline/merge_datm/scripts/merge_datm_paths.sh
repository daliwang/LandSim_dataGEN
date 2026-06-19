#!/bin/bash
# Resolve paths for parallel_pipeline merge_datm SLURM scripts.
# Works when sbatch copies the batch script to spool (uses BASH_SOURCE, not $0).

_merge_datm_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MERGE_DATM_PROJECT_ROOT="$(cd "${_merge_datm_script_dir}/../.." && pwd)"
readonly MERGE_DATM_SCRIPT_DIR="${_merge_datm_script_dir}"
readonly MERGE_DATM_PIPELINE_SCRIPTS="${MERGE_DATM_PROJECT_ROOT}/scripts"
readonly MERGE_DATM_LEGACY_SCRIPT_DIR="${MERGE_DATM_PROJECT_ROOT}/scripts/merge_datm"

if [[ ! -f "${MERGE_DATM_LEGACY_SCRIPT_DIR}/env_cray_mpi.sh" ]]; then
    echo "[merge_datm] FATAL: missing ${MERGE_DATM_LEGACY_SCRIPT_DIR}/env_cray_mpi.sh" >&2
    exit 1
fi

SCRIPT_DIR="${MERGE_DATM_SCRIPT_DIR}"
export SCRIPT_DIR MERGE_DATM_PROJECT_ROOT MERGE_DATM_PIPELINE_SCRIPTS MERGE_DATM_LEGACY_SCRIPT_DIR
