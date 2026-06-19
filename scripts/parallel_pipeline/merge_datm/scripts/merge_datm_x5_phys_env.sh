# shellcheck shell=bash
# Example env for x5_phys merge_datm jobs (38,325 ha2x3h files).
# Source after merge_datm_paths.sh. Override any variable before sourcing.

: "${MERGE_DATM_PROJECT_ROOT:?source merge_datm_paths.sh first}"

_scratch="${SCRATCH:-${HOME}}"

export CONFIG_INPUT="${CONFIG_INPUT:-${MERGE_DATM_PROJECT_ROOT}/config/CNP_merge_datm_x5_phys_test.txt}"
export FORCING_OUT_DIR="${FORCING_OUT_DIR:-${_scratch}/landsim_inputs/striped/forcing_netcdf_datm_x5_phys_1901_2023}"
export FORCE_REBUILD="${FORCE_REBUILD:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-200}"

# DATM_ROOT and other paths live in CONFIG_INPUT — copy/edit the config for your site.
