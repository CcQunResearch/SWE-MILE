#!/usr/bin/env bash
# Resolve all user paths before starting processes or creating outputs.
SWE_MILE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SWE_MILE_ROOT}"
export PYTHONPATH="${SWE_MILE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${EXP_ROOT:?Set EXP_ROOT to your experiment directory}"
case "${EXP_ROOT}" in *\<*|*\>*) echo "Replace the EXP_ROOT placeholder with a real path" >&2; exit 2 ;; esac
export EXP_ROOT
export DATA_ROOT="${DATA_ROOT:-${EXP_ROOT}/data}"
if [[ "${SWE_MILE_REQUIRE_MODEL:-0}" == "1" ]]; then
    : "${MODEL_PATH:?Set MODEL_PATH to a local model or checkpoint directory}"
    case "${MODEL_PATH}" in *\<*|*\>*) echo "Replace the MODEL_PATH placeholder with a real path" >&2; exit 2 ;; esac
    [[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH is not a directory: ${MODEL_PATH}" >&2; exit 2; }
    export MODEL_PATH
fi

export SWE_MILE_IMAGE_MAP_FILE="${SWE_MILE_IMAGE_MAP_FILE:-${MINISANDBOX_IMAGE_MAP_FILE:-}}"
export MINISANDBOX_IMAGE_MAP_FILE="${MINISANDBOX_IMAGE_MAP_FILE:-${SWE_MILE_IMAGE_MAP_FILE}}"
