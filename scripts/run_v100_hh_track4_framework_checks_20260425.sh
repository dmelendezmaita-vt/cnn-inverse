#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK_ENV_PATH="${FRAMEWORK_ENV_PATH:?framework env path required}"
FRAMEWORK_KIND="${FRAMEWORK_KIND:?framework kind required}"
REPO_ROOT="${REPO_ROOT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch}"
SAVE_DIR="${SAVE_DIR:?save dir required}"

if [[ -f "${FRAMEWORK_ENV_PATH}/bin/activate" ]]; then
  source "${FRAMEWORK_ENV_PATH}/bin/activate"
else
  source activate "${FRAMEWORK_ENV_PATH}"
fi
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export NC_STAGE_SPLIT_CACHE_TO_TMPDIR="${NC_STAGE_SPLIT_CACHE_TO_TMPDIR:-1}"
export NC_LOCAL_STAGE_ROOT="${NC_LOCAL_STAGE_ROOT:-${TMPDIR:-/localscratch/${SLURM_JOB_ID:-}}}"

cd "${REPO_ROOT}"

case "${FRAMEWORK_KIND}" in
  bayesflow)
    python scripts/run_hh_track4_aligned_multicurrent_bayesflow_20260425.py \
      --params pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml \
      --save-dir "${SAVE_DIR}"
    ;;
  swyft)
    python scripts/run_hh_track4_aligned_multicurrent_swyft_20260425.py \
      --params pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml \
      --save-dir "${SAVE_DIR}"
    ;;
  *)
    echo "Unknown FRAMEWORK_KIND=${FRAMEWORK_KIND}" >&2
    exit 1
    ;;
esac
