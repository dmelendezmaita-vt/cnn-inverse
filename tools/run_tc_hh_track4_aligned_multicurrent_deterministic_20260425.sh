#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROW_ID="${ROW_ID:?ROW_ID is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
BASELINE="${BASELINE:?BASELINE is required}"
AGGREGATION="${AGGREGATION:?AGGREGATION is required}"
FEATURE_MODE="${FEATURE_MODE:?FEATURE_MODE is required}"
SVD_COMPONENTS="${SVD_COMPONENTS:-0}"
SAVE_DIR="${SAVE_DIR:?SAVE_DIR is required}"
SEED="${SEED:-20260425}"
N_TRAIN="${N_TRAIN:-4096}"
N_VALIDATE="${N_VALIDATE:-1024}"
N_TEST="${N_TEST:-1024}"
CURRENTS="${CURRENTS:-0.1,0.2,0.3,0.4,0.5}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env
export PYTHONPATH="${REPO_ROOT}/vendor/dlkit:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NC_STAGE_SPLIT_CACHE_TO_TMPDIR="${NC_STAGE_SPLIT_CACHE_TO_TMPDIR:-0}"
export NC_LOCAL_STAGE_ROOT="${NC_LOCAL_STAGE_ROOT:-${TMPDIR:-/localscratch/${SLURM_JOB_ID:-}}}"

cd "${REPO_ROOT}"

python tools/run_hh_track4_aligned_multicurrent_classical_20260425.py \
  --params "${PARAMS_FILE}" \
  --baseline "${BASELINE}" \
  --aggregation "${AGGREGATION}" \
  --feature-mode "${FEATURE_MODE}" \
  --svd-components "${SVD_COMPONENTS}" \
  --save-dir "${SAVE_DIR}" \
  --seed "${SEED}" \
  --n-train "${N_TRAIN}" \
  --n-validate "${N_VALIDATE}" \
  --n-test "${N_TEST}" \
  --currents "${CURRENTS}"
