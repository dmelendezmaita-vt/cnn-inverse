#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
METHOD="${METHOD:?METHOD is required}"
AGGREGATION="${AGGREGATION:?AGGREGATION is required}"
FEATURE_MODE="${FEATURE_MODE:?FEATURE_MODE is required}"
SAVE_DIR="${SAVE_DIR:?SAVE_DIR is required}"
SEED="${SEED:-20260425}"
N_TRAIN="${N_TRAIN:-4096}"
N_VALIDATE="${N_VALIDATE:-1024}"
N_TEST="${N_TEST:-128}"
CURRENTS="${CURRENTS:-0.1,0.2,0.3,0.4,0.5}"

module load Miniforge3
SBI_ENV_PATH="${SBI_ENV_PATH:-/projects/neuro-collab/conda/neuro-collab-env}"
if [[ -f "${SBI_ENV_PATH}/bin/activate" ]]; then
  source "${SBI_ENV_PATH}/bin/activate"
else
  source activate "${SBI_ENV_PATH}"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/vendor/dlkit:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NC_STAGE_SPLIT_CACHE_TO_TMPDIR="${NC_STAGE_SPLIT_CACHE_TO_TMPDIR:-0}"
export NC_LOCAL_STAGE_ROOT="${NC_LOCAL_STAGE_ROOT:-${TMPDIR:-/localscratch/${SLURM_JOB_ID:-}}}"

cd "${REPO_ROOT}"

python tools/run_hh_track4_aligned_multicurrent_sbi_20260425.py \
  --params "${PARAMS_FILE}" \
  --method "${METHOD}" \
  --aggregation "${AGGREGATION}" \
  --feature-mode "${FEATURE_MODE}" \
  --save-dir "${SAVE_DIR}" \
  --seed "${SEED}" \
  --n-train "${N_TRAIN}" \
  --n-validate "${N_VALIDATE}" \
  --n-test "${N_TEST}" \
  --currents "${CURRENTS}" \
  --device cuda
