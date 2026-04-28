#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch}"
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
source activate /projects/neuro-collab/conda/neuro-collab-env
export PYTHONNOUSERSITE=1
export PYTHONPATH="/projects/neuro-collab/code/dl-kit-main:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=

cd "${REPO_ROOT}"

python scripts/run_hh_track4_aligned_multicurrent_sbi_20260425.py \
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
  --device cpu
