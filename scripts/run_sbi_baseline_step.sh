#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
SBI_METHOD="${SBI_METHOD:-snpe}"
DENSITY_ESTIMATOR="${DENSITY_ESTIMATOR:-maf}"
DATA_DIR="${DATA_DIR:-}"
DATA_PREFIX="${DATA_PREFIX:-}"
CURR="${CURR:-0.1}"
N_TRAIN="${N_TRAIN:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
DEVICE="${DEVICE:-auto}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
EMBEDDING_HIDDEN="${EMBEDDING_HIDDEN:-256}"
HIDDEN_FEATURES="${HIDDEN_FEATURES:-128}"
NUM_TRANSFORMS="${NUM_TRANSFORMS:-5}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-4}"
STOP_AFTER_EPOCHS="${STOP_AFTER_EPOCHS:-15}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-80}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-256}"
SEED="${SEED:-0}"

PYTHON_BIN="/projects/neuro-collab/conda/neuro-collab-env/bin/python"
DLKIT="${REPO_ROOT}/vendor/dlkit"

export PYTHONPATH="${DLKIT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

mkdir -p "${RUN_OUTPUT_ROOT}"
SAVE_DIR="${RUN_OUTPUT_ROOT}/${SLURM_JOB_ID}_${RUN_ID}"

PARAMS_PATH="${PARAMS_FILE}"
if [[ "${PARAMS_PATH}" != /* ]]; then
  PARAMS_PATH="${REPO_ROOT}/${PARAMS_PATH}"
fi

cd /tmp
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_hh_sbi_baseline.py" \
  --params "${PARAMS_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --method "${SBI_METHOD}" \
  --density-estimator "${DENSITY_ESTIMATOR}" \
  ${DATA_DIR:+--data-dir "${DATA_DIR}"} \
  ${DATA_PREFIX:+--data-prefix "${DATA_PREFIX}"} \
  --curr "${CURR}" \
  ${N_TRAIN:+--n-train "${N_TRAIN}"} \
  ${EVAL_LIMIT:+--eval-limit "${EVAL_LIMIT}"} \
  --device "${DEVICE}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --embedding-hidden "${EMBEDDING_HIDDEN}" \
  --hidden-features "${HIDDEN_FEATURES}" \
  --num-transforms "${NUM_TRANSFORMS}" \
  --training-batch-size "${TRAINING_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --stop-after-epochs "${STOP_AFTER_EPOCHS}" \
  --max-num-epochs "${MAX_NUM_EPOCHS}" \
  --posterior-samples "${POSTERIOR_SAMPLES}" \
  --seed "${SEED}"
