#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
SBI_METHOD="${SBI_METHOD:-snpe}"
MODEL_NAME="${MODEL_NAME:-maf}"
SAMPLE_WITH="${SAMPLE_WITH:-}"
DATA_DIR="${DATA_DIR:-}"
DATA_PREFIX="${DATA_PREFIX:-}"
CURR="${CURR:-0.1}"
N_TRAIN="${N_TRAIN:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
DEVICE="${DEVICE:-cuda}"
FEATURE_MODE="${FEATURE_MODE:-raw}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
EMBEDDING_HIDDEN="${EMBEDDING_HIDDEN:-256}"
HIDDEN_FEATURES="${HIDDEN_FEATURES:-128}"
NUM_TRANSFORMS="${NUM_TRANSFORMS:-5}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-4}"
STOP_AFTER_EPOCHS="${STOP_AFTER_EPOCHS:-15}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-80}"
TRAIN_ADDITIVE_NOISE_STD="${TRAIN_ADDITIVE_NOISE_STD:-0.0}"
TRAIN_MULTIPLICATIVE_NOISE_STD="${TRAIN_MULTIPLICATIVE_NOISE_STD:-0.0}"
TRAIN_BASELINE_DRIFT_STD="${TRAIN_BASELINE_DRIFT_STD:-0.0}"
TRAIN_MASK_FRACTION="${TRAIN_MASK_FRACTION:-0.0}"
EVAL_ADDITIVE_NOISE_STD="${EVAL_ADDITIVE_NOISE_STD:-0.0}"
EVAL_MULTIPLICATIVE_NOISE_STD="${EVAL_MULTIPLICATIVE_NOISE_STD:-0.0}"
EVAL_BASELINE_DRIFT_STD="${EVAL_BASELINE_DRIFT_STD:-0.0}"
EVAL_MASK_FRACTION="${EVAL_MASK_FRACTION:-0.0}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-16}"
SEED="${SEED:-0}"
COMPUTE_MAP="${COMPUTE_MAP:-0}"
MAP_NUM_ITER="${MAP_NUM_ITER:-}"
MAP_NUM_TO_OPTIMIZE="${MAP_NUM_TO_OPTIMIZE:-}"
MAP_LEARNING_RATE="${MAP_LEARNING_RATE:-}"
MAP_NUM_INIT_SAMPLES="${MAP_NUM_INIT_SAMPLES:-}"
DECISION_RULE="${DECISION_RULE:-mean}"
SELECTION_SPLIT="${SELECTION_SPLIT:-validate}"
SELECTION_COV_LAMBDA="${SELECTION_COV_LAMBDA:-1.0}"
SELECTION_CANDIDATE_RULES="${SELECTION_CANDIDATE_RULES:-mean,median}"

PYTHON_BIN="/projects/neuro-collab/conda/neuro-collab-env/bin/python"
DLKIT="${REPO}/vendor/dlkit"

export PYTHONPATH="${DLKIT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

mkdir -p "${RUN_OUTPUT_ROOT}"
SAVE_DIR="${RUN_OUTPUT_ROOT}/${SLURM_JOB_ID}_${RUN_ID}"

PARAMS_PATH="${PARAMS_FILE}"
if [[ "${PARAMS_PATH}" != /* ]]; then
  PARAMS_PATH="${REPO_ROOT}/${PARAMS_PATH}"
fi

cd /tmp
CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_hh_sbi_baseline_20260422.py"
  --params "${PARAMS_PATH}"
  --save-dir "${SAVE_DIR}"
  --method "${SBI_METHOD}"
  --density-estimator "${MODEL_NAME}"
  --curr "${CURR}"
  --device "${DEVICE}"
  --feature-mode "${FEATURE_MODE}"
  --embedding-dim "${EMBEDDING_DIM}"
  --embedding-hidden "${EMBEDDING_HIDDEN}"
  --hidden-features "${HIDDEN_FEATURES}"
  --num-transforms "${NUM_TRANSFORMS}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --stop-after-epochs "${STOP_AFTER_EPOCHS}"
  --max-num-epochs "${MAX_NUM_EPOCHS}"
  --train-additive-noise-std "${TRAIN_ADDITIVE_NOISE_STD}"
  --train-multiplicative-noise-std "${TRAIN_MULTIPLICATIVE_NOISE_STD}"
  --train-baseline-drift-std "${TRAIN_BASELINE_DRIFT_STD}"
  --train-mask-fraction "${TRAIN_MASK_FRACTION}"
  --eval-additive-noise-std "${EVAL_ADDITIVE_NOISE_STD}"
  --eval-multiplicative-noise-std "${EVAL_MULTIPLICATIVE_NOISE_STD}"
  --eval-baseline-drift-std "${EVAL_BASELINE_DRIFT_STD}"
  --eval-mask-fraction "${EVAL_MASK_FRACTION}"
  --posterior-samples "${POSTERIOR_SAMPLES}"
  --seed "${SEED}"
  --decision-rule "${DECISION_RULE}"
  --selection-split "${SELECTION_SPLIT}"
  --selection-cov-lambda "${SELECTION_COV_LAMBDA}"
  --selection-candidate-rules "${SELECTION_CANDIDATE_RULES}"
)

if [[ -n "${DATA_DIR}" ]]; then
  CMD+=(--data-dir "${DATA_DIR}")
fi
if [[ -n "${DATA_PREFIX}" ]]; then
  CMD+=(--data-prefix "${DATA_PREFIX}")
fi
if [[ -n "${N_TRAIN}" ]]; then
  CMD+=(--n-train "${N_TRAIN}")
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
  CMD+=(--eval-limit "${EVAL_LIMIT}")
fi
if [[ -n "${SAMPLE_WITH}" ]]; then
  CMD+=(--sample-with "${SAMPLE_WITH}")
fi
if [[ "${COMPUTE_MAP}" == "1" ]]; then
  CMD+=(--compute-map)
fi
if [[ -n "${MAP_NUM_ITER}" ]]; then
  CMD+=(--map-num-iter "${MAP_NUM_ITER}")
fi
if [[ -n "${MAP_NUM_TO_OPTIMIZE}" ]]; then
  CMD+=(--map-num-to-optimize "${MAP_NUM_TO_OPTIMIZE}")
fi
if [[ -n "${MAP_LEARNING_RATE}" ]]; then
  CMD+=(--map-learning-rate "${MAP_LEARNING_RATE}")
fi
if [[ -n "${MAP_NUM_INIT_SAMPLES}" ]]; then
  CMD+=(--map-num-init-samples "${MAP_NUM_INIT_SAMPLES}")
fi

"${CMD[@]}"
