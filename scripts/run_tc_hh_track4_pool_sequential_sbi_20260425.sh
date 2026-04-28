#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
METHOD="${METHOD:?METHOD is required}"
DENSITY_ESTIMATOR="${DENSITY_ESTIMATOR:-maf}"
FEATURE_MODE="${FEATURE_MODE:-raw_plus_fft256_summary12}"
SAVE_DIR="${SAVE_DIR:?SAVE_DIR is required}"
DATA_DIR="${DATA_DIR:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track4_hh_full/concatenated_data}"
DATA_PREFIX="${DATA_PREFIX:-concatenated_data}"
CURR="${CURR:-0.1}"
SEED="${SEED:-20260425}"
POOL_SIZE="${POOL_SIZE:-4096}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-2048}"
N_VALIDATE="${N_VALIDATE:-1024}"
N_TEST="${N_TEST:-1024}"
ROUNDS="${ROUNDS:-3}"
INITIAL_TRAIN_SIZE="${INITIAL_TRAIN_SIZE:-}"
ROUND_TRAIN_SIZES="${ROUND_TRAIN_SIZES:-}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-1024}"
CANDIDATE_POSTERIOR_SAMPLES="${CANDIDATE_POSTERIOR_SAMPLES:-4}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-8}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-4}"
STOP_AFTER_EPOCHS="${STOP_AFTER_EPOCHS:-12}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-60}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
EMBEDDING_HIDDEN="${EMBEDDING_HIDDEN:-256}"
HIDDEN_FEATURES="${HIDDEN_FEATURES:-128}"
NUM_TRANSFORMS="${NUM_TRANSFORMS:-5}"
DECISION_RULE="${DECISION_RULE:-mean}"
SAMPLE_WITH="${SAMPLE_WITH:-}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env
export PYTHONPATH="/projects/neuro-collab/code/dl-kit-main:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=
export NC_STAGE_SPLIT_CACHE_TO_TMPDIR="${NC_STAGE_SPLIT_CACHE_TO_TMPDIR:-1}"
export NC_LOCAL_STAGE_ROOT="${NC_LOCAL_STAGE_ROOT:-${TMPDIR:-/localscratch/${SLURM_JOB_ID:-}}}"

cd "${REPO_ROOT}"

cmd=(
  python
  scripts/run_hh_track4_pool_sequential_sbi_20260425.py
  --params "${PARAMS_FILE}"
  --method "${METHOD}"
  --density-estimator "${DENSITY_ESTIMATOR}"
  --feature-mode "${FEATURE_MODE}"
  --save-dir "${SAVE_DIR}"
  --data-dir "${DATA_DIR}"
  --data-prefix "${DATA_PREFIX}"
  --curr "${CURR}"
  --seed "${SEED}"
  --pool-size "${POOL_SIZE}"
  --final-train-size "${FINAL_TRAIN_SIZE}"
  --n-validate "${N_VALIDATE}"
  --n-test "${N_TEST}"
  --rounds "${ROUNDS}"
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
  --candidate-posterior-samples "${CANDIDATE_POSTERIOR_SAMPLES}"
  --posterior-samples "${POSTERIOR_SAMPLES}"
  --training-batch-size "${TRAINING_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --stop-after-epochs "${STOP_AFTER_EPOCHS}"
  --max-num-epochs "${MAX_NUM_EPOCHS}"
  --embedding-dim "${EMBEDDING_DIM}"
  --embedding-hidden "${EMBEDDING_HIDDEN}"
  --hidden-features "${HIDDEN_FEATURES}"
  --num-transforms "${NUM_TRANSFORMS}"
  --decision-rule "${DECISION_RULE}"
  --device cpu
)

if [[ -n "${INITIAL_TRAIN_SIZE}" ]]; then
  cmd+=(--initial-train-size "${INITIAL_TRAIN_SIZE}")
fi
if [[ -n "${ROUND_TRAIN_SIZES}" ]]; then
  cmd+=(--round-train-sizes "${ROUND_TRAIN_SIZES}")
fi
if [[ -n "${SAMPLE_WITH}" ]]; then
  cmd+=(--sample-with "${SAMPLE_WITH}")
fi

"${cmd[@]}"
