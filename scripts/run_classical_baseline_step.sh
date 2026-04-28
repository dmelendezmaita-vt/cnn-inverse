#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
BASELINE_NAME="${BASELINE_NAME:?BASELINE_NAME is required}"
FEATURE_MODE="${FEATURE_MODE:-raw}"
DATA_DIR="${DATA_DIR:-}"
DATA_PREFIX="${DATA_PREFIX:-}"
CURR="${CURR:-0.1}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

DLKIT="${REPO_ROOT}/vendor/dlkit"
export PYTHONPATH="${DLKIT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"

mkdir -p "${RUN_OUTPUT_ROOT}"
SAVE_DIR="${RUN_OUTPUT_ROOT}/${SLURM_JOB_ID}_${RUN_ID}"

cd "${REPO_ROOT}"
python scripts/run_hh_classical_baseline.py \
  --params "${PARAMS_FILE}" \
  --baseline "${BASELINE_NAME}" \
  --feature-mode "${FEATURE_MODE}" \
  --save-dir "${SAVE_DIR}" \
  ${DATA_DIR:+--data-dir "${DATA_DIR}"} \
  ${DATA_PREFIX:+--data-prefix "${DATA_PREFIX}"} \
  --curr "${CURR}"
