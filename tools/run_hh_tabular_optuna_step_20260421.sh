#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
BASELINE_NAME="${BASELINE_NAME:?BASELINE_NAME is required}"
DATA_DIR="${DATA_DIR:-}"
DATA_PREFIX="${DATA_PREFIX:-}"
CURR="${CURR:-0.1}"
TRIALS="${TRIALS:-30}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

DLKIT="${REPO}/vendor/dlkit"
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${DLKIT}:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"

mkdir -p "${RUN_OUTPUT_ROOT}"
SAVE_DIR="${RUN_OUTPUT_ROOT}/${SLURM_JOB_ID}_${RUN_ID}"

cd "${REPO_ROOT}"
python tools/run_hh_tabular_optuna_search_20260421.py \
  --params "${PARAMS_FILE}" \
  --study-spec "${BASELINE_NAME}" \
  --save-dir "${SAVE_DIR}" \
  --trials "${TRIALS}" \
  ${DATA_DIR:+--data-dir "${DATA_DIR}"} \
  ${DATA_PREFIX:+--data-prefix "${DATA_PREFIX}"} \
  --curr "${CURR}"
