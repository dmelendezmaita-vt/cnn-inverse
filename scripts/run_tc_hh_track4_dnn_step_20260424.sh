#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
DATA_PREFIX="${DATA_PREFIX:?DATA_PREFIX is required}"
CURR="${CURR:-0.1}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-}"
TAR_PATH="${TAR_PATH:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-test}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

DLKIT="/projects/neuro-collab/code/dl-kit-main"
export PYTHONPATH="${DLKIT}:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-6}}"
export NC_STAGE_SPLIT_CACHE_TO_TMPDIR="${NC_STAGE_SPLIT_CACHE_TO_TMPDIR:-1}"
export NC_LOCAL_STAGE_ROOT="${NC_LOCAL_STAGE_ROOT:-${TMPDIR:-/localscratch/${SLURM_JOB_ID:-}}}"
if (( OMP_NUM_THREADS < 1 )); then
  export OMP_NUM_THREADS=1
fi

if [[ -n "${SHARED_DATA_DIR}" && -d "${SHARED_DATA_DIR}" ]]; then
  DATA_DIR="${SHARED_DATA_DIR}"
elif [[ -n "${TAR_PATH}" ]]; then
  DATA_DIR="${TAR_PATH}"
else
  echo "No usable data source provided" >&2
  exit 1
fi

cd "${REPO_ROOT}"

python pytorch/run_dnn.py \
  --params "${PARAMS_FILE}" \
  --mode train_eval \
  --save_dir_base "${RUN_OUTPUT_ROOT}" \
  --data_dir "${DATA_DIR}" \
  --data_prefix "${DATA_PREFIX}" \
  --curr "${CURR}" \
  --save_predictions "${SAVE_PREDICTIONS}"
