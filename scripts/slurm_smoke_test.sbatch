#!/usr/bin/env bash
#SBATCH --job-name=tst_l40s_1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=runs/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARAMS_FILE="${PARAMS_FILE:-src/pytorch/configs/params_dnn_tar_smoke.yaml}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
DATA_DIR="${DATA_DIR:-}"
DATA_PREFIX="${DATA_PREFIX:-}"
CURR="${CURR:-0.1}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="${REPO_ROOT}/vendor/dlkit:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
"${PYTHON_BIN}" src/pytorch/run_dnn.py \
  --params "${PARAMS_FILE}" \
  --mode train \
  ${DATA_DIR:+--data_dir "${DATA_DIR}"} \
  ${DATA_PREFIX:+--data_prefix "${DATA_PREFIX}"} \
  --curr "${CURR}"

"${PYTHON_BIN}" src/pytorch/run_dnn.py \
  --params "${PARAMS_FILE}" \
  --mode eval \
  ${DATA_DIR:+--data_dir "${DATA_DIR}"} \
  ${DATA_PREFIX:+--data_prefix "${DATA_PREFIX}"} \
  --curr "${CURR}"
