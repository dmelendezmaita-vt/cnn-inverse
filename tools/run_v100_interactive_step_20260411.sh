#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ALLOC_JOB_ID="${ALLOC_JOB_ID:?ALLOC_JOB_ID is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
TAR_PATH="${TAR_PATH:?TAR_PATH is required}"
DATA_PREFIX="${DATA_PREFIX:?DATA_PREFIX is required}"
CURR="${CURR:-0.1}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:?MASTER_PORT is required}"
STEP_NNODES="${STEP_NNODES:?STEP_NNODES is required}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DATA_ACCESS_MODE="${DATA_ACCESS_MODE:-copy_to_node}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-test}"
SPLIT_EVAL_AFTER_TRAIN="${SPLIT_EVAL_AFTER_TRAIN:-0}"
EVAL_ONLY_CHECKPOINT="${EVAL_ONLY_CHECKPOINT:-}"
SKIP_INLINE_SPLIT_EVAL="${SKIP_INLINE_SPLIT_EVAL:-0}"
LAUNCH_BACKEND="${LAUNCH_BACKEND:-torchrun}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

DLKIT="${REPO}/vendor/dlkit"
export PYTHONPATH="${DLKIT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_DIST_TIMEOUT_SECONDS="${TORCH_DIST_TIMEOUT_SECONDS:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,veth}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( (${SLURM_CPUS_PER_TASK:-24}) / 2 ))}"
if (( OMP_NUM_THREADS < 1 )); then
  export OMP_NUM_THREADS=1
fi

TMP_BASE="${FOLLOWUP_TMP_BASE:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
NODE_TAG="${SLURMD_NODENAME:-${HOSTNAME:-node}}"
WORK_ROOT="${TMP_BASE}/${ALLOC_JOB_ID}_${RUN_ID}"
WORK="${WORK_ROOT}/${NODE_TAG}_proc${SLURM_PROCID:-0}"
TMPDIR="${WORK}/tmp"
export TMPDIR
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${WORK_ROOT}" "${WORK}" "${TMPDIR}"

if [[ -n "${SHARED_DATA_DIR}" ]]; then
  if [[ ! -d "${SHARED_DATA_DIR}" ]] || ! find "${SHARED_DATA_DIR}" -mindepth 1 -maxdepth 1 | grep -q .; then
    python "${REPO_ROOT}/tools/shared_data_utils.py" \
      --tar-path "${TAR_PATH}" \
      --shared-dir "${SHARED_DATA_DIR}"
  fi
fi

if [[ -n "${SHARED_DATA_DIR}" && -d "${SHARED_DATA_DIR}" ]]; then
  DATA_DIR="${SHARED_DATA_DIR}"
elif [[ "${DATA_ACCESS_MODE}" == "copy_to_node" ]]; then
  mkdir -p "${WORK}/ds"
  cp -f "${TAR_PATH}" "${WORK}/dataset.tar"
  tar -xf "${WORK}/dataset.tar" -C "${WORK}/ds"
  TOPDIR="$(find "${WORK}/ds" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  if [[ -n "${TOPDIR}" ]]; then
    DATA_DIR="${TOPDIR}"
  else
    DATA_DIR="${WORK}/ds"
  fi
else
  DATA_DIR="${TAR_PATH}"
fi

echo "$(date -Is) [interactive-step] run_id=${RUN_ID} alloc_job_id=${ALLOC_JOB_ID} nodeid=${SLURM_NODEID:-NA} procid=${SLURM_PROCID:-NA}"
echo "$(date -Is) [interactive-step] data_access_mode=${DATA_ACCESS_MODE} data_dir=${DATA_DIR}"
echo "$(date -Is) [interactive-step] master=${MASTER_ADDR}:${MASTER_PORT} step_nnodes=${STEP_NNODES} nproc_per_node=${NPROC_PER_NODE}"
echo "$(date -Is) [interactive-step] split_eval_after_train=${SPLIT_EVAL_AFTER_TRAIN}"
echo "$(date -Is) [interactive-step] eval_only_checkpoint=${EVAL_ONLY_CHECKPOINT:-none}"
echo "$(date -Is) [interactive-step] skip_inline_split_eval=${SKIP_INLINE_SPLIT_EVAL}"
echo "$(date -Is) [interactive-step] launch_backend=${LAUNCH_BACKEND}"

cd "${REPO_ROOT}"

# Ensure a unique output directory below RUN_OUTPUT_ROOT for each logical task execution.
export SLURM_JOB_ID="${ALLOC_JOB_ID}_${RUN_ID}"

SAVE_PREDICTIONS_ARGS=()
if [[ -n "${SAVE_PREDICTIONS}" && "${SAVE_PREDICTIONS}" != "inherit_from_config" ]]; then
  SAVE_PREDICTIONS_ARGS=(--save_predictions "${SAVE_PREDICTIONS}")
fi

if [[ -n "${EVAL_ONLY_CHECKPOINT}" ]]; then
  if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
    exit 0
  fi
  LOAD_PATH="${EVAL_ONLY_CHECKPOINT}"
  if [[ -d "${LOAD_PATH}" ]]; then
    CANDIDATE="$(find "${LOAD_PATH}" -type f -name 'net_e*.pt' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${CANDIDATE}" ]]; then
      echo "$(date -Is) [interactive-step] eval-only salvage failed: no checkpoint found under ${LOAD_PATH}" >&2
      exit 1
    fi
    LOAD_PATH="${CANDIDATE}"
  elif [[ ! -f "${LOAD_PATH}" ]]; then
    echo "$(date -Is) [interactive-step] eval-only salvage failed: checkpoint path does not exist: ${LOAD_PATH}" >&2
    exit 1
  fi

  echo "$(date -Is) [interactive-step] eval-only salvage using checkpoint ${LOAD_PATH}"
  unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
  unset SLURM_NTASKS SLURM_LOCALID
  export CUDA_VISIBLE_DEVICES=
  python src/pytorch/run_dnn.py \
    --params "${PARAMS_FILE}" \
    --mode eval \
    --load_dir "${LOAD_PATH}" \
    --save_dir_base "${RUN_OUTPUT_ROOT}" \
    --data_dir "${DATA_DIR}" \
    --data_prefix "${DATA_PREFIX}" \
    --curr "${CURR}" \
    "${SAVE_PREDICTIONS_ARGS[@]}"
  exit 0
fi

TRAIN_MODE="train_eval"
if [[ "${SPLIT_EVAL_AFTER_TRAIN}" == "1" ]]; then
  TRAIN_MODE="train"
fi

if [[ "${LAUNCH_BACKEND}" == "slurm_direct" ]]; then
  python src/pytorch/run_dnn.py \
    --params "${PARAMS_FILE}" \
    --mode "${TRAIN_MODE}" \
    --save_dir_base "${RUN_OUTPUT_ROOT}" \
    --data_dir "${DATA_DIR}" \
    --data_prefix "${DATA_PREFIX}" \
    --curr "${CURR}" \
    "${SAVE_PREDICTIONS_ARGS[@]}"
else
  torchrun \
    --nnodes="${STEP_NNODES}" \
    --node_rank="${SLURM_PROCID}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${ALLOC_JOB_ID}_${RUN_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    src/pytorch/run_dnn.py \
      --params "${PARAMS_FILE}" \
      --mode "${TRAIN_MODE}" \
      --save_dir_base "${RUN_OUTPUT_ROOT}" \
      --data_dir "${DATA_DIR}" \
      --data_prefix "${DATA_PREFIX}" \
      --curr "${CURR}" \
      "${SAVE_PREDICTIONS_ARGS[@]}"
fi

if [[ "${SPLIT_EVAL_AFTER_TRAIN}" == "1" && "${SKIP_INLINE_SPLIT_EVAL}" != "1" && "${SLURM_PROCID:-0}" == "0" ]]; then
  SAVE_DIR="${RUN_OUTPUT_ROOT}/${ALLOC_JOB_ID}_${RUN_ID}"
  CKPT_PATH="$(find "${SAVE_DIR}/checkpoints" -type f -name 'net_e*.pt' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "$(date -Is) [interactive-step] split-eval failed: no checkpoint found under ${SAVE_DIR}/checkpoints" >&2
    exit 1
  fi

  echo "$(date -Is) [interactive-step] split-eval using checkpoint ${CKPT_PATH}"

  unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
  unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID
  # CPU-only eval is slower but much more reliable than GPU eval after the
  # distributed training step for these post-local SGD runs.
  export CUDA_VISIBLE_DEVICES=

  python src/pytorch/run_dnn.py \
    --params "${PARAMS_FILE}" \
    --mode eval \
    --load_dir "${CKPT_PATH}" \
    --save_dir_base "${RUN_OUTPUT_ROOT}" \
    --data_dir "${DATA_DIR}" \
    --data_prefix "${DATA_PREFIX}" \
    --curr "${CURR}" \
    "${SAVE_PREDICTIONS_ARGS[@]}"
fi
