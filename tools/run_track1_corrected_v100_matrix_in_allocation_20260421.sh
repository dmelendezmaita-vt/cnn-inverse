#!/usr/bin/env bash
set -euo pipefail

ALLOCATION_JOB_ID="${ALLOCATION_JOB_ID:-}"
if [[ -z "${ALLOCATION_JOB_ID}" ]]; then
  echo "ERROR: set ALLOCATION_JOB_ID to a running Falcon allocation job id" >&2
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLKIT="${REPO}/vendor/dlkit"
DATA_DIR="${REPO}/data/2020-12-09"
CONDA_ENV="/projects/neuro-collab/conda/neuro-collab-env"

SEEDS_STR="${SEEDS:-301 302 303 304 305}"
NODE_COUNTS_STR="${NODE_COUNTS:-1 2 4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NPROC="${NPROC:-2}"

SUITE="track1_corrected_v100"
CONFIG_DIR="${REPO}/src/pytorch/configs/track1_corrected_v100"
CSV="${REPO}/data/important_notes/first_track_paper_parity/v100_track1_corrected_training_jobs_rerun.csv"
RUN_ROOT="/projects/neuro-collab/data/runs/${SUITE}"

format_hms() {
  local total="${1:-0}"
  local hh mm ss
  hh=$(( total / 3600 ))
  mm=$(( (total % 3600) / 60 ))
  ss=$(( total % 60 ))
  printf '%02d:%02d:%02d' "${hh}" "${mm}" "${ss}"
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

NODELIST="$(scontrol show job "${ALLOCATION_JOB_ID}" | tr ' ' '\n' | awk -F= '/^NodeList=/{print $2; exit}')"
if [[ -z "${NODELIST}" ]]; then
  echo "ERROR: could not resolve NodeList for allocation ${ALLOCATION_JOB_ID}" >&2
  exit 1
fi
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
if [[ "${#HOSTS[@]}" -lt 1 ]]; then
  echo "ERROR: no hosts found for allocation ${ALLOCATION_JOB_ID}" >&2
  exit 1
fi

PARTITION="$(scontrol show job "${ALLOCATION_JOB_ID}" | tr ' ' '\n' | awk -F= '/^Partition=/{print $2; exit}')"
QOS="$(scontrol show job "${ALLOCATION_JOB_ID}" | tr ' ' '\n' | awk -F= '/^QOS=/{print $2; exit}')"
REQUESTED_TIME_LIMIT="$(scontrol show job "${ALLOCATION_JOB_ID}" | tr ' ' '\n' | awk -F= '/^TimeLimit=/{print $2; exit}')"

mkdir -p "${RUN_ROOT}"
mkdir -p "$(dirname "${CSV}")"

printf '%s\n' \
  'Suite,Cluster,GPU,Job name,Seed,Nodes,GPUs/node,CPUs/node,RAM/node (GiB),Requested time limit,QoS,Partition,Params file,Data access mode,Job ID,Status,Elapsed job time,Submitted at (local),Run dir,Train last,Val last,Train best,Val best,Notes' \
  > "${CSV}"

for seed in ${SEEDS_STR}; do
  PARAMS_FILE="${CONFIG_DIR}/params_dnn_tar_seed${seed}.yaml"
  if [[ ! -f "${PARAMS_FILE}" ]]; then
    echo "ERROR: missing params file ${PARAMS_FILE}" >&2
    exit 1
  fi

  for nodes in ${NODE_COUNTS_STR}; do
    if (( nodes > ${#HOSTS[@]} )); then
      echo "ERROR: requested ${nodes} nodes but allocation only has ${#HOSTS[@]}" >&2
      exit 1
    fi

    STEP_HOSTS=( "${HOSTS[@]:0:${nodes}}" )
    STEP_NODELIST="$(join_by_comma "${STEP_HOSTS[@]}")"
    MASTER_ADDR="${STEP_HOSTS[0]}"
    MASTER_PORT="$(( 29500 + (seed - 300) * 10 + nodes ))"

    JOB_NAME="trn_V100_${nodes}n_t1c${seed}"
    SAVE_DIR_BASE="${RUN_ROOT}/${JOB_NAME}"
    RUN_DIR="${SAVE_DIR_BASE}/${ALLOCATION_JOB_ID}"
    START_LOCAL="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    TMP_LOG="$(mktemp)"

    start_epoch="$(date +%s)"
    set +e
    srun \
      --jobid="${ALLOCATION_JOB_ID}" \
      --nodes="${nodes}" \
      --ntasks="${nodes}" \
      --ntasks-per-node=1 \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --gres="gpu:v100:${GPUS_PER_NODE}" \
      --nodelist="${STEP_NODELIST}" \
      --kill-on-bad-exit=1 \
      bash -lc "
        set -euo pipefail
        module load Miniforge3
        source activate '${CONDA_ENV}'
        export PYTHONPATH='${DLKIT}:${REPO}:\${PYTHONPATH:-}'
        export OMP_NUM_THREADS=\$(( \${SLURM_CPUS_PER_TASK:-1} / ${NPROC} ))
        export OMP_NUM_THREADS=\$(( OMP_NUM_THREADS<1 ? 1 : OMP_NUM_THREADS ))
        cd '${REPO}'
        torchrun \
          --nnodes=\"\${SLURM_NNODES}\" \
          --node_rank=\"\${SLURM_NODEID}\" \
          --nproc_per_node='${NPROC}' \
          --rdzv_backend=c10d \
          --rdzv_id='${ALLOCATION_JOB_ID}_${JOB_NAME}' \
          --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
          src/pytorch/run_dnn.py \
            --params '${PARAMS_FILE}' \
            --mode train_eval \
            --save_dir_base '${SAVE_DIR_BASE}' \
            --data_dir '${DATA_DIR}' \
            --save_predictions test
      " 2>&1 | tee "${TMP_LOG}"
    rc="${PIPESTATUS[0]}"
    set -e
    elapsed_hms="$(format_hms "$(( $(date +%s) - start_epoch ))")"

    STATUS="COMPLETED"
    if (( rc != 0 )); then
      STATUS="FAILED"
    fi

    TRAIN_LAST=""
    VAL_LAST=""
    TRAIN_BEST=""
    VAL_BEST=""
    if [[ -s "${RUN_DIR}/loss.txt" ]]; then
      TRAIN_LAST="$(awk 'END{print $1}' "${RUN_DIR}/loss.txt")"
      VAL_LAST="$(awk 'END{print $2}' "${RUN_DIR}/loss.txt")"
      TRAIN_BEST="$(awk 'NR==1{m=$1} $1<m{m=$1} END{print m}' "${RUN_DIR}/loss.txt")"
      VAL_BEST="$(awk 'NR==1{m=$2} $2<m{m=$2} END{print m}' "${RUN_DIR}/loss.txt")"
    fi

    NOTE="allocation_job_id=${ALLOCATION_JOB_ID}; lane=corrected; step_hosts=${STEP_NODELIST}"
    printf '%s\n' \
      "${SUITE},Falcon,V100,${JOB_NAME},${seed},${nodes},${GPUS_PER_NODE},${CPUS_PER_TASK},NA,${REQUESTED_TIME_LIMIT},${QOS},${PARTITION},${PARAMS_FILE},shared_fs,${ALLOCATION_JOB_ID},${STATUS},${elapsed_hms},${START_LOCAL},${RUN_DIR},${TRAIN_LAST},${VAL_LAST},${TRAIN_BEST},${VAL_BEST},${NOTE}" \
      >> "${CSV}"

    rm -f "${TMP_LOG}"

    if (( rc != 0 )); then
      echo "ERROR: ${JOB_NAME} failed" >&2
      exit "${rc}"
    fi
  done
done

echo "corrected matrix complete: ${CSV}"
