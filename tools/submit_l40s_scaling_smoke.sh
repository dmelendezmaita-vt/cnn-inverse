#!/usr/bin/env bash
set -euo pipefail

SBATCH_SCRIPT="${SBATCH_SCRIPT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch}"
PARTITION="${PARTITION:-l40s_normal_q}"
QOS="${QOS:-fal_l40s_normal_short}"
TIME_LIMIT="${TIME_LIMIT:-00:12:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
MEM_PER_NODE="${MEM_PER_NODE:-32G}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NPROC="${NPROC:-4}"
LOG_DIR="${LOG_DIR:-/projects/neuro-collab/data/runs/logs}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: sbatch script not found: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_MAP="${LOG_DIR}/l40s_scaling_jobs_${STAMP}.tsv"

echo -e "label\tjob_id\tnodes\tpartition\tqos\ttime\tgpus_per_node\tcpus_per_task\tmem_per_node\tnproc\tsbatch_script" > "${JOB_MAP}"

submit_job() {
  local label="$1"
  local nodes="$2"
  local -a qos_arg=()

  if [[ -n "${QOS}" ]]; then
    qos_arg=(--qos="${QOS}")
  fi

  local job_raw job_id
  job_raw="$(sbatch --parsable \
    --job-name="${label}" \
    --partition="${PARTITION}" \
    "${qos_arg[@]}" \
    --nodes="${nodes}" \
    --ntasks="${nodes}" \
    --ntasks-per-node=1 \
    --gres="gpu:${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM_PER_NODE}" \
    --time="${TIME_LIMIT}" \
    --export="ALL,NPROC=${NPROC}" \
    "${SBATCH_SCRIPT}")"

  job_id="${job_raw%%;*}"
  echo "${label}: ${job_id}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" \
    "${job_id}" \
    "${nodes}" \
    "${PARTITION}" \
    "${QOS}" \
    "${TIME_LIMIT}" \
    "${GPUS_PER_NODE}" \
    "${CPUS_PER_TASK}" \
    "${MEM_PER_NODE}" \
    "${NPROC}" \
    "${SBATCH_SCRIPT}" >> "${JOB_MAP}"
}

submit_job "fhn_smoke_1n" 1
submit_job "fhn_smoke_2n" 2
submit_job "fhn_smoke_4n" 4

echo "job map saved: ${JOB_MAP}"
