#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"
if [[ "${MODE}" != "plan" && "${MODE}" != "submit" && "${MODE}" != "test-only" ]]; then
  echo "usage: $0 [plan|submit|test-only]" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${REPO}/slurm_train.sbatch"
PLAN_CSV="${REPO}/data/important_notes/falcon_seeded_training_plan_20260306.csv"
TRACK_CSV="${REPO}/data/important_notes/falcon_seeded_training_jobs.csv"

TAR_PATH="/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar"
DATA_PREFIX="publication_2020"
CURR="0.1"
DATA_ACCESS_MODE="copy_to_node"
COMMON_EXPORTS="ALL,DATA_ACCESS_MODE=${DATA_ACCESS_MODE},TAR_PATH=${TAR_PATH},DATA_PREFIX=${DATA_PREFIX},CURR=${CURR},TORCH_NCCL_ASYNC_ERROR_HANDLING=1,TORCH_NCCL_BLOCKING_WAIT=1,NCCL_DEBUG=WARN"
SEEDS=(301 302 303 304 305 306 307 308 309 310 311 312)

find_existing_job() {
  local job_name="$1"
  local line
  line="$(run_falcon_bash "squeue -h -u \$(whoami) -o '%i|%T|%M|%j' | awk -F'|' '\$4==\"${job_name}\"{print \$1\"|\"\$2\"|\"\$3; exit}'")"
  if [[ -n \"${line}\" ]]; then
    printf '%s\n' \"${line}\"
    return 0
  fi
  line="$(run_falcon_bash \"sacct -X -S 2026-03-01 -n -P -o JobIDRaw,JobName,State,Elapsed | awk -F'|' '\\$2==\\\"${job_name}\\\" {print \\$1 \\\"|\\\" \\$3 \\\"|\\\" \\$4; exit}'\")"
  [[ -n \"${line}\" ]] && printf '%s\n' \"${line}\"
}

run_falcon_bash() {
  local cmd="$1"
  local quoted
  printf -v quoted "%q" "${cmd}"
  ssh -o BatchMode=yes tinkercliffs1 ssh -o BatchMode=yes falcon1.arc.vt.edu bash -lc "${quoted}"
}

query_status() {
  local job_id="$1"
  local line=""
  line="$(run_falcon_bash "squeue -h --jobs=${job_id} --format='%T|%M' | head -n 1")"
  if [[ -z "${line}" ]]; then
    line="$(run_falcon_bash "sacct -X -j ${job_id} -n -P -o State,Elapsed | tail -n 1")"
  fi
  line="$(printf "%s" "${line}" | tr -d '\r' | head -n 1)"
  if [[ -z "${line}" ]]; then
    line="PENDING|00:00:00"
  fi
  printf "%s\n" "${line}"
}

append_row() {
  local csv_file="$1"
  shift
  local out=""
  local field
  for field in "$@"; do
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${field}"
  done
  printf "%s\n" "${out}" >> "${csv_file}"
}

submit_row() {
  local seed="$1"
  local gpu="$2"
  local node_type="$3"
  local job_stem="$4"
  local partition="$5"
  local qos="$6"
  local nodes="$7"
  local gpus_per_node="$8"
  local cpus_per_node="$9"
  local mem_gib="${10}"
  local ram_rank_gib="${11}"
  local gpu_token="${12}"

  local params_file="src/pytorch/configs/falcon_seed_repeats/params_dnn_tar_seed${seed}.yaml"
  local job_name="${job_stem}_s${seed}"
  local gpu_total=$(( nodes * gpus_per_node ))
  local cpu_total=$(( nodes * cpus_per_node ))
  local ram_total=$(( nodes * mem_gib ))
  local export_block="${COMMON_EXPORTS},PARAMS_FILE=${params_file}"
  local sbatch_flag="--parsable"

  if [[ "${MODE}" == "test-only" ]]; then
    sbatch_flag="--test-only"
  fi

  local cmd="cd ${REPO} && sbatch ${sbatch_flag} --job-name=${job_name} --account=neuro-collab --partition=${partition} --qos=${qos} --nodes=${nodes} --ntasks-per-node=1 --cpus-per-task=${cpus_per_node} --mem=${mem_gib}G --gres=gpu:${gpu_token}:${gpus_per_node} --time=1-00:00:00 --export=${export_block} ${TRAIN_SCRIPT}"

  if [[ "${MODE}" == "test-only" ]]; then
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "falcon_seeded_baseline_20260306" "${seed}" "Falcon" "${gpu}" "${node_type}" "${job_name}" \
      "1-00:00:00" "${qos}" "${partition}" "${params_file}" "${DATA_ACCESS_MODE}" \
      "${nodes}" "${gpus_per_node}" "${gpu_total}" "${cpus_per_node}" "${cpu_total}" \
      "${mem_gib}" "${ram_rank_gib}" "${ram_total}"
    run_falcon_bash "${cmd}"
    return 0
  fi

  local job_id status elapsed status_line
  if [[ "${MODE}" == "submit" ]]; then
    existing="$(find_existing_job "${job_name}")"
    if [[ -n "${existing}" ]]; then
      IFS='|' read -r job_id status elapsed <<< "${existing}"
    else
      job_id="$(run_falcon_bash "${cmd}")"
      job_id="${job_id%%;*}"
      status_line="$(query_status "${job_id}")"
      IFS='|' read -r status elapsed <<< "${status_line}"
    fi
  else
    status_line="$(query_status "${job_name}")"
    job_id="${status_line%%|*}"
    status="${status_line#*|}"; status="${status%%|*}"
    elapsed="${status_line##*|}"
  fi

  append_row "${TMP_CSV}" \
    "falcon_seeded_baseline_20260306" "${seed}" "Falcon" "${gpu}" "${node_type}" "${job_name}" \
    "${job_id}" "${status}" "${elapsed}" "1-00:00:00" "${qos}" "${partition}" \
    "${params_file}" "${DATA_ACCESS_MODE}" "${nodes}" "${gpus_per_node}" "${gpu_total}" \
    "${cpus_per_node}" "${cpu_total}" "${mem_gib}" "${ram_rank_gib}" "${ram_total}"
}

ROWS=(
  "V100|fal-v100|trn_V100_1n|v100_normal_q|fal_v100_normal_base|1|2|24|64|32|v100"
  "V100|fal-v100|trn_V100_2n|v100_normal_q|fal_v100_normal_base|2|2|24|64|32|v100"
  "V100|fal-v100|trn_V100_4n|v100_normal_q|fal_v100_normal_base|4|2|24|64|32|v100"
  "A30|fal-a30|trn_A30_1n|a30_normal_q|fal_a30_normal_short|1|4|48|64|16|a30"
  "A30|fal-a30|trn_A30_2n|a30_normal_q|fal_a30_normal_short|2|4|48|64|16|a30"
  "A30|fal-a30|trn_A30_4n|a30_normal_q|fal_a30_normal_short|4|4|48|64|16|a30"
  "L40S|fal-l40s|trn_L40S_1n|l40s_normal_q|fal_l40s_normal_short|1|4|48|64|16|l40s"
  "L40S|fal-l40s|trn_L40S_2n|l40s_normal_q|fal_l40s_normal_short|2|4|48|64|16|l40s"
  "L40S|fal-l40s|trn_L40S_4n|l40s_normal_q|fal_l40s_normal_short|4|4|48|64|16|l40s"
)

if [[ "${MODE}" == "plan" ]]; then
  cat "${PLAN_CSV}"
  exit 0
fi

TMP_CSV="$(mktemp)"
printf "%s\n" "Suite,Seed,Cluster,GPU,Node type,Job name,Job ID,Status,Elapsed job time,Requested time limit,QoS,Partition,Params file,Data access mode,Nodes,GPUs/node,GPUs total,CPUs/node,CPUs total,RAM/node (GiB),RAM/rank (GiB),RAM total (GiB)" > "${TMP_CSV}"

for seed in "${SEEDS[@]}"; do
  for row in "${ROWS[@]}"; do
    IFS='|' read -r gpu node_type job_stem partition qos nodes gpus_per_node cpus_per_node mem_gib ram_rank_gib gpu_token <<< "${row}"
    submit_row "${seed}" "${gpu}" "${node_type}" "${job_stem}" "${partition}" "${qos}" "${nodes}" "${gpus_per_node}" "${cpus_per_node}" "${mem_gib}" "${ram_rank_gib}" "${gpu_token}"
  done
done

mv "${TMP_CSV}" "${TRACK_CSV}"
