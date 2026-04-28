#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-submit}"
CLUSTER_FILTER="${2:-all}"
if [[ "${MODE}" != "submit" && "${MODE}" != "test-only" && "${MODE}" != "refresh" ]]; then
  echo "usage: $0 [submit|test-only|refresh] [all|Falcon|Tinkercliffs]" >&2
  exit 2
fi
if [[ "${CLUSTER_FILTER}" != "all" && "${CLUSTER_FILTER}" != "Falcon" && "${CLUSTER_FILTER}" != "Tinkercliffs" ]]; then
  echo "invalid cluster filter: ${CLUSTER_FILTER}" >&2
  exit 2
fi

REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
TRAIN_SCRIPT="${REPO}/slurm_train.sbatch"
SMOKE_SCRIPT="${REPO}/slurm_smoke_test.sbatch"
TRAIN_CSV="${REPO}/data/important_notes/training_jobs.csv"
TEST_CSV="${REPO}/data/important_notes/testing_jobs.csv"
MANIFEST="${REPO}/data/important_notes/resubmission_manifest_20260306.tsv"

TAR_PATH="/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar"
DATA_PREFIX="publication_2020"
CURR="0.1"
TRAIN_PARAMS="pytorch/configs/params_dnn_tar.yaml"
DATA_ACCESS_MODE="copy_to_node"

COMMON_EXPORTS="ALL,DATA_ACCESS_MODE=${DATA_ACCESS_MODE},TAR_PATH=${TAR_PATH},DATA_PREFIX=${DATA_PREFIX},CURR=${CURR},TORCH_NCCL_ASYNC_ERROR_HANDLING=1,TORCH_NCCL_BLOCKING_WAIT=1,NCCL_DEBUG=WARN,NCCL_IB_DISABLE=0,NCCL_SOCKET_IFNAME=^lo,docker,veth"

run_tc_bash() {
  local cmd="$1"
  local quoted
  printf -v quoted '%q' "${cmd}"
  ssh -o BatchMode=yes tinkercliffs1 bash -lc "${quoted}"
}

run_falcon_bash() {
  local cmd="$1"
  local quoted
  printf -v quoted '%q' "${cmd}"
  ssh -o BatchMode=yes tinkercliffs1 ssh -o BatchMode=yes falcon1.arc.vt.edu bash -lc "${quoted}"
}

submit_or_test() {
  local target="$1"
  local cmd="$2"
  if [[ "${target}" == "Falcon" ]]; then
    run_falcon_bash "${cmd}"
  else
    run_tc_bash "${cmd}"
  fi
}

query_status() {
  local target="$1"
  local job_id="$2"
  local status=""
  if [[ "${target}" == "Falcon" ]]; then
    status="$(run_falcon_bash "squeue -h --jobs=${job_id} --format='%T|%M' | head -n 1")"
    if [[ -z "${status}" ]]; then
      status="$(run_falcon_bash "sacct -X -j ${job_id} -n -P -o State,Elapsed | tail -n 1")"
    fi
  else
    status="$(run_tc_bash "squeue --noheader --jobs=${job_id} --format='%T|%M' | head -n 1")"
  fi
  status="$(printf '%s' "${status}" | tr -d '\r' | head -n 1)"
  if [[ -z "${status}" ]]; then
    status="PENDING|00:00:00"
  fi
  printf '%s\n' "${status}"
}

query_job_by_name() {
  local target="$1"
  local job_name="$2"
  local result
  if [[ "${target}" == "Falcon" ]]; then
    result="$(run_falcon_bash "squeue -h -u \$(whoami) -o '%i|%j|%T|%M' | awk -F'|' '\$2==\"${job_name}\" {print \$1 \"|\" \$3 \"|\" \$4; exit}'")"
    if [[ -z "${result}" ]]; then
      result="$(run_falcon_bash "sacct -X -S $(date +%F) -n -P -o JobIDRaw,JobName,State,Elapsed | awk -F'|' '\$2==\"${job_name}\" {print \$1 \"|\" \$3 \"|\" \$4}' | tail -n 1")"
    fi
  else
    result="$(run_tc_bash "squeue -h -u \$(whoami) -o '%i|%j|%T|%M' | awk -F'|' '\$2==\"${job_name}\" {print \$1 \"|\" \$3 \"|\" \$4; exit}'")"
  fi
  result="$(printf '%s' "${result}" | tr -d '\r' | head -n 1)"
  printf '%s\n' "${result}"
}

append_csv_row() {
  local csv_file="$1"
  shift
  local joined=""
  local value
  for value in "$@"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${value}"
  done
  printf '%s\n' "${joined}" >> "${csv_file}"
}

cluster_enabled() {
  local cluster="$1"
  [[ "${CLUSTER_FILTER}" == "all" || "${CLUSTER_FILTER}" == "${cluster}" ]]
}

submit_row() {
  local suite_kind="$1"
  local cluster="$2"
  local gpu="$3"
  local node_type="$4"
  local job_name="$5"
  local partition="$6"
  local qos="$7"
  local nodes="$8"
  local gpus_per_node="$9"
  local cpus_per_node="${10}"
  local mem_gib="${11}"
  local ram_rank_gib="${12}"
  local requested_time="${13}"
  local constraint="${14}"
  local gpu_token="${15}"
  local target="${16}"

  local script_path="${TRAIN_SCRIPT}"
  local params_export=",PARAMS_FILE=${TRAIN_PARAMS}"
  if [[ "${suite_kind}" == "test" ]]; then
    script_path="${SMOKE_SCRIPT}"
    params_export=""
  fi

  local gpu_total=$(( nodes * gpus_per_node ))
  local cpu_total=$(( nodes * cpus_per_node ))
  local ram_total=$(( nodes * mem_gib ))
  local constraint_arg=""
  if [[ -n "${constraint}" ]]; then
    constraint_arg=" --constraint=${constraint}"
  fi

  local export_block="${COMMON_EXPORTS}${params_export}"
  local sbatch_flag="--parsable"
  if [[ "${MODE}" == "test-only" ]]; then
    sbatch_flag="--test-only"
  fi

  local cmd
  cmd="cd ${REPO} && sbatch ${sbatch_flag} --job-name=${job_name} --account=neuro-collab --partition=${partition} --qos=${qos} --nodes=${nodes} --ntasks-per-node=1 --cpus-per-task=${cpus_per_node} --mem=${mem_gib}G --gres=gpu:${gpu_token}:${gpus_per_node} --time=${requested_time}${constraint_arg} --export=${export_block} ${script_path}"

  if [[ "${MODE}" == "test-only" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t' "${suite_kind}" "${cluster}" "${gpu}" "${node_type}" "${job_name}"
    submit_or_test "${target}" "${cmd}"
    return 0
  fi

  local job_id=""
  local status=""
  local elapsed=""

  if [[ "${MODE}" == "submit" ]]; then
    job_id="$(submit_or_test "${target}" "${cmd}")"
    job_id="${job_id%%;*}"
    local status_line
    status_line="$(query_status "${target}" "${job_id}")"
    status="${status_line%%|*}"
    elapsed="${status_line##*|}"
  else
    local job_line
    job_line="$(query_job_by_name "${target}" "${job_name}")"
    if [[ -z "${job_line}" ]]; then
      echo "missing active queue entry for ${job_name} on ${target}" >&2
      exit 1
    fi
    IFS='|' read -r job_id status elapsed <<< "${job_line}"
  fi

  append_csv_row "${MANIFEST}" \
    "${suite_kind}" "${cluster}" "${gpu}" "${node_type}" "${job_name}" "${job_id}" "${status}" "${elapsed}" \
    "${requested_time}" "${qos}" "${partition}" "${constraint:-none}" "${nodes}" "${gpus_per_node}" "${gpu_total}" \
    "${cpus_per_node}" "${cpu_total}" "${mem_gib}" "${ram_rank_gib}" "${ram_total}" "${DATA_ACCESS_MODE}"

  if [[ "${suite_kind}" == "train" ]]; then
    append_csv_row "${TRAIN_TMP}" \
      "${cluster}" "${gpu}" "${node_type}" "${job_name}" "${job_id}" "${status}" "${elapsed}" "${requested_time}" "${qos}" \
      "${nodes}" "${gpus_per_node}" "${gpu_total}" "${cpus_per_node}" "${cpu_total}" "${mem_gib}" "${ram_rank_gib}" "${ram_total}" "${DATA_ACCESS_MODE}"
  else
    append_csv_row "${TEST_TMP}" \
      "${cluster}" "${gpu}" "${node_type}" "${job_name}" "${job_id}" "${status}" "${elapsed}" "${requested_time}" "${qos}" \
      "${nodes}" "${gpus_per_node}" "${gpu_total}" "${cpus_per_node}" "${cpu_total}" "${mem_gib}" "${ram_rank_gib}" "${ram_total}" "${DATA_ACCESS_MODE}"
  fi
}

TRAIN_ROWS=(
  "Falcon|V100|fal-v100|trn_V100_1n|v100_normal_q|fal_v100_normal_base|1|2|24|64|32|1-00:00:00||v100|Falcon"
  "Falcon|V100|fal-v100|trn_V100_2n|v100_normal_q|fal_v100_normal_base|2|2|24|64|32|1-00:00:00||v100|Falcon"
  "Falcon|V100|fal-v100|trn_V100_4n|v100_normal_q|fal_v100_normal_base|4|2|24|64|32|1-00:00:00||v100|Falcon"
  "Falcon|A30|fal-a30|trn_A30_1n|a30_normal_q|fal_a30_normal_short|1|4|48|64|16|1-00:00:00||a30|Falcon"
  "Falcon|A30|fal-a30|trn_A30_2n|a30_normal_q|fal_a30_normal_short|2|4|48|64|16|1-00:00:00||a30|Falcon"
  "Falcon|A30|fal-a30|trn_A30_4n|a30_normal_q|fal_a30_normal_short|4|4|48|64|16|1-00:00:00||a30|Falcon"
  "Falcon|L40S|fal-l40s|trn_L40S_1n|l40s_normal_q|fal_l40s_normal_short|1|4|48|64|16|1-00:00:00||l40s|Falcon"
  "Falcon|L40S|fal-l40s|trn_L40S_2n|l40s_normal_q|fal_l40s_normal_short|2|4|48|64|16|1-00:00:00||l40s|Falcon"
  "Falcon|L40S|fal-l40s|trn_L40S_4n|l40s_normal_q|fal_l40s_normal_short|4|4|48|64|16|1-00:00:00||l40s|Falcon"
  "Tinkercliffs|A100|dgx-A100|trn_A100_1n_dgx|a100_normal_q|tc_a100_normal_short|1|8|48|128|16|1-00:00:00|dgx-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|dgx-A100|trn_A100_2n_dgx|a100_normal_q|tc_a100_normal_short|2|8|48|128|16|1-00:00:00|dgx-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|dgx-A100|trn_A100_4n_dgx|a100_normal_q|tc_a100_normal_short|4|8|48|128|16|1-00:00:00|dgx-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|hpe-A100|trn_A100_1n_hpe|a100_normal_q|tc_a100_normal_short|1|8|48|128|16|1-00:00:00|hpe-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|hpe-A100|trn_A100_2n_hpe|a100_normal_q|tc_a100_normal_short|2|8|48|128|16|1-00:00:00|hpe-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|hpe-A100|trn_A100_4n_hpe|a100_normal_q|tc_a100_normal_short|4|8|48|128|16|1-00:00:00|hpe-A100|a100|Tinkercliffs"
  "Tinkercliffs|H200|tc-xe|trn_H200_1n|h200_normal_q|tc_h200_normal_short|1|8|48|128|16|1-00:00:00||h200|Tinkercliffs"
  "Tinkercliffs|H200|tc-xe|trn_H200_2n|h200_normal_q|tc_h200_normal_short|2|8|48|128|16|1-00:00:00||h200|Tinkercliffs"
  "Tinkercliffs|H200|tc-xe|trn_H200_4n|h200_normal_q|tc_h200_normal_short|4|5|42|80|16|1-00:00:00||h200|Tinkercliffs"
)

TEST_ROWS=(
  "Falcon|V100|fal-v100|tst_V100_1n|v100_normal_q|fal_v100_normal_base|1|2|24|32|16|00:30:00||v100|Falcon"
  "Falcon|V100|fal-v100|tst_V100_2n|v100_normal_q|fal_v100_normal_base|2|2|24|32|16|00:30:00||v100|Falcon"
  "Falcon|V100|fal-v100|tst_V100_4n|v100_normal_q|fal_v100_normal_base|4|2|24|32|16|00:30:00||v100|Falcon"
  "Falcon|A30|fal-a30|tst_A30_1n|a30_normal_q|fal_a30_normal_short|1|4|48|32|8|00:30:00||a30|Falcon"
  "Falcon|A30|fal-a30|tst_A30_2n|a30_normal_q|fal_a30_normal_short|2|4|48|32|8|00:30:00||a30|Falcon"
  "Falcon|A30|fal-a30|tst_A30_4n|a30_normal_q|fal_a30_normal_short|4|4|48|32|8|00:30:00||a30|Falcon"
  "Falcon|L40S|fal-l40s|tst_L40S_1n|l40s_normal_q|fal_l40s_normal_short|1|4|48|32|8|00:30:00||l40s|Falcon"
  "Falcon|L40S|fal-l40s|tst_L40S_2n|l40s_normal_q|fal_l40s_normal_short|2|4|48|32|8|00:30:00||l40s|Falcon"
  "Falcon|L40S|fal-l40s|tst_L40S_4n|l40s_normal_q|fal_l40s_normal_short|4|4|48|32|8|00:30:00||l40s|Falcon"
  "Tinkercliffs|A100|dgx-A100|tst_A100_1n_dgx|a100_normal_q|tc_a100_normal_short|1|8|48|32|4|00:30:00|dgx-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|dgx-A100|tst_A100_2n_dgx|a100_normal_q|tc_a100_normal_short|2|8|48|32|4|00:30:00|dgx-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|dgx-A100|tst_A100_4n_dgx|a100_normal_q|tc_a100_normal_short|4|8|48|32|4|00:30:00|dgx-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|hpe-A100|tst_A100_1n_hpe|a100_normal_q|tc_a100_normal_short|1|8|48|32|4|00:30:00|hpe-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|hpe-A100|tst_A100_2n_hpe|a100_normal_q|tc_a100_normal_short|2|8|48|32|4|00:30:00|hpe-A100|a100|Tinkercliffs"
  "Tinkercliffs|A100|hpe-A100|tst_A100_4n_hpe|a100_normal_q|tc_a100_normal_short|4|8|48|32|4|00:30:00|hpe-A100|a100|Tinkercliffs"
  "Tinkercliffs|H200|tc-xe|tst_H200_1n|h200_normal_q|tc_h200_normal_short|1|8|48|32|4|00:30:00||h200|Tinkercliffs"
  "Tinkercliffs|H200|tc-xe|tst_H200_2n|h200_normal_q|tc_h200_normal_short|2|8|48|32|4|00:30:00||h200|Tinkercliffs"
  "Tinkercliffs|H200|tc-xe|tst_H200_4n|h200_normal_q|tc_h200_normal_short|4|5|42|32|6.40|00:30:00||h200|Tinkercliffs"
)

if [[ "${MODE}" == "submit" || "${MODE}" == "refresh" ]]; then
  TRAIN_TMP="$(mktemp)"
  TEST_TMP="$(mktemp)"
  : > "${MANIFEST}"

  printf '%s\n' "Cluster,GPU,Node type,Job name,Job ID,Status,Elapsed job time,Requested time limit,QoS,Nodes,GPUs/node,GPUs total,CPUs/node,CPUs total,RAM/node (GiB),RAM/rank (GiB),RAM total (GiB),Data access mode" > "${TRAIN_TMP}"
  printf '%s\n' "Cluster,GPU,Node type,Job name,Job ID,Status,Elapsed job time,Requested time limit,QoS,Nodes,GPUs/node,GPUs total,CPUs/node,CPUs total,RAM/node (GiB),RAM/rank (GiB),RAM total (GiB),Data access mode" > "${TEST_TMP}"
  printf '%s\n' "suite_kind	cluster	gpu	node_type	job_name	job_id	status	elapsed	requested_time	qos	partition	constraint	nodes	gpus_per_node	gpus_total	cpus_per_node	cpus_total	ram_node_gib	ram_rank_gib	ram_total_gib	data_access_mode" > "${MANIFEST}"
fi

row=""
for row in "${TRAIN_ROWS[@]}"; do
  IFS='|' read -r cluster gpu node_type job_name partition qos nodes gpus_per_node cpus_per_node mem_gib ram_rank_gib requested_time constraint gpu_token target <<< "${row}"
  cluster_enabled "${cluster}" || continue
  submit_row "train" "${cluster}" "${gpu}" "${node_type}" "${job_name}" "${partition}" "${qos}" "${nodes}" "${gpus_per_node}" "${cpus_per_node}" "${mem_gib}" "${ram_rank_gib}" "${requested_time}" "${constraint}" "${gpu_token}" "${target}"
done

for row in "${TEST_ROWS[@]}"; do
  IFS='|' read -r cluster gpu node_type job_name partition qos nodes gpus_per_node cpus_per_node mem_gib ram_rank_gib requested_time constraint gpu_token target <<< "${row}"
  cluster_enabled "${cluster}" || continue
  submit_row "test" "${cluster}" "${gpu}" "${node_type}" "${job_name}" "${partition}" "${qos}" "${nodes}" "${gpus_per_node}" "${cpus_per_node}" "${mem_gib}" "${ram_rank_gib}" "${requested_time}" "${constraint}" "${gpu_token}" "${target}"
done

if [[ "${MODE}" == "submit" || "${MODE}" == "refresh" ]]; then
  if [[ "${CLUSTER_FILTER}" == "all" ]]; then
    mv "${TRAIN_TMP}" "${TRAIN_CSV}"
    mv "${TEST_TMP}" "${TEST_CSV}"
  else
    MERGED_TRAIN="$(mktemp)"
    MERGED_TEST="$(mktemp)"
    head -n 1 "${TRAIN_TMP}" > "${MERGED_TRAIN}"
    tail -n +2 "${TRAIN_TMP}" >> "${MERGED_TRAIN}"
    awk -F, -v cluster="${CLUSTER_FILTER}" 'NR>1 && $1 != cluster { print }' "${TRAIN_CSV}" >> "${MERGED_TRAIN}"
    head -n 1 "${TEST_TMP}" > "${MERGED_TEST}"
    tail -n +2 "${TEST_TMP}" >> "${MERGED_TEST}"
    awk -F, -v cluster="${CLUSTER_FILTER}" 'NR>1 && $1 != cluster { print }' "${TEST_CSV}" >> "${MERGED_TEST}"
    mv "${MERGED_TRAIN}" "${TRAIN_CSV}"
    mv "${MERGED_TEST}" "${TEST_CSV}"
  fi
fi
