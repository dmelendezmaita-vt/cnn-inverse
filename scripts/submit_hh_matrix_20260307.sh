#!/usr/bin/env bash
set -euo pipefail

# HH (third track) submission helper.
# Default mode is test-only (shows sbatch commands without submitting).
# Usage: bash scripts/submit_hh_matrix_20260307.sh [submit|test-only]

MODE="${1:-test-only}"
if [[ "${MODE}" != "submit" && "${MODE}" != "test-only" ]]; then
  echo "usage: $0 [submit|test-only]" >&2
  exit 2
fi

REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
TRAIN_SCRIPT="${REPO}/slurm_train.sbatch"
SMOKE_SCRIPT="${REPO}/slurm_smoke_test.sbatch"
PLAN_SEEDED="${REPO}/data/important_notes/third_track_hh/hh_seeded_training_plan_20260307.csv"
TRACK_SEEDED="${REPO}/data/important_notes/third_track_hh/hh_seeded_training_jobs.csv"
TRACK_TEST="${REPO}/data/important_notes/third_track_hh/hh_testing_jobs.csv"

# TODO: point to the HH dataset tarball/prefix actually in use.
TAR_PATH_HH="${TAR_PATH_HH:-/projects/neuro-collab/data/tar_files/SET_ME_HH_DATA.tar}"
DATA_PREFIX_HH="${DATA_PREFIX_HH:-SET_ME_HH_PREFIX}"
PARAMS_FILE_HH="pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml"
DATA_ACCESS_MODE="copy_to_node"
CURR="0.1"

if [[ "${TAR_PATH_HH}" == *SET_ME* || ! -f "${TAR_PATH_HH}" ]]; then
  echo "Set TAR_PATH_HH to the HH dataset tarball before submitting. Current: ${TAR_PATH_HH}" >&2
  exit 3
fi
if [[ "${DATA_PREFIX_HH}" == SET_ME_HH_PREFIX ]]; then
  echo "Set DATA_PREFIX_HH to the HH data prefix before submitting." >&2
  exit 3
fi

COMMON_EXPORTS="ALL,DATA_ACCESS_MODE=${DATA_ACCESS_MODE},TAR_PATH=${TAR_PATH_HH},DATA_PREFIX=${DATA_PREFIX_HH},CURR=${CURR},TORCH_NCCL_ASYNC_ERROR_HANDLING=1,TORCH_NCCL_BLOCKING_WAIT=1,NCCL_DEBUG=WARN,NCCL_IB_DISABLE=0,NCCL_SOCKET_IFNAME=^lo,docker,veth"

run_tc_bash() {
  local cmd="$1"
  local quoted
  printf -v quoted '%q' "${cmd}"
  ssh -n -o BatchMode=yes tinkercliffs1 bash -lc "${quoted}"
}

submit_or_test() {
  local cmd="$1"
  if [[ "${MODE}" == "test-only" ]]; then
    run_tc_bash "${cmd}"
  else
    run_tc_bash "${cmd}"
  fi
}

append_csv_row() {
  local csv_file="$1"
  shift
  printf '%s\n' "$(IFS=,; echo "$*")" >> "${csv_file}"
}

submit_training_from_plan() {
  local plan="${PLAN_SEEDED}"
  tail -n +2 "${plan}" | while IFS=, read -r suite seed cluster gpu node_type job_name time_limit qos partition params_file data_mode nodes gpn gtotal cpus node_cpus_total ram ram_per_rank ram_total job_id status elapsed sched_start queue_detail; do
    # Only Tinkercliffs rows are expected.
    local gpu_token
    case "${gpu}" in
      A100) gpu_token="a100" ;;
      H200) gpu_token="h200" ;;
      *) echo "Unknown GPU type in plan: ${gpu}" >&2; exit 4 ;;
    esac
    local cmd="cd ${REPO} && sbatch --parsable --job-name=${job_name} --account=neuro-collab --partition=${partition} --qos=${qos} --nodes=${nodes} --ntasks-per-node=1 --cpus-per-task=${cpus} --mem=${ram}G --gres=gpu:${gpu_token}:${gpn} --time=${time_limit} --export=${COMMON_EXPORTS},PARAMS_FILE=${PARAMS_FILE_HH},SEED=${seed} ${TRAIN_SCRIPT}"
    if [[ "${MODE}" == "test-only" ]]; then
      printf '[test-only] %s\n' "${cmd}"
      continue
    fi
    local new_job_id
    new_job_id="$(submit_or_test "${cmd}")"
    new_job_id="${new_job_id%%;*}"
    append_csv_row "${TRACK_SEEDED}" "${cluster}" "${gpu}" "${node_type}" "${job_name}" "${new_job_id}" "SUBMITTED" "0:00" "${time_limit}" "${qos}" "${nodes}" "${gpn}" "${gtotal}" "${cpus}" "${node_cpus_total}" "${ram}" "${ram_per_rank}" "${ram_total}" "${data_mode}"
  done
}

submit_testing_matrix() {
  # 30-minute HH smoke tests across node types.
  local test_rows=(
    "A100|dgx-A100|tst_HH_A100_1n_dgx|1|8|48|32|copy_to_node|00:30:00|tc_a100_normal_short|a100_normal_q"
    "A100|dgx-A100|tst_HH_A100_2n_dgx|2|8|48|32|copy_to_node|00:30:00|tc_a100_normal_short|a100_normal_q"
    "A100|dgx-A100|tst_HH_A100_4n_dgx|4|8|48|32|copy_to_node|00:30:00|tc_a100_normal_short|a100_normal_q"
    "A100|hpe-A100|tst_HH_A100_1n_hpe|1|8|48|32|copy_to_node|00:30:00|tc_a100_normal_short|a100_normal_q"
    "A100|hpe-A100|tst_HH_A100_2n_hpe|2|8|48|32|copy_to_node|00:30:00|tc_a100_normal_short|a100_normal_q"
    "A100|hpe-A100|tst_HH_A100_4n_hpe|4|8|48|32|copy_to_node|00:30:00|tc_a100_normal_short|a100_normal_q"
    "H200|tc-xe|tst_HH_H200_1n|1|8|48|32|copy_to_node|00:30:00|tc_h200_normal_short|h200_normal_q"
    "H200|tc-xe|tst_HH_H200_2n|2|8|48|32|copy_to_node|00:30:00|tc_h200_normal_short|h200_normal_q"
    "H200|tc-xe|tst_HH_H200_4n|4|5|42|32|copy_to_node|00:30:00|tc_h200_normal_short|h200_normal_q"
  )
  local gpu gpu_token node_type job_name nodes gpn cpus mem data_mode wall qos partition
  for row in "${test_rows[@]}"; do
    IFS='|' read -r gpu node_type job_name nodes gpn cpus mem data_mode wall qos partition <<< "${row}"
    case "${gpu}" in
      A100) gpu_token="a100" ;;
      H200) gpu_token="h200" ;;
    esac
    local cmd="cd ${REPO} && sbatch --parsable --job-name=${job_name} --account=neuro-collab --partition=${partition} --qos=${qos} --nodes=${nodes} --ntasks-per-node=1 --cpus-per-task=${cpus} --mem=${mem}G --gres=gpu:${gpu_token}:${gpn} --time=${wall} --export=${COMMON_EXPORTS},PARAMS_FILE=${PARAMS_FILE_HH} ${SMOKE_SCRIPT}"
    if [[ "${MODE}" == "test-only" ]]; then
      printf '[test-only] %s\n' "${cmd}"
      continue
    fi
    local job_id
    job_id="$(submit_or_test "${cmd}")"
    job_id="${job_id%%;*}"
    local gpu_total=$(( nodes * gpn ))
    local cpu_total=$(( nodes * cpus ))
    local ram_total=$(( nodes * mem ))
    append_csv_row "${TRACK_TEST}" "Tinkercliffs" "${gpu}" "${node_type}" "${job_name}" "${job_id}" "SUBMITTED" "0:00" "${wall}" "${qos}" "${nodes}" "${gpn}" "${gpu_total}" "${cpus}" "${cpu_total}" "${mem}" "16" "${ram_total}" "${data_mode}"
  done
}

submit_training_from_plan
submit_testing_matrix

echo "HH staging complete (mode=${MODE})."
