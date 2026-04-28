#!/usr/bin/env bash
set -euo pipefail

# Submit HH_full_v2 (fifth track) unseeded matrix on Tinkercliffs and both seeded/unseeded on Falcon.
# Default mode is test-only. Use MODE=submit (or TEST_ONLY=0 MODE=submit) to actually submit.
#
# Env you may override:
#   TAR_PATH_HH (default: /projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar)
#   DATA_PREFIX_HH (default: concatenated_data)
#   CURR (default: 0.1)
#   MODE (submit|test-only) or TEST_ONLY=1
#   MIN_RAM_PER_RANK_GIB (default: 48)
#   FALCON_GPU_FILTER (ALL|V100|A30|L40S; default: ALL)

MODE="${MODE:-test-only}"
if [[ "${TEST_ONLY:-0}" == "1" ]]; then MODE="test-only"; fi
if [[ "${MODE}" != "submit" && "${MODE}" != "test-only" ]]; then
  echo "MODE must be submit or test-only" >&2
  exit 2
fi
TARGET="${TARGET:-ALL}"
if [[ "${TARGET}" != "ALL" && "${TARGET}" != "TC" && "${TARGET}" != "FAL" ]]; then
  echo "TARGET must be ALL, TC, or FAL" >&2
  exit 2
fi
FALCON_GPU_FILTER="${FALCON_GPU_FILTER:-ALL}"
FALCON_GPU_FILTER_UPPER="${FALCON_GPU_FILTER^^}"
if [[ "${FALCON_GPU_FILTER_UPPER}" != "ALL" && "${FALCON_GPU_FILTER_UPPER}" != "V100" && "${FALCON_GPU_FILTER_UPPER}" != "A30" && "${FALCON_GPU_FILTER_UPPER}" != "L40S" ]]; then
  echo "FALCON_GPU_FILTER must be ALL, V100, A30, or L40S" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${REPO}/slurm_train.sbatch"
SMOKE_SCRIPT="${REPO}/slurm_smoke_test.sbatch"
PARAMS_FILE_HH="src/pytorch/configs/fifth_track_hh_full_refined/params_dnn_tar_hh_full_refined.yaml"
TAR_PATH_HH="${TAR_PATH_HH:-/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar}"
DATA_PREFIX_HH="${DATA_PREFIX_HH:-concatenated_data}"
CURR="${CURR:-0.1}"
DATA_ACCESS_MODE_DEFAULT="${DATA_ACCESS_MODE_DEFAULT:-copy_to_node}"
FALCON_V100_DATA_ACCESS_MODE="${FALCON_V100_DATA_ACCESS_MODE:-direct_tar}"
MIN_RAM_PER_RANK_GIB="${MIN_RAM_PER_RANK_GIB:-48}"
TMP_NVME_GIB="${TMP_NVME_GIB:-280}"
TMP_NVME_GIB_FAL_V100="${TMP_NVME_GIB_FAL_V100:-20}"
COMMON_EXPORTS_BASE="ALL,TAR_PATH=${TAR_PATH_HH},DATA_PREFIX=${DATA_PREFIX_HH},CURR=${CURR},TORCH_NCCL_ASYNC_ERROR_HANDLING=1,TORCH_NCCL_BLOCKING_WAIT=1,NCCL_DEBUG=WARN,NCCL_IB_DISABLE=0,NCCL_SOCKET_IFNAME=^lo,docker,veth"

TRAIN_CSV_TC="${REPO}/data/important_notes/fifth_track_hh_full_refined/hh_full_v2_training_jobs.csv"
TEST_CSV_TC="${REPO}/data/important_notes/fifth_track_hh_full_refined/hh_full_v2_testing_jobs.csv"
SEED_CSV_TC="${REPO}/data/important_notes/fifth_track_hh_full_refined/hh_full_v2_seeded_training_jobs.csv"
SEED_TEST_CSV_TC="${REPO}/data/important_notes/fifth_track_hh_full_refined/hh_full_v2_seeded_testing_jobs.csv"
TRAIN_CSV_FAL="${TRAIN_CSV_TC}"
TEST_CSV_FAL="${TEST_CSV_TC}"
SEED_CSV_FAL="${SEED_CSV_TC}"
SEED_TEST_CSV_FAL="${SEED_TEST_CSV_TC}"

run_tc() {
  local cmd="$1"
  ssh -n -o BatchMode=yes tinkercliffs1 bash -lc "$cmd"
}
run_fal() {
  local cmd="$1"
  local quoted
  printf -v quoted '%q' "${cmd}"
  ssh -n -o BatchMode=yes tinkercliffs1 "ssh -o BatchMode=yes falcon1.arc.vt.edu bash -lc ${quoted}"
}

submit_row() {
  local target="$1" cluster="$2" job_name="$3" partition="$4" qos="$5" nodes="$6" gpn="$7" cpus="$8" mem_gib="$9" wall="${10}" gpu_token="${11}" params_extra="${12}" csv_file="${13}"
  local data_access_mode="${DATA_ACCESS_MODE_DEFAULT}"
  local tmp_gib="${TMP_NVME_GIB}"
  # Falcon V100 nodes have limited local TmpDisk; use direct_tar and a feasible --tmp request.
  if [[ "${target}" == "FAL" && "${gpu_token,,}" == "v100" ]]; then
    data_access_mode="${FALCON_V100_DATA_ACCESS_MODE}"
    tmp_gib="${TMP_NVME_GIB_FAL_V100}"
  fi
  local common_exports="${COMMON_EXPORTS_BASE},DATA_ACCESS_MODE=${data_access_mode}"
  local min_mem_gib=$(( gpn * MIN_RAM_PER_RANK_GIB ))
  if (( mem_gib < min_mem_gib )); then
    echo "[WARN] ${job_name}: bumping mem from ${mem_gib}G to ${min_mem_gib}G to satisfy ${MIN_RAM_PER_RANK_GIB} GiB/rank." >&2
    mem_gib="${min_mem_gib}"
  fi
  local ram_rank_gib=$(( mem_gib / gpn ))
  local sb_cmd="cd ${REPO} && sbatch --parsable --job-name=${job_name} --account=neuro-collab --partition=${partition} --qos=${qos} --nodes=${nodes} --ntasks-per-node=1 --cpus-per-task=${cpus} --mem=${mem_gib}G --tmp=${tmp_gib}G --gres=gpu:${gpu_token}:${gpn} --time=${wall} --export=${common_exports}${params_extra} ${SMOKE_SCRIPT}"
  if [[ "${csv_file}" == *training* ]]; then
    sb_cmd="${sb_cmd/$SMOKE_SCRIPT/$TRAIN_SCRIPT}"
  fi
  if [[ "${MODE}" == "test-only" ]]; then
    printf "[test-only][%s] %s\n" "${cluster}" "${sb_cmd}"
    echo "JOBID_PLACEHOLDER,${job_name}" > /dev/null
    return
  fi
  local out
  if [[ "${target}" == "TC" ]]; then
    out="$(run_tc "${sb_cmd}")"
  else
    out="$(run_fal "${sb_cmd}")"
  fi
  local job_id="${out%%;*}"
  local gpu_total=$(( nodes * gpn ))
  local cpu_total=$(( nodes * cpus ))
  local ram_total=$(( nodes * mem_gib ))
  # Append to CSV
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${cluster}" "${gpu_token^^}" "$( [[ ${target} == TC ]] && echo ${node_type:-} || echo ${node_type:-})" \
    "${job_name}" "${job_id}" "SUBMITTED" "0:00" "${wall}" "${qos}" \
    "${nodes}" "${gpn}" "${gpu_total}" "${cpus}" "${cpu_total}" "${mem_gib}" "${ram_rank_gib}" "${ram_total}" "${data_access_mode}" >> "${csv_file}"
}

tc_suffix() {
  local node_type="$1"
  case "${node_type}" in
    dgx-A100) echo "_dgx" ;;
    hpe-A100) echo "_hpe" ;;
    *) echo "" ;;
  esac
}

falcon_gpu_allowed() {
  local gpu="$1"
  if [[ "${FALCON_GPU_FILTER_UPPER}" == "ALL" ]]; then
    return 0
  fi
  [[ "${gpu^^}" == "${FALCON_GPU_FILTER_UPPER}" ]]
}

# TC unseeded training/testing matrices
tc_jobs_train=(
  "A100|dgx-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|01:30:00|a100"
  "A100|dgx-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|01:30:00|a100"
  "A100|dgx-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|01:30:00|a100"
  "A100|hpe-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|01:30:00|a100"
  "A100|hpe-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|01:30:00|a100"
  "A100|hpe-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|01:30:00|a100"
  "H200|tc-h200|h200_normal_q|tc_h200_normal_short|1|8|48|256|01:30:00|h200"
  "H200|tc-h200|h200_normal_q|tc_h200_normal_short|2|8|48|256|01:30:00|h200"
  "H200|tc-h200|h200_normal_q|tc_h200_normal_short|4|5|42|160|01:30:00|h200"
)
tc_jobs_test=(
  "A100|dgx-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|00:30:00|a100"
  "A100|dgx-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|00:30:00|a100"
  "A100|dgx-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|00:30:00|a100"
  "A100|hpe-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|00:30:00|a100"
  "A100|hpe-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|00:30:00|a100"
  "A100|hpe-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|00:30:00|a100"
  "H200|tc-h200|h200_normal_q|tc_h200_normal_short|1|8|48|256|00:30:00|h200"
  "H200|tc-h200|h200_normal_q|tc_h200_normal_short|2|8|48|256|00:30:00|h200"
  "H200|tc-h200|h200_normal_q|tc_h200_normal_short|4|5|42|160|00:30:00|h200"
)

# Falcon training/testing (seeded and unseeded)
fal_train_unseed=(
  "V100|fal-v100|v100_normal_q|fal_v100_normal_base|1|2|24|64|01:30:00|v100"
  "V100|fal-v100|v100_normal_q|fal_v100_normal_base|2|2|24|64|01:30:00|v100"
  "V100|fal-v100|v100_normal_q|fal_v100_normal_base|4|2|24|64|01:30:00|v100"
  "A30|fal-a30|a30_normal_q|fal_a30_normal_short|1|4|48|128|01:30:00|a30"
  "A30|fal-a30|a30_normal_q|fal_a30_normal_short|2|4|48|128|01:30:00|a30"
  "A30|fal-a30|a30_normal_q|fal_a30_normal_short|4|4|48|128|01:30:00|a30"
  "L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|1|4|48|128|01:30:00|l40s"
  "L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|2|4|48|128|01:30:00|l40s"
  "L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|4|4|48|128|01:30:00|l40s"
)
fal_test_unseed=(
  "V100|fal-v100|v100_normal_q|fal_v100_normal_base|1|2|24|64|00:30:00|v100"
  "V100|fal-v100|v100_normal_q|fal_v100_normal_base|2|2|24|64|00:30:00|v100"
  "V100|fal-v100|v100_normal_q|fal_v100_normal_base|4|2|24|64|00:30:00|v100"
  "A30|fal-a30|a30_normal_q|fal_a30_normal_short|1|4|48|128|00:30:00|a30"
  "A30|fal-a30|a30_normal_q|fal_a30_normal_short|2|4|48|128|00:30:00|a30"
  "A30|fal-a30|a30_normal_q|fal_a30_normal_short|4|4|48|128|00:30:00|a30"
  "L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|1|4|48|128|00:30:00|l40s"
  "L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|2|4|48|128|00:30:00|l40s"
  "L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|4|4|48|128|00:30:00|l40s"
)
fal_train_seeded=()
tc_train_seeded=()
tc_test_seeded=()
fal_test_seeded=()
seeds=(301 302 303 304 305 306 307 308 309 310 311 312)
for seed in "${seeds[@]}"; do
  tc_train_seeded+=("A100|dgx-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|01:30:00|a100|${seed}")
  tc_train_seeded+=("A100|dgx-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|01:30:00|a100|${seed}")
  tc_train_seeded+=("A100|dgx-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|01:30:00|a100|${seed}")
  tc_train_seeded+=("A100|hpe-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|01:30:00|a100|${seed}")
  tc_train_seeded+=("A100|hpe-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|01:30:00|a100|${seed}")
  tc_train_seeded+=("A100|hpe-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|01:30:00|a100|${seed}")
  tc_train_seeded+=("H200|tc-h200|h200_normal_q|tc_h200_normal_short|1|8|48|256|01:30:00|h200|${seed}")
  tc_train_seeded+=("H200|tc-h200|h200_normal_q|tc_h200_normal_short|2|8|48|256|01:30:00|h200|${seed}")
  tc_train_seeded+=("H200|tc-h200|h200_normal_q|tc_h200_normal_short|4|5|42|160|01:30:00|h200|${seed}")

  tc_test_seeded+=("A100|dgx-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|00:30:00|a100|${seed}")
  tc_test_seeded+=("A100|dgx-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|00:30:00|a100|${seed}")
  tc_test_seeded+=("A100|dgx-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|00:30:00|a100|${seed}")
  tc_test_seeded+=("A100|hpe-A100|a100_normal_q|tc_a100_normal_short|1|8|48|256|00:30:00|a100|${seed}")
  tc_test_seeded+=("A100|hpe-A100|a100_normal_q|tc_a100_normal_short|2|8|48|256|00:30:00|a100|${seed}")
  tc_test_seeded+=("A100|hpe-A100|a100_normal_q|tc_a100_normal_short|4|8|48|256|00:30:00|a100|${seed}")
  tc_test_seeded+=("H200|tc-h200|h200_normal_q|tc_h200_normal_short|1|8|48|256|00:30:00|h200|${seed}")
  tc_test_seeded+=("H200|tc-h200|h200_normal_q|tc_h200_normal_short|2|8|48|256|00:30:00|h200|${seed}")
  tc_test_seeded+=("H200|tc-h200|h200_normal_q|tc_h200_normal_short|4|5|42|160|00:30:00|h200|${seed}")

  fal_train_seeded+=("V100|fal-v100|v100_normal_q|fal_v100_normal_base|1|2|24|64|01:30:00|v100|${seed}")
  fal_train_seeded+=("V100|fal-v100|v100_normal_q|fal_v100_normal_base|2|2|24|64|01:30:00|v100|${seed}")
  fal_train_seeded+=("V100|fal-v100|v100_normal_q|fal_v100_normal_base|4|2|24|64|01:30:00|v100|${seed}")
  fal_train_seeded+=("A30|fal-a30|a30_normal_q|fal_a30_normal_short|1|4|48|128|01:30:00|a30|${seed}")
  fal_train_seeded+=("A30|fal-a30|a30_normal_q|fal_a30_normal_short|2|4|48|128|01:30:00|a30|${seed}")
  fal_train_seeded+=("A30|fal-a30|a30_normal_q|fal_a30_normal_short|4|4|48|128|01:30:00|a30|${seed}")
  fal_train_seeded+=("L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|1|4|48|128|01:30:00|l40s|${seed}")
  fal_train_seeded+=("L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|2|4|48|128|01:30:00|l40s|${seed}")
  fal_train_seeded+=("L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|4|4|48|128|01:30:00|l40s|${seed}")

  fal_test_seeded+=("V100|fal-v100|v100_normal_q|fal_v100_normal_base|1|2|24|64|00:30:00|v100|${seed}")
  fal_test_seeded+=("V100|fal-v100|v100_normal_q|fal_v100_normal_base|2|2|24|64|00:30:00|v100|${seed}")
  fal_test_seeded+=("V100|fal-v100|v100_normal_q|fal_v100_normal_base|4|2|24|64|00:30:00|v100|${seed}")
  fal_test_seeded+=("A30|fal-a30|a30_normal_q|fal_a30_normal_short|1|4|48|128|00:30:00|a30|${seed}")
  fal_test_seeded+=("A30|fal-a30|a30_normal_q|fal_a30_normal_short|2|4|48|128|00:30:00|a30|${seed}")
  fal_test_seeded+=("A30|fal-a30|a30_normal_q|fal_a30_normal_short|4|4|48|128|00:30:00|a30|${seed}")
  fal_test_seeded+=("L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|1|4|48|128|00:30:00|l40s|${seed}")
  fal_test_seeded+=("L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|2|4|48|128|00:30:00|l40s|${seed}")
  fal_test_seeded+=("L40S|fal-l40s|l40s_normal_q|fal_l40s_normal_short|4|4|48|128|00:30:00|l40s|${seed}")
done

# Submit TC unseeded/seeded
if [[ "${TARGET}" == "ALL" || "${TARGET}" == "TC" ]]; then
  idx=0
  for row in "${tc_jobs_train[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token <<< "${row}"
    local_suffix="$(tc_suffix "${node_type}")"
    job_name="trn_HH_full_v2_${gpu}_${nodes}n${local_suffix}"
    node_type="${node_type}"
    submit_row "TC" "Tinkercliffs" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH}" "${TRAIN_CSV_TC}"
  done
  for row in "${tc_jobs_test[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token <<< "${row}"
    local_suffix="$(tc_suffix "${node_type}")"
    job_name="tst_HH_full_v2_${gpu}_${nodes}n${local_suffix}"
    node_type="${node_type}"
    submit_row "TC" "Tinkercliffs" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH}" "${TEST_CSV_TC}"
  done

  for row in "${tc_train_seeded[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token seed <<< "${row}"
    local_suffix="$(tc_suffix "${node_type}")"
    job_name="trn_HH_full_v2_${gpu}_${nodes}n${local_suffix}_s${seed}"
    node_type="${node_type}"
    submit_row "TC" "Tinkercliffs" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH},SEED=${seed}" "${SEED_CSV_TC}"
  done
  for row in "${tc_test_seeded[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token seed <<< "${row}"
    local_suffix="$(tc_suffix "${node_type}")"
    job_name="tst_HH_full_v2_${gpu}_${nodes}n${local_suffix}_s${seed}"
    node_type="${node_type}"
    submit_row "TC" "Tinkercliffs" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH},SEED=${seed}" "${SEED_TEST_CSV_TC}"
  done
fi

# Falcon unseeded
if [[ "${TARGET}" == "ALL" || "${TARGET}" == "FAL" ]]; then
  for row in "${fal_train_unseed[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token <<< "${row}"
    if ! falcon_gpu_allowed "${gpu}"; then
      continue
    fi
    job_name="trn_HH_full_v2_${gpu}_${nodes}n"
    node_type="${node_type}"
    submit_row "FAL" "Falcon" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH}" "${TRAIN_CSV_FAL}"
  done
  for row in "${fal_test_unseed[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token <<< "${row}"
    if ! falcon_gpu_allowed "${gpu}"; then
      continue
    fi
    job_name="tst_HH_full_v2_${gpu}_${nodes}n"
    node_type="${node_type}"
    submit_row "FAL" "Falcon" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH}" "${TEST_CSV_FAL}"
  done

  for row in "${fal_train_seeded[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token seed <<< "${row}"
    if ! falcon_gpu_allowed "${gpu}"; then
      continue
    fi
    job_name="trn_HH_full_v2_${gpu}_${nodes}n_s${seed}"
    node_type="${node_type}"
    submit_row "FAL" "Falcon" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH},SEED=${seed}" "${SEED_CSV_FAL}"
  done
  for row in "${fal_test_seeded[@]}"; do
    IFS='|' read -r gpu node_type part qos nodes gpn cpus mem wall token seed <<< "${row}"
    if ! falcon_gpu_allowed "${gpu}"; then
      continue
    fi
    job_name="tst_HH_full_v2_${gpu}_${nodes}n_s${seed}"
    node_type="${node_type}"
    submit_row "FAL" "Falcon" "${job_name}" "${part}" "${qos}" "${nodes}" "${gpn}" "${cpus}" "${mem}" "${wall}" "${token}" ",PARAMS_FILE=${PARAMS_FILE_HH},SEED=${seed}" "${SEED_TEST_CSV_FAL}"
  done
fi

echo "HH_full_v2 Falcon+TC unseeded/seeded submission complete (mode=${MODE}, target=${TARGET})."
