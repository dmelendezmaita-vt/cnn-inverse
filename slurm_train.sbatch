#!/usr/bin/env bash
#SBATCH --job-name=trn_l40s_4n
#SBATCH --account=neuro-collab
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/neuro-collab/data/runs/logs/%x-%j.out
#SBATCH --signal=B:USR1@300


set -euo pipefail

echo "$(date -Is) debug: DEBUG=${DEBUG-UNSET}"

export PS4='+ $(date -Is) ${BASH_SOURCE}:${LINENO}: '
if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

mkdir -p /projects/neuro-collab/data/runs/logs

# Job-name convention: <mode>_<gpu>_<nodes>n
# Examples:
#   sbatch --job-name=trn_a100_1n --nodes=1 slurm_train.sbatch
#   sbatch --job-name=trn_h200_4n --nodes=4 slurm_train.sbatch

# Defaults (override via: sbatch --export=ALL,TAR_PATH=...,CURR=...,DATA_PREFIX=...,PARAMS_FILE=...,SHARED_DATA_DIR=...)
TAR_PATH="${TAR_PATH:-/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar}"
CURR="${CURR:-0.1}"
DATA_PREFIX="${DATA_PREFIX:-publication_2020}"
PARAMS_FILE="${PARAMS_FILE:-pytorch/configs/params_dnn_tar.yaml}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
DLKIT="/projects/neuro-collab/code/dl-kit-main"
export PYTHONPATH="${DLKIT}:${REPO}:${PYTHONPATH:-}"

RUN_DIR="/projects/neuro-collab/data/runs/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/stdout.log") 2>&1

REPORT_INTERVAL="${REPORT_INTERVAL:-60}"
RUN_MODE="${RUN_MODE:-train}"
RUN_START_LOCAL="$(date '+%Y-%m-%d %H:%M:%S %Z')"
RUN_START_EPOCH="$(date +%s)"
REPORT_NOTIFY="${REPORT_NOTIFY:-1}"
REPORT_NOTIFY_EVERY="${REPORT_NOTIFY_EVERY:-12}"

# ----------------------------
# ntfy notifications
# ----------------------------
NTFY_TOPIC="dmelendezmaita"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"
NTFY_TAGS="${NTFY_TAGS:-slurm,train}"
NTFY_TIMEOUT="${NTFY_TIMEOUT:-10}"
NTFY_CONNECT_TIMEOUT="${NTFY_CONNECT_TIMEOUT:-5}"
NTFY_CRITICAL_RETRIES="${NTFY_CRITICAL_RETRIES:-3}"
NTFY_REPORT_RETRIES="${NTFY_REPORT_RETRIES:-1}"
NTFY_RETRY_DELAY_SECONDS="${NTFY_RETRY_DELAY_SECONDS:-2}"
NTFY_START_JITTER_MAX="${NTFY_START_JITTER_MAX:-10}"
NTFY_429_BASE_DELAY_SECONDS="${NTFY_429_BASE_DELAY_SECONDS:-5}"
NTFY_429_MAX_DELAY_SECONDS="${NTFY_429_MAX_DELAY_SECONDS:-120}"

if ! [[ "${REPORT_NOTIFY_EVERY}" =~ ^[0-9]+$ ]] || (( REPORT_NOTIFY_EVERY < 1 )); then
  REPORT_NOTIFY_EVERY=1
fi
if ! [[ "${NTFY_CRITICAL_RETRIES}" =~ ^[0-9]+$ ]] || (( NTFY_CRITICAL_RETRIES < 1 )); then
  NTFY_CRITICAL_RETRIES=1
fi
if ! [[ "${NTFY_REPORT_RETRIES}" =~ ^[0-9]+$ ]] || (( NTFY_REPORT_RETRIES < 1 )); then
  NTFY_REPORT_RETRIES=1
fi
if ! [[ "${NTFY_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || (( NTFY_RETRY_DELAY_SECONDS < 0 )); then
  NTFY_RETRY_DELAY_SECONDS=0
fi
if ! [[ "${NTFY_START_JITTER_MAX}" =~ ^[0-9]+$ ]] || (( NTFY_START_JITTER_MAX < 0 )); then
  NTFY_START_JITTER_MAX=0
fi
if ! [[ "${NTFY_429_BASE_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || (( NTFY_429_BASE_DELAY_SECONDS < 0 )); then
  NTFY_429_BASE_DELAY_SECONDS=0
fi
if ! [[ "${NTFY_429_MAX_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || (( NTFY_429_MAX_DELAY_SECONDS < 0 )); then
  NTFY_429_MAX_DELAY_SECONDS=0
fi

REPORT_NOTIFY_COUNTER=0

ntfy_send() {
  local title="$1"
  local body="$2"
  local phase="${3:-report}"
  local attempts attempt rc curl_out http_code sleep_s jitter_s

  attempts="${NTFY_REPORT_RETRIES}"
  if [[ "${phase}" != "report" ]]; then
    attempts="${NTFY_CRITICAL_RETRIES}"
  fi
  if ! [[ "${attempts}" =~ ^[0-9]+$ ]] || (( attempts < 1 )); then
    attempts=1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "$(date -Is) ntfy_send skipped: curl unavailable phase=${phase} title=${title}"
    return 0
  fi

  if [[ "${phase}" == "start" ]] && (( NTFY_START_JITTER_MAX > 0 )); then
    jitter_s=$(( RANDOM % (NTFY_START_JITTER_MAX + 1) ))
    if (( jitter_s > 0 )); then
      echo "$(date -Is) ntfy_send jitter phase=${phase} sleep=${jitter_s}s title=${title}"
      sleep "${jitter_s}"
    fi
  fi

  for (( attempt=1; attempt<=attempts; attempt++ )); do
    http_code=""
    if curl_out="$(
      curl -sS \
        --max-time "${NTFY_TIMEOUT}" \
        --connect-timeout "${NTFY_CONNECT_TIMEOUT}" \
        -o /dev/null \
        -w '%{http_code}' \
        -H "Title: ${title}" \
        -H "Priority: default" \
        -H "Tags: ${NTFY_TAGS}" \
        -d "${body}" \
        "${NTFY_URL}" 2>&1
    )"; then
      rc=0
      http_code="$(printf '%s' "${curl_out}" | tr -d '\n\r')"
      if [[ "${http_code}" =~ ^2[0-9][0-9]$ ]]; then
        echo "$(date -Is) ntfy_send ok phase=${phase} attempt=${attempt}/${attempts} http=${http_code} title=${title}"
        return 0
      fi
      echo "$(date -Is) ntfy_send failed phase=${phase} attempt=${attempt}/${attempts} rc=0 http=${http_code} title=${title}"
    else
      rc=$?
      echo "$(date -Is) ntfy_send failed phase=${phase} attempt=${attempt}/${attempts} rc=${rc} result=$(printf '%s' "${curl_out}" | tr '\n' ' ') title=${title}"
    fi

    if (( attempt < attempts )); then
      sleep_s="${NTFY_RETRY_DELAY_SECONDS}"
      if [[ "${http_code}" == "429" ]]; then
        sleep_s=$(( NTFY_429_BASE_DELAY_SECONDS * (2 ** (attempt - 1)) ))
        if (( NTFY_429_MAX_DELAY_SECONDS > 0 && sleep_s > NTFY_429_MAX_DELAY_SECONDS )); then
          sleep_s="${NTFY_429_MAX_DELAY_SECONDS}"
        fi
        if (( NTFY_START_JITTER_MAX > 0 )); then
          jitter_s=$(( RANDOM % (NTFY_START_JITTER_MAX + 1) ))
          sleep_s=$(( sleep_s + jitter_s ))
        fi
        echo "$(date -Is) ntfy_send backoff phase=${phase} attempt=${attempt}/${attempts} http=429 sleep=${sleep_s}s title=${title}"
      fi
      if (( sleep_s > 0 )); then
        sleep "${sleep_s}"
      fi
    fi
  done

  return 0
}

JOB_SUBMIT_LOCAL=""

format_seconds_hms() {
  local total="${1:-0}"
  local hh mm ss

  if ! [[ "${total}" =~ ^[0-9]+$ ]]; then
    echo "NA"
    return
  fi

  hh=$(( total / 3600 ))
  mm=$(( (total % 3600) / 60 ))
  ss=$(( total % 60 ))
  printf '%02d:%02d:%02d\n' "${hh}" "${mm}" "${ss}"
}

elapsed_since_start_hms() {
  local now elapsed
  now="$(date +%s)"
  if [[ "${RUN_START_EPOCH:-}" =~ ^[0-9]+$ ]] && [[ "${now}" =~ ^[0-9]+$ ]]; then
    elapsed=$(( now - RUN_START_EPOCH ))
    if (( elapsed < 0 )); then
      elapsed=0
    fi
    format_seconds_hms "${elapsed}"
  else
    echo "NA"
  fi
}

get_job_submit_local() {
  local submit

  if [[ -n "${JOB_SUBMIT_LOCAL:-}" ]]; then
    echo "${JOB_SUBMIT_LOCAL}"
    return
  fi

  submit=""
  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    submit="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | tr ' ' '\n' | awk -F= '/^SubmitTime=/{print $2; exit}')"
  fi

  if [[ -z "${submit}" || "${submit}" == "N/A" || "${submit}" == "Unknown" ]]; then
    if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
      submit="$(squeue -h -j "${SLURM_JOB_ID}" -o "%V" 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    fi
  fi

  if [[ -z "${submit}" || "${submit}" == "N/A" || "${submit}" == "Unknown" ]]; then
    submit="NA"
  fi

  JOB_SUBMIT_LOCAL="${submit}"
  echo "${JOB_SUBMIT_LOCAL}"
}

get_job_time_remaining() {
  local remaining
  remaining=""

  if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    remaining="$(squeue -h -j "${SLURM_JOB_ID}" -o "%L" 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  fi

  case "${remaining}" in
    ""|"N/A"|"NOT_SET"|"INVALID")
      echo "NA"
      ;;
    *)
      echo "${remaining}"
      ;;
  esac
}

get_elapsed_runtime_hms() {
  local elapsed
  elapsed=""

  if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    elapsed="$(squeue -h -j "${SLURM_JOB_ID}" -o "%M" 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  fi

  case "${elapsed}" in
    ""|"N/A"|"NOT_SET"|"INVALID")
      elapsed="$(elapsed_since_start_hms)"
      ;;
  esac

  if [[ -z "${elapsed}" ]]; then
    elapsed="NA"
  fi
  echo "${elapsed}"
}

send_start_notification() {
  local gpn cpn tar_name submit_local remaining_local body

  gpn="$(first_int_or_na "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}")"
  cpn="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-NA}}"
  tar_name="$(basename "${TAR_PATH}")"
  submit_local="$(get_job_submit_local)"
  remaining_local="$(get_job_time_remaining)"

  body="$(
    echo "[RUN START]"
    echo "Process: ${SLURM_JOB_NAME:-fhn_train}"
    echo "Job ID: ${SLURM_JOB_ID:-NA}"
    echo "Mode: ${RUN_MODE}"
    echo "Submit (local): ${submit_local}"
    echo "Start (local): ${RUN_START_LOCAL}"
    if [[ "${remaining_local}" != "NA" ]]; then
      echo "Time remaining: ${remaining_local}"
    fi
    echo "Nodes: ${SLURM_NNODES:-NA}"
    echo "GPUs / node: ${gpn}"
    echo "CPUs / node: ${cpn}"
    echo "Tar: ${tar_name}"
    echo "Run dir: ${RUN_DIR}"
  )"

  ntfy_send "Job ${SLURM_JOB_ID} start" "${body}" "start"
}

send_done_notification() {
  local exit_code="$1"
  local reason="${2:-}"
  local status end_local submit_local remaining_local elapsed_local body

  status="SUCCESS"
  if [[ "${exit_code}" != "0" ]]; then
    status="FAILED"
  fi

  end_local="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  submit_local="$(get_job_submit_local)"
  remaining_local="$(get_job_time_remaining)"
  elapsed_local="$(get_elapsed_runtime_hms)"
  body="$(
    echo "[RUN DONE]"
    echo "Process: ${SLURM_JOB_NAME:-fhn_train}"
    echo "Job ID: ${SLURM_JOB_ID:-NA}"
    echo "Mode: ${RUN_MODE}"
    echo "Submit (local): ${submit_local}"
    echo "Start (local): ${RUN_START_LOCAL}"
    echo "End (local): ${end_local}"
    echo "Elapsed run: ${elapsed_local}"
    if [[ "${remaining_local}" != "NA" ]]; then
      echo "Time remaining: ${remaining_local}"
    fi
    echo "Status: ${status}"
    echo "Exit: ${exit_code}"
    if [[ -n "${reason}" ]]; then
      echo "Reason: ${reason}"
    fi
    echo "Run dir: ${RUN_DIR}"
  )"

  ntfy_send "Job ${SLURM_JOB_ID} done (${status})" "${body}" "done"
}

first_int_or_na() {
  local raw="${1:-}"
  local out

  out="$(printf '%s' "${raw}" | grep -Eo '[0-9]+' | head -n 1 || true)"
  if [[ -n "${out}" ]]; then
    printf '%s\n' "${out}"
  else
    printf 'NA\n'
  fi
}

parse_mem_to_gib() {
  local raw="${1:-}"
  local value unit

  raw="${raw//[[:space:]]/}"
  raw="${raw%[cn]}"

  if [[ -z "${raw}" || "${raw}" == "N/A" || "${raw}" == "0" ]]; then
    echo "NA"
    return
  fi

  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    # SLURM_MEM_PER_NODE is typically in MiB when unitless.
    echo $(( (raw + 1023) / 1024 ))
    return
  fi

  if [[ "${raw}" =~ ^([0-9]+)([KMGT])$ ]]; then
    value="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    case "${unit}" in
      K) echo $(( (value + 1048575) / 1048576 )) ;;
      M) echo $(( (value + 1023) / 1024 )) ;;
      G) echo "${value}" ;;
      T) echo $(( value * 1024 )) ;;
      *) echo "NA" ;;
    esac
    return
  fi

  echo "NA"
}

detect_requested_mem_gib() {
  local raw gib info min_node min_cpu cpus

  raw="${SLURM_MEM_PER_NODE:-}"
  gib="$(parse_mem_to_gib "${raw}")"
  if [[ "${gib}" != "NA" ]]; then
    echo "${gib}"
    return
  fi

  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    info="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | tr ' ' '\n' || true)"

    min_node="$(echo "${info}" | awk -F= '/^MinMemoryNode=/{print $2; exit}')"
    gib="$(parse_mem_to_gib "${min_node}")"
    if [[ "${gib}" != "NA" ]]; then
      echo "${gib}"
      return
    fi

    min_cpu="$(echo "${info}" | awk -F= '/^MinMemoryCPU=/{print $2; exit}')"
    gib="$(parse_mem_to_gib "${min_cpu}")"
    if [[ "${gib}" != "NA" ]]; then
      cpus="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
      if [[ "${cpus}" =~ ^[0-9]+$ ]] && (( cpus > 0 )); then
        echo $(( gib * cpus ))
        return
      fi
    fi
  fi

  echo "NA"
}

print_local_metrics_record() {
  local user nice system idle iowait irq softirq steal
  local total1 idle1 total2 idle2 dt di cpu_pct
  local mem_total_kb mem_avail_kb mem_used_kb ram_used_gib
  local gpu_count gpu_util_sum gpu_utils
  local vram_used_mib vram_total_mib gpu_util_avg vram_used_gib vram_total_gib
  local util used total idx
  local tmp_base monitor_file util_samples_sum util_samples_cnt
  local -a peak_util peak_used peak_total seen

  read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
  total1=$((user + nice + system + idle + iowait + irq + softirq + steal))
  idle1=$((idle + iowait))
  sleep 1
  read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
  total2=$((user + nice + system + idle + iowait + irq + softirq + steal))
  idle2=$((idle + iowait))
  dt=$((total2 - total1))
  di=$((idle2 - idle1))
  if (( dt > 0 )); then
    cpu_pct=$(( (100 * (dt - di) + dt / 2) / dt ))
  else
    cpu_pct=0
  fi

  mem_total_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  mem_avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  mem_used_kb=$((mem_total_kb - mem_avail_kb))
  ram_used_gib=$(( (mem_used_kb + 1048575) / 1048576 ))

  gpu_count=0
  gpu_util_sum=0
  gpu_utils="NA"
  vram_used_mib=0
  vram_total_mib=0
  gpu_util_avg=0
  util_samples_sum=0
  util_samples_cnt=0

  tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  monitor_file="${GPU_MONITOR_FILE:-${tmp_base}/${SLURM_JOB_ID}/metrics/gpu_metrics.csv}"

  if [[ -s "${monitor_file}" ]]; then
    while IFS=',' read -r idx util used total _; do
      idx="${idx//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      used="${used//[[:space:]]/}"
      total="${total//[[:space:]]/}"
      [[ "${idx}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ && "${used}" =~ ^[0-9]+$ && "${total}" =~ ^[0-9]+$ ]] || continue

      seen[idx]=1
      if [[ -z "${peak_util[idx]:-}" || "${util}" -gt "${peak_util[idx]}" ]]; then
        peak_util[idx]="${util}"
      fi
      if [[ -z "${peak_used[idx]:-}" || "${used}" -gt "${peak_used[idx]}" ]]; then
        peak_used[idx]="${used}"
      fi
      if [[ -z "${peak_total[idx]:-}" || "${total}" -gt "${peak_total[idx]}" ]]; then
        peak_total[idx]="${total}"
      fi

      util_samples_sum=$((util_samples_sum + util))
      util_samples_cnt=$((util_samples_cnt + 1))
    done < <(tail -n 4096 "${monitor_file}" 2>/dev/null || true)
  fi

  if (( util_samples_cnt > 0 )); then
    gpu_utils=""
    for idx in $(printf '%s\n' "${!seen[@]}" | sort -n); do
      [[ -n "${idx}" ]] || continue
      gpu_count=$((gpu_count + 1))
      vram_used_mib=$((vram_used_mib + peak_used[idx]))
      vram_total_mib=$((vram_total_mib + peak_total[idx]))
      if [[ -z "${gpu_utils}" ]]; then
        gpu_utils="${peak_util[idx]}%"
      else
        gpu_utils="${gpu_utils},${peak_util[idx]}%"
      fi
    done
    gpu_util_avg=$(( (util_samples_sum + util_samples_cnt / 2) / util_samples_cnt ))
  elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_utils=""
    while IFS=',' read -r idx util used total _; do
      util="${util//[[:space:]]/}"
      used="${used//[[:space:]]/}"
      total="${total//[[:space:]]/}"
      [[ "${util}" =~ ^[0-9]+$ && "${used}" =~ ^[0-9]+$ && "${total}" =~ ^[0-9]+$ ]] || continue

      gpu_count=$((gpu_count + 1))
      gpu_util_sum=$((gpu_util_sum + util))
      vram_used_mib=$((vram_used_mib + used))
      vram_total_mib=$((vram_total_mib + total))

      if [[ -z "${gpu_utils}" ]]; then
        gpu_utils="${util}%"
      else
        gpu_utils="${gpu_utils},${util}%"
      fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)

    if (( gpu_count > 0 )); then
      gpu_util_avg=$(( (gpu_util_sum + gpu_count / 2) / gpu_count ))
    fi
  fi

  if (( gpu_count == 0 )); then
    gpu_utils="NA"
  fi

  vram_used_gib=$(( (vram_used_mib + 512) / 1024 ))
  vram_total_gib=$(( (vram_total_mib + 512) / 1024 ))

  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$(hostname -s)" \
    "${cpu_pct}" \
    "${ram_used_gib}" \
    "${gpu_util_avg}" \
    "${gpu_utils}" \
    "${vram_used_gib}" \
    "${vram_total_gib}" \
    "${gpu_count}"
}

collect_metrics_for_node() {
  local node="$1"
  local payload out
  local local_short local_full

  local_short="$(hostname -s)"
  local_full="$(hostname)"
  if [[ "${node}" == "${local_short}" || "${node}" == "${local_full}" ]]; then
    print_local_metrics_record
    return 0
  fi

  payload="$(declare -f print_local_metrics_record); print_local_metrics_record"
  out="$(timeout 20s srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --nodelist="${node}" --exact bash -lc "${payload}" 2>/dev/null || true)"

  if [[ -z "${out}" ]]; then
    out="$(timeout 20s srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --nodelist="${node}" --exact bash -lc "${payload}" 2>/dev/null || true)"
  fi

  if [[ -n "${out}" ]]; then
    echo "${out}" | tail -n 1
  else
    printf '%s|NA|NA|NA|NA|NA|NA|0\n' "${node}"
  fi
}

util_report() {
  local run_dir="$1"
  local forced_status="${2:-}"
  local stamp body process_name mode job_id job_status error_line
  local node_expr node_count requested_mem_gib cpus_per_node gpus_per_node
  local rec node cpu ram gpu_avg gpu_list vram_used vram_total gpu_count
  local cpu_sum cpu_cnt ram_used_sum cluster_ram_total cluster_ram_pct
  local gpu_util_weighted gpu_cnt_total vram_used_sum vram_total_sum
  local cluster_cpu_avg cluster_gpu_avg cluster_vram_pct cluster_vram
  local node_ram_pct node_vram_pct per_node_lines submit_local remaining_local elapsed_local
  local is_terminal_report
  local -a nodes

  stamp="$(date -Is)"
  process_name="${SLURM_JOB_NAME:-fhn_train}"
  mode="${RUN_MODE}"
  job_id="${SLURM_JOB_ID:-NA}"
  submit_local="$(get_job_submit_local)"
  remaining_local="$(get_job_time_remaining)"
  elapsed_local="$(get_elapsed_runtime_hms)"
  node_expr="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
  cpus_per_node="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-NA}}"
  requested_mem_gib="$(detect_requested_mem_gib)"

  if [[ -n "${node_expr}" ]] && command -v scontrol >/dev/null 2>&1; then
    mapfile -t nodes < <(scontrol show hostnames "${node_expr}" 2>/dev/null || true)
  fi
  if [[ ${#nodes[@]} -eq 0 ]]; then
    nodes=("$(hostname -s)")
  fi
  node_count="${#nodes[@]}"

  gpus_per_node="$(first_int_or_na "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}")"

  error_line=""
  if [[ "${forced_status}" != "COMPLETED" ]] && [[ -f "${run_dir}/stdout.log" ]]; then
    # Prefer explicit fatal markers emitted by this script or Python tracebacks.
    error_line="$(grep -E 'ERROR line=|ERROR: missing or empty output files|Traceback \(most recent call last\):|[[:alnum:]_]+Error:' "${run_dir}/stdout.log" | tail -n 1 || true)"
    # Fallback for failed reports when no explicit marker was captured.
    if [[ -z "${error_line}" && "${forced_status}" == "FAILED" ]]; then
      error_line="$(grep -E 'ERROR|Traceback|RuntimeError|CUDA out of memory|NCCL ERROR|NCCL WARN' "${run_dir}/stdout.log" | tail -n 1 || true)"
    fi
  fi
  if [[ -n "${error_line}" ]]; then
    error_line="${error_line//$'\n'/ }"
    if (( ${#error_line} > 220 )); then
      error_line="${error_line:0:217}..."
    fi
  fi
  if [[ -n "${forced_status}" ]]; then
    job_status="${forced_status}"
  elif [[ -n "${error_line}" ]]; then
    job_status="FAILED"
  else
    job_status="RUNNING"
  fi
  case "${job_status}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY)
      is_terminal_report=1
      ;;
    *)
      is_terminal_report=0
      ;;
  esac

  cpu_sum=0
  cpu_cnt=0
  ram_used_sum=0
  gpu_util_weighted=0
  gpu_cnt_total=0
  vram_used_sum=0
  vram_total_sum=0
  per_node_lines=""

  for node in "${nodes[@]}"; do
    rec="$(collect_metrics_for_node "${node}")"
    IFS='|' read -r node cpu ram gpu_avg gpu_list vram_used vram_total gpu_count <<< "${rec}"

    if [[ "${cpu}" =~ ^[0-9]+$ ]]; then
      cpu_sum=$((cpu_sum + cpu))
      cpu_cnt=$((cpu_cnt + 1))
    fi

    if [[ "${ram}" =~ ^[0-9]+$ ]]; then
      ram_used_sum=$((ram_used_sum + ram))
    fi

    if [[ "${gpu_avg}" =~ ^[0-9]+$ && "${gpu_count}" =~ ^[0-9]+$ ]]; then
      gpu_util_weighted=$((gpu_util_weighted + (gpu_avg * gpu_count)))
      gpu_cnt_total=$((gpu_cnt_total + gpu_count))
    fi

    if [[ "${vram_used}" =~ ^[0-9]+$ ]]; then
      vram_used_sum=$((vram_used_sum + vram_used))
    fi
    if [[ "${vram_total}" =~ ^[0-9]+$ ]]; then
      vram_total_sum=$((vram_total_sum + vram_total))
    fi

    if [[ "${requested_mem_gib}" =~ ^[0-9]+$ && "${ram}" =~ ^[0-9]+$ && "${requested_mem_gib}" -gt 0 ]]; then
      node_ram_pct=$(( (100 * ram + requested_mem_gib / 2) / requested_mem_gib ))
    else
      node_ram_pct="NA"
    fi

    if [[ "${vram_total}" =~ ^[0-9]+$ && "${vram_used}" =~ ^[0-9]+$ && "${vram_total}" -gt 0 ]]; then
      node_vram_pct=$(( (100 * vram_used + vram_total / 2) / vram_total ))
    else
      node_vram_pct="NA"
    fi

    if [[ -z "${per_node_lines}" ]]; then
      per_node_lines="${node}: CPU ${cpu}% | RAM ${ram}/${requested_mem_gib} GiB (${node_ram_pct}%) | GPU util avg ${gpu_avg}% [${gpu_list}] | VRAM ${vram_used}/${vram_total} GiB (${node_vram_pct}%)"
    else
      per_node_lines+=$'\n'"${node}: CPU ${cpu}% | RAM ${ram}/${requested_mem_gib} GiB (${node_ram_pct}%) | GPU util avg ${gpu_avg}% [${gpu_list}] | VRAM ${vram_used}/${vram_total} GiB (${node_vram_pct}%)"
    fi
  done

  if (( cpu_cnt > 0 )); then
    cluster_cpu_avg=$(( (cpu_sum + cpu_cnt / 2) / cpu_cnt ))
  else
    cluster_cpu_avg="NA"
  fi

  if [[ "${requested_mem_gib}" =~ ^[0-9]+$ ]]; then
    cluster_ram_total=$((requested_mem_gib * node_count))
    if (( cluster_ram_total > 0 )); then
      cluster_ram_pct=$(( (100 * ram_used_sum + cluster_ram_total / 2) / cluster_ram_total ))
    else
      cluster_ram_pct="NA"
    fi
  else
    cluster_ram_total="NA"
    cluster_ram_pct="NA"
  fi

  if (( gpu_cnt_total > 0 )); then
    cluster_gpu_avg=$(( (gpu_util_weighted + gpu_cnt_total / 2) / gpu_cnt_total ))
  else
    cluster_gpu_avg="NA"
  fi

  if (( vram_total_sum > 0 )); then
    cluster_vram_pct=$(( (100 * vram_used_sum + vram_total_sum / 2) / vram_total_sum ))
    cluster_vram="${vram_used_sum}/${vram_total_sum} GiB (${cluster_vram_pct}%)"
  else
    cluster_vram="NA"
  fi

  if [[ "${gpus_per_node}" == "NA" && ${node_count} -gt 0 && ${gpu_cnt_total} -gt 0 ]]; then
    gpus_per_node=$((gpu_cnt_total / node_count))
  fi

  body="$(
    echo "[RUN STATUS]"
    echo "Process: ${process_name}"
    echo "Job ID: ${job_id}"
    echo "Mode: ${mode}"
    echo "Submit (local): ${submit_local}"
    echo "Start (local): ${RUN_START_LOCAL}"
    echo "Elapsed run: ${elapsed_local}"
    if [[ "${remaining_local}" != "NA" ]]; then
      echo "Time remaining: ${remaining_local}"
    fi
    echo "Nodes: ${node_count}"
    echo "GPUs / node: ${gpus_per_node}"
    echo "CPUs / node: ${cpus_per_node}"
    echo "Status: ${job_status}"
    echo
    echo "[CLUSTER CURRENT]"
    echo "CPU avg: ${cluster_cpu_avg}%"
    echo "RAM: ${ram_used_sum}/${cluster_ram_total} GiB (${cluster_ram_pct}%)"
    echo "GPU util avg: ${cluster_gpu_avg}%"
    echo "VRAM avg: ${cluster_vram}"
    echo
    echo "[PER NODE]"
    echo "${per_node_lines}"
    echo
    echo "[ERROR]"
    if [[ -n "${error_line}" ]]; then
      echo "- summary: ${error_line}"
    else
      echo "- none"
    fi
  )"
  echo "${body}" > "${run_dir}/report_${stamp}.txt" || true
  if [[ "${REPORT_NOTIFY}" == "1" ]]; then
    REPORT_NOTIFY_COUNTER=$((REPORT_NOTIFY_COUNTER + 1))
    if (( is_terminal_report == 1 )); then
      echo "$(date -Is) ntfy_send skipped phase=report terminal_status=${job_status}"
    elif (( REPORT_NOTIFY_COUNTER % REPORT_NOTIFY_EVERY == 0 )); then
      ntfy_send "Job ${job_id} report (${job_status})" "${body}" "report"
    else
      echo "$(date -Is) ntfy_send skipped phase=report counter=${REPORT_NOTIFY_COUNTER} every=${REPORT_NOTIFY_EVERY}"
    fi
  fi
}

handle_usr1() {
  echo "$(date -Is) received SIGUSR1, writing report"
  util_report "${RUN_DIR}" || true
}

trap handle_usr1 USR1

REPORT_PID=""

start_periodic_reports() {
  local run_dir="$1"
  local interval="$2"

  (
    set +x
    while true; do
      sleep "$interval"
      util_report "$run_dir" || true
    done
  ) &

  REPORT_PID=$!
  echo "$(date -Is) periodic reports started pid=${REPORT_PID} interval=${interval}"
}

cleanup() {
  if [[ -n "${REPORT_PID:-}" ]]; then
    kill "${REPORT_PID}" 2>/dev/null || true
    wait "${REPORT_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

validate_run_outputs() {
  local run_dir="$1"
  local -a required_files
  local -a missing_files
  local missing_list_file
  local f

  required_files=(
    "params.yaml"
    "split_indices.npz"
    "net.txt"
    "metrics_summary.json"
    "metrics_summary.csv"
    "predictions.npz"
    "predictions_schema.json"
  )
  missing_files=()

  for f in "${required_files[@]}"; do
    if [[ ! -s "${run_dir}/${f}" ]]; then
      missing_files+=("${f}")
    fi
  done

  if (( ${#missing_files[@]} > 0 )); then
    missing_list_file="${run_dir}/missing_outputs.txt"
    printf '%s\n' "${missing_files[@]}" > "${missing_list_file}" || true
    echo "$(date -Is) ERROR: missing or empty output files in ${run_dir} (${#missing_files[@]} missing; see ${missing_list_file})"
    return 1
  fi

  python - "${run_dir}/metrics_summary.json" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

for key in ("mse", "mae", "r2"):
    if key not in payload:
        raise SystemExit(f"missing metrics key: {key}")

def walk(value):
    if isinstance(value, dict):
        for sub in value.values():
            walk(sub)
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            walk(sub)
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SystemExit("found non-finite metric value")

walk(payload)
print("metrics_summary.json validation OK")
PY
}

send_start_notification

echo "$(date -Is) multi-node launch prep: TAR_PATH=${TAR_PATH} TMPDIR=${TMPDIR} JOBID=${SLURM_JOB_ID} NNODES=${SLURM_NNODES}"
echo "$(date -Is) step: after launch prep, before rendezvous setup"

echo "$(date -Is) step: cd repo"
cd "${REPO}"
echo "$(date -Is) step: cd repo done"

echo "$(date -Is) step: resolve master"
NODELIST="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
echo "$(date -Is) step: NODELIST=${NODELIST}"

echo "$(date -Is) step: scontrol hostnames begin"
HOSTS="$(timeout 10s scontrol show hostnames "${NODELIST}" || true)"
echo "$(date -Is) step: scontrol hostnames rc=$? lines=$(echo "${HOSTS}" | wc -l)"

export MASTER_ADDR="$(echo "${HOSTS}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 20000) ))}"

echo "$(date -Is) rendezvous: MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "$(date -Is) ERROR: MASTER_ADDR empty, cannot continue"
  exit 2
fi

echo "$(date -Is) step: rendezvous setup done"

REPORT_INTERVAL="${REPORT_INTERVAL:-60}"

start_periodic_reports "${RUN_DIR}" "${REPORT_INTERVAL}"
echo "$(date -Is) step: periodic reports setup returned pid=${REPORT_PID:-EMPTY}"

cd "${REPO}"

NODELIST="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
export MASTER_ADDR="$(scontrol show hostnames "$NODELIST" | head -n 1)"

export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 20000) ))}"

echo "$(date -Is) rendezvous: MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODELIST=${NODELIST}"
echo "$(date -Is) scheduler: partition=${SLURM_JOB_PARTITION:-NA} qos=${SLURM_JOB_QOS:-NA} constraints=${SLURM_JOB_CONSTRAINTS:-NONE}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "$(date -Is) ERROR: could not determine MASTER_ADDR from NODELIST=${NODELIST}"
  exit 2
fi
if [[ -z "${NPROC:-}" ]]; then
  NPROC="$(first_int_or_na "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}")"
  if [[ "${NPROC}" == "NA" ]]; then
    NPROC=4
  fi
fi
export NPROC
export TAR_PATH CURR DATA_PREFIX PARAMS_FILE SHARED_DATA_DIR
DATA_ACCESS_MODE="${DATA_ACCESS_MODE:-copy_to_node}"
case "${DATA_ACCESS_MODE}" in
  copy_to_node) ;;
  direct|direct_tar) DATA_ACCESS_MODE="direct_tar" ;;
  *)
    echo "$(date -Is) ERROR: invalid DATA_ACCESS_MODE='${DATA_ACCESS_MODE}' (expected copy_to_node|direct_tar)"
    exit 2
    ;;
esac
export DATA_ACCESS_MODE
echo "$(date -Is) data access mode: ${DATA_ACCESS_MODE}"
export PYTHONUNBUFFERED=1
export TORCH_DIST_TIMEOUT_SECONDS="${TORCH_DIST_TIMEOUT_SECONDS:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-${NCCL_ASYNC_ERROR_HANDLING:-1}}"
unset NCCL_ASYNC_ERROR_HANDLING || true
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,veth}"

echo "$(date -Is) launching: nnodes=${SLURM_NNODES} nproc_per_node=${NPROC} gpus_per_node=${SLURM_GPUS_ON_NODE:-NA} master=${MASTER_ADDR}:${MASTER_PORT}"

echo "$(date -Is) step: about to start srun torchrun block"
set +e
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    on_err() {
      local ec=$?
      echo "$(date -Is) ERROR line=$LINENO cmd=$BASH_COMMAND exit=$ec"
      env | grep -E "SLURM|CUDA|NCCL|MASTER|TMPDIR" || true
      exit "$ec"
    }
    trap on_err ERR

    if [[ "${DEBUG:-0}" == "1" ]]; then
      set -x
    fi

    # Ensure the same Python environment on every node
    module load Miniforge3
    source activate /projects/neuro-collab/conda/neuro-collab-env

    REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
    DLKIT="/projects/neuro-collab/code/dl-kit-main"
    export PYTHONPATH="${DLKIT}:${REPO}:${PYTHONPATH:-}"

    TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
    WORK="${TMP_BASE}/${SLURM_JOB_ID}"
    TMPDIR="${WORK}/tmp"
    export TMPDIR
    mkdir -p "${WORK}" "${TMPDIR}"

    STAGE_MIN_FREE_BYTES="${STAGE_MIN_FREE_BYTES:-21474836480}"
    STAGE_BUFFER_BYTES="${STAGE_BUFFER_BYTES:-21474836480}"
    STAGE_MIN_FREE_INODES="${STAGE_MIN_FREE_INODES:-10000}"
    STAGE_SYNC_TIMEOUT="${STAGE_SYNC_TIMEOUT:-900}"
    STATUS_DIR="/projects/neuro-collab/data/runs/${SLURM_JOB_ID}/stage_status"
    mkdir -p "${STATUS_DIR}"
    NODE_STATUS_FILE="${STATUS_DIR}/node_${SLURM_NODEID}.mode"

    TAR_SIZE_BYTES="$(stat -c%s "${TAR_PATH}" 2>/dev/null || echo 0)"
    if [[ "${TAR_SIZE_BYTES}" =~ ^[0-9]+$ ]] && [[ "${STAGE_BUFFER_BYTES}" =~ ^[0-9]+$ ]] && [[ "${STAGE_MIN_FREE_BYTES}" =~ ^[0-9]+$ ]] && (( TAR_SIZE_BYTES > 0 )); then
      # copy_to_node needs room for both the tar copy and extracted data, plus a safety buffer.
      REQUIRED_BYTES=$(( TAR_SIZE_BYTES * 2 + STAGE_BUFFER_BYTES ))
      if (( REQUIRED_BYTES > STAGE_MIN_FREE_BYTES )); then
        STAGE_MIN_FREE_BYTES="${REQUIRED_BYTES}"
      fi
    fi

    get_free_bytes() {
      df -PB1 "$1" | awk "NR==2 {print \$4}"
    }
    get_free_inodes() {
      df -Pi "$1" | awk "NR==2 {print \$4}"
    }

    NODE_DATA_MODE=""
    DATA_DIR=""
    if [[ "${DATA_ACCESS_MODE}" == "copy_to_node" ]]; then
      echo "$(date -Is) [node ${SLURM_NODEID}] staging/extract start: TMP_BASE=${TMP_BASE} TAR_PATH=${TAR_PATH}"
      mkdir -p "${WORK}/ds"
      FREE_BYTES="$(get_free_bytes "${WORK}" || echo 0)"
      FREE_INODES="$(get_free_inodes "${WORK}" || echo 0)"
      echo "$(date -Is) [node ${SLURM_NODEID}] local space check: free_bytes=${FREE_BYTES} free_inodes=${FREE_INODES}"
      if [[ "${FREE_BYTES}" -lt "${STAGE_MIN_FREE_BYTES}" || "${FREE_INODES}" -lt "${STAGE_MIN_FREE_INODES}" ]]; then
        echo "$(date -Is) [node ${SLURM_NODEID}] WARN: insufficient local scratch; switching to direct mode"
        NODE_DATA_MODE="direct_tar"
      else
        STAGE_OK=1
        cp -f "${TAR_PATH}" "${WORK}/dataset.tar" || STAGE_OK=0
        if [[ "${STAGE_OK}" -eq 1 ]]; then
          tar -xf "${WORK}/dataset.tar" -C "${WORK}/ds" || STAGE_OK=0
        fi
        if [[ "${STAGE_OK}" -eq 1 ]]; then
          TOPDIR="$(find "${WORK}/ds" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
          if [[ -n "${TOPDIR}" ]]; then
            DATA_DIR="${TOPDIR}"
          else
            DATA_DIR="${WORK}/ds"
          fi
          NODE_DATA_MODE="copy_to_node"
          echo "$(date -Is) [node ${SLURM_NODEID}] staging/extract done: DATA_DIR=${DATA_DIR}"
        else
          echo "$(date -Is) [node ${SLURM_NODEID}] WARN: staging/extract failed; switching to direct mode"
          rm -rf "${WORK}/dataset.tar" "${WORK}/ds" || true
          NODE_DATA_MODE="direct_tar"
        fi
      fi
    else
      NODE_DATA_MODE="direct_tar"
    fi

    if [[ "${NODE_DATA_MODE}" == "direct_tar" ]]; then
      if [[ -n "${SHARED_DATA_DIR:-}" && -d "${SHARED_DATA_DIR}" ]]; then
        DATA_DIR="${SHARED_DATA_DIR}"
      else
        DATA_DIR="${TAR_PATH}"
      fi
    fi

    echo "${NODE_DATA_MODE}" > "${NODE_STATUS_FILE}"

    EXPECTED_NODES="${SLURM_NNODES:-1}"
    SECONDS_WAITED=0
    while true; do
      READY_COUNT="$(find "${STATUS_DIR}" -maxdepth 1 -name "node_*.mode" | wc -l)"
      if [[ "${READY_COUNT}" -ge "${EXPECTED_NODES}" ]]; then
        break
      fi
      if [[ "${SECONDS_WAITED}" -ge "${STAGE_SYNC_TIMEOUT}" ]]; then
        echo "$(date -Is) [node ${SLURM_NODEID}] WARN: stage status sync timeout after ${SECONDS_WAITED}s"
        break
      fi
      sleep 1
      SECONDS_WAITED=$((SECONDS_WAITED + 1))
    done

    GLOBAL_DATA_MODE="copy_to_node"
    if grep -q "direct_tar" "${STATUS_DIR}"/node_*.mode 2>/dev/null; then
      GLOBAL_DATA_MODE="direct_tar"
    fi

    if [[ "${GLOBAL_DATA_MODE}" == "direct_tar" ]]; then
      if [[ "${NODE_DATA_MODE}" == "copy_to_node" ]]; then
        rm -rf "${WORK}/dataset.tar" "${WORK}/ds" || true
      fi
      NODE_DATA_MODE="direct_tar"
      if [[ -n "${SHARED_DATA_DIR:-}" && -d "${SHARED_DATA_DIR}" ]]; then
        DATA_DIR="${SHARED_DATA_DIR}"
      else
        DATA_DIR="${TAR_PATH}"
      fi
      echo "$(date -Is) [node ${SLURM_NODEID}] unified data mode=direct_tar DATA_DIR=${DATA_DIR}"
    else
      echo "$(date -Is) [node ${SLURM_NODEID}] unified data mode=copy_to_node DATA_DIR=${DATA_DIR}"
    fi

    cd "${REPO}"

    export TORCH_DIST_TIMEOUT_SECONDS="${TORCH_DIST_TIMEOUT_SECONDS:-1800}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-${NCCL_ASYNC_ERROR_HANDLING:-1}}"
    unset NCCL_ASYNC_ERROR_HANDLING || true
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,veth}"

    export OMP_NUM_THREADS=$(( (${SLURM_CPUS_PER_TASK:-1}) / (${NPROC}) ))
    export OMP_NUM_THREADS=$(( OMP_NUM_THREADS<1 ? 1 : OMP_NUM_THREADS ))

    MON_DIR="${WORK}/metrics"
    mkdir -p "${MON_DIR}"
    GPU_MONITOR_FILE="${MON_DIR}/gpu_metrics.csv"
    GPU_MONITOR_PID=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      : > "${GPU_MONITOR_FILE}"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 1 >> "${GPU_MONITOR_FILE}" 2>/dev/null &
      GPU_MONITOR_PID=$!
      echo "$(date -Is) [node ${SLURM_NODEID}] gpu monitor started pid=${GPU_MONITOR_PID} file=${GPU_MONITOR_FILE}"
    fi
    cleanup_gpu_monitor() {
      if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
      fi
    }
    trap cleanup_gpu_monitor EXIT

    torchrun \
      --nnodes="${SLURM_NNODES}" \
      --node_rank="${SLURM_NODEID}" \
      --nproc_per_node="${NPROC}" \
      --rdzv_backend=c10d \
      --rdzv_id="${SLURM_JOB_ID}" \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      pytorch/run_dnn.py \
        --params "${PARAMS_FILE}" \
        --mode train_eval \
        --save_dir_base /projects/neuro-collab/data/runs \
        --data_dir "${DATA_DIR}" \
        --data_prefix "${DATA_PREFIX}" \
        --curr "${CURR}" \
        --save_predictions test
  '
step_rc=$?
set -e
echo "$(date -Is) step: srun torchrun block returned with rc=${step_rc}"

if (( step_rc != 0 )); then
  util_report "${RUN_DIR}" "FAILED" || true
  send_done_notification "${step_rc}" "step_failed"
  exit "${step_rc}"
fi

if ! validate_run_outputs "${RUN_DIR}"; then
  util_report "${RUN_DIR}" "FAILED" || true
  send_done_notification "3" "artifact_validation_failed"
  exit 3
fi

util_report "${RUN_DIR}" "COMPLETED"
send_done_notification "0"
