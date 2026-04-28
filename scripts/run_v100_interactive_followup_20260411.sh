#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
PHASE_SET="phaseA"
DRY_RUN=0
REBUILD_MATRIX=0
LIMIT=0
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase-set)
      PHASE_SET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --rebuild-matrix)
      REBUILD_MATRIX=1
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}" ]]; then
  echo "This wrapper should be run inside the live interactive allocation." >&2
  echo "Missing SLURM_JOB_ID or SLURM_JOB_NODELIST." >&2
  exit 2
fi

cd "${REPO}"

if [[ "${REBUILD_MATRIX}" == "1" || ! -f "${REPO}/data/important_notes/optimization_track_20260411_v100_interactive/tables/v100_interactive_followup_matrix_20260411_v100_interactive.csv" ]]; then
  python scripts/build_v100_interactive_followup_matrix_20260411.py
fi

LAUNCH_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  LAUNCH_ARGS+=(--dry-run)
fi
if [[ "${LIMIT}" != "0" ]]; then
  LAUNCH_ARGS+=(--limit "${LIMIT}")
fi
LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")

run_phase() {
  local phase="$1"
  echo "$(date -Is) running follow-up phase ${phase}"
  python scripts/launch_v100_interactive_followup_20260411.py --phase "${phase}" "${LAUNCH_ARGS[@]}"
}

case "${PHASE_SET}" in
  phaseA)
    run_phase phaseA_baseline_latency
    run_phase phaseA_throughput_geometry
    ;;
  phaseB)
    bash scripts/prepare_v100_interactive_shared_data_20260411.sh
    run_phase phaseB_codepath_screen
    ;;
  all)
    run_phase phaseA_baseline_latency
    run_phase phaseA_throughput_geometry
    bash scripts/prepare_v100_interactive_shared_data_20260411.sh
    run_phase phaseB_codepath_screen
    ;;
  *)
    echo "Unknown --phase-set ${PHASE_SET}. Expected phaseA|phaseB|all." >&2
    exit 2
    ;;
esac

python scripts/summarize_v100_interactive_followup_20260411.py
