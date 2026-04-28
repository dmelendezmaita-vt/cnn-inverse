#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
ALLOC_JOB_ID="${1:?usage: run_a30_hh_track4_pending_baseline_followups_20260420.sh <alloc_job_id> <alloc_nodelist>}"
ALLOC_NODELIST="${2:?usage: run_a30_hh_track4_pending_baseline_followups_20260420.sh <alloc_job_id> <alloc_nodelist>}"
CLUSTER="${3:-falcon}"

cd "${REPO}"

python scripts/build_a30_hh_track4_classical_robustness_20260420.py
python scripts/test_launch_a30_hh_track4_classical_robustness_20260420.py
python scripts/launch_a30_hh_track4_classical_robustness_20260420.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}"

python scripts/build_a30_hh_track4_extra_trees_tuning_20260420.py
python scripts/test_launch_a30_hh_track4_extra_trees_tuning_20260420.py
python scripts/launch_a30_hh_track4_extra_trees_tuning_20260420.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}"
