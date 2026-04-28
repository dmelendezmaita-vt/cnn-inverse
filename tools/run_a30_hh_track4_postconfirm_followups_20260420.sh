#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALLOC_JOB_ID="${1:?usage: run_a30_hh_track4_postconfirm_followups_20260420.sh <alloc_job_id> <alloc_nodelist>}"
ALLOC_NODELIST="${2:?usage: run_a30_hh_track4_postconfirm_followups_20260420.sh <alloc_job_id> <alloc_nodelist>}"
CLUSTER="${3:-falcon}"

cd "${REPO}"

python tools/build_a30_hh_track4_noise_robustness_20260420.py
python tools/test_launch_a30_hh_track4_noise_robustness_20260420.py
python tools/launch_a30_hh_track4_noise_robustness_20260420.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}"

python tools/build_a30_hh_track4_classical_baselines_20260420.py
python tools/test_launch_a30_hh_track4_classical_baselines_20260420.py
python tools/launch_a30_hh_track4_classical_baselines_20260420.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}"
