#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch}"
ALLOC_JOB_ID="${ALLOC_JOB_ID:?ALLOC_JOB_ID is required}"
ALLOC_NODELIST="${ALLOC_NODELIST:?ALLOC_NODELIST is required}"
CLUSTER="${CLUSTER:-falcon}"

cd "${REPO_ROOT}"

python scripts/build_v100_hh_track4_random_forest_targeted_tuning_20260421.py
python scripts/build_v100_hh_track4_classical_stacking_20260421.py
python scripts/build_v100_hh_track4_classical_data_scaling_20260421.py

python scripts/launch_hh_classical_matrix_in_allocation_20260421.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}" \
  --matrix-csv "data/important_notes/optimization_track_20260421_v100_hh_track4_random_forest_targeted_tuning/tables/v100_hh_track4_random_forest_targeted_tuning_matrix_20260421_v100_hh_track4_random_forest_targeted_tuning.csv" \
  --registry-csv "data/important_notes/optimization_track_20260421_v100_hh_track4_random_forest_targeted_tuning/tables/v100_hh_track4_random_forest_targeted_tuning_registry_20260421_v100_hh_track4_random_forest_targeted_tuning.csv" \
  --run-log-json "data/important_notes/optimization_track_20260421_v100_hh_track4_random_forest_targeted_tuning/notes/v100_hh_track4_random_forest_targeted_tuning_run_log_20260421_v100_hh_track4_random_forest_targeted_tuning.json"

python scripts/launch_hh_classical_matrix_in_allocation_20260421.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}" \
  --matrix-csv "data/important_notes/optimization_track_20260421_v100_hh_track4_classical_stacking/tables/v100_hh_track4_classical_stacking_matrix_20260421_v100_hh_track4_classical_stacking.csv" \
  --registry-csv "data/important_notes/optimization_track_20260421_v100_hh_track4_classical_stacking/tables/v100_hh_track4_classical_stacking_registry_20260421_v100_hh_track4_classical_stacking.csv" \
  --run-log-json "data/important_notes/optimization_track_20260421_v100_hh_track4_classical_stacking/notes/v100_hh_track4_classical_stacking_run_log_20260421_v100_hh_track4_classical_stacking.json"

python scripts/launch_hh_classical_matrix_in_allocation_20260421.py \
  --cluster "${CLUSTER}" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}" \
  --matrix-csv "data/important_notes/optimization_track_20260421_v100_hh_track4_classical_data_scaling/tables/v100_hh_track4_classical_data_scaling_matrix_20260421_v100_hh_track4_classical_data_scaling.csv" \
  --registry-csv "data/important_notes/optimization_track_20260421_v100_hh_track4_classical_data_scaling/tables/v100_hh_track4_classical_data_scaling_registry_20260421_v100_hh_track4_classical_data_scaling.csv" \
  --run-log-json "data/important_notes/optimization_track_20260421_v100_hh_track4_classical_data_scaling/notes/v100_hh_track4_classical_data_scaling_run_log_20260421_v100_hh_track4_classical_data_scaling.json"

python scripts/analyze_hh_track4_checkpoint1_20260421.py
