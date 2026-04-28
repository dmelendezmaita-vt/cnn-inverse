#!/usr/bin/env bash
#SBATCH --job-name=v100_large_bucket
#SBATCH --account=neuro-collab
#SBATCH --partition=v100_normal_q
#SBATCH --qos=fal_v100_normal_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=350G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/neuro-collab/data/runs/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch}"
TASK_CSV="${TASK_CSV:?TASK_CSV is required}"
NODES_BUCKET="${NODES_BUCKET:?NODES_BUCKET is required}"
PROGRESS_CSV="${PROGRESS_CSV:?PROGRESS_CSV is required}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
SUMMARY_JSON="${SUMMARY_JSON:?SUMMARY_JSON is required}"
MAX_TASKS="${MAX_TASKS:-0}"

mkdir -p /projects/neuro-collab/data/runs/logs

echo "$(date -Is) [v100-large] start job=${SLURM_JOB_ID:-NA} nodes=${SLURM_NNODES:-NA} bucket=${NODES_BUCKET}"
echo "$(date -Is) [v100-large] task_csv=${TASK_CSV}"
echo "$(date -Is) [v100-large] progress_csv=${PROGRESS_CSV}"
echo "$(date -Is) [v100-large] results_root=${RESULTS_ROOT}"
echo "$(date -Is) [v100-large] summary_json=${SUMMARY_JSON}"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

python "${REPO_ROOT}/scripts/run_v100_large_bucket.py" \
  --task-csv "${TASK_CSV}" \
  --nodes-bucket "${NODES_BUCKET}" \
  --progress-csv "${PROGRESS_CSV}" \
  --results-root "${RESULTS_ROOT}" \
  --summary-json "${SUMMARY_JSON}" \
  --repo "${REPO_ROOT}" \
  --max-tasks "${MAX_TASKS}"

rc=$?
echo "$(date -Is) [v100-large] done job=${SLURM_JOB_ID:-NA} rc=${rc}"
exit ${rc}
