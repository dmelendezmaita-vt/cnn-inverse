#!/usr/bin/env bash
#SBATCH --job-name=a30_hh_t4_knnr_retry_20260421
#SBATCH --account=neuro-collab
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a30:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/dmm96/interactive_alloc/a30_hh_t4_knnr_retry_20260421-%j.out
#SBATCH --error=/home/dmm96/interactive_alloc/a30_hh_t4_knnr_retry_20260421-%j.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REPO}/data/important_notes/optimization_track_20260421_a30_hh_track4_knn_robustness_tuning/tables/a30_hh_track4_knn_robustness_tuning_registry_20260421_a30_hh_track4_knn_robustness_tuning.csv"

module load Miniforge3
source activate /projects/neuro-collab/conda/neuro-collab-env

cd "${REPO}"

mapfile -t FAILED_ROW_IDS < <(
  python - <<'PY'
import csv
from pathlib import Path

p = Path("data/important_notes/optimization_track_20260421_a30_hh_track4_knn_robustness_tuning/tables/a30_hh_track4_knn_robustness_tuning_registry_20260421_a30_hh_track4_knn_robustness_tuning.csv")
if p.exists():
    rows = list(csv.DictReader(p.open()))
    for row in rows:
        if row.get("status") == "FAILED":
            print(row["row_id"])
PY
)

echo "[$(date -Is)] registry=${REGISTRY}"
if ((${#FAILED_ROW_IDS[@]} == 0)); then
  echo "[$(date -Is)] No failed kNN robustness rows found; exiting."
  exit 0
fi

echo "[$(date -Is)] Rerunning failed row ids: ${FAILED_ROW_IDS[*]}"

ARGS=(--cluster falcon --force-rerun)
for rid in "${FAILED_ROW_IDS[@]}"; do
  ARGS+=(--row-id "${rid}")
done

python tools/launch_a30_hh_track4_knn_robustness_tuning_20260421.py "${ARGS[@]}"
