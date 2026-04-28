#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?alloc job id required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DET_REGISTRY="${REPO}/data/important_notes/optimization_track_20260425_tc_hh_track4_aligned_multicurrent_deterministic/tables/tc_hh_track4_aligned_multicurrent_deterministic_registry_20260425_tc_hh_track4_aligned_multicurrent_deterministic.csv"

python - <<'PY' "$DET_REGISTRY"
import csv
import sys
import time
from pathlib import Path

registry = Path(sys.argv[1])
while True:
    if registry.exists():
        rows = list(csv.DictReader(registry.open()))
        terminal = [r for r in rows if r.get("status") in {"COMPLETED", "FAILED"}]
        if len(rows) >= 4 and len(terminal) == len(rows):
            break
    time.sleep(60)
PY

HOST=$(scontrol show hostnames $(squeue -j "$ALLOC_JOB_ID" -h -o %N) | head -n 1)
srun --jobid="$ALLOC_JOB_ID" --overlap -N1 -n1 -c1 -w "$HOST" bash -lc "
  module load Miniforge3 >/dev/null 2>&1
  source activate /projects/neuro-collab/conda/neuro-collab-env
  export PYTHONPATH=${REPO}/vendor/dlkit:${REPO}:\${PYTHONPATH:-}
  export PYTHONUNBUFFERED=1
  export NC_STAGE_SPLIT_CACHE_TO_TMPDIR=1
  export NC_LOCAL_STAGE_ROOT=\${TMPDIR:-/localscratch/\${SLURM_JOB_ID:-}}
  python ${REPO}/tools/warm_hh_split_cache_tmpdir_20260425.py \
    --params ${REPO}/src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml \
    --currents 0.1 \
    --n-train 4096 \
    --n-validate 1024 \
    --n-test 1024 \
    --ensure-source-cache
"

cd /home/dmm96
python "${REPO}/tools/launch_tc_hh_track4_pool_sequential_sbi_20260425.py" --alloc-job-id "${ALLOC_JOB_ID}"
