#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?alloc job id required}"
REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
DET_REGISTRY="${REPO}/data/important_notes/optimization_track_20260425_tc_hh_track4_aligned_multicurrent_deterministic/tables/tc_hh_track4_aligned_multicurrent_deterministic_registry_20260425_tc_hh_track4_aligned_multicurrent_deterministic.csv"
RUN_ROOT="${REPO}/data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan/runs"
mkdir -p "${RUN_ROOT}"

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

HOST=$(ssh -o BatchMode=yes -J dmm96@falcon2.arc.vt.edu dmm96@fal101 "scontrol show hostnames \$(squeue -j ${ALLOC_JOB_ID} -h -o %N) | head -n 1")

srun --jobid="${ALLOC_JOB_ID}" --overlap -N1 -n1 -c1 -w "${HOST}" bash -lc "
  module load Miniforge3 >/dev/null 2>&1
  source activate /projects/neuro-collab/conda/neuro-collab-env
  export PYTHONPATH=/projects/neuro-collab/code/dl-kit-main:${REPO}:\${PYTHONPATH:-}
  export PYTHONUNBUFFERED=1
  export NC_STAGE_SPLIT_CACHE_TO_TMPDIR=1
  export NC_LOCAL_STAGE_ROOT=\${TMPDIR:-/localscratch/\${SLURM_JOB_ID:-}}
  python ${REPO}/scripts/warm_hh_split_cache_tmpdir_20260425.py \
    --params ${REPO}/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml \
    --currents 0.1,0.2,0.3,0.4,0.5 \
    --n-train 4096 \
    --n-validate 1024 \
    --n-test 1024 \
    --ensure-source-cache
"

if [[ -x /home/dmm96/venvs/bayesflow_20260425_py312/bin/python ]]; then
  srun --jobid="${ALLOC_JOB_ID}" --exclusive --mem=0 -N1 -n1 -c6 -w "${HOST}" bash -lc \
    "FRAMEWORK_ENV_PATH=/home/dmm96/venvs/bayesflow_20260425_py312 FRAMEWORK_KIND=bayesflow SAVE_DIR=${RUN_ROOT}/bayesflow_v100_20260425 bash ${REPO}/scripts/run_v100_hh_track4_framework_checks_20260425.sh"
fi

if [[ -x /home/dmm96/venvs/swyft_20260425/bin/python ]]; then
  srun --jobid="${ALLOC_JOB_ID}" --exclusive --mem=0 -N1 -n1 -c6 -w "${HOST}" bash -lc \
    "FRAMEWORK_ENV_PATH=/home/dmm96/venvs/swyft_20260425 FRAMEWORK_KIND=swyft SAVE_DIR=${RUN_ROOT}/swyft_v100_20260425 bash ${REPO}/scripts/run_v100_hh_track4_framework_checks_20260425.sh"
fi
