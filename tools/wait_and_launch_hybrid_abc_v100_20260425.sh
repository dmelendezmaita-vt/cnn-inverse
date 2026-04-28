#!/usr/bin/env bash
set -euo pipefail
ALLOC_JOB_ID="${1:?alloc job id required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DET_REGISTRY="${REPO}/data/important_notes/optimization_track_20260425_tc_hh_track4_aligned_multicurrent_deterministic/tables/tc_hh_track4_aligned_multicurrent_deterministic_registry_20260425_tc_hh_track4_aligned_multicurrent_deterministic.csv"
python - <<'PY' "$DET_REGISTRY"
import csv, sys, time
from pathlib import Path
p=Path(sys.argv[1])
while True:
    if p.exists():
        rows=list(csv.DictReader(p.open()))
        terminal=[r for r in rows if r.get('status') in {'COMPLETED','FAILED'}]
        if len(rows) >= 4 and len(terminal)==len(rows):
            break
    time.sleep(60)
PY
HOST=$(scontrol show hostnames $(squeue -j "$ALLOC_JOB_ID" -h -o %N) | head -n 1)
OUT_ROOT="${REPO}/data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan/runs"
mkdir -p "$OUT_ROOT"
srun --jobid="$ALLOC_JOB_ID" --exclusive --mem=0 -N1 -n1 -c8 -w "$HOST" bash -lc '
  module load Miniforge3 >/dev/null 2>&1
  source activate /projects/neuro-collab/conda/neuro-collab-env
  export PYTHONUNBUFFERED=1
  python /projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/tools/run_hh_track4_assumption_conditioned_wasserstein_abc_20260425.py --save-dir /projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan/runs/wasserstein_abc_v100_20260425 --n-particles 256 --keep-top 24 --n-exemplars 3 --trace-length 2000'
srun --jobid="$ALLOC_JOB_ID" --exclusive --mem=0 -N1 -n1 -c8 -w "$HOST" bash -lc '
  module load Miniforge3 >/dev/null 2>&1
  source activate /projects/neuro-collab/conda/neuro-collab-env
  export PYTHONUNBUFFERED=1
  python /projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/tools/run_hh_track4_assumption_conditioned_hybrid_refinement_20260425.py --save-dir /projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan/runs/hybrid_refinement_v100_20260425 --init-mode midpoint --n-exemplars 3 --trace-length 2000 --maxiter 120'
