#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?alloc job id required}"
ALLOC_NODELIST="${2:?alloc nodelist required}"
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

cd /home/dmm96
python "${REPO}/tools/launch_v100_hh_track4_aligned_multicurrent_sbi_in_allocation_20260425.py" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}"
