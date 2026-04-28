#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?alloc job id required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUDI_REGISTRY="${REPO}/data/important_notes/optimization_track_20260424_tc_hh_track4_rudi_closure_block/tables/tc_hh_track4_rudi_closure_block_registry_20260424_tc_hh_track4_rudi_closure_block.csv"

python - <<'PY' "$RUDI_REGISTRY"
import csv
import sys
import time
from pathlib import Path

registry = Path(sys.argv[1])
while True:
    if registry.exists():
        rows = list(csv.DictReader(registry.open()))
        terminal = [r for r in rows if r.get("status") in {"COMPLETED", "FAILED"}]
        if len(rows) >= 12 and len(terminal) == len(rows):
            break
    time.sleep(60)
PY

cd /home/dmm96
python "${REPO}/tools/launch_tc_hh_track4_aligned_multicurrent_deterministic_20260425.py" --alloc-job-id "${ALLOC_JOB_ID}"
