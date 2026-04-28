#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?alloc job id required}"
START_ROW_ID="${2:?start row id required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUDI_REGISTRY="${REPO}/data/important_notes/optimization_track_20260424_tc_hh_track4_rudi_closure_block/tables/tc_hh_track4_rudi_closure_block_registry_20260424_tc_hh_track4_rudi_closure_block.csv"
WAIT_ROW_ID="${WAIT_ROW_ID:-tcrudi_0011}"

python - <<'PY' "$RUDI_REGISTRY" "$WAIT_ROW_ID"
import csv
import sys
import time
from pathlib import Path

registry = Path(sys.argv[1])
wait_row_id = sys.argv[2]

while True:
    if registry.exists():
        rows = list(csv.DictReader(registry.open()))
        matches = [row for row in rows if row.get("row_id") == wait_row_id]
        if matches and matches[-1].get("status") in {"COMPLETED", "FAILED"}:
            break
    time.sleep(30)
PY

cd /home/dmm96
python "${REPO}/tools/launch_tc_hh_track4_rudi_closure_block_20260424.py" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --max-parallel 1 \
  --start-row-id "${START_ROW_ID}"
