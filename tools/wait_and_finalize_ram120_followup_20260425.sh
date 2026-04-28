#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO}/data/important_notes/optimization_track_20260425_v100_hh_track4_ram120g_followup/runs/snre_meanstd_ram120g/metrics_summary.json"

while [[ ! -f "${TARGET}" ]]; do
  sleep 30
done

python "${REPO}/tools/finalize_ram120_followup_20260425.py"
