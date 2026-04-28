#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?alloc job id required}"
ALLOC_NODELIST="${2:?alloc nodelist required}"
SBI_ENV_PATH="${3:?sbi env path required}"
REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"

while true; do
  if [[ -x "${SBI_ENV_PATH}/bin/python" ]]; then
    if "${SBI_ENV_PATH}/bin/python" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda is not None and torch.backends.cuda.is_built()) else 1)'; then
      break
    fi
  fi
  sleep 60
done

cd /home/dmm96
SBI_ENV_PATH="${SBI_ENV_PATH}" python "${REPO}/scripts/launch_v100_hh_track4_aligned_multicurrent_sbi_in_allocation_20260425.py" \
  --alloc-job-id "${ALLOC_JOB_ID}" \
  --alloc-nodelist "${ALLOC_NODELIST}" \
  --rerun-failed
