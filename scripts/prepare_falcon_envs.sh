#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-public-complete-core.txt
python - <<'PY'
import torch, sbi, swyft, pytorch_lightning, psutil
print({
    "torch_cuda": torch.cuda.is_available(),
    "cuda_devices": torch.cuda.device_count(),
    "sbi": getattr(sbi, "__version__", "unknown"),
    "swyft": getattr(swyft, "__version__", "unknown"),
    "psutil": getattr(psutil, "__version__", "unknown"),
})
PY

python -m venv .venv-bayesflow
source .venv-bayesflow/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-public-bayesflow.txt
python - <<'PY'
import bayesflow, keras, numpy, psutil
print({
    "bayesflow": bayesflow.__version__,
    "keras": keras.__version__,
    "numpy": numpy.__version__,
    "psutil": getattr(psutil, "__version__", "unknown"),
})
PY

source .venv/bin/activate
echo "Prepared and verified .venv and .venv-bayesflow under ${REPO_ROOT}"
