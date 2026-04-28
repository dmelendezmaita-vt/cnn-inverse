#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
RESOURCE_DIR="$REPO/data/important_notes/simformer_gpu_verify_20260427/resource_monitor"
VENV_PY="$REPO/.venvs/simformer_py312_jax0423/bin/python"
SIM_NVIDIA_ROOT="$REPO/.venvs/simformer_py312_jax0423/lib/python3.12/site-packages/nvidia"
SIM_LIB_PATHS="$SIM_NVIDIA_ROOT/cudnn/lib:$SIM_NVIDIA_ROOT/cublas/lib:$SIM_NVIDIA_ROOT/cusolver/lib:$SIM_NVIDIA_ROOT/cusparse/lib:$SIM_NVIDIA_ROOT/cufft/lib:$SIM_NVIDIA_ROOT/nccl/lib:$SIM_NVIDIA_ROOT/nvjitlink/lib:$SIM_NVIDIA_ROOT/cuda_runtime/lib"

cd "$REPO"
/projects/neuro-collab/conda/neuro-collab-env/bin/python \
  scripts/run_hh_track4_monitored_command_20260427.py \
  --resource-dir "$RESOURCE_DIR" \
  --gpu-expected 1 \
  --command "cd $REPO && env PYTHONPATH=\"$REPO:/projects/neuro-collab/code/dl-kit-main:\${PYTHONPATH:-}\" LD_LIBRARY_PATH=\"$SIM_LIB_PATHS:\${LD_LIBRARY_PATH:-}\" JAX_PLATFORMS=cuda $VENV_PY scripts/verify_simformer_jax_gpu_20260427.py --size 4096 --repeats 8"
