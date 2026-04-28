# Neural Inverse Inference Workflows for FitzHugh-Nagumo and Hodgkin-Huxley Models

This repository ships the core source tree, a small FitzHugh-Nagumo starter dataset, a curated script surface, a static results dashboard, and the vendored `dlkit` dependency required by the PyTorch workflow.

## Repository Surface

| Path | Role |
| --- | --- |
| `src/` | source code for the PyTorch and TensorFlow workflows |
| `scripts/` | curated execution and analysis helpers |
| `dashboard/` | static website for the current canonical result summary |
| `vendor/dlkit/` | vendored dependency used by the PyTorch code |
| `data/2020-12-09/` | shipped baseline FitzHugh-Nagumo data |

## Baseline Run

```bash
python -m pip install -e vendor/dlkit
python -m pip install -r src/pytorch/requirements.txt
python src/pytorch/run_dnn.py --params src/pytorch/configs/params_dnn.yaml --mode train
python src/pytorch/run_dnn.py --params src/pytorch/configs/params_dnn.yaml --mode eval
```

## Curated Script Surface

| Script | Role |
| --- | --- |
| `scripts/shared_data_utils.py` | staged tar extraction helper |
| `scripts/run_interactive_dnn_step.sh` | distributed DNN execution step |
| `scripts/run_classical_baseline_step.sh` | classical baseline execution step |
| `scripts/run_sbi_baseline_step.sh` | SBI execution step |
| `scripts/run_hh_classical_baseline.py` | classical HH baseline driver |
| `scripts/run_hh_sbi_baseline.py` | SBI HH baseline driver |
| `scripts/slurm_smoke_test.sbatch` | smoke job template |
| `scripts/slurm_train.sbatch` | train job template |

The shipped templates are parameterized and do not assume a particular user account, notification channel, or private directory layout. The Hodgkin-Huxley helpers still require external data, because the full arrays are not part of this repository.

## Hodgkin-Huxley Data Placement

To run the current Hodgkin-Huxley workflows from a fresh checkout:

1. download the full dataset archive as `concatenated_data.tar.gz`
2. place that file in the repository root

The public Hodgkin-Huxley configs and helpers are written so that this root-level
tarball is the only required external data artifact. On the first HH run, the
repository prepares a reusable extracted working directory under
`.prepared_data/concatenated_data/`, after which subsequent runs reuse that
local copy automatically.

Smoke example for the Hodgkin-Huxley DNN path:

```bash
python src/pytorch/run_dnn.py --params src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml --mode train
python src/pytorch/run_dnn.py --params src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml --mode eval
```

## Data Boundary

The full Hodgkin-Huxley data are not shipped here.

| Artifact | Size |
| --- | ---: |
| `concatenated_data.tar.gz` | `77205166166` bytes |
| one full-current target array such as `concatenated_data_0.1_curr.npy` | `24000000000` bytes |

The public repository therefore ships no full Hodgkin-Huxley arrays, because those files are far beyond GitHub's practical and hard upload limits.

## Static Dashboard

The repository includes a static website under `dashboard/`, which summarizes the current canonical results using the comparability classes that are defended in the thesis and supporting reports.

## Untouched Upstream Snapshot

The inherited upstream zip referenced during benchmark reconstruction was stored in the original workspace at:

- `/projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip`
- zip comment: `eb676a34bb32d880b172e70f9faf6f41a2d9fe3c`

Verification command:

```bash
unzip -z /projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip
```
