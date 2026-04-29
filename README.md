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

## Falcon Setup

The public batch suites are designed to be launched from within the Falcon cluster, using the same A30 allocation contract as the repository's canonical reproduction path.

Log into Falcon first:

```bash
ssh falcon
```

Create a clean repository-local environment, without `--system-site-packages`, then install the public PyTorch stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-public-pytorch.txt
```

If you intend to run the complete canonical branch surface, which now includes the BayesFlow, Swyft, and assumption-conditioned surrogate families, use two repository-local environments under Falcon `Python/3.12.3-GCCcore-13.3.0`:

```bash
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-public-complete-core.txt

python -m venv .venv-bayesflow
source .venv-bayesflow/bin/activate
python -m pip install -r requirements-public-bayesflow.txt

source .venv/bin/activate
```

The split is necessary because the BayesFlow 2.x branches require `numpy>=2.2.6`, while the existing PyTorch, SBI, and `dlkit` stack requires `numpy<2`.

## Baseline Run

```bash
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
| `scripts/run_hh_track4_aligned_multicurrent_bayesflow_20260425.py` and related BayesFlow variants | aligned native BayesFlow framework branches |
| `scripts/run_hh_track4_aligned_multicurrent_swyft_20260425.py` and related Swyft variants | aligned native Swyft framework branches |
| `scripts/run_hh_track4_assumption_conditioned_*` | compact-HH surrogate branches, including active design, Wasserstein ABC, hybrid refinement, and ASNPE |
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

The batch runner also accepts an HH tarball from any location, so the canonical
batch interface does not require a root-level copy when `--hh-tar-path` is
provided explicitly.

Smoke example for the Hodgkin-Huxley DNN path:

```bash
python src/pytorch/run_dnn.py --params src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml --mode train
python src/pytorch/run_dnn.py --params src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml --mode eval
```

## Batch Runner

The repository now includes a batch-aware dispatcher:

```bash
python scripts/run_batch.py --list
```

Three suites are declared:

| Suite | Role |
| --- | --- |
| `falcon_a30_smoke` | smoke validation of the public FHN and HH workflows, designed to run inside the default Falcon A30 allocation |
| `canonical_complete_smoke` | reduced smoke validation for the full canonical branch set, including BayesFlow, Swyft, posterior decision-rule analysis, and the assumption-conditioned surrogate families |
| `scientific_smoke` | Falcon-sized branch reproductions for the public experiment threads |
| `canonical_complete` | the full public canonical branch set, including the native BayesFlow, native Swyft, and assumption-conditioned surrogate families |
| `canonical_falcon` | Falcon-cluster batches for the current environment-anchored HH reproduction path |

Examples:

```bash
python scripts/run_batch.py \
  --suite falcon_a30_smoke \
  --hh-tar-path /path/to/concatenated_data.tar.gz \
  --allocation-job-id "$SLURM_JOB_ID"

python scripts/run_batch.py \
  --suite scientific_smoke \
  --hh-tar-path /path/to/concatenated_data.tar.gz \
  --allocation-job-id "$SLURM_JOB_ID"

python scripts/run_batch.py \
  --suite canonical_complete_smoke \
  --hh-tar-path /path/to/concatenated_data.tar.gz \
  --allocation-job-id "$SLURM_JOB_ID"

python scripts/run_batch.py \
  --suite canonical_complete \
  --hh-tar-path /path/to/concatenated_data.tar.gz \
  --allocation-job-id "$SLURM_JOB_ID"

python scripts/run_batch.py \
  --batch fal01_hh_4node_a30_dnn \
  --hh-tar-path /path/to/concatenated_data.tar.gz \
  --allocation-job-id "$SLURM_JOB_ID"
```

All five suites are designed for execution from within the documented Falcon Slurm allocation, using the same A30-oriented cluster contract and the same external HH tarball contract.

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
