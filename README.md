# Neural Inverse Inference Workflows for FitzHugh-Nagumo and Hodgkin-Huxley Models

This repository contains the public source tree for the FitzHugh-Nagumo and Hodgkin-Huxley inverse-inference workflows, together with a curated execution surface, a small FitzHugh-Nagumo starter dataset, a static dashboard, and the vendored `dlkit` dependency used by the PyTorch code.

## Repository Layout

| Path | Role |
| --- | --- |
| `src/` | PyTorch and TensorFlow source trees |
| `scripts/` | curated execution, staging, and analysis entrypoints |
| `dashboard/` | static result dashboard |
| `data/2020-12-09/` | shipped FitzHugh-Nagumo starter dataset |
| `vendor/dlkit/` | vendored dependency used by the PyTorch workflows |

## Falcon Setup

The batch suites are intended to run from within the Falcon cluster.

```bash
ssh falcon
module load Python/3.12.3-GCCcore-13.3.0
cd /path/to/repository
```

If you only need the public PyTorch stack, create one environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-public-pytorch.txt
```

If you intend to run the complete smoke or full batch suites, use the repository bootstrap script, which creates and verifies both required environments:

```bash
bash scripts/prepare_falcon_envs.sh
```

That script prepares:

| Environment | Purpose |
| --- | --- |
| `.venv` | core PyTorch, SBI, Swyft, and utility stack |
| `.venv-bayesflow` | BayesFlow-specific stack, which is separated because it requires a different `numpy` line |

## Data

The repository ships the small FitzHugh-Nagumo starter dataset, while the full Hodgkin-Huxley arrays are external.

To run the Hodgkin-Huxley workflows, provide the archive `concatenated_data.tar.gz`, either:

| Method | How |
| --- | --- |
| explicit path | pass `--hh-tar-path /path/to/concatenated_data.tar.gz` |
| repository-local tarball | place `concatenated_data.tar.gz` at the repository root |

The current default staging strategy is `nvme_full_extract`, under which each node:

1. copies the archive to local NVMe,
2. extracts it once,
3. reuses the extracted tree across later experiments on that same node.

The batch runner and the HH helper scripts support the same staging contract. If you already maintain a reusable extracted tree elsewhere, you can point the helpers at it with `--prepared-data-root`.

## Baseline Runs

FitzHugh-Nagumo DNN baseline:

```bash
python src/pytorch/run_dnn.py --params src/pytorch/configs/params_dnn.yaml --mode train
python src/pytorch/run_dnn.py --params src/pytorch/configs/params_dnn.yaml --mode eval
```

Hodgkin-Huxley smoke DNN baseline:

```bash
python src/pytorch/run_dnn.py --params src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml --mode train
python src/pytorch/run_dnn.py --params src/pytorch/configs/hh/params_dnn_tar_hh_smoke.yaml --mode eval
```

## Batch Runner

The public control surface is `scripts/run_batch.py`.

List available suites and batches:

```bash
python scripts/run_batch.py --list
```

Declared suites:

| Suite | Purpose |
| --- | --- |
| `falcon_a30_smoke` | smoke validation of the public FHN and HH workflows |
| `canonical_complete_smoke` | smoke validation of the full public experiment surface, including BayesFlow, Swyft, distributed launcher paths, posterior decision rules, and surrogate branches |
| `scientific_smoke` | reduced scientific reproduction suite |
| `canonical_complete` | full public experiment suite |
| `canonical_falcon` | Falcon-oriented HH reproduction suite |

Typical commands:

```bash
python scripts/run_batch.py \
  --suite falcon_a30_smoke \
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
```

For targeted runs, the same runner also supports:

| Control | Role |
| --- | --- |
| `--batch <name>` | run one batch |
| `--through <name>` | stop after a named batch |
| `--resume` | continue past completed batches |
| `--prepared-data-root <path>` | reuse an existing extracted HH dataset root |

## Resource Monitoring

Slurm-backed batch steps write resource-monitor artifacts under each step output root. The monitor records:

| Metric class | Coverage |
| --- | --- |
| process tree | CPU percent, RSS, VMS, top resident processes |
| node | per-CPU utilization, RAM usage, swap usage |
| GPU | per-GPU compute utilization, VRAM usage, step-attributed VRAM |

These artifacts are produced automatically by the batch runner and the distributed interactive launcher.

## Dashboard

The repository includes a static dashboard under `dashboard/`, which summarizes the shipped experiment surface.
