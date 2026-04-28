# Neural Inverse Inference Workflows for FitzHugh-Nagumo and Hodgkin-Huxley Models

This repository is a public, reproducibility-focused staging tree prepared from the active neuro-collab working copy. It contains executable source code, configuration files, baseline FitzHugh-Nagumo data, the vendored `dlkit` dependency that the PyTorch workflow requires, and a curated subset of text-first evidence artifacts that support the current canonical Hodgkin-Huxley conclusions.

## Start Here

The primary human entrypoint is the numbered reproduction layer under `repro/`, because the canonical workflow is no longer the same as the older internal track chronology preserved in `data/important_notes/`.

## Public Repository Scope

| Area | Included in this staging repo | Technical role |
| --- | --- | --- |
| `pytorch/`, `tensorflow/`, `utils/` | Yes | Core model code and baseline training and evaluation entry points |
| `scripts/` | Yes | Experiment builders, launchers, analyzers, and reporting utilities used in the live workflow |
| `third_party/dl-kit-main/` | Yes | Required sibling dependency for `dlkit.*` imports used by the PyTorch code |
| `data/2020-12-09/` | Yes | Small FitzHugh-Nagumo baseline dataset, which fits within GitHub file-size limits |
| `data/important_notes/` | Curated subset only | Text-first manifests, reports, notes, and tables that support the current canonical claims |
| Runtime logs, live run directories, shared scratch mirrors, model checkpoints | No | Omitted to keep the public tree clean, portable, and within GitHub storage constraints |
| Large Hodgkin-Huxley tar archives and local cluster scratch paths | No | These remain external data dependencies and are described in `data/README.md` |

## Environment Setup

The PyTorch workflow depends on the vendored `dlkit` package and on the project-specific requirements file.

```bash
python -m pip install -e third_party/dl-kit-main
python -m pip install -r pytorch/requirements.txt
```

Convenience requirements files are also provided:

```bash
python -m pip install -r requirements-public-pytorch.txt
python -m pip install -r requirements-public-tensorflow.txt
```

## Baseline FitzHugh-Nagumo Run

```bash
python pytorch/run_dnn.py --params pytorch/configs/params_dnn.yaml --mode train
python pytorch/run_dnn.py --params pytorch/configs/params_dnn.yaml --mode eval
```

## Reproduction Surface

| Surface | Role |
| --- | --- |
| `repro/00_source_snapshot/` | exact reference and command path for the untouched upstream zip |
| `repro/10_fhn_benchmark_reconstruction/` | inherited benchmark reconstruction evidence |
| `repro/20_hh_canonical_clean_a30/` | canonical clean A30 manifest and analysis path |
| `repro/30_hh_closure_boundary/` | bounded closure claim, efficiency evidence, and provenance blocker |
| `repro/40_state_of_the_art_followup/` | later executable literature-aligned and assumption-conditioned follow-up |
| `repro/90_legacy_upload_bundle/` | earlier packaging logic retained as legacy context |

The staging rules and omitted artifact classes are documented in `docs/github_upload_scope.md`.
