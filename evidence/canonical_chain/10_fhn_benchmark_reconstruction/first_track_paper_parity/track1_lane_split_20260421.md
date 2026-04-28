# Track 1 Lane Split Note (2026-04-21)

## Purpose

Record the canonical separation between:

1. the Track 1 benchmark lane that preserves paper-behavior split semantics
2. the corrected-split lane that should be treated as a follow-on Track 1.5 /
   `track1_corrected`

## Canonical lanes

### Track 1 benchmark lane

Use this lane when the goal is:

- benchmarking against original paper/code behavior
- preserving the legacy validate/test overlap semantics
- building the seeded benchmark matrix for reference comparisons

Canonical config paths:

- single benchmark:
  - `pytorch/configs/track1_exact_reproduction/params_dnn_paper_behavior.yaml`
- seeded V100 matrix:
  - `pytorch/configs/track1_benchmark_v100/params_dnn_paper_behavior_seed301.yaml`
  - `...seed302.yaml`
  - `...seed303.yaml`
  - `...seed304.yaml`
  - `...seed305.yaml`

Canonical tracking/output paths:

- benchmark note:
  - `data/important_notes/first_track_paper_parity/track1_split_integrity_and_benchmark_equivalence_20260421.md`
- seeded matrix CSV:
  - `data/important_notes/first_track_paper_parity/v100_track1_benchmark_training_jobs.csv`

Key semantic marker:

- `data.legacy_validate_test_overlap: true`

### Track 1 corrected lane (`track1_corrected` / Track 1.5)

Use this lane when the goal is:

- evaluating the corrected disjoint split
- comparing against a scientifically cleaner holdout test
- studying the effect of the split fix itself

Canonical config paths:

- `pytorch/configs/track1_corrected_v100/params_dnn_tar_seed301.yaml`
- `...seed302.yaml`
- `...seed303.yaml`
- `...seed304.yaml`
- `...seed305.yaml`

Canonical tracking/output paths:

- corrected seeded matrix CSV:
  - `data/important_notes/first_track_paper_parity/v100_track1_corrected_training_jobs.csv`
- rerun CSV (if rerun through the new allocation runner):
  - `data/important_notes/first_track_paper_parity/v100_track1_corrected_training_jobs_rerun.csv`

## Legacy names

These older names now map to the corrected lane and should be treated as
historical/legacy labels:

- `paper_equivalence_v100`
- `paper_data_parity_v100`
- `v100_paper_data_parity_training_jobs.csv`

They are useful for provenance, but they are no longer the canonical benchmark
lane names.

## Runner

For the actual Track 1 benchmark matrix, use the dedicated single-GPU runner:

- `scripts/run_track1_v100_benchmark_matrix_in_allocation_20260421.sh`

Usage:

```bash
ALLOCATION_JOB_ID=<running_falcon_v100_jobid> \
bash scripts/run_track1_v100_benchmark_matrix_in_allocation_20260421.sh
```

For the corrected lane only, use:

- `scripts/run_track1_corrected_v100_matrix_in_allocation_20260421.sh`

Usage:

```bash
ALLOCATION_JOB_ID=<running_falcon_v100_jobid> \
bash scripts/run_track1_corrected_v100_matrix_in_allocation_20260421.sh
```

## Bottom line

Going forward:

- `Track 1` means the paper-behavior benchmark lane
- `track1_corrected` / `Track 1.5` means the corrected-split lane

## 2026-04-21 correction note

An initial distributed paper-behavior attempt was made on 2026-04-21, but that
lane has been removed from the working tree because it followed the wrong
distributed `1n/2n/4n` matrix shape.

Only the canonical benchmark lane and the corrected lane should remain in the
working tree going forward.

The canonical seeded benchmark lane is instead:

- config root:
  `pytorch/configs/track1_benchmark_v100/`
- output root:
  `/projects/neuro-collab/data/runs/track1_benchmark_v100/`
- tracking CSV:
  `data/important_notes/first_track_paper_parity/v100_track1_benchmark_training_jobs.csv`
