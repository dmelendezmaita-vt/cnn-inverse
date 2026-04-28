# Track 1 Split Integrity and Benchmark Equivalence Note (2026-04-21)

## Purpose

Document the validation/test split issue found in the original reference code,
verify whether it was real, and clarify what changed in the Track 1 parity
reruns after the split logic was corrected.

## Short answer

Yes. The original stored reference code did not implement a proper disjoint
validation/test split on the legacy `2020-12-09` path.

For the paper-matched benchmark configuration (`Ntrain=1000`, `Nvalidate=2000`,
`Ntest=2000`), the original code made `validate` and `test` the same subset.
It also dropped the true last `Ntest` rows of the source arrays from evaluation.

The current Track 1 parity reruns use corrected split logic, so they are
scientifically cleaner than the archived benchmark. But because of that, they
are no longer evaluation-split-equivalent to the original paper/code behavior.

## Original-code verification

Reference code path:

- `data/important_notes/first_track_paper_parity/fhn_dnn_original_unmodified_20260302_050951/fhn_dnn-1-implementation-in-pytorch/pytorch/data.py`

### Code evidence

In the original loader, the legacy `2020` branch does:

1. Load the full array.
2. Truncate it to `all[:-Ntest]` and store that in the pool variable.
3. Build `test` from the tail of that already-truncated pool.
4. Later build `validate` from the tail of the same pool.

Relevant lines in the stored reference file:

- features:
  - `data.py:98-100`
- targets:
  - `data.py:115-117`
- split into train/validate/test:
  - `data.py:251-253`
  - `data.py:258-260`

That logic is effectively:

```text
pool = all[:-Ntest]
test = pool[-Ntest:]
validate = pool[-Nvalidate:]
train = pool[:Ntrain]
```

So when `Nvalidate == Ntest`, `validate == test`.

For the Track 1 paper-matched case:

- `Ntrain = 1000`
- `Nvalidate = 2000`
- `Ntest = 2000`

Therefore:

- `validate` and `test` are the same rows
- the true last `2000` rows of the original array are never used as test data

## Run-artifact verification

Archived benchmark run:

- `data/important_notes/first_track_paper_parity/fhn_dnn_original_unmodified_20260302_050951/benchmark_records/job_242036_20260304_194449/runs_dnn/run_dnn_info.log`

The benchmark log reports identical validate/test metrics:

- validate MSE/MAE/R2:
  - `0.0027437591925263405 / 0.03488574177026749 / 0.973210334777832`
- test MSE/MAE/R2:
  - `0.0027437591925263405 / 0.03488574177026749 / 0.973210334777832`

This matches the code diagnosis above and is strong evidence that the archived
benchmark used an overlapped validate/test split.

## What changed in the working repo

Current working loader:

- `pytorch/data.py`

The split bug was explicitly corrected in the current loader for the legacy
path:

- `pytorch/data.py:554-625`

The corrected logic now does:

```text
pool = all[:-Ntest]
test = all[-Ntest:]
```

Then it performs the train/validate split inside the pool using explicit split
indices:

- `pytorch/data.py:788-835`

For the default sequential strategy:

- `train = first Ntrain rows of pool`
- `validate = last Nvalidate rows of pool`
- `test = last Ntest rows of full array`

This makes train/validate/test disjoint as long as `Ntrain + Nvalidate <= len(pool)`.

## What Track 1 parity runs actually used

Modern Track 1 parity config:

- `pytorch/configs/params_dnn_tar.yaml`
- seeded copies under:
  - `pytorch/configs/paper_equivalence_v100/`

Key differences from the archived benchmark:

- `dataset_layout: tar_singlefile_split`
- tar-backed publication data via `data_prefix: publication_2020`
- corrected split logic in current `pytorch/data.py`
- distributed-capable runtime in current `pytorch/run_dnn.py`

The current runtime also persists the chosen split indices automatically:

- `pytorch/run_dnn.py:570-573`

Example corrected Track 1 run:

- `/projects/neuro-collab/data/runs/242167`

Saved config:

- `/projects/neuro-collab/data/runs/242167/params.yaml`

Saved split indices:

- `/projects/neuro-collab/data/runs/242167/split_indices.npz`

Verified properties for that run:

- `idx_train.shape = (1000,)`
- `idx_validate.shape = (2000,)`
- `idx_train` and `idx_validate` have `0` overlap
- with sequential splitting, the run uses:
  - train = first `1000` rows of the pool
  - validate = last `2000` rows of the pool
  - test = last `2000` rows of the full array

The run metrics are no longer identical between validate and test:

- validate:
  - `MSE = 0.0025913212448358536`
  - `MAE = 0.034083541482686996`
  - `R2 = 0.9744073748588562`
- test:
  - `MSE = 0.00306942337192595`
  - `MAE = 0.036326583474874496`
  - `R2 = 0.9711751341819763`

This is exactly what we would expect from a corrected disjoint holdout split.

## Implication for the word "benchmark"

There are now two different notions of benchmark in Track 1:

1. Paper-behavior benchmark
   - archived/original code behavior
   - preserves the original split bug
   - closest to reproducing what the original code actually did

2. Corrected-split parity rerun
   - current code behavior
   - uses a proper disjoint holdout test set
   - scientifically cleaner, but not strictly equivalent to the original paper/code evaluation split

Because of this, the archived benchmark and the current parity reruns are:

- configuration-matched on the main scientific knobs
- data-identical at the file-content level
- not evaluation-split-equivalent

So if we compare current Track 1 reruns to the archived benchmark, part of the
difference is expected to come from the split correction itself, not necessarily
from model drift or distributed-execution effects.

## Recommended wording going forward

Use these labels explicitly:

- `paper-behavior benchmark` for the archived original-code run(s)
- `corrected-split parity reruns` for the current tar/DDP Track 1 runs

Avoid calling the corrected-split Track 1 runs "paper-equivalent benchmark"
without qualification.

## Current fix status in the working repo

An explicit paper-behavior reproduction mode now exists in the working loader.

Code path:

- `pytorch/data.py`

Config added:

- `pytorch/configs/track1_exact_reproduction/params_dnn_paper_behavior.yaml`

Key setting:

- `data.legacy_validate_test_overlap: true`

This opt-in mode restores the original Track 1 split semantics for benchmark
use while leaving the corrected split as the default for ordinary modern runs.

## 2026-04-21 reproduction rerun using current code

Run output:

- `/projects/neuro-collab/data/runs/track1_exact_reproduction_benchmark_20260421/368801`

What was run:

- current `pytorch/run_dnn.py`
- current `pytorch/data.py`
- config:
  `pytorch/configs/track1_exact_reproduction/params_dnn_paper_behavior.yaml`
- Falcon V100, single-GPU step inside active allocation `368801`

Observed metrics from the rerun:

- train:
  - `MSE = 0.002474844688549638`
  - `MAE = 0.032615579664707184`
  - `R2 = 0.9769289493560791`
- validate:
  - `MSE = 0.0027407100424170494`
  - `MAE = 0.03491185978055`
  - `R2 = 0.9732168912887573`
- test:
  - `MSE = 0.0027407100424170494`
  - `MAE = 0.03491185978055`
  - `R2 = 0.9732168912887573`

Comparison against archived original benchmark `242036`:

- validate/test `MSE` delta:
  - `-3.0491501092910767e-06` (`-0.1111%`)
- validate/test `MAE` delta:
  - `+2.611801028251648e-05` (`+0.0749%`)
- validate/test `R2` delta:
  - `+6.556510925292969e-06` (`+0.00067%`)

Interpretation:

- The working-code reproduction mode is now close enough to the archived
  original benchmark to serve as an exact paper-behavior benchmark in
  practical terms.
- Validate and test are again numerically identical, as required for matching
  the original behavior.
- The remaining tiny metric differences are on the order normally expected from
  runtime/library/environment drift rather than a substantive split mismatch.

## Bottom line

The split issue was real.

We did fix it in the working code.

After that fix, the current Track 1 parity runs became a better scientific
evaluation, but they stopped being strictly equivalent to the archived
paper-behavior benchmark on the validation/test split semantics.

As of 2026-04-21, the working repo now also has an explicit paper-behavior
reproduction mode and a verified rerun that restores benchmark equivalence for
Track 1 benchmark purposes.
