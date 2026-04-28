# HH Track4 Aligned-Current Literature Exhaustion Plan

Workspace-clock generated: `2026-04-25`

## Scope

This note answers a narrower question than the main closure report:

> If we explicitly assume that rows are aligned across current files `0.1` to `0.5`, what method families remain to be tested before we can say the literature-backed approach space is practically exhausted under that assumption?

This note is:

- assumption-conditioned
- not provenance-recovered
- intended to guide the final remaining testing lanes

## New Assumption

Assume:

> row `i` in each current-labeled file corresponds to the same latent HH parameter vector, observed under different injected-current settings.

Operational consequence:

- each example becomes a multi-observation inverse problem
- the scientifically relevant unit is no longer a single trace only
- methods that pool information across multiple current conditions become legitimate remaining literature lanes

## What Is Already Covered

The following broad classes are already covered enough that they should not be reopened without a new structural idea:

1. classical point predictors on the fixed clean task
2. single-observation supervised inverse nets on the fixed clean task
3. major amortized SBI families already available in `sbi`
   - `SNPE`
   - `FMPE`
   - `NPSE`
   - `SNLE`
   - `SNRE`
4. assumption-conditioned compact-HH direct fitting
   - differential evolution
   - BPTT / gradient fitting
   - hybrid trace+feature objective
   - feature-only objective

The live bounded Rudi-style closure block reinforces this:

- the baseline repeat is tight across seeds
- the raw+FFT proxy, wider model, and low-LR stability variant stay in the same weak neural band
- current evidence does not indicate a hidden large gain inside the plain single-observation inverse-net family

## Remaining Method Families Under The Alignment Assumption

### 1. Multi-current deterministic inverse maps

This becomes mandatory once alignment is assumed.

Why:

- Rudi-style inverse maps previously consumed only one current condition at a time
- if five aligned current responses exist, a fair deterministic baseline must test joint conditioning

Minimal literature-complete block:

- concatenation baseline:
  - stack all five current traces as channels or as a long concatenated input
- shared encoder + per-current branch aggregator
- set-encoder variant:
  - DeepSets-style pooling or permutation-invariant set aggregation
- masked-current training:
  - random current dropout during training so the model can use subsets of protocols

Why this matters:

- this is the cleanest remaining path for improving point-estimate accuracy at scale
- it can exploit more information without needing simulator recovery

### 2. Hierarchical / multi-observation SBI

This also becomes mandatory once alignment is assumed.

Primary literature anchors:

- HNPE, NeurIPS 2021
- BayesFlow, 2020

Why:

- the aligned setting is no longer a single-observation posterior estimation problem
- it is a repeated-observation problem with shared latent parameters

Minimal literature-complete block:

- one HNPE-style normalizing-flow posterior over shared parameters
- one BayesFlow-style invertible conditional posterior over sets of observations
- one ablation over number of current conditions:
  - 1 current
  - 2 currents
  - all 5 currents

Evaluation should include:

- point summaries:
  - posterior mean
  - posterior median
  - MAP or approximate MAP
- calibration:
  - coverage
  - rank behavior
- practical scaling:
  - train time
  - inference time per held-out example

Why this matters:

- if alignment is true, this is likely the strongest remaining accuracy lane in the literature-backed amortized family

### 3. SBI validation and recalibration

This is still missing in a modern sense.

Primary literature anchors:

- SBC, Talts et al. 2018
- L-C2ST, NeurIPS 2023
- differentiable coverage calibration, NeurIPS 2023

Why:

- we trained posterior estimators, but we have not exhausted the posterior-diagnostic literature
- better calibrated posteriors can improve decision quality even if base point accuracy changes little

Minimal block:

- `SBC` on the final `FMPE` and feature-aware `SNPE` representatives
- `L-C2ST` or equivalent local posterior diagnostic on representative held-out observations
- post-hoc coverage recalibration on the same representatives

This should be run both for:

- single-current posterior references
- aligned multi-current posterior references, if implemented

### 4. TMNRE, not just vanilla SNRE

Vanilla `SNRE` was already covered and looked weak.

That does **not** fully close the ratio-estimation literature, because `TMNRE` is a more targeted modern ratio-estimation family.

Why it still matters:

- it is specifically motivated by simulation efficiency and low-dimensional marginal posterior quality
- it is the natural follow-up if we want to honestly close the modern ratio-based SBI lane

Minimal block:

- one `TMNRE` representative on the aligned-current task
- compare against `FMPE` and feature-aware `SNPE`

Expected value:

- more likely to strengthen closure than to produce a breakthrough
- still important for literature completeness

### 5. Hybrid amortized initialization plus local refinement

This remains a distinct open family in the assumption-conditioned simulator lane.

Why:

- pure amortized inference and pure mechanistic fitting were both tested
- the hybrid literature-motivated bridge has not been tested directly

Minimal block:

- initialize the compact-HH sandbox optimizer from:
  - the deterministic inverse-net prediction
  - the posterior mean from the best SBI model
- compare against random initialization

This is still:

- assumption-conditioned
- not provenance-recovered

But under the current assumptions it is a legitimate remaining family.

### 6. Wasserstein evaluation stack

Wasserstein is useful primarily as an evaluation layer, and secondarily as an inference family through ABC / SMC.

What already exists locally:

- per-target univariate Wasserstein diagnostics from `predictions.npz`

What is still missing:

1. per-target Wasserstein on the final Track4 representatives
   - raw space
   - `log10` space

2. joint sliced-Wasserstein over the full 6D target vector
   - to detect whether a method matches marginals but distorts joint geometry

3. posterior-predictive Wasserstein
   - compare observed and posterior-predictive trace summaries across current conditions

4. if the aligned-current SBI path is run:
   - Wasserstein as a posterior-predictive fit diagnostic across subsets of conditions

Why it matters:

- `MAE` and `MSE` are samplewise point-error metrics
- Wasserstein detects distributional mismatch and can reveal mode collapse or shrinkage behavior

Why it is not enough alone:

- a model can have decent Wasserstein distance while still making poor samplewise inverse predictions
- so Wasserstein should be secondary to task accuracy for point estimators

### 7. Sliced-Wasserstein ABC / Wasserstein SMC

This is the main remaining optimal-transport inference family.

Why it is distinct:

- it is not equivalent to DE, BPTT, or NPE/NRE/NLE
- it uses OT discrepancy directly inside likelihood-free inference

Why it is lower priority:

- it is expensive
- it is assumption-conditioned
- it is unlikely to dominate amortized inference on scale

When it becomes worth running:

- after the aligned multi-current deterministic and hierarchical SBI branches
- after the hybrid initialization bridge

### 8. Active sequential posterior estimation

This is real literature, but lower priority here.

Why:

- its biggest value is simulation efficiency when simulator calls are expensive
- on fixed arrays, the benefit is limited

It becomes more relevant only if:

- we keep the aligned-current assumption
- and we want to choose optimal subsets of current conditions or protocol order

## What We Do Not Need To Reopen

The following do not look like necessary remaining branches under the alignment assumption:

- more generic single-current CNN / EfficientNet / ConvResNet tuning
- more vanilla `SNLE` or vanilla `SNRE`
- more compact-HH optimizer families without a new structural lever
- waiting for different GPU hardware to repeat the same experiments

## Recommended Order

1. finish the live bounded Rudi closure block
2. extend evaluation with:
   - per-target Wasserstein
   - joint sliced-Wasserstein
3. build one aligned multi-current deterministic baseline
4. build one aligned hierarchical SBI baseline
5. run `SBC + L-C2ST + coverage recalibration`
6. add one `TMNRE` block
7. run hybrid amortized-init plus local refinement in the compact-HH sandbox
8. only then, if ambiguity remains, run sliced-Wasserstein ABC

## Bottom Line

If alignment is assumed, the strongest remaining accuracy-at-scale lane is:

> methods that pool information across aligned current conditions, either deterministically or probabilistically.

The strongest remaining validation lane is:

> posterior calibration and local posterior diagnostics, plus Wasserstein-based distributional checks.

The strongest remaining assumption-conditioned mechanistic lane is:

> amortized initialization plus local refinement.

Under this alignment assumption, the campaign is **not** yet literature-exhausted. But the remaining open space is now much more specific:

- aligned multi-current inference
- modern posterior validation
- one targeted ratio-estimation follow-up
- Wasserstein-based evaluation and, if needed, OT-based ABC
