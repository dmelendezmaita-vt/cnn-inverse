# HH Track4 Final Closure Report

Workspace-clock generated: `2026-04-23`

## Executive Claim

The fixed HH Track4 inverse-problem workload is now closed as a workspace-supported practical frontier, with an explicit boundary:

- clean overall winner: `random_forest_500` on the promoted `raw_plus_fft256_summary12` representation
- cheap classical representative: `knn_k11`
- clean SBI winner: `FMPE` on `raw_plus_fft256_summary12`
- promoted clean SNPE reference: feature-aware `SNPE-MAF` on `raw_plus_fft256_summary12`
- robust SBI path for the localized checkpoint6 failures: shift-specific matched corruption-aware SNPE
- direct fitting / differentiable simulator: blocked by missing simulator provenance, not scientifically exhausted

The strongest safe conclusion is therefore not "all scientifically relevant methods are exhausted." It is:

> The repo-supported and practical literature-backed branches that can be executed from the current workspace have been pushed to a decision-grade frontier on the fixed HH Track4 task. Remaining stronger closure is blocked mainly by the absence of a runnable HH simulator contract and parameter semantics for direct fitting.

## Live Operational State

Live state was refreshed during this continuation before making current operational claims. Workspace/Falcon clock at verification: `2026-04-23 23:19 EDT`.

- Falcon V100 allocation `368063` is `RUNNING` on `fal[101-103,108]`, with `4` nodes, `96` CPUs, and `gres/gpu:v100:2` per node.
- Falcon A30 allocations `368038` and `368759` remain `PENDING`.
- The short Falcon allocation `368960` was cancelled at the user's request and is no longer active.
- Tinkercliffs CPU allocation `5122799` is `RUNNING` on `tc047`, with `18` CPUs and a 14-day walltime.
- Tinkercliffs A100 allocation `5014617` remains `PENDING`.

This means the continuing active resources are the longer Falcon V100 allocation and the Tinkercliffs CPU allocation. CPU work should remain focused on provenance, aggregation, reports, and classical diagnostics; GPU work should be reserved for SBI/neural tasks that can materially advance closure.

## Gate Outcomes

| gate | status | closure result |
|---|---|---|
| `G1` | completed | validate-calibrated SNPE closed negative |
| `G2` | completed positive | `raw_plus_fft256_summary12` promoted for SNPE |
| `G3` | completed positive | matched train-time drift and matched train-time masking are robust, shift-specific SNPE wins |
| `G3b` | completed mixed | `FMPE` improves clean SBI, `NPSE` is covered but not promoted, `FMPE` does not replace robust SNPE |
| `G4` | blocked, evidence audited | direct fitting cannot be run without simulator provenance |
| `G5` | completed with caveats | classical microbenchmark plus standardized existing-run V100 SBI timing now cover final representatives |
| `G6` | completed | this final closure report states the safe claim and remaining gap |

## Clean-Data Frontier

The clean-data conclusion has two layers that should not be merged:

- overall clean winner by MAE: `random_forest_500`
- clean SBI winner: `FMPE`

Clean representative means:

| representative | family | MAE | MSE | R2 | status |
|---|---|---:|---:|---:|---|
| `random_forest_500` | classical | `441.033` | `2.349e6` | `0.1989` | overall clean MAE winner |
| `knn_k11` | classical | `501.381` | `2.732e6` | `0.1192` | cheap classical representative |
| `FMPE` + `raw_plus_fft256_summary12` | SBI | `582.303` | `2.005e6` | `0.2679` | clean SBI winner |
| feature-aware `SNPE-MAF` | SBI | `597.208` | `2.180e6` | `0.2587` | promoted clean SNPE reference |
| `NPSE` + `raw_plus_fft256_summary12` | SBI | `620.612` | `2.073e6` | `0.2598` | covered, not promoted |
| `effnet_beta005_invvar_e500` | neural | `879.462` | `4.847e6` | `-0.1029` | dominated proxy |

Interpretation:

- `random_forest_500` remains the best clean point-estimation representative on MAE.
- `FMPE` is the best clean SBI method seen so far and clears the promoted feature-aware SNPE reference.
- `NPSE` is slower and not accuracy-positive enough to promote.
- the neural representative is clearly dominated on clean accuracy and is not a systems winner.

## Robustness Frontier

The robustness result is shift-specific. There is no universal robust-SNPE model promoted from the tested designs.

Localized checkpoint6 instability modes:

- `drift010`
- `mask20`

Shift-specific SBI robustness means:

| shift | representative | MAE | MSE | R2 | comparison |
|---|---|---:|---:|---:|---|
| `drift010` | clean-trained feature-aware SNPE | `944.97` | `4.409e6` | `-0.1833` | unstable reference |
| `drift010` | matched train-time drift SNPE | `676.31` | `2.834e6` | `0.1408` | promoted robust drift path |
| `drift010` | clean-trained FMPE | `988.47` | `4.600e6` | `-0.2074` | not robust |
| `mask20` | clean-trained feature-aware SNPE | `981.08` | `4.633e6` | `-0.2403` | unstable reference |
| `mask20` | matched train-time masking SNPE | `618.98` | `2.290e6` | `0.2157` | promoted robust mask path |
| `mask20` | clean-trained FMPE | `1078.45` | `5.135e6` | `-0.2861` | not robust |

The combined-corruption guardrail is negative:

- clean-trained feature-aware SNPE on clean eval: `MAE 597.208`, `MSE 2.180e6`, `R2 0.2587`
- drift010+mask20 train-corrupted SNPE on clean eval: `MAE 996.121`, `MSE 4.740e6`, `R2 -0.2405`

Therefore:

- matched corruption-aware SNPE is a real robustness mitigation path
- it should remain shift-specific
- the naive combined corruption design should not be reopened without a genuinely different mixture or augmentation-preserving training design

## Systems-Efficiency Closure

The final G5 package now has:

- fresh CPU microbenchmarks for `random_forest_500` and `knn_k11`
- extracted V100 train/eval/total timing from completed clean SNPE, clean FMPE, NPSE, and matched-corruption SNPE seed blocks
- host peak RSS for the final SBI representatives joined from Slurm step accounting
- three direct V100 GPU-memory observations for each final SBI representative are now recorded in checkpoint7 rerun packages
- the prior neural proxy kept visible as dominated
- a direct-fitting blocker row

Final systems table:

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv`

SBI Slurm host-memory audit:

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_sbi_slurm_memory_audit_20260423.csv`

SBI chained V100 GPU-memory rerun audit:

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_vram_chain_audit_20260423.csv`

Key timing means:

| representative | hardware | train | eval | total | ms/eval example | note |
|---|---|---:|---:|---:|---:|---|
| `knn_k11` | CPU | `0.003s` | `0.159s` | `3.88s` | `0.156` | cheapest completed representative |
| `random_forest_500` | CPU | `213.69s` | `0.097s` | `221.30s` | `0.095` | best clean MAE |
| feature-aware SNPE | V100 | `104.53s` | `8.43s` | `125.58s` | `32.9` | promoted clean SNPE reference |
| `FMPE` | V100 | `53.98s` | `102.27s` | `166.72s` | `399.5` | clean SBI winner, slower evaluation |
| matched drift SNPE | V100 | `101.91s` | `8.83s` | `126.14s` | `34.5` | robust drift path |
| matched mask SNPE | V100 | `112.87s` | `9.40s` | `137.36s` | `36.7` | robust mask path |
| `NPSE` | V100 | `92.31s` | `374.81s` | `482.24s` | `1464.1` | covered, not promoted |
| neural proxy | A30+V100 | `235.76s` | `25.88s` | `261.64s` | `25.3` | dominated accuracy |

Systems caveats:

- classical rows use CPU, `n_train=4096`, and `n_eval=1024`
- SBI rows use V100, `n_train=12288`, `n_eval=256`, and `posterior_samples=16`
- classical peak RSS is recorded directly; SBI host peak RSS is joined from Slurm step accounting
- the original completed SBI blocks did not log GPU VRAM, but checkpoint7 reruns now provide three direct V100 GPU-memory observations for each surviving SBI representative
- these are standardized enough for a stronger V100 memory comparison across the surviving representatives, but still not a multi-hardware GPU-VRAM-complete benchmark

## Assumption-Conditioned Aligned Framework Follow-Up

After the initial closure package, the user explicitly required that the remaining aligned multi-current literature branches be implemented natively rather than left at placeholder or framework-check level. Under the user-approved alignment assumption, the following additional branches are now completed:

| representative | family | implementation level | MAE | MSE | R2 | note |
|---|---|---|---:|---:|---:|---|
| native `BayesFlow`, `meanstd` | BayesFlow | native V100 offline amortized workflow | `594.6757` | `3165560.25` | `0.0951` | strongest completed native framework result in this aligned lane |
| native `BayesFlow`, `concat` | BayesFlow | native V100 offline amortized workflow | `670.0533` | `3487251.0` | `-0.0040` | weaker than the native `meanstd` BayesFlow variant |
| native `Swyft`, `meanstd` | Swyft marginal-ratio | native V100 implementation | `803.3525` | `4381718.5` | `-0.2610` | supersedes the earlier Swyft placeholder |
| native `Swyft`, `concat` reduced-width no-pairwise | Swyft marginal-ratio | native V100 implementation | `904.5768` | `5459437.5` | `-0.4341` | concat is possible only after shrinking the model, and remains clearly noncompetitive |
| native `Swyft`, explicit TMNRE-style truncation follow-up | Swyft marginal-ratio | native V100 implementation | `876.4901` | `5338647.0` | `-0.2826` | first explicit rectangular-truncation stage ran successfully, but did not prune the bank enough to improve the result |
| pool-sequential `FMPE` | sequential SBI | repaired rerun | `752.4014` | `2784573.75` | `0.1440` | fixed-pool sequential SBI, not a new frontier winner |
| pool-sequential `SNPE` | sequential SBI | repaired rerun | `991.0191` | `4464426.0` | `-0.1738` | clearly noncompetitive |

Summary table:

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan/tables/hh_track4_native_framework_and_sequential_summary_20260425.csv`

Interpretation:

- these branches are assumption-conditioned, because they rely on the user-approved row-alignment assumption rather than recovered provenance
- native `BayesFlow` `meanstd` is the strongest completed result from this aligned native-framework follow-up, but it does not beat the clean-data frontier already established in the main workload
- native `Swyft` was implemented and run successfully, which closes the earlier implementation gap for that family at the framework level, but its completed representative is not competitive with the main clean-data frontier
- `Swyft` `concat` required material model simplification to become executable on V100 and still remained clearly noncompetitive
- the first explicit `TMNRE`-style truncation follow-up also failed to change the decision, because the selected truncation region effectively retained the full candidate bank and the second-stage result stayed weak
- repaired sequential SBI also fails to reopen the clean-data frontier

Therefore, the aligned native-framework follow-up reduces the remaining literature gap materially, but it does not overturn the main closure statement. These results should be treated as additional assumption-conditioned evidence, not as replacements for the main clean-data frontier.

## Direct Fitting Boundary

Direct fitting remains the important unresolved scientific family, but the blocker is workspace provenance, not lack of effort.

Additional provenance audit artifacts were added:

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/reports/hh_track4_simulator_provenance_audit_20260423.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/tables/hh_track4_simulator_provenance_evidence_20260423.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/tables/hh_track4_raw_data_schema_20260423.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/tables/hh_track4_target_ranges_by_current_20260423.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_literature_provenance_inference/reports/hh_track4_literature_provenance_inference_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_literature_provenance_inference/tables/hh_track4_literature_backed_provenance_inferences_20260424.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_registry/reports/hh_track4_assumption_registry_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_registry/tables/hh_track4_assumption_registry_20260424.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_sandbox/reports/hh_track4_assumption_conditioned_compact_hh_reevaluation_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_sandbox/tables/hh_track4_assumption_conditioned_compact_hh_decision_20260424.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_sandbox/reports/hh_track4_assumption_conditioned_compact_hh_battery_summary_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_sandbox/reports/hh_track4_assumption_conditioned_compact_hh_optimization_results_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_sandbox/tables/hh_track4_assumption_conditioned_compact_hh_optimization_decision_20260424.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search/reports/hh_track4_assumption_conditioned_fit_family_decision_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search/tables/hh_track4_assumption_conditioned_fit_family_decision_20260424.csv`

The audit recovered the fixed supervised data contract:

- Track4 tars expose array payloads only, with no generator/protocol/code payload.
- current labels recoverable from filenames: `0.1`, `0.2`, `0.3`, `0.4`, `0.5`
- each current has `15000` feature rows and `15000` target rows
- each feature row is a raw float32 vector of length `400000`
- each target row has six raw float32 parameters
- the base config deliberately keeps target names neutral as `hh_param_1` through `hh_param_6`

The literature-backed note improves interpretation but does not remove the blocker:

- best-effort inference: Track4 is most consistent with a compact HH-style current-clamp inverse task
- direct fitting remains scientifically relevant in the literature
- exact parameter semantics, protocol waveform details, and simulator identity still are not recoverable from literature alone
- an explicit assumption registry now records what may be assumed for exploratory, assumption-conditioned analysis without upgrading those assumptions into provenance claims
- an assumption-conditioned compact HH screen, small variant battery, and bounded optimization battery were run
- those tests improved the surrogate fit somewhat but still did not show a compelling missed-winner lane under the current surrogate assumptions
- a broader fit-family battery then covered both differential-evolution and BPTT optimizers, in both hybrid and feature-only objective modes, and still did not reveal a compelling missed-winner lane

The workspace still lacks:

- a runnable HH simulator contract for Track4
- data-generation script or authoritative protocol metadata
- explicit semantic mapping for `hh_param_1` through `hh_param_6`
- time-grid/current-injection metadata needed to define a mechanistic forward fit
- BluePyOpt/profile-likelihood/direct-fitting implementation path

Do not phrase this as direct fitting being scientifically exhausted. The safe phrase is:

> direct fitting and differentiable-simulator approaches remain scientifically relevant but are not executable from the current workspace without inventing missing simulator provenance.

## Final Safe Closure Statement

The campaign can now be closed at the strongest safe workspace level:

1. The implemented classical, neural, SNPE, corruption-aware SNPE, and representative modern-SBI branches have decision-grade coverage on this fixed HH Track4 workload.
2. The clean-data winner is `random_forest_500`; the clean SBI winner is `FMPE`; the robustness path for the localized `drift010` and `mask20` failures is matched corruption-aware SNPE.
3. No already-tested negative branch should be reopened without a concrete new design reason.
4. Systems efficiency is now standardized enough to support the representative comparison, with memory caveats for SBI.
5. Additional assumption-conditioned aligned native-framework testing has now been completed, including native `Swyft` and native `BayesFlow`, and none of those branches overturns the established clean-data frontier.
6. The remaining gap to a literature-complete closure claim is direct fitting, blocked by missing simulator provenance, plus any stronger TMNRE-specific truncation workflow or broader posterior-diagnostic package if one insists on exhausting those families beyond the now-completed native baseline implementations.

## What Would Change This Claim

The claim should be reopened only if one of these happens:

- simulator provenance is recovered, enabling a real direct-fitting or differentiable-simulator baseline
- a new robust-SBI training design is introduced that preserves clean examples while addressing drift and masking together
- a new primary-source method family is identified that is both scientifically relevant and executable without inventing missing HH semantics
- a memory-complete single-hardware benchmark is required for publication-grade systems claims

Until then, the current endpoint is a safe closure with a named remaining gap, not an unlimited open-ended search.
