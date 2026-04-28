# HH Track4 Closure Plan (2026-04-23)

## Purpose
- turn the remaining open-ended “are we done?” question into an explicit closure program
- define the smallest remaining set of method, robustness, and systems-efficiency gates required before the strongest safe frontier-closure claim

## Current Status
- `Checkpoint 0` through `Checkpoint 6` are materially complete.
- `Checkpoint 7` is materially complete with systems caveats via:
  - partial saturation: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/notes/notes.md`
  - efficiency audit: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_efficiency_audit/notes/notes.md`
  - standardized efficiency package: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/notes/hh_track4_checkpoint7_standardized_efficiency_notes_20260423_hh_track4_checkpoint7_standardized_efficiency.md`
- final closure report: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_closure_plan/reports/hh_track4_final_closure_report_20260423.md`
- current best clean MAE representative: `random_forest_500`
- historical pre-feature SBI MSE anchor: `snpe_maf_h192_t8_n8192`
- current promoted SNPE representation: `snpe_maf_h192_t8_n12288__raw_plus_fft256_summary12`
- current best clean SBI representative: `fmpe_mlp_h192_l8_n12288__raw_plus_fft256_summary12`
- current best robust SBI path: shift-specific matched train-time corruption for SNPE under `drift010` and `mask20`
- fastest completed representative: `knn_k11`
- direct fitting is still blocked by missing simulator-generation metadata in the current workspace

## Strongest Safe Claim Today
- within the currently implemented families and the fixed HH Track4 task contract, the frontier is well characterized
- no obvious easy win remains inside the already tested classical and neural branches
- the SNPE branch did still contain useful work: feature-aware inputs and matched train-time corruption both produced material gains
- the strongest literature-complete closure claim is **not** yet available because:
  - the direct-fitting / differentiable-simulator family is not executable from the current workspace
- the promoted robust-SNPE branch now has a negative combined-corruption clean-cost guardrail, so it should remain shift-specific
- systems-efficiency evidence is standardized enough for the safe representative comparison, with explicit caveats for cross-hardware equivalence and missing SBI peak-memory logs

## Material Improvement Threshold
Use the threshold already defined in the exhaustive plan:
- at least `2%` relative gain on the primary pointwise metric family
- or a clean win on `2/3` of `MSE`, `MAE`, `R2`
- and the gain must survive fresh-seed confirmation
- and the gain must survive at least one hardware replication or equivalent fixed-checkpoint evaluation

If a candidate fails this threshold, do not promote it as a frontier change.

## Agent Findings
- simulator/direct-fitting audit: no runnable simulator contract is present locally; the workspace contains saved arrays, priors, and targets, but not the semantic HH parameter mapping or differentiable forward model needed for a real direct-fitting baseline
- smallest feasible SBI extension: validation-selected posterior decision rules on top of the current SNPE runner; this closed negative
- feature-aware SNPE: `raw_plus_fft256_summary12` materially improved the top SNPE branch on clean data and the localized drift/mask shifts
- misspecification-aware SNPE: matched train-time drift and masking both materially improved the promoted feature-aware SNPE branch under the corresponding shifted evaluations
- modern SBI gap check: `FMPE` improved clean SBI accuracy over feature-aware SNPE, while `NPSE` was slower and not promoted
- `FMPE` shift follow-up: clean-trained `FMPE` did not survive the localized `drift010` and `mask20` failure modes, so it does not replace corruption-aware SNPE for robustness

## Updated Primary Sources Consulted
- feature-aware HH SBI: Beck et al., 2022, `Efficient identification of informative features in simulation-based inference`
  - https://arxiv.org/abs/2210.11915
- more flexible amortized SBI: Gloeckler et al., 2024, `All-in-one simulation-based inference`
  - https://arxiv.org/abs/2404.09636
- misspecification-aware SBI calibration: Wehenkel et al., 2024, `Addressing Misspecification in Simulation-based Inference through Data-driven Calibration`
  - https://arxiv.org/abs/2405.08719
- robust statistics for misspecified SBI: Huang et al., 2023, `Learning Robust Statistics for Simulation-based Inference under Model Misspecification`
  - https://arxiv.org/abs/2305.15871
- differentiable HH/state-space fitting remains a live path: Tanoh et al., 2025, `Identifying multi-compartment Hodgkin-Huxley models with high-density extracellular voltage recordings`
  - https://arxiv.org/abs/2506.20233

## Remaining Closure Gates
The machine-readable table is:
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_closure_plan/tables/closure_gates.csv`

Interpretation:
- `G1` is completed and did not change the frontier.
- `G2` is completed positive: `raw_plus_fft256_summary12` is now the promoted SNPE representation.
- `G3` is completed positive for matched drift and matched masking; its combo clean-cost guardrail was negative, so the robust result is shift-specific rather than one universal corruption model.
- `G3b` is completed mixed: modern `FMPE` is a clean-data SBI improvement, but not a robustness replacement.
- `G4` is a real workspace blocker, not just an unrun matrix.
- `G5` is completed with caveats: classical rows have fresh CPU microbenchmarks, and final SBI representatives now have standardized existing-run V100 timing; SBI peak memory and single-hardware equivalence remain caveats.
- `G6` is completed by the final closure report.

## Hardware Routing
- no immediate GPU-worthy follow-up is required for the current closure claim
- before using any allocation, verify live scheduler state with `squeue`
- use GPUs only if simulator provenance is recovered or a genuinely new robust-SBI design is introduced

## Post-Closure Execution Rule
1. Do not reopen already-negative branches without a concrete new design reason.
2. Treat direct fitting as blocked unless simulator provenance, parameter semantics, and protocol metadata are recovered.
3. Keep the robust-SNPE conclusion shift-specific: matched drift for `drift010`, matched masking for `mask20`.
4. `G5` has been closed with the standardized efficiency package and final representative addendum.
5. `G6` has been closed with the final closure report.

## Stop Conditions
- If `G1` to `G3` are flat and `G4` remains blocked after a serious provenance search, the strongest safe conclusion becomes:
  - the implemented workspace has reached a practically closed frontier for this workload, but the literature-complete claim remains limited by unavailable simulator provenance
- If `G4` becomes executable and also closes negative, and `G5` shows no accuracy-efficiency upset, then the strongest closure claim becomes supportable for both accuracy and systems efficiency on the defined HH Track4 workload

## Final Closure Addendum
- final report: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_closure_plan/reports/hh_track4_final_closure_report_20260423.md`
- final efficiency table: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv`
- SBI seed-level efficiency rows: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_sbi_seed_rows_20260423.csv`

Safe closure language:
- the current workspace-supported practical frontier is closed for the implemented classical, neural, SNPE, corruption-aware SNPE, and representative modern-SBI branches
- direct fitting remains a scientifically relevant but workspace-blocked family
- do not state literature-complete exhaustion unless simulator provenance is recovered and the direct-fitting family can be tested or explicitly ruled out from primary-source evidence
