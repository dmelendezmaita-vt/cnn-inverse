# HH Track4 A30 Reporting Synthesis (20260427_hh_track4_a30_literal_rerun)

## Scope

This refresh was regenerated entirely from the supplied campaign root, and it only uses campaign-local matrix tables, registry tables, per-row metrics summaries, per-target metrics, and point-prediction arrays.

## Campaign Coverage

- campaign root: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun`
- discovered matrix tables: 0
- discovered registry tables: 0
- completed runs with usable metrics: 0
- strategy aggregates: 0
- branch aggregates: 0

## Core Findings

| result space | row | metric | note |
|---|---|---:|---|

## Branch Status

| branch | status | completed_rows | total_rows | best_strategy_id | best_mae |
|---|---|---:|---:|---|---:|

## Strategy Comparison

| phase | strategy_base | noise_label | mean_mae | std_mae | mean_r2 | completed_runs |
|---|---|---|---:|---:|---:|---:|

## Limitations

| id | scope | severity | statement |
|---|---|---|---|
| `L1` | `coverage` | `high` | The refresh reads only matrix, registry, metrics, and prediction artifacts present under the supplied campaign root, so missing or pruned rerun rows remain outside scope. |
| `L2` | `comparability` | `high` | These A30-only summaries compare direct point-estimation reruns across local branches, noise settings, and schedules, and they should not be merged numerically with posterior-family analyses from separate campaign roots. |
| `L3` | `diagnostics` | `medium` | The diagnostics are deterministic error-shape diagnostics derived from point predictions and per-target metrics, because posterior-sample calibration artifacts are not part of the A30 direct-output contract. |
| `L4` | `noise` | `medium` | Noise robustness summaries depend on campaign-local strategy naming and metrics metadata, so rows without explicit clean or noise tags remain visible but cannot contribute to clean-vs-noisy deltas. |
| `L5` | `runtime` | `medium` | Runtime columns prefer campaign-local training metadata, then registry elapsed time, which means they reflect the rerun's local execution contract rather than a globally standardized benchmark protocol. |

## Generated Artifacts

- `tables/hh_track4_a30_run_comparison_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_strategy_comparison_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_noise_robustness_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_prediction_diagnostics_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_target_diagnostics_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_literature_branch_registry_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_reporting_key_results_20260427_hh_track4_a30_literal_rerun.csv`
- `tables/hh_track4_a30_reporting_limitations_20260427_hh_track4_a30_literal_rerun.csv`
- `reports/hh_track4_a30_literature_branch_registry_20260427_hh_track4_a30_literal_rerun.md`
- `reports/hh_track4_a30_reporting_synthesis_20260427_hh_track4_a30_literal_rerun.md`
