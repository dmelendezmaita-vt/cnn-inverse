# HH Track4 Checkpoint7 Standardized Efficiency Microbenchmark (2026-04-23)

## Purpose
- define a smallest useful G5 standardized CPU-side efficiency package for classical representatives
- benchmark `random_forest_500` and `knn_k11` on the frozen Track4 split
- hold `feature_mode=raw_plus_fft256_summary12` fixed across all rows

## Package Layout
- matrix: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_matrix_20260423_hh_track4_checkpoint7_standardized_efficiency.csv`
- append-only result table target: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_results_20260423_hh_track4_checkpoint7_standardized_efficiency.csv`
- run outputs: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/runs`
- logs: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/logs`
- runner: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/run_hh_track4_checkpoint7_standardized_efficiency_20260423.py`

## Matrix
- rows: `6` = 3 repeats x 2 representatives
- status after build: `PENDING`; no jobs are submitted or run by this builder

## Runner Measurements
- feature/load time, fit time, eval time, total time
- peak RSS via Python `resource` when available
- n_train, n_eval, sec_per_1k_train, ms_per_eval_example
- MAE, MSE, R2 after inverse target scaling/transform

## Reuse Notes
- runner imports loader/model/target inverse helpers from `run_hh_classical_baseline_20260420.py`
- the fixed feature mode matches the raw+FFT256+summary12 helper behavior used by the SBI baseline path without copying large helper blocks

## Status
- rows completed: `6/6`
- representatives: `random_forest_500`, `knn_k11`
- repeats: `3`
- feature mode: `raw_plus_fft256_summary12`
- split cache: shared Track4 `4096/1024` train/test cache
- hardware: CPU shell on the current cluster environment
- results table: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_results_20260423_hh_track4_checkpoint7_standardized_efficiency.csv`

## Results

| representative | repeats | MAE | MSE | R2 | feature/load mean | fit mean | eval mean | total mean | peak RSS mean | sec/1k train | ms/eval example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random_forest_500` | 3 | `441.033` | `2.349e6` | `0.1989` | `7.51s` | `213.69s` | `0.097s` | `221.30s` | `13058 MB` | `52.17` | `0.095` |
| `knn_k11` | 3 | `501.381` | `2.732e6` | `0.1192` | `3.72s` | `0.003s` | `0.159s` | `3.88s` | `13065 MB` | `0.0008` | `0.156` |

## Interpretation
- `random_forest_500` remains the clean classical accuracy representative, but its fit cost is roughly two orders of magnitude above `knn_k11`
- `knn_k11` remains the cheap classical representative; most of its measured time is feature/cache materialization rather than fitting
- both classical representatives have sub-millisecond per-example evaluation on this frozen `1024`-example test pass
- peak RSS is dominated by loading/materializing the shared HH arrays and feature representation, so these measurements are most useful as package-level memory estimates rather than pure estimator memory

## Final Representative Addendum
- final representative summary: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv`
- SBI seed rows: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_standardized_efficiency_sbi_seed_rows_20260423.csv`
- summary json: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/notes/hh_track4_checkpoint7_standardized_efficiency_final_representatives_summary_20260423.json`

The addendum standardizes the remaining non-classical evidence by extracting train/eval/total runtime from completed V100 `metrics_summary.json` files rather than rerunning finished science. It covers:
- clean promoted feature-aware SNPE
- clean FMPE, the clean SBI winner
- NPSE as the modern-SBI covered but non-promoted contrast
- matched-drift and matched-mask SNPE, the shift-specific robust SBI path
- the prior dominated neural proxy
- the direct-fitting blocker row

Representative means:

| family | representative | role | eval | runs | MAE | MSE | R2 | train | eval time | total | ms/eval example |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| classical | `random_forest_500` | clean accuracy | clean | 3 | `441.033` | `2.349e6` | `0.1989` | `213.69s` | `0.097s` | `221.30s` | `0.095` |
| classical | `knn_k11` | cheap classical | clean | 3 | `501.381` | `2.732e6` | `0.1192` | `0.003s` | `0.159s` | `3.88s` | `0.156` |
| SBI | `sbi_feature_aware_snpe_clean` | promoted SNPE reference | clean | 4 | `597.208` | `2.180e6` | `0.2587` | `104.53s` | `8.43s` | `125.58s` | `32.9` |
| SBI | `sbi_fmpe_clean` | clean SBI winner | clean | 4 | `582.303` | `2.005e6` | `0.2679` | `53.98s` | `102.27s` | `166.72s` | `399.5` |
| SBI | `sbi_npse_clean_not_promoted` | covered, not promoted | clean | 4 | `620.612` | `2.073e6` | `0.2598` | `92.31s` | `374.81s` | `482.24s` | `1464.1` |
| SBI | `sbi_snpe_matched_drift010` | robust drift path | `drift010` | 4 | `676.306` | `2.835e6` | `0.1408` | `101.91s` | `8.83s` | `126.14s` | `34.5` |
| SBI | `sbi_snpe_matched_mask20` | robust mask path | `mask20` | 4 | `618.976` | `2.290e6` | `0.2157` | `112.87s` | `9.40s` | `137.36s` | `36.7` |
| neural | `effnet_beta005_invvar_e500` | dominated proxy | clean | proxy | `879.462` | `4.847e6` | `-0.1029` | `235.76s` | `25.88s` | `261.64s` | `25.3` |
| direct fitting | `blocked_missing_simulator_metadata` | workspace blocker | n/a | n/a |  |  |  |  |  |  |  |

Systems interpretation:
- the classical rows remain the only fully fresh CPU microbenchmark rows with peak RSS recorded
- the SBI rows now have consistent seed-level V100 train/eval/total timing from the completed representative runs
- `FMPE` is faster to train than feature-aware SNPE but much slower at posterior evaluation under the current `posterior_samples=16`, `sample_with=ode` setup
- the shift-specific matched SNPE robustness path has SNPE-like evaluation cost and does not introduce an order-of-magnitude runtime penalty
- SBI host peak RSS was joined from Slurm step accounting for the final seed rows
- checkpoint7 V100 reruns now add three direct GPU-memory observations for each final SBI representative:
  `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/tables/hh_track4_checkpoint7_vram_chain_audit_20260423.csv`
- the systems claim is therefore materially stronger on V100 memory behavior, but it is still not a multi-hardware GPU-VRAM-complete benchmark across all families
- neural remains dominated on clean accuracy and is not a systems winner
- direct fitting remains unbenchmarkable until simulator provenance is recovered
