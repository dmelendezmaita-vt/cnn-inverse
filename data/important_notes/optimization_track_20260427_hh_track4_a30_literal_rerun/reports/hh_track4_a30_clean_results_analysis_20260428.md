# HH Track4 Clean A30 Results Analysis

## Methods

- campaign root: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun`
- task count completed under the clean homogeneous campaign: `622`
- direct task phases completed: `502`
- all result interpretation below is based on the fresh A30-local campaign root, its manifest and registry, task-local `metrics_summary.json` artifacts, task-local `resource_summary.json` sidecars, and the clean A30-local literature, decision-rule, hybrid, and active-policy tables regenerated inside that root

## Coverage

- direct summary rows: `140`
- active-policy rows: `14`
- aligned framework rows: `17`
- decision-rule rows: `63`
- hybrid initializer rows: `77`

## Findings

- best direct strategy by clean mean MAE: `extra_trees_500_depth20` with `500.430226` in `phaseAP_track4_a30_baseline_finalists_confirmation`
- best aligned framework row: `a30runall__bayesflow_native_meanstd_v100_20260425` with `MAE 595.630310` and `R2 0.106437`
- best comparable posterior decision-rule row: `a30runall__bayesflow_set_posterior_v100_20260426::median` with `MAE 564.638123`
- best hybrid initializer row: `hybrid_refinement_surrogate_bundle_prediction_snpe_maf_r3_pool4096_final2048_medoid_20260427` with `objective 2.644513`
- best heuristic active-design row: `a30surr__active_sequential_wasserstein_smoke_20260425` with `mean_abs_log10_error_params_mean 0.464670` under policy `wasserstein`
- best ASNPE-style row: `a30surr__asnpe_extended_v100_20260428` with `mean_abs_log10_error_params_mean 0.655782`
- best non-comparable upstream Simformer decision-rule row: `a30lit__simformer_official_hh_primary_20260426::mean` with `MAE 7.624424`

## Active Design Ladder

| run | policy | budget | mean_abs_log10_error_params_mean | runtime_sec |
|---|---|---|---:|---:|
| `a30surr__asnpe_extended_v100_20260428` | `asnpe_style` | `extended` | 0.655782 | 376.271882 |
| `a30surr__asnpe_primary_v100_20260426` | `asnpe_style` | `primary` | 0.794177 | 367.729573 |
| `a30surr__asnpe_secondary_v100_20260426` | `asnpe_style` | `secondary` | 0.842372 | 108.736677 |
| `a30surr__asnpe_smoke_local_20260426` | `asnpe_style` | `smoke` | 1.075431 | 30.676366 |
| `a30surr__active_sequential_disagreement_extended_20260428` | `disagreement` | `extended` | 0.837470 | 152.114490 |
| `a30surr__active_sequential_default_20260425` | `disagreement` | `full` | 0.611035 | 37.591469 |
| `a30surr__active_sequential_disagreement_full_20260425` | `disagreement` | `full` | 0.611035 | 37.758825 |
| `a30surr__active_sequential_smoke_20260425` | `disagreement` | `smoke` | 0.466811 | 4.746905 |
| `a30surr__active_sequential_uncertainty_extended_20260428` | `uncertainty` | `extended` | 0.823975 | 154.339422 |
| `a30surr__active_sequential_uncertainty_full_20260425` | `uncertainty` | `full` | 0.613888 | 37.792729 |
| `a30surr__active_sequential_uncertainty_smoke_20260428` | `uncertainty` | `smoke` | 0.520432 | 4.700259 |
| `a30surr__active_sequential_wasserstein_extended_20260428` | `wasserstein` | `extended` | 0.827883 | 152.535622 |
| `a30surr__active_sequential_wasserstein_full_20260425` | `wasserstein` | `full` | 0.613765 | 37.929654 |
| `a30surr__active_sequential_wasserstein_smoke_20260425` | `wasserstein` | `smoke` | 0.464670 | 4.698010 |

## Resource Findings

| phase | n_completed | elapsed_sec_mean | cpu_pct_mean | rss_gb_max | gpu_util_max_mean |
|---|---:|---:|---:|---:|---:|
| `phaseAK_track4_a30_replication` | 16 | 165.050 | 176.562 | 19.218 | 87.500 |
| `phaseAL_track4_a30_gpu_profile` | 6 | 149.533 | 177.833 | 19.217 | 85.500 |
| `phaseAM_track4_a30_effnet_schedule_retune` | 12 | 241.271 | 187.833 | 19.219 | 94.250 |
| `phaseANALYSIS_direct` | 1 | 1.558 | 120.000 | 0.086 | 0.000 |
| `phaseANALYSIS_hybrid` | 1 | 1.045 | 33.000 | 0.085 | 0.000 |
| `phaseANALYSIS_literature_surrogate` | 1 | 26.527 | 64.000 | 0.947 | 0.000 |
| `phaseAN_track4_a30_effnet_tail_schedule_retune` | 12 | 209.792 | 187.083 | 19.218 | 94.500 |
| `phaseAO_track4_a30_effnet_local_matrix` | 36 | 210.591 | 188.833 | 19.220 | 93.639 |
| `phaseAP_track4_a30_baseline_finalists_confirmation` | 16 | 11.492 | 2730.750 | 1.239 | 0.000 |
| `phaseAP_track4_a30_classical_baselines` | 12 | 7.748 | 1207.583 | 1.240 | 0.000 |
| `phaseAP_track4_a30_classical_robustness` | 40 | 8.114 | 1466.450 | 1.240 | 0.000 |
| `phaseAP_track4_a30_extra_trees_targeted_confirmation` | 16 | 15.110 | 3076.250 | 12.800 | 0.000 |
| `phaseAP_track4_a30_extra_trees_targeted_tuning` | 32 | 14.578 | 3442.531 | 12.799 | 0.000 |
| `phaseAP_track4_a30_extra_trees_tuning` | 28 | 13.311 | 2898.357 | 1.595 | 0.000 |
| `phaseAP_track4_a30_frontier_head_to_head` | 64 | 11.968 | 2651.984 | 12.799 | 0.000 |
| `phaseAP_track4_a30_knn_robustness_tuning` | 80 | 5.725 | 297.663 | 12.799 | 0.000 |
| `phaseAP_track4_a30_knn_targeted_tuning` | 20 | 7.008 | 272.350 | 12.800 | 0.000 |
| `phaseAP_track4_a30_noise_robustness` | 20 | 20.852 | 500.550 | 18.910 | 0.000 |
| `phaseAP_track4_a30_top3_confirmation` | 12 | 209.680 | 188.917 | 19.217 | 93.250 |
| `phaseAP_track4_a30_tuned_tree_robustness` | 80 | 11.778 | 2779.262 | 12.800 | 0.000 |
| `phaseLIT_aligned_runall` | 19 | 536.639 | 245.368 | 19.020 | 53.941 |
| `phaseLIT_new_methods` | 3 | 51.762 | 162.667 | 2.126 | 89.500 |
| `phaseMANIFEST_hybrid_append` | 1 | 0.994 | 47.000 | 0.104 | 0.000 |
| `phaseSURR_active` | 14 | 114.921 | 218.643 | 1.707 | 0.000 |
| `phaseSURR_aux` | 1 | 62.204 | 321.000 | 1.490 | 80.000 |
| `phaseSURR_direct_fitting` | 79 | 19.205 | 94.684 | 0.104 | 0.000 |

## Limitations

- direct fixed-array rows, aligned posterior rows, surrogate direct-fitting rows, active-design rows, and upstream Simformer rows should not be pooled into one numerical leaderboard without comparability labels
- the active-design ladder is broader than before, but each rung is still represented by a single clean A30 rerun, so budget-by-seed variance is not fully characterized
- the ASNPE rows use their own manifest schema, and the clean A30 analysis normalizes them into the active-policy table using top-level posterior error fields rather than a shared aggregate-metrics block

## Output Files

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_methods_20260428.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_direct_summary_20260428.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_active_policy_ladder_20260428.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_resource_phase_summary_20260428.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_key_findings_20260428.csv`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/reports/hh_track4_a30_clean_results_analysis_20260428.md`
