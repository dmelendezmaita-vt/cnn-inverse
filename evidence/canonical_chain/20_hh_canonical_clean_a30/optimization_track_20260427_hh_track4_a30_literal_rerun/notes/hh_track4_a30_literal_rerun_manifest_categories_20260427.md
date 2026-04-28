# HH Track4 A30 Literal Rerun Manifest Categories

## Literal policy
This rerun manifest keeps the full literal A30 HH Track4 branch history that is still represented by existing matrices and registries, which means that analyzed-negative neural phases are preserved as tasks instead of being filtered away to only the promoted classical branches.

## Analysis anchors
- checkpoint6 reference rank signature: `random_forest_clean_winner > knn_robustness_winner > efficientnet_neural_baseline`
- checkpoint6 unstable shift labels: `drift010, mask20`
- checkpoint7 best clean family by MAE: `classical`
- checkpoint7 neural representative role: `dominated_neural_reference`
- checkpoint7 classical clean representative role: `clean_accuracy_representative`
- checkpoint7 kNN representative role: `cheap_classical_representative`
- family audit neural runtime note: `Training and evaluation proxies come from separate representative packages.`
- family audit classical runtime note: `Combined runtime only; train/eval not logged separately for this family.`

## Category table

| category | task_count | phase_count | meaning | evidence files |
|---|---:|---:|---|---|
| `literal_negative_analyzed_neural` | 88 | 5 | Rows from the A30 neural search and confirmation line that were later analyzed and retained as dominated or non-promoted evidence. | `optimization_track_20260423_hh_track4_checkpoint6_shift_stability/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_efficiency_audit/family_efficiency_audit.csv`<br>`optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv` |
| `literal_negative_analyzed_neural_eval_only` | 20 | 1 | Eval-only neural follow-up rows that depend on confirmed top-3 checkpoints and remain part of the literal negative branch. | `optimization_track_20260423_hh_track4_checkpoint6_shift_stability/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv` |
| `promoted_classical_confirmation` | 32 | 2 | Fresh-seed classical confirmation rows that stabilize finalist claims. | `optimization_track_20260423_hh_track4_checkpoint6_shift_stability/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv` |
| `promoted_classical_head_to_head` | 64 | 1 | Literal head-to-head comparison rows between the strongest tuned tree and kNN contenders. | `optimization_track_20260423_hh_track4_checkpoint6_shift_stability/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv` |
| `promoted_classical_robustness` | 200 | 3 | Classical robustness rows that feed the checkpoint6 shift-stability comparison. | `optimization_track_20260423_hh_track4_checkpoint6_shift_stability/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint6_shift_stability/family_shift_metrics.csv`<br>`optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/summary.json` |
| `promoted_classical_search` | 92 | 4 | Classical search rows that feed the later clean-data and robustness contenders. | `optimization_track_20260423_hh_track4_checkpoint6_shift_stability/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint6_shift_stability/family_shift_metrics.csv`<br>`optimization_track_20260423_hh_track4_checkpoint7_partial_saturation/summary.json`<br>`optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv` |
| `support_profile_only` | 6 | 1 | Profiling and instrumentation rows that support the family-efficiency readout rather than a headline accuracy claim. | `optimization_track_20260423_hh_track4_checkpoint7_efficiency_audit/family_efficiency_audit.csv` |

## Phase mapping

| phase | category | tasks | example_strategy |
|---|---|---:|---|
| `phaseAK_track4_a30_replication` | `literal_negative_analyzed_neural` | 16 | `avgpool_baseline` |
| `phaseAL_track4_a30_gpu_profile` | `support_profile_only` | 6 | `avgpool_baseline` |
| `phaseAM_track4_a30_effnet_schedule_retune` | `literal_negative_analyzed_neural` | 12 | `effnet_beta005_invstd_e400` |
| `phaseAN_track4_a30_effnet_tail_schedule_retune` | `literal_negative_analyzed_neural` | 12 | `effnet_beta005_invstd_e400` |
| `phaseAO_track4_a30_effnet_local_matrix` | `literal_negative_analyzed_neural` | 36 | `effnet_beta003_invstd_e400` |
| `phaseAP_track4_a30_baseline_finalists_confirmation` | `promoted_classical_confirmation` | 16 | `extra_trees_500` |
| `phaseAP_track4_a30_classical_baselines` | `promoted_classical_search` | 12 | `ridge_svd128_alpha1` |
| `phaseAP_track4_a30_classical_robustness` | `promoted_classical_robustness` | 40 | `extra_trees_200_clean` |
| `phaseAP_track4_a30_extra_trees_targeted_confirmation` | `promoted_classical_confirmation` | 16 | `extra_trees_500` |
| `phaseAP_track4_a30_extra_trees_targeted_tuning` | `promoted_classical_search` | 32 | `extra_trees_500` |
| `phaseAP_track4_a30_extra_trees_tuning` | `promoted_classical_search` | 28 | `extra_trees_200` |
| `phaseAP_track4_a30_frontier_head_to_head` | `promoted_classical_head_to_head` | 64 | `extra_trees_500_clean` |
| `phaseAP_track4_a30_knn_robustness_tuning` | `promoted_classical_robustness` | 80 | `knn_k5_clean` |
| `phaseAP_track4_a30_knn_targeted_tuning` | `promoted_classical_search` | 20 | `knn_k3` |
| `phaseAP_track4_a30_noise_robustness` | `literal_negative_analyzed_neural_eval_only` | 20 | `effnet_beta005_invvar_e500_clean` |
| `phaseAP_track4_a30_top3_confirmation` | `literal_negative_analyzed_neural` | 12 | `effnet_beta003_invstd_e400` |
| `phaseAP_track4_a30_tuned_tree_robustness` | `promoted_classical_robustness` | 80 | `extra_trees_500_clean` |

## Dependency policy

- `phaseAP_track4_a30_top3_confirmation` depends on the earlier local-matrix anchor for the same EfficientNet strategy.
- `phaseAP_track4_a30_noise_robustness` depends on the explicit top-3 checkpoint source recorded in each matrix row.
- classical robustness and confirmation phases depend on the earlier baseline or tuning anchor that produced the same baseline family.
- frontier head-to-head rows depend on the targeted tree confirmation anchor or the tuned kNN robustness anchor for the matching baseline family.

## Output policy

- each task declares deterministic mirrored outputs under the new campaign root using `runs/{phase}/{launch_group}/{task_id}`
- direct A30 tasks execute the underlying row step scripts inside the clean queue allocation, which keeps monitoring one-to-one with the experiment row and avoids nesting another scheduler under the queue
