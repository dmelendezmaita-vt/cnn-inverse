# HH Track4 Active Sequential Policy Comparison

- summary csv: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/tables/hh_track4_active_sequential_policy_comparison_20260425.csv`

| run | policy | mean_abs_log10_error_params_mean | mean_relative_error_params_mean | runtime_sec |
|---|---|---:|---:|---:|
| `active_sequential_smoke_20260425` | `` | 0.467097 | 1.002963 | 7.804222 |
| `active_sequential_wasserstein_smoke_20260425` | `wasserstein` | 0.474531 | 0.651896 | 42.103852 |
| `active_sequential_disagreement_full_20260425` | `disagreement` | 0.611035 | 0.869772 | 35.158450 |
| `active_sequential_default_20260425` | `` | 0.611464 | 0.872677 | 30.917326 |
| `active_sequential_wasserstein_full_20260425` | `wasserstein` | 0.613765 | 0.869013 | 35.323478 |
| `active_sequential_uncertainty_full_20260425` | `uncertainty` | 0.613888 | 0.868874 | 35.198447 |
| `asnpe_primary_v100_20260426` | `asnpe_style` | 0.754448 | 20.834720 | 12.793871 |
| `asnpe_smoke_local_20260426` | `asnpe_style` | 0.859409 | 4.422506 | 9.051952 |
| `asnpe_secondary_v100_20260426` | `asnpe_style` | 0.915400 | 26.649282 | 12.448345 |
