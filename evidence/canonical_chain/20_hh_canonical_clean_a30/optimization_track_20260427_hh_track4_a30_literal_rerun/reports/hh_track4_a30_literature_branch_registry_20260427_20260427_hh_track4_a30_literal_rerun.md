# HH Track4 A30 Literature and Surrogate Branch Registry

- source csv: `hh_track4_a30_literature_branch_registry_20260427_20260427_hh_track4_a30_literal_rerun.csv`

| branch | status | completed_runs | running_tasks | pending_tasks | failed_tasks | comparability_class | note |
|---|---|---:|---:|---:|---:|---|---|
| `bayesflow` | `completed` | 9 | 0 | 0 | 0 | `assumption_conditioned_aligned_multicurrent_a30` | decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions |
| `swyft` | `completed` | 3 | 0 | 0 | 0 | `assumption_conditioned_aligned_multicurrent_a30` | decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions |
| `tmnre` | `completed` | 4 | 0 | 0 | 0 | `assumption_conditioned_aligned_multicurrent_a30` | decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions |
| `poolseq_sbi` | `completed` | 2 | 0 | 0 | 0 | `assumption_conditioned_poolseq_sbi_a30` | mean, median, selected, and medoid rules are locally derivable from campaign-stored posterior outputs |
| `active_sequential` | `completed` | 10 | 0 | 0 | 0 | `assumption_conditioned_surrogate_active_design_a30` |  |
| `asnpe` | `completed` | 4 | 0 | 0 | 0 | `assumption_conditioned_surrogate_active_design_a30` |  |
| `simformer` | `completed` | 2 | 0 | 0 | 0 | `upstream_hh_benchmark_not_directly_comparable` | official upstream HH benchmark, not directly comparable to local aligned or surrogate rows |
| `generalized_bayes_npe` | `completed` | 1 | 0 | 0 | 0 | `assumption_conditioned_local_adaptation_not_exact_paper_code` | bounded local beta-conditioned adaptation, which should not be reported as the exact paper implementation |
| `temporal_deterministic` | `completed` | 1 | 0 | 0 | 0 | `assumption_conditioned_aligned_multicurrent_a30` |  |
