# HH Track4 A30 Literature and Surrogate Refresh

- campaign root: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun`
- discovered later-branch runs: `43`

## Scope

This refresh is restricted to the later aligned literature and surrogate branches under the supplied A30 campaign root, namely the BayesFlow, Swyft, TMNRE, pool-sequential SBI, surrogate active-sequential, ASNPE, Simformer, and generalized-Bayes branches, while the earlier direct A30 classical refresh remains out of scope for this script.

## Coverage

| branch | status | completed_runs | note |
|---|---|---:|---|
| `bayesflow` | `completed` | 9 | decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions |
| `swyft` | `completed` | 3 | decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions |
| `tmnre` | `completed` | 4 | decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions |
| `poolseq_sbi` | `completed` | 2 | mean, median, selected, and medoid rules are locally derivable from campaign-stored posterior outputs |
| `active_sequential` | `completed` | 10 |  |
| `asnpe` | `completed` | 4 |  |
| `simformer` | `completed` | 2 | official upstream HH benchmark, not directly comparable to local aligned or surrogate rows |
| `generalized_bayes_npe` | `completed` | 1 | bounded local beta-conditioned adaptation, which should not be reported as the exact paper implementation |
| `temporal_deterministic` | `completed` | 1 |  |

## Key Tables

| table | rows | interpretation |
|---|---:|---|
| framework snapshot | 17 | aligned framework-level snapshot from campaign-local metrics_summary artifacts |
| posterior decision rules | 63 | campaign-local rule rows, which include explicit rule files plus mean rows that can be derived directly from raw predictions |
| active policy comparison | 14 | surrogate active-sequential and ASNPE policy rows |
| normalized method comparison | 100 | direct comparison table with explicit comparability classes |

## Best Current Rows

| result space | best row | primary metric | note |
|---|---|---:|---|
| aligned framework snapshot | `a30runall__bayesflow_native_meanstd_v100_20260425` | 595.630310 | `bayesflow_native` |
| posterior decision rules | `a30lit__simformer_official_hh_primary_20260426::mean` | 7.624424 | `simformer` |
| heuristic active policy | `a30surr__active_sequential_wasserstein_smoke_20260425` | 0.464670 | `wasserstein` |
| ASNPE | `a30surr__asnpe_extended_v100_20260428` | 0.655782 | `asnpe_style` |
| Simformer upstream HH | `a30lit__simformer_official_hh_primary_20260426` | 0.963940 | `secondary metric is r2` |
| generalized-Bayes beta sweep | `a30lit__generalized_bayes_npe_main_20260426::beta_0.25` | 2.221421 | `beta=0.25` |

## Limitations

| id | scope | severity | statement |
|---|---|---|---|
| `L1` | `comparability` | `high` | Do not rank aligned BayesFlow, Swyft, TMNRE, pool-sequential SBI, surrogate active-design, Simformer upstream HH, and generalized-Bayes adaptation rows in one undifferentiated table, because they target different observation contracts and posterior objectives. |
| `L2` | `decision_rules` | `high` | For normalized BayesFlow and Swyft posterior runs, this refresh can reconstruct mean rows directly from local predictions_test artifacts, while median, medoid, and map-bank rows are regenerated only when explicit predictions_* files are present under the campaign root. |
| `L3` | `active_design` | `high` | Active sequential and ASNPE rows remain assumption-conditioned surrogate branches, so they should be reported as active-design evidence rather than as direct Track4 frontier claims. |
| `L4` | `simformer` | `high` | Simformer rows remain upstream HH benchmark references, which are methodologically useful but not numerically comparable to the local aligned or surrogate Track4-style workloads. |
| `L5` | `generalized_bayes` | `medium` | The generalized-Bayes NPE branch is a bounded local beta-conditioned adaptation, which should be documented as such, because it is not the authors' exact released workflow. |
