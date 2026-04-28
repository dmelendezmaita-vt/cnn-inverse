# HH Track4 ASNPE Branch 20260426

## Scope

This branch implements a bounded ASNPE-style posterior-estimation loop over the assumption-conditioned compact HH simulator, following the paper's Algorithm 1 structure, in which proposal updates are sequential, candidate acquisition is performed over proposal-sampled parameter pools, and the posterior estimator is retrained after each round on the accumulated simulation dataset.

## Approximation to p(phi|D)

The intractable Bayesian posterior over NDE parameters, `p(phi|D)`, is approximated by an equally weighted bootstrap deep ensemble of conditional diagonal-Gaussian neural posterior estimators. Each ensemble member is independently initialized, trained on a bootstrap resample of the accumulated dataset `D`, and optimized with a proposal-corrected weighted negative log likelihood, where sample weights are proportional to the inverse proposal density at the time the sample was acquired, then clipped and renormalized for stability.

The acquisition function used in the implementation is

`alpha(theta) = q_bar(theta | x0) * Var_k[q_phi_k(theta | x0)]`,

where `q_bar` is the equally weighted mixture over ensemble members, and `Var_k` is the empirical variance over ensemble-member densities evaluated at the candidate parameter.

## Queue Artifacts

- backlog: `tables/hh_track4_asnpe_backlog_20260426.json`
- registry: `tables/hh_track4_asnpe_registry_20260426.csv`

## Launches

Two real ASNPE runs were launched through the existing backlog mechanism:

1. `acdhh_14001`, label `asnpe_style_loop_primary_v100`
2. `acdhh_14002`, label `asnpe_style_loop_secondary_v100`

Both runs targeted Falcon allocation `368063`, and both completed successfully through the existing queue runner.

## Completed Run Artifacts

1. `runs/active_sequential/asnpe_primary_v100_20260426/asnpe_manifest.json`
2. `runs/active_sequential/asnpe_secondary_v100_20260426/asnpe_manifest.json`

## Aggregate Metrics

| run | mean_abs_log10_error_params_mean | mean_relative_error_params_mean | posterior_std_norm_mean_mean | runtime_sec |
|---|---:|---:|---:|---:|
| `primary_v100` | `0.754448` | `20.834720` | `0.968478` | `12.793871` |
| `secondary_v100` | `0.915400` | `26.649282` | `0.631387` | `12.448345` |
