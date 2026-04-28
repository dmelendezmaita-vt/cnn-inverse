# First Track (Paper-Parity, FHN) Summary — 2026-03-07

## What we did
- Matched data to the publication: verified SHA256 equality between the runtime tar (`/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar`) and the archived paper data (`.../data/2020-12-09`).
- Matched configs to Table 1/2 of the paper: Ntrain=1000, Nvalidate=2000, Ntest=2000, batch=32, Adam lr=0.002, swish ConvNet [8,16,32]+[32,32], epochs=200 (noise-free), using `pytorch/configs/params_dnn_tar.yaml`.
- Re-ran the isolated benchmark once for ground truth: job 242036 (`benchmark_records/job_242036_20260304_194449/runs_dnn`) under the paper-matched config.
- Ran cluster parity jobs on Falcon V100: jobs 234717/234718/234719 (1n/2n/4n) with the same config and dataset layout.
- Assessed robustness with multiple seeds: 5 seeds per node count (15 runs) tracked in `v100_paper_data_parity_training_jobs.csv`; metrics extracted into `v100_paper_data_parity_analysis_jobs.csv` and summarized in `..._analysis_summary.csv` and `..._analysis_detailed.md`.
- Verified completion: all 15 training runs `COMPLETED`; no pending/failed entries in the first-track folder.
- Captured reproducibility: snapshots `paper_data_parity_reproducibility_snapshot_*` store code/config/data at the validated state.

## Why we did it
- Establish fidelity to the published FHN setup before extending to other tracks (HH, scalability).
- Provide a trusted benchmark and cluster comparison for later cross-node/GPU analyses.
- Quantify seed-level stability and ensure distributed execution does not degrade quality.
- Maintain provenance so future changes can be audited or reproduced quickly.

## Key findings
- Cluster runs stay within <0.5σ shifts across 1n/2n/4n; quality close to benchmark.
- Test MSE/MAE/R2 deltas vs benchmark are small (see `paper_match_benchmark_and_v100_cluster_rerun_analysis_20260302.md`).
- Scaling up nodes did not hurt model quality in this setup; throughput remained cluster-bound, not math-bound.

## Pointers
- Benchmark artifacts: `.../benchmark_records/job_242036_20260304_194449/runs_dnn`
- Cluster runs: `/projects/neuro-collab/data/runs/234717`, `/projects/neuro-collab/data/runs/234718`, `/projects/neuro-collab/data/runs/234719`
- Tracking/analysis: `v100_paper_data_parity_training_jobs.csv`, `v100_paper_data_parity_analysis_jobs.csv`, `v100_paper_data_parity_analysis_detailed.md`
