# Paper-Matched Re-run: Benchmark + V100 Cluster Implementations

Date: 2026-03-02 21:45:00

## Why we were not using N=2000 for training

- Table 1 in the paper uses `Ntrain=1000`, `Nvalidate=2000`, `Ntest=2000`.
- `N=2000` applies to validation/test, not train.
- Earlier configs were inherited defaults (`Ntrain=1024`, `Nvalidate=1000`, `Ntest=2000`), so they were not paper-matched.

## Final Match Status (Table 1/2)

| Item | Paper target | Final benchmark (`242036`) | Final cluster (`234717/234718/234719`) | Status |
|---|---|---|---|---|
| Ntrain | 1000 | 1000 | 1000 | matched |
| Nvalidate | 2000 | 2000 | 2000 | matched |
| Ntest | 2000 | 2000 | 2000 | matched |
| Batch size | 32 | 32 | global=32 | matched |
| Optimizer | Adam | Adam | Adam | matched |
| Learning rate | 0.002 | 0.002 | 0.002 | matched |
| Epochs (noise-free) | 200 | 200 | 200 | matched |
| Activation | swish | swish | swish | matched |
| Conv kernel/stride | 3 / 2 | 3 / 2 | 3 / 2 | matched |
| Conv filters | [8,16,32] | [8,16,32] | [8,16,32] | matched |
| AvgPool | kernel=2, stride=2 | yes (`net.txt`) | yes (`net.txt`) | matched |
| Dense head | [32,32] | [32,32] | [32,32] | matched |

## Job IDs and Runtime

| Case | Job ID | State | Queue wait (s) | Elapsed (s) | Nodes | GPUs total |
|---|---:|---|---:|---:|---:|---:|
| benchmark_paper_pool_fix | 242036 | COMPLETED | 1 | 31 | 1 | 1 |
| v100_1n_paper_pool | 234717 | COMPLETED | 273 | 258 | 1 | 2 |
| v100_2n_paper_pool | 234718 | COMPLETED | 273 | 267 | 2 | 4 |
| v100_4n_paper_pool | 234719 | COMPLETED | 0 | 269 | 4 | 8 |

## Benchmark Metrics (`242036`)

| Split | MSE | MAE | R2 |
|---|---:|---:|---:|
| train | 0.002500837668776512 | 0.032848943024873734 | 0.9767111539840698 |
| validate | 0.0027437591925263405 | 0.03488574177026749 | 0.973210334777832 |
| test | 0.0027437591925263405 | 0.03488574177026749 | 0.973210334777832 |

## Cluster vs Benchmark (same final config)

### Train

| Case | MSE | ΔMSE vs benchmark | MAE | ΔMAE vs benchmark | R2 | ΔR2 vs benchmark |
|---|---:|---:|---:|---:|---:|---:|
| v100_1n_paper_pool | 0.0024542659521102905 | -1.86% | 0.032382816076278687 | -1.42% | 0.9771553874015808 | +0.000444 |
| v100_2n_paper_pool | 0.002515696920454502 | +0.59% | 0.03295856714248657 | +0.33% | 0.9766370058059692 | -0.000074 |
| v100_4n_paper_pool | 0.002467602491378784 | -1.33% | 0.032553400844335556 | -0.90% | 0.9770311117172241 | +0.000320 |

### Validate

| Case | MSE | ΔMSE vs benchmark | MAE | ΔMAE vs benchmark | R2 | ΔR2 vs benchmark |
|---|---:|---:|---:|---:|---:|---:|
| v100_1n_paper_pool | 0.0027143508195877075 | -1.07% | 0.03475416451692581 | -0.38% | 0.9734803438186646 | +0.000270 |
| v100_2n_paper_pool | 0.002754855901002884 | +0.40% | 0.03488680720329285 | +0.00% | 0.9731142520904541 | -0.000096 |
| v100_4n_paper_pool | 0.002730301348492503 | -0.49% | 0.03491009399294853 | +0.07% | 0.9733633995056152 | +0.000153 |

### Test

| Case | MSE | ΔMSE vs benchmark | MAE | ΔMAE vs benchmark | R2 | ΔR2 vs benchmark |
|---|---:|---:|---:|---:|---:|---:|
| v100_1n_paper_pool | 0.003133655060082674 | +14.22% | 0.036608725786209106 | +4.94% | 0.9709117412567139 | -0.002299 |
| v100_2n_paper_pool | 0.003133385442197323 | +14.21% | 0.03669731318950653 | +5.19% | 0.9709341526031494 | -0.002276 |
| v100_4n_paper_pool | 0.003157856874167919 | +15.10% | 0.036835767328739166 | +5.59% | 0.9707349538803101 | -0.002475 |

## Throughput Normalization

- Expected updates: `epochs * floor(Ntrain/global_batch) = 200 * floor(1000/32) = 6200`.

| Case | Elapsed (s) | Normalized updates/s | Normalized samples/s |
|---|---:|---:|---:|
| benchmark_paper_pool_fix | 31 | 200.000 | 6400.0 |
| v100_1n_paper_pool | 258 | 24.031 | 769.0 |
| v100_2n_paper_pool | 267 | 23.221 | 743.1 |
| v100_4n_paper_pool | 269 | 23.048 | 737.5 |

## Verification (2026-03-07)

- Benchmark metrics pulled from `run_dnn_info.log` in `job_242036_20260304_194449/runs_dnn`.
- Cluster metrics pulled from `metrics_summary.csv` in `/projects/neuro-collab/data/runs/234717`, `/projects/neuro-collab/data/runs/234718`, `/projects/neuro-collab/data/runs/234719`.
- Status scan: `v100_paper_data_parity_training_jobs.csv` shows 15/15 `COMPLETED`; no PENDING/FAILED/CANCEL/RUNNING found under `first_track_paper_parity/`.
- All deltas in this note recomputed from those artifacts after switching benchmark ID to 242036.

## Notes

- `234716` (`orig_V100_bench_paper_pool`) is superseded. It was submitted before patching pooling support in the isolated benchmark `nets.py`, so it did not show AvgPool in `net.txt`.
- Final benchmark for comparisons is `242036`.

## Artifact Paths

- Final benchmark run: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/fhn_dnn_original_unmodified_20260302_050951/benchmark_records/job_242036_20260304_194449/runs_dnn`
- Cluster run dir 1n: `/projects/neuro-collab/data/runs/234717`
- Cluster run dir 2n: `/projects/neuro-collab/data/runs/234718`
- Cluster run dir 4n: `/projects/neuro-collab/data/runs/234719`
