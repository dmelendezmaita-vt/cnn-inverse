# V100 1n/2n/4n vs Original Benchmark (Comparable Methodology)

Date: 2026-03-02 15:58:56 

## Scope

- Benchmark baseline: original unmodified run `233768` (`orig_V100_bench`).
- Compared training runs: `233844` (V100_1n), `233845` (V100_2n), `233846` (V100_4n).
- All comparisons use the same formulas and data sources.

## Methodology (applied identically to 1n/2n/4n)

1. Accuracy metrics are read from each run `metrics_summary.csv` and compared to benchmark `run_dnn_info.log` for train/validate/test: MSE, MAE, R2.
2. Loss-tail robustness is compared using `loss.txt`: final loss, mean of last 20 epochs, mean of last 50 epochs.
3. Runtime is compared at two levels:
   - Inner training/eval runtime from run logs (`Runtime - train [sec]`, `Runtime - eval [sec]`).
   - Scheduler elapsed and queue wait from `sacct` (`Elapsed`, `Start-Submit`).
4. Throughput is normalized to publication-equivalent optimization updates:
   - Expected global updates = `epochs * (Ntrain / effective_global_train_batch_size)` = 1600 updates in all cases.
   - Normalized global updates/sec = `1600 / scheduler_elapsed_seconds`.
   - Normalized global samples/sec = `(epochs * Ntrain) / scheduler_elapsed_seconds`.
5. Config parity is checked against benchmark `params.yaml` for core training/architecture knobs.

## Config Parity Check (Core Parameters)

| Parameter | Benchmark | V100_1n | V100_2n | V100_4n |
|---|---:|---:|---:|---:|
| data.features_type | `TIME_NOISE` | `TIME_NOISE` | `TIME_NOISE` | `TIME_NOISE` |
| data.targets_type | `ODE` | `ODE` | `ODE` | `ODE` |
| data.targets_normalize | `False` | `False` | `False` | `False` |
| data.Ntrain | `1024` | `1024` | `1024` | `1024` |
| data.Nvalidate | `1000` | `1000` | `1000` | `1000` |
| data.Ntest | `2000` | `2000` | `2000` | `2000` |
| data.train_batch_size | `256` | `256` | `256` | `256` |
| data.eval_batch_size | `256` | `256` | `256` | `256` |
| optimizer.learning_rate | `0.01` | `0.01` | `0.01` | `0.01` |
| optimizer.type | `AdamW` | `AdamW` | `AdamW` | `AdamW` |
| optimizer.weight_decay | `0.01` | `0.01` | `0.01` | `0.01` |
| training.epochs | `400` | `400` | `400` | `400` |
| runconfig.save_checkpoints_epochs | `50` | `50` | `50` | `50` |
| net.type | `ConvNet` | `ConvNet` | `ConvNet` | `ConvNet` |
| net.conv_layer_sizes | `[8, 16, 32]` | `[8, 16, 32]` | `[8, 16, 32]` | `[8, 16, 32]` |
| net.dense_layer_sizes | `[128, 128, 128, 128]` | `[128, 128, 128, 128]` | `[128, 128, 128, 128]` | `[128, 128, 128, 128]` |

Additional run-path differences (non-core math) for all three training runs: tar-based data path and `tar_singlefile_split` layout with explicit `file_names` and `data_prefix`.

## Resource and Runtime Summary

| Case | Job ID | Nodes | GPUs total | Queue wait (s) | Elapsed (s) | Elapsed vs bench | Global updates/sec (norm) | Global samples/sec (norm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V100_1n | 233844 | 1 | 2 | 1 | 247 | +488.10% | 6.478 (-83.00%) | 1658.3 (-83.00%) |
| V100_2n | 233845 | 2 | 4 | 1 | 254 | +504.76% | 6.299 (-83.46%) | 1612.6 (-83.46%) |
| V100_4n | 233846 | 4 | 8 | 3 | 257 | +511.90% | 6.226 (-83.66%) | 1593.8 (-83.66%) |

Benchmark reference: elapsed `42s`, queue wait `1s`, normalized updates/sec `38.095`, normalized samples/sec `9752.4`.

## Accuracy Comparison vs Benchmark

### Train

| Case | MSE | ΔMSE vs bench | MAE | ΔMAE vs bench | R2 | ΔR2 vs bench |
|---|---:|---:|---:|---:|---:|---:|
| V100_1n | 0.00072480388917 | -2.16% | 0.0145810395479 | +0.19% | 0.992294788361 | +0.000162 |
| V100_2n | 0.000718568975572 | -3.00% | 0.0145813785493 | +0.20% | 0.992348074913 | +0.000215 |
| V100_4n | 0.000704753212631 | -4.86% | 0.0144017459825 | -1.04% | 0.992522656918 | +0.000390 |

### Validate

| Case | MSE | ΔMSE vs bench | MAE | ΔMAE vs bench | R2 | ΔR2 vs bench |
|---|---:|---:|---:|---:|---:|---:|
| V100_1n | 0.00241823517717 | -0.64% | 0.0288021937013 | -0.72% | 0.975016236305 | +0.000052 |
| V100_2n | 0.00244706496596 | +0.55% | 0.0289631597698 | -0.17% | 0.974796175957 | -0.000168 |
| V100_4n | 0.00244533433579 | +0.47% | 0.0286990739405 | -1.08% | 0.974796772003 | -0.000167 |

### Test

| Case | MSE | ΔMSE vs bench | MAE | ΔMAE vs bench | R2 | ΔR2 vs bench |
|---|---:|---:|---:|---:|---:|---:|
| V100_1n | 0.00262781744823 | +7.49% | 0.0296968556941 | +3.37% | 0.973934650421 | -0.000924 |
| V100_2n | 0.00258467113599 | +5.72% | 0.0296485349536 | +3.20% | 0.974591672421 | -0.000267 |
| V100_4n | 0.00255890982226 | +4.67% | 0.0295343380421 | +2.80% | 0.974576830864 | -0.000282 |

## Loss Tail Comparison vs Benchmark

| Case | Final loss | Δ final vs bench | Last-20 mean | Δ last-20 vs bench | Last-50 mean | Δ last-50 vs bench |
|---|---:|---:|---:|---:|---:|---:|
| V100_1n | 0.000736439280445 | +0.65% | 0.000710152789907 | -1.35% | 0.000717492294061 | -0.64% |
| V100_2n | 0.000697878582287 | -4.62% | 0.000711023732583 | -1.23% | 0.000713664755749 | -1.17% |
| V100_4n | 0.000709950269083 | -2.97% | 0.000727125294725 | +1.01% | 0.000731856695929 | +1.35% |

## Inner Runtime (from logs)

| Case | Train sec | Δ train vs bench | Eval sec | Δ eval vs bench | Effective global batch | World size | Per-rank batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| V100_1n | 9.193321 | -6.62% | 0.664519 | +117.87% | 256 | 2 | 128 |
| V100_2n | 11.093101 | +12.67% | 0.576716 | +89.08% | 256 | 4 | 64 |
| V100_4n | 12.171099 | +23.62% | 0.608708 | +99.57% | 256 | 8 | 32 |

Benchmark: train `9.845294s`, eval `0.305007s`, effective global batch `256`, world size `1`, per-rank batch `256`.

## Findings

1. Core training/architecture parameters are aligned across benchmark and all three V100 runs; methodology is comparable by design.
2. Best test accuracy among V100 node counts is: MSE `V100_4n`, MAE `V100_4n`, R2 `V100_2n` (same or near-same case expected due deterministic-ish setup).
3. Best scheduler elapsed among V100 runs is `V100_1n` (`247s`), but all are much slower than benchmark `42s`.
4. Scaling from 1n -> 2n -> 4n does not improve wall-clock for this workload; elapsed slightly increases while normalized throughput decreases.
5. Across all three V100 cases, generalization on test split remains close to benchmark but typically slightly worse in MSE/MAE/R2 than the original benchmark run.

## Source Artifacts

- Benchmark run dir: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/fhn_dnn_original_unmodified_20260302_050951/benchmark_records/job_233768_20260302_051033/runs_dnn`
- V100_1n: `/projects/neuro-collab/data/runs/233844`
- V100_2n: `/projects/neuro-collab/data/runs/233845`
- V100_4n: `/projects/neuro-collab/data/runs/233846`

---
Generated automatically from run artifacts and sacct records.
