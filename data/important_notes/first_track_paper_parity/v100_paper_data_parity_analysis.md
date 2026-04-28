# V100 Paper Equivalence Analysis

- Input CSV: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_training_jobs.csv`
- Per-job metrics CSV: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_analysis_jobs.csv`
- Summary CSV: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_analysis_summary.csv`
- Jobs analyzed: `15`
- Missing artifacts count: `0`

## Node-level Means (5 seeds each)

| nodes | train_last | val_last | train_best | val_best | test_mse | test_mae | test_r2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002258832 | 0.001000221 | 0.002041326 | 0.000626474 | 0.002989553 | 0.036327342 | 0.971744525 |
| 2 | 0.002182303 | 0.000907779 | 0.002039016 | 0.000639211 | 0.002980718 | 0.036230034 | 0.971857142 |
| 4 | 0.002230235 | 0.001050949 | 0.002053943 | 0.000679683 | 0.002985428 | 0.036303666 | 0.971799219 |

## Relative Delta vs 1-node Mean (%)

| nodes | train_last | val_last | train_best | val_best | test_mse | test_mae | test_r2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | -3.39 | -9.24 | -0.11 | 2.03 | -0.30 | -0.27 | 0.0116 |
| 4 | -1.27 | 5.07 | 0.62 | 8.49 | -0.14 | -0.07 | 0.0056 |

## Variability (std across 5 seeds)

| nodes | train_last_std | val_last_std | test_mse_std | test_mae_std | test_r2_std |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000177648 | 0.000217649 | 0.000158688 | 0.001264453 | 0.001437747 |
| 2 | 0.000139694 | 0.000188347 | 0.000156046 | 0.001314278 | 0.001448158 |
| 4 | 0.000134006 | 0.000073857 | 0.000148956 | 0.001097995 | 0.001375677 |

## Consistency Check
- 2n vs 1n mean-shift ratios: train_last: 0.48σ, val_last: 0.46σ, test_mse: 0.06σ, test_mae: 0.08σ
- 4n vs 1n mean-shift ratios: train_last: 0.18σ, val_last: 0.35σ, test_mse: 0.03σ, test_mae: 0.02σ
