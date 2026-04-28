# V100 Paper-Equivalence Analysis (Detailed)

## Scope
This report evaluates whether distributed training on Falcon V100 GPUs is statistically consistent across 1-node, 2-node, and 4-node runs when using the same paper-matched configuration and multiple random seeds.

## Data Sources
- Suite tracking CSV: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_training_jobs.csv`
- Per-job extracted metrics: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_analysis_jobs.csv`
- Node-level summary statistics: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_analysis_summary.csv`

## Experimental Design
- Cluster: Falcon
- GPU: V100
- Node counts: 1, 2, 4
- Seeds: 301, 302, 303, 304, 305
- Runs per node count: 5
- Total runs: 15
- Fixed resources per node: 2 GPUs, 24 CPUs, 64 GiB RAM
- QoS/partition: `fal_v100_normal_base` / `v100_normal_q`
- Data mode: `copy_to_node`

## Run Completion Status

| Status | Count |
|---|---:|
| COMPLETED | 15 |

All 15 runs reached `COMPLETED` state.

## Per-Job Outcomes

| Job ID | Job Name | Nodes | Seed | Status | Elapsed | Train Last | Val Last | Train Best | Val Best | Test MSE | Test MAE | Test R2 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 242167 | trn_V100_1n_eq301 | 1 | 301 | COMPLETED | 00:04:18 | 0.002098569 | 0.000901909 | 0.002098569 | 0.000628386 | 0.003069423 | 0.036326583 | 0.971175134 |
| 242168 | trn_V100_1n_eq302 | 1 | 302 | COMPLETED | 00:04:08 | 0.002136673 | 0.001049995 | 0.001865898 | 0.000601214 | 0.002737954 | 0.034814894 | 0.973718286 |
| 242169 | trn_V100_1n_eq303 | 1 | 303 | COMPLETED | 00:04:16 | 0.002301315 | 0.000990479 | 0.002028413 | 0.000651741 | 0.003007117 | 0.036545515 | 0.971564829 |
| 242170 | trn_V100_1n_eq304 | 1 | 304 | COMPLETED | 00:04:14 | 0.002544691 | 0.001325629 | 0.002074085 | 0.000701281 | 0.002969543 | 0.035705663 | 0.972410142 |
| 242171 | trn_V100_1n_eq305 | 1 | 305 | COMPLETED | 00:04:18 | 0.002212914 | 0.000733094 | 0.002139667 | 0.000549747 | 0.003163727 | 0.038244054 | 0.969854236 |
| 242172 | trn_V100_2n_eq301 | 2 | 301 | COMPLETED | 00:04:18 | 0.002212280 | 0.001191033 | 0.001955812 | 0.000640230 | 0.003019909 | 0.035865299 | 0.971674919 |
| 242173 | trn_V100_2n_eq302 | 2 | 302 | COMPLETED | 00:04:24 | 0.001954907 | 0.000841196 | 0.001882610 | 0.000649430 | 0.002735469 | 0.034674071 | 0.973808765 |
| 242174 | trn_V100_2n_eq303 | 2 | 303 | COMPLETED | 00:04:12 | 0.002309281 | 0.000669126 | 0.002088900 | 0.000669126 | 0.003029472 | 0.036874600 | 0.971372604 |
| 242175 | trn_V100_2n_eq304 | 2 | 304 | COMPLETED | 00:04:19 | 0.002159016 | 0.000914529 | 0.002143431 | 0.000620459 | 0.002956972 | 0.035617992 | 0.972535014 |
| 242176 | trn_V100_2n_eq305 | 2 | 305 | COMPLETED | 00:04:22 | 0.002276031 | 0.000923009 | 0.002124328 | 0.000616807 | 0.003161765 | 0.038118206 | 0.969894409 |
| 242177 | trn_V100_4n_eq301 | 4 | 301 | COMPLETED | 00:04:31 | 0.002305329 | 0.001137587 | 0.002126134 | 0.000687892 | 0.003070614 | 0.036120430 | 0.971104622 |
| 242178 | trn_V100_4n_eq302 | 4 | 302 | COMPLETED | 00:04:33 | 0.002026710 | 0.001068418 | 0.001898207 | 0.000607779 | 0.002761901 | 0.035187900 | 0.973537743 |
| 242179 | trn_V100_4n_eq303 | 4 | 303 | COMPLETED | 00:04:25 | 0.002181630 | 0.001027914 | 0.002029312 | 0.000654094 | 0.002969557 | 0.036592912 | 0.971924722 |
| 242180 | trn_V100_4n_eq304 | 4 | 304 | COMPLETED | 00:04:22 | 0.002260720 | 0.000939028 | 0.002117920 | 0.000695711 | 0.002964078 | 0.035593968 | 0.972511709 |
| 242181 | trn_V100_4n_eq305 | 4 | 305 | COMPLETED | 00:04:31 | 0.002376784 | 0.001081796 | 0.002098145 | 0.000752939 | 0.003160992 | 0.038023122 | 0.969917297 |

## Node-Level Summary (Mean ± Std Across 5 Seeds)

| Nodes | Train Last | Val Last | Train Best | Val Best | Test MSE | Test MAE | Test R2 | Avg Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002258832 ± 0.000177648 | 0.001000221 ± 0.000217649 | 0.002041326 ± 0.000106022 | 0.000626474 ± 0.000056488 | 0.002989553 ± 0.000158688 | 0.036327342 ± 0.001264453 | 0.971744525 ± 0.001437747 | 00:04:15 |
| 2 | 0.002182303 ± 0.000139694 | 0.000907779 ± 0.000188347 | 0.002039016 ± 0.000114077 | 0.000639211 ± 0.000021529 | 0.002980718 ± 0.000156046 | 0.036230034 ± 0.001314278 | 0.971857142 ± 0.001448158 | 00:04:19 |
| 4 | 0.002230235 ± 0.000134006 | 0.001050949 ± 0.000073857 | 0.002053943 ± 0.000095025 | 0.000679683 ± 0.000053656 | 0.002985428 ± 0.000148956 | 0.036303666 ± 0.001097995 | 0.971799219 ± 0.001375677 | 00:04:28 |

## Relative Change vs 1-Node Mean (%)

| Nodes | Train Last | Val Last | Train Best | Val Best | Test MSE | Test MAE | Test R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| 2 | -3.39 | -9.24 | -0.11 | 2.03 | -0.30 | -0.27 | 0.0116 |
| 4 | -1.27 | 5.07 | 0.62 | 8.49 | -0.14 | -0.07 | 0.0056 |

## Consistency Check (Mean-Shift / Pooled-Std, sigma units)

| Nodes vs 1n | Train Last | Val Last | Test MSE | Test MAE |
|---|---:|---:|---:|---:|
| 2n vs 1n | 0.48 | 0.46 | 0.06 | 0.08 |
| 4n vs 1n | 0.18 | 0.35 | 0.03 | 0.02 |

Interpretation: all shifts are below 0.5 sigma for the reported metrics, indicating no meaningful degradation from 1n to 2n/4n within observed seed variability.

## Conclusions
1. The full suite completed successfully (15/15).
2. Quality metrics (`test_mse`, `test_mae`, `test_r2`) are statistically consistent across 1n, 2n, and 4n.
3. Observed cross-node differences are small compared to run-to-run seed variance.
4. For this V100 paper-matched setup, distributed scaling (up to 4 nodes) preserves model quality.

## Limitations and Follow-Ups
1. This analysis is specific to Falcon V100 and this exact configuration.
2. Equivalent multi-seed studies should be repeated for A30/L40S/A100/H200 if cross-GPU equivalence is required.
3. If strict paper replication is needed, keep using the same reporting rule consistently (final epoch and/or best-checkpoint selection).
