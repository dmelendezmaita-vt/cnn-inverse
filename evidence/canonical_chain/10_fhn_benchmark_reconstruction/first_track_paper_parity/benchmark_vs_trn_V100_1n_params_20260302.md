# Benchmark vs Tracked `trn_V100_1n` Parameters

Date: 2026-03-02

Compared runs:

- Original-code benchmark: job `233768` (`orig_V100_bench`)
- Tracked CSV training run: job `233134` (`trn_V100_1n` in `training_jobs.csv`)

## 1) Requested scheduler resources

| Field | Benchmark `233768` | Tracked train `233134` |
|---|---|---|
| Cluster | Falcon | Falcon |
| GPU type | V100 | V100 |
| Nodes | 1 | 1 |
| GPUs per node | 1 | 2 |
| GPUs total | 1 | 2 |
| CPUs per node | 24 | 24 |
| CPUs total | 24 | 24 |
| RAM per node | 64 GiB | 64 GiB |
| Time limit | 04:00:00 | 1-00:00:00 |
| Partition | `v100_normal_q` | `v100_normal_q` |
| QoS | `fal_v100_normal_base` | `fal_v100_normal_base` |
| Data access mode | n/a (no `DATA_ACCESS_MODE` in original runner) | `direct` |

## 2) Model/data run parameters (from saved `params.yaml`)

### Shared core settings

- `net.type: ConvNet`
- `net.conv_layer_sizes: [8, 16, 32]`
- `net.dense_layer_sizes: [128, 128, 128, 128]`
- `optimizer.type: AdamW`
- `learning_rate: 0.01`
- `training.epochs: 400`
- `Ntrain/Nvalidate/Ntest: 1024 / 1000 / 2000`

### Key differences

| Parameter area | Benchmark `233768` (original code) | Tracked train `233134` |
|---|---|---|
| Config file | `configs/params_dnn.yaml` | `pytorch/configs/params_dnn_tar.yaml` |
| Data source | `../data/2020-12-09` | `/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar` |
| Dataset layout | legacy file layout | `tar_singlefile_split` |
| Feature type | `TIME_NOISE` | `TIME` |
| Target dimensionality | ODE targets (2 params in output layer) | ODE targets with `targets_cols_num: 6` |
| Extra tar fields | none | `data_prefix`, `curr`, `file_names`, `features_cols_num`, `targets_cols_num`, `use_mmap` |
| Checkpoint frequency | `save_checkpoints_epochs: 50` | `save_checkpoints_epochs: 1` |
| Eval random sub-begin | not present | `features_sub_begin_random_eval: false` |

## 3) Locations after move to `important_notes`

- Moved isolated original code + benchmark bundle: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/fhn_dnn_original_unmodified_20260302_050951`
- Benchmark record: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/fhn_dnn_original_unmodified_20260302_050951/benchmark_records/original_code_v100_benchmark_233768.md`
- Benchmark index: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/fhn_dnn_original_unmodified_20260302_050951/benchmark_records/benchmark_index.csv`
