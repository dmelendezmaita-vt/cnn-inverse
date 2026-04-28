# Session Notes: Paper-Data Parity Rename and Verification

Date: 2026-03-04 (EST)

## 1) Goal
Rename V100 parity-tracking artifacts so naming clearly indicates these runs are based on the same dataset/protocol family as the paper, then verify data identity against the paper dataset.

## 2) Renames Performed

### 2.1 Important Notes files

| Old name | New name |
|---|---|
| `v100_paper_equivalence_training_jobs.csv` | `v100_paper_data_parity_training_jobs.csv` |
| `v100_paper_equivalence_analysis_jobs.csv` | `v100_paper_data_parity_analysis_jobs.csv` |
| `v100_paper_equivalence_analysis_summary.csv` | `v100_paper_data_parity_analysis_summary.csv` |
| `v100_paper_equivalence_analysis.md` | `v100_paper_data_parity_analysis.md` |
| `v100_paper_equivalence_analysis_detailed.md` | `v100_paper_data_parity_analysis_detailed.md` |
| `v100_paper_equivalence_plan.txt` | `v100_paper_data_parity_plan.txt` |

### 2.2 Script rename

| Old name | New name |
|---|---|
| `scripts/update_v100_equivalence_csv.sh` | `scripts/update_v100_paper_data_parity_csv.sh` |

### 2.3 Reference updates
- Updated internal references in renamed files/script to the new names.
- Updated suite label in CSV from `paper_equivalence_v100` to `paper_data_parity_v100`.
- Verified no stale references remain to old names.

## 3) Verification: Are we using the same data as the paper?

### 3.1 Paper reference data location (archived original code bundle)
`/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/fhn_dnn_original_unmodified_20260302_050951/fhn_dnn-1-implementation-in-pytorch/data/2020-12-09`

Contains (among others):
- `fhn_T200_samplePrior_state0.npy`
- `fhn_T200_samplePrior_theta.npy`
- `noise_correlated_Nt1000_Nsim10000_data.npy`

### 3.2 Current runtime dataset package
`/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar`

Tar entries include exactly:
- `fhn_T200_samplePrior_state0.npy`
- `fhn_T200_samplePrior_theta.npy`
- `noise_correlated_Nt1000_Nsim10000_data.npy`

### 3.3 SHA256 identity check (paper data file vs tar member bytes)

| File | SHA256 in paper data dir | SHA256 in tar stream | Match |
|---|---|---|---|
| `fhn_T200_samplePrior_state0.npy` | `82fbc382bb84c53b5aaf2fef4bc3fbffab6cbe497675fd74e63f4420f357fff5` | `82fbc382bb84c53b5aaf2fef4bc3fbffab6cbe497675fd74e63f4420f357fff5` | yes |
| `fhn_T200_samplePrior_theta.npy` | `e1061afc5dc50b5bd1d1e773a7d83044347d47802d10899ec22bbc44cda4af86` | `e1061afc5dc50b5bd1d1e773a7d83044347d47802d10899ec22bbc44cda4af86` | yes |
| `noise_correlated_Nt1000_Nsim10000_data.npy` | `d5ff5ffbd38112edcd2c40255ef8f18da8411e95c1719afb6e338bca27eb321c` | `d5ff5ffbd38112edcd2c40255ef8f18da8411e95c1719afb6e338bca27eb321c` | yes |

### 3.4 Config-level parity checks
Compared paper-matched archived config (`params_dnn_paper_match.yaml`) with active tar-based config (`params_dnn_tar.yaml`) for key scientific settings:
- `Ntrain/Nvalidate/Ntest`: match
- `train_batch_size/eval_batch_size`: match
- `random_seed`: match (`123` baseline)
- model architecture: match (`ConvNet`, `[8,16,32]`, dense `[32,32]`, `swish`)
- optimizer and LR: match (`Adam`, `0.002`)
- `epochs`: match (`200`)

Note:
- Archived paper config does not explicitly list `data.file_names` entries, while tar config does.
- The checksum validation above confirms the tar members are byte-identical to the canonical paper data files.

## 4) Operational Verification
- Ran renamed updater script successfully:
  - `scripts/update_v100_paper_data_parity_csv.sh`
  - Updated file: `data/important_notes/v100_paper_data_parity_training_jobs.csv`

## 5) Outcome
- Naming now clearly indicates paper-data parity.
- Dataset identity with the paper source files has been verified via SHA256 checks.
- Session changes are documented and reproducible.
