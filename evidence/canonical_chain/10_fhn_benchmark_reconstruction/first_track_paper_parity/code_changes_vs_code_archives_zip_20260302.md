# Code Changes vs `/code/archives` Zip Baseline

Date: 2026-03-02

## Baseline used

- Archive: `/projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip`
- Extracted baseline path: `/tmp/fhn_zip_baseline.mataPv/fhn_dnn-1-implementation-in-pytorch`
- Comparison target: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch`
- Method: file-level diff (`added`, `modified`, `deleted`) and content diffs with `git diff --no-index`.
- Exclusions for classification: `__pycache__/`, `*.pyc`.

## Backup created before redo

- Backup archive: `/projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_20260302_034513.tar`
- Sidecars:
- `/projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_20260302_034513.tar.sha256`
- `/projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_20260302_034513.tar.files.sha256`
- `/projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_20260302_034513.tar.filelist.txt`
- `/projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_20260302_034513.tar.manifest.txt`

## High-level result

- Added files: `17`
- Modified files: `5`
- Deleted files: `0`

## Modified files (present in baseline and current)

### `README.md`

- Added a new **Backup and Restore** section with quickstart commands.
- Added links to new docs:
- `docs/backup_restore.md`
- `docs/restore_runbook.md`
- `docs/notification_formats.md`

### `pytorch/configs/params_dnn.yaml`

- Added `features_sub_begin_random_eval: false` in `data` config.

### `pytorch/data.py`

- Large refactor/extension (net +622/-203 lines).
- Added tar-aware loading support:
- `_resolve_tar_member_name(...)`
- `_load_array_from_tar(...)`
- `_load_array_maybe_tar(...)`
- Added explicit support for dataset layout `tar_singlefile_split`.
- Added safer handling for raw binary payloads and shape inference errors.
- Added mmap behavior guardrails for tar-stream based loading.
- Expanded split/filter logic (`Nvalidate`, optional target filtering, weighted/random split handling).
- Added timesteps loading path support for tar layouts.

### `pytorch/run_dnn.py`

- Major runtime update (net +557/-305 lines).
- Added distributed execution support:
- `torch.distributed`
- DDP wrapping
- env detection for `torchrun` and SLURM (`WORLD_SIZE`, `RANK`, `SLURM_*`)
- Added per-rank logging directories and rank-aware seeding.
- Added signal handlers for `SIGUSR1`/`SIGTERM` to save interrupt checkpoints.
- Added distributed-aware dataloader/sampler and rank synchronization points.
- Added CLI overrides and run-directory handling improvements (`--data_dir`, `--data_prefix`, `--save_dir_base`, SLURM job ID based output).
- Added explicit script entrypoint guard (`if __name__ == "__main__":`).

### `utils/utils.py`

- Updated `Mode.any(...)` to use bitwise logic (`self & modes`) with compatibility fallback.

## Added files (not present in zip baseline)

- `data/important_notes/tables_validation.txt`
- `data/important_notes/testing_jobs.csv`
- `data/important_notes/training_jobs.csv`
- `docs/backup_restore.md`
- `docs/notification_formats.md`
- `docs/restore_runbook.md`
- `fix-tracking/2026-02-28-falcon-orchestration-fixes.md`
- `fix-tracking/README.md`
- `pytorch/configs/params_dnn_tar.yaml`
- `pytorch/configs/params_dnn_tar_smoke.yaml`
- `scripts/backup_project.sh`
- `scripts/restore_project.sh`
- `scripts/submit_l40s_scaling_smoke.sh`
- `slurm_smoke_test.sbatch`
- `slurm_smoke_test_direct.sbatch`
- `slurm_train.sbatch`
- `slurm_train_direct.sbatch`

## Added-file intent summary

- New Slurm orchestration stack for smoke/train runs, including direct and copy-to-node data-access workflows.
- New backup/restore automation with checksum/manifests and runbooks.
- New tar-specific PyTorch configs for archive-backed datasets.
- New ops tracking docs and important-notes job tables.

