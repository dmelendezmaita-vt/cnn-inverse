# Backup and Restore Guide

This guide describes the backup and restore tooling for the project at:

`/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch`

## Purpose

The scripts provide a repeatable way to:

1. Save critical code/config state to a timestamped archive.
2. Validate archive integrity with SHA-256 checksum files.
3. Restore project state safely, with optional pre-restore backup.

Scripts:

- `scripts/backup_project.sh`
- `scripts/restore_project.sh`

## Default Behavior

### Backup defaults

- Source: `/projects/neuro-collab`
- Destination: `/projects/neuro-collab/archives/fhn_dnn`
- Prefix: `fhn_dnn_backup`
- Format: `tar` (uncompressed)
- Scope: `key`

### Restore defaults

- Target: `/projects/neuro-collab`
- Pre-restore safety mode: `prompt`
- Pre-restore backup destination: `/projects/neuro-collab/archives/fhn_dnn`
- Pre-restore backup format: `tar`
- Pre-restore backup scope: `repo` (target-relative; does not depend on `--scope key` allowlist paths)

## What Gets Backed Up By Default (`--scope key`)

Included:

- `code/fhn_dnn-1-implementation-in-pytorch/pytorch/`
- `code/fhn_dnn-1-implementation-in-pytorch/tensorflow/`
- `code/fhn_dnn-1-implementation-in-pytorch/utils/`
- `code/fhn_dnn-1-implementation-in-pytorch/scripts/`
- `code/fhn_dnn-1-implementation-in-pytorch/docs/`
- `code/fhn_dnn-1-implementation-in-pytorch/slurm_train.sbatch`
- `code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch`
- `code/fhn_dnn-1-implementation-in-pytorch/README.md`
- `code/fhn_dnn-1-implementation-in-pytorch/WARP.md`
- `code/fhn_dnn-1-implementation-in-pytorch/LICENSE`
- `code/dl-kit-main/dlkit/opt/train.py`
- `code/dl-kit-main/dlkit/opt/train_utils.py`
- `DMM research progress.pdf`

Intentionally excluded by default:

- repository `data/`
- caches (`__pycache__`, `*.pyc`, etc.)
- runtime artifacts
- large dataset tarballs in `/projects/neuro-collab/data/tar_files`

## Backup Script Usage

## Command

```bash
bash scripts/backup_project.sh [OPTIONS]
```

## Options

- `--source PATH`: project root to back up.
- `--dest PATH`: output directory for archive and sidecars.
- `--prefix NAME`: archive filename prefix.
- `--format tar|tar.gz`: archive format.
- `--scope key|repo`: `key` allowlist or `repo` full repo minus excludes.
- `--dry-run`: show planned actions only.
- `-h, --help`: print built-in usage documentation.

## Output files

Given archive `<archive>`:

- `<archive>`: backup archive (`.tar` or `.tar.gz`)
- `<archive>.sha256`: archive checksum sidecar
- `<archive>.files.sha256`: per-file checksum sidecar for archived content
- `<archive>.filelist.txt`: exact relative file list packed in archive
- `<archive>.manifest.txt`: metadata manifest (options, source/dest, integrity artifacts)

During backup creation, integrity is verified by:

1. Generating per-file checksums from current source files.
2. Creating archive and verifying archive checksum.
3. Extracting archive to temporary staging.
4. Verifying extracted file list matches source file list exactly.
5. Verifying extracted file checksums match source file checksums exactly.

## Examples

Create a default backup:

```bash
cd /projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch
bash scripts/backup_project.sh
```

Create compressed backup:

```bash
bash scripts/backup_project.sh --format tar.gz
```

Preview backup actions without writing files:

```bash
bash scripts/backup_project.sh --dry-run
```

## Restore Script Usage

## Command

```bash
bash scripts/restore_project.sh --archive PATH [OPTIONS]
```

## Options

- `--archive PATH`: backup archive to restore from (required).
- `--target PATH`: restore destination.
- `--backup-current prompt|always|never`: pre-restore backup behavior.
- `--backup-dest PATH`: destination for pre-restore backup archives.
- `--format tar|tar.gz`: format for the pre-restore backup.
- `--yes`: non-interactive mode.
- `--dry-run`: validate and print actions only.
- `-h, --help`: print built-in usage documentation.

## Restore safety model

1. Verifies archive exists.
2. Verifies archive checksum (`<archive>.sha256`).
3. Extracts archive to staging and verifies file checksums (`<archive>.files.sha256`).
4. Handles pre-restore backup based on `--backup-current`.
   - When pre-restore backup is created, it uses `--scope repo` against the selected `--target`.
5. If pre-restore backup is not created, requires explicit overwrite confirmation (unless `--yes`).
6. Restores files into target path.
7. Verifies restored files in target match `<archive>.files.sha256`.

## Quick examples

Dry-run restore:

```bash
bash scripts/restore_project.sh \
  --archive /projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_YYYYmmdd_HHMMSS.tar \
  --dry-run
```

Restore with explicit safety backup:

```bash
bash scripts/restore_project.sh \
  --archive /projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_YYYYmmdd_HHMMSS.tar \
  --backup-current always
```

Non-interactive restore:

```bash
bash scripts/restore_project.sh \
  --archive /projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_YYYYmmdd_HHMMSS.tar \
  --backup-current always \
  --yes
```

## Archive Validation Commands

List archive contents:

```bash
tar -tf /projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_YYYYmmdd_HHMMSS.tar | head
```

Validate checksum:

```bash
cd /projects/neuro-collab/archives/fhn_dnn
sha256sum -c fhn_dnn_backup_YYYYmmdd_HHMMSS.tar.sha256
```

Validate per-file checksums against a restored target:

```bash
cd /projects/neuro-collab
sha256sum -c /projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_YYYYmmdd_HHMMSS.tar.files.sha256
```

## Troubleshooting

Checksum mismatch:

- Cause: archive changed after checksum generation.
- Action: do not restore; regenerate or copy a valid archive.

Missing checksum sidecars:

- Behavior: restore aborts if `<archive>.sha256` or `<archive>.files.sha256` is missing.
- Action: ensure sidecars are kept with the archive and were not renamed.

Permission errors writing backups:

- Cause: destination path permissions.
- Action: choose writable `--dest` or `--backup-dest`.

Wrong target path:

- Risk: unintended overwrite.
- Action: run restore with `--dry-run` first and verify target.

## Operational Notes

- Use uncompressed `tar` for fast backup/restore on HPC filesystems.
- Prefer `--dry-run` before all restores.
- Keep backup archives under `/projects/neuro-collab/archives/fhn_dnn`.
- Set a retention policy (for example: keep last N daily + weekly snapshots).
