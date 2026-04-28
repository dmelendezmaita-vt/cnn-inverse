# Falcon Orchestration Fix Log (2026-02-28)

## Scope

This log captures:

1. The scheduler fix applied now (after diagnosing failed smoke jobs).
2. Earlier changes already present in the repository that were made for orchestration, notifications, and backup/restore reliability.

## A) Newly Applied Fix: LR Scheduler Crash in 1-Epoch Smoke Runs

### Symptom

- Jobs `232640` and `232641` failed during training with:
  - `ZeroDivisionError: integer modulo by zero`
  - failure location: `lr_scheduler.step()` in `dlkit/opt/train.py`
  - internal traceback path points to PyTorch cosine scheduler logic.

### Root Cause

- The previous scheduler construction always created:
  - `LinearLR` + `ConstantLR` + `CosineAnnealingLR`.
- For smoke runs with `training.epochs: 1`, the computed cosine horizon became invalid (`T_max=0`), which is not allowed by cosine scheduler internals.

### Online Documentation Basis

- `CosineAnnealingLR` requires a positive `T_max` horizon:
  - https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- `SequentialLR` transitions depend on valid stage lengths and milestones:
  - https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html
- Stage scheduler semantics:
  - https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
  - https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html

### Code Changes Applied

1. `code/dl-kit-main/dlkit/opt/scheduler.py`
   - Added early return for `n_epochs <= 1` to disable LR scheduling on 1-epoch smoke runs.
   - Added validation for `learning_rate > 0`.
   - Clamped stage lengths so linear+constant stages never consume all epochs.
   - Built scheduler stages dynamically (skip zero-length stages).
   - Guaranteed cosine stage receives `T_max >= 1`.
   - Returned single scheduler directly when only one stage exists; otherwise used `SequentialLR`.

2. `code/dl-kit-main/dlkit/opt/train.py`
   - Wrapped `lr_scheduler.step()` in explicit runtime error handling.
   - New error message includes epoch and scheduler class, so scheduler misconfiguration is immediately diagnosable from logs.

### Why This Fix

- Preserves full behavior for normal multi-epoch training.
- Prevents crash in short smoke tests used for orchestration validation.
- Improves failure diagnosability if scheduler config is invalid again.

### Validation Performed

- Local scheduler smoke check (`n_epochs` = 1, 2, 10):
  - `n_epochs=1` -> scheduler disabled (`None`), no crash.
  - `n_epochs=2` -> valid cosine scheduler, no crash.
  - `n_epochs=10` -> valid sequential scheduler, no crash.

## A2) Newly Applied Fix: False-Positive `[ERROR]` Summary in Completed Reports

### Symptom

- Job `232678` completed successfully (`Exit 0`, required artifacts present), but report notification still showed:
  - `[ERROR] - summary: Exception raised from recvBytes ...`

### Root Cause

- `util_report()` error extraction in both sbatch files used broad pattern matching (`ERROR|...|Exception`).
- During normal distributed shutdown, non-fatal rendezvous/TCP warnings can include the string `Exception`.
- The parser treated those as report errors even when forced status was `COMPLETED`.

### Code Changes Applied

1. `code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch`
2. `code/fhn_dnn-1-implementation-in-pytorch/slurm_train.sbatch`

Both now:

- skip error-line parsing when `forced_status == COMPLETED`,
- prioritize explicit fatal markers:
  - `ERROR line=`
  - `ERROR: missing or empty output files`
  - `Traceback (most recent call last):`
  - `...Error:`
- use broader fallback patterns only when `forced_status == FAILED`.

### Why This Fix

- Preserves real failure detection while preventing noisy false alarms on successful jobs.
- Keeps notification trust high for rapid triage.

### Validation Performed

- Shell syntax checks passed for both scripts (`bash -n`).
- Existing completed jobs keep valid status semantics:
  - `COMPLETED`/`SUCCESS` no longer require empty warning logs.

## B) Previously Applied Changes (Verified in Current Tree)

These items were already present before the scheduler patch above and are documented here for continuity.

### 1) Notification Format and Capitalization Normalization (smoke + train scripts)

- Files:
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch`
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_train.sbatch`
- Current format includes:
  - `Job ID` label (capitalized correctly)
  - submission/start/end timestamps
  - elapsed runtime and time remaining
  - run directory and status sections
- Reasoning:
  - Make alerts actionable for queued/running/failed/completed states and remove ambiguous report formatting.

### 2) Periodic Runtime Status Reports + Node-Level Telemetry

- Files:
  - same two sbatch files above
- Current reports include:
  - `[RUN STATUS]`, `[CLUSTER CURRENT]`, `[PER NODE]`, `[ERROR]`
  - CPU/RAM/GPU/VRAM summaries
  - report write-out to `report_*.txt` under each run directory
- Reasoning:
  - Provide orchestration visibility without manually tailing distributed rank logs.

### 3) Output Validation Gate for Smoke/Train Runs

- Files:
  - same two sbatch files above
- Current behavior:
  - validates required artifacts (`params.yaml`, `split_indices.npz`, `net.txt`, `metrics_summary.json`, `metrics_summary.csv`, `predictions.npz`, `predictions_schema.json`)
  - validates metrics JSON structure and finite numeric values
  - writes `missing_outputs.txt` when required outputs are absent
- Reasoning:
  - prevent false positives where job exits but does not produce expected outputs.

### 4) `run_dnn.py` Instrumentation and Distributed-Orchestration Hardening

- File:
  - `code/fhn_dnn-1-implementation-in-pytorch/pytorch/run_dnn.py`
- Current behavior includes:
  - progress markers around data-load boundary
  - persisted split indices path
  - distributed-aware data loader usage
  - metrics summary and prediction schema outputs
- Reasoning:
  - improve multi-rank observability and enforce artifact contracts used by the sbatch validators.

### 5) Backup/Restore Integrity Workflow

- Files:
  - `code/fhn_dnn-1-implementation-in-pytorch/scripts/backup_project.sh`
  - `code/fhn_dnn-1-implementation-in-pytorch/scripts/restore_project.sh`
- Current behavior includes:
  - archive checksum and file-level checksum manifests
  - extracted copy verification against manifests
  - restore post-copy checksum verification
  - repo-scope excludes for `archives/*`, `data/*`, `runs/*`
- Reasoning:
  - support safe rollback while avoiding accidental inclusion of runtime data and prior archive outputs in repo-scope backups.

## C) Pending/Watch Items

- Re-run Falcon smoke jobs after this scheduler fix to confirm:
  - no scheduler crash for 1/2/4-node tests,
  - required output artifacts are produced,
  - notification/report format remains intact.

## D) Execution Record (2026-02-28, EST)

- After applying the report parser fix:
  - pending job cancelled: `232679` (`fhn_smoke_4n`)
  - replacement submitted: `232698` (`fhn_smoke_4n`, 4 nodes, 12 minutes)
- Status snapshot:
  - `232677` (`fhn_smoke_1n`): `COMPLETED` (`Exit 0`)
  - `232678` (`fhn_smoke_2n`): `COMPLETED` (`Exit 0`)
  - `232698` (`fhn_smoke_4n`): `PENDING` (at submission snapshot)

## E) Job Naming Convention Update

- Updated both sbatch scripts to use and document the naming format:
  - `<mode>_<gpu>_<nodes>n`
- Header defaults now align with format:
  - `slurm_smoke_test.sbatch`: `tst_l40s_1n`
  - `slurm_train.sbatch`: `trn_l40s_4n`
- Added explicit submission examples for A100/H200 naming overrides in each script header comments.

## F) Backup Script Key-Scope Fix (Stale PDF Include)

### Symptom

- `backup_project.sh` failed in default key-scope mode with:
  - `ERROR: Required key-scope path missing from source: DMM research progress.pdf`

### Root Cause

- `KEY_INCLUDE_PATHS` still listed `DMM research progress.pdf`, but that file is no longer guaranteed to exist at repository root.

### Code Change Applied

- File: `code/fhn_dnn-1-implementation-in-pytorch/scripts/backup_project.sh`
- Removed `DMM research progress.pdf` from `KEY_INCLUDE_PATHS`.

### Result

- Default key-scope backup now completes successfully again.

## G) Large Repo-Scope Backup Root Cause and Cleanup

### What happened

- The large archive `fhn_dnn_backup_20260228_225555.tar` grew to ~192G.
- A later backup exists (`fhn_dnn_backup_20260228_232311.tar`) and validates successfully (`sha256sum -c` OK).

### Root cause

- The large backup was run with `--scope repo`, which includes almost all files under `/projects/neuro-collab` except patterns in `REPO_EXCLUDES`.

## H) GPU Utilization Reporting Stabilization (2026-03-01)

### Symptom

- Runtime notifications frequently showed `GPU util avg: 0%` and `VRAM avg: 0/...` even for runs that clearly executed on CUDA devices.

### Root Cause

- The report path relied on a single-point `nvidia-smi` snapshot.
- Short/bursty kernels and timing mismatch between sampling and kernel execution often produced zero snapshots.

### Code Changes Applied

- Files updated:
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch`
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_train.sbatch`
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test_direct.sbatch`
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_train_direct.sbatch`
- Each node now starts a background GPU monitor before `torchrun`:
  - `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total ... -l 1`
- Metrics are written to job-local, node-local path:
  - `${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${SLURM_JOB_ID}/metrics/gpu_metrics.csv`
- `util_report()` now prefers this monitor history and computes:
  - utilization average from sampled history
  - per-GPU peak list from sampled history
  - VRAM aggregate from sampled history
- Existing one-shot `nvidia-smi` path is retained as fallback.

### Parallel Safety

- No cross-job overwrite risk:
  - path is scoped by `SLURM_JOB_ID`
- No cross-node overwrite risk:
  - writes occur to each node's local temp filesystem
- Cleanup is explicit:
  - monitor process is stopped via `trap ... EXIT` in each node task shell.

## I) 2026-03-01 Resubmission Triage: New Failures and Root Causes

### Actions Performed

- Canceled all pending tracked jobs and resubmitted equivalent replacements.
- Updated tracked job IDs in:
  - `data/important_notes/testing_jobs.csv`
  - `data/important_notes/training_jobs.csv`

### Root Cause 1: Falcon V100 direct-mode jobs (FAILED)

- Affected jobs:
  - `232996` (`trn_V100_2n`)
  - `232997` (`tst_V100_4n`)
  - `232998` (`tst_V100_2n`)
  - `232999` (`trn_V100_4n`)
- Observed error in `stdout.log`:
  - `FileNotFoundError: ... /localscratch/<job>/<job>/ds/y/reduced_data_0.1_curr.npy`
- Interpretation:
  - direct-mode extraction path/layout remains inconsistent with runtime `data_dir` resolution for these V100 jobs.

### Root Cause 2: Tinkercliffs A100 jobs (FAILED immediately)

- Affected jobs:
  - `4703173`, `4703177`, `4703178`, `4703179`, `4703182`, `4703184`
- Observed error in `stdout.log`:
  - `srun: error: Unable to create step ... Invalid generic resource (gres) specification`
- `scontrol show job` evidence:
  - `ReqTRES=...gres/gpu=8`
  - `AllocTRES=...gres/gpu=4,gres/gpu:a100=4`
  - `TresPerNode=gres/gpu:8,gres/gpu:4`
- Interpretation:
  - conflicting GPU requests are being combined (script header `#SBATCH --gres=gpu:4` with CLI `--gpus-per-node=8`) on A100 partitions, causing invalid step GRES state.

## J) 2026-03-01 Non-Extraction + GRES Consistency Fix (Applied to all 4 sbatch scripts)

### Requested behavior

- Do not copy/extract dataset tar contents to node-local storage.
- Keep resource requests consistent to avoid GRES conflicts.

### Changes applied

1. `pytorch/data.py`
   - Added tar-stream loading support for array files:
     - resolves tar member names with and without `reduced_data/` prefix.
     - loads arrays directly from tar members (no filesystem extraction).
     - preserves raw-binary fallback logic.
   - When `data_dir` points to a tar file, disables `mmap_mode` and logs warning.
   - Added tar-aware timesteps loading when `file_names.timesteps` is specified.

2. All four sbatch scripts:
   - `slurm_smoke_test.sbatch`
   - `slurm_train.sbatch`
   - `slurm_smoke_test_direct.sbatch`
   - `slurm_train_direct.sbatch`
   - Removed `#SBATCH --gres=gpu:4` to prevent mixed `gres`/`gpus-per-node` conflicts.
   - Replaced data staging/extraction sections with direct tar mode:
     - `DATA_DIR="${TAR_PATH}"`
   - Kept GPU monitor/reporting instrumentation intact.

### Validation

- `bash -n` passed for all 4 sbatch scripts.
- `python -m py_compile` passed for `pytorch/data.py`.
- Verified expected required tar members exist in `/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar`.
- At that time, `REPO_EXCLUDES` did not include `other/*`.
- File-list evidence showed:
  - `other/*` entries were present (22 files, mostly large video files),
  - top-level `other` directory size is ~190G,
  - `conda` was also included (~2.5G), but `other` dominated total size.

### Fix applied

- `code/fhn_dnn-1-implementation-in-pytorch/scripts/backup_project.sh`
  - bumped script version to `1.1.1`,
  - added `other/*` to `REPO_EXCLUDES`.

### Cleanup performed

- Deleted the failed/aborted large backup set:
  - `fhn_dnn_backup_20260228_225555.tar`
  - `.tar.filelist.txt`
  - `.tar.files.sha256`
  - `.tar.sha256`

## H) Backup/Restore End-to-End Validation (Isolation + Real Files)

### Isolation test

- Built a synthetic source tree under `/tmp/backup_restore_iso_*`.
- Ran `backup_project.sh --scope repo` and restored into an isolated target using `restore_project.sh`.
- Verified expected include/exclude behavior from file list and restored output:
  - Included: `code/*`, `docs/*`, `scripts/*`, `conda/*`
  - Excluded: `other/*`, `data/*`, `archives/*`, `runs/*`, `.codex/*`

### Real-files restore test

- Created fresh key-scope backup:
  - `fhn_dnn_backup_20260301_000007.tar`
- Captured pre-delete checksums, deleted real files:
  - `code/fhn_dnn-1-implementation-in-pytorch/README.md`
  - `code/fhn_dnn-1-implementation-in-pytorch/WARP.md`
  - `code/fhn_dnn-1-implementation-in-pytorch/scripts/backup_project.sh`
- Restored from backup with:
  - `restore_project.sh --backup-current never --yes`
- Verified exact checksum match after restore (`sha256sum -c` all `OK`).

### Result

- Backup and restore functionality is confirmed working both in isolation and against real project files, including explicit delete-and-restore behavior.

## I) Notification Formatting + Utilization Sampling Adjustment (2026-03-01)

### Symptom

- Notification text could render poorly on mobile due to padded labels before `:`.
- CPU/GPU utilization in reports often appeared as `0%` for short jobs.

### Root Cause

- Labels were formatted with alignment spaces before `:`.
- Default report interval was `1200s` (20 min), so short jobs often only emitted a final snapshot after workload completion/failure, when instantaneous utilization is usually low.

### Code Changes Applied

- Files:
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch`
  - `code/fhn_dnn-1-implementation-in-pytorch/slurm_train.sbatch`
- Changes:
  - normalized labels to compact `Key: value` style (no padded whitespace before `:`),
  - normalized per-node line prefix to `${node}: ...`,
  - reduced default `REPORT_INTERVAL` from `1200` to `60` (both definition points in each script).

### Validation

- `bash -n` passed for both scripts.
- no tab-prefixed `:` label formatting remains in either script.

## J) V100 Direct-Mode Failures + A30 2-Node Timing Analysis (2026-03-01)

### Symptom

- V100 direct-mode jobs failed quickly on Falcon:
  - `tst_V100_2n` / `trn_V100_2n` / `tst_V100_4n` / `trn_V100_4n`.
- A30 `trn_A30_2n` completed but took noticeably longer than `trn_A30_1n` and `trn_A30_4n`.

### Root Cause (V100)

- The previous `_direct` scripts still extracted the **entire** tar archive per node into local scratch.
- Logs show tar write failures on local scratch:
  - `Cannot open: No space left on device`
  - `ERROR ... cmd=tar -xf "${TAR_PATH}" -C "${WORK}/ds" exit=2`
- This means `_direct` was only removing the extra `cp`, but still not "direct access" in practice.

### Why A30 2-node was slower

- Slurm elapsed time is run time (`Start -> End`), not queue wait.
- `trn_A30_2n` has run elapsed `00:07:06` vs `00:02:38` (1n) and `00:02:57` (4n).
- During shutdown it logged a distributed rendezvous teardown warning:
  - `RendezvousConnectionError` during shutdown.
- Job still exited successfully (`rc=0`) and output validation passed.

### Fix Applied (Direct Scripts)

- Files updated:
  - `slurm_smoke_test_direct.sbatch`
  - `slurm_train_direct.sbatch`
- Replaced full-tar extract with **selective tar member extraction** for the active `curr` value:
  - required:
    - `y/${data_prefix}_${curr}_curr.npy`
    - `generated_params/${data_prefix}_${curr}_curr.npy`
  - optional:
    - `support_stats/${data_prefix}_${curr}_curr.npy`
- Added explicit missing-member checks before extraction.
- Kept a per-job cross-node barrier (`barrier_direct`) before launching `torchrun` to reduce startup skew/rendezvous timing races.

### Validation

- `bash -n` passes for both updated direct scripts.

### Follow-up correction

- First selective-extract attempt failed because tar members are stored under a top-level `reduced_data/` prefix.
- Added robust tar member resolution in both direct scripts, trying:
  - `${member}`
  - `./${member}`
  - `reduced_data/${member}`
  - `./reduced_data/${member}`
- Re-submitted V100 direct jobs with patched scripts:
  - `tst_V100_2n` -> `232998`
  - `tst_V100_4n` -> `232997`
  - `trn_V100_2n` -> `232996`
  - `trn_V100_4n` -> `232999`

## K) Data-Access Mode Hardening Across All 4 Slurm Scripts (2026-03-05)

### Symptom

- `_direct` scripts had drifted to be effectively identical to non-direct scripts.
- In practice, `_direct` still defaulted to `copy_to_node`, which can reintroduce known local-scratch failures.

### Root Cause

- `DATA_ACCESS_MODE` default remained `copy_to_node` in all four scripts.
- There was no strict mode validation, so mis-typed mode values were not rejected early.
- `copy_to_node` path failed hard on copy/extract errors, even though direct tar access is available.

### Code Changes Applied

- Files updated:
  - `slurm_smoke_test.sbatch`
  - `slurm_train.sbatch`
  - `slurm_smoke_test_direct.sbatch`
  - `slurm_train_direct.sbatch`
- Added `DATA_ACCESS_MODE` validation/canonicalization in all four:
  - accepted: `copy_to_node`, `direct_tar`
  - alias: `direct` -> canonicalized to `direct_tar`
  - invalid values now fail fast with explicit error.
- `_direct` variants now default to:
  - `DATA_ACCESS_MODE=direct_tar`
- Added runtime fallback in `copy_to_node` path:
  - if copy or extract fails, script logs warning and falls back to `direct_tar` mode instead of failing immediately.

### Validation

- `bash -n` passed for all four scripts after edits.
- Diff checks confirm the remaining intended difference between regular and `_direct` scripts is now the default `DATA_ACCESS_MODE`.
