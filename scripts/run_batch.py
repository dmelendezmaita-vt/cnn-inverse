#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from batches import (
    BatchDefinition,
    BatchStep,
    batch_to_dict,
    batches_for_suite,
    batches_through,
    get_batch,
    list_batches,
    suite_names,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run or plan the public experiment batches. The default execution contract is the current Falcon A30 allocation, "
            "under which smoke, scientific-smoke, and canonical batches can be staged through one runner."
        )
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="List the available batches and suites.")
    mode.add_argument("--batch", help="Run exactly one batch.")
    mode.add_argument("--through", help="Run all batches up to and including the requested batch.")
    mode.add_argument("--suite", choices=suite_names(), help="Run the named suite in its declared order.")

    ap.add_argument("--hh-tar-path", default=None, help="Path to the external Hodgkin-Huxley tar.gz archive.")
    ap.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "batches"), help="Root directory for batch outputs and status files.")
    ap.add_argument("--python-bin", default=sys.executable, help="Python interpreter used for Python-based steps.")
    ap.add_argument("--monitor-python", default=None, help="Interpreter used to run the repo resource-monitoring tools.")
    ap.add_argument("--curr", default="0.1", help="Current identifier passed to HH workflows.")
    ap.add_argument("--resume", action="store_true", help="Skip batches that already have a completed status file.")
    ap.add_argument("--force", action="store_true", help="Rerun batches even when a completed status file already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Print the resolved commands and environments without executing them.")

    ap.add_argument("--allocation-job-id", default=None, help="Slurm allocation job id, defaults to $SLURM_JOB_ID.")
    ap.add_argument("--master-addr", default=None, help="Distributed rendezvous address for Falcon batches.")
    ap.add_argument("--master-port", default="29500", help="Distributed rendezvous port for Falcon batches.")
    ap.add_argument("--step-nnodes", type=int, default=4, help="Default node count for Falcon DNN batches.")
    ap.add_argument("--nproc-per-node", type=int, default=2, help="Default processes per node for Falcon DNN batches.")
    ap.add_argument("--launch-backend", default="torchrun", choices=["torchrun", "slurm_direct"], help="Launcher backend for Falcon DNN batches.")
    ap.add_argument(
        "--data-access-mode",
        default="nvme_full_extract",
        choices=["copy_to_node", "tar_in_place", "shm_curr_copy", "shm_full_copy", "nvme_full_extract"],
        help="Data staging policy for Falcon DNN batches.",
    )
    ap.add_argument("--shared-data-dir", default=None, help="Optional shared extracted-data directory for Falcon batches.")
    ap.add_argument("--prepared-data-root", default=None, help="Optional shared prepared-data root used before any node-local tmpfs staging.")
    ap.add_argument("--srun-bin", default="srun", help="Slurm launcher used for distributed Falcon batches.")
    ap.add_argument("--srun-cpus-per-task", type=int, default=None, help="Optional cpus-per-task override for distributed Falcon batches.")
    return ap.parse_args()


def resolve_batches(args: argparse.Namespace) -> Tuple[BatchDefinition, ...]:
    if args.batch:
        return (get_batch(args.batch),)
    if args.through:
        return batches_through(args.through)
    if args.suite:
        return batches_for_suite(args.suite)
    return ()


def status_root(run_root: Path) -> Path:
    return run_root / "_status"


def manifest_root(run_root: Path) -> Path:
    return run_root / "_manifests"


def batch_root(run_root: Path, batch: BatchDefinition) -> Path:
    return run_root / batch.batch_id


def batch_status_path(run_root: Path, batch: BatchDefinition) -> Path:
    return status_root(run_root) / f"{batch.batch_id}.json"


def batch_manifest_path(run_root: Path, batch: BatchDefinition) -> Path:
    return manifest_root(run_root) / f"{batch.batch_id}.json"


def ensure_dirs(run_root: Path, batch: BatchDefinition) -> None:
    batch_root(run_root, batch).mkdir(parents=True, exist_ok=True)
    status_root(run_root).mkdir(parents=True, exist_ok=True)
    manifest_root(run_root).mkdir(parents=True, exist_ok=True)


def load_status(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_hh_tar(batch: BatchDefinition, args: argparse.Namespace) -> Optional[Path]:
    if not batch.requires_hh_tar:
        return None
    if not args.hh_tar_path:
        raise SystemExit(f"Batch {batch.batch_id} requires --hh-tar-path.")
    tar_path = Path(args.hh_tar_path).expanduser().resolve()
    if not tar_path.exists():
        raise SystemExit(f"HH tar archive does not exist: {tar_path}")
    return tar_path


def derive_data_prefix(tar_path: Path) -> str:
    name = tar_path.name
    for suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return tar_path.stem


def default_prepared_data_root(args: argparse.Namespace) -> Path:
    if args.prepared_data_root:
        return Path(args.prepared_data_root).expanduser()
    return REPO_ROOT / ".prepared_data"


def _job_nodelist(allocation_job_id: str) -> Optional[str]:
    if shutil.which("scontrol") is None:
        return None
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "job", str(allocation_job_id)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    for token in out.split():
        if token.startswith("NodeList="):
            value = token.split("=", 1)[1].strip()
            if value and value != "(null)":
                return value
    return None


def derive_master_addr(args: argparse.Namespace, allocation_job_id: Optional[str] = None) -> str:
    if args.master_addr:
        return args.master_addr
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist and allocation_job_id:
        nodelist = _job_nodelist(str(allocation_job_id))
    if nodelist and shutil.which("scontrol"):
        try:
            out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).strip().splitlines()
            if out:
                return out[0].strip()
        except Exception:
            pass
    env_master = os.environ.get("MASTER_ADDR")
    if env_master:
        return env_master
    return os.environ.get("HOSTNAME", "localhost")


def derive_step_master_port(base_port: str, step_id: str) -> str:
    try:
        base = int(base_port)
    except Exception:
        return str(base_port)
    offset = (sum(step_id.encode("utf-8")) % 1000) + 1
    return str(base + offset)


def require_slurm(batch: BatchDefinition, args: argparse.Namespace) -> str:
    allocation_job_id = args.allocation_job_id or os.environ.get("SLURM_JOB_ID")
    if not batch.requires_slurm:
        return allocation_job_id or "manual"
    if not allocation_job_id:
        if args.dry_run:
            return "DRYRUN_ALLOCATION"
        raise SystemExit(
            f"Batch {batch.batch_id} requires a Slurm allocation. Run it inside the documented cluster environment, "
            f"or pass --allocation-job-id explicitly."
        )
    if batch.min_nodes:
        env_nodes = os.environ.get("SLURM_NNODES")
        if env_nodes is not None:
            try:
                if int(env_nodes) < int(batch.min_nodes):
                    if args.dry_run:
                        return str(allocation_job_id)
                    raise SystemExit(
                        f"Batch {batch.batch_id} requires at least {batch.min_nodes} allocated nodes, but SLURM_NNODES={env_nodes}."
                    )
            except ValueError:
                pass
    return str(allocation_job_id)


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def step_output_root(run_root: Path, batch: BatchDefinition, step: BatchStep) -> Path:
    return batch_root(run_root, batch) / step.step_id


def source_step_root(run_root: Path, batch: BatchDefinition, step_id: str) -> Path:
    source_step = next(item for item in batch.steps if item.step_id == step_id)
    return step_output_root(run_root, batch, source_step)


def latest_checkpoint_under(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    patterns = ("net_e*.pt", "*.ckpt", "model.keras")
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(path.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def wrap_single_node_srun(args: argparse.Namespace, allocation_job_id: str, command: Sequence[str]) -> List[str]:
    wrapped = [
        args.srun_bin,
        "--jobid",
        str(allocation_job_id),
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--ntasks-per-node",
        "1",
        "--kill-on-bad-exit=1",
    ]
    cpus_per_task = args.srun_cpus_per_task or os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task:
        wrapped += ["--cpus-per-task", str(cpus_per_task)]
    wrapped += list(command)
    return wrapped


def resolve_python_bin(args: argparse.Namespace, step: BatchStep) -> str:
    if step.python_bin_override:
        path = Path(step.python_bin_override)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path)
    return args.python_bin


def resolve_monitor_python(args: argparse.Namespace, step: BatchStep) -> str:
    if args.monitor_python:
        path = Path(args.monitor_python)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path)
    env_override = os.environ.get("RESOURCE_MONITOR_PYTHON", "").strip()
    if env_override:
        return env_override
    return resolve_python_bin(args, step)


def wrap_with_resource_monitor(
    *,
    monitor_python: str,
    step_root: Path,
    step_id: str,
    gpu_expected: int,
    command: Sequence[str],
) -> List[str]:
    monitor_dir = step_root / "resource_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return [
        monitor_python,
        "scripts/run_monitored_command.py",
        "--resource-dir",
        str(monitor_dir),
        "--label",
        step_id,
        "--gpu-expected",
        str(int(gpu_expected)),
        "--monitor-python",
        monitor_python,
        "--",
        *list(command),
    ]


def gpu_expected_for_step(
    args: argparse.Namespace,
    batch: BatchDefinition,
    step: BatchStep,
) -> int:
    if not batch.requires_slurm:
        return 0
    if not step.wrap_srun:
        return 0
    if step.kind == "interactive_dnn":
        if step.load_from_step:
            return 0
        return max(1, int(step.nproc_per_node or 1))
    if step.kind == "classical":
        return 0
    if step.kind == "sbi":
        extra_args = tuple(str(token).lower() for token in step.extra_args)
        if "--device" in extra_args:
            idx = extra_args.index("--device")
            if idx + 1 < len(extra_args) and extra_args[idx + 1] == "cpu":
                return 0
        return 1
    if step.kind == "dnn":
        return 1
    if step.kind == "script":
        extra_args = tuple(str(token).lower() for token in step.extra_args)
        if "--device" in extra_args:
            idx = extra_args.index("--device")
            if idx + 1 < len(extra_args) and extra_args[idx + 1] == "cpu":
                return 0
            if idx + 1 < len(extra_args) and extra_args[idx + 1] in {"gpu", "cuda", "auto"}:
                return max(1, int(step.nproc_per_node or 1))
        script_path = (step.script_path or "").lower()
        if "swyft" in script_path:
            return max(1, int(step.nproc_per_node or 1))
        if step.python_bin_override and "bayesflow" in step.python_bin_override.lower():
            return max(1, int(step.nproc_per_node or 1))
        if step.nproc_per_node is not None and step.step_nnodes is not None and int(step.step_nnodes) > 1:
            return max(1, int(step.nproc_per_node))
        if step.step_nnodes is not None and int(step.step_nnodes) > 1 and batch.preferred_gpu:
            return 1
    return 0


def render_extra_args(
    extra_args: Sequence[str],
    *,
    run_root: Path,
    batch: BatchDefinition,
    step: BatchStep,
) -> List[str]:
    rendered: List[str] = []
    for token in extra_args:
        value = str(token)
        value = value.replace("{REPO_ROOT}", str(REPO_ROOT))
        value = value.replace("{RUN_ROOT}", str(run_root))
        value = value.replace("{BATCH_ROOT}", str(batch_root(run_root, batch)))
        value = value.replace("{STEP_ROOT}", str(step_output_root(run_root, batch, step)))
        if step.load_from_step:
            value = value.replace("{LOAD_FROM_STEP_ROOT}", str(source_step_root(run_root, batch, step.load_from_step)))
        rendered.append(value)
    return rendered


def step_cohorts(steps: Sequence[BatchStep]) -> List[List[BatchStep]]:
    cohorts: List[List[BatchStep]] = []
    current: List[BatchStep] = []
    current_group: Optional[str] = None
    for step in steps:
        group = step.parallel_group
        if not group:
            if current:
                cohorts.append(current)
                current = []
                current_group = None
            cohorts.append([step])
            continue
        if current and current_group == group:
            current.append(step)
            continue
        if current:
            cohorts.append(current)
        current = [step]
        current_group = group
    if current:
        cohorts.append(current)
    return cohorts


def resolve_load_checkpoint(
    run_root: Path,
    batch: BatchDefinition,
    step: BatchStep,
    step_records: Dict[str, Dict[str, object]],
    *,
    placeholder_ok: bool = False,
) -> Path:
    if not step.load_from_step:
        raise RuntimeError(f"Step {step.step_id} does not declare load_from_step.")
    source_root = step_output_root(run_root, batch, next(item for item in batch.steps if item.step_id == step.load_from_step))
    source_record = step_records.get(step.load_from_step)
    if source_record and source_record.get("latest_checkpoint"):
        return Path(str(source_record["latest_checkpoint"]))
    checkpoint = latest_checkpoint_under(source_root)
    if checkpoint is None:
        if placeholder_ok:
            return source_root / "__latest_checkpoint__.pt"
        raise SystemExit(
            f"Unable to resolve a checkpoint for step {step.step_id}. No checkpoint was found under {source_root}."
        )
    return checkpoint


def make_step_env(
    args: argparse.Namespace,
    batch: BatchDefinition,
    step: BatchStep,
    hh_tar_path: Optional[Path],
    allocation_job_id: str,
    step_records: Dict[str, Dict[str, object]],
    run_root: Path,
    *,
    placeholder_ok: bool = False,
) -> Dict[str, str]:
    env = os.environ.copy()
    python_bin = resolve_python_bin(args, step)
    monitor_python = resolve_monitor_python(args, step)
    env["PYTHONUNBUFFERED"] = "1"
    env["REPO_ROOT"] = str(REPO_ROOT)
    env["PYTHON_BIN"] = python_bin
    env["RESOURCE_MONITOR_PYTHON"] = monitor_python
    torchrun_path = Path(python_bin).parent / "torchrun"
    if torchrun_path.exists():
        env["TORCHRUN_BIN"] = str(torchrun_path)
    if batch.requires_slurm and hh_tar_path is not None:
        data_prefix = derive_data_prefix(hh_tar_path)
        prepared_root = default_prepared_data_root(args)
        env["ALLOC_JOB_ID"] = allocation_job_id
        env["RUN_ID"] = step.step_id
        env["RUN_OUTPUT_ROOT"] = str(step_output_root(run_root, batch, step))
        env["PARAMS_FILE"] = step.params_file
        env["TAR_PATH"] = str(hh_tar_path)
        env["DATA_PREFIX"] = data_prefix
        env["MASTER_ADDR"] = derive_master_addr(args, allocation_job_id)
        env["MASTER_PORT"] = derive_step_master_port(str(args.master_port), step.step_id)
        env["STEP_NNODES"] = str(step.step_nnodes or args.step_nnodes)
        env["NPROC_PER_NODE"] = str(step.nproc_per_node or args.nproc_per_node)
        env["DATA_ACCESS_MODE"] = str(args.data_access_mode)
        env["NC_HH_STAGE_MODE"] = str(
            args.data_access_mode if (args.data_access_mode.startswith("shm_") or args.data_access_mode.startswith("nvme_")) else "none"
        )
        env["NC_HH_PREPARED_ROOT"] = str(prepared_root)
        env["NC_HH_CACHE_IDENTITY"] = f"{hh_tar_path.resolve()}::{data_prefix}"
        env["NC_HH_SPLIT_CACHE_ROOT"] = str(prepared_root / data_prefix / ".split_array_cache")
        env["NC_HH_SCALE_CACHE_ROOT"] = str(prepared_root / data_prefix / ".scale_cache")
        env["LAUNCH_BACKEND"] = str(args.launch_backend)
        env["RESOURCE_MONITOR_DIR"] = str(step_output_root(run_root, batch, step) / "resource_monitor")
        env["RESOURCE_MONITOR_LABEL"] = str(step.step_id)
        env["RESOURCE_MONITOR_INTERVAL_SEC"] = os.environ.get("RESOURCE_MONITOR_INTERVAL_SEC", "5.0")
        if args.shared_data_dir:
            env["SHARED_DATA_DIR"] = str(Path(args.shared_data_dir).expanduser())
        if step.pass_curr:
            env["CURR"] = str(args.curr)
        if step.save_predictions:
            env["SAVE_PREDICTIONS"] = step.save_predictions
        if step.split_eval_after_train:
            env["SPLIT_EVAL_AFTER_TRAIN"] = "1"
        if step.skip_inline_split_eval:
            env["SKIP_INLINE_SPLIT_EVAL"] = "1"
        if step.load_from_step:
            env["EVAL_ONLY_CHECKPOINT"] = str(
                resolve_load_checkpoint(run_root, batch, step, step_records, placeholder_ok=placeholder_ok)
            )
    for key, value in step.env_overrides:
        env[str(key)] = str(value)
    return env


def build_step_command(
    args: argparse.Namespace,
    batch: BatchDefinition,
    step: BatchStep,
    hh_tar_path: Optional[Path],
    allocation_job_id: str,
    step_records: Dict[str, Dict[str, object]],
    run_root: Path,
    *,
    placeholder_ok: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    step_root = step_output_root(run_root, batch, step)
    step_root.mkdir(parents=True, exist_ok=True)
    env = make_step_env(
        args,
        batch,
        step,
        hh_tar_path,
        allocation_job_id,
        step_records,
        run_root,
        placeholder_ok=placeholder_ok,
    )

    if step.kind == "dnn":
        python_bin = resolve_python_bin(args, step)
        base_command = [
            python_bin,
            "src/pytorch/run_dnn.py",
            "--params",
            step.params_file,
            "--mode",
            str(step.mode),
            "--save_dir_base",
            str(step_root),
        ]
        if hh_tar_path is not None:
            base_command += ["--data_dir", str(hh_tar_path)]
        if step.pass_curr:
            base_command += ["--curr", str(args.curr)]
        if step.save_predictions:
            base_command += ["--save_predictions", step.save_predictions]
        if step.load_from_step:
            base_command += [
                "--load_dir",
                str(resolve_load_checkpoint(run_root, batch, step, step_records, placeholder_ok=placeholder_ok)),
            ]
        base_command += render_extra_args(step.extra_args, run_root=run_root, batch=batch, step=step)
        monitored_command = wrap_with_resource_monitor(
            monitor_python=resolve_monitor_python(args, step),
            step_root=step_root,
            step_id=step.step_id,
            gpu_expected=gpu_expected_for_step(args, batch, step),
            command=base_command,
        )
        command = wrap_single_node_srun(args, allocation_job_id, monitored_command) if batch.requires_slurm and step.wrap_srun else monitored_command
        return command, env

    if step.kind == "classical":
        python_bin = resolve_python_bin(args, step)
        base_command = [
            python_bin,
            "scripts/run_hh_classical_baseline.py",
            "--params",
            step.params_file,
            "--baseline",
            str(step.baseline_name),
            "--feature-mode",
            str(step.feature_mode),
            "--save-dir",
            str(step_root),
        ]
        if hh_tar_path is not None:
            base_command += ["--data-dir", str(hh_tar_path)]
        if step.pass_curr:
            base_command += ["--curr", str(args.curr)]
        base_command += render_extra_args(step.extra_args, run_root=run_root, batch=batch, step=step)
        monitored_command = wrap_with_resource_monitor(
            monitor_python=resolve_monitor_python(args, step),
            step_root=step_root,
            step_id=step.step_id,
            gpu_expected=gpu_expected_for_step(args, batch, step),
            command=base_command,
        )
        command = wrap_single_node_srun(args, allocation_job_id, monitored_command) if batch.requires_slurm and step.wrap_srun else monitored_command
        return command, env

    if step.kind == "sbi":
        python_bin = resolve_python_bin(args, step)
        base_command = [
            python_bin,
            "scripts/run_hh_sbi_baseline.py",
            "--params",
            step.params_file,
            "--save-dir",
            str(step_root),
            "--method",
            str(step.sbi_method),
            "--density-estimator",
            str(step.density_estimator),
            "--feature-mode",
            str(step.feature_mode),
        ]
        if hh_tar_path is not None:
            base_command += ["--data-dir", str(hh_tar_path)]
        if step.pass_curr:
            base_command += ["--curr", str(args.curr)]
        if step.n_train is not None:
            base_command += ["--n-train", str(step.n_train)]
        if step.eval_limit is not None:
            base_command += ["--eval-limit", str(step.eval_limit)]
        if step.posterior_samples is not None:
            base_command += ["--posterior-samples", str(step.posterior_samples)]
        if step.stop_after_epochs is not None:
            base_command += ["--stop-after-epochs", str(step.stop_after_epochs)]
        if step.max_num_epochs is not None:
            base_command += ["--max-num-epochs", str(step.max_num_epochs)]
        if step.seed is not None:
            base_command += ["--seed", str(step.seed)]
        base_command += render_extra_args(step.extra_args, run_root=run_root, batch=batch, step=step)
        monitored_command = wrap_with_resource_monitor(
            monitor_python=resolve_monitor_python(args, step),
            step_root=step_root,
            step_id=step.step_id,
            gpu_expected=gpu_expected_for_step(args, batch, step),
            command=base_command,
        )
        command = wrap_single_node_srun(args, allocation_job_id, monitored_command) if batch.requires_slurm and step.wrap_srun else monitored_command
        return command, env

    if step.kind == "script":
        if not step.script_path:
            raise SystemExit(f"Batch step {batch.batch_id}:{step.step_id} is missing script_path.")
        python_bin = resolve_python_bin(args, step)
        base_command = [
            python_bin,
            step.script_path,
            "--save-dir",
            str(step_root),
        ]
        if step.params_file:
            base_command += ["--params", step.params_file]
        if hh_tar_path is not None and step.pass_data_dir:
            base_command += ["--data-dir", str(hh_tar_path), "--data-prefix", derive_data_prefix(hh_tar_path)]
        if step.pass_curr:
            base_command += ["--curr", str(args.curr)]
        if step.feature_mode:
            base_command += ["--feature-mode", str(step.feature_mode)]
        if step.seed is not None:
            base_command += ["--seed", str(step.seed)]
        base_command += render_extra_args(step.extra_args, run_root=run_root, batch=batch, step=step)
        monitored_command = wrap_with_resource_monitor(
            monitor_python=resolve_monitor_python(args, step),
            step_root=step_root,
            step_id=step.step_id,
            gpu_expected=gpu_expected_for_step(args, batch, step),
            command=base_command,
        )
        command = wrap_single_node_srun(args, allocation_job_id, monitored_command) if batch.requires_slurm and step.wrap_srun else monitored_command
        return command, env

    if step.kind == "interactive_dnn":
        nnodes = int(step.step_nnodes or args.step_nnodes)
        command = [
            args.srun_bin,
            "--overlap",
            "--jobid",
            str(allocation_job_id),
            "--nodes",
            str(nnodes),
            "--ntasks",
            str(nnodes),
            "--ntasks-per-node",
            "1",
            "--kill-on-bad-exit=1",
        ]
        cpus_per_task = args.srun_cpus_per_task or os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus_per_task:
            command += ["--cpus-per-task", str(cpus_per_task)]
        command += ["bash", "scripts/run_interactive_dnn_step.sh"]
        return command, env

    raise SystemExit(f"Unsupported step kind: {step.kind}")


def summarize_step_outputs(run_root: Path, batch: BatchDefinition, step: BatchStep) -> Dict[str, object]:
    root = step_output_root(run_root, batch, step)
    outputs: Dict[str, object] = {"output_root": str(root)}
    checkpoint = latest_checkpoint_under(root)
    if checkpoint is not None:
        outputs["latest_checkpoint"] = str(checkpoint)
    metric_files = sorted(root.rglob("metrics_summary.json"))
    if metric_files:
        outputs["metrics_summary_files"] = [str(path) for path in metric_files]
    raw_monitor_files = sorted(root.rglob("resource_monitor.json"))
    if raw_monitor_files:
        outputs["resource_monitor_files"] = [str(path) for path in raw_monitor_files]
        monitors: List[Dict[str, object]] = []
        monitor_details: List[Dict[str, object]] = []
        for path in raw_monitor_files:
            try:
                raw_monitor = json.loads(path.read_text())
            except Exception:
                continue
            monitors.append(raw_monitor)
            resource_dir = path.parent
            detail: Dict[str, object] = {
                "resource_dir": str(resource_dir),
                "label": raw_monitor.get("label", ""),
                "hostname": raw_monitor.get("hostname", ""),
                "slurm_step_id": raw_monitor.get("slurm_step_id", ""),
                "raw_summary": raw_monitor.get("summary", {}),
            }
            task_env_path = resource_dir / "task_env.json"
            if task_env_path.exists():
                try:
                    detail["task_env"] = json.loads(task_env_path.read_text())
                except Exception:
                    pass
            resource_summary_path = resource_dir / "resource_summary.json"
            if resource_summary_path.exists():
                try:
                    detail["resource_summary"] = json.loads(resource_summary_path.read_text())
                except Exception:
                    pass
            monitor_details.append(detail)
        if monitor_details:
            outputs["resource_monitor_details"] = monitor_details
        if monitors:
            summary = {
                "hostnames": sorted({str(item.get("hostname")) for item in monitors if item.get("hostname")}),
                "slurm_step_ids": sorted({str(item.get("slurm_step_id")) for item in monitors if item.get("slurm_step_id")}),
                "max_tree_cpu_percent": max(
                    float(item.get("summary", {}).get("max_tree_cpu_percent") or 0.0) for item in monitors
                ),
                "max_tree_rss_mb": max(
                    float(item.get("summary", {}).get("max_tree_rss_mb") or 0.0) for item in monitors
                ),
                "max_tree_vms_mb": max(
                    float(item.get("summary", {}).get("max_tree_vms_mb") or 0.0) for item in monitors
                ),
                "max_node_cpu_percent_mean": max(
                    float(item.get("summary", {}).get("max_node_cpu_percent_mean") or 0.0) for item in monitors
                ),
                "max_node_memory_used_mb": max(
                    float(item.get("summary", {}).get("max_node_memory_used_mb") or 0.0) for item in monitors
                ),
                "max_node_memory_percent": max(
                    float(item.get("summary", {}).get("max_node_memory_percent") or 0.0) for item in monitors
                ),
                "max_node_swap_used_mb": max(
                    float(item.get("summary", {}).get("max_node_swap_used_mb") or 0.0) for item in monitors
                ),
                "node_by_hostname": {},
                "gpu_by_uuid": {},
            }
            node_by_hostname: Dict[str, Dict[str, object]] = {}
            gpu_by_uuid: Dict[str, Dict[str, object]] = {}
            for item in monitors:
                hostname = str(item.get("hostname") or "")
                node_summary = item.get("summary", {})
                if hostname:
                    per_cpu = [float(x) for x in (node_summary.get("max_node_cpu_percent_percpu") or [])]
                    record = node_by_hostname.setdefault(
                        hostname,
                        {
                            "max_node_cpu_percent_mean": 0.0,
                            "max_node_cpu_percent_percpu": per_cpu,
                            "max_node_memory_used_mb": 0.0,
                            "max_node_memory_percent": 0.0,
                            "max_node_swap_used_mb": 0.0,
                        },
                    )
                    record["max_node_cpu_percent_mean"] = max(
                        float(record["max_node_cpu_percent_mean"]),
                        float(node_summary.get("max_node_cpu_percent_mean") or 0.0),
                    )
                    if per_cpu:
                        existing = [float(x) for x in record.get("max_node_cpu_percent_percpu", [])]
                        width = max(len(existing), len(per_cpu))
                        merged = []
                        for idx in range(width):
                            left = existing[idx] if idx < len(existing) else 0.0
                            right = per_cpu[idx] if idx < len(per_cpu) else 0.0
                            merged.append(max(left, right))
                        record["max_node_cpu_percent_percpu"] = merged
                    record["max_node_memory_used_mb"] = max(
                        float(record["max_node_memory_used_mb"]),
                        float(node_summary.get("max_node_memory_used_mb") or 0.0),
                    )
                    record["max_node_memory_percent"] = max(
                        float(record["max_node_memory_percent"]),
                        float(node_summary.get("max_node_memory_percent") or 0.0),
                    )
                    record["max_node_swap_used_mb"] = max(
                        float(record["max_node_swap_used_mb"]),
                        float(node_summary.get("max_node_swap_used_mb") or 0.0),
                    )
                for uuid, gpu in (item.get("summary", {}).get("gpu_by_uuid") or {}).items():
                    record = gpu_by_uuid.setdefault(
                        str(uuid),
                        {
                            "index": gpu.get("index"),
                            "name": gpu.get("name"),
                            "max_utilization_gpu_percent": 0.0,
                            "max_utilization_memory_percent": 0.0,
                            "max_memory_used_mb": 0.0,
                            "max_step_process_gpu_memory_mb": 0.0,
                        },
                    )
                    record["max_utilization_gpu_percent"] = max(
                        float(record["max_utilization_gpu_percent"]),
                        float(gpu.get("max_utilization_gpu_percent") or 0.0),
                    )
                    record["max_utilization_memory_percent"] = max(
                        float(record["max_utilization_memory_percent"]),
                        float(gpu.get("max_utilization_memory_percent") or 0.0),
                    )
                    record["max_memory_used_mb"] = max(
                        float(record["max_memory_used_mb"]),
                        float(gpu.get("max_memory_used_mb") or 0.0),
                    )
                    record["max_step_process_gpu_memory_mb"] = max(
                        float(record["max_step_process_gpu_memory_mb"]),
                        float(gpu.get("max_step_process_gpu_memory_mb") or 0.0),
                    )
            summary["node_by_hostname"] = node_by_hostname
            summary["gpu_by_uuid"] = gpu_by_uuid
            outputs["resource_monitor_summary"] = summary
    resource_summary_files = sorted(root.rglob("resource_summary.json"))
    if resource_summary_files:
        outputs["resource_summary_files"] = [str(path) for path in resource_summary_files]
    resource_time_files = sorted(root.rglob("resource_time.txt"))
    if resource_time_files:
        outputs["resource_time_files"] = [str(path) for path in resource_time_files]
    sacct_files = sorted(root.rglob("sacct_step.txt"))
    if sacct_files:
        outputs["sacct_step_files"] = [str(path) for path in sacct_files]
    gpu_csv_files = sorted(root.rglob("gpu_monitor.csv"))
    if gpu_csv_files:
        outputs["gpu_monitor_files"] = [str(path) for path in gpu_csv_files]
    return outputs


def validate_step_outputs(args: argparse.Namespace, run_root: Path, batch: BatchDefinition, step: BatchStep, outputs: Dict[str, object]) -> None:
    root = step_output_root(run_root, batch, step)
    errors: List[str] = []

    if not root.exists():
        errors.append(f"output root does not exist: {root}")

    if step.kind in {"dnn", "interactive_dnn"} and not step.load_from_step:
        if "latest_checkpoint" not in outputs:
            errors.append("expected a checkpoint artifact, but no checkpoint was found")

    expects_metrics = False
    if step.kind in {"classical", "sbi"}:
        expects_metrics = True
    elif step.kind == "dnn" and str(step.mode).lower() == "eval":
        expects_metrics = True
    elif step.kind == "interactive_dnn" and step.load_from_step:
        expects_metrics = True
    if expects_metrics and "metrics_summary_files" not in outputs:
        errors.append("expected at least one metrics_summary.json artifact, but none was found")
    if batch.requires_slurm and "resource_summary_files" not in outputs:
        errors.append("expected at least one resource_summary.json artifact, but none was found")
    if batch.requires_slurm and "resource_monitor_files" not in outputs:
        errors.append("expected at least one resource_monitor.json artifact, but none was found")

    if step.required_globs:
        for pattern in step.required_globs:
            matches = list(root.rglob(pattern))
            if not matches:
                errors.append(f"expected at least one artifact matching {pattern!r}, but none was found")

    if step.kind == "script" and not step.required_globs:
        if not any(path.is_file() for path in root.rglob("*")):
            errors.append("script step produced no files under its output root")

    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"Output validation failed for {batch.batch_id}:{step.step_id}: {joined}")

    if batch.requires_slurm:
        monitor_python = resolve_monitor_python(args, step)
        for summary_path in outputs.get("resource_summary_files", []):
            resource_dir = str(Path(summary_path).parent)
            subprocess.run(
                [
                    monitor_python,
                    "scripts/validate_monitoring_contract.py",
                    "--resource-dir",
                    resource_dir,
                ],
                cwd=str(REPO_ROOT),
                check=True,
            )


def print_batch_listing() -> None:
    print("Batches")
    for batch in list_batches():
        reqs = []
        if batch.requires_hh_tar:
            reqs.append("HH tar")
        if batch.requires_slurm:
            reqs.append(batch.cluster_label or "Slurm")
        req_text = ", ".join(reqs) if reqs else "none"
        print(f"  {batch.batch_id:28s} suite={batch.suite:16s} requires={req_text}")
        print(f"    {batch.title}")
    print("\nSuites")
    for suite in suite_names():
        batch_ids = [batch.batch_id for batch in batches_for_suite(suite)]
        print(f"  {suite}: {', '.join(batch_ids)}")


def resolved_plan(
    args: argparse.Namespace,
    batches: Sequence[BatchDefinition],
) -> List[Dict[str, object]]:
    plan = []
    for batch in batches:
        hh_tar_path = require_hh_tar(batch, args)
        allocation_job_id = require_slurm(batch, args)
        step_records: Dict[str, Dict[str, object]] = {}
        batch_plan = {
            "batch": batch.batch_id,
            "title": batch.title,
            "suite": batch.suite,
            "steps": [],
        }
        for step in batch.steps:
            command, env = build_step_command(
                args=args,
                batch=batch,
                step=step,
                hh_tar_path=hh_tar_path,
                allocation_job_id=allocation_job_id,
                step_records=step_records,
                run_root=Path(args.run_root).expanduser().resolve(),
                placeholder_ok=True,
            )
            batch_plan["steps"].append(
                {
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "description": step.description,
                    "command": command,
                    "env_subset": {
                        key: env[key]
                        for key in sorted(
                            key
                            for key in env.keys()
                            if key
                            in {
                                "ALLOC_JOB_ID",
                                "CURR",
                                "DATA_ACCESS_MODE",
                                "DATA_PREFIX",
                                "EVAL_ONLY_CHECKPOINT",
                                "LAUNCH_BACKEND",
                                "MASTER_ADDR",
                                "MASTER_PORT",
                                "NPROC_PER_NODE",
                                "NC_HH_PREPARED_ROOT",
                                "NC_HH_STAGE_MODE",
                                "PARAMS_FILE",
                                "RUN_ID",
                                "RUN_OUTPUT_ROOT",
                                "SAVE_PREDICTIONS",
                                "SHARED_DATA_DIR",
                                "SPLIT_EVAL_AFTER_TRAIN",
                                "STEP_NNODES",
                                "TAR_PATH",
                            }
                        )
                    },
                }
            )
        plan.append(batch_plan)
    return plan


def execute_batches(args: argparse.Namespace, batches: Sequence[BatchDefinition]) -> None:
    run_root = Path(args.run_root).expanduser().resolve()

    for batch in batches:
        ensure_dirs(run_root, batch)
        status_path = batch_status_path(run_root, batch)
        manifest_path = batch_manifest_path(run_root, batch)
        existing = load_status(status_path)
        if existing and existing.get("status") == "completed":
            if args.resume:
                print(f"[resume] skipping completed batch {batch.batch_id}")
                continue
            if not args.force:
                raise SystemExit(
                    f"Batch {batch.batch_id} is already marked completed at {status_path}. "
                    f"Use --resume to skip it or --force to rerun it."
                )

        hh_tar_path = require_hh_tar(batch, args)
        allocation_job_id = require_slurm(batch, args)

        manifest = {
            "created_at": utc_now_iso(),
            "repo_root": str(REPO_ROOT),
            "run_root": str(run_root),
            "hh_tar_path": str(hh_tar_path) if hh_tar_path else None,
            "batch": batch_to_dict(batch),
            "resolved_args": {
                "curr": args.curr,
                "python_bin": args.python_bin,
                "allocation_job_id": allocation_job_id,
                "master_addr": derive_master_addr(args, allocation_job_id) if batch.requires_slurm else None,
                "master_port": args.master_port,
                "step_nnodes": args.step_nnodes,
                "nproc_per_node": args.nproc_per_node,
                "launch_backend": args.launch_backend,
                "data_access_mode": args.data_access_mode,
                "shared_data_dir": args.shared_data_dir,
            },
        }
        write_json(manifest_path, manifest)

        step_records: Dict[str, Dict[str, object]] = {}
        batch_status: Dict[str, object] = {
            "batch_id": batch.batch_id,
            "title": batch.title,
            "status": "running",
            "started_at": utc_now_iso(),
            "completed_at": None,
            "repo_root": str(REPO_ROOT),
            "run_root": str(run_root),
            "hh_tar_path": str(hh_tar_path) if hh_tar_path else None,
            "allocation_job_id": allocation_job_id,
            "steps": [],
        }
        write_json(status_path, batch_status)

        try:
            for cohort in step_cohorts(batch.steps):
                cohort_specs: List[Dict[str, object]] = []
                for step in cohort:
                    command, env = build_step_command(
                        args=args,
                        batch=batch,
                        step=step,
                        hh_tar_path=hh_tar_path,
                        allocation_job_id=allocation_job_id,
                        step_records=step_records,
                        run_root=run_root,
                        placeholder_ok=False,
                    )
                    env_subset = {
                        key: env[key]
                        for key in sorted(
                            key
                            for key in env.keys()
                            if key
                            in {
                                "ALLOC_JOB_ID",
                                "CURR",
                                "DATA_ACCESS_MODE",
                                "DATA_PREFIX",
                                "EVAL_ONLY_CHECKPOINT",
                                "LAUNCH_BACKEND",
                                "MASTER_ADDR",
                                "MASTER_PORT",
                                "NPROC_PER_NODE",
                                "PARAMS_FILE",
                                "RUN_ID",
                                "RUN_OUTPUT_ROOT",
                                "SAVE_PREDICTIONS",
                                "SHARED_DATA_DIR",
                                "SPLIT_EVAL_AFTER_TRAIN",
                                "STEP_NNODES",
                                "TAR_PATH",
                            }
                        )
                    }
                    print(f"[batch {batch.batch_id}] step {step.step_id}")
                    print(f"  command: {format_command(command)}")
                    if env_subset:
                        print(f"  env: {json.dumps(env_subset, sort_keys=True)}")

                    step_record: Dict[str, object] = {
                        "step_id": step.step_id,
                        "kind": step.kind,
                        "description": step.description,
                        "command": command,
                        "env_subset": env_subset,
                        "started_at": utc_now_iso(),
                        "completed_at": None,
                        "status": "running",
                    }
                    batch_status["steps"].append(step_record)
                    cohort_specs.append(
                        {
                            "step": step,
                            "command": command,
                            "env": env,
                            "step_record": step_record,
                        }
                    )
                write_json(status_path, batch_status)

                if args.dry_run:
                    continue

                active: List[Tuple[Dict[str, object], subprocess.Popen[object]]] = []
                for spec in cohort_specs:
                    proc = subprocess.Popen(
                        spec["command"],
                        cwd=str(REPO_ROOT),
                        env=spec["env"],
                    )
                    active.append((spec, proc))

                first_failure: Optional[subprocess.CalledProcessError] = None
                failure_step_id: Optional[str] = None
                for spec, proc in active:
                    rc = proc.wait()
                    step = spec["step"]
                    step_record = spec["step_record"]
                    if rc != 0:
                        if first_failure is None:
                            first_failure = subprocess.CalledProcessError(rc, spec["command"])
                            failure_step_id = step.step_id
                            for other_spec, other_proc in active:
                                if other_proc is not proc and other_proc.poll() is None:
                                    other_proc.terminate()
                        step_record["completed_at"] = utc_now_iso()
                        step_record["status"] = "failed"
                        step_record["failure"] = {
                            "returncode": rc,
                            "command": spec["command"],
                        }
                        write_json(status_path, batch_status)
                        continue

                    outputs = summarize_step_outputs(run_root, batch, step)
                    validate_step_outputs(args, run_root, batch, step, outputs)
                    step_record["outputs"] = outputs
                    step_record["completed_at"] = utc_now_iso()
                    step_record["status"] = "completed"
                    step_records[step.step_id] = outputs
                    write_json(status_path, batch_status)

                for spec, proc in active:
                    if proc.poll() is None:
                        proc.wait()
                    step_record = spec["step_record"]
                    if step_record.get("status") == "running":
                        rc = int(proc.returncode or 0)
                        if rc == 0:
                            continue
                        step_record["completed_at"] = utc_now_iso()
                        step_record["status"] = "failed"
                        step_record["failure"] = {
                            "returncode": rc,
                            "command": spec["command"],
                        }
                        write_json(status_path, batch_status)
                        if first_failure is None:
                            first_failure = subprocess.CalledProcessError(rc, spec["command"])
                            failure_step_id = spec["step"].step_id

                if first_failure is not None:
                    batch_status["status"] = "failed"
                    batch_status["completed_at"] = utc_now_iso()
                    batch_status["failed_step"] = failure_step_id
                    batch_status["failure"] = {
                        "returncode": first_failure.returncode,
                        "command": first_failure.cmd,
                    }
                    write_json(status_path, batch_status)
                    raise SystemExit(first_failure.returncode) from first_failure

            batch_status["status"] = "completed"
            batch_status["completed_at"] = utc_now_iso()
            write_json(status_path, batch_status)
        except subprocess.CalledProcessError as exc:
            batch_status["status"] = "failed"
            batch_status["completed_at"] = utc_now_iso()
            batch_status["failed_step"] = batch_status.get("failed_step")
            batch_status["failure"] = {
                "returncode": exc.returncode,
                "command": exc.cmd,
            }
            write_json(status_path, batch_status)
            raise SystemExit(exc.returncode) from exc
        except Exception as exc:
            batch_status["status"] = "failed"
            batch_status["completed_at"] = utc_now_iso()
            batch_status["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            write_json(status_path, batch_status)
            raise


def main() -> None:
    args = parse_args()
    if args.list:
        print_batch_listing()
        return

    batches = resolve_batches(args)
    if not batches:
        raise SystemExit("No batches were resolved from the provided arguments.")

    if args.dry_run:
        plan = resolved_plan(args, batches)
        print(json.dumps(plan, indent=2))
        return

    execute_batches(args, batches)


if __name__ == "__main__":
    main()
