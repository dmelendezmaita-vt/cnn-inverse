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
    ap.add_argument("--data-access-mode", default="copy_to_node", choices=["copy_to_node", "tar_in_place"], help="Data staging policy for Falcon DNN batches.")
    ap.add_argument("--shared-data-dir", default=None, help="Optional shared extracted-data directory for Falcon batches.")
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
    candidates = sorted(path.rglob("net_e*.pt"))
    if not candidates:
        return None
    return candidates[-1]


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
    env["PYTHONUNBUFFERED"] = "1"
    env["REPO_ROOT"] = str(REPO_ROOT)
    if batch.requires_slurm and hh_tar_path is not None:
        env["ALLOC_JOB_ID"] = allocation_job_id
        env["RUN_ID"] = step.step_id
        env["RUN_OUTPUT_ROOT"] = str(step_output_root(run_root, batch, step))
        env["PARAMS_FILE"] = step.params_file
        env["TAR_PATH"] = str(hh_tar_path)
        env["DATA_PREFIX"] = derive_data_prefix(hh_tar_path)
        env["MASTER_ADDR"] = derive_master_addr(args, allocation_job_id)
        env["MASTER_PORT"] = str(args.master_port)
        env["STEP_NNODES"] = str(step.step_nnodes or args.step_nnodes)
        env["NPROC_PER_NODE"] = str(step.nproc_per_node or args.nproc_per_node)
        env["DATA_ACCESS_MODE"] = str(args.data_access_mode)
        env["LAUNCH_BACKEND"] = str(args.launch_backend)
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
        base_command = [
            args.python_bin,
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
        command = wrap_single_node_srun(args, allocation_job_id, base_command) if batch.requires_slurm else list(base_command)
        return command, env

    if step.kind == "classical":
        base_command = [
            args.python_bin,
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
        command = wrap_single_node_srun(args, allocation_job_id, base_command) if batch.requires_slurm else list(base_command)
        return command, env

    if step.kind == "sbi":
        base_command = [
            args.python_bin,
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
        command = wrap_single_node_srun(args, allocation_job_id, base_command) if batch.requires_slurm else list(base_command)
        return command, env

    if step.kind == "script":
        if not step.script_path:
            raise SystemExit(f"Batch step {batch.batch_id}:{step.step_id} is missing script_path.")
        base_command = [
            args.python_bin,
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
        command = wrap_single_node_srun(args, allocation_job_id, base_command) if batch.requires_slurm else list(base_command)
        return command, env

    if step.kind == "interactive_dnn":
        nnodes = int(step.step_nnodes or args.step_nnodes)
        command = [
            args.srun_bin,
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
    return outputs


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
            for step in batch.steps:
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
                write_json(status_path, batch_status)

                if not args.dry_run:
                    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)

                outputs = summarize_step_outputs(run_root, batch, step)
                step_record["outputs"] = outputs
                step_record["completed_at"] = utc_now_iso()
                step_record["status"] = "completed"
                step_records[step.step_id] = outputs
                write_json(status_path, batch_status)

            batch_status["status"] = "completed"
            batch_status["completed_at"] = utc_now_iso()
            write_json(status_path, batch_status)
        except subprocess.CalledProcessError as exc:
            batch_status["status"] = "failed"
            batch_status["completed_at"] = utc_now_iso()
            batch_status["failure"] = {
                "returncode": exc.returncode,
                "command": exc.cmd,
            }
            write_json(status_path, batch_status)
            raise SystemExit(exc.returncode) from exc


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
