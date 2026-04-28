#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_tasks(path: Path, nodes_bucket: int) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    out = [r for r in rows if int(r["nodes"]) == nodes_bucket]
    out.sort(key=lambda r: int(r["task_id"]))
    return out


def read_completed(progress_csv: Path) -> set[int]:
    if not progress_csv.exists():
        return set()
    rows = list(csv.DictReader(progress_csv.open()))
    done = set()
    for r in rows:
        if (r.get("status") or "") == "COMPLETED":
            try:
                done.add(int(r["task_id"]))
            except Exception:
                pass
    return done


def append_progress(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "timestamp",
        "task_id",
        "task_slug",
        "track_key",
        "policy",
        "nodes",
        "batch_profile",
        "learning_rate",
        "features_sub_length",
        "seed",
        "params_file",
        "status",
        "return_code",
        "elapsed_sec",
        "runner_job_id",
        "notes",
    ]
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if not exists:
            w.writeheader()
        w.writerow(row)


def first_host_from_nodelist() -> str:
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        raise RuntimeError("Missing SLURM_NODELIST/SLURM_JOB_NODELIST")
    proc = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
        capture_output=True,
        check=False,
    )
    hosts = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if proc.returncode != 0 or not hosts:
        raise RuntimeError(f"Failed to resolve hostnames from nodelist={nodelist}")
    return hosts[0]


def run_task(
    *,
    task: Dict[str, str],
    repo: Path,
    master_addr: str,
    master_port: int,
    slurm_nnodes: int,
    slurm_cpus_per_task: int,
    runner_job_id: str,
    results_root: Path,
) -> subprocess.CompletedProcess:
    task_id = int(task["task_id"])
    params_file = task["params_file"]
    task_slug = task["task_slug"]

    task_save_base = results_root / f"nodes_{task['nodes']}" / f"task_{task_id:04d}_{task_slug}"
    task_save_base.mkdir(parents=True, exist_ok=True)

    # Unique rendezvous id per task.
    rdzv_id = f"{runner_job_id}_{task_id}"

    inner = "\n".join(
        [
            "set -euo pipefail",
            "module load Miniforge3",
            "source activate /projects/neuro-collab/conda/neuro-collab-env",
            f"REPO={shlex.quote(str(repo))}",
            "DLKIT=/projects/neuro-collab/code/dl-kit-main",
            "export PYTHONPATH=\"${DLKIT}:${REPO}:${PYTHONPATH:-}\"",
            "export PYTHONUNBUFFERED=1",
            "export TORCH_DIST_TIMEOUT_SECONDS=1800",
            "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
            "unset NCCL_ASYNC_ERROR_HANDLING || true",
            "export TORCH_NCCL_BLOCKING_WAIT=1",
            "export NCCL_DEBUG=WARN",
            "export NCCL_IB_DISABLE=0",
            "export NCCL_SOCKET_IFNAME=^lo,docker,veth",
            "export OMP_NUM_THREADS=$(( (${SLURM_CPUS_PER_TASK:-1}) / 2 ))",
            "export OMP_NUM_THREADS=$(( OMP_NUM_THREADS<1 ? 1 : OMP_NUM_THREADS ))",
            "cd \"${REPO}\"",
            # Force unique run directory per task by overriding the env var used by run_dnn.
            f"export SLURM_JOB_ID={shlex.quote(runner_job_id + '_' + str(task_id))}",
            "torchrun "
            f"--nnodes={slurm_nnodes} "
            "--node_rank=\"${SLURM_NODEID}\" "
            "--nproc_per_node=2 "
            "--rdzv_backend=c10d "
            f"--rdzv_id={shlex.quote(rdzv_id)} "
            f"--rdzv_endpoint={shlex.quote(master_addr + ':' + str(master_port))} "
            "pytorch/run_dnn.py "
            f"--params {shlex.quote(params_file)} "
            "--mode train_eval "
            f"--save_dir_base {shlex.quote(str(task_save_base))} "
            "--save_predictions test",
        ]
    )

    cmd = [
        "srun",
        "--export=ALL",
        f"--ntasks={slurm_nnodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={slurm_cpus_per_task}",
        "--kill-on-bad-exit=1",
        "bash",
        "-lc",
        inner,
    ]

    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run V100 large-matrix tasks sequentially within one Slurm allocation.")
    ap.add_argument("--task-csv", required=True)
    ap.add_argument("--nodes-bucket", type=int, required=True)
    ap.add_argument("--progress-csv", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--repo", default="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
    ap.add_argument("--max-tasks", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    repo = Path(args.repo)
    task_csv = Path(args.task_csv)
    progress_csv = Path(args.progress_csv)
    results_root = Path(args.results_root)
    summary_json = Path(args.summary_json)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "manual")
    slurm_nnodes = int(os.environ.get("SLURM_NNODES", str(args.nodes_bucket)))
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "24"))

    if slurm_nnodes != args.nodes_bucket:
        raise SystemExit(
            f"Runner nodes mismatch: allocation has {slurm_nnodes}, expected bucket {args.nodes_bucket}"
        )

    tasks = load_tasks(task_csv, args.nodes_bucket)
    already_done = read_completed(progress_csv)
    pending_tasks = [t for t in tasks if int(t["task_id"]) not in already_done]
    if args.max_tasks > 0:
        pending_tasks = pending_tasks[: args.max_tasks]

    master_addr = first_host_from_nodelist()

    run_started = datetime.now().astimezone().isoformat()
    completed = 0
    failed = 0

    for idx, task in enumerate(pending_tasks, start=1):
        task_id = int(task["task_id"])
        # Vary port to reduce chance of stale reuse collisions.
        base = int(os.environ.get("SLURM_JOB_ID", "0") or 0)
        master_port = 20000 + ((base + task_id) % 20000)

        t0 = time.time()
        proc = run_task(
            task=task,
            repo=repo,
            master_addr=master_addr,
            master_port=master_port,
            slurm_nnodes=slurm_nnodes,
            slurm_cpus_per_task=slurm_cpus_per_task,
            runner_job_id=slurm_job_id,
            results_root=results_root,
        )
        elapsed = time.time() - t0

        status = "COMPLETED" if proc.returncode == 0 else "FAILED"
        if status == "COMPLETED":
            completed += 1
        else:
            failed += 1

        notes = ""
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().replace("\n", " | ")
            out = (proc.stdout or "").strip().replace("\n", " | ")
            notes = f"stderr={err[:3000]} ; stdout={out[:3000]}"

        append_progress(
            progress_csv,
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "task_id": task["task_id"],
                "task_slug": task["task_slug"],
                "track_key": task["track_key"],
                "policy": task["policy"],
                "nodes": task["nodes"],
                "batch_profile": task["batch_profile"],
                "learning_rate": task["learning_rate"],
                "features_sub_length": task["features_sub_length"],
                "seed": task["seed"],
                "params_file": task["params_file"],
                "status": status,
                "return_code": str(proc.returncode),
                "elapsed_sec": f"{elapsed:.3f}",
                "runner_job_id": slurm_job_id,
                "notes": notes,
            },
        )

        print(
            json.dumps(
                {
                    "runner_job_id": slurm_job_id,
                    "nodes_bucket": args.nodes_bucket,
                    "task_index": idx,
                    "task_total": len(pending_tasks),
                    "task_id": task_id,
                    "status": status,
                    "return_code": proc.returncode,
                    "elapsed_sec": round(elapsed, 3),
                }
            ),
            flush=True,
        )

    run_ended = datetime.now().astimezone().isoformat()
    summary = {
        "runner_job_id": slurm_job_id,
        "nodes_bucket": args.nodes_bucket,
        "run_started": run_started,
        "run_ended": run_ended,
        "tasks_in_bucket": len(tasks),
        "tasks_already_completed_before_run": len(already_done),
        "tasks_executed_this_run": len(pending_tasks),
        "completed_this_run": completed,
        "failed_this_run": failed,
        "progress_csv": str(progress_csv),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
