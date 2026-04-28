#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

DATE_TAG = "20260416_v100_hh_datapath_redesign_r4"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
RUNS = OUT_ROOT / "runs"

MATRIX_CSV = TABLES / f"v100_hh_datapath_redesign_r4_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_datapath_redesign_r4_registry_{DATE_TAG}.csv"
RECONCILE = REPO / "tools" / "reconcile_v100_hh_datapath_redesign_r4_20260416.py"
SUMMARIZE = REPO / "tools" / "summarize_v100_hh_datapath_redesign_r4_20260416.py"
STEP_SCRIPT = REPO / "tools" / "run_v100_interactive_step_20260411.sh"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_matches(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.row_ids and row["row_id"] not in args.row_ids:
        return False
    if args.phases and row["phase"] not in args.phases:
        return False
    if args.strategies and row["strategy_id"] not in args.strategies:
        return False
    return True


def row_run_root(row: dict[str, str]) -> Path:
    return RUNS / row["phase"] / row["launch_mode"] / row["row_id"]


def row_metrics_exist(row: dict[str, str]) -> bool:
    run_root = row_run_root(row)
    if not run_root.exists():
        return False
    return any(run_root.rglob("metrics_summary.json"))


def row_params(row: dict[str, str]) -> dict | None:
    try:
        return yaml.safe_load((REPO / row["params_file"]).read_text())
    except Exception:
        return None


def row_checkpoint_path(row: dict[str, str]) -> str | None:
    run_root = row_run_root(row)
    if not run_root.exists():
        return None

    params = row_params(row) or {}
    training_epochs = int(params.get("training", {}).get("epochs", 0) or 0)
    checkpoint_every = params.get("runconfig", {}).get("save_checkpoints_epochs", None)
    try:
        checkpoint_every = int(checkpoint_every) if checkpoint_every is not None else None
    except Exception:
        checkpoint_every = None

    candidates: list[tuple[int, Path]] = []
    for ckpt in run_root.rglob("net_e*.pt"):
        m = re.search(r"net_e(\d+)\.pt$", ckpt.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), ckpt))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    if training_epochs > 0 and checkpoint_every and checkpoint_every > 0:
        min_epoch = max(checkpoint_every, training_epochs - checkpoint_every)
        viable = [item for item in candidates if item[0] >= min_epoch]
        if viable:
            return str(viable[-1][1])
        return None
    return str(candidates[-1][1])


def hostnames(nodelist: str) -> list[str]:
    proc = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or f"failed to expand nodelist {nodelist}")
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def active_steps(job_id: str) -> list[str]:
    proc = subprocess.run(
        ["squeue", "-s", "-h", "-j", str(job_id), "-o", "%i"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    steps = []
    for ln in (proc.stdout or "").splitlines():
        ln = ln.strip()
        if re.fullmatch(r"\d+\.\d+", ln):
            steps.append(ln)
    return steps


def clear_active_steps(job_id: str) -> list[str]:
    cleared = []
    for step_id in active_steps(job_id):
        subprocess.run(["scancel", step_id], text=True, capture_output=True, check=False)
        cleared.append(step_id)
    return cleared


def resolve_allocation(
    row: dict[str, str],
    registry_row: dict[str, str] | None,
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    job_id = args.alloc_job_id
    nodelist = args.alloc_nodelist
    if not job_id and registry_row:
        job_id = registry_row.get("alloc_job_id") or None
    if not nodelist and registry_row:
        nodelist = registry_row.get("alloc_nodelist") or None
    if not job_id:
        job_id = os.environ.get("SLURM_JOB_ID")
    if not nodelist:
        nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    return job_id, nodelist


def salvage_eval_only(job_id: str, nodelist: str, row: dict[str, str], checkpoint_path: str) -> int:
    hosts = hostnames(nodelist)
    first_host = hosts[0] if hosts else nodelist.split(",", 1)[0]
    run_output_root = row_run_root(row)

    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "ALLOC_JOB_ID": str(job_id),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_output_root),
            "PARAMS_FILE": row["params_file"],
            "TAR_PATH": row["tar_path"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "STEP_NNODES": "1",
            "NPROC_PER_NODE": "1",
            "DATA_ACCESS_MODE": row["data_access_mode"],
            "SHARED_DATA_DIR": row["shared_data_dir"],
            "SAVE_PREDICTIONS": row["save_predictions"],
            "SPLIT_EVAL_AFTER_TRAIN": "0",
            "EVAL_ONLY_CHECKPOINT": checkpoint_path,
            "FOLLOWUP_TMP_BASE": "/projects/neuro-collab/other/v100if_tmp",
        }
    )
    cmd = [
        "srun",
        f"--jobid={job_id}",
        "--exclusive",
        "-N1",
        f"-w{first_host}",
        "--ntasks=1",
        f"--cpus-per-task={row['cpus_per_node']}",
        "--gres=gpu:v100:1",
        "bash",
        str(STEP_SCRIPT),
    ]
    proc = subprocess.run(cmd, text=True, check=False, env=env)
    return proc.returncode


def reconcile_and_summarize() -> None:
    subprocess.run([sys.executable, str(RECONCILE)], check=False)
    subprocess.run([sys.executable, str(SUMMARIZE)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--clear-active-steps", action="store_true")
    args = ap.parse_args()

    matrix_rows = [row for row in load_csv(MATRIX_CSV) if row_matches(row, args)]
    registry_rows = {row["row_id"]: row for row in load_csv(REGISTRY_CSV)}

    report = {
        "date_tag": DATE_TAG,
        "selected_rows": len(matrix_rows),
        "already_complete": [],
        "salvaged": [],
        "no_checkpoint": [],
        "missing_allocation": [],
        "failed": [],
        "cleared_steps": {},
    }

    for row in matrix_rows:
        if row_metrics_exist(row):
            report["already_complete"].append(row["row_id"])
            continue

        checkpoint_path = row_checkpoint_path(row)
        if checkpoint_path is None:
            report["no_checkpoint"].append(row["row_id"])
            continue

        registry_row = registry_rows.get(row["row_id"])
        job_id, nodelist = resolve_allocation(row, registry_row, args)
        if not job_id or not nodelist:
            report["missing_allocation"].append(row["row_id"])
            continue

        if args.clear_active_steps:
            cleared = clear_active_steps(job_id)
            if cleared:
                report["cleared_steps"][row["row_id"]] = cleared

        rc = salvage_eval_only(job_id, nodelist, row, checkpoint_path)
        reconcile_and_summarize()
        if rc == 0 and row_metrics_exist(row):
            report["salvaged"].append(row["row_id"])
        else:
            report["failed"].append({"row_id": row["row_id"], "return_code": rc})

    reconcile_and_summarize()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
