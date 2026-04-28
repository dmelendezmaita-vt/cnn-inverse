#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260425_tc_hh_track4_pool_sequential_sbi"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"tc_hh_track4_pool_sequential_sbi_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"tc_hh_track4_pool_sequential_sbi_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"tc_hh_track4_pool_sequential_sbi_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "scripts" / "run_tc_hh_track4_pool_sequential_sbi_20260425.sh"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_node(job_id: str) -> str:
    proc = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%N"], text=True, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"Could not resolve node for allocation {job_id}")
    nodelist = proc.stdout.strip().splitlines()[0].strip()
    if "[" not in nodelist:
        return nodelist
    proc2 = subprocess.run(["scontrol", "show", "hostnames", nodelist], text=True, capture_output=True, check=False)
    if proc2.returncode != 0 or not proc2.stdout.strip():
        raise RuntimeError(f"Could not expand node list for allocation {job_id}: {nodelist}")
    return proc2.stdout.strip().splitlines()[0].strip()


def registry_fields() -> List[str]:
    return [
        "row_id",
        "strategy_id",
        "method",
        "density_estimator",
        "rounds",
        "final_train_size",
        "status",
        "return_code",
        "alloc_job_id",
        "assigned_node",
        "started_at",
        "ended_at",
        "elapsed_sec",
        "save_dir",
        "stdout_log",
        "stderr_log",
        "notes",
    ]


def build_env(row: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    mapping = {
        "PARAMS_FILE": "params_file",
        "METHOD": "method",
        "DENSITY_ESTIMATOR": "density_estimator",
        "FEATURE_MODE": "feature_mode",
        "SAVE_DIR": "save_dir",
        "DATA_DIR": "data_dir",
        "DATA_PREFIX": "data_prefix",
        "CURR": "curr",
        "SEED": "seed",
        "POOL_SIZE": "pool_size",
        "FINAL_TRAIN_SIZE": "final_train_size",
        "N_VALIDATE": "n_validate",
        "N_TEST": "n_test",
        "ROUNDS": "rounds",
        "INITIAL_TRAIN_SIZE": "initial_train_size",
        "ROUND_TRAIN_SIZES": "round_train_sizes",
        "CANDIDATE_POOL_SIZE": "candidate_pool_size",
        "CANDIDATE_POSTERIOR_SAMPLES": "candidate_posterior_samples",
        "POSTERIOR_SAMPLES": "posterior_samples",
        "TRAINING_BATCH_SIZE": "training_batch_size",
        "LEARNING_RATE": "learning_rate",
        "STOP_AFTER_EPOCHS": "stop_after_epochs",
        "MAX_NUM_EPOCHS": "max_num_epochs",
        "EMBEDDING_DIM": "embedding_dim",
        "EMBEDDING_HIDDEN": "embedding_hidden",
        "HIDDEN_FEATURES": "hidden_features",
        "NUM_TRANSFORMS": "num_transforms",
        "DECISION_RULE": "decision_rule",
        "SAMPLE_WITH": "sample_with",
    }
    for env_key, row_key in mapping.items():
        value = row.get(row_key, "")
        if value != "":
            env[env_key] = value
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args()

    rows = load_csv(MATRIX_CSV)
    if not rows:
        raise SystemExit(f"Empty matrix: {MATRIX_CSV}")

    node = resolve_node(args.alloc_job_id)
    registry = load_csv(REGISTRY_CSV)
    if args.rerun_failed:
        done = {row["row_id"] for row in registry if row.get("status") == "COMPLETED"}
    else:
        done = {row["row_id"] for row in registry if row.get("status") in {"COMPLETED", "FAILED"}}

    LOGS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    RUN_LOG_JSON.write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "alloc_job_id": args.alloc_job_id,
                "assigned_node": node,
                "rows_total": len(rows),
                "rows_already_terminal": len(done),
                "rerun_failed": bool(args.rerun_failed),
            },
            indent=2,
        )
        + "\n"
    )

    for row in rows:
        if row["row_id"] in done:
            continue
        stdout_log = LOGS / f"{row['row_id']}.out"
        stderr_log = LOGS / f"{row['row_id']}.err"
        started_at = now_iso()
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "srun",
                f"--jobid={args.alloc_job_id}",
                "--exclusive",
                "--mem=0",
                "-N1",
                "-n1",
                "-c4",
                f"-w{node}",
                "bash",
                str(STEP_SCRIPT),
            ],
            text=True,
            stdout=stdout_log.open("w"),
            stderr=stderr_log.open("w"),
            env=build_env(row),
            check=False,
        )
        elapsed_sec = time.perf_counter() - t0
        ended_at = now_iso()

        registry = [existing for existing in registry if existing["row_id"] != row["row_id"]]
        registry.append(
            {
                "row_id": row["row_id"],
                "strategy_id": row["strategy_id"],
                "method": row["method"],
                "density_estimator": row["density_estimator"],
                "rounds": row["rounds"],
                "final_train_size": row["final_train_size"],
                "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
                "return_code": str(proc.returncode),
                "alloc_job_id": args.alloc_job_id,
                "assigned_node": node,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_sec": f"{elapsed_sec:.3f}",
                "save_dir": row["save_dir"],
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "notes": row["notes"],
            }
        )
        write_csv(REGISTRY_CSV, registry, registry_fields())
        time.sleep(1)


if __name__ == "__main__":
    main()
