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

DATE_TAG = "20260425_tc_hh_track4_aligned_multicurrent_sbi"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"tc_hh_track4_aligned_multicurrent_sbi_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"tc_hh_track4_aligned_multicurrent_sbi_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"tc_hh_track4_aligned_multicurrent_sbi_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "scripts" / "run_tc_hh_track4_aligned_multicurrent_sbi_20260425.sh"


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
    return proc.stdout.strip().splitlines()[0].strip()


def registry_fields() -> List[str]:
    return [
        "row_id",
        "strategy_id",
        "method",
        "aggregation",
        "feature_mode",
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
    env.update(
        {
            "PARAMS_FILE": row["params_file"],
            "METHOD": row["method"],
            "AGGREGATION": row["aggregation"],
            "FEATURE_MODE": row["feature_mode"],
            "SAVE_DIR": row["save_dir"],
            "SEED": row["seed"],
            "N_TRAIN": row["n_train"],
            "N_VALIDATE": row["n_validate"],
            "N_TEST": row["n_test"],
            "CURRENTS": row["currents"],
        }
    )
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-job-id", required=True)
    args = ap.parse_args()

    rows = load_csv(MATRIX_CSV)
    if not rows:
        raise SystemExit(f"Empty matrix: {MATRIX_CSV}")

    node = resolve_node(args.alloc_job_id)
    registry = load_csv(REGISTRY_CSV)
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
        ended_at = now_iso()
        registry = [r for r in registry if r["row_id"] != row["row_id"]]
        registry.append(
            {
                "row_id": row["row_id"],
                "strategy_id": row["strategy_id"],
                "method": row["method"],
                "aggregation": row["aggregation"],
                "feature_mode": row["feature_mode"],
                "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
                "return_code": str(proc.returncode),
                "alloc_job_id": args.alloc_job_id,
                "assigned_node": node,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_sec": "",
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
