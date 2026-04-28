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
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"tc_hh_track4_aligned_multicurrent_sbi_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_track4_aligned_multicurrent_sbi_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_hh_track4_aligned_multicurrent_sbi_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "tools" / "run_v100_hh_track4_aligned_multicurrent_sbi_step_20260425.sh"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hostnames(nodelist: str) -> List[str]:
    proc = subprocess.run(["scontrol", "show", "hostnames", nodelist], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to resolve hostnames for {nodelist}: {proc.stderr}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


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
    ap.add_argument("--alloc-nodelist")
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args()

    rows = load_csv(MATRIX_CSV)
    if not rows:
        raise SystemExit(f"Empty matrix: {MATRIX_CSV}")
    registry = load_csv(REGISTRY_CSV)
    terminal_statuses = {"COMPLETED"} if args.rerun_failed else {"COMPLETED", "FAILED"}
    done = {row["row_id"] for row in registry if row.get("status") in terminal_statuses}

    if args.alloc_nodelist:
        hosts = hostnames(args.alloc_nodelist)
    else:
        proc = subprocess.run(
            ["scontrol", "show", "job", str(args.alloc_job_id)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"Could not inspect allocation {args.alloc_job_id}: {proc.stderr}")
        token = next((part.split("=", 1)[1] for part in proc.stdout.split() if part.startswith("NodeList=")), None)
        if not token:
            raise SystemExit(f"Could not resolve NodeList from allocation {args.alloc_job_id}")
        hosts = hostnames(token)

    LOGS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    RUN_LOG_JSON.write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "alloc_job_id": args.alloc_job_id,
                "hosts": hosts,
                "rows_total": len(rows),
                "rows_already_terminal": len(done),
            },
            indent=2,
        )
        + "\n"
    )

    active = []
    remaining = [row for row in rows if row["row_id"] not in done]
    host_slots = hosts[: min(len(hosts), 3)]

    def launch_one(row: Dict[str, str], host: str):
        nonlocal registry
        stdout_log = LOGS / f"{row['row_id']}.out"
        stderr_log = LOGS / f"{row['row_id']}.err"
        started_at = now_iso()
        proc = subprocess.Popen(
            [
                "srun",
                f"--jobid={args.alloc_job_id}",
                "--exclusive",
                "--mem=0",
                "-N1",
                "-n1",
                "-c12",
                f"-w{host}",
                "--gres=gpu:v100:1",
                "bash",
                str(STEP_SCRIPT),
            ],
            text=True,
            stdout=stdout_log.open("w"),
            stderr=stderr_log.open("w"),
            env=build_env(row),
        )
        active.append(
            {
                "proc": proc,
                "row": row,
                "host": host,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "started_at": started_at,
                "start_ts": time.time(),
            }
        )
        registry = [reg for reg in registry if reg["row_id"] != row["row_id"]]
        registry.append(
            {
                "row_id": row["row_id"],
                "strategy_id": row["strategy_id"],
                "method": row["method"],
                "aggregation": row["aggregation"],
                "feature_mode": row["feature_mode"],
                "status": "RUNNING",
                "return_code": "",
                "alloc_job_id": args.alloc_job_id,
                "assigned_node": host,
                "started_at": started_at,
                "ended_at": "",
                "elapsed_sec": "",
                "save_dir": row["save_dir"],
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "notes": row["notes"],
            }
        )
        write_csv(REGISTRY_CSV, registry, registry_fields())

    def poll():
        nonlocal active, registry
        next_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                next_active.append(item)
                continue
            for reg in registry:
                if reg["row_id"] == item["row"]["row_id"]:
                    reg["status"] = "COMPLETED" if ret == 0 else "FAILED"
                    reg["return_code"] = str(ret)
                    reg["ended_at"] = now_iso()
                    reg["elapsed_sec"] = f"{time.time() - item['start_ts']:.3f}"
                    break
            write_csv(REGISTRY_CSV, registry, registry_fields())
        active = next_active

    while remaining or active:
        while remaining and len(active) < len(host_slots):
            busy_hosts = {item["host"] for item in active}
            free = next((h for h in host_slots if h not in busy_hosts), None)
            if free is None:
                break
            launch_one(remaining.pop(0), free)
            time.sleep(1)
        time.sleep(5)
        poll()


if __name__ == "__main__":
    main()
