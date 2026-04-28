#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260424_tc_hh_track4_rudi_closure_block"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"tc_hh_track4_rudi_closure_block_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"tc_hh_track4_rudi_closure_block_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"tc_hh_track4_rudi_closure_block_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "scripts" / "run_tc_hh_track4_dnn_step_20260424.sh"


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
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def resolve_node(job_id: str) -> str:
    proc = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%N"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"Could not resolve node for allocation {job_id}: {proc.stderr.strip()}")
    nodelist = proc.stdout.strip().splitlines()[0].strip()
    if "[" not in nodelist:
        return nodelist
    proc2 = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc2.returncode != 0 or not proc2.stdout.strip():
        raise RuntimeError(f"Could not expand node list for allocation {job_id}: {nodelist}")
    return proc2.stdout.strip().splitlines()[0].strip()


def registry_fields() -> List[str]:
    return [
        "row_id",
        "phase",
        "launch_mode",
        "launch_group",
        "strategy_id",
        "track_key",
        "seed",
        "status",
        "return_code",
        "alloc_job_id",
        "assigned_node",
        "started_at",
        "ended_at",
        "elapsed_sec",
        "run_output_root",
        "stdout_log",
        "stderr_log",
        "params_file",
        "shared_data_dir",
        "notes",
    ]


def make_run_output_root(row: Dict[str, str]) -> Path:
    return RUNS / row["phase"] / row["launch_mode"] / row["row_id"]


def make_logs(row: Dict[str, str]) -> tuple[Path, Path]:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"{row['row_id']}.out", LOGS / f"{row['row_id']}.err"


def build_env(row: Dict[str, str], run_output_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_output_root),
            "PARAMS_FILE": row["params_file"],
            "TAR_PATH": row["tar_path"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
            "SHARED_DATA_DIR": row["shared_data_dir"],
            "SAVE_PREDICTIONS": row["save_predictions"],
        }
    )
    return env


def build_srun_cmd(job_id: str, node: str, step_script: Path, row: Dict[str, str]) -> List[str]:
    return [
        "srun",
        f"--jobid={job_id}",
        "--exclusive",
        "--mem=0",
        "-N1",
        "-n1",
        f"-c{row['cpus_per_task']}",
        f"-w{node}",
        "bash",
        str(step_script),
    ]


def launch_one(row: Dict[str, str], job_id: str, node: str) -> subprocess.Popen:
    run_output_root = make_run_output_root(row)
    run_output_root.mkdir(parents=True, exist_ok=True)
    step_script_copy = run_output_root / STEP_SCRIPT.name
    shutil.copy2(STEP_SCRIPT, step_script_copy)
    stdout_log, stderr_log = make_logs(row)
    cmd = build_srun_cmd(job_id, node, step_script_copy, row)
    env = build_env(row, run_output_root)
    out = stdout_log.open("w")
    err = stderr_log.open("w")
    proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, text=True)
    proc._stdout_handle = out  # type: ignore[attr-defined]
    proc._stderr_handle = err  # type: ignore[attr-defined]
    return proc


def close_proc_files(proc: subprocess.Popen) -> None:
    for attr in ("_stdout_handle", "_stderr_handle"):
        handle = getattr(proc, attr, None)
        if handle is not None:
            handle.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--start-row-id")
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args()

    rows = load_csv(MATRIX_CSV)
    if not rows:
        raise SystemExit(f"Matrix not found or empty: {MATRIX_CSV}")
    if args.start_row_id:
        row_ids = [row["row_id"] for row in rows]
        if args.start_row_id not in row_ids:
            raise SystemExit(f"start-row-id not found in matrix: {args.start_row_id}")
        rows = rows[row_ids.index(args.start_row_id):]

    node = resolve_node(args.alloc_job_id)
    registry_rows: List[Dict[str, str]] = load_csv(REGISTRY_CSV)
    terminal_statuses = {"COMPLETED"} if args.rerun_failed else {"COMPLETED", "FAILED"}
    done = {row["row_id"] for row in registry_rows if row.get("status") in terminal_statuses}
    rows = [row for row in rows if row["row_id"] not in done]
    running: List[Dict[str, object]] = []

    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    run_log = {
        "started_at": now_iso(),
        "alloc_job_id": args.alloc_job_id,
        "assigned_node": node,
        "max_parallel": args.max_parallel,
        "rows_total": len(rows),
    }
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2) + "\n")

    def flush_registry() -> None:
        write_csv(REGISTRY_CSV, registry_rows, registry_fields())

    def poll_running(block: bool) -> None:
        nonlocal running
        while True:
            progressed = False
            next_running: List[Dict[str, object]] = []
            for item in running:
                proc: subprocess.Popen = item["proc"]  # type: ignore[assignment]
                ret = proc.poll()
                if ret is None:
                    next_running.append(item)
                    continue
                progressed = True
                close_proc_files(proc)
                ended_at = now_iso()
                elapsed = time.time() - float(item["start_ts"])
                row = item["row"]  # type: ignore[assignment]
                for reg_row in registry_rows:
                    if reg_row["row_id"] == row["row_id"]:
                        reg_row.update(
                            {
                                "status": "COMPLETED" if ret == 0 else "FAILED",
                                "return_code": str(ret),
                                "ended_at": ended_at,
                                "elapsed_sec": f"{elapsed:.3f}",
                            }
                        )
                        break
            running = next_running
            if progressed:
                flush_registry()
            if not block or progressed or not running:
                return
            time.sleep(5)

    for row in rows:
        while len(running) >= args.max_parallel:
            poll_running(block=True)
        run_output_root = make_run_output_root(row)
        stdout_log, stderr_log = make_logs(row)
        started_at = now_iso()
        proc = launch_one(row, args.alloc_job_id, node)
        found = False
        for reg_row in registry_rows:
            if reg_row["row_id"] == row["row_id"]:
                reg_row.update(
                    {
                        "phase": row["phase"],
                        "launch_mode": row["launch_mode"],
                        "launch_group": row["launch_group"],
                        "strategy_id": row["strategy_id"],
                        "track_key": row["track_key"],
                        "seed": row["seed"],
                        "status": "RUNNING",
                        "return_code": "",
                        "alloc_job_id": args.alloc_job_id,
                        "assigned_node": node,
                        "started_at": started_at,
                        "ended_at": "",
                        "elapsed_sec": "",
                        "run_output_root": str(run_output_root),
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                        "params_file": row["params_file"],
                        "shared_data_dir": row["shared_data_dir"],
                        "notes": row["notes"],
                    }
                )
                found = True
                break
        if not found:
            registry_rows.append(
                {
                    "row_id": row["row_id"],
                    "phase": row["phase"],
                    "launch_mode": row["launch_mode"],
                    "launch_group": row["launch_group"],
                    "strategy_id": row["strategy_id"],
                    "track_key": row["track_key"],
                    "seed": row["seed"],
                    "status": "RUNNING",
                    "return_code": "",
                    "alloc_job_id": args.alloc_job_id,
                    "assigned_node": node,
                    "started_at": started_at,
                    "ended_at": "",
                    "elapsed_sec": "",
                    "run_output_root": str(run_output_root),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                    "params_file": row["params_file"],
                    "shared_data_dir": row["shared_data_dir"],
                    "notes": row["notes"],
                }
            )
        flush_registry()
        running.append(
            {
                "proc": proc,
                "row": row,
                "start_ts": time.time(),
                "started_at": started_at,
                "run_output_root": run_output_root,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            }
        )
        time.sleep(1)

    while running:
        poll_running(block=True)

    run_log["ended_at"] = now_iso()
    run_log["rows_completed"] = sum(1 for r in registry_rows if r["status"] == "COMPLETED")
    run_log["rows_failed"] = sum(1 for r in registry_rows if r["status"] == "FAILED")
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2) + "\n")


if __name__ == "__main__":
    main()
