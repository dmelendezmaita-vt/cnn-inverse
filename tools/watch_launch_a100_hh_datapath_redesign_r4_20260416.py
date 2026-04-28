#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATE_TAG = "20260416_a100_hh_datapath_redesign_r4"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
NOTES = OUT_ROOT / "notes"
STATE_JSON = NOTES / f"a100_hh_datapath_redesign_r4_watch_state_{DATE_TAG}.json"
LAUNCHER = REPO / "tools" / "launch_a100_hh_datapath_redesign_r4_20260416.py"
RECONCILE = REPO / "tools" / "reconcile_a100_hh_datapath_redesign_r4_20260416.py"
SUMMARIZE = REPO / "tools" / "summarize_a100_hh_datapath_redesign_r4_20260416.py"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_state(state: dict) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")


def show_job(job_id: str) -> str:
    proc = subprocess.run(["scontrol", "show", "job", str(job_id)], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to query job {job_id}")
    return proc.stdout


def parse_field(raw: str, key: str) -> str:
    token = f"{key}="
    for part in raw.split():
        if part.startswith(token):
            return part[len(token):]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--poll-sec", type=float, default=60.0)
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    args = ap.parse_args()

    job_id = str(args.job_id)
    launch_cmd = [
        sys.executable,
        str(LAUNCHER),
        "--alloc-job-id",
        job_id,
    ]
    for phase in args.phases or []:
        launch_cmd.extend(["--phase", phase])
    for strategy in args.strategies or []:
        launch_cmd.extend(["--strategy", strategy])

    while True:
        raw = show_job(job_id)
        state = parse_field(raw, "JobState")
        nodelist = parse_field(raw, "NodeList")
        sched_nodelist = parse_field(raw, "SchedNodeList")
        save_state(
            {
                "timestamp": now_iso(),
                "job_id": job_id,
                "job_state": state,
                "node_list": nodelist,
                "sched_node_list": sched_nodelist,
                "launch_cmd": launch_cmd,
            }
        )

        if state == "RUNNING" and nodelist and nodelist != "(null)":
            launch_cmd_full = launch_cmd + ["--alloc-nodelist", nodelist]
            subprocess.run(launch_cmd_full, cwd=str(REPO), check=True)
            subprocess.run([sys.executable, str(RECONCILE)], cwd=str(REPO), check=False)
            subprocess.run([sys.executable, str(SUMMARIZE)], cwd=str(REPO), check=False)
            save_state(
                {
                    "timestamp": now_iso(),
                    "job_id": job_id,
                    "job_state": "LAUNCHED_AND_SUMMARIZED",
                    "node_list": nodelist,
                    "sched_node_list": sched_nodelist,
                    "launch_cmd": launch_cmd_full,
                }
            )
            return

        if state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "BOOT_FAIL", "NODE_FAIL"}:
            save_state(
                {
                    "timestamp": now_iso(),
                    "job_id": job_id,
                    "job_state": state,
                    "node_list": nodelist,
                    "sched_node_list": sched_nodelist,
                    "launch_cmd": launch_cmd,
                    "error": "Allocation ended before automatic launch could begin.",
                }
            )
            raise SystemExit(f"Allocation {job_id} ended in state={state} before launch.")

        time.sleep(max(5.0, args.poll_sec))


if __name__ == "__main__":
    main()
