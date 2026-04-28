#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATE_TAG = "20260413_v100_hh_scaling_replication_r2"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_scaling_replication_r2_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_scaling_replication_r2_registry_{DATE_TAG}.csv"
STATE_JSON = NOTES / f"v100_hh_scaling_replication_r2_driver_state_{DATE_TAG}.json"
LAUNCHER = REPO / "tools" / "launch_v100_hh_scaling_replication_r2_20260413.py"
RECONCILE = REPO / "tools" / "reconcile_v100_hh_scaling_replication_r2_20260413.py"
SUMMARIZE = REPO / "tools" / "summarize_v100_hh_scaling_replication_r2_20260413.py"

FALCON_HOST = "falcon1.arc.vt.edu"
TOPIC = "dmelendezmaita"
ALLOC_IDS = ["353562", "355055"]
POLL_SEC = 60


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ssh_capture(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", FALCON_HOST, cmd],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"ssh command failed: {cmd}")
    return proc


def local_capture(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"command failed: {cmd}")
    return proc


def send_ntfy(title: str, message: str) -> None:
    subprocess.run(
        [
            "curl",
            "-fsS",
            "-H",
            f"Title: {title}",
            "-d",
            message,
            f"https://ntfy.sh/{TOPIC}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def parse_job_info(job_id: str) -> dict:
    proc = ssh_capture(f"scontrol show job {job_id} 2>/dev/null || true")
    txt = proc.stdout or ""
    info = {"job_id": job_id, "raw": txt, "state": None, "node_list": "", "sched_node_list": "", "start_time": "", "end_time": ""}
    if not txt.strip():
        return info
    m = re.search(r"JobState=(\S+)", txt)
    if m:
        info["state"] = m.group(1)
    m = re.search(r"\bNodeList=([^\s]+)", txt)
    if m:
        info["node_list"] = m.group(1)
    m = re.search(r"\bSchedNodeList=([^\s]+)", txt)
    if m:
        info["sched_node_list"] = m.group(1)
    m = re.search(r"\bStartTime=([^\s]+)", txt)
    if m:
        info["start_time"] = m.group(1)
    m = re.search(r"\bEndTime=([^\s]+)", txt)
    if m:
        info["end_time"] = m.group(1)
    return info


def active_steps(job_id: str) -> list[str]:
    proc = ssh_capture(f"squeue -s -h -j {job_id} -o '%i'")
    steps = []
    for ln in (proc.stdout or "").splitlines():
        ln = ln.strip()
        if re.fullmatch(r"\d+\.\d+", ln):
            steps.append(ln)
    return steps


def choose_running_allocation() -> dict | None:
    for job_id in ALLOC_IDS:
        info = parse_job_info(job_id)
        if info["state"] == "RUNNING":
            return info
    return None


def next_incomplete_row_id() -> str | None:
    matrix_rows = load_csv(MATRIX_CSV)
    completed = {row["row_id"] for row in load_csv(REGISTRY_CSV) if row.get("status") == "COMPLETED"}
    for row in matrix_rows:
        if row["row_id"] not in completed:
            return row["row_id"]
    return None


def reconcile_and_summarize() -> None:
    local_capture([sys.executable, str(RECONCILE)], check=False)
    local_capture([sys.executable, str(SUMMARIZE)], check=False)


def save_state(state: dict) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")


def launch_row(job_id: str, nodelist: str, row_id: str) -> subprocess.CompletedProcess:
    cmd = (
        f"cd {shlex.quote(str(REPO))} && "
        f"python {shlex.quote(str(LAUNCHER.relative_to(REPO)))} "
        f"--alloc-job-id {shlex.quote(job_id)} "
        f"--alloc-nodelist {shlex.quote(nodelist)} "
        f"--row-id {shlex.quote(row_id)}"
    )
    return ssh_capture(cmd, check=False)


def main() -> int:
    send_ntfy(
        "neuro-collab v100 R2 driver",
        "Started persistent R2 sequential driver. It will keep launching incomplete rows in package 20260413, using V100 allocation 353562 first and switching to 355055 when it starts.",
    )

    while True:
        reconcile_and_summarize()
        row_id = next_incomplete_row_id()
        running = choose_running_allocation()

        state = {
            "timestamp": now_iso(),
            "next_incomplete_row_id": row_id,
            "running_allocation": running,
            "allocations": [parse_job_info(job_id) for job_id in ALLOC_IDS],
        }
        save_state(state)

        if row_id is None:
            send_ntfy(
                "neuro-collab v100 R2 driver",
                "All rows in the 20260413 V100 HH scaling replication package are now complete. The driver is exiting.",
            )
            return 0

        if running is None:
            time.sleep(POLL_SEC)
            continue

        steps = active_steps(running["job_id"])
        if steps:
            time.sleep(POLL_SEC)
            continue

        nodelist = running["node_list"]
        if not nodelist or nodelist in {"(null)", "None"}:
            time.sleep(POLL_SEC)
            continue

        send_ntfy(
            "neuro-collab v100 R2 driver",
            f"Launching next incomplete R2 row {row_id} in running allocation {running['job_id']} on {nodelist}.",
        )
        proc = launch_row(running["job_id"], nodelist, row_id)
        reconcile_and_summarize()

        send_ntfy(
            "neuro-collab v100 R2 driver",
            f"Launcher returned for row {row_id} in allocation {running['job_id']} with rc={proc.returncode}. The driver will continue with the next incomplete row when the allocation is free.",
        )

        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
