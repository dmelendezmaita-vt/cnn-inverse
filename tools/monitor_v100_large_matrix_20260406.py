#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260406_v100_large"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}/submission"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

RUNNER_JOBS_CSV = TABLES / f"v100_large_runner_jobs_registry_{DATE_TAG}.csv"
TASK_MATRIX_CSV = TABLES / f"v100_large_task_matrix_{DATE_TAG}.csv"
RUNNER_STATUS_SNAPSHOT_CSV = TABLES / f"v100_large_runner_status_snapshot_{DATE_TAG}.csv"
SUMMARY_JSON = NOTES / f"v100_large_status_summary_{DATE_TAG}.json"


def run_falcon_bash(cmd: str) -> subprocess.CompletedProcess:
    escaped = (
        cmd.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    full_cmd = (
        'ssh -n -o BatchMode=yes tinkercliffs1 '
        f'"ssh -n -o BatchMode=yes falcon1.arc.vt.edu \\"{escaped}\\""'
    )
    return subprocess.run(full_cmd, shell=True, text=True, capture_output=True)


def normalize_state(state: str) -> str:
    s = (state or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if " " in s:
        s = s.split(" ", 1)[0]
    return s


def classify(state: str) -> str:
    s = normalize_state(state)
    if s in {"PENDING", "CONFIGURING", "REQUEUE_HOLD", "RESV_DEL_HOLD", "SUSPENDED"}:
        return "PENDING"
    if s in {"RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING"}:
        return "RUNNING"
    if s == "COMPLETED":
        return "COMPLETED"
    if s in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}:
        return "FAILED"
    if s == "UNKNOWN":
        return "UNKNOWN"
    return "OTHER"


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
        for r in rows:
            w.writerow(r)


def query_runner_states(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if not job_ids:
        return {}
    ids_arg = ",".join(job_ids)

    sq_cmd = f"squeue -h -j {ids_arg} -o '%i|%T|%M|%L|%R'"
    sq_proc = run_falcon_bash(sq_cmd)
    sq_rows: Dict[str, Dict[str, str]] = {}
    if sq_proc.returncode == 0:
        for ln in (sq_proc.stdout or "").splitlines():
            t = ln.strip().split("|", 4)
            if len(t) != 5:
                continue
            jid, state, elapsed, left, reason = t
            if jid.isdigit():
                sq_rows[jid] = {
                    "state": normalize_state(state),
                    "elapsed": elapsed,
                    "time_left": left,
                    "reason": reason,
                    "source": "squeue",
                    "exit_code": "",
                }

    sa_cmd = f"sacct -n -P -X -j {ids_arg} --format=JobIDRaw,State,Elapsed,ExitCode"
    sa_proc = run_falcon_bash(sa_cmd)
    sa_rows: Dict[str, Dict[str, str]] = {}
    if sa_proc.returncode == 0:
        for ln in (sa_proc.stdout or "").splitlines():
            t = ln.strip().split("|", 3)
            if len(t) != 4:
                continue
            jid, state, elapsed, exit_code = t
            if jid.isdigit():
                sa_rows[jid] = {
                    "state": normalize_state(state),
                    "elapsed": elapsed,
                    "time_left": "",
                    "reason": "",
                    "source": "sacct",
                    "exit_code": exit_code,
                }

    out: Dict[str, Dict[str, str]] = {}
    for jid in job_ids:
        if jid in sq_rows:
            out[jid] = sq_rows[jid]
        elif jid in sa_rows:
            out[jid] = sa_rows[jid]
        else:
            out[jid] = {
                "state": "UNKNOWN",
                "elapsed": "",
                "time_left": "",
                "reason": "",
                "source": "none",
                "exit_code": "",
            }
    return out


def task_progress_counts(runner_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for rr in runner_rows:
        rid = rr["runner_id"]
        p = Path(rr["progress_csv"])
        if not p.exists():
            counts[rid] = {"COMPLETED": 0, "FAILED": 0, "TOTAL_ROWS": 0}
            continue
        rows = list(csv.DictReader(p.open()))
        c = Counter((r.get("status") or "UNKNOWN") for r in rows)
        counts[rid] = {
            "COMPLETED": int(c.get("COMPLETED", 0)),
            "FAILED": int(c.get("FAILED", 0)),
            "TOTAL_ROWS": len(rows),
        }
    return counts


def main() -> None:
    runner_rows = load_csv(RUNNER_JOBS_CSV)
    if not runner_rows:
        raise SystemExit(f"Missing or empty runner jobs registry: {RUNNER_JOBS_CSV}")

    task_rows = load_csv(TASK_MATRIX_CSV)
    tasks_total = len(task_rows)
    tasks_by_nodes = Counter(int(r["nodes"]) for r in task_rows)

    job_ids = [r["job_id"] for r in runner_rows if (r.get("job_id") or "").isdigit()]
    live = query_runner_states(job_ids)
    progress = task_progress_counts(runner_rows)

    snapshot_rows: List[Dict[str, str]] = []
    state_counts = Counter()
    class_counts = Counter()

    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")

    for r in runner_rows:
        jid = r.get("job_id", "")
        info = live.get(jid, {
            "state": "UNKNOWN",
            "elapsed": "",
            "time_left": "",
            "reason": "",
            "source": "none",
            "exit_code": "",
        })
        state = normalize_state(info.get("state", "UNKNOWN"))
        status_class = classify(state)
        state_counts[state] += 1
        class_counts[status_class] += 1

        p = progress.get(r["runner_id"], {"COMPLETED": 0, "FAILED": 0, "TOTAL_ROWS": 0})
        snapshot_rows.append(
            {
                "timestamp": now,
                "runner_id": r["runner_id"],
                "job_name": r["job_name"],
                "job_id": jid,
                "nodes": r["nodes"],
                "status_state": state,
                "status_class": status_class,
                "elapsed": info.get("elapsed", ""),
                "time_left": info.get("time_left", ""),
                "reason": info.get("reason", ""),
                "source": info.get("source", "none"),
                "exit_code": info.get("exit_code", ""),
                "tasks_completed": str(p["COMPLETED"]),
                "tasks_failed": str(p["FAILED"]),
                "progress_rows": str(p["TOTAL_ROWS"]),
            }
        )

    write_csv(
        RUNNER_STATUS_SNAPSHOT_CSV,
        snapshot_rows,
        [
            "timestamp",
            "runner_id",
            "job_name",
            "job_id",
            "nodes",
            "status_state",
            "status_class",
            "elapsed",
            "time_left",
            "reason",
            "source",
            "exit_code",
            "tasks_completed",
            "tasks_failed",
            "progress_rows",
        ],
    )

    tasks_completed_total = sum(int(r["tasks_completed"]) for r in snapshot_rows)
    tasks_failed_total = sum(int(r["tasks_failed"]) for r in snapshot_rows)

    summary = {
        "date_tag": DATE_TAG,
        "timestamp": now,
        "runner_jobs_csv": str(RUNNER_JOBS_CSV),
        "task_matrix_csv": str(TASK_MATRIX_CSV),
        "runner_status_snapshot_csv": str(RUNNER_STATUS_SNAPSHOT_CSV),
        "runner_status_class_counts": dict(class_counts),
        "runner_state_counts": dict(state_counts),
        "tasks_total": tasks_total,
        "tasks_by_nodes": {str(k): int(v) for k, v in sorted(tasks_by_nodes.items())},
        "tasks_completed_total": tasks_completed_total,
        "tasks_failed_total": tasks_failed_total,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
