#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260406_v100"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}/submission"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

JOBS_CSV = TABLES / f"optimization_jobs_registry_{DATE_TAG}.csv"
SNAPSHOT_CSV = TABLES / f"optimization_jobs_status_snapshot_{DATE_TAG}.csv"
SUMMARY_JSON = NOTES / f"optimization_status_summary_{DATE_TAG}.json"


def load_registry(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing jobs registry: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty jobs registry: {path}")
    return rows


def run_falcon_bash(cmd: str) -> subprocess.CompletedProcess:
    # Build one nested SSH shell command so inner command arguments are preserved.
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
    # sacct may report variants like "CANCELLED by ..."
    if " " in s:
        s = s.split(" ", 1)[0]
    return s


def parse_squeue(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if not job_ids:
        return {}
    ids_arg = ",".join(job_ids)
    cmd = (
        "squeue -h "
        f"-j {ids_arg} "
        "-o '%i|%T|%M|%L|%D|%R|%j'"
    )
    proc = run_falcon_bash(cmd)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        jid, state, elapsed, left, nodes, reason, name = [p.strip() for p in parts]
        if not jid.isdigit():
            continue
        rows[jid] = {
            "state": normalize_state(state),
            "elapsed": elapsed,
            "time_left": left,
            "nodes": nodes,
            "reason": reason,
            "job_name_live": name,
            "source": "squeue",
            "exit_code": "",
            "start": "",
            "end": "",
        }
    return rows


def parse_sacct(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if not job_ids:
        return {}
    ids_arg = ",".join(job_ids)
    cmd = (
        "sacct -n -P -X "
        f"-j {ids_arg} "
        "--format=JobIDRaw,State,Elapsed,ExitCode,NodeList,Start,End,JobName"
    )
    proc = run_falcon_bash(cmd)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|", 7)
        if len(parts) != 8:
            continue
        jid, state, elapsed, exit_code, nodes, start, end, name = [p.strip() for p in parts]
        if not jid.isdigit():
            continue
        rows[jid] = {
            "state": normalize_state(state),
            "elapsed": elapsed,
            "time_left": "",
            "nodes": nodes,
            "reason": "",
            "job_name_live": name,
            "source": "sacct",
            "exit_code": exit_code,
            "start": start,
            "end": end,
        }
    return rows


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


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    registry = load_registry(JOBS_CSV)
    job_ids = [r.get("job_id", "").strip() for r in registry if (r.get("job_id", "").strip().isdigit())]
    if not job_ids:
        raise SystemExit("No numeric job_ids found in registry.")

    sq = parse_squeue(job_ids)
    sa = parse_sacct(job_ids)

    snapshot_rows: List[Dict[str, str]] = []
    state_counts = Counter()
    class_counts = Counter()
    source_counts = Counter()

    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")

    for r in registry:
        jid = r.get("job_id", "").strip()
        live = None
        if jid in sq:
            live = sq[jid]
        elif jid in sa:
            live = sa[jid]

        if live is None:
            state = "UNKNOWN"
            live = {
                "state": state,
                "elapsed": "",
                "time_left": "",
                "nodes": "",
                "reason": "",
                "job_name_live": "",
                "source": "none",
                "exit_code": "",
                "start": "",
                "end": "",
            }
        else:
            state = normalize_state(live.get("state", "UNKNOWN"))

        status_class = classify(state)
        state_counts[state] += 1
        class_counts[status_class] += 1
        source_counts[live.get("source", "none")] += 1

        snapshot_rows.append(
            {
                "timestamp": now,
                "job_id": jid,
                "job_name": r.get("job_name", ""),
                "phase": r.get("phase", ""),
                "track_key": r.get("track_key", ""),
                "nodes_requested": r.get("nodes", ""),
                "status_state": state,
                "status_class": status_class,
                "elapsed": live.get("elapsed", ""),
                "time_left": live.get("time_left", ""),
                "reason": live.get("reason", ""),
                "source": live.get("source", "none"),
                "exit_code": live.get("exit_code", ""),
                "start": live.get("start", ""),
                "end": live.get("end", ""),
            }
        )

    fieldnames = [
        "timestamp",
        "job_id",
        "job_name",
        "phase",
        "track_key",
        "nodes_requested",
        "status_state",
        "status_class",
        "elapsed",
        "time_left",
        "reason",
        "source",
        "exit_code",
        "start",
        "end",
    ]
    write_csv(SNAPSHOT_CSV, snapshot_rows, fieldnames)

    summary = {
        "date_tag": DATE_TAG,
        "timestamp": now,
        "registry_csv": str(JOBS_CSV),
        "snapshot_csv": str(SNAPSHOT_CSV),
        "total_jobs": len(snapshot_rows),
        "status_class_counts": dict(class_counts),
        "status_state_counts": dict(state_counts),
        "source_counts": dict(source_counts),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
