#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def node_step_counts(job_id: str) -> dict[str, int]:
    proc = subprocess.run(
        [
            "ssh",
            "falcon2",
            f"sacct -j {job_id} --format=JobID,State,Elapsed,NodeList --noheader | tail -n 40",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    counts: dict[str, int] = {}
    if proc.returncode != 0:
        return counts
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        jobid, state, _elapsed, nodelist = parts[:4]
        if "." not in jobid:
            continue
        if state != "RUNNING":
            continue
        counts[nodelist] = counts.get(nodelist, 0) + 1
    return counts


def submit_step(job_id: str, node: str, command: str, log_out: Path, log_err: Path) -> subprocess.Popen:
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_err.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ssh",
            "falcon2",
            f"srun --jobid={job_id} --exclusive --mem=0 -N1 -n1 -c12 -w {node} --gres=gpu:v100:1 bash -lc {shlex.quote(command)}",
        ],
        stdout=log_out.open("w"),
        stderr=log_err.open("w"),
        text=True,
    )


def update_registry_row(rows: list[dict[str, str]], row_id: str, new_row: dict[str, str]) -> list[dict[str, str]]:
    filtered = [row for row in rows if row.get("row_id") != row_id]
    filtered.append(new_row)
    return filtered


def main() -> None:
    ap = argparse.ArgumentParser(description="Drain the remaining HH Track4 assumption-conditioned backlog inside a live Falcon allocation.")
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--backlog-json", required=True)
    ap.add_argument("--registry-csv", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--nodes", default="fal101,fal102,fal103,fal108")
    ap.add_argument("--poll-sec", type=float, default=5.0)
    args = ap.parse_args()

    backlog = read_json(Path(args.backlog_json))
    if not isinstance(backlog, list):
        raise SystemExit("backlog json must be a list of task objects")
    registry_path = Path(args.registry_csv)
    registry = load_registry(registry_path)
    log_dir = Path(args.log_dir)
    nodes = [part.strip() for part in args.nodes.split(",") if part.strip()]

    active: dict[str, dict[str, object]] = {}
    completed_states = {"COMPLETED", "FAILED", "SHOWN_USELESS", "BLOCKED"}
    while True:
        # reconcile finished local subprocesses
        finished_ids = []
        for row_id, payload in active.items():
            proc = payload["proc"]
            ret = proc.poll()
            if ret is None:
                continue
            finished_ids.append(row_id)
            status = "COMPLETED" if ret == 0 else "FAILED"
            registry = update_registry_row(
                registry,
                row_id,
                {
                    "row_id": row_id,
                    "label": str(payload["label"]),
                    "status": status,
                    "node": str(payload["node"]),
                    "started_at": str(payload["started_at"]),
                    "ended_at": now_iso(),
                    "return_code": str(ret),
                    "command": str(payload["command"]),
                    "stdout_log": str(payload["stdout_log"]),
                    "stderr_log": str(payload["stderr_log"]),
                },
            )
        for row_id in finished_ids:
            active.pop(row_id, None)
        if finished_ids:
            write_csv(registry_path, registry)

        completed_ids = {row["row_id"] for row in registry if row.get("status") in completed_states}
        running_ids = set(active.keys())
        pending = [row for row in backlog if row["row_id"] not in completed_ids and row["row_id"] not in running_ids]

        if not pending and not active:
            break

        counts = node_step_counts(args.alloc_job_id)
        free_nodes = [node for node in nodes if counts.get(node, 0) == 0]

        while pending and free_nodes:
            task = pending.pop(0)
            node = free_nodes.pop(0)
            row_id = str(task["row_id"])
            label = str(task["label"])
            command = str(task["command"])
            stdout_log = log_dir / f"{row_id}.out"
            stderr_log = log_dir / f"{row_id}.err"
            proc = submit_step(args.alloc_job_id, node, command, stdout_log, stderr_log)
            started_at = now_iso()
            active[row_id] = {
                "proc": proc,
                "label": label,
                "node": node,
                "started_at": started_at,
                "command": command,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            }
            registry = update_registry_row(
                registry,
                row_id,
                {
                    "row_id": row_id,
                    "label": label,
                    "status": "RUNNING",
                    "node": node,
                    "started_at": started_at,
                    "ended_at": "",
                    "return_code": "",
                    "command": command,
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                },
            )
            write_csv(registry_path, registry)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()

