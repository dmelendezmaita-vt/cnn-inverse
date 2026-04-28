#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


QUEUE_SCRIPT = (
    "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/"
    "run_hh_track4_a30_literal_queue_20260427.py"
)
BUILD_SCRIPT = (
    "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/"
    "build_hh_track4_a30_direct_manifest_20260427.py"
)
APPEND_LATE_SCRIPT = (
    "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/"
    "append_hh_track4_a30_late_tasks_20260427.py"
)
DECORATE_SCRIPT = (
    "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/"
    "decorate_hh_track4_manifest_with_monitoring_20260427.py"
)
RESTART_SCRIPT = (
    "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/"
    "watch_hh_track4_a30_queue_restart_20260427.py"
)
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
NTFY_TOPIC = "dmelendezmaita"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Wait for a fresh A30 allocation to start, then relaunch the literal monitored HH Track4 queue "
            "against that allocation."
        )
    )
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--registry-csv", required=True)
    ap.add_argument("--queue-session", default="hh_track4_a30_literal_20260427")
    ap.add_argument("--restart-watch-session", default="hh_track4_a30_restart_watch_20260427")
    ap.add_argument("--poll-sec", type=float, default=60.0)
    return ap.parse_args()


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def send_ntfy(title: str, message: str) -> None:
    run(
        [
            "curl",
            "-fsS",
            "-H",
            f"Title: {title}",
            "-d",
            message,
            f"https://ntfy.sh/{NTFY_TOPIC}",
        ],
        check=False,
    )


def allocation_state(job_id: str) -> str:
    proc = run(["squeue", "-j", job_id, "-h", "-o", "%T"], check=False)
    state = (proc.stdout or "").strip()
    if state:
        return state.splitlines()[0].strip().upper()
    proc = run(
        ["sacct", "-j", job_id, "--format=State", "-n", "-P"],
        check=False,
    )
    state = (proc.stdout or "").strip()
    if not state:
        return "UNKNOWN"
    return state.split("|", 1)[0].strip().upper()


def stale_alloc_ids(registry_csv: str) -> list[str]:
    path = Path(registry_csv)
    if not path.exists():
        return []
    out = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in {"RUNNING", "LAUNCHING"}:
                continue
            alloc = row.get("alloc_job_id", "").strip()
            if alloc and alloc not in out:
                out.append(alloc)
    return out


def prepare_manifest(alloc_job_id: str, manifest_json: str) -> None:
    run([PYTHON_BIN, BUILD_SCRIPT, "--alloc-job-id", alloc_job_id, "--out-json", manifest_json], check=True)
    run([PYTHON_BIN, APPEND_LATE_SCRIPT, "--manifest-json", manifest_json], check=True)
    run([PYTHON_BIN, DECORATE_SCRIPT, "--manifest-json", manifest_json], check=True)


def main() -> int:
    args = parse_args()
    last_state = None

    while True:
        state = allocation_state(args.alloc_job_id)
        if state != last_state:
            print(f"[allocation-watch] job={args.alloc_job_id} state={state}", flush=True)
            last_state = state

        if state == "RUNNING":
            prepare_manifest(args.alloc_job_id, args.manifest_json)
            run(["tmux", "kill-session", "-t", args.queue_session], check=False)
            run(["tmux", "kill-session", "-t", args.restart_watch_session], check=False)
            run(["pkill", "-f", "run_hh_track4_a30_literal_queue_20260427.py"], check=False)
            for stale_alloc in stale_alloc_ids(args.registry_csv):
                run(["pkill", "-f", f"srun --jobid={stale_alloc}"], check=False)

            queue_cmd = (
                f"{PYTHON_BIN} {QUEUE_SCRIPT} --manifest-json {args.manifest_json} "
                f"--alloc-job-id {args.alloc_job_id} --rerun-failed"
            )
            run(["tmux", "new-session", "-d", "-s", args.queue_session, queue_cmd], check=True)

            restart_cmd = (
                f"{PYTHON_BIN} {RESTART_SCRIPT} --registry-csv {args.registry_csv} "
                f"--manifest-json {args.manifest_json} --alloc-job-id {args.alloc_job_id} "
                f"--queue-session {args.queue_session}"
            )
            run(["tmux", "new-session", "-d", "-s", args.restart_watch_session, restart_cmd], check=True)
            send_ntfy(
                "A30 rerun queue rebound",
                (
                    f"alloc_job_id={args.alloc_job_id} queue_session={args.queue_session} "
                    f"restart_watch_session={args.restart_watch_session}"
                ),
            )
            return 0

        if state in {"CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}:
            send_ntfy(
                "A30 allocation failed before queue launch",
                f"alloc_job_id={args.alloc_job_id} state={state}",
            )
            return 1

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
