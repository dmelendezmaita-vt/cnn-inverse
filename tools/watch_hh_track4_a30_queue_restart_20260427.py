#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Wait for the A30 queue to drain current live tasks, then restart it if "
            "failed, blocked, missing, or pending work remains."
        )
    )
    ap.add_argument("--registry-csv", required=True)
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--queue-session", default="hh_track4_a30_literal_20260427")
    ap.add_argument("--poll-sec", type=float, default=60.0)
    return ap.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    registry = Path(args.registry_csv)
    manifest = Path(args.manifest_json)

    while True:
        rows = load_rows(registry)
        manifest_obj = json.loads(manifest.read_text()) if manifest.exists() else {"tasks": []}
        manifest_ids = {str(task.get("task_id", "")).strip() for task in manifest_obj.get("tasks", [])}
        registry_ids = {row.get("task_id", "").strip() for row in rows}
        running = sum(1 for row in rows if row.get("status") in {"RUNNING", "LAUNCHING"})
        pending = sum(1 for row in rows if row.get("status") == "PENDING")
        failed = sum(1 for row in rows if row.get("status") == "FAILED")
        blocked = sum(1 for row in rows if row.get("status") == "BLOCKED")
        missing = len(manifest_ids - registry_ids)
        if running == 0:
            if failed > 0 or blocked > 0 or missing > 0 or pending > 0:
                subprocess.run(["tmux", "kill-session", "-t", args.queue_session], check=False)
                subprocess.run(
                    [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        args.queue_session,
                        (
                            "/projects/neuro-collab/conda/neuro-collab-env/bin/python "
                            "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/tools/"
                            "run_hh_track4_a30_literal_queue_20260427.py "
                            f"--manifest-json {manifest} --alloc-job-id {args.alloc_job_id} --rerun-failed"
                        ),
                    ],
                    check=True,
                )
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
