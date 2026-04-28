#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260406_v100_large"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}/submission"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

RUNNER_PLAN_CSV = TABLES / f"v100_large_runner_plan_{DATE_TAG}.csv"
RUNNER_JOBS_CSV = TABLES / f"v100_large_runner_jobs_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_large_submission_run_log_{DATE_TAG}.json"


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


def run_falcon_bash(cmd: str) -> subprocess.CompletedProcess:
    quoted = shlex.quote(cmd)
    return subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "tinkercliffs1",
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "falcon1.arc.vt.edu",
            "bash",
            "-lc",
            quoted,
        ],
        text=True,
        capture_output=True,
    )


def submit_one(row: Dict[str, str]) -> Dict[str, str]:
    export_block = ",".join(
        [
            "ALL",
            f"REPO_ROOT={REPO}",
            f"TASK_CSV={row['task_csv']}",
            f"NODES_BUCKET={row['nodes']}",
            f"PROGRESS_CSV={row['progress_csv']}",
            f"RESULTS_ROOT={row['results_root']}",
            f"SUMMARY_JSON={row['summary_json']}",
            "MAX_TASKS=0",
        ]
    )

    sbatch_cmd = (
        f"cd {REPO} && "
        f"sbatch --parsable "
        f"--job-name={row['job_name']} "
        f"--account=neuro-collab "
        f"--partition={row['partition']} "
        f"--qos={row['qos']} "
        f"--nodes={row['nodes']} "
        f"--ntasks-per-node=1 "
        f"--cpus-per-task={row['cpus_per_node']} "
        f"--mem={row['mem_gib']}G "
        f"--gres=gpu:v100:{row['gpus_per_node']} "
        f"--time={row['time_limit']} "
        f"--export={export_block} "
        f"{row['sbatch_script']}"
    )

    proc = run_falcon_bash(sbatch_cmd)
    if proc.returncode != 0:
        return {
            "job_id": "",
            "status": "SUBMIT_FAILED",
            "submit_action": "submit_failed",
            "submit_error": (proc.stderr or proc.stdout or "").strip().replace("\n", " | "),
        }

    out = (proc.stdout or "").strip()
    job_id = out.split(";")[0].strip()
    if not job_id.isdigit():
        return {
            "job_id": "",
            "status": "SUBMIT_FAILED",
            "submit_action": "submit_failed_parse",
            "submit_error": f"unparsable sbatch output: {out}",
        }

    return {
        "job_id": job_id,
        "status": "SUBMITTED",
        "submit_action": "submitted_now",
        "submit_error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit V100 large-matrix runner jobs.")
    parser.add_argument("--limit", type=int, default=0, help="optional max number of new submissions (0 means no limit)")
    args = parser.parse_args()

    if not RUNNER_PLAN_CSV.exists():
        raise SystemExit(f"Missing runner plan CSV: {RUNNER_PLAN_CSV}")

    plan_rows = load_csv(RUNNER_PLAN_CSV)
    if not plan_rows:
        raise SystemExit(f"Runner plan CSV is empty: {RUNNER_PLAN_CSV}")

    existing = load_csv(RUNNER_JOBS_CSV)
    existing_by_name = {r.get("job_name", ""): r for r in existing}

    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")

    out_rows: List[Dict[str, str]] = []
    submitted = 0
    skipped_existing = 0
    failed = 0

    for row in plan_rows:
        job_name = row.get("job_name", "")
        if job_name in existing_by_name:
            prev = existing_by_name[job_name]
            rec = dict(row)
            rec.update(
                {
                    "job_id": prev.get("job_id", ""),
                    "status": prev.get("status", "SUBMITTED"),
                    "submitted_at": prev.get("submitted_at", now),
                    "submit_action": "already_recorded",
                    "submit_error": prev.get("submit_error", ""),
                }
            )
            skipped_existing += 1
            out_rows.append(rec)
            continue

        if args.limit > 0 and submitted >= args.limit:
            rec = dict(row)
            rec.update(
                {
                    "job_id": "",
                    "status": "NOT_SUBMITTED_LIMIT",
                    "submitted_at": now,
                    "submit_action": "not_submitted_limit",
                    "submit_error": "",
                }
            )
            out_rows.append(rec)
            continue

        result = submit_one(row)
        rec = dict(row)
        rec.update(
            {
                "job_id": result["job_id"],
                "status": result["status"],
                "submitted_at": now,
                "submit_action": result["submit_action"],
                "submit_error": result["submit_error"],
            }
        )
        out_rows.append(rec)
        if result["status"] == "SUBMITTED":
            submitted += 1
        else:
            failed += 1

    fieldnames = list(plan_rows[0].keys()) + [
        "job_id",
        "status",
        "submitted_at",
        "submit_action",
        "submit_error",
    ]
    write_csv(RUNNER_JOBS_CSV, out_rows, fieldnames)

    run_log = {
        "date_tag": DATE_TAG,
        "runner_plan_csv": str(RUNNER_PLAN_CSV),
        "runner_jobs_csv": str(RUNNER_JOBS_CSV),
        "considered_rows": len(plan_rows),
        "submitted_now": submitted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "limit": args.limit,
        "timestamp": now,
    }
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2) + "\n")

    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
