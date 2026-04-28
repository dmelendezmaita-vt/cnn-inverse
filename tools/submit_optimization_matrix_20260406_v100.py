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

DATE_TAG = "20260406_v100"
REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "slurm_train.sbatch"
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}/submission"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

PLAN_CSV = TABLES / f"optimization_submission_matrix_{DATE_TAG}.csv"
JOBS_CSV = TABLES / f"optimization_jobs_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"optimization_submission_run_log_{DATE_TAG}.json"

LANES = [
    "phaseA_recovery_ntrain",
    "phaseB_scaling_improve",
]


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
    report_interval = row.get("report_interval_sec", "15")
    export_block = ",".join(
        [
            "ALL",
            f"DATA_ACCESS_MODE={row['data_access_mode']}",
            f"TAR_PATH={row['tar_path']}",
            f"DATA_PREFIX={row['data_prefix']}",
            "CURR=0.1",
            f"PARAMS_FILE={row['params_file']}",
            f"REPORT_INTERVAL={report_interval}",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
            "TORCH_NCCL_BLOCKING_WAIT=1",
            "NCCL_DEBUG=WARN",
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
        f"--gres=gpu:{row['gpu_token']}:{row['gpus_per_node']} "
        f"--time={row['time_limit']} "
        f"--export={export_block} "
        f"{TRAIN_SCRIPT}"
    )

    proc = run_falcon_bash(sbatch_cmd)
    if proc.returncode != 0:
        return {
            "job_id": "",
            "status": "SUBMIT_FAILED",
            "elapsed": "0:00",
            "submit_action": "submit_failed",
            "submit_error": (proc.stderr or proc.stdout or "").strip().replace("\n", " | "),
        }

    out = (proc.stdout or "").strip()
    job_id = out.split(";")[0].strip()
    if not job_id.isdigit():
        return {
            "job_id": "",
            "status": "SUBMIT_FAILED",
            "elapsed": "0:00",
            "submit_action": "submit_failed_parse",
            "submit_error": f"unparsable sbatch output: {out}",
        }

    return {
        "job_id": job_id,
        "status": "SUBMITTED",
        "elapsed": "0:00",
        "submit_action": "submitted_now",
        "submit_error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit optimization matrix jobs.")
    parser.add_argument("--lane", default="all", choices=["all"] + LANES)
    parser.add_argument("--limit", type=int, default=0, help="optional max number of new submissions (0 means no limit)")
    args = parser.parse_args()

    if not PLAN_CSV.exists():
        raise SystemExit(f"Missing plan CSV: {PLAN_CSV}")

    plan_rows = load_csv(PLAN_CSV)
    if not plan_rows:
        raise SystemExit(f"Plan CSV is empty: {PLAN_CSV}")

    existing = load_csv(JOBS_CSV)
    existing_by_name = {r.get("job_name", ""): r for r in existing}

    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")

    out_rows: List[Dict[str, str]] = []
    submitted = 0
    skipped_existing = 0
    failed = 0
    considered = 0

    for row in plan_rows:
        if args.lane != "all" and row.get("submission_lane", "") != args.lane:
            continue
        considered += 1
        job_name = row.get("job_name", "")
        if job_name in existing_by_name:
            prev = existing_by_name[job_name]
            rec = dict(row)
            rec.update(
                {
                    "job_id": prev.get("job_id", ""),
                    "status": prev.get("status", "SUBMITTED"),
                    "elapsed": prev.get("elapsed", "0:00"),
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
                    "elapsed": "0:00",
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
                "elapsed": result["elapsed"],
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
        "elapsed",
        "submitted_at",
        "submit_action",
        "submit_error",
    ]
    write_csv(JOBS_CSV, out_rows, fieldnames)

    run_log = {
        "date_tag": DATE_TAG,
        "lane": args.lane,
        "limit": args.limit,
        "plan_csv": str(PLAN_CSV),
        "jobs_csv": str(JOBS_CSV),
        "considered_rows": considered,
        "submitted_now": submitted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "timestamp": now,
    }
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2))

    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
