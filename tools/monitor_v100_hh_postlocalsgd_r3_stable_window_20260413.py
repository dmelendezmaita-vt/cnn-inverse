#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

DATE_TAG = "20260413_v100_hh_postlocalsgd_r3_stable"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

REGISTRY_CSV = TABLES / f"v100_hh_postlocalsgd_r3_stable_registry_{DATE_TAG}.csv"
STATE_JSON = NOTES / f"v100_hh_postlocalsgd_r3_stable_driver_state_{DATE_TAG}.json"

DRIVER_SCRIPT = REPO / "tools" / "run_v100_hh_postlocalsgd_r3_stable_driver_20260413.py"
DRIVER_SESSION = "neuro_r3_stable_driver"
FALCON_HOST = "falcon1.arc.vt.edu"
TOPIC = "dmelendezmaita"
ALLOC_JOB_ID = "355055"

INTERVAL_SEC = 12 * 60
WINDOW_SEC = 2 * 60 * 60
STALE_STEP_AGE_SEC = 15 * 60


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ssh_capture(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", FALCON_HOST, cmd],
        text=True,
        capture_output=True,
        check=False,
    )


def run_local(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


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


def tmux_session_exists(name: str) -> bool:
    proc = run_local(["tmux", "has-session", "-t", name])
    return proc.returncode == 0


def restart_driver() -> bool:
    run_local(["tmux", "kill-session", "-t", DRIVER_SESSION])
    proc = run_local(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            DRIVER_SESSION,
            f"python -u {DRIVER_SCRIPT}",
        ]
    )
    return proc.returncode == 0


def active_steps(job_id: str) -> list[str]:
    proc = ssh_capture(f"squeue -s -h -j {job_id} -o '%i'")
    steps = []
    for ln in (proc.stdout or "").splitlines():
        ln = ln.strip()
        if re.fullmatch(r"\d+\.\d+", ln):
            steps.append(ln)
    return steps


def parse_state() -> dict | None:
    if not STATE_JSON.exists():
        return None
    return json.loads(STATE_JSON.read_text())


def next_incomplete_row_id() -> str | None:
    state = parse_state()
    if state:
        return state.get("next_incomplete_row_id")
    return None


def row_run_root(row_id: str) -> Path:
    return RUNS / "phaseR3_postlocalsgd_stable" / "dedicated" / row_id


def row_log_paths(row_id: str) -> list[Path]:
    return [LOGS / f"{row_id}.out", LOGS / f"{row_id}.err"]


def row_metrics_exist(row_id: str) -> bool:
    root = row_run_root(row_id)
    if not root.exists():
        return False
    for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if (subdir / "metrics_summary.json").exists():
            return True
    return False


def row_last_activity_age_sec(row_id: str) -> float | None:
    paths = [p for p in row_log_paths(row_id) if p.exists()]
    root = row_run_root(row_id)
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                paths.append(p)
    if not paths:
        return None
    newest = max(p.stat().st_mtime for p in paths)
    return max(0.0, time.time() - newest)


def clear_active_steps(job_id: str) -> list[str]:
    steps = active_steps(job_id)
    for step in steps:
        ssh_capture(f"scancel {step} 2>/dev/null || true")
    return steps


def completed_count() -> int:
    return sum(row.get("status") == "COMPLETED" for row in load_csv(REGISTRY_CSV))


def log_line(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def main() -> int:
    start_ts = time.time()
    checks = 0
    interventions = 0
    send_ntfy(
        "neuro-collab v100 R3 stable watch",
        "Started a 2-hour autonomous monitor for the stable R3 package. It will check every 12 minutes, fix stalled rows/driver issues, and keep the queue moving on allocation 355055.",
    )

    while (time.time() - start_ts) < WINDOW_SEC:
        checks += 1
        row_id = next_incomplete_row_id()
        steps = active_steps(ALLOC_JOB_ID)
        count = completed_count()

        log_line(f"check={checks} completed_rows={count} next_incomplete={row_id} active_steps={steps}")

        if not tmux_session_exists(DRIVER_SESSION):
            interventions += 1
            restart_driver()
            send_ntfy(
                "neuro-collab v100 R3 stable watch",
                f"Driver session {DRIVER_SESSION} was missing and has been restarted during the 2-hour monitoring window.",
            )

        if row_id and steps:
            age = row_last_activity_age_sec(row_id)
            if age is not None and age >= STALE_STEP_AGE_SEC:
                interventions += 1
                cleared = clear_active_steps(ALLOC_JOB_ID)
                if row_metrics_exist(row_id):
                    send_ntfy(
                        "neuro-collab v100 R3 stable watch",
                        f"Detected stale row {row_id} after metrics existed; cleared child steps {cleared} so the stable driver can continue.",
                    )
                else:
                    send_ntfy(
                        "neuro-collab v100 R3 stable watch",
                        f"Detected stale row {row_id} with no new activity for {int(age)}s; cleared child steps {cleared} so the stable driver can retry it.",
                    )
                time.sleep(10)

        elif row_id and not steps:
            # No active step but still incomplete work. Nudge the driver if needed.
            interventions += 1
            restart_driver()
            send_ntfy(
                "neuro-collab v100 R3 stable watch",
                f"No active child step was present while row {row_id} remained incomplete. Restarted the stable driver to resume the queue.",
            )

        remaining = WINDOW_SEC - (time.time() - start_ts)
        if remaining <= 0:
            break
        time.sleep(min(INTERVAL_SEC, max(0, int(remaining))))

    send_ntfy(
        "neuro-collab v100 R3 stable watch",
        f"Finished the 2-hour monitoring window. checks={checks} interventions={interventions} completed_rows={completed_count()} next_incomplete={next_incomplete_row_id()}",
    )
    log_line(f"finished window checks={checks} interventions={interventions} completed_rows={completed_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
