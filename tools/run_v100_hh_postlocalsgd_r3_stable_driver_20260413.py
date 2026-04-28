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

import yaml

DATE_TAG = "20260413_v100_hh_postlocalsgd_r3_stable"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
LAUNCH_LOG_DIR = NOTES / "launcher_bg"

MATRIX_CSV = TABLES / f"v100_hh_postlocalsgd_r3_stable_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_postlocalsgd_r3_stable_registry_{DATE_TAG}.csv"
STATE_JSON = NOTES / f"v100_hh_postlocalsgd_r3_stable_driver_state_{DATE_TAG}.json"
DEFERRED_JSON = NOTES / f"v100_hh_postlocalsgd_r3_stable_deferred_rows_{DATE_TAG}.json"
LAUNCHER = REPO / "tools" / "launch_v100_hh_postlocalsgd_r3_stable_20260413.py"
RECONCILE = REPO / "tools" / "reconcile_v100_hh_postlocalsgd_r3_stable_20260413.py"
SUMMARIZE = REPO / "tools" / "summarize_v100_hh_postlocalsgd_r3_stable_20260413.py"
STEP_SCRIPT = REPO / "tools" / "run_v100_interactive_step_20260411.sh"

FALCON_HOST = "falcon1.arc.vt.edu"
TOPIC = "dmelendezmaita"
ALLOC_IDS = ["355055"]
POLL_SEC = 60
STALE_STEP_AGE_SEC = 900


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_deferred_rows() -> list[str]:
    if not DEFERRED_JSON.exists():
        return []
    try:
        payload = json.loads(DEFERRED_JSON.read_text())
    except Exception:
        return []
    rows = payload.get("deferred_rows", [])
    return [str(x) for x in rows]


def add_deferred_row(row_id: str, reason: str) -> None:
    payload = {}
    if DEFERRED_JSON.exists():
        try:
            payload = json.loads(DEFERRED_JSON.read_text())
        except Exception:
            payload = {}
    rows = [str(x) for x in payload.get("deferred_rows", [])]
    if row_id not in rows:
        rows.append(row_id)
    reasons = payload.get("reason", {})
    if not isinstance(reasons, dict):
        reasons = {}
    reasons[row_id] = reason
    payload["deferred_rows"] = rows
    payload["reason"] = reasons
    payload["updated_at"] = now_iso()
    DEFERRED_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def matrix_row_by_id(row_id: str) -> dict | None:
    for row in load_csv(MATRIX_CSV):
        if row["row_id"] == row_id:
            return row
    return None


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


def expand_nodelist(nodelist: str) -> list[str]:
    proc = ssh_capture(f"scontrol show hostnames {shlex.quote(nodelist)}", check=False)
    hosts = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if hosts:
        return hosts
    return [part.strip() for part in nodelist.split(",") if part.strip()]


def choose_running_allocation() -> dict | None:
    for job_id in ALLOC_IDS:
        info = parse_job_info(job_id)
        if info["state"] == "RUNNING":
            return info
    return None


def next_incomplete_row_id() -> str | None:
    matrix_rows = load_csv(MATRIX_CSV)
    completed = {
        row["row_id"]
        for row in load_csv(REGISTRY_CSV)
        if row.get("status") == "COMPLETED" and row_metrics_exist(row["row_id"])
    }
    deferred = set(load_deferred_rows())
    for row in matrix_rows:
        if row["row_id"] not in completed and row["row_id"] not in deferred:
            return row["row_id"]
    return None


def _row_run_root(row_id: str) -> Path:
    return OUT_ROOT / "runs" / "phaseR3_postlocalsgd_stable" / "dedicated" / row_id


def _row_log_paths(row_id: str) -> list[Path]:
    return [
        OUT_ROOT / "logs" / f"{row_id}.out",
        OUT_ROOT / "logs" / f"{row_id}.err",
    ]


def row_metrics_exist(row_id: str) -> bool:
    run_root = _row_run_root(row_id)
    if not run_root.exists():
        return False
    for subdir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
        if (subdir / "metrics_summary.json").exists():
            return True
    return False


def _row_params(row_id: str) -> dict | None:
    row = matrix_row_by_id(row_id)
    if row is None:
        return None
    try:
        return yaml.safe_load(Path(row["params_file"]).read_text())
    except Exception:
        return None


def row_checkpoint_path(row_id: str) -> str | None:
    run_root = _row_run_root(row_id)
    if not run_root.exists():
        return None
    params = _row_params(row_id) or {}
    training_epochs = int(params.get("training", {}).get("epochs", 0) or 0)
    checkpoint_every = params.get("runconfig", {}).get("save_checkpoints_epochs", None)
    try:
        checkpoint_every = int(checkpoint_every) if checkpoint_every is not None else None
    except Exception:
        checkpoint_every = None

    candidates = []
    for ckpt in run_root.rglob("net_e*.pt"):
        m = re.search(r"net_e(\d+)\.pt$", ckpt.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), ckpt))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    if training_epochs > 0 and checkpoint_every and checkpoint_every > 0:
        min_epoch = max(checkpoint_every, training_epochs - checkpoint_every)
        viable = [item for item in candidates if item[0] >= min_epoch]
        if viable:
            return str(viable[-1][1])
        return None
    return str(candidates[-1][1])


def row_last_activity_age_sec(row_id: str) -> float | None:
    paths = [p for p in _row_log_paths(row_id) if p.exists()]
    run_root = _row_run_root(row_id)
    if run_root.exists():
        for p in run_root.rglob("*"):
            if p.is_file():
                paths.append(p)
    if not paths:
        return None
    newest = max(p.stat().st_mtime for p in paths)
    return max(0.0, time.time() - newest)


def clear_active_steps(job_id: str) -> None:
    steps = active_steps(job_id)
    for step_id in steps:
        ssh_capture(f"scancel {step_id} 2>/dev/null || true", check=False)


def salvage_eval_only(job_id: str, nodelist: str, row_id: str) -> bool:
    row = matrix_row_by_id(row_id)
    ckpt = row_checkpoint_path(row_id)
    if row is None or ckpt is None:
        return False

    hosts = expand_nodelist(nodelist)
    first_host = hosts[0] if hosts else nodelist.split(",", 1)[0]
    run_output_root = OUT_ROOT / "runs" / row["phase"] / row["launch_mode"] / row_id
    env_assign = " ".join(
        shlex.quote(item)
        for item in [
            f"REPO_ROOT={REPO}",
            f"ALLOC_JOB_ID={job_id}",
            f"RUN_ID={row_id}",
            f"RUN_OUTPUT_ROOT={run_output_root}",
            f"PARAMS_FILE={row['params_file']}",
            f"TAR_PATH={row['tar_path']}",
            f"DATA_PREFIX={row['data_prefix']}",
            f"CURR={row['curr']}",
            "MASTER_ADDR=127.0.0.1",
            "MASTER_PORT=29500",
            "STEP_NNODES=1",
            "NPROC_PER_NODE=1",
            f"DATA_ACCESS_MODE={row['data_access_mode']}",
            f"SHARED_DATA_DIR={row['shared_data_dir']}",
            f"SAVE_PREDICTIONS={row['save_predictions']}",
            "SPLIT_EVAL_AFTER_TRAIN=0",
            f"EVAL_ONLY_CHECKPOINT={ckpt}",
            "FOLLOWUP_TMP_BASE=/projects/neuro-collab/other/v100if_tmp",
        ]
    )
    cmd = (
        f"srun --jobid={job_id} --exclusive -N1 -w {first_host} --ntasks=1 "
        f"--cpus-per-task={row['cpus_per_node']} --gres=gpu:v100:1 "
        f"bash -lc {shlex.quote(env_assign + ' bash ' + str(STEP_SCRIPT))}"
    )
    proc = ssh_capture(f"cd {shlex.quote(str(REPO))} && {cmd}", check=False)
    return proc.returncode == 0


def reconcile_and_summarize() -> None:
    local_capture([sys.executable, str(RECONCILE)], check=False)
    local_capture([sys.executable, str(SUMMARIZE)], check=False)


def save_state(state: dict) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")


def launch_row(job_id: str, nodelist: str, row_id: str) -> subprocess.CompletedProcess:
    LAUNCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    launch_log = LAUNCH_LOG_DIR / f"{row_id}.log"
    cmd = (
        f"cd {shlex.quote(str(REPO))} && "
        f"python {shlex.quote(str(LAUNCHER.relative_to(REPO)))} "
        f"--alloc-job-id {shlex.quote(job_id)} "
        f"--alloc-nodelist {shlex.quote(nodelist)} "
        f"--row-id {shlex.quote(row_id)}"
    )
    bg_cmd = (
        f"cd {shlex.quote(str(REPO))} && "
        f"nohup bash -lc {shlex.quote(cmd)} "
        f"> {shlex.quote(str(launch_log))} 2>&1 < /dev/null & echo $!"
    )
    return ssh_capture(bg_cmd, check=False)


def complete_from_checkpoint_if_possible(job_id: str, nodelist: str, row_id: str) -> bool:
    if row_metrics_exist(row_id):
        return True
    if row_checkpoint_path(row_id) is None:
        return False
    send_ntfy(
        "neuro-collab v100 R3 stable driver",
        f"Row {row_id} has a final checkpoint but no metrics; running separate CPU-only eval completion step.",
    )
    ok = salvage_eval_only(job_id, nodelist, row_id)
    reconcile_and_summarize()
    if row_metrics_exist(row_id):
        send_ntfy(
            "neuro-collab v100 R3 stable driver",
            f"Separate eval completion produced metrics for row {row_id} (srun rc ok={ok}).",
        )
        return True
    send_ntfy(
        "neuro-collab v100 R3 stable driver",
        f"Separate eval completion did not produce metrics for row {row_id}; it will remain incomplete.",
    )
    return False


def main() -> int:
    send_ntfy(
        "neuro-collab v100 R3 stable driver",
        "Started persistent R3 stable post-local SGD driver. It will keep launching incomplete rows in the separate stable 20260413 package on the running 7-day V100 allocation 355055.",
    )

    while True:
        reconcile_and_summarize()
        row_id = next_incomplete_row_id()
        running = choose_running_allocation()

        state = {
            "timestamp": now_iso(),
            "next_incomplete_row_id": row_id,
            "deferred_rows": load_deferred_rows(),
            "running_allocation": running,
            "allocations": [parse_job_info(job_id) for job_id in ALLOC_IDS],
        }
        save_state(state)

        if row_id is None:
            send_ntfy(
                "neuro-collab v100 R3 stable driver",
                "All rows in the 20260413 V100 HH post-local SGD R3 stable package are now complete. The driver is exiting.",
            )
            return 0

        if running is None:
            time.sleep(POLL_SEC)
            continue

        steps = active_steps(running["job_id"])
        if steps:
            active_row = row_id
            age_sec = row_last_activity_age_sec(active_row)
            if age_sec is not None and age_sec >= STALE_STEP_AGE_SEC:
                if row_metrics_exist(active_row):
                    send_ntfy(
                        "neuro-collab v100 R3 stable driver",
                        f"Detected stale step for row {active_row} after metrics were already written; clearing active child steps so the queue can continue.",
                    )
                    clear_active_steps(running["job_id"])
                elif row_checkpoint_path(active_row) is not None:
                    send_ntfy(
                        "neuro-collab v100 R3 stable driver",
                        f"Detected stale row {active_row} with a finished checkpoint but no metrics; attempting eval-only salvage after clearing child steps.",
                    )
                    clear_active_steps(running["job_id"])
                    time.sleep(5)
                    if salvage_eval_only(running["job_id"], running["node_list"], active_row):
                        send_ntfy(
                            "neuro-collab v100 R3 stable driver",
                            f"Eval-only salvage succeeded for row {active_row}; continuing the queue.",
                        )
                    else:
                        send_ntfy(
                            "neuro-collab v100 R3 stable driver",
                            f"Eval-only salvage failed for row {active_row}; deferring it so the queue can continue.",
                        )
                        add_deferred_row(
                            active_row,
                            "Deferred after stale distributed step produced a checkpoint but CPU-only eval salvage did not produce metrics.",
                        )
                else:
                    send_ntfy(
                        "neuro-collab v100 R3 stable driver",
                        f"Detected stale active row {active_row} with no new log activity for {int(age_sec)}s; clearing child steps so the row can be retried automatically.",
                    )
                    clear_active_steps(running["job_id"])
                time.sleep(5)
                reconcile_and_summarize()
                continue
            time.sleep(POLL_SEC)
            continue

        nodelist = running["node_list"]
        if not nodelist or nodelist in {"(null)", "None"}:
            time.sleep(POLL_SEC)
            continue

        if row_checkpoint_path(row_id) is not None:
            if complete_from_checkpoint_if_possible(running["job_id"], nodelist, row_id):
                time.sleep(5)
                continue
            add_deferred_row(
                row_id,
                "Deferred after final checkpoint existed but separate CPU-only eval completion did not produce metrics.",
            )
            reconcile_and_summarize()
            time.sleep(5)
            continue

        send_ntfy(
            "neuro-collab v100 R3 stable driver",
            f"Launching next incomplete R3 stable row {row_id} in running allocation {running['job_id']} on {nodelist}.",
        )
        proc = launch_row(running["job_id"], nodelist, row_id)

        send_ntfy(
            "neuro-collab v100 R3 stable driver",
            f"Launcher backgrounded row {row_id} in allocation {running['job_id']} with rc={proc.returncode} and token={proc.stdout.strip() or 'none'}. The driver will keep polling for progress and stale-step recovery.",
        )

        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
