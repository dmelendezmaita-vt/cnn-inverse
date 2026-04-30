#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCE_MONITOR_SCRIPT = SCRIPT_DIR / "resource_monitor.py"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a command with the unified repo monitoring contract.")
    ap.add_argument("--resource-dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--gpu-expected", type=int, default=0)
    ap.add_argument("--monitor-python", default=os.environ.get("RESOURCE_MONITOR_PYTHON", sys.executable))
    ap.add_argument("--interval-sec", type=float, default=float(os.environ.get("RESOURCE_MONITOR_INTERVAL_SEC", "5.0")))
    ap.add_argument("command", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    if ns.command and ns.command[0] == "--":
        ns.command = ns.command[1:]
    if not ns.command:
        raise SystemExit("run_monitored_command.py requires a command after '--'.")
    return ns


def maybe_compose_step_id(job_id: str, step_id: str) -> str:
    if not step_id:
        return ""
    if "." in step_id:
        return step_id
    if not job_id:
        return step_id
    return f"{job_id}.{step_id}"


def start_gpu_monitor(resource_dir: Path, gpu_expected: int) -> subprocess.Popen[str] | None:
    if gpu_expected <= 0:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    interval_sec = os.environ.get("GPU_MONITOR_INTERVAL_SEC", "1").strip() or "1"
    query = "index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    monitor_path = resource_dir / "gpu_monitor.csv"
    monitor_cmd = [
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            f"echo 'timestamp,{query}' > {shlex.quote(str(monitor_path))}; "
            "IDS=${CUDA_VISIBLE_DEVICES:-}; "
            "{ while true; do "
            "TS=$(date -Is); "
            "if [ -n \"$IDS\" ]; then "
            f"nvidia-smi --id=\"$IDS\" --query-gpu={query} --format=csv,noheader,nounits; "
            "else "
            f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits; "
            "fi | awk -v ts=\"$TS\" '{print ts \",\" $0}'; "
            f"sleep {shlex.quote(interval_sec)}; "
            "done; } >> "
            f"{shlex.quote(str(monitor_path))}"
        ),
    ]
    return subprocess.Popen(
        monitor_cmd,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_time_file(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ": " in line:
            key, value = line.rsplit(": ", 1)
        elif ":" in line:
            key, value = line.rsplit(":", 1)
        else:
            continue
        out[key.strip()] = value.strip()
    return out


def parse_gpu_monitor(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"gpu_sample_count": 0, "per_gpu": []}
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    by_gpu: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_gpu.setdefault(row["index"].strip(), []).append(row)
    per_gpu = []
    for idx, gpu_rows in sorted(by_gpu.items()):
        util = [float(r["utilization.gpu"].strip()) for r in gpu_rows if r["utilization.gpu"].strip()]
        util_mem = [float(r["utilization.memory"].strip()) for r in gpu_rows if r["utilization.memory"].strip()]
        mem = [float(r["memory.used"].strip()) for r in gpu_rows if r["memory.used"].strip()]
        power = [float(r["power.draw"].strip()) for r in gpu_rows if r["power.draw"].strip()]
        per_gpu.append(
            {
                "gpu_index": idx.strip(),
                "sample_count": len(gpu_rows),
                "utilization_gpu_mean": sum(util) / len(util) if util else None,
                "utilization_gpu_max": max(util) if util else None,
                "utilization_memory_mean": sum(util_mem) / len(util_mem) if util_mem else None,
                "utilization_memory_max": max(util_mem) if util_mem else None,
                "memory_used_mb_mean": sum(mem) / len(mem) if mem else None,
                "memory_used_mb_max": max(mem) if mem else None,
                "power_draw_w_mean": sum(power) / len(power) if power else None,
                "power_draw_w_max": max(power) if power else None,
            }
        )
    return {"gpu_sample_count": len(rows), "per_gpu": per_gpu}


def fetch_sacct_text(job_id: str, step_id: str, out_path: Path) -> str:
    candidate_ids = []
    composed = maybe_compose_step_id(job_id, step_id)
    for item in (composed, step_id, job_id):
        item = item.strip()
        if item and item not in candidate_ids:
            candidate_ids.append(item)
    if not candidate_ids:
        return ""

    best_text = ""
    for _attempt in range(5):
        for candidate in candidate_ids:
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        f"sacct -j {shlex.quote(candidate)} "
                        "--format=JobID,NodeList,Elapsed,State,MaxRSS,AveRSS,AllocTRES -P -n 2>/dev/null || true"
                    ),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            text = (proc.stdout or "").strip()
            if text:
                best_text = text
                out_path.write_text(text + "\n")
                return text
        time.sleep(2)

    if best_text:
        out_path.write_text(best_text + "\n")
    return best_text


def main() -> int:
    args = parse_args()
    resource_dir = Path(args.resource_dir)
    resource_dir.mkdir(parents=True, exist_ok=True)

    env_snapshot = {
        "started_at": now_iso(),
        "label": args.label,
        "gpu_expected": int(args.gpu_expected),
        "monitor_python": str(args.monitor_python),
        "hostname": os.environ.get("HOSTNAME", ""),
        "alloc_job_id": os.environ.get("ALLOC_JOB_ID", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID", ""),
        "slurmd_nodename": os.environ.get("SLURMD_NODENAME", ""),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE", ""),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }
    (resource_dir / "task_env.json").write_text(json.dumps(env_snapshot, indent=2) + "\n")

    raw_monitor_path = resource_dir / "resource_monitor.json"
    gpu_proc = start_gpu_monitor(resource_dir, args.gpu_expected)
    time_file = resource_dir / "resource_time.txt"
    started = time.time()
    proc = subprocess.run(
        [
            str(args.monitor_python),
            str(RESOURCE_MONITOR_SCRIPT),
            "--output-json",
            str(raw_monitor_path),
            "--label",
            args.label,
            "--interval-sec",
            str(args.interval_sec),
            "--",
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_file),
            *list(args.command),
        ],
        text=True,
        check=False,
    )
    elapsed_sec = time.time() - started

    if gpu_proc is not None:
        gpu_proc.terminate()
        try:
            gpu_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gpu_proc.kill()
            gpu_proc.wait(timeout=5)

    sacct_file = resource_dir / "sacct_step.txt"
    alloc_job_id = os.environ.get("ALLOC_JOB_ID", "")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_id = alloc_job_id or slurm_job_id
    step_id = os.environ.get("SLURM_STEP_ID", "")
    sacct_text = fetch_sacct_text(job_id, step_id, sacct_file)

    raw_monitor = {}
    if raw_monitor_path.exists():
        raw_monitor = json.loads(raw_monitor_path.read_text())

    summary = {
        "completed_at": now_iso(),
        "label": args.label,
        "gpu_expected": int(args.gpu_expected),
        "monitor_python": str(args.monitor_python),
        "return_code": proc.returncode,
        "elapsed_sec": elapsed_sec,
        "alloc_job_id": alloc_job_id,
        "slurm_job_id": job_id,
        "slurm_job_id_runtime": slurm_job_id,
        "slurm_step_id": step_id,
        "time_metrics": parse_time_file(time_file),
        "gpu_metrics": parse_gpu_monitor(resource_dir / "gpu_monitor.csv"),
        "resource_monitor_file": str(raw_monitor_path) if raw_monitor_path.exists() else "",
        "resource_monitor_summary": raw_monitor.get("summary", {}),
        "sacct_step_file": str(sacct_file) if sacct_file.exists() else "",
        "sacct_present": bool(sacct_text),
    }
    (resource_dir / "resource_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
