#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import psutil


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a command while sampling process and GPU resource usage.")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--interval-sec", type=float, default=float(os.environ.get("RESOURCE_MONITOR_INTERVAL_SEC", "5.0")))
    ap.add_argument("command", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    if ns.command and ns.command[0] == "--":
        ns.command = ns.command[1:]
    if not ns.command:
        raise SystemExit("resource_monitor.py requires a command after '--'.")
    return ns


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value in {"[Not Supported]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def collect_processes(root: psutil.Process) -> List[psutil.Process]:
    try:
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        procs = [root]
    alive: List[psutil.Process] = []
    seen = set()
    for proc in procs:
        if proc.pid in seen:
            continue
        seen.add(proc.pid)
        try:
            if proc.is_running():
                alive.append(proc)
        except psutil.Error:
            continue
    return alive


def initialize_cpu_counters(procs: Iterable[psutil.Process]) -> None:
    for proc in procs:
        try:
            proc.cpu_percent(None)
        except psutil.Error:
            continue


def refresh_tracked_processes(
    root: psutil.Process,
    tracked: Dict[int, psutil.Process],
) -> tuple[List[psutil.Process], set[int]]:
    procs = collect_processes(root)
    alive: List[psutil.Process] = []
    seen: set[int] = set()
    new_pids: set[int] = set()
    for proc in procs:
        pid = proc.pid
        seen.add(pid)
        if pid not in tracked:
            tracked[pid] = proc
            new_pids.add(pid)
        alive.append(tracked[pid])
    stale_pids = [pid for pid in tracked if pid not in seen]
    for pid in stale_pids:
        tracked.pop(pid, None)
    return alive, new_pids


def sample_process_tree(
    root: psutil.Process,
    tracked: Dict[int, psutil.Process],
) -> Dict[str, object]:
    procs, new_pids = refresh_tracked_processes(root, tracked)
    initialize_cpu_counters(tracked[pid] for pid in new_pids)
    pid_set = {proc.pid for proc in procs}
    cpu_percent_sum = 0.0
    rss_bytes_sum = 0
    vms_bytes_sum = 0
    top_processes: List[Dict[str, object]] = []
    for proc in procs:
        try:
            with proc.oneshot():
                if proc.pid in new_pids:
                    cpu_percent = 0.0
                else:
                    cpu_percent = float(proc.cpu_percent(None))
                mem = proc.memory_info()
                name = proc.name()
        except psutil.Error:
            continue
        cpu_percent_sum += cpu_percent
        rss_bytes_sum += int(mem.rss)
        vms_bytes_sum += int(mem.vms)
        top_processes.append(
            {
                "pid": proc.pid,
                "name": name,
                "cpu_percent": cpu_percent,
                "rss_mb": float(mem.rss / (1024.0 * 1024.0)),
                "vms_mb": float(mem.vms / (1024.0 * 1024.0)),
            }
        )
    top_processes.sort(key=lambda item: float(item["rss_mb"]), reverse=True)
    return {
        "process_count": len(procs),
        "pid_set": sorted(pid_set),
        "tree_cpu_percent": float(cpu_percent_sum),
        "tree_rss_mb": float(rss_bytes_sum / (1024.0 * 1024.0)),
        "tree_vms_mb": float(vms_bytes_sum / (1024.0 * 1024.0)),
        "top_processes_by_rss": top_processes[:8],
    }


def sample_node_resources() -> Dict[str, object]:
    cpu_percent_percpu = [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)]
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    try:
        load1, load5, load15 = os.getloadavg()
        loadavg = {
            "load1": float(load1),
            "load5": float(load5),
            "load15": float(load15),
        }
    except (AttributeError, OSError):
        loadavg = None
    return {
        "cpu_count_logical": int(psutil.cpu_count(logical=True) or 0),
        "cpu_count_physical": int(psutil.cpu_count(logical=False) or 0),
        "cpu_percent_percpu": cpu_percent_percpu,
        "cpu_percent_mean": float(sum(cpu_percent_percpu) / len(cpu_percent_percpu)) if cpu_percent_percpu else 0.0,
        "loadavg": loadavg,
        "memory_total_mb": float(vm.total / (1024.0 * 1024.0)),
        "memory_available_mb": float(vm.available / (1024.0 * 1024.0)),
        "memory_used_mb": float(vm.used / (1024.0 * 1024.0)),
        "memory_percent": float(vm.percent),
        "swap_total_mb": float(swap.total / (1024.0 * 1024.0)),
        "swap_used_mb": float(swap.used / (1024.0 * 1024.0)),
        "swap_percent": float(swap.percent),
    }


def query_nvidia_smi() -> List[Dict[str, object]]:
    if not shutil_which("nvidia-smi"):
        return []
    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc_cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_raw = subprocess.check_output(gpu_cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    process_rows: List[Dict[str, object]] = []
    try:
        proc_raw = subprocess.check_output(proc_cmd, text=True, stderr=subprocess.DEVNULL)
        for line in proc_raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            process_rows.append(
                {
                    "pid": pid,
                    "gpu_uuid": parts[1],
                    "used_gpu_memory_mb": safe_float(parts[2]),
                }
            )
    except Exception:
        process_rows = []
    gpus: List[Dict[str, object]] = []
    for line in gpu_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu_percent": safe_float(parts[3]),
                "utilization_memory_percent": safe_float(parts[4]),
                "memory_used_mb": safe_float(parts[5]),
                "memory_total_mb": safe_float(parts[6]),
                "processes": process_rows,
            }
        )
    return gpus


def add_gpu_attribution(gpus: List[Dict[str, object]], pid_set: set[int]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for gpu in gpus:
        step_processes = [
            row
            for row in gpu.get("processes", [])
            if isinstance(row, dict)
            and int(row.get("pid", -1)) in pid_set
            and str(row.get("gpu_uuid")) == str(gpu.get("uuid"))
        ]
        step_gpu_memory_mb = sum(float(row.get("used_gpu_memory_mb") or 0.0) for row in step_processes)
        record = dict(gpu)
        record["step_process_gpu_memory_mb"] = float(step_gpu_memory_mb)
        record["step_processes"] = step_processes
        record.pop("processes", None)
        enriched.append(record)
    return enriched


def shutil_which(binary: str) -> str | None:
    import shutil

    return shutil.which(binary)


def summarize_samples(samples: List[Dict[str, object]]) -> Dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "max_tree_cpu_percent": None,
            "mean_tree_cpu_percent": None,
            "max_tree_rss_mb": None,
            "max_tree_vms_mb": None,
            "gpu_by_uuid": {},
        }
    cpu_values = [float(sample["process_tree"]["tree_cpu_percent"]) for sample in samples]
    rss_values = [float(sample["process_tree"]["tree_rss_mb"]) for sample in samples]
    vms_values = [float(sample["process_tree"]["tree_vms_mb"]) for sample in samples]
    gpu_by_uuid: Dict[str, Dict[str, object]] = {}
    for sample in samples:
        for gpu in sample.get("gpus", []):
            uuid = str(gpu.get("uuid"))
            entry = gpu_by_uuid.setdefault(
                uuid,
                {
                    "index": gpu.get("index"),
                    "name": gpu.get("name"),
                    "max_utilization_gpu_percent": 0.0,
                    "max_utilization_memory_percent": 0.0,
                    "max_memory_used_mb": 0.0,
                    "max_step_process_gpu_memory_mb": 0.0,
                },
            )
            entry["max_utilization_gpu_percent"] = max(
                float(entry["max_utilization_gpu_percent"]),
                float(gpu.get("utilization_gpu_percent") or 0.0),
            )
            entry["max_utilization_memory_percent"] = max(
                float(entry["max_utilization_memory_percent"]),
                float(gpu.get("utilization_memory_percent") or 0.0),
            )
            entry["max_memory_used_mb"] = max(
                float(entry["max_memory_used_mb"]),
                float(gpu.get("memory_used_mb") or 0.0),
            )
            entry["max_step_process_gpu_memory_mb"] = max(
                float(entry["max_step_process_gpu_memory_mb"]),
                float(gpu.get("step_process_gpu_memory_mb") or 0.0),
            )
    max_node_cpu_percpu: List[float] = []
    cpu_width = max(len(sample.get("node", {}).get("cpu_percent_percpu", [])) for sample in samples)
    for idx in range(cpu_width):
        max_node_cpu_percpu.append(
            max(
                float(sample.get("node", {}).get("cpu_percent_percpu", [0.0] * cpu_width)[idx] or 0.0)
                for sample in samples
            )
        )
    node_memory_used = [float(sample.get("node", {}).get("memory_used_mb") or 0.0) for sample in samples]
    node_memory_percent = [float(sample.get("node", {}).get("memory_percent") or 0.0) for sample in samples]
    node_swap_used = [float(sample.get("node", {}).get("swap_used_mb") or 0.0) for sample in samples]
    node_cpu_mean = [float(sample.get("node", {}).get("cpu_percent_mean") or 0.0) for sample in samples]
    return {
        "sample_count": len(samples),
        "max_tree_cpu_percent": max(cpu_values),
        "mean_tree_cpu_percent": sum(cpu_values) / len(cpu_values),
        "max_tree_rss_mb": max(rss_values),
        "max_tree_vms_mb": max(vms_values),
        "max_node_cpu_percent_mean": max(node_cpu_mean),
        "max_node_cpu_percent_percpu": max_node_cpu_percpu,
        "max_node_memory_used_mb": max(node_memory_used),
        "max_node_memory_percent": max(node_memory_percent),
        "max_node_swap_used_mb": max(node_swap_used),
        "gpu_by_uuid": gpu_by_uuid,
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now_iso()
    command_started_ts = time.time()
    proc = subprocess.Popen(args.command)
    root = psutil.Process(proc.pid)
    tracked_processes: Dict[int, psutil.Process] = {}
    initialize_cpu_counters(collect_processes(root))
    psutil.cpu_percent(interval=None, percpu=True)

    samples: List[Dict[str, object]] = []
    first_sample = True
    while True:
        if not first_sample:
            time.sleep(args.interval_sec)
        first_sample = False
        process_tree = sample_process_tree(root, tracked_processes)
        node = sample_node_resources()
        pid_set = set(process_tree["pid_set"])
        gpus = add_gpu_attribution(query_nvidia_smi(), pid_set)
        samples.append(
            {
                "timestamp": utc_now_iso(),
                "elapsed_sec": float(time.time() - command_started_ts),
                "node": node,
                "process_tree": process_tree,
                "gpus": gpus,
            }
        )
        if proc.poll() is not None:
            break

    payload = {
        "label": args.label,
        "command": args.command,
        "hostname": socket.gethostname(),
        "monitor_pid": os.getpid(),
        "root_pid": proc.pid,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "slurm_nodeid": os.environ.get("SLURM_NODEID"),
        "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
        "poll_interval_sec": float(args.interval_sec),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "returncode": proc.returncode,
        "cpu_count": os.cpu_count(),
        "samples": samples,
        "summary": summarize_samples(samples),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
