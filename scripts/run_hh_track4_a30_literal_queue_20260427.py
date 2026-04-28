#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

SSH_HOST = "falcon2"
NTFY_TOPIC = "dmelendezmaita"
RESOURCE_CLASS_TO_GPUS = {
    "cpu_only": 0,
    "gpu1": 1,
    "gpu2": 2,
    "gpu4": 4,
}
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "BLOCKED"}
RUNNING_STATUSES = {"RUNNING", "LAUNCHING"}
DEFAULT_GPUS_PER_NODE = 4
DEFAULT_MIN_SLEEP_SEC = 5.0
DEFAULT_MAX_SLEEP_SEC = 60.0
DEFAULT_IDLE_SLEEP_SEC = 10.0
DEFAULT_MIN_HEARTBEAT_SEC = 60.0
DEFAULT_MAX_HEARTBEAT_SEC = 300.0
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
MONITOR_WRAPPER = REPO / "scripts" / "run_hh_track4_monitored_command_20260427.py"


@dataclass(frozen=True)
class Task:
    task_id: str
    phase: str
    label: str
    command: str
    resource_class: str
    cpus_per_task: int
    gpus_per_task: int
    expected_runtime_sec: float
    depends_on: tuple[str, ...]
    output_paths: tuple[str, ...]
    monitoring_wrapped: bool


@dataclass
class NodeCapacity:
    node: str
    cpus_total: int
    gpus_total: int
    cpus_used: int = 0
    gpus_used: int = 0

    def free_cpus(self) -> int:
        return self.cpus_total - self.cpus_used

    def free_gpus(self) -> int:
        return self.gpus_total - self.gpus_used


@dataclass
class RunningTask:
    task: Task
    node: str
    launch_name: str
    proc: subprocess.Popen[str]
    stdout_handle: object
    stderr_handle: object
    stdout_log: Path
    stderr_log: Path
    started_at: str
    start_ts: float
    expected_end_ts: float


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Drain a manifest-defined HH Track4 A30 campaign inside a live Falcon allocation, "
            "using ssh falcon2 plus srun with per-task cpu_only, gpu1, gpu2, and gpu4 placement."
        )
    )
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--rerun-failed", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    return ap.parse_args()


def log_line(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def ssh_capture(command: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST, command],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ssh failed: {command}")
    return proc


def ssh_popen(command: str, *, stdout_handle, stderr_handle) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST, command],
        text=True,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )


def send_ntfy(title: str, message: str) -> None:
    try:
        subprocess.run(
            [
                "curl",
                "-fsS",
                "--max-time",
                "5",
                "-H",
                f"Title: {title}",
                "-d",
                message,
                f"https://ntfy.sh/{NTFY_TOPIC}",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        pass


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "campaign"


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def registry_fields() -> list[str]:
    return [
        "task_id",
        "phase",
        "label",
        "status",
        "resource_class",
        "cpus_per_task",
        "gpus_per_task",
        "expected_runtime_sec",
        "depends_on",
        "alloc_job_id",
        "assigned_node",
        "started_at",
        "ended_at",
        "elapsed_sec",
        "return_code",
        "command",
        "stdout_log",
        "stderr_log",
        "notes",
    ]


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = row.get("task_id", "").strip()
        if task_id:
            out[task_id] = row
    return out


def write_registry(path: Path, rows_by_task_id: dict[str, dict[str, str]], task_order: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=registry_fields(), lineterminator="\n")
        writer.writeheader()
        for task_id in task_order:
            writer.writerow(rows_by_task_id[task_id])


def parse_nodes(raw_nodes: object) -> list[str]:
    if isinstance(raw_nodes, list):
        return [str(item).strip() for item in raw_nodes if str(item).strip()]
    if isinstance(raw_nodes, str):
        text = raw_nodes.strip()
        if not text:
            return []
        if "[" in text or "," in text:
            proc = ssh_capture(f"scontrol show hostnames {shlex.quote(text)}", check=False)
            hosts = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if hosts:
                return hosts
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def resolve_alloc_nodelist(alloc_job_id: str) -> str:
    proc = ssh_capture(f"squeue -j {shlex.quote(alloc_job_id)} -h -o '%N' | head -n 1", check=True)
    nodelist = (proc.stdout or "").strip()
    if not nodelist:
        raise RuntimeError(f"Could not resolve a running nodelist for allocation {alloc_job_id}")
    return nodelist.splitlines()[0].strip()


def allocation_state(alloc_job_id: str) -> str:
    proc = ssh_capture(f"squeue -j {shlex.quote(alloc_job_id)} -h -o '%T'", check=False)
    state = (proc.stdout or "").strip()
    if state:
        return state.splitlines()[0].strip().upper()
    proc = ssh_capture(
        f"sacct -j {shlex.quote(alloc_job_id)} --format=State -n -P | head -n 1",
        check=False,
    )
    state = (proc.stdout or "").strip()
    if not state:
        return "UNKNOWN"
    return state.split("|", 1)[0].strip().upper()


def expand_nodes_from_allocation(alloc_job_id: str) -> list[str]:
    nodelist = resolve_alloc_nodelist(alloc_job_id)
    proc = ssh_capture(f"scontrol show hostnames {shlex.quote(nodelist)}", check=True)
    hosts = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not hosts:
        raise RuntimeError(f"Could not expand allocation nodelist {nodelist} for {alloc_job_id}")
    return hosts


def resolve_node_cpu_capacity(nodes: list[str], override: object) -> dict[str, int]:
    if isinstance(override, int):
        return {node: int(override) for node in nodes}
    if isinstance(override, str) and override.strip():
        return {node: int(float(override)) for node in nodes}
    if isinstance(override, dict):
        out: dict[str, int] = {}
        for node in nodes:
            if node not in override:
                raise SystemExit(f"allocation_defaults.cpus_per_node is missing node {node}")
            out[node] = int(float(str(override[node])))
        return out

    out: dict[str, int] = {}
    for node in nodes:
        proc = ssh_capture(f"scontrol show node {shlex.quote(node)}", check=True)
        match = re.search(r"\bCPUTot=(\d+)", proc.stdout or "")
        if not match:
            raise RuntimeError(f"Could not parse CPUTot for node {node}")
        out[node] = int(match.group(1))
    return out


def infer_resource_dir(task: Task, campaign_root: Path) -> Path:
    if task.output_paths:
        raw = Path(task.output_paths[0])
        if raw.suffix:
            return raw.parent / "resource_monitor" / task.task_id
        return raw / "resource_monitor" / task.task_id
    return campaign_root / "resources" / task.task_id


def monitoring_wrap_command(task: Task, campaign_root: Path) -> tuple[str, Path]:
    resource_dir = infer_resource_dir(task, campaign_root)
    if task.monitoring_wrapped:
        command = re.sub(
            r"(--resource-dir\s+)(\S+)",
            lambda m: f"{m.group(1)}{shlex.quote(str(resource_dir))}",
            task.command,
            count=1,
        )
        return command, resource_dir
    wrapped = (
        f"cd {shlex.quote(str(REPO))} && "
        f"{shlex.join([PYTHON_BIN, str(MONITOR_WRAPPER), '--resource-dir', str(resource_dir), '--gpu-expected', str(task.gpus_per_task), '--command', task.command])}"
    )
    return wrapped, resource_dir


def decorated_task(task: Task, campaign_root: Path) -> Task:
    command, _resource_dir = monitoring_wrap_command(task, campaign_root)
    return replace(task, command=command, monitoring_wrapped=True)


def base_registry_row(task: Task, alloc_job_id: str) -> dict[str, str]:
    return {
        "task_id": task.task_id,
        "phase": task.phase,
        "label": task.label,
        "status": "PENDING",
        "resource_class": task.resource_class,
        "cpus_per_task": str(task.cpus_per_task),
        "gpus_per_task": str(task.gpus_per_task),
        "expected_runtime_sec": f"{task.expected_runtime_sec:.3f}",
        "depends_on": json.dumps(list(task.depends_on)),
        "alloc_job_id": alloc_job_id,
        "assigned_node": "",
        "started_at": "",
        "ended_at": "",
        "elapsed_sec": "",
        "return_code": "",
        "command": task.command,
        "stdout_log": "",
        "stderr_log": "",
        "notes": "",
    }


def latest_remote_step_state(alloc_job_id: str, launch_name: str) -> tuple[str, str] | None:
    proc = ssh_capture(
        (
            f"sacct -j {shlex.quote(alloc_job_id)} "
            "--format=JobIDRaw,JobName,State -P -n | "
            f"awk -F'|' '$2==\"{launch_name}\" {{print $1 \"|\" $3}}' | tail -n 1"
        ),
        check=False,
    )
    text = (proc.stdout or "").strip()
    if not text or "|" not in text:
        return None
    step_id, state = text.split("|", 1)
    return step_id.strip(), state.strip().upper()


def outputs_exist(task: Task) -> bool:
    if not task.output_paths:
        return True
    file_paths = [Path(path) for path in task.output_paths if Path(path).suffix]
    dir_paths = [Path(path) for path in task.output_paths if not Path(path).suffix]
    if any(path.exists() for path in file_paths):
        return True

    expected_names = {path.name for path in file_paths}
    for dir_path in dir_paths:
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        if not expected_names:
            if any(dir_path.iterdir()):
                return True
            continue
        for child in dir_path.rglob("*"):
            if child.is_file() and child.name in expected_names:
                return True
    return False


def normalize_depends_on(raw_value: object, task_id: str) -> tuple[str, ...]:
    if raw_value in (None, "", []):
        return ()
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
        return tuple(parts)
    if isinstance(raw_value, list):
        out = []
        for item in raw_value:
            dep = str(item).strip()
            if dep:
                out.append(dep)
        return tuple(out)
    raise SystemExit(f"Task {task_id} has unsupported depends_on={raw_value!r}")


def parse_task(obj: object) -> Task:
    if not isinstance(obj, dict):
        raise SystemExit("Each task entry must be an object.")

    required = [
        "task_id",
        "phase",
        "label",
        "command",
        "resource_class",
        "cpus_per_task",
        "gpus_per_task",
        "expected_runtime_sec",
        "depends_on",
    ]
    missing = [key for key in required if key not in obj]
    if missing:
        raise SystemExit(f"Task entry is missing required keys: {missing}")

    task_id = str(obj["task_id"]).strip()
    if not task_id:
        raise SystemExit("task_id must be non-empty.")

    resource_class = str(obj["resource_class"]).strip()
    if resource_class not in RESOURCE_CLASS_TO_GPUS:
        raise SystemExit(f"Task {task_id} has unsupported resource_class={resource_class!r}")

    cpus_per_task = int(float(str(obj["cpus_per_task"])))
    gpus_per_task = int(float(str(obj["gpus_per_task"])))
    expected_runtime_sec = float(str(obj["expected_runtime_sec"]))
    if cpus_per_task <= 0:
        raise SystemExit(f"Task {task_id} must request cpus_per_task > 0")
    if expected_runtime_sec <= 0:
        raise SystemExit(f"Task {task_id} must provide expected_runtime_sec > 0")

    expected_gpus = RESOURCE_CLASS_TO_GPUS[resource_class]
    if gpus_per_task != expected_gpus:
        raise SystemExit(
            f"Task {task_id} has gpus_per_task={gpus_per_task}, which does not match resource_class={resource_class}"
        )

    depends_on = normalize_depends_on(obj["depends_on"], task_id)
    return Task(
        task_id=task_id,
        phase=str(obj["phase"]).strip(),
        label=str(obj["label"]).strip(),
        command=str(obj["command"]),
        resource_class=resource_class,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        expected_runtime_sec=expected_runtime_sec,
        depends_on=depends_on,
        output_paths=tuple(str(item) for item in obj.get("output_paths", []) if str(item).strip()),
        monitoring_wrapped=bool(obj.get("monitoring_wrapped")),
    )


def load_manifest(path: Path) -> tuple[str, Path, dict[str, object], list[Task]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit("Manifest JSON must be a top-level object.")

    for key in ("campaign_name", "campaign_root", "allocation_defaults", "tasks"):
        if key not in payload:
            raise SystemExit(f"Manifest is missing required top-level key {key!r}")

    campaign_name = str(payload["campaign_name"]).strip()
    campaign_root = Path(str(payload["campaign_root"])).expanduser()
    allocation_defaults = payload["allocation_defaults"]
    if not isinstance(allocation_defaults, dict):
        raise SystemExit("allocation_defaults must be an object.")
    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SystemExit("tasks must be a non-empty list.")

    tasks = [parse_task(item) for item in raw_tasks]
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("Manifest task_id values must be unique.")

    known = set(task_ids)
    for task in tasks:
        for dep in task.depends_on:
            if dep not in known:
                raise SystemExit(f"Task {task.task_id} depends on unknown task {dep}")

    return campaign_name, campaign_root, allocation_defaults, tasks


def merge_manifest_tasks_into_registry(
    *,
    tasks: list[Task],
    rows_by_task_id: dict[str, dict[str, str]],
    task_order: list[str],
    alloc_job_id: str,
) -> bool:
    changed = False
    known = set(rows_by_task_id)
    for task in tasks:
        if task.task_id not in known:
            row = base_registry_row(task, alloc_job_id)
            row["command"] = task.command
            rows_by_task_id[task.task_id] = row
            task_order.append(task.task_id)
            known.add(task.task_id)
            changed = True
            continue

        existing = rows_by_task_id[task.task_id]
        existing["phase"] = task.phase
        existing["label"] = task.label
        existing["resource_class"] = task.resource_class
        existing["cpus_per_task"] = str(task.cpus_per_task)
        existing["gpus_per_task"] = str(task.gpus_per_task)
        existing["expected_runtime_sec"] = f"{task.expected_runtime_sec:.3f}"
        existing["depends_on"] = json.dumps(list(task.depends_on))
        existing["command"] = task.command
        existing["alloc_job_id"] = alloc_job_id
    return changed


def choose_node(task: Task, nodes: dict[str, NodeCapacity]) -> Optional[str]:
    candidates: list[tuple[tuple[object, ...], str]] = []
    for node, capacity in nodes.items():
        if capacity.free_cpus() < task.cpus_per_task:
            continue
        if capacity.free_gpus() < task.gpus_per_task:
            continue

        # Keep CPU-only and GPU-bearing work on separate nodes in the local scheduler
        # model. This matches the observed Slurm behavior much better than naive
        # co-location, and it avoids long-lived step-admission stalls.
        if task.gpus_per_task == 0 and capacity.gpus_used > 0:
            continue
        if task.gpus_per_task > 0 and capacity.gpus_used == 0 and capacity.cpus_used > 0:
            continue

        leftover_cpus = capacity.free_cpus() - task.cpus_per_task
        leftover_gpus = capacity.free_gpus() - task.gpus_per_task

        if task.resource_class == "gpu2":
            score = (
                0 if (capacity.gpus_used == 0 and capacity.cpus_used == 0) else 1,
                capacity.gpus_used,
                leftover_cpus,
                node,
            )
        elif task.resource_class == "gpu1":
            score = (
                0 if (capacity.gpus_used == 0 and capacity.cpus_used == 0) else 1,
                capacity.gpus_used,
                leftover_cpus,
                node,
            )
        else:
            score = (
                0 if (capacity.gpus_used == 0 and capacity.cpus_used == 0) else 1,
                capacity.gpus_used,
                leftover_cpus,
                node,
            )
        candidates.append((score, node))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def allocate(node: NodeCapacity, task: Task) -> None:
    node.cpus_used += task.cpus_per_task
    node.gpus_used += task.gpus_per_task


def release(node: NodeCapacity, task: Task) -> None:
    node.cpus_used = max(0, node.cpus_used - task.cpus_per_task)
    node.gpus_used = max(0, node.gpus_used - task.gpus_per_task)


def summarize_counts(rows_by_task_id: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows_by_task_id.values():
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def dependency_statuses(task: Task, rows_by_task_id: dict[str, dict[str, str]]) -> list[str]:
    return [rows_by_task_id[dep]["status"] for dep in task.depends_on]


def task_is_ready(task: Task, rows_by_task_id: dict[str, dict[str, str]]) -> bool:
    statuses = dependency_statuses(task, rows_by_task_id)
    return all(status == "COMPLETED" for status in statuses)


def task_is_blocked(task: Task, rows_by_task_id: dict[str, dict[str, str]]) -> bool:
    statuses = dependency_statuses(task, rows_by_task_id)
    return any(status in {"FAILED", "BLOCKED"} for status in statuses)


def render_node_snapshot(nodes: dict[str, NodeCapacity]) -> str:
    parts = []
    for node in nodes.values():
        parts.append(
            f"{node.node}:cpu={node.cpus_used}/{node.cpus_total},gpu={node.gpus_used}/{node.gpus_total}"
        )
    return "; ".join(parts)


def heartbeat_interval_sec(
    active: dict[str, RunningTask],
    pending_tasks: list[Task],
    allocation_defaults: dict[str, object],
) -> float:
    min_heartbeat = float(allocation_defaults.get("min_heartbeat_sec", DEFAULT_MIN_HEARTBEAT_SEC))
    max_heartbeat = float(allocation_defaults.get("max_heartbeat_sec", DEFAULT_MAX_HEARTBEAT_SEC))
    if active:
        runtimes = sorted(item.task.expected_runtime_sec for item in active.values())
    elif pending_tasks:
        runtimes = sorted(task.expected_runtime_sec for task in pending_tasks)
    else:
        return max_heartbeat
    median_runtime = runtimes[len(runtimes) // 2]
    return max(min_heartbeat, min(max_heartbeat, median_runtime * 0.75))


def sleep_interval_sec(
    active: dict[str, RunningTask],
    next_heartbeat_at: float,
    allocation_defaults: dict[str, object],
) -> float:
    now_ts = time.time()
    min_sleep = float(allocation_defaults.get("min_sleep_sec", DEFAULT_MIN_SLEEP_SEC))
    max_sleep = float(allocation_defaults.get("max_sleep_sec", DEFAULT_MAX_SLEEP_SEC))
    idle_sleep = float(allocation_defaults.get("idle_sleep_sec", DEFAULT_IDLE_SLEEP_SEC))
    until_heartbeat = max(0.0, next_heartbeat_at - now_ts)
    if not active:
        return max(min_sleep, min(idle_sleep, until_heartbeat if until_heartbeat > 0 else idle_sleep))

    remaining = [max(1.0, item.expected_end_ts - now_ts) for item in active.values()]
    soonest = min(remaining)
    adaptive = max(min_sleep, min(max_sleep, soonest * 0.35))
    if until_heartbeat <= 0:
        return min_sleep
    return max(min_sleep, min(adaptive, until_heartbeat))


def build_remote_srun(task: Task, alloc_job_id: str, node: str, launch_name: str) -> str:
    cmd = [
        "srun",
        f"--jobid={alloc_job_id}",
        "--exact",
        "-N1",
        f"-w{node}",
        "--ntasks=1",
        "--ntasks-per-node=1",
        f"--cpus-per-task={task.cpus_per_task}",
        "--kill-on-bad-exit=1",
        f"--job-name={launch_name}",
    ]
    if task.gpus_per_task > 0:
        cmd.append(f"--gres=gpu:a30:{task.gpus_per_task}")
    cmd.extend(["bash", "-lc", task.command])
    return shlex.join(cmd)


def close_handles(running: RunningTask) -> None:
    try:
        running.stdout_handle.close()
    except Exception:
        pass
    try:
        running.stderr_handle.close()
    except Exception:
        pass


def reconcile_finished(
    active: dict[str, RunningTask],
    nodes: dict[str, NodeCapacity],
    rows_by_task_id: dict[str, dict[str, str]],
    registry_path: Path,
    task_order: list[str],
    alloc_job_id: str,
) -> bool:
    changed = False
    finished_ids = []
    for task_id, running in active.items():
        row = rows_by_task_id[task_id]
        if row["status"] == "LAUNCHING":
            remote = latest_remote_step_state(alloc_job_id, running.launch_name)
            if remote is not None:
                row["status"] = "RUNNING"
                row["notes"] = f"launch_name={running.launch_name} remote_step_id={remote[0]} remote_state={remote[1]}"
                changed = True
        ret = running.proc.poll()
        if ret is None:
            continue

        release(nodes[running.node], running.task)
        close_handles(running)
        finished_ids.append(task_id)
        changed = True

        ended_at = now_iso()
        elapsed = max(0.0, time.time() - running.start_ts)
        if ret == 0 and outputs_exist(running.task):
            row["status"] = "COMPLETED"
        else:
            row["status"] = "FAILED"
        row["ended_at"] = ended_at
        row["elapsed_sec"] = f"{elapsed:.3f}"
        row["return_code"] = str(ret)
        if ret == 0 and row["status"] == "FAILED":
            row["notes"] = "wrapper exited 0 but required outputs were not found"

        if ret == 0:
            send_ntfy(
                f"{row['task_id']} completed",
                (
                    f"phase={row['phase']} task={row['task_id']} label={row['label']} node={running.node} "
                    f"elapsed_sec={elapsed:.1f} expected_runtime_sec={running.task.expected_runtime_sec:.1f}"
                ),
            )
        else:
            send_ntfy(
                f"{row['task_id']} failed",
                (
                    f"task={row['task_id']} label={row['label']} node={running.node} return_code={ret} "
                    f"stdout_log={row['stdout_log']} stderr_log={row['stderr_log']}"
                ),
            )
        log_line(
            f"task {task_id} finished status={row['status']} node={running.node} elapsed_sec={elapsed:.1f}"
        )

    for task_id in finished_ids:
        active.pop(task_id, None)

    if changed:
        write_registry(registry_path, rows_by_task_id, task_order)
    return changed


def refresh_manifest_growth(
    *,
    manifest_path: Path,
    alloc_job_id: str,
    rows_by_task_id: dict[str, dict[str, str]],
    task_order: list[str],
) -> tuple[str, Path, dict[str, object], list[Task], bool]:
    campaign_name, campaign_root, allocation_defaults, refreshed_tasks = load_manifest(manifest_path)
    refreshed_tasks = [decorated_task(task, campaign_root) for task in refreshed_tasks]
    changed = merge_manifest_tasks_into_registry(
        tasks=refreshed_tasks,
        rows_by_task_id=rows_by_task_id,
        task_order=task_order,
        alloc_job_id=alloc_job_id,
    )
    return campaign_name, campaign_root, allocation_defaults, refreshed_tasks, changed


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    campaign_name, campaign_root, allocation_defaults, tasks = load_manifest(manifest_path)
    alloc_job_id = str(
        args.alloc_job_id
        or allocation_defaults.get("alloc_job_id")
        or allocation_defaults.get("allocation_job_id")
        or ""
    ).strip()
    if not alloc_job_id:
        raise SystemExit("Need --alloc-job-id or allocation_defaults.alloc_job_id in the manifest.")
    alloc_state = allocation_state(alloc_job_id)
    if alloc_state not in {"RUNNING"}:
        raise SystemExit(f"Allocation {alloc_job_id} is not RUNNING, current state={alloc_state!r}.")

    campaign_slug = slugify(campaign_name)
    logs_dir = campaign_root / "logs"
    tables_dir = campaign_root / "tables"
    notes_dir = campaign_root / "notes"
    registry_path = tables_dir / f"{campaign_slug}_registry.csv"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    raw_nodes = allocation_defaults.get("nodes")
    nodes_list = parse_nodes(raw_nodes) if raw_nodes is not None else []
    if not nodes_list:
        nodes_list = expand_nodes_from_allocation(alloc_job_id)

    gpus_per_node = int(float(str(allocation_defaults.get("gpus_per_node", DEFAULT_GPUS_PER_NODE))))
    cpu_by_node = resolve_node_cpu_capacity(nodes_list, allocation_defaults.get("cpus_per_node"))
    tasks = [decorated_task(task, campaign_root) for task in tasks]

    node_state = {
        node: NodeCapacity(node=node, cpus_total=cpu_by_node[node], gpus_total=gpus_per_node)
        for node in nodes_list
    }

    if args.validate_only:
        print(json.dumps(
            {
                "campaign_name": campaign_name,
                "campaign_root": str(campaign_root),
                "alloc_job_id": alloc_job_id,
                "nodes": nodes_list,
                "cpus_per_node": cpu_by_node,
                "gpus_per_node": gpus_per_node,
                "task_count": len(tasks),
            },
            indent=2,
        ))
        return 0

    max_cpu_capacity = max(capacity.cpus_total for capacity in node_state.values())
    for task in tasks:
        if task.gpus_per_task > gpus_per_node:
            raise SystemExit(
                f"Task {task.task_id} requests {task.gpus_per_task} GPUs, but each node only has {gpus_per_node}."
            )
        if task.cpus_per_task > max_cpu_capacity:
            raise SystemExit(
                f"Task {task.task_id} requests {task.cpus_per_task} CPUs, but max node capacity is {max_cpu_capacity}."
            )
        if choose_node(task, node_state) is None:
            raise SystemExit(
                f"Task {task.task_id} cannot fit on any configured node with the current per-node limits."
            )

    task_order = [task.task_id for task in tasks]
    rows_by_task_id = load_registry(registry_path)
    merge_manifest_tasks_into_registry(
        tasks=tasks,
        rows_by_task_id=rows_by_task_id,
        task_order=task_order,
        alloc_job_id=alloc_job_id,
    )

    for task in tasks:
        existing = rows_by_task_id[task.task_id]
        if existing["status"] in RUNNING_STATUSES:
            note = (
                f"Recovered stale {existing['status']} row at {now_iso()}; "
                "the prior ssh-backed step cannot be resumed, so the task returned to PENDING."
            )
            existing["status"] = "PENDING"
            existing["assigned_node"] = ""
            existing["started_at"] = ""
            existing["ended_at"] = ""
            existing["elapsed_sec"] = ""
            existing["return_code"] = ""
            existing["stdout_log"] = ""
            existing["stderr_log"] = ""
            existing["notes"] = note
        elif args.rerun_failed and existing["status"] in {"FAILED", "BLOCKED"}:
            existing["status"] = "PENDING"
            existing["assigned_node"] = ""
            existing["started_at"] = ""
            existing["ended_at"] = ""
            existing["elapsed_sec"] = ""
            existing["return_code"] = ""
            existing["stdout_log"] = ""
            existing["stderr_log"] = ""
            existing["notes"] = f"Reset to PENDING at {now_iso()} by --rerun-failed."

    write_registry(registry_path, rows_by_task_id, task_order)

    send_ntfy(
        f"{campaign_name} started",
        (
            f"campaign_name={campaign_name} alloc_job_id={alloc_job_id} nodes={','.join(nodes_list)} "
            f"task_count={len(tasks)} registry_csv={registry_path}"
        ),
    )
    log_line(
        f"campaign started name={campaign_name} alloc_job_id={alloc_job_id} nodes={','.join(nodes_list)}"
    )

    active: dict[str, RunningTask] = {}
    pending_tasks = [task for task in tasks if rows_by_task_id[task.task_id]["status"] not in TERMINAL_STATUSES]
    next_heartbeat_at = time.time() + heartbeat_interval_sec(active, pending_tasks, allocation_defaults)

    try:
        while True:
            alloc_state = allocation_state(alloc_job_id)
            if alloc_state not in {"RUNNING"}:
                interrupted_at = now_iso()
                for task_id, running in list(active.items()):
                    if running.proc.poll() is None:
                        running.proc.terminate()
                        try:
                            running.proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            running.proc.kill()
                            running.proc.wait(timeout=5)
                    release(node_state[running.node], running.task)
                    close_handles(running)
                    row = rows_by_task_id[task_id]
                    row["status"] = "PENDING"
                    row["assigned_node"] = ""
                    row["ended_at"] = interrupted_at
                    row["elapsed_sec"] = f"{max(0.0, time.time() - running.start_ts):.3f}"
                    row["return_code"] = "ALLOC_LOST"
                    row["notes"] = (
                        f"Allocation {alloc_job_id} left RUNNING state and became {alloc_state!r} at {interrupted_at}; "
                        "task reset to PENDING for replay on a fresh allocation."
                    )
                    active.pop(task_id, None)
                write_registry(registry_path, rows_by_task_id, task_order)
                send_ntfy(
                    f"{campaign_name} allocation lost",
                    (
                        f"campaign_name={campaign_name} alloc_job_id={alloc_job_id} "
                        f"state={alloc_state} interrupted_at={interrupted_at}"
                    ),
                )
                raise SystemExit(f"Allocation {alloc_job_id} left RUNNING state: {alloc_state}")

            finished = reconcile_finished(active, node_state, rows_by_task_id, registry_path, task_order, alloc_job_id)

            refreshed_name, refreshed_root, refreshed_defaults, refreshed_tasks, manifest_changed = refresh_manifest_growth(
                manifest_path=manifest_path,
                alloc_job_id=alloc_job_id,
                rows_by_task_id=rows_by_task_id,
                task_order=task_order,
            )
            if manifest_changed:
                campaign_name = refreshed_name
                campaign_root = refreshed_root
                allocation_defaults = refreshed_defaults
                tasks = refreshed_tasks
                write_registry(registry_path, rows_by_task_id, task_order)
                log_line(f"manifest growth detected and merged; task_count={len(tasks)}")

            blocked_changed = False
            for task in tasks:
                row = rows_by_task_id[task.task_id]
                if row["status"] != "PENDING":
                    continue
                if task_is_blocked(task, rows_by_task_id):
                    row["status"] = "BLOCKED"
                    row["ended_at"] = now_iso()
                    row["notes"] = f"Blocked because at least one dependency did not complete: {list(task.depends_on)}"
                    blocked_changed = True
                    log_line(f"task {task.task_id} blocked by failed dependency")
            if blocked_changed:
                write_registry(registry_path, rows_by_task_id, task_order)

            launched = False
            while True:
                made_progress = False
                for task in tasks:
                    row = rows_by_task_id[task.task_id]
                    if row["status"] != "PENDING":
                        continue
                    if not task_is_ready(task, rows_by_task_id):
                        continue

                    node_name = choose_node(task, node_state)
                    if node_name is None:
                        continue

                    capacity = node_state[node_name]
                    allocate(capacity, task)

                    stdout_log = logs_dir / f"{task.task_id}.out"
                    stderr_log = logs_dir / f"{task.task_id}.err"
                    stdout_handle = stdout_log.open("w")
                    stderr_handle = stderr_log.open("w")
                    launch_name = f"{task.task_id}__{uuid.uuid4().hex[:8]}"
                    remote_srun = build_remote_srun(task, alloc_job_id, node_name, launch_name)
                    proc = ssh_popen(remote_srun, stdout_handle=stdout_handle, stderr_handle=stderr_handle)
                    started_at = now_iso()
                    start_ts = time.time()
                    active[task.task_id] = RunningTask(
                        task=task,
                        node=node_name,
                        launch_name=launch_name,
                        proc=proc,
                        stdout_handle=stdout_handle,
                        stderr_handle=stderr_handle,
                        stdout_log=stdout_log,
                        stderr_log=stderr_log,
                        started_at=started_at,
                        start_ts=start_ts,
                        expected_end_ts=start_ts + task.expected_runtime_sec,
                    )

                    row["status"] = "LAUNCHING"
                    row["assigned_node"] = node_name
                    row["started_at"] = started_at
                    row["ended_at"] = ""
                    row["elapsed_sec"] = ""
                    row["return_code"] = ""
                    row["stdout_log"] = str(stdout_log)
                    row["stderr_log"] = str(stderr_log)
                    row["notes"] = f"launch_name={launch_name} waiting for remote step admission"
                    write_registry(registry_path, rows_by_task_id, task_order)

                    send_ntfy(
                        f"{task.task_id} launched",
                        (
                            f"campaign_name={campaign_name} task={task.task_id} label={task.label} node={node_name} "
                            f"resource_class={task.resource_class} cpus={task.cpus_per_task} "
                            f"gpus={task.gpus_per_task} expected_runtime_sec={task.expected_runtime_sec:.1f}"
                        ),
                    )
                    log_line(
                        f"task {task.task_id} launched node={node_name} resource_class={task.resource_class} "
                        f"cpus={task.cpus_per_task} gpus={task.gpus_per_task}"
                    )
                    launched = True
                    made_progress = True
                if not made_progress:
                    break

            statuses = {task_id: rows_by_task_id[task_id]["status"] for task_id in task_order}
            if not active and all(status in TERMINAL_STATUSES for status in statuses.values()):
                refreshed_name, refreshed_root, refreshed_defaults, refreshed_tasks, manifest_changed = refresh_manifest_growth(
                    manifest_path=manifest_path,
                    alloc_job_id=alloc_job_id,
                    rows_by_task_id=rows_by_task_id,
                    task_order=task_order,
                )
                if manifest_changed:
                    campaign_name = refreshed_name
                    campaign_root = refreshed_root
                    allocation_defaults = refreshed_defaults
                    tasks = refreshed_tasks
                    write_registry(registry_path, rows_by_task_id, task_order)
                    continue
                break

            if not active:
                unresolved = [task for task in tasks if rows_by_task_id[task.task_id]["status"] == "PENDING"]
                if unresolved:
                    for task in unresolved:
                        rows_by_task_id[task.task_id]["status"] = "BLOCKED"
                        rows_by_task_id[task.task_id]["ended_at"] = now_iso()
                        rows_by_task_id[task.task_id]["notes"] = (
                            f"Blocked because dependencies never resolved: {list(task.depends_on)}"
                        )
                    write_registry(registry_path, rows_by_task_id, task_order)
                break

            now_ts = time.time()
            pending_tasks = [task for task in tasks if rows_by_task_id[task.task_id]["status"] == "PENDING"]
            if now_ts >= next_heartbeat_at:
                counts = summarize_counts(rows_by_task_id)
                send_ntfy(
                    f"{campaign_name} heartbeat",
                    (
                        f"campaign_name={campaign_name} alloc_job_id={alloc_job_id} "
                        f"running={counts.get('RUNNING', 0)} pending={counts.get('PENDING', 0)} "
                        f"completed={counts.get('COMPLETED', 0)} failed={counts.get('FAILED', 0)} "
                        f"blocked={counts.get('BLOCKED', 0)} nodes={render_node_snapshot(node_state)}"
                    ),
                )
                next_heartbeat_at = time.time() + heartbeat_interval_sec(active, pending_tasks, allocation_defaults)

            if launched or finished or blocked_changed or manifest_changed:
                next_heartbeat_at = time.time() + heartbeat_interval_sec(active, pending_tasks, allocation_defaults)
                continue
            sleep_for = sleep_interval_sec(active, next_heartbeat_at, allocation_defaults)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        interrupted_at = now_iso()
        for task_id, running in list(active.items()):
            if running.proc.poll() is None:
                running.proc.terminate()
                try:
                    running.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    running.proc.kill()
                    running.proc.wait(timeout=5)
            release(node_state[running.node], running.task)
            close_handles(running)
            row = rows_by_task_id[task_id]
            row["status"] = "PENDING"
            row["assigned_node"] = ""
            row["ended_at"] = interrupted_at
            row["elapsed_sec"] = f"{max(0.0, time.time() - running.start_ts):.3f}"
            row["return_code"] = "INTERRUPTED"
            row["notes"] = (
                f"Interrupted at {interrupted_at}; partial logs remain at "
                f"{row['stdout_log']} and {row['stderr_log']}."
            )
            active.pop(task_id, None)
        write_registry(registry_path, rows_by_task_id, task_order)
        send_ntfy(
            f"{campaign_name} interrupted",
            f"campaign_name={campaign_name} alloc_job_id={alloc_job_id} interrupted_at={interrupted_at}",
        )
        raise
    finally:
        for running in list(active.values()):
            close_handles(running)

    counts = summarize_counts(rows_by_task_id)
    send_ntfy(
        f"{campaign_name} ended",
        (
            f"campaign_name={campaign_name} alloc_job_id={alloc_job_id} "
            f"completed={counts.get('COMPLETED', 0)} failed={counts.get('FAILED', 0)} "
            f"blocked={counts.get('BLOCKED', 0)} registry_csv={registry_path}"
        ),
    )
    log_line(
        f"campaign ended completed={counts.get('COMPLETED', 0)} failed={counts.get('FAILED', 0)} "
        f"blocked={counts.get('BLOCKED', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
