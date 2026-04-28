#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260427_hh_track4_a30_literal_rerun"
)
MANIFEST_JSON = CAMPAIGN_ROOT / "tables" / "hh_track4_a30_literal_rerun_manifest_20260427.json"
REGISTRY_CSV = CAMPAIGN_ROOT / "tables" / "hh_track4_a30_literal_rerun_20260427_registry.csv"
MONITOR_SCRIPT = REPO / "tools" / "run_hh_track4_monitored_command_20260427.py"
RESET_REPORT_JSON = CAMPAIGN_ROOT / "notes" / "hh_track4_a30_invalid_reset_20260427.json"


@dataclass(frozen=True)
class TaskMeta:
    task_id: str
    gpus_per_task: int
    depends_on: tuple[str, ...]
    output_paths: tuple[str, ...]
    resource_dir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Reset invalid clean A30 rerun tasks by deleting incorrect outputs and "
            "returning only the affected registry rows to PENDING."
        )
    )
    ap.add_argument("--campaign-root", default=str(CAMPAIGN_ROOT))
    ap.add_argument("--manifest-json", default=str(MANIFEST_JSON))
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    ap.add_argument("--watcher-fix-ts", default="")
    ap.add_argument("--report-json", default=str(RESET_REPORT_JSON))
    ap.add_argument("--apply", action="store_true")
    return ap.parse_args()


def load_manifest(path: Path) -> dict[str, TaskMeta]:
    obj = json.loads(path.read_text())
    out: dict[str, TaskMeta] = {}
    for task in obj["tasks"]:
        resource_dir = str(task.get("resource_dir", "")).strip()
        out[str(task["task_id"])] = TaskMeta(
            task_id=str(task["task_id"]),
            gpus_per_task=int(task.get("gpus_per_task", 0)),
            depends_on=tuple(str(dep) for dep in task.get("depends_on", []) if str(dep).strip()),
            output_paths=tuple(str(path) for path in task.get("output_paths", []) if str(path).strip()),
            resource_dir=resource_dir,
        )
    return out


def load_registry(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, {row["task_id"]: row for row in rows}


def write_registry(path: Path, fieldnames: list[str], rows_by_task: dict[str, dict[str, str]]) -> None:
    ordered = [rows_by_task[task_id] for task_id in sorted(rows_by_task)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def watcher_fix_timestamp(path: Path) -> str:
    return path.stat().st_mtime_ns and __import__("datetime").datetime.fromtimestamp(
        path.stat().st_mtime, __import__("datetime").timezone(__import__("datetime").timedelta(hours=-4))
    ).isoformat()


def read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def task_run_root(task: TaskMeta) -> Path | None:
    dir_candidates = [Path(path) for path in task.output_paths if Path(path).suffix == ""]
    if dir_candidates:
        return dir_candidates[0]
    file_candidates = [Path(path) for path in task.output_paths if Path(path).suffix]
    if file_candidates:
        parent = file_candidates[0].parent
        if "runs" in parent.parts:
            return parent
    return None


def candidate_resource_dirs(task: TaskMeta) -> list[Path]:
    candidates: list[Path] = []
    if task.resource_dir:
        candidates.append(Path(task.resource_dir))
    root = task_run_root(task)
    if root is not None:
        candidates.append(root / "resource_monitor" / task.task_id)
        candidates.append(root / "resource_monitor")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def resource_summary_path(task: TaskMeta) -> Path | None:
    for resource_dir in candidate_resource_dirs(task):
        summary = resource_dir / "resource_summary.json"
        if summary.exists():
            return summary
    return None


def direct_failure_seeds(
    manifest: dict[str, TaskMeta],
    registry: dict[str, dict[str, str]],
    watcher_fix_ts: str,
) -> tuple[set[str], dict[str, str]]:
    seeds: set[str] = set()
    reasons: dict[str, str] = {}
    for task_id, row in registry.items():
        status = row["status"]
        task = manifest.get(task_id)
        if task is None:
            continue

        if status in {"FAILED", "BLOCKED"}:
            seeds.add(task_id)
            reasons[task_id] = f"status={status}"
            continue

        if status != "COMPLETED":
            continue

        summary_path = resource_summary_path(task)
        summary = read_summary(summary_path) if summary_path else None
        if summary is None:
            if task.gpus_per_task > 0:
                seeds.add(task_id)
                reasons[task_id] = "missing_resource_summary_for_completed_gpu_task"
            continue

        if task.gpus_per_task <= 0:
            continue

        gpu_metrics = summary.get("gpu_metrics") or {}
        completed_at = str(summary.get("completed_at", ""))
        gpu_sample_count = gpu_metrics.get("gpu_sample_count")
        if not completed_at:
            seeds.add(task_id)
            reasons[task_id] = "missing_completed_at_for_completed_gpu_task"
            continue
        if completed_at < watcher_fix_ts and (gpu_sample_count is None or int(gpu_sample_count) <= 4):
            seeds.add(task_id)
            reasons[task_id] = (
                f"pre_fix_gpu_monitor completed_at={completed_at} gpu_sample_count={gpu_sample_count}"
            )
    return seeds, reasons


def dependency_closure(seed_ids: set[str], manifest: dict[str, TaskMeta]) -> set[str]:
    dependents: dict[str, list[str]] = defaultdict(list)
    for task in manifest.values():
        for dep in task.depends_on:
            dependents[dep].append(task.task_id)

    closure = set(seed_ids)
    queue = deque(seed_ids)
    while queue:
        task_id = queue.popleft()
        for child in dependents.get(task_id, []):
            if child not in closure:
                closure.add(child)
                queue.append(child)
    return closure


def delete_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def clear_task_artifacts(task: TaskMeta, row: dict[str, str]) -> dict[str, int]:
    counts = Counter()
    run_root = task_run_root(task)
    if run_root is not None and delete_path(run_root):
        counts["run_roots_deleted"] += 1

    deleted_resource_dirs: set[Path] = set()
    for resource_dir in candidate_resource_dirs(task):
        if resource_dir.exists() and resource_dir not in deleted_resource_dirs:
            delete_path(resource_dir)
            deleted_resource_dirs.add(resource_dir)
            counts["resource_dirs_deleted"] += 1

    for key in ["stdout_log", "stderr_log"]:
        value = row.get(key, "").strip()
        if value and delete_path(Path(value)):
            counts["log_files_deleted"] += 1

    for path_str in task.output_paths:
        path = Path(path_str)
        if run_root is not None and path == run_root:
            continue
        if any(resource_dir in path.parents for resource_dir in deleted_resource_dirs):
            continue
        if delete_path(path):
            counts["output_paths_deleted"] += 1

    return dict(counts)


def reset_registry_row(row: dict[str, str], note: str) -> None:
    row["status"] = "PENDING"
    row["assigned_node"] = ""
    row["started_at"] = ""
    row["ended_at"] = ""
    row["elapsed_sec"] = ""
    row["return_code"] = ""
    row["stdout_log"] = ""
    row["stderr_log"] = ""
    row["notes"] = note


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    manifest_json = Path(args.manifest_json).expanduser().resolve()
    registry_csv = Path(args.registry_csv).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve()
    watcher_fix_ts = args.watcher_fix_ts or watcher_fix_timestamp(MONITOR_SCRIPT)

    manifest = load_manifest(manifest_json)
    fieldnames, registry = load_registry(registry_csv)

    seed_ids, reasons = direct_failure_seeds(manifest, registry, watcher_fix_ts)
    reset_ids = dependency_closure(seed_ids, manifest)

    artifact_counts = Counter()
    reset_status_counts = Counter()
    for task_id in sorted(reset_ids):
        row = registry.get(task_id)
        task = manifest.get(task_id)
        if row is None or task is None:
            continue
        reset_status_counts[row["status"]] += 1
        if args.apply:
            artifact_counts.update(clear_task_artifacts(task, row))
            reason = reasons.get(task_id, "upstream_dependency_reset")
            reset_registry_row(row, f"Reset at watcher_fix_ts={watcher_fix_ts}; reason={reason}")

    legacy_resource_dir = campaign_root / "tables" / "resource_monitor"
    if args.apply and legacy_resource_dir.exists():
        shutil.rmtree(legacy_resource_dir)
        artifact_counts["legacy_shared_resource_dir_deleted"] += 1

    if args.apply:
        write_registry(registry_csv, fieldnames, registry)

    report = {
        "campaign_root": str(campaign_root),
        "manifest_json": str(manifest_json),
        "registry_csv": str(registry_csv),
        "watcher_fix_ts": watcher_fix_ts,
        "seed_task_count": len(seed_ids),
        "reset_task_count": len(reset_ids),
        "seed_reasons": {task_id: reasons[task_id] for task_id in sorted(seed_ids)},
        "reset_status_counts": dict(reset_status_counts),
        "artifact_counts": dict(artifact_counts),
        "apply": bool(args.apply),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
