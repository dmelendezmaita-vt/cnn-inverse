#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO / "data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun"
TABLES = CAMPAIGN_ROOT / "tables"
MANIFEST_JSON = TABLES / "hh_track4_a30_literal_rerun_manifest_20260427.json"
REGISTRY_CSV = TABLES / "hh_track4_a30_literal_rerun_20260427_registry.csv"
MONITOR_SCRIPT = REPO / "tools/run_hh_track4_monitored_command_20260427.py"
DEFAULT_REPORT_JSON = TABLES / "hh_track4_a30_invalid_reset_plan_20260427.json"
DEFAULT_REPORT_CSV = TABLES / "hh_track4_a30_invalid_reset_plan_20260427.csv"
RESET_FIELDS = (
    "status",
    "alloc_job_id",
    "assigned_node",
    "started_at",
    "ended_at",
    "elapsed_sec",
    "return_code",
    "stdout_log",
    "stderr_log",
    "notes",
)


@dataclass(frozen=True)
class ResetCandidate:
    task_id: str
    old_status: str
    reasons: tuple[str, ...]
    manifest_task: dict[str, object]
    registry_row: dict[str, str]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Identify and reset invalid tasks in the clean HH Track4 A30 rerun campaign, "
            "specifically FAILED/BLOCKED rows and completed GPU rows launched before the "
            "post-fix GPU watcher boundary."
        )
    )
    ap.add_argument("--manifest-json", default=str(MANIFEST_JSON))
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    ap.add_argument("--campaign-root", default=str(CAMPAIGN_ROOT))
    ap.add_argument(
        "--gpu-watcher-fix-iso",
        default=None,
        help=(
            "ISO-8601 boundary for the corrected per-GPU watcher deployment. "
            "Completed GPU tasks started before this timestamp are reset. "
            "Default is the monitored-command wrapper mtime."
        ),
    )
    ap.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    ap.add_argument("--report-csv", default=str(DEFAULT_REPORT_CSV))
    ap.add_argument("--apply", action="store_true")
    return ap.parse_args()


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_registry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_registry(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def iso_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def delete_path(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    shutil.rmtree(path)
    return True


def task_resource_dir(task: dict[str, object]) -> Path | None:
    raw = str(task.get("resource_dir", "")).strip()
    return Path(raw) if raw else None


def task_output_paths(task: dict[str, object]) -> list[Path]:
    out = []
    for item in task.get("output_paths", []) or []:
        raw = str(item).strip()
        if raw:
            out.append(Path(raw))
    return out


def task_log_paths(row: dict[str, str]) -> list[Path]:
    out = []
    for key in ("stdout_log", "stderr_log"):
        raw = row.get(key, "").strip()
        if raw:
            out.append(Path(raw))
    return out


def candidate_reasons(row: dict[str, str], boundary_dt: datetime) -> tuple[str, ...]:
    reasons: list[str] = []
    status = row.get("status", "").strip().upper()
    if status == "FAILED":
        reasons.append("failed_status")
    if status == "BLOCKED":
        reasons.append("blocked_status")
    if status == "COMPLETED":
        gpus = int(float(row.get("gpus_per_task", "0") or "0"))
        started_at = row.get("started_at", "").strip()
        if gpus > 0 and started_at:
            if parse_iso(started_at) < boundary_dt:
                reasons.append("completed_gpu_pre_watcher_fix")
    return tuple(reasons)


def collect_candidates(
    manifest_obj: dict[str, object],
    registry_rows: list[dict[str, str]],
    boundary_dt: datetime,
) -> list[ResetCandidate]:
    manifest_tasks = {
        str(task.get("task_id", "")).strip(): task
        for task in manifest_obj.get("tasks", [])
        if str(task.get("task_id", "")).strip()
    }
    out: list[ResetCandidate] = []
    for row in registry_rows:
        task_id = row.get("task_id", "").strip()
        task = manifest_tasks.get(task_id)
        if not task:
            continue
        reasons = candidate_reasons(row, boundary_dt)
        if reasons:
            out.append(
                ResetCandidate(
                    task_id=task_id,
                    old_status=row.get("status", "").strip(),
                    reasons=reasons,
                    manifest_task=task,
                    registry_row=row,
                )
            )
    return out


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in sorted({p for p in paths}, key=lambda p: len(str(p)), reverse=True):
        if path in seen:
            continue
        if any(parent in seen for parent in path.parents):
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def deletion_plan(candidate: ResetCandidate, campaign_root: Path) -> tuple[list[Path], list[Path]]:
    deleteable: list[Path] = []
    skipped: list[Path] = []
    for path in task_output_paths(candidate.manifest_task):
        (deleteable if path_is_within(path, campaign_root) else skipped).append(path)
    resource_dir = task_resource_dir(candidate.manifest_task)
    if resource_dir is not None:
        (deleteable if path_is_within(resource_dir, campaign_root) else skipped).append(resource_dir)
    for path in task_log_paths(candidate.registry_row):
        (deleteable if path_is_within(path, campaign_root) else skipped).append(path)
    return unique_paths(deleteable), unique_paths(skipped)


def apply_candidate_reset(
    candidate: ResetCandidate,
    registry_by_task_id: dict[str, dict[str, str]],
    campaign_root: Path,
) -> dict[str, object]:
    deleted: list[str] = []
    missing: list[str] = []
    skipped_outside_root: list[str] = []
    deleteable, skipped = deletion_plan(candidate, campaign_root)
    for path in skipped:
        skipped_outside_root.append(str(path))
    for path in deleteable:
        if delete_path(path):
            deleted.append(str(path))
        else:
            missing.append(str(path))
    row = dict(registry_by_task_id[candidate.task_id])
    for field in RESET_FIELDS:
        row[field] = ""
    row["status"] = "PENDING"
    registry_by_task_id[candidate.task_id] = row
    return {
        "task_id": candidate.task_id,
        "deleted_paths": deleted,
        "missing_paths": missing,
        "skipped_outside_root": skipped_outside_root,
    }


def plan_record(
    candidate: ResetCandidate,
    boundary_iso: str,
    boundary_source: str,
    campaign_root: Path,
) -> dict[str, object]:
    deleteable, skipped = deletion_plan(candidate, campaign_root)
    task = candidate.manifest_task
    row = candidate.registry_row
    return {
        "task_id": candidate.task_id,
        "phase": str(task.get("phase", "")),
        "label": str(task.get("label", "")),
        "old_status": candidate.old_status,
        "reasons": list(candidate.reasons),
        "started_at": row.get("started_at", ""),
        "ended_at": row.get("ended_at", ""),
        "resource_class": row.get("resource_class", ""),
        "gpus_per_task": row.get("gpus_per_task", ""),
        "boundary_iso": boundary_iso,
        "boundary_source": boundary_source,
        "resource_dir": str(task.get("resource_dir", "")),
        "stdout_log": row.get("stdout_log", ""),
        "stderr_log": row.get("stderr_log", ""),
        "deleteable_paths": [str(p) for p in deleteable],
        "skipped_outside_root": [str(p) for p in skipped],
    }


def write_plan_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "phase",
        "label",
        "old_status",
        "reasons",
        "started_at",
        "ended_at",
        "resource_class",
        "gpus_per_task",
        "boundary_iso",
        "boundary_source",
        "resource_dir",
        "stdout_log",
        "stderr_log",
        "deleteable_paths",
        "skipped_outside_root",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["reasons"] = json.dumps(row["reasons"])
            row["deleteable_paths"] = json.dumps(row["deleteable_paths"])
            row["skipped_outside_root"] = json.dumps(row["skipped_outside_root"])
            writer.writerow(row)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    reason_counts = Counter()
    phase_counts = Counter()
    for rec in records:
        phase_counts[str(rec["phase"])] += 1
        for reason in rec["reasons"]:
            reason_counts[str(reason)] += 1
    return {
        "reset_task_count": len(records),
        "reason_counts": dict(sorted(reason_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_json)
    registry_path = Path(args.registry_csv)
    campaign_root = Path(args.campaign_root)

    if args.gpu_watcher_fix_iso:
        boundary_iso = args.gpu_watcher_fix_iso
        boundary_source = "explicit_arg"
    else:
        boundary_iso = iso_from_mtime(MONITOR_SCRIPT)
        boundary_source = f"monitor_script_mtime:{MONITOR_SCRIPT}"
    boundary_dt = parse_iso(boundary_iso)

    manifest_obj = load_manifest(manifest_path)
    registry_fields, registry_rows = load_registry(registry_path)
    candidates = collect_candidates(manifest_obj, registry_rows, boundary_dt)
    records = [plan_record(c, boundary_iso, boundary_source, campaign_root) for c in candidates]

    summary = summarize(records)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "campaign_root": str(campaign_root),
        "manifest_json": str(manifest_path),
        "registry_csv": str(registry_path),
        "gpu_watcher_fix_boundary_iso": boundary_iso,
        "gpu_watcher_fix_boundary_source": boundary_source,
        "criteria": {
            "failed_or_blocked": "Reset any task whose registry status is FAILED or BLOCKED.",
            "completed_gpu_pre_watcher_fix": (
                "Reset any COMPLETED task with gpus_per_task > 0 whose started_at "
                "timestamp is earlier than the GPU watcher fix boundary, because those "
                "runs were launched before the corrected high-resolution per-GPU watcher."
            ),
            "deletion_scope": (
                "Delete only task output paths, task resource_dir, and task-scoped stdout/stderr logs, "
                "and only when those paths resolve under the clean campaign root."
            ),
        },
        "summary": summary,
        "tasks": records,
    }

    report_json = Path(args.report_json)
    report_csv = Path(args.report_csv)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    write_plan_csv(report_csv, records)

    if args.apply:
        registry_by_task_id = {row["task_id"]: row for row in registry_rows}
        applied = [
            apply_candidate_reset(candidate, registry_by_task_id, campaign_root)
            for candidate in candidates
        ]
        ordered_rows = [registry_by_task_id[row["task_id"]] for row in registry_rows]
        write_registry(registry_path, registry_fields, ordered_rows)
        report["applied"] = {
            "applied_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "details": applied,
        }
        report_json.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["summary"], indent=2))
    print(
        json.dumps(
            {
                "gpu_watcher_fix_boundary_iso": boundary_iso,
                "gpu_watcher_fix_boundary_source": boundary_source,
                "report_json": str(report_json),
                "report_csv": str(report_csv),
                "apply": bool(args.apply),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
