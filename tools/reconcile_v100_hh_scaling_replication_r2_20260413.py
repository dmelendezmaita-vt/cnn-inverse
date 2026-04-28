#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

DATE_TAG = "20260413_v100_hh_scaling_replication_r2"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"v100_hh_scaling_replication_r2_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_scaling_replication_r2_registry_{DATE_TAG}.csv"


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def registry_fields():
    return [
        "row_id", "phase", "launch_mode", "launch_group", "strategy_id", "track_key", "policy", "nodes", "seed",
        "status", "return_code", "alloc_job_id", "alloc_nodelist", "assigned_nodes", "started_at", "ended_at",
        "elapsed_sec", "run_output_root", "stdout_log", "stderr_log", "params_file", "shared_data_dir", "notes",
    ]


def main():
    matrix_rows = {row["row_id"]: row for row in load_csv(MATRIX_CSV)}
    registry_rows = {row["row_id"]: row for row in load_csv(REGISTRY_CSV)}
    reconciled = 0

    for row_id, row in matrix_rows.items():
        run_root = RUNS / row["phase"] / row["launch_mode"] / row_id
        if not run_root.exists():
            continue
        subdirs = [p for p in run_root.iterdir() if p.is_dir()]
        if not subdirs:
            continue
        task_dir = sorted(subdirs)[0]
        metrics_path = task_dir / "metrics_summary.json"
        if not metrics_path.exists():
            continue

        existing = registry_rows.get(row_id)
        if existing and existing.get("status") == "COMPLETED":
            continue

        started_at = datetime.fromtimestamp(run_root.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        ended_at = datetime.fromtimestamp(metrics_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        try:
            elapsed_sec = f"{max(0.0, datetime.fromisoformat(ended_at).timestamp() - datetime.fromisoformat(started_at).timestamp()):.3f}"
        except Exception:
            elapsed_sec = ""

        registry_rows[row_id] = {
            "row_id": row_id,
            "phase": row["phase"],
            "launch_mode": row["launch_mode"],
            "launch_group": row["launch_group"],
            "strategy_id": row["strategy_id"],
            "track_key": row["track_key"],
            "policy": row["policy"],
            "nodes": row["nodes"],
            "seed": row["seed"],
            "status": "COMPLETED",
            "return_code": "0",
            "alloc_job_id": (task_dir.name.split("_", 1)[0] if "_" in task_dir.name else ""),
            "alloc_nodelist": existing.get("alloc_nodelist", "") if existing else "",
            "assigned_nodes": existing.get("assigned_nodes", "") if existing else "",
            "started_at": existing.get("started_at", started_at) if existing else started_at,
            "ended_at": existing.get("ended_at", ended_at) if existing else ended_at,
            "elapsed_sec": existing.get("elapsed_sec", elapsed_sec) if existing else elapsed_sec,
            "run_output_root": str(run_root),
            "stdout_log": str(LOGS / f"{row_id}.out"),
            "stderr_log": str(LOGS / f"{row_id}.err"),
            "params_file": row["params_file"],
            "shared_data_dir": row["shared_data_dir"],
            "notes": (existing.get("notes", "") + "; reconciled_from_run_dir") if existing else "reconciled_from_run_dir",
        }
        reconciled += 1

    ordered = [registry_rows[k] for k in sorted(registry_rows)]
    write_csv(REGISTRY_CSV, ordered, registry_fields())
    print(json.dumps({"reconciled_rows": reconciled, "registry_csv": str(REGISTRY_CSV)}, indent=2))


if __name__ == "__main__":
    main()
