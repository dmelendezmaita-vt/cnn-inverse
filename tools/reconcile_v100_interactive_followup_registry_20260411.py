#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260411_v100_interactive"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"v100_interactive_followup_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_interactive_followup_registry_{DATE_TAG}.csv"


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
        for row in rows:
            w.writerow(row)


def registry_fields() -> List[str]:
    return [
        "row_id",
        "phase",
        "launch_mode",
        "launch_group",
        "strategy_id",
        "track_key",
        "policy",
        "nodes",
        "seed",
        "status",
        "return_code",
        "alloc_job_id",
        "alloc_nodelist",
        "assigned_nodes",
        "started_at",
        "ended_at",
        "elapsed_sec",
        "run_output_root",
        "stdout_log",
        "stderr_log",
        "params_file",
        "shared_data_dir",
        "notes",
    ]


def parse_runtime(log_path: Path):
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="ignore")
    mt = re.search(r"Runtime - train \[sec\]:\s+([0-9.eE+-]+)", text)
    me = re.search(r"Runtime - eval \[sec\]:\s+([0-9.eE+-]+)", text)
    train = float(mt.group(1)) if mt else None
    eval_ = float(me.group(1)) if me else None
    return train, eval_


def main() -> None:
    matrix_rows = {row["row_id"]: row for row in load_csv(MATRIX_CSV)}
    registry_rows = {row["row_id"]: row for row in load_csv(REGISTRY_CSV)}

    reconciled = 0

    for row_id, row in matrix_rows.items():
        run_output_root = RUNS / row["phase"] / row["launch_mode"] / row_id
        if not run_output_root.exists():
            continue
        subdirs = [p for p in run_output_root.iterdir() if p.is_dir()]
        if not subdirs:
            continue
        task_dir = sorted(subdirs)[0]
        metrics_path = task_dir / "metrics_summary.json"
        if not metrics_path.exists():
            continue

        reg = registry_rows.get(row_id)
        if reg and reg.get("status") == "COMPLETED":
            continue

        started_at = ""
        ended_at = ""
        elapsed_sec = ""
        if reg:
            started_at = reg.get("started_at", "")
            ended_at = reg.get("ended_at", "")
            elapsed_sec = reg.get("elapsed_sec", "")

        # If missing, derive coarse timestamps from filesystem mtimes.
        if not started_at:
            started_at = run_output_root.stat().st_mtime_ns
            started_at = __import__("datetime").datetime.fromtimestamp(started_at / 1e9).astimezone().isoformat(timespec="seconds")
        if not ended_at:
            ended_at = metrics_path.stat().st_mtime_ns
            ended_at = __import__("datetime").datetime.fromtimestamp(ended_at / 1e9).astimezone().isoformat(timespec="seconds")
        if not elapsed_sec:
            try:
                start_ts = __import__("datetime").datetime.fromisoformat(started_at).timestamp()
                end_ts = __import__("datetime").datetime.fromisoformat(ended_at).timestamp()
                elapsed_sec = f"{max(0.0, end_ts - start_ts):.3f}"
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
            "alloc_nodelist": reg.get("alloc_nodelist", "") if reg else "",
            "assigned_nodes": reg.get("assigned_nodes", "") if reg else "",
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_sec": elapsed_sec,
            "run_output_root": str(run_output_root),
            "stdout_log": str(LOGS / f"{row_id}.out"),
            "stderr_log": str(LOGS / f"{row_id}.err"),
            "params_file": row["params_file"],
            "shared_data_dir": row["shared_data_dir"],
            "notes": (reg.get("notes", "") if reg else "") + ("; reconciled_from_run_dir" if reg else "reconciled_from_run_dir"),
        }
        reconciled += 1

    ordered = [registry_rows[k] for k in sorted(registry_rows)]
    write_csv(REGISTRY_CSV, ordered, registry_fields())
    print(json.dumps({"reconciled_rows": reconciled, "registry_csv": str(REGISTRY_CSV)}, indent=2))


if __name__ == "__main__":
    main()
