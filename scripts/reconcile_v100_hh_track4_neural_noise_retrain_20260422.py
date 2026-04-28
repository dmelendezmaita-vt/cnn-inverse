#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260422_v100_hh_track4_neural_noise_retrain"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"

DEST_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
DEST_TABLES = DEST_ROOT / "tables"
DEST_NOTES = DEST_ROOT / "notes"
DEST_RUNS = DEST_ROOT / "runs"
DEST_LOGS = DEST_ROOT / "logs"

MATRIX_CSV = DEST_TABLES / f"v100_hh_track4_neural_noise_retrain_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = DEST_TABLES / f"v100_hh_track4_neural_noise_retrain_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = DEST_NOTES / f"v100_hh_track4_neural_noise_retrain_run_log_{DATE_TAG}.json"
NOTES_MD = DEST_NOTES / f"v100_hh_track4_neural_noise_retrain_notes_{DATE_TAG}.md"

SOURCE_ROOT = IMPORTANT / "optimization_track_20260422_v100_hh_track4_neural_noise_probe"
SOURCE_RUNS = SOURCE_ROOT / "runs" / "phaseCP4_track4_v100_neural_noise_retrain" / "concurrent_4x1n"
SOURCE_LOGS = SOURCE_ROOT / "logs"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_csv(path: Path) -> List[Dict[str, str]]:
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


def parse_started_at(stdout_log: Path) -> str:
    if not stdout_log.exists():
        return ""
    for line in stdout_log.read_text().splitlines():
        if line.startswith("2026-") and "[interactive-step]" in line:
            return line.split(" [interactive-step]", 1)[0]
    return ""


def parse_assigned_node(stdout_log: Path) -> str:
    if not stdout_log.exists():
        return ""
    for line in stdout_log.read_text().splitlines():
        if "master=" in line:
            fragment = line.split("master=", 1)[1].split(":", 1)[0].strip()
            return fragment
    return ""


def materialized_run_dir(run_root: Path) -> Path | None:
    subdirs = sorted([p for p in run_root.iterdir() if p.is_dir()]) if run_root.exists() else []
    return subdirs[0] if subdirs else None


def move_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def patch_note(summary: dict, mean_by_strategy: Dict[str, dict]) -> None:
    if not NOTES_MD.exists():
        return
    text = NOTES_MD.read_text()
    marker = "## Variants\n- `effnet_beta005_invvar_e500_noise001`\n- `effnet_beta003_invvar_e400_noise001`\n"
    if marker not in text:
        return
    addition = (
        "\n## Reconciliation\n"
        "- the first live Falcon wave landed under the sibling `v100_hh_track4_neural_noise_probe` package because of a launcher out-root bug\n"
        "- this package was reconciled after completion by moving the finished `v100t4nr_*` runs and logs into the correct retrain root and rebuilding the registry\n"
        f"- reconciled at `{summary['reconciled_at']}`\n"
        "\n## Status\n"
        f"- registry status: `{summary['status_counts']}`\n"
        f"- rows reconciled: `{summary['rows_reconciled']}`\n"
        "\n## Mean Results\n"
        f"- `effnet_beta005_invvar_e500_noise001`: `MSE = {mean_by_strategy['effnet_beta005_invvar_e500_noise001']['mse_mean']:.3f}`, "
        f"`MAE = {mean_by_strategy['effnet_beta005_invvar_e500_noise001']['mae_mean']:.3f}`, "
        f"`R2 = {mean_by_strategy['effnet_beta005_invvar_e500_noise001']['r2_mean']:.6f}`\n"
        f"- `effnet_beta003_invvar_e400_noise001`: `MSE = {mean_by_strategy['effnet_beta003_invvar_e400_noise001']['mse_mean']:.3f}`, "
        f"`MAE = {mean_by_strategy['effnet_beta003_invvar_e400_noise001']['mae_mean']:.3f}`, "
        f"`R2 = {mean_by_strategy['effnet_beta003_invvar_e400_noise001']['r2_mean']:.6f}`\n"
        "\n## Read\n"
        "- the low-noise retraining block does not materially move the neural frontier on the noisy Track 4 task\n"
        "- both variants remain clustered near the previous neural baseline quality rather than opening a new clean or robust frontier\n"
    )
    if "## Reconciliation\n" not in text:
        text = text + addition
        NOTES_MD.write_text(text)


def main() -> None:
    rows = load_csv(MATRIX_CSV)
    registry_rows: List[Dict[str, str]] = []
    metrics_summary: Dict[str, List[dict]] = {}

    for row in rows:
        row_id = row["row_id"]
        src_run_root = SOURCE_RUNS / row_id
        dst_run_root = DEST_RUNS / row["phase"] / row["launch_mode"] / row_id
        src_stdout = SOURCE_LOGS / f"{row_id}.out"
        src_stderr = SOURCE_LOGS / f"{row_id}.err"
        dst_stdout = DEST_LOGS / f"{row_id}.out"
        dst_stderr = DEST_LOGS / f"{row_id}.err"

        if src_run_root.exists():
            move_if_needed(src_run_root, dst_run_root)
        if src_stdout.exists():
            move_if_needed(src_stdout, dst_stdout)
        if src_stderr.exists():
            move_if_needed(src_stderr, dst_stderr)

        run_dir = materialized_run_dir(dst_run_root)
        metrics_json = run_dir / "metrics_summary.json" if run_dir else None
        status = "FAILED"
        return_code = "1"
        if metrics_json and metrics_json.exists():
            status = "COMPLETED"
            return_code = "0"
            data = json.loads(metrics_json.read_text())
            metrics_summary.setdefault(row["strategy_id"], []).append(
                {
                    "mse": float(data["mse"]["test"]),
                    "mae": float(data["mae"]["test"]),
                    "r2": float(data["r2"]["test"]),
                }
            )

        started_at = parse_started_at(dst_stdout)
        ended_at = (
            datetime.fromtimestamp(metrics_json.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            if metrics_json and metrics_json.exists()
            else ""
        )
        elapsed_sec = (
            f"{max(0.0, metrics_json.stat().st_mtime - dst_stdout.stat().st_mtime):.3f}"
            if metrics_json and metrics_json.exists() and dst_stdout.exists()
            else ""
        )
        assigned_node = parse_assigned_node(dst_stdout)
        registry_rows.append(
            {
                "row_id": row_id,
                "phase": row["phase"],
                "launch_mode": row["launch_mode"],
                "launch_group": row["launch_group"],
                "strategy_id": row["strategy_id"],
                "track_key": row["track_key"],
                "policy": row["policy"],
                "nodes": row["nodes"],
                "seed": row["seed"],
                "status": status,
                "return_code": return_code,
                "alloc_job_id": "368960",
                "alloc_nodelist": "fal[117-120]",
                "assigned_nodes": assigned_node,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_sec": elapsed_sec,
                "run_output_root": str(dst_run_root),
                "stdout_log": str(dst_stdout),
                "stderr_log": str(dst_stderr),
                "params_file": row["params_file"],
                "shared_data_dir": row["shared_data_dir"],
                "notes": "Reconciled from the default neural noise probe out root after the live Falcon wave completed.",
            }
        )

    write_csv(REGISTRY_CSV, registry_rows, registry_fields())

    mean_by_strategy = {}
    for strategy_id, values in metrics_summary.items():
        mean_by_strategy[strategy_id] = {
            "mse_mean": sum(v["mse"] for v in values) / len(values),
            "mae_mean": sum(v["mae"] for v in values) / len(values),
            "r2_mean": sum(v["r2"] for v in values) / len(values),
            "n": len(values),
        }

    status_counts: Dict[str, int] = {}
    for row in registry_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    run_log = {
        "reconciled_at": now_iso(),
        "source_root": str(SOURCE_ROOT),
        "dest_root": str(DEST_ROOT),
        "rows_reconciled": len(registry_rows),
        "status_counts": status_counts,
        "mean_by_strategy": mean_by_strategy,
    }
    RUN_LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2) + "\n")
    patch_note(run_log, mean_by_strategy)
    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
