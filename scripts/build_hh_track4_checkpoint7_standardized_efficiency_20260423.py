#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

DATE_TAG = "20260423_hh_track4_checkpoint7_standardized_efficiency"
DATE_LABEL = "2026-04-23"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

RUNNER = REPO / "scripts" / "run_hh_track4_checkpoint7_standardized_efficiency_20260423.py"
BASE_PARAMS = REPO / "pytorch" / "configs" / "fourth_track_hh_full" / "params_dnn_tar_hh_full.yaml"
MATRIX_CSV = TABLES / f"hh_track4_checkpoint7_standardized_efficiency_matrix_{DATE_TAG}.csv"
SUMMARY_CSV = TABLES / f"hh_track4_checkpoint7_standardized_efficiency_results_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"hh_track4_checkpoint7_standardized_efficiency_notes_{DATE_TAG}.md"
MANIFEST_JSON = NOTES / "manifest.json"


def shell_quote(path: Path | str) -> str:
    raw = str(path)
    return "'" + raw.replace("'", "'\"'\"'") + "'"


def main() -> None:
    for path in (TABLES, NOTES, RUNS, LOGS):
        path.mkdir(parents=True, exist_ok=True)

    baselines = ["random_forest_500", "knn_k11"]
    repeats = [1, 2, 3]
    rows = []
    row_id = 0
    for repeat in repeats:
        for baseline in baselines:
            row_id += 1
            run_id = f"cp7eff_{row_id:04d}_{baseline}_r{repeat}"
            save_dir = RUNS / run_id
            cmd = (
                f"python {shell_quote(RUNNER)} "
                f"--params {shell_quote(BASE_PARAMS)} "
                f"--baseline {baseline} "
                f"--feature-mode raw_plus_fft256_summary12 "
                f"--repeat {repeat} "
                f"--save-dir {shell_quote(save_dir)} "
                f"--summary-csv {shell_quote(SUMMARY_CSV)}"
            )
            rows.append(
                {
                    "row_id": f"cp7eff_{row_id:04d}",
                    "run_id": run_id,
                    "repeat": str(repeat),
                    "baseline": baseline,
                    "feature_mode": "raw_plus_fft256_summary12",
                    "params_file": str(BASE_PARAMS.relative_to(REPO)),
                    "save_dir": str(save_dir),
                    "stdout_log": str(LOGS / f"{run_id}.out"),
                    "stderr_log": str(LOGS / f"{run_id}.err"),
                    "status": "PENDING",
                    "command": cmd,
                    "notes": "CPU-side local/cluster-shell microbenchmark row; builder intentionally does not launch it.",
                }
            )

    with MATRIX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "date_label": DATE_LABEL,
        "package_root": str(OUT_ROOT),
        "runner": str(RUNNER),
        "matrix_csv": str(MATRIX_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "rows": len(rows),
        "repeats": repeats,
        "baselines": baselines,
        "feature_mode": "raw_plus_fft256_summary12",
        "launch_policy": "not_launched_by_builder",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n")

    note_lines = [
        f"# HH Track4 Checkpoint7 Standardized Efficiency Microbenchmark ({DATE_LABEL})",
        "",
        "## Purpose",
        "- define a smallest useful G5 standardized CPU-side efficiency package for classical representatives",
        "- benchmark `random_forest_500` and `knn_k11` on the frozen Track4 split",
        "- hold `feature_mode=raw_plus_fft256_summary12` fixed across all rows",
        "",
        "## Package Layout",
        f"- matrix: `{MATRIX_CSV}`",
        f"- append-only result table target: `{SUMMARY_CSV}`",
        f"- run outputs: `{RUNS}`",
        f"- logs: `{LOGS}`",
        f"- runner: `{RUNNER}`",
        "",
        "## Matrix",
        f"- rows: `{len(rows)}` = 3 repeats x 2 representatives",
        "- status after build: `PENDING`; no jobs are submitted or run by this builder",
        "",
        "## Runner Measurements",
        "- feature/load time, fit time, eval time, total time",
        "- peak RSS via Python `resource` when available",
        "- n_train, n_eval, sec_per_1k_train, ms_per_eval_example",
        "- MAE, MSE, R2 after inverse target scaling/transform",
        "",
        "## Reuse Notes",
        "- runner imports loader/model/target inverse helpers from `run_hh_classical_baseline_20260420.py`",
        "- the fixed feature mode matches the raw+FFT256+summary12 helper behavior used by the SBI baseline path without copying large helper blocks",
    ]
    NOTES_MD.write_text("\n".join(note_lines) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Wrote notes: {NOTES_MD}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
