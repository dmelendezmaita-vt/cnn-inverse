#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_SBI = REPO / "data/important_notes/optimization_track_20260425_tc_hh_track4_aligned_multicurrent_sbi/runs"
RAM_ROOT = REPO / "data/important_notes/optimization_track_20260425_v100_hh_track4_ram120g_followup"
RAM_RUNS = RAM_ROOT / "runs"
TABLES = RAM_ROOT / "tables"
REPORTS = RAM_ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("fmpe_concat", "fmpe_concat_ram120g"),
        ("snpe_concat", "snpe_concat_ram120g"),
        ("snre_meanstd", "snre_meanstd_ram120g"),
    ]

    rows = []
    for old_name, ram_name in comparisons:
        old_path = OLD_SBI / old_name / "metrics_summary.json"
        ram_path = RAM_RUNS / ram_name / "metrics_summary.json"
        if not old_path.exists() or not ram_path.exists():
            continue
        old = load_json(old_path)
        ram = load_json(ram_path)
        rows.append(
            {
                "run": old_name,
                "old_total_runtime_sec": old.get("total_runtime_sec"),
                "ram120_total_runtime_sec": ram.get("total_runtime_sec"),
                "delta_sec": ram.get("total_runtime_sec") - old.get("total_runtime_sec"),
                "old_training_runtime_sec": old.get("training_runtime_sec"),
                "ram120_training_runtime_sec": ram.get("training_runtime_sec"),
                "old_evaluation_runtime_sec": old.get("evaluation_runtime_sec"),
                "ram120_evaluation_runtime_sec": ram.get("evaluation_runtime_sec"),
            }
        )

    out_csv = TABLES / "ram120_runtime_comparison_final_20260425.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    det_metrics_path = RAM_RUNS / "meanstd_mlp_ram120g" / "metrics_summary.json"
    det_metrics = load_json(det_metrics_path) if det_metrics_path.exists() else None

    report = REPORTS / "ram120_followup_final_20260425.md"
    lines = [
        "# RAM120 Follow-up Final, 2026-04-25",
        "",
        "## Completed RAM-backed runs",
        "",
        "- `fmpe_concat_ram120g`",
        "- `snpe_concat_ram120g`",
        "- `snre_meanstd_ram120g`" if (RAM_RUNS / "snre_meanstd_ram120g" / "metrics_summary.json").exists() else "- `snre_meanstd_ram120g` pending",
        "- representative deterministic rerun `meanstd_mlp_ram120g`" if det_metrics else "- representative deterministic rerun pending",
        "",
        "## Runtime comparison",
        "",
        f"- [ram120_runtime_comparison_final_20260425.csv]({out_csv})" if rows else "- comparison CSV not written yet",
        "",
    ]

    if rows:
        lines.extend(
            [
                "| run | old total runtime sec | RAM120 total runtime sec | delta sec |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['run']}` | {row['old_total_runtime_sec']:.6f} | "
                f"{row['ram120_total_runtime_sec']:.6f} | {row['delta_sec']:.6f} |"
            )
        lines.append("")

    if det_metrics:
        m = det_metrics["metrics_test"]
        lines.extend(
            [
                "## Deterministic representative",
                "",
                "- `meanstd_mlp_ram120g`",
                f"  - `MSE {m['mse']}`",
                f"  - `MAE {m['mae']}`",
                f"  - `R2 {m['r2']}`",
                "",
            ]
        )

    report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
