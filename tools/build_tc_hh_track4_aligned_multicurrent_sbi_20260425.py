#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260425_tc_hh_track4_aligned_multicurrent_sbi"
DATE_LABEL = "2026-04-25"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"tc_hh_track4_aligned_multicurrent_sbi_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"tc_hh_track4_aligned_multicurrent_sbi_notes_{DATE_TAG}.md"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    params_file = "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"
    rows = []
    variants = [
        ("amcsbi_0001", "fmpe_concat", "fmpe", "concat", "raw_plus_fft256_summary12", "High-value aligned multi-current FMPE baseline."),
        ("amcsbi_0002", "snpe_concat", "snpe", "concat", "raw_plus_fft256_summary12", "High-value aligned multi-current SNPE baseline."),
        ("amcsbi_0003", "snre_meanstd", "snre", "meanstd", "raw_plus_fft256_summary12", "Low-value but explicitly included aligned multi-current ratio-estimation baseline."),
    ]
    for row_id, strategy_id, method, aggregation, feature_mode, desc in variants:
        rows.append(
            {
                "row_id": row_id,
                "phase": "phaseAMCSBI_track4_aligned_multicurrent_sbi",
                "phase_label": "Track4 aligned multicurrent SBI",
                "launch_mode": "tc_cpu_serial",
                "launch_group": "track4_aligned_multicurrent_sbi",
                "family": "track4_aligned_multicurrent_sbi",
                "method": method,
                "aggregation": aggregation,
                "feature_mode": feature_mode,
                "params_file": params_file,
                "currents": "0.1,0.2,0.3,0.4,0.5",
                "n_train": "4096",
                "n_validate": "1024",
                "n_test": "128",
                "seed": "20260425",
                "strategy_id": strategy_id,
                "strategy_description": desc,
                "save_dir": str(OUT_ROOT / "runs" / strategy_id),
                "notes": "Alignment-conditioned multi-current SBI block on CPU.",
            }
        )
    with MATRIX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    NOTES_MD.write_text(
        "\n".join(
            [
                f"# TC HH Track4 Aligned Multicurrent SBI ({DATE_LABEL})",
                "",
                f"- matrix csv: `{MATRIX_CSV}`",
                f"- rows: `{len(rows)}`",
                "",
                "This block assumes row alignment across current files.",
            ]
        )
        + "\n"
    )
    print(f"Wrote {MATRIX_CSV}")


if __name__ == "__main__":
    main()
