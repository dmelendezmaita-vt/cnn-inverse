#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260425_tc_hh_track4_aligned_multicurrent_deterministic"
DATE_LABEL = "2026-04-25"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"tc_hh_track4_aligned_multicurrent_deterministic_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"tc_hh_track4_aligned_multicurrent_deterministic_notes_{DATE_TAG}.md"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    rows = []
    params_file = "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"
    currents = "0.1,0.2,0.3,0.4,0.5"
    variants = [
        ("concat_ridge", "ridge", "concat", 0, "Aligned multi-current deterministic baseline using concatenated current-specific richer-observation features and Ridge."),
        ("meanstd_ridge", "ridge", "meanstd", 0, "Aligned multi-current deterministic set-aggregation baseline using mean/std pooling across currents and Ridge."),
        ("concat_mlp_svd512", "mlp", "concat", 512, "Aligned multi-current deterministic neural baseline using concatenated current-specific richer-observation features, SVD compression, and MLP."),
        ("meanstd_mlp", "mlp", "meanstd", 0, "Aligned multi-current deterministic set-aggregation neural baseline using mean/std pooled features and MLP."),
    ]
    for idx, (strategy_id, baseline, aggregation, svd_components, desc) in enumerate(variants, start=1):
        rows.append(
            {
                "row_id": f"tcmcdet_{idx:04d}",
                "phase": "phaseAMC_track4_aligned_multicurrent_deterministic",
                "phase_label": "Track4 aligned multicurrent deterministic",
                "launch_mode": "tc_cpu_serial",
                "launch_group": "track4_aligned_multicurrent_deterministic",
                "family": "track4_aligned_multicurrent_deterministic",
                "baseline": baseline,
                "aggregation": aggregation,
                "feature_mode": "raw_plus_fft256_summary12",
                "svd_components": str(svd_components),
                "params_file": params_file,
                "currents": currents,
                "n_train": "4096",
                "n_validate": "1024",
                "n_test": "1024",
                "seed": "20260425",
                "strategy_id": strategy_id,
                "strategy_description": desc,
                "save_dir": str(OUT_ROOT / "runs" / strategy_id),
                "notes": "Alignment-conditioned multi-current deterministic baseline block.",
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
                f"# TC HH Track4 Aligned Multicurrent Deterministic ({DATE_LABEL})",
                "",
                f"- matrix csv: `{MATRIX_CSV}`",
                f"- rows: `{len(rows)}`",
                "",
                "This block assumes row alignment across current files `0.1` to `0.5`.",
            ]
        )
        + "\n"
    )
    print(f"Wrote {MATRIX_CSV}")


if __name__ == "__main__":
    main()
