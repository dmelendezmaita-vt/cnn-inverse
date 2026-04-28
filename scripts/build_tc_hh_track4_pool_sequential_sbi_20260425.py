#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260425_tc_hh_track4_pool_sequential_sbi"
DATE_LABEL = "2026-04-25"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"tc_hh_track4_pool_sequential_sbi_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"tc_hh_track4_pool_sequential_sbi_notes_{DATE_TAG}.md"

SHARED_DATA_DIR = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    params_file = "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"
    common = {
        "phase": "phaseTC_track4_pool_sequential_sbi",
        "phase_label": "Track4 pool sequential SBI",
        "launch_mode": "tc_cpu_serial",
        "launch_group": "track4_pool_sequential_sbi",
        "family": "track4_pool_sequential_sbi",
        "params_file": params_file,
        "data_dir": str(SHARED_DATA_DIR),
        "data_prefix": "concatenated_data",
        "curr": "0.1",
        "feature_mode": "raw_plus_fft256_summary12",
        "pool_size": "4096",
        "final_train_size": "2048",
        "n_validate": "1024",
        "n_test": "1024",
        "candidate_pool_size": "1024",
        "candidate_posterior_samples": "4",
        "posterior_samples": "8",
        "training_batch_size": "128",
        "learning_rate": "5.0e-4",
        "stop_after_epochs": "12",
        "max_num_epochs": "60",
        "embedding_dim": "64",
        "embedding_hidden": "256",
        "hidden_features": "128",
        "num_transforms": "5",
        "decision_rule": "mean",
        "seed": "20260425",
    }
    variants = [
        {
            "row_id": "pssbi_0001",
            "strategy_id": "fmpe_mlp_r2_pool4096_final2048",
            "strategy_description": "Two-round FMPE baseline over a fixed 4096-example cached HH pool, finishing on 2048 acquired examples.",
            "method": "fmpe",
            "density_estimator": "mlp",
            "sample_with": "ode",
            "rounds": "2",
            "initial_train_size": "1024",
        },
        {
            "row_id": "pssbi_0002",
            "strategy_id": "snpe_maf_r3_pool4096_final2048",
            "strategy_description": "Three-round SNPE baseline over the same fixed HH pool, using uncertainty-guided pool expansion on CPU.",
            "method": "snpe",
            "density_estimator": "maf",
            "sample_with": "direct",
            "rounds": "3",
            "initial_train_size": "512",
        },
    ]

    rows = []
    for variant in variants:
        strategy_id = variant["strategy_id"]
        row = {
            **common,
            **variant,
            "save_dir": str(OUT_ROOT / "runs" / strategy_id),
            "notes": "CPU-capable pool-based sequential SBI over precomputed HH split arrays.",
        }
        rows.append(row)

    with MATRIX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    NOTES_MD.write_text(
        "\n".join(
            [
                f"# TC HH Track4 Pool Sequential SBI ({DATE_LABEL})",
                "",
                f"- matrix csv: `{MATRIX_CSV}`",
                f"- rows: `{len(rows)}`",
                "",
                "This block reuses the cached HH split arrays in the shared data mirror,",
                "treats the train split as a fixed candidate pool, and grows the SBI train set",
                "round by round by ranking remaining observations with posterior uncertainty.",
            ]
        )
        + "\n"
    )
    print(f"Wrote {MATRIX_CSV}")


if __name__ == "__main__":
    main()
