#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260423_v100_hh_track4_checkpoint2_snpe_feature_shift_followup"
DATE_LABEL = "2026-04-23"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_snpe_feature_shift_followup_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_snpe_feature_shift_followup_notes_{DATE_TAG}.md"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    shared = str(
        IMPORTANT
        / "optimization_track_20260411_v100_interactive"
        / "shared_data"
        / "track4_hh_full"
        / "concatenated_data"
    )
    params_file = (
        "src/pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_largebudget_followup/"
        "track4_hh_full/params_t4cp2large_snpe_maf_h192_t8_n12288_s1201.yaml"
    )
    seeds = [2001, 2002, 2003, 2004]
    variants = [
        ("raw", "raw", "0.0", "0.10", "0.0", "Canonical raw-trace SNPE under drift010 shift.", "block1"),
        ("raw", "raw", "0.0", "0.0", "0.20", "Canonical raw-trace SNPE under mask20 shift.", "block1"),
        (
            "raw_plus_fft256_summary12",
            "raw_plus_fft256_summary12",
            "0.0",
            "0.10",
            "0.0",
            "Promoted feature-aware SNPE under drift010 shift.",
            "block2",
        ),
        (
            "raw_plus_fft256_summary12",
            "raw_plus_fft256_summary12",
            "0.0",
            "0.0",
            "0.20",
            "Promoted feature-aware SNPE under mask20 shift.",
            "block2",
        ),
    ]

    rows = []
    row_num = 0
    for seed in seeds:
        for feature_slug, feature_mode, mult_std, drift_std, mask_frac, desc, block in variants:
            row_num += 1
            shift_slug = "drift010" if drift_std != "0.0" else "mask20"
            rows.append(
                {
                    "row_id": f"v100cp2fs_{row_num:04d}",
                    "phase": "phaseCP2_track4_v100_snpe_feature_shift_followup",
                    "phase_label": "Track4 Checkpoint 2 V100 SNPE feature shift follow-up",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": f"track4_checkpoint2_snpe_feature_shift_{block}_v100",
                    "family": "track4_checkpoint2_v100_snpe_feature_shift_followup",
                    "track_key": "track4_hh_full",
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "12",
                    "gpus_per_node": "1",
                    "seed": str(seed),
                    "strategy_id": f"snpe_maf_h192_t8_n12288_{feature_slug}_{shift_slug}_seed{seed}",
                    "strategy_description": f"{desc} Seed={seed}.",
                    "method_name": "snpe",
                    "model_name": "maf",
                    "sample_with": "direct",
                    "params_file": params_file,
                    "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
                    "data_prefix": "concatenated_data",
                    "curr": "0.1",
                    "data_access_mode": "shared_split_cache",
                    "shared_data_dir": shared,
                    "task_slug": f"track4_hh_full_snpe_{feature_slug}_{shift_slug}_seed{seed}",
                    "n_train": "12288",
                    "eval_limit": "256",
                    "device": "cuda",
                    "feature_mode": feature_mode,
                    "embedding_dim": "64",
                    "embedding_hidden": "256",
                    "hidden_features": "192",
                    "num_transforms": "8",
                    "training_batch_size": "128",
                    "learning_rate": "5.0e-4",
                    "stop_after_epochs": "20",
                    "max_num_epochs": "100",
                    "eval_additive_noise_std": "0.0",
                    "eval_multiplicative_noise_std": mult_std,
                    "eval_baseline_drift_std": drift_std,
                    "eval_mask_fraction": mask_frac,
                    "posterior_samples": "16",
                    "compute_map": "0",
                    "decision_rule": "mean",
                    "selection_split": "validate",
                    "selection_cov_lambda": "1.0",
                    "selection_candidate_rules": "mean,median",
                    "notes": "Checkpoint 2 follow-up to test whether the promoted feature-aware SNPE representation improves the localized drift/mask failure modes from checkpoint6.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SNPE Feature Shift Follow-Up ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- follow up the positive clean-data feature-aware SNPE result on the two localized unstable shift families from checkpoint6",
        "- compare raw-trace SNPE against the promoted `raw_plus_fft256_summary12` representation under `drift010` and `mask20` with matched seeds",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
