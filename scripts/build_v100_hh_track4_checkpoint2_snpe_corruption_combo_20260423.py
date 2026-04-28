#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260423_v100_hh_track4_checkpoint2_snpe_corruption_combo"
DATE_LABEL = "2026-04-23"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_snpe_corruption_combo_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_snpe_corruption_combo_notes_{DATE_TAG}.md"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    shared = str(
        IMPORTANT
        / "optimization_track_20260411_v100_interactive"
        / "shared_data"
        / "track4_hh_full"
        / "concatenated_data"
    )
    params_file = (
        "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_largebudget_followup/"
        "track4_hh_full/params_t4cp2large_snpe_maf_h192_t8_n12288_s1201.yaml"
    )
    seeds = [2001, 2002, 2003, 2004]
    variants = [
        (
            "none",
            "0.0",
            "0.0",
            "SNPE clean-train baseline for clean-cost guardrail.",
        ),
        (
            "combo",
            "0.010",
            "0.20",
            "SNPE robust-combo train corruption with drift010 plus mask20 for clean-cost guardrail.",
        ),
    ]

    rows = []
    row_num = 0
    for seed in seeds:
        for train_mode, train_drift, train_mask, desc in variants:
            row_num += 1
            rows.append(
                {
                    "row_id": f"v100cp2combo_{row_num:04d}",
                    "phase": "phaseCP2_track4_v100_snpe_corruption_combo",
                    "phase_label": "Track4 Checkpoint 2 V100 SNPE robust-combo clean-cost guardrail",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": "track4_checkpoint2_snpe_corruption_combo_clean_v100",
                    "family": "track4_checkpoint2_v100_snpe_corruption_combo",
                    "track_key": "track4_hh_full",
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "12",
                    "gpus_per_node": "1",
                    "seed": str(seed),
                    "strategy_id": f"snpe_maf_h192_t8_n12288_train{train_mode}_clean_seed{seed}",
                    "strategy_description": f"{desc} Eval shift is clean only. Seed={seed}.",
                    "method_name": "snpe",
                    "model_name": "maf",
                    "sample_with": "direct",
                    "params_file": params_file,
                    "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
                    "data_prefix": "concatenated_data",
                    "curr": "0.1",
                    "data_access_mode": "shared_split_cache",
                    "shared_data_dir": shared,
                    "task_slug": f"track4_hh_full_snpe_train{train_mode}_clean_seed{seed}",
                    "n_train": "12288",
                    "eval_limit": "256",
                    "device": "cuda",
                    "feature_mode": "raw_plus_fft256_summary12",
                    "embedding_dim": "64",
                    "embedding_hidden": "256",
                    "hidden_features": "192",
                    "num_transforms": "8",
                    "training_batch_size": "128",
                    "learning_rate": "5.0e-4",
                    "stop_after_epochs": "20",
                    "max_num_epochs": "100",
                    "train_corruption_mode": train_mode,
                    "train_additive_noise_std": "0.0",
                    "train_multiplicative_noise_std": "0.0",
                    "train_baseline_drift_std": train_drift,
                    "train_mask_fraction": train_mask,
                    "eval_shift": "clean",
                    "eval_additive_noise_std": "0.0",
                    "eval_multiplicative_noise_std": "0.0",
                    "eval_baseline_drift_std": "0.0",
                    "eval_mask_fraction": "0.0",
                    "posterior_samples": "16",
                    "compute_map": "0",
                    "decision_rule": "mean",
                    "selection_split": "validate",
                    "selection_cov_lambda": "1.0",
                    "selection_candidate_rules": "mean,median",
                    "notes": "Compact first-wave clean-performance guardrail before shifted robust-combo evaluation; combo train corruption uses drift_std=0.010 and mask_fraction=0.20.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SNPE Robust-Combo First Wave ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- run a compact clean-performance guardrail before shifted combo evaluation",
        "- compare clean-train `none` against robust-combo train corruption using `train_baseline_drift_std=0.010` and `train_mask_fraction=0.20`",
        "- keep eval shifts clean-only in this first wave, with additive, multiplicative, drift, and mask eval corruptions all zero",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
