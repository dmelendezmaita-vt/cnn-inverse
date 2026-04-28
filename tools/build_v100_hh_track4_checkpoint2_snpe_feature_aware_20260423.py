#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260423_v100_hh_track4_checkpoint2_snpe_feature_aware"
DATE_LABEL = "2026-04-23"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_snpe_feature_aware_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_snpe_feature_aware_notes_{DATE_TAG}.md"


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
    feature_variants = [
        (
            "summary12",
            "summary12",
            "HH summary-statistics-only SNPE following the informative-features direction from the HH SBI literature.",
        ),
        (
            "raw_plus_fft256_summary12",
            "raw_plus_fft256_summary12",
            "Hybrid raw+FFT+summary SNPE using the strongest observation-richness representation from the classical checkpoint4 package.",
        ),
    ]

    rows = []
    row_num = 0
    for feature_slug, feature_mode, desc in feature_variants:
        for seed in seeds:
            row_num += 1
            rows.append(
                {
                    "row_id": f"v100cp2feat_{row_num:04d}",
                    "phase": "phaseCP2_track4_v100_snpe_feature_aware",
                    "phase_label": "Track4 Checkpoint 2 V100 SNPE feature-aware follow-up",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": "track4_checkpoint2_snpe_feature_aware_v100",
                    "family": "track4_checkpoint2_v100_snpe_feature_aware",
                    "track_key": "track4_hh_full",
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "12",
                    "gpus_per_node": "1",
                    "seed": str(seed),
                    "strategy_id": f"snpe_maf_h192_t8_n12288_{feature_slug}_seed{seed}",
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
                    "task_slug": f"track4_hh_full_snpe_{feature_slug}_seed{seed}",
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
                    "posterior_samples": "16",
                    "compute_map": "0",
                    "decision_rule": "mean",
                    "selection_split": "validate",
                    "selection_cov_lambda": "1.0",
                    "selection_candidate_rules": "mean,median",
                    "notes": "Checkpoint 2 feature-aware SNPE follow-up after validate-calibrated decision-rule selection failed to clear the material-improvement threshold.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SNPE Feature-Aware Follow-Up ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- move to the next closure gate after the validate-calibrated point-rule branch closed negative",
        "- test whether feature-aware observation encodings materially improve the strongest current SNPE configuration",
        "- keep seeds, train budget, and flow capacity fixed so this block isolates the representation effect",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
