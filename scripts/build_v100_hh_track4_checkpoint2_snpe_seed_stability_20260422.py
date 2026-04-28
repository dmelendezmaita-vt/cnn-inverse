#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260422_v100_hh_track4_checkpoint2_snpe_seed_stability"
DATE_LABEL = "2026-04-22"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_snpe_seed_stability_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_snpe_seed_stability_notes_{DATE_TAG}.md"


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

    configs = [
        (
            "snpe_maf_h192_t8_n12288",
            "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_largebudget_followup/track4_hh_full/params_t4cp2large_snpe_maf_h192_t8_n12288_s1201.yaml",
            "12288",
            "192",
            "8",
            "Best large-budget SNPE candidate.",
        ),
        (
            "snpe_maf_h192_t8_n8192",
            "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_largebudget_followup/track4_hh_full/params_t4cp2large_snpe_maf_h192_t8_n8192_s1201.yaml",
            "8192",
            "192",
            "8",
            "Best medium-budget SNPE comparison candidate.",
        ),
    ]
    seeds = [2001, 2002, 2003, 2004]

    rows = []
    row_num = 0
    for strategy_prefix, params_file, n_train, hidden_features, num_transforms, desc in configs:
        for seed in seeds:
            row_num += 1
            rows.append(
                {
                    "row_id": f"v100cp2seed_{row_num:04d}",
                    "phase": "phaseCP2_track4_v100_snpe_seed_stability",
                    "phase_label": "Track4 Checkpoint 2 V100 SNPE seed stability",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": "track4_checkpoint2_snpe_seed_stability_v100",
                    "family": "track4_checkpoint2_v100_snpe_seed_stability",
                    "track_key": "track4_hh_full",
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "12",
                    "gpus_per_node": "1",
                    "seed": str(seed),
                    "strategy_id": f"{strategy_prefix}_seed{seed}",
                    "strategy_description": f"{desc} Stability run with model seed={seed}.",
                    "method_name": "snpe",
                    "model_name": "maf",
                    "sample_with": "direct",
                    "params_file": params_file,
                    "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
                    "data_prefix": "concatenated_data",
                    "curr": "0.1",
                    "data_access_mode": "shared_split_cache",
                    "shared_data_dir": shared,
                    "task_slug": f"track4_hh_full_{strategy_prefix}_seed{seed}",
                    "n_train": n_train,
                    "eval_limit": "256",
                    "device": "cuda",
                    "embedding_dim": "64",
                    "embedding_hidden": "256",
                    "hidden_features": hidden_features,
                    "num_transforms": num_transforms,
                    "training_batch_size": "128",
                    "learning_rate": "5.0e-4",
                    "stop_after_epochs": "20",
                    "max_num_epochs": "100",
                    "posterior_samples": "16",
                    "compute_map": "0",
                    "notes": "Checkpoint 2 SNPE seed-stability confirmation on the strongest large-budget candidates.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SNPE Seed Stability ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- verify that the best SNPE gains survive model-seed variation before using them as the checkpoint-2 family conclusion",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
