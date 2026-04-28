#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260422_v100_hh_track4_checkpoint2_snpe_map_followup"
DATE_LABEL = "2026-04-22"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_snpe_map_followup_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_snpe_map_followup_notes_{DATE_TAG}.md"


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
    configs = {
        "snpe_maf_h192_t8": (
            "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_followup/track4_hh_full/params_t4cp2snpe_snpe_maf_h192_t8_s1201.yaml",
            "Top MAF follow-up by posterior-mean MSE.",
        ),
        "snpe_maf_h128_t8": (
            "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_followup/track4_hh_full/params_t4cp2snpe_snpe_maf_h128_t8_s1201.yaml",
            "Second-best MAF follow-up with stronger coverage.",
        ),
        "snpe_maf_h128_t5": (
            "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_followup/track4_hh_full/params_t4cp2snpe_snpe_maf_h128_t5_s1201.yaml",
            "Third-best MAF follow-up by posterior-mean MSE.",
        ),
        "snpe_nsf_h128_t5": (
            "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_followup/track4_hh_full/params_t4cp2snpe_snpe_nsf_h128_t5_s1201.yaml",
            "Best NSF baseline from the packed follow-up sweep.",
        ),
    }

    rows = []
    row_num = 0
    for posterior_samples in (16, 32):
        for strategy_id, (params_file, desc) in configs.items():
            row_num += 1
            model_name = "maf" if "_maf_" in strategy_id else "nsf"
            hidden_features = "192" if "_h192_" in strategy_id else "128"
            num_transforms = "8" if "_t8" in strategy_id else "5"
            rows.append(
                {
                    "row_id": f"v100cp2map_{row_num:04d}",
                    "phase": "phaseCP2_track4_v100_snpe_map_followup",
                    "phase_label": "Track4 Checkpoint 2 V100 SNPE MAP follow-up",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": "track4_checkpoint2_snpe_map_followup_v100",
                    "family": "track4_checkpoint2_v100_snpe_map_followup",
                    "track_key": "track4_hh_full",
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "12",
                    "gpus_per_node": "1",
                    "seed": "1201",
                    "strategy_id": f"{strategy_id}_ps{posterior_samples}",
                    "strategy_description": f"{desc} MAP-enabled decision follow-up with posterior_samples={posterior_samples}.",
                    "method_name": "snpe",
                    "model_name": model_name,
                    "sample_with": "direct",
                    "params_file": params_file,
                    "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
                    "data_prefix": "concatenated_data",
                    "curr": "0.1",
                    "data_access_mode": "shared_split_cache",
                    "shared_data_dir": shared,
                    "task_slug": f"track4_hh_full_{strategy_id}_map_ps{posterior_samples}_s1201",
                    "n_train": "4096",
                    "eval_limit": "64",
                    "device": "cuda",
                    "embedding_dim": "64",
                    "embedding_hidden": "256",
                    "hidden_features": hidden_features,
                    "num_transforms": num_transforms,
                    "training_batch_size": "128",
                    "learning_rate": "5.0e-4",
                    "stop_after_epochs": "20",
                    "max_num_epochs": "100",
                    "posterior_samples": str(posterior_samples),
                    "compute_map": "1",
                    "map_num_iter": "30",
                    "map_num_to_optimize": "20",
                    "map_learning_rate": "0.05",
                    "map_num_init_samples": "200",
                    "notes": "Checkpoint 2 MAP follow-up on the top SNPE rows, packed to use all 8 Falcon V100s.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SNPE MAP Follow-up ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- compute MAP-aware decision metrics on the strongest SNPE rows",
        "- pack the full Falcon V100 allocation while comparing posterior moment stability across two posterior-sample budgets",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
