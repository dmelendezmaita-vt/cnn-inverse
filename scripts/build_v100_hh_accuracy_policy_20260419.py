#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260419_v100_hh_accuracy_policy"
DATE_LABEL = "2026-04-19"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_accuracy_policy_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_accuracy_policy_notes_{DATE_TAG}.md"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    rows = []
    row_num = 0

    def add_row(
        *,
        phase: str,
        phase_label: str,
        launch_mode: str,
        launch_group: str,
        track_key: str,
        nodes: int,
        seed: int,
        strategy_id: str,
        strategy_description: str,
        params_file: str,
        tar_path: str,
        data_prefix: str,
        shared_data_dir: str,
        batch_profile: str,
        learning_rate: str,
        features_sub_length: str,
        ntrain: str,
        global_train_batch_size: str,
        save_predictions: str,
        save_diagnostic_plots: str,
        save_checkpoints_epochs: str,
        notes: str,
    ) -> None:
        nonlocal row_num
        row_num += 1
        rows.append(
            {
                "row_id": f"v100accpol_{row_num:04d}",
                "phase": phase,
                "phase_label": phase_label,
                "launch_mode": launch_mode,
                "launch_group": launch_group,
                "family": "accuracy_policy",
                "track_key": track_key,
                "policy": "accuracy",
                "nodes": str(nodes),
                "gpus_per_node": "2",
                "cpus_per_node": "24",
                "gpu_type": "V100",
                "batch_profile": batch_profile,
                "learning_rate": learning_rate,
                "features_sub_length": features_sub_length,
                "Ntrain": ntrain,
                "global_train_batch_size": global_train_batch_size,
                "seed": str(seed),
                "strategy_id": strategy_id,
                "strategy_description": strategy_description,
                "params_file": params_file,
                "tar_path": tar_path,
                "data_prefix": data_prefix,
                "curr": "0.1",
                "data_access_mode": "direct_tar",
                "shared_data_dir": shared_data_dir,
                "save_predictions": save_predictions,
                "save_diagnostic_plots": save_diagnostic_plots,
                "save_checkpoints_epochs": save_checkpoints_epochs,
                "task_slug": f"{track_key}_{strategy_id}_n{nodes}_s{seed}",
                "notes": notes,
            }
        )

    shared_root = IMPORTANT / "optimization_track_20260411_v100_interactive" / "shared_data"
    track3_shared = str(shared_root / "track3_hh_reduced" / "reduced_data")
    track4_shared = str(shared_root / "track4_hh_full" / "concatenated_data")

    track3_accuracy = {
        974: "pytorch/configs/reruns_20260418_v100_hh_track3_g64e600_confirmation/track3_hh_reduced/params_t3_confirm_g64_e600_n1_s974.yaml",
        975: "pytorch/configs/reruns_20260418_v100_hh_track3_g64e600_confirmation/track3_hh_reduced/params_t3_confirm_g64_e600_n1_s975.yaml",
        976: "pytorch/configs/reruns_20260418_v100_hh_track3_g64e600_confirmation/track3_hh_reduced/params_t3_confirm_g64_e600_n1_s976.yaml",
        977: "pytorch/configs/reruns_20260418_v100_hh_track3_g64e600_confirmation/track3_hh_reduced/params_t3_confirm_g64_e600_n1_s977.yaml",
    }
    track4_accuracy = {
        1007: "pytorch/configs/reruns_20260419_v100_hh_track4_avgpool_confirmation/track4_hh_full/params_t4_avgpool_beta0p1_n1_s1007.yaml",
        1008: "pytorch/configs/reruns_20260419_v100_hh_track4_avgpool_confirmation/track4_hh_full/params_t4_avgpool_beta0p1_n1_s1008.yaml",
        1009: "pytorch/configs/reruns_20260419_v100_hh_track4_avgpool_confirmation/track4_hh_full/params_t4_avgpool_beta0p1_n1_s1009.yaml",
        1010: "pytorch/configs/reruns_20260419_v100_hh_track4_avgpool_confirmation/track4_hh_full/params_t4_avgpool_beta0p1_n1_s1010.yaml",
    }

    for seed in [974, 975, 976, 977]:
        add_row(
            phase="phaseA_accuracy_policy",
            phase_label="Track3 accuracy default",
            launch_mode="concurrent_4x1n",
            launch_group="track3_accuracy_default",
            track_key="track3_hh_reduced",
            nodes=1,
            seed=seed,
            strategy_id="track3_accuracy_default_g64_e600",
            strategy_description="Track3 accuracy default: g64/e600 confirmed winner.",
            params_file=track3_accuracy[seed],
            tar_path="/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            data_prefix="reduced_data",
            shared_data_dir=track3_shared,
            batch_profile="gbs64_e600",
            learning_rate="3.0e-4",
            features_sub_length="2000",
            ntrain="4096",
            global_train_batch_size="64",
            save_predictions="none",
            save_diagnostic_plots="False",
            save_checkpoints_epochs="null",
            notes="Confirmed track3 pointwise-accuracy winner.",
        )

    for seed in [1007, 1008, 1009, 1010]:
        add_row(
            phase="phaseB_accuracy_policy",
            phase_label="Track4 accuracy default",
            launch_mode="concurrent_4x1n",
            launch_group="track4_accuracy_default",
            track_key="track4_hh_full",
            nodes=1,
            seed=seed,
            strategy_id="track4_accuracy_default_beta0p1_avgpool",
            strategy_description="Track4 accuracy default: huber beta=0.1 + avg-pool confirmed winner.",
            params_file=track4_accuracy[seed],
            tar_path="/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            data_prefix="concatenated_data",
            shared_data_dir=track4_shared,
            batch_profile="gbs128_e400_beta0p1_avgpool",
            learning_rate="3.0e-4",
            features_sub_length="2000",
            ntrain="4096",
            global_train_batch_size="128",
            save_predictions="test",
            save_diagnostic_plots="True",
            save_checkpoints_epochs="10",
            notes="Confirmed track4 pointwise-accuracy winner.",
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Accuracy Policy Package ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Modes",
        "- `track3_accuracy_default_g64_e600`",
        "- `track4_accuracy_default_beta0p1`",
        "",
        "## Purpose",
        "- package the confirmed per-track pointwise-accuracy winners into named rerunnable defaults",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
