#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260418_v100_hh_default_policy"
DATE_LABEL = "2026-04-18"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_default_policy_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_default_policy_notes_{DATE_TAG}.md"


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
        policy: str,
        nodes: int,
        gpus_per_node: int,
        cpus_per_node: int,
        batch_profile: str,
        learning_rate: str,
        features_sub_length: str,
        ntrain: str,
        global_train_batch_size: str,
        seed: int,
        strategy_id: str,
        strategy_description: str,
        params_file: str,
        tar_path: str,
        data_prefix: str,
        data_access_mode: str,
        shared_data_dir: str,
        save_predictions: str,
        save_diagnostic_plots: str,
        save_checkpoints_epochs: str,
        notes: str,
    ) -> None:
        nonlocal row_num
        row_num += 1
        rows.append(
            {
                "row_id": f"v100pol_{row_num:04d}",
                "phase": phase,
                "phase_label": phase_label,
                "launch_mode": launch_mode,
                "launch_group": launch_group,
                "family": "default_policy",
                "track_key": track_key,
                "policy": policy,
                "nodes": str(nodes),
                "gpus_per_node": str(gpus_per_node),
                "cpus_per_node": str(cpus_per_node),
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
                "data_access_mode": data_access_mode,
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

    track3_1n = {
        971: "src/pytorch/configs/reruns_20260418_v100_hh_1n_factorization/track3_hh_reduced/params_t3_factor_s5_n1_s971_a.yaml",
        972: "src/pytorch/configs/reruns_20260418_v100_hh_1n_factorization/track3_hh_reduced/params_t3_factor_s5_n1_s972_a.yaml",
        973: "src/pytorch/configs/reruns_20260418_v100_hh_1n_factorization/track3_hh_reduced/params_t3_factor_s5_n1_s973_a.yaml",
    }
    track4_1n = {
        971: "src/pytorch/configs/reruns_20260418_v100_hh_1n_factorization/track4_hh_full/params_t4_factor_s5_n1_s971_a.yaml",
        972: "src/pytorch/configs/reruns_20260418_v100_hh_1n_factorization/track4_hh_full/params_t4_factor_s5_n1_s972_a.yaml",
        973: "src/pytorch/configs/reruns_20260418_v100_hh_1n_factorization/track4_hh_full/params_t4_factor_s5_n1_s973_a.yaml",
    }
    track3_4n = {
        971: "src/pytorch/configs/reruns_20260417_v100_hh_ddp_tuning_r5c/track3_hh_reduced/params_t3_ddpopt_gabv1_b100_n4_s971.yaml",
        972: "src/pytorch/configs/reruns_20260418_v100_hh_ddp_tuning_r5c_confirmation/track3_hh_reduced/params_t3_ddpopt_gabv1_b100_n4_s972.yaml",
    }
    track4_4n = {
        971: "src/pytorch/configs/reruns_20260413_v100_hh_scaling_replication_r2/track4_hh_full/params_t4_R1_accum2_nosync_n4_s971.yaml",
        972: "src/pytorch/configs/reruns_20260413_v100_hh_scaling_replication_r2/track4_hh_full/params_t4_R1_accum2_nosync_n4_s972.yaml",
    }

    for seed, repeat_label in [(971, "a"), (972, "a"), (973, "a"), (971, "b")]:
        add_row(
            phase="phaseP0_default_policy",
            phase_label="Track3 broad-sweep default",
            launch_mode="concurrent_4x1n",
            launch_group="track3_default_1n",
            track_key="track3_hh_reduced",
            policy="strong",
            nodes=1,
            gpus_per_node=2,
            cpus_per_node=24,
            batch_profile="strong_gbs128",
            learning_rate="3.0e-4",
            features_sub_length="2000",
            ntrain="4096",
            global_train_batch_size="128",
            seed=seed,
            strategy_id="track3_default_1n_s5",
            strategy_description="Track3 broad-sweep default: 1n factor_s5 no-AMP lean-artifact path.",
            params_file=track3_1n[seed],
            tar_path="/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            data_prefix="reduced_data",
            data_access_mode="direct_tar",
            shared_data_dir=track3_shared,
            save_predictions="none",
            save_diagnostic_plots="False",
            save_checkpoints_epochs="null",
            notes=f"Concurrent track3 broad-sweep default ({repeat_label}).",
        )

    for seed, repeat_label in [(971, "a"), (972, "a"), (973, "a"), (971, "b")]:
        add_row(
            phase="phaseP0_default_policy",
            phase_label="Track4 broad-sweep default",
            launch_mode="concurrent_4x1n",
            launch_group="track4_default_1n",
            track_key="track4_hh_full",
            policy="strong",
            nodes=1,
            gpus_per_node=2,
            cpus_per_node=24,
            batch_profile="strong_gbs128",
            learning_rate="3.0e-4",
            features_sub_length="2000",
            ntrain="8192",
            global_train_batch_size="128",
            seed=seed,
            strategy_id="track4_default_1n_s5",
            strategy_description="Track4 broad-sweep default: 1n factor_s5 no-AMP lean-artifact path.",
            params_file=track4_1n[seed],
            tar_path="/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            data_prefix="concatenated_data",
            data_access_mode="direct_tar",
            shared_data_dir=track4_shared,
            save_predictions="test",
            save_diagnostic_plots="True",
            save_checkpoints_epochs="10",
            notes=f"Concurrent track4 broad-sweep default ({repeat_label}).",
        )

    for seed in [971, 972]:
        add_row(
            phase="phaseP0_default_policy",
            phase_label="Track3 targeted multi-node candidate",
            launch_mode="dedicated",
            launch_group="",
            track_key="track3_hh_reduced",
            policy="strong",
            nodes=4,
            gpus_per_node=2,
            cpus_per_node=24,
            batch_profile="strong_gbs128",
            learning_rate="3.0e-4",
            features_sub_length="2000",
            ntrain="4096",
            global_train_batch_size="128",
            seed=seed,
            strategy_id="track3_targeted_4n_b100",
            strategy_description="Track3 targeted multi-node candidate: tuned 4n b100 path.",
            params_file=track3_4n[seed],
            tar_path="/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            data_prefix="reduced_data",
            data_access_mode="direct_tar",
            shared_data_dir=track3_shared,
            save_predictions="none",
            save_diagnostic_plots="False",
            save_checkpoints_epochs="null",
            notes="Targeted track3 multi-node candidate.",
        )

    for seed in [971, 972]:
        add_row(
            phase="phaseP0_default_policy",
            phase_label="Track4 targeted latency mode",
            launch_mode="dedicated",
            launch_group="",
            track_key="track4_hh_full",
            policy="strong",
            nodes=4,
            gpus_per_node=2,
            cpus_per_node=24,
            batch_profile="strong_gbs128",
            learning_rate="3.0e-4",
            features_sub_length="2000",
            ntrain="8192",
            global_train_batch_size="128",
            seed=seed,
            strategy_id="track4_targeted_4n_accum2",
            strategy_description="Track4 targeted latency mode: untuned 4n accum2 path.",
            params_file=track4_4n[seed],
            tar_path="/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            data_prefix="concatenated_data",
            data_access_mode="direct_tar",
            shared_data_dir=track4_shared,
            save_predictions="test",
            save_diagnostic_plots="True",
            save_checkpoints_epochs="10",
            notes="Targeted track4 latency mode.",
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Default Policy Package ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Modes",
        "- `track3_default_1n_s5`",
        "- `track4_default_1n_s5`",
        "- `track3_targeted_4n_b100`",
        "- `track4_targeted_4n_accum2`",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
