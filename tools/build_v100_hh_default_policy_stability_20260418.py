#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260418_v100_hh_default_policy_stability"
DATE_LABEL = "2026-04-18"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_default_policy_stability_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_default_policy_stability_notes_{DATE_TAG}.md"


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
        save_predictions: str,
        save_diagnostic_plots: str,
        save_checkpoints_epochs: str,
        notes: str,
    ) -> None:
        nonlocal row_num
        row_num += 1
        rows.append(
            {
                "row_id": f"v100pols_{row_num:04d}",
                "phase": phase,
                "phase_label": phase_label,
                "launch_mode": launch_mode,
                "launch_group": launch_group,
                "family": "default_policy_stability",
                "track_key": track_key,
                "policy": "strong",
                "nodes": str(nodes),
                "gpus_per_node": "2",
                "cpus_per_node": "24",
                "gpu_type": "V100",
                "batch_profile": "strong_gbs128",
                "learning_rate": "3.0e-4",
                "features_sub_length": "2000",
                "Ntrain": "4096" if track_key == "track3_hh_reduced" else "8192",
                "global_train_batch_size": "128",
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

    # Two grouped cycles per track for the broad defaults.
    for cycle, seeds in [("cycleA", [971, 972, 973, 971]), ("cycleB", [972, 973, 971, 972])]:
        for seed in seeds:
            add_row(
                phase="phaseP1_default_policy_stability",
                phase_label="Track3 broad default stability",
                launch_mode="concurrent_4x1n",
                launch_group=f"track3_default_1n_{cycle}",
                track_key="track3_hh_reduced",
                nodes=1,
                seed=seed,
                strategy_id="track3_default_1n_s5",
                strategy_description="Track3 broad-sweep default stability repeat.",
                params_file=track3_1n[seed],
                tar_path="/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
                data_prefix="reduced_data",
                shared_data_dir=track3_shared,
                save_predictions="none",
                save_diagnostic_plots="False",
                save_checkpoints_epochs="null",
                notes=f"Track3 default stability {cycle}.",
            )
        for seed in seeds:
            add_row(
                phase="phaseP1_default_policy_stability",
                phase_label="Track4 broad default stability",
                launch_mode="concurrent_4x1n",
                launch_group=f"track4_default_1n_{cycle}",
                track_key="track4_hh_full",
                nodes=1,
                seed=seed,
                strategy_id="track4_default_1n_s5",
                strategy_description="Track4 broad-sweep default stability repeat.",
                params_file=track4_1n[seed],
                tar_path="/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
                data_prefix="concatenated_data",
                shared_data_dir=track4_shared,
                save_predictions="none",
                save_diagnostic_plots="False",
                save_checkpoints_epochs="null",
                notes=f"Track4 default stability {cycle}.",
            )

    # Single targeted reference rows.
    for seed in [971, 972]:
        add_row(
            phase="phaseP1_default_policy_stability",
            phase_label="Track3 targeted reference",
            launch_mode="dedicated",
            launch_group="",
            track_key="track3_hh_reduced",
            nodes=4,
            seed=seed,
            strategy_id="track3_targeted_4n_b100",
            strategy_description="Track3 tuned 4n reference row.",
            params_file=track3_4n[seed],
            tar_path="/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            data_prefix="reduced_data",
            shared_data_dir=track3_shared,
            save_predictions="none",
            save_diagnostic_plots="False",
            save_checkpoints_epochs="null",
            notes="Track3 targeted reference row.",
        )
    for seed in [971, 972]:
        add_row(
            phase="phaseP1_default_policy_stability",
            phase_label="Track4 targeted reference",
            launch_mode="dedicated",
            launch_group="",
            track_key="track4_hh_full",
            nodes=4,
            seed=seed,
            strategy_id="track4_targeted_4n_accum2",
            strategy_description="Track4 4n latency reference row.",
            params_file=track4_4n[seed],
            tar_path="/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            data_prefix="concatenated_data",
            shared_data_dir=track4_shared,
            save_predictions="test",
            save_diagnostic_plots="True",
            save_checkpoints_epochs="10",
            notes="Track4 targeted reference row.",
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Default Policy Stability ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- repeat the settled broad defaults in grouped mode to quantify stability and variance",
        "- keep a small targeted 4n reference lane for context",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
