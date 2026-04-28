#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260418_v100_hh_throughput_orchestration"
DATE_LABEL = "2026-04-18"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_throughput_orchestration_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_throughput_orchestration_notes_{DATE_TAG}.md"


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
                "row_id": f"v100thr_{row_num:04d}",
                "phase": phase,
                "phase_label": phase_label,
                "launch_mode": launch_mode,
                "launch_group": launch_group,
                "family": "throughput_orchestration",
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

    track_meta = {
        "track3_hh_reduced": {
            "tar_path": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            "data_prefix": "reduced_data",
            "shared_data_dir": str(
                IMPORTANT
                / "optimization_track_20260411_v100_interactive"
                / "shared_data"
                / "track3_hh_reduced"
                / "reduced_data"
            ),
            "ntrain": "4096",
            "one_n_params": {
                971: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track3_hh_reduced/params_t3_strong_gbs128_n1_lr3p0em4_sub2000_s971.yaml",
                972: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track3_hh_reduced/params_t3_strong_gbs128_n1_lr3p0em4_sub2000_s972.yaml",
                973: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track3_hh_reduced/params_t3_strong_gbs128_n1_lr3p0em4_sub2000_s973.yaml",
                974: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track3_hh_reduced/params_t3_strong_gbs128_n1_lr3p0em4_sub2000_s974.yaml",
            },
            "four_n_params": {
                971: "src/pytorch/configs/reruns_20260417_v100_hh_ddp_tuning_r5c/track3_hh_reduced/params_t3_ddpopt_gabv1_b100_n4_s971.yaml",
                972: "src/pytorch/configs/reruns_20260418_v100_hh_ddp_tuning_r5c_confirmation/track3_hh_reduced/params_t3_ddpopt_gabv1_b100_n4_s972.yaml",
            },
            "four_n_strategy": "track3_r5c_b100_4n",
            "four_n_desc": "Targeted 4n latency mode: R1_accum2_nosync with ddp_gradient_as_bucket_view=True and ddp_bucket_cap_mb=100.",
        },
        "track4_hh_full": {
            "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "data_prefix": "concatenated_data",
            "shared_data_dir": str(
                IMPORTANT
                / "optimization_track_20260411_v100_interactive"
                / "shared_data"
                / "track4_hh_full"
                / "concatenated_data"
            ),
            "ntrain": "8192",
            "one_n_params": {
                971: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s971.yaml",
                972: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s972.yaml",
                973: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s973.yaml",
                974: "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s974.yaml",
            },
            "four_n_params": {
                971: "src/pytorch/configs/reruns_20260413_v100_hh_scaling_replication_r2/track4_hh_full/params_t4_R1_accum2_nosync_n4_s971.yaml",
                972: "src/pytorch/configs/reruns_20260413_v100_hh_scaling_replication_r2/track4_hh_full/params_t4_R1_accum2_nosync_n4_s972.yaml",
            },
            "four_n_strategy": "track4_r1_accum2_4n",
            "four_n_desc": "Selected 4n latency mode: R1_accum2_nosync baseline without R5c tuning.",
        },
    }

    for track_key, meta in track_meta.items():
        group = f"{track_key}_throughput_1n"
        for seed in [971, 972, 973, 971]:
            add_row(
                phase="phaseT6_throughput_orchestration",
                phase_label="Throughput 1n production group",
                launch_mode="concurrent_4x1n",
                launch_group=group,
                track_key=track_key,
                policy="strong",
                nodes=1,
                gpus_per_node=2,
                cpus_per_node=24,
                batch_profile="strong_gbs128",
                learning_rate="3.0e-4",
                features_sub_length="2000",
                ntrain=meta["ntrain"],
                global_train_batch_size="128",
                seed=seed,
                strategy_id="throughput_1n_s4",
                strategy_description="Practical 1n throughput-first production mode using the lean S4 path.",
                params_file=meta["one_n_params"][seed],
                tar_path=meta["tar_path"],
                data_prefix=meta["data_prefix"],
                data_access_mode="direct_tar",
                shared_data_dir=meta["shared_data_dir"],
                save_predictions="none",
                save_diagnostic_plots="False",
                save_checkpoints_epochs="null",
                notes="Concurrent 4x1n production-oriented group.",
            )

        for seed in [971, 972]:
            add_row(
                phase="phaseT6_throughput_orchestration",
                phase_label="Selected 4n latency mode",
                launch_mode="dedicated",
                launch_group="",
                track_key=track_key,
                policy="strong",
                nodes=4,
                gpus_per_node=2,
                cpus_per_node=24,
                batch_profile="strong_gbs128",
                learning_rate="3.0e-4",
                features_sub_length="2000",
                ntrain=meta["ntrain"],
                global_train_batch_size="128",
                seed=seed,
                strategy_id=meta["four_n_strategy"],
                strategy_description=meta["four_n_desc"],
                params_file=meta["four_n_params"][seed],
                tar_path=meta["tar_path"],
                data_prefix=meta["data_prefix"],
                data_access_mode="direct_tar",
                shared_data_dir=meta["shared_data_dir"],
                save_predictions="none" if track_key == "track3_hh_reduced" else "test",
                save_diagnostic_plots="False" if track_key == "track3_hh_reduced" else "True",
                save_checkpoints_epochs="null" if track_key == "track3_hh_reduced" else "10",
                notes="Selected 4n latency-oriented comparison row.",
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Throughput Orchestration ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- compare concurrent `1n` production mode against selected `4n` latency mode",
        "- answer the jobs-per-hour question more directly than more DDP micro-tuning",
        "",
        "## Tracks",
        "- `track3_hh_reduced`",
        "- `track4_hh_full`",
        "",
        "## Modes",
        "- `throughput_1n_s4`: 4 concurrent `1n` rows per track (using seeds `971`, `972`, `973`, plus one repeated row to saturate the 4-node allocation)",
        "- `track3_r5c_b100_4n`: targeted tuned `4n` candidate for track3",
        "- `track4_r1_accum2_4n`: untuned selected `4n` latency mode for track4",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
