#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

DATE_TAG = "20260418_v100_hh_track4_1n_default_confirmation"
DATE_LABEL = "2026-04-18"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_1n_default_confirmation_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_1n_default_confirmation_notes_{DATE_TAG}.md"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    rows = []
    row_num = 0

    def add_row(seed: int, repeat_label: str, strategy_id: str, params_file: str) -> None:
        nonlocal row_num
        row_num += 1
        rows.append(
            {
                "row_id": f"v100t4def_{row_num:04d}",
                "phase": "phaseT7_track4_1n_default_confirmation",
                "phase_label": "Track4 1n default confirmation",
                "launch_mode": "concurrent_4x1n",
                "launch_group": f"track4_{strategy_id}_1n_confirmation",
                "family": "track4_1n_default_confirmation",
                "track_key": "track4_hh_full",
                "policy": "strong",
                "nodes": "1",
                "gpus_per_node": "2",
                "cpus_per_node": "24",
                "gpu_type": "V100",
                "batch_profile": "strong_gbs128",
                "learning_rate": "3.0e-4",
                "features_sub_length": "2000",
                "Ntrain": "8192",
                "global_train_batch_size": "128",
                "seed": str(seed),
                "strategy_id": strategy_id,
                "strategy_description": {
                    "track4_1n_s2": "Track4 1n candidate using S2_shareddata_nolosssync.",
                    "track4_1n_s4": "Track4 1n candidate using S4_leanartifacts_amp_shareddata_nolosssync.",
                }[strategy_id],
                "params_file": params_file,
                "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
                "data_prefix": "concatenated_data",
                "curr": "0.1",
                "data_access_mode": "direct_tar",
                "shared_data_dir": str(
                    IMPORTANT
                    / "optimization_track_20260411_v100_interactive"
                    / "shared_data"
                    / "track4_hh_full"
                    / "concatenated_data"
                ),
                "save_predictions": "test",
                "save_diagnostic_plots": "True" if strategy_id == "track4_1n_s2" else "False",
                "save_checkpoints_epochs": "10" if strategy_id == "track4_1n_s2" else "null",
                "task_slug": f"track4_{strategy_id}_n1_s{seed}_{repeat_label}",
                "notes": "Grouped 1n confirmation for the track4 broad-sweep default path.",
            }
        )

    s2 = {
        971: "pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S2_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s971.yaml",
        972: "pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S2_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s972.yaml",
        973: "pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S2_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s973.yaml",
    }
    s4 = {
        971: "pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s971.yaml",
        972: "pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s972.yaml",
        973: "pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/S4_leanartifacts_amp_shareddata_nolosssync/track4_hh_full/params_t4_strong_gbs128_n1_lr3p0em4_sub2000_s973.yaml",
    }

    for strategy_id, params_map in [("track4_1n_s2", s2), ("track4_1n_s4", s4)]:
        for seed, repeat_label in [(971, "a"), (972, "a"), (973, "a"), (971, "b")]:
            add_row(seed, repeat_label, strategy_id, params_map[seed])

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 1n Default Confirmation ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- grouped `1n` confirmation for the track4 broad-sweep default path",
        "- compare `S2_shareddata_nolosssync` vs `S4_leanartifacts_amp_shareddata_nolosssync`",
        "- keep the grouped `4x1n` launch mode so the result reflects the actual throughput use case",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
