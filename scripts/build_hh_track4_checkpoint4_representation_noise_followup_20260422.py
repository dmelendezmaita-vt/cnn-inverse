#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260422_hh_track4_checkpoint4_representation_noise_followup"
DATE_LABEL = "2026-04-22"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"hh_track4_checkpoint4_representation_noise_followup_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"hh_track4_checkpoint4_representation_noise_followup_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def create_config(base_rel: str, out_rel: str, *, seed: int, noise_std: float) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    params = yaml.safe_load(read_text(base_path))

    params.setdefault("data", {})
    params["data"]["features_normalize"] = True
    params["data"]["features_scale_cache_enabled"] = True
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["features_sub_length"] = 2000
    params["data"]["features_additive_noise_std"] = noise_std
    params["data"]["random_seed"] = seed
    params["data"]["Ntrain"] = 4096
    params["data"]["Nvalidate"] = 1024
    params["data"]["Ntest"] = 1024
    params.setdefault("runconfig", {})
    params["runconfig"]["save_predictions"] = "test"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return out_rel


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_key = "track4_hh_full"
    meta = {
        "short": "t4cp4noise",
        "base": "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
        "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
        "prefix": "concatenated_data",
        "shared": str(
            IMPORTANT
            / "optimization_track_20260411_v100_interactive"
            / "shared_data"
            / "track4_hh_full"
            / "concatenated_data"
        ),
    }

    baselines = {
        "extra_trees_500": "Promoted clean-data MSE anchor.",
        "random_forest_500": "Promoted clean-data MAE/R2 anchor.",
        "knn_k11": "Promoted robust local-method anchor.",
    }
    noise_levels = [
        ("clean", 0.0),
        ("noise005", 0.005),
        ("noise010", 0.01),
        ("noise020", 0.02),
    ]

    seed = 1103
    feature_mode = "raw_plus_fft256_summary12"
    rows = []
    row_num = 0
    for baseline_name, baseline_desc in baselines.items():
        group = f"track4_checkpoint4_repr_noise_{baseline_name}_seed_1103"
        for noise_label, noise_std in noise_levels:
            row_num += 1
            cfg_rel = (
                f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{baseline_name}_{noise_label}_n1_s{seed}.yaml"
            )
            create_config(meta["base"], cfg_rel, seed=seed, noise_std=noise_std)
            strategy_id = f"{baseline_name}__{noise_label}"
            rows.append(
                {
                    "row_id": f"cp4noise_{row_num:04d}",
                    "phase": "phaseCP4_track4_representation_noise_followup",
                    "phase_label": "Checkpoint4 representation noise follow-up",
                    "launch_mode": "serial_1x1n",
                    "launch_group": group,
                    "family": "track4_checkpoint4_representation_noise_followup",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "18",
                    "seed": str(seed),
                    "strategy_id": strategy_id,
                    "strategy_description": (
                        f"{baseline_desc} Promoted representation under additive feature noise std={noise_std}."
                    ),
                    "baseline_name": baseline_name,
                    "feature_mode": feature_mode,
                    "params_file": cfg_rel,
                    "tar_path": meta["tar"],
                    "data_prefix": meta["prefix"],
                    "curr": "0.1",
                    "data_access_mode": "direct_tar",
                    "shared_data_dir": meta["shared"],
                    "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}",
                    "notes": (
                        "Checkpoint 4 / robustness bridge block: test whether the promoted observation representation "
                        "changes the practical ranking under additive feature noise."
                    ),
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# HH Track4 Checkpoint4 Representation Noise Follow-Up ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- test whether the promoted `raw_plus_fft256_summary12` representation changes the practical noisy-data ranking",
        "- focus only on the promoted top models to keep the block targeted",
        "",
        "## Baselines",
        *[f"- `{name}`" for name in baselines],
        "",
        "## Noise levels",
        *[f"- `{label}` = `{std}`" for label, std in noise_levels],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
