#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260422_hh_track4_checkpoint4_representation_promotion"
DATE_LABEL = "2026-04-22"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"hh_track4_checkpoint4_representation_promotion_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"hh_track4_checkpoint4_representation_promotion_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def create_config(base_rel: str, out_rel: str, *, seed: int) -> str:
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
        "short": "t4cp4prom",
        "base": "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
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
        "extra_trees_500": "Clean-data MSE anchor.",
        "extra_trees_500_leaf5": "Clean-data R2 anchor.",
        "extra_trees_500_depth20": "Clean-data MAE anchor.",
        "extra_trees_500_leaf3": "Balanced tree frontier representative.",
        "knn_k11": "Robustness-oriented local-method representative.",
        "random_forest_500": "Random-forest anchor from the classical stacking package.",
        "blend_et500_rf500_knn11_mean": "Simple classical blend.",
        "stack_et500_rf500_knn11_ridge": "Validation-time classical stacker.",
    }

    seed = 1103
    feature_mode = "raw_plus_fft256_summary12"
    group = "track4_checkpoint4_representation_promotion_seed_1103"
    rows = []
    row_num = 0
    for baseline_name, baseline_desc in baselines.items():
        row_num += 1
        cfg_rel = (
            f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
            f"params_{meta['short']}_{baseline_name}_{feature_mode}_n1_s{seed}.yaml"
        )
        create_config(meta["base"], cfg_rel, seed=seed)
        strategy_id = f"{baseline_name}__{feature_mode}"
        rows.append(
            {
                "row_id": f"cp4prom_{row_num:04d}",
                "phase": "phaseCP4_track4_representation_promotion",
                "phase_label": "Checkpoint4 representation promotion",
                "launch_mode": "serial_1x1n",
                "launch_group": group,
                "family": "track4_checkpoint4_representation_promotion",
                "track_key": track_key,
                "policy": "strong",
                "nodes": "1",
                "cpus_per_node": "18",
                "seed": str(seed),
                "strategy_id": strategy_id,
                "strategy_description": f"{baseline_desc} Promoted representation: raw + FFT(256) + summary12.",
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
                    "Checkpoint 4 promotion block: apply the best observation representation from the canary "
                    "study across the broader clean-data frontier models on the fixed HH task."
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
        f"# HH Track4 Checkpoint4 Representation Promotion ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- promote `raw_plus_fft256_summary12` across the broader clean-data frontier models",
        "- compare the best observation representation against the already-saved raw-trace baselines from the same fixed HH contract",
        "",
        "## Promoted representation",
        "- `raw_plus_fft256_summary12`",
        "",
        "## Baselines",
        *[f"- `{name}`" for name in baselines],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
