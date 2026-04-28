#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260421_a30_hh_track4_tuned_tree_robustness"
DATE_LABEL = "2026-04-21"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"a30_hh_track4_tuned_tree_robustness_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"a30_hh_track4_tuned_tree_robustness_notes_{DATE_TAG}.md"


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
    params["data"]["features_scale_cache_enabled"] = False
    params["data"]["split_array_cache_enabled"] = False
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["features_sub_length"] = 2000
    params["data"]["features_additive_noise_std"] = float(noise_std)
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
        "short": "t4",
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
        "extra_trees_500": "ExtraTreesRegressor (500 trees, best clean-data MSE).",
        "extra_trees_500_leaf3": "ExtraTreesRegressor (500 trees, leaf=3, strongest balanced targeted-tree sidecar).",
        "extra_trees_500_leaf5": "ExtraTreesRegressor (500 trees, leaf=5, best clean-data R2).",
        "extra_trees_500_depth20": "ExtraTreesRegressor (500 trees, max_depth=20, best clean-data MAE).",
        "knn_k5": "k-nearest-neighbors regression (k=5, strongest robustness anchor).",
    }
    noise_levels = {
        "clean": 0.0,
        "noise001": 0.01,
        "noise002": 0.02,
        "noise005": 0.05,
    }

    rows = []
    row_num = 0
    for baseline_name, baseline_desc in baselines.items():
        for noise_tag, noise_std in noise_levels.items():
            group = f"track4_a30_tuned_tree_robustness_{baseline_name}_{noise_tag}"
            for seed in [1087, 1088, 1089, 1090]:
                row_num += 1
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{baseline_name}_{noise_tag}_n1_s{seed}.yaml"
                )
                create_config(meta["base"], cfg_rel, seed=seed, noise_std=noise_std)
                rows.append(
                    {
                        "row_id": f"a30t4crob_{row_num:04d}",
                        "phase": "phaseAP_track4_a30_tuned_tree_robustness",
                        "phase_label": "Track4 A30 tuned tree robustness",
                        "launch_mode": "concurrent_4x1n",
                        "launch_group": group,
                        "family": "track4_a30_tuned_tree_robustness",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "cpus_per_node": "64",
                        "seed": str(seed),
                        "strategy_id": f"{baseline_name}_{noise_tag}",
                        "strategy_description": f"{baseline_desc} with additive feature noise std={noise_std}.",
                        "baseline_name": baseline_name,
                        "params_file": cfg_rel,
                        "tar_path": meta["tar"],
                        "data_prefix": meta["prefix"],
                        "curr": "0.1",
                        "data_access_mode": "direct_tar",
                        "shared_data_dir": meta["shared"],
                        "task_slug": f"{track_key}_{baseline_name}_{noise_tag}_n1_s{seed}",
                        "notes": "Tuned-tree robustness sweep on the HH inverse task under additive feature noise with fresh seeds.",
                    }
                )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# A30 HH Track4 Tuned Tree Robustness ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- compare the strongest tuned-tree clean-data finalists against a kNN robustness anchor under additive-noise conditions",
        "",
        "## Baselines",
        "- `extra_trees_500`",
        "- `extra_trees_500_leaf3`",
        "- `extra_trees_500_leaf5`",
        "- `extra_trees_500_depth20`",
        "- `knn_k5`",
        "",
        "## Noise Levels",
        "- `clean`",
        "- `noise001`",
        "- `noise002`",
        "- `noise005`",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
