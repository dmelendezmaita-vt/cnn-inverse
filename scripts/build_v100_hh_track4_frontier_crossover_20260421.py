#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260421_v100_hh_track4_frontier_crossover"
DATE_LABEL = "2026-04-21"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_frontier_crossover_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_frontier_crossover_notes_{DATE_TAG}.md"


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
        "extra_trees_500": "ExtraTreesRegressor (500 trees, best clean-data MSE).",
        "extra_trees_500_leaf5": "ExtraTreesRegressor (500 trees, best clean-data R2).",
        "extra_trees_500_depth20": "ExtraTreesRegressor (500 trees, best clean-data MAE).",
        "knn_k11": "k-nearest-neighbors regression (k=11, strongest tuned robustness candidate).",
    }
    noise_levels = {
        "clean": 0.0,
        "noise001": 0.01,
        "noise002": 0.02,
        "noise005": 0.05,
        "noise010": 0.10,
        "noise020": 0.20,
    }

    rows = []
    row_num = 0
    for seed in [1103, 1104, 1105, 1106]:
        for noise_tag, noise_std in noise_levels.items():
            group = f"track4_v100_frontier_crossover_seed_{seed}_{noise_tag}"
            for baseline_name, baseline_desc in baselines.items():
                row_num += 1
                cfg_rel = (
                    f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{baseline_name}_{noise_tag}_n1_s{seed}.yaml"
                )
                create_config(meta["base"], cfg_rel, seed=seed, noise_std=noise_std)
                rows.append(
                    {
                        "row_id": f"v100t4xov_{row_num:04d}",
                        "phase": "phaseAP_track4_v100_frontier_crossover",
                        "phase_label": "Track4 V100 frontier crossover",
                        "launch_mode": "concurrent_4x1n",
                        "launch_group": group,
                        "family": "track4_v100_frontier_crossover",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "cpus_per_node": "24",
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
                        "notes": "Direct crossover comparison between the clean-data tree winners and tuned knn_k11 across a finer noise ladder.",
                    }
                )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Frontier Crossover ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- locate the practical crossover point where the clean-data tree winners give way to the tuned kNN under increasing additive noise",
        "",
        "## Finalists",
        *[f"- `{k}`" for k in baselines],
        "",
        "## Noise Levels",
        *[f"- `{k}`" for k in noise_levels],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
