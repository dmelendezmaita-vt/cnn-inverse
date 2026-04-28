#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260421_v100_hh_track4_classical_stacking"
DATE_LABEL = "2026-04-21"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_classical_stacking_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_track4_classical_stacking_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_hh_track4_classical_stacking_run_log_{DATE_TAG}.json"
NOTES_MD = NOTES / f"v100_hh_track4_classical_stacking_notes_{DATE_TAG}.md"


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
    params["data"]["features_scale_cache_enabled"] = False
    params["data"]["split_array_cache_enabled"] = False
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

    blocks = [
        (
            "a",
            [
                ("extra_trees_500", "Current clean-data ExtraTrees anchor."),
                ("random_forest_500", "Current random-forest anchor."),
                ("knn_k11", "Current robustness-oriented kNN anchor."),
            ],
        ),
        (
            "b",
            [
                ("blend_et500_rf500_knn11_mean", "Simple mean blend of the three strongest current classical families."),
                ("stack_et500_rf500_knn11_ridge", "Validation-time ridge stacker over the three strongest current classical families."),
            ],
        ),
    ]

    rows = []
    row_num = 0
    for seed in [1115, 1116, 1117, 1118]:
        for block_name, baselines in blocks:
            group = f"track4_v100_stacking_seed_{seed}_block_{block_name}"
            for baseline_name, baseline_desc in baselines:
                row_num += 1
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{baseline_name}_n1_s{seed}.yaml"
                )
                create_config(meta["base"], cfg_rel, seed=seed)
                rows.append(
                    {
                        "row_id": f"v100t4stk_{row_num:04d}",
                        "phase": "phaseCP1_track4_v100_classical_stacking",
                        "phase_label": "Track4 V100 classical stacking",
                        "launch_mode": "concurrent_4x1n",
                        "launch_group": group,
                        "family": "track4_v100_classical_stacking",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "cpus_per_node": "24",
                        "seed": str(seed),
                        "strategy_id": baseline_name,
                        "strategy_description": baseline_desc,
                        "baseline_name": baseline_name,
                        "params_file": cfg_rel,
                        "tar_path": meta["tar"],
                        "data_prefix": meta["prefix"],
                        "curr": "0.1",
                        "data_access_mode": "direct_tar",
                        "shared_data_dir": meta["shared"],
                        "task_slug": f"{track_key}_{baseline_name}_n1_s{seed}",
                        "notes": "Checkpoint 1 ensemble/stacking study using the strongest current classical families.",
                    }
                )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Classical Stacking ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- test whether simple averaging or validation-time ridge stacking beats the strongest current single classical families",
        "",
        "## Baselines",
        *[f"- `{name}`" for _block, baselines in blocks for name, _desc in baselines],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
