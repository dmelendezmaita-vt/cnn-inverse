#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260421_v100_hh_track4_classical_optuna_hpo"
DATE_LABEL = "2026-04-21"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_classical_optuna_hpo_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_track4_classical_optuna_hpo_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_hh_track4_classical_optuna_hpo_run_log_{DATE_TAG}.json"
NOTES_MD = NOTES / f"v100_hh_track4_classical_optuna_hpo_notes_{DATE_TAG}.md"


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
    params["data"]["Ntrain"] = 12288
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

    studies = {
        "extra_trees_mse": "Optuna/TPE search over ExtraTrees hyperparameters optimizing validation MSE.",
        "random_forest_mse": "Optuna/TPE search over RandomForest hyperparameters optimizing validation MSE.",
        "random_forest_mae": "Optuna/TPE search over RandomForest hyperparameters optimizing validation MAE.",
        "random_forest_r2": "Optuna/TPE search over RandomForest hyperparameters optimizing validation R2.",
    }

    rows = []
    row_num = 0
    seed = 1123
    group = "track4_v100_classical_optuna_hpo"
    for study_spec, desc in studies.items():
        row_num += 1
        cfg_rel = (
            f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
            f"params_{meta['short']}_{study_spec}_n1_s{seed}.yaml"
        )
        create_config(meta["base"], cfg_rel, seed=seed)
        rows.append(
            {
                "row_id": f"v100t4hpo_{row_num:04d}",
                "phase": "phaseCP1_track4_v100_classical_optuna_hpo",
                "phase_label": "Track4 V100 classical Optuna HPO",
                "launch_mode": "concurrent_4x1n",
                "launch_group": group,
                "family": "track4_v100_classical_optuna_hpo",
                "track_key": track_key,
                "policy": "strong",
                "nodes": "1",
                "cpus_per_node": "24",
                "seed": str(seed),
                "strategy_id": study_spec,
                "strategy_description": desc,
                "baseline_name": study_spec,
                "params_file": cfg_rel,
                "tar_path": meta["tar"],
                "data_prefix": meta["prefix"],
                "curr": "0.1",
                "data_access_mode": "direct_tar",
                "shared_data_dir": meta["shared"],
                "task_slug": f"{track_key}_{study_spec}_n1_s{seed}",
                "notes": "Checkpoint 1 optimizer-guided search over the strongest current classical families using Optuna/TPE.",
            }
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Classical Optuna HPO ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- finish the optimizer-guided search part of Checkpoint 1 inside the current classical frontier",
        "",
        "## Studies",
        *[f"- `{k}`" for k in studies],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
