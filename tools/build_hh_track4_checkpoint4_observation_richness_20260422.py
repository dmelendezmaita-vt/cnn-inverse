#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260422_hh_track4_checkpoint4_observation_richness"
DATE_LABEL = "2026-04-22"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"hh_track4_checkpoint4_observation_richness_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"hh_track4_checkpoint4_observation_richness_notes_{DATE_TAG}.md"


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
        "short": "t4cp4obs",
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
        "extra_trees_500_leaf3": "Balanced clean-data tree frontier representative.",
        "knn_k11": "Robustness-oriented local-method representative.",
    }
    feature_modes = {
        "raw": "Raw cropped trace only.",
        "summary12": "Hand-engineered time-domain summaries only.",
        "raw_plus_summary12": "Raw cropped trace plus hand-engineered summaries.",
        "raw_plus_fft256_summary12": "Raw cropped trace plus FFT and hand-engineered summaries.",
    }

    rows = []
    row_num = 0
    for seed in [1103]:
        for baseline_name, baseline_desc in baselines.items():
            group = f"track4_checkpoint4_obs_{baseline_name}_seed_{seed}"
            for feature_mode, feature_desc in feature_modes.items():
                row_num += 1
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{baseline_name}_{feature_mode}_n1_s{seed}.yaml"
                )
                create_config(meta["base"], cfg_rel, seed=seed)
                strategy_id = f"{baseline_name}__{feature_mode}"
                rows.append(
                    {
                        "row_id": f"cp4obs_{row_num:04d}",
                        "phase": "phaseCP4_track4_observation_richness",
                        "phase_label": "Checkpoint4 observation richness",
                        "launch_mode": "serial_1x1n",
                        "launch_group": group,
                        "family": "track4_checkpoint4_observation_richness",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "cpus_per_node": "18",
                        "seed": str(seed),
                        "strategy_id": strategy_id,
                        "strategy_description": f"{baseline_desc} Representation: {feature_desc}",
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
                            "Checkpoint 4 E2 block: compare raw-trace-only versus summary-augmented "
                            "observation representations on the fixed HH task while holding the estimator family fixed."
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
        f"# HH Track4 Checkpoint4 Observation Richness ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- start Checkpoint 4 on the non-blocked observation-richness branch",
        "- test whether richer representations improve parameter recovery more than estimator swaps do on the fixed HH task",
        "- keep the task contract fixed at `track4_hh_full`, `curr=0.1`, `Ntrain=4096`, `Nvalidate=1024`, `Ntest=1024`",
        "",
        "## Why this block now",
        "- Checkpoint 2 is already satisfied by the SBI benchmark block",
        "- Checkpoint 3 direct-fitting remains blocked by missing simulator-generation metadata in the workspace",
        "- Checkpoint 4 E2 can proceed immediately with the existing HH archive",
        "",
        "## Estimator families",
        "- `extra_trees_500_leaf3`",
        "- `knn_k11`",
        "",
        "## Observation representations",
        "- `raw`",
        "- `summary12`",
        "- `raw_plus_summary12`",
        "- `raw_plus_fft256_summary12`",
        "",
        "## Execution note",
        "- this block keeps a single canonical classical seed because the fixed sequential split and deterministic estimator settings make repeated seed rows redundant here",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
