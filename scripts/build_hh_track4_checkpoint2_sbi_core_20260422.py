#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260422_hh_track4_checkpoint2_sbi_core"
DATE_LABEL = "2026-04-22"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"hh_track4_checkpoint2_sbi_core_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"hh_track4_checkpoint2_sbi_core_notes_{DATE_TAG}.md"


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
    params["data"]["features_additive_noise_std"] = 0.0
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
        "short": "t4cp2",
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

    variants = [
        {
            "strategy_id": "snpe_maf_n4096",
            "strategy_description": "SNPE with a MAF posterior head and a small learned embedding over the same Checkpoint 1 HH task contract.",
            "density_estimator": "maf",
            "n_train": "4096",
        },
        {
            "strategy_id": "snpe_nsf_n4096",
            "strategy_description": "SNPE with an NSF posterior head to test whether a more flexible normalizing flow changes the first SBI verdict.",
            "density_estimator": "nsf",
            "n_train": "4096",
        },
    ]

    rows = []
    seed = 1201
    group = "track4_checkpoint2_sbi_core"
    for idx, variant in enumerate(variants, start=1):
        cfg_rel = (
            f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
            f"params_{meta['short']}_{variant['strategy_id']}_s{seed}.yaml"
        )
        create_config(meta["base"], cfg_rel, seed=seed)
        rows.append(
            {
                "row_id": f"cp2sbi_{idx:04d}",
                "phase": "phaseCP2_track4_sbi_core",
                "phase_label": "Track4 Checkpoint 2 SBI core",
                "launch_mode": "dedicated_1x1n",
                "launch_group": group,
                "family": "track4_checkpoint2_sbi_core",
                "track_key": track_key,
                "policy": "strong",
                "nodes": "1",
                "cpus_per_node": "18",
                "seed": str(seed),
                "strategy_id": variant["strategy_id"],
                "strategy_description": variant["strategy_description"],
                "method_name": "snpe",
                "density_estimator": variant["density_estimator"],
                "params_file": cfg_rel,
                "tar_path": meta["tar"],
                "data_prefix": meta["prefix"],
                "curr": "0.1",
                "data_access_mode": "shared_split_cache",
                "shared_data_dir": meta["shared"],
                "task_slug": f"{track_key}_{variant['strategy_id']}_s{seed}",
                "n_train": variant["n_train"],
                "eval_limit": "256",
                "device": "cpu",
                "embedding_dim": "64",
                "embedding_hidden": "256",
                "hidden_features": "128",
                "num_transforms": "5",
                "training_batch_size": "128",
                "learning_rate": "5.0e-4",
                "stop_after_epochs": "15",
                "max_num_epochs": "80",
                "posterior_samples": "16",
                "notes": "Checkpoint 2 begins with an SBI-first block on the frozen Checkpoint 1 HH task contract, using a 256-observation held-out subset for the first CPU-only decision pass.",
            }
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# HH Track4 Checkpoint 2 SBI Core ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- start Checkpoint 2 with a first serious posterior-inference benchmark before deciding whether the SBI family merits broader expansion",
        "- keep the HH task contract fixed to the same normalized 2000-step single-protocol setup used in Checkpoint 1",
        "- keep the first CPU-only pass tractable by using 256 held-out observations and 16 posterior draws per observation before any broader rerun on GPU",
        "",
        "## Initial Variants",
        *[f"- `{row['strategy_id']}`" for row in rows],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
