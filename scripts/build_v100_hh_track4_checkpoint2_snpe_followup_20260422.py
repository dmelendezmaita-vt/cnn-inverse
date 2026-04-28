#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260422_v100_hh_track4_checkpoint2_snpe_followup"
DATE_LABEL = "2026-04-22"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_snpe_followup_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_snpe_followup_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def create_config(base_rel: str, out_rel: str, *, seed: int, n_train: int) -> str:
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
    params["data"]["Ntrain"] = n_train
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
        "short": "t4cp2snpe",
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
        ("snpe_nsf_h128_t5", "snpe", "nsf", 128, 5),
        ("snpe_nsf_h128_t8", "snpe", "nsf", 128, 8),
        ("snpe_nsf_h192_t5", "snpe", "nsf", 192, 5),
        ("snpe_nsf_h192_t8", "snpe", "nsf", 192, 8),
        ("snpe_maf_h128_t5", "snpe", "maf", 128, 5),
        ("snpe_maf_h128_t8", "snpe", "maf", 128, 8),
        ("snpe_maf_h192_t5", "snpe", "maf", 192, 5),
        ("snpe_maf_h192_t8", "snpe", "maf", 192, 8),
    ]

    rows = []
    seed = 1201
    group = "track4_checkpoint2_snpe_followup_v100"
    for idx, (strategy_id, method_name, model_name, hidden_features, num_transforms) in enumerate(variants, start=1):
        cfg_rel = (
            f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
            f"params_{meta['short']}_{strategy_id}_s{seed}.yaml"
        )
        create_config(meta["base"], cfg_rel, seed=seed, n_train=4096)
        rows.append(
            {
                "row_id": f"v100cp2snpe_{idx:04d}",
                "phase": "phaseCP2_track4_v100_snpe_followup",
                "phase_label": "Track4 Checkpoint 2 V100 SNPE follow-up",
                "launch_mode": "concurrent_8x1n",
                "launch_group": group,
                "family": "track4_checkpoint2_v100_snpe_followup",
                "track_key": track_key,
                "policy": "strong",
                "nodes": "1",
                "cpus_per_node": "12",
                "gpus_per_node": "1",
                "seed": str(seed),
                "strategy_id": strategy_id,
                "strategy_description": f"{model_name.upper()} SNPE follow-up with hidden_features={hidden_features}, num_transforms={num_transforms}.",
                "method_name": method_name,
                "model_name": model_name,
                "sample_with": "direct",
                "params_file": cfg_rel,
                "tar_path": meta["tar"],
                "data_prefix": meta["prefix"],
                "curr": "0.1",
                "data_access_mode": "shared_split_cache",
                "shared_data_dir": meta["shared"],
                "task_slug": f"{track_key}_{strategy_id}_s{seed}",
                "n_train": "4096",
                "eval_limit": "256",
                "device": "cuda",
                "embedding_dim": "64",
                "embedding_hidden": "256",
                "hidden_features": str(hidden_features),
                "num_transforms": str(num_transforms),
                "training_batch_size": "128",
                "learning_rate": "5.0e-4",
                "stop_after_epochs": "20",
                "max_num_epochs": "100",
                "posterior_samples": "16",
                "compute_map": "0",
                "notes": "Checkpoint 2 SNPE-only follow-up packed at two 1-GPU rows per V100 node to use the full Falcon allocation efficiently.",
            }
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SNPE Follow-up ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- fix the earlier under-utilization by packing two independent 1-GPU SNPE rows per V100 node",
        "- focus the next Checkpoint 2 expansion on the only SBI branch that looked promising enough to merit more GPU time",
        "",
        "## Variants",
        *[f"- `{row['strategy_id']}`" for row in rows],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
