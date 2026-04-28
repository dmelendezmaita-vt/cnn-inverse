#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260422_v100_hh_track4_checkpoint2_sbi_family_block"
DATE_LABEL = "2026-04-22"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint2_sbi_family_block_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint2_sbi_family_block_notes_{DATE_TAG}.md"


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
        "short": "t4cp2v100",
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

    variants = [
        {
            "strategy_id": "snpe_maf_n4096",
            "method_name": "snpe",
            "model_name": "maf",
            "sample_with": "direct",
            "n_train": 4096,
            "eval_limit": 256,
            "posterior_samples": 16,
            "compute_map": "0",
            "description": "Run the strongest current SNPE family on V100 using the cached 4096-train HH split for an immediate GPU-backed comparison.",
        },
        {
            "strategy_id": "snpe_nsf_n4096",
            "method_name": "snpe",
            "model_name": "nsf",
            "sample_with": "direct",
            "n_train": 4096,
            "eval_limit": 256,
            "posterior_samples": 16,
            "compute_map": "0",
            "description": "Test whether a more flexible flow improves the cached-split SNPE branch on V100.",
        },
        {
            "strategy_id": "snle_maf_n4096",
            "method_name": "snle",
            "model_name": "maf",
            "sample_with": "mcmc",
            "n_train": 4096,
            "eval_limit": 8,
            "posterior_samples": 8,
            "compute_map": "0",
            "description": "Add a likelihood-estimation representative to the Checkpoint 2 SBI family block with a tightly bounded MCMC evaluation subset.",
        },
        {
            "strategy_id": "snre_resnet_n4096",
            "method_name": "snre",
            "model_name": "resnet",
            "sample_with": "mcmc",
            "n_train": 4096,
            "eval_limit": 8,
            "posterior_samples": 8,
            "compute_map": "0",
            "description": "Add a ratio-estimation representative to the Checkpoint 2 SBI family block with a tightly bounded MCMC evaluation subset.",
        },
    ]

    rows = []
    seed = 1201
    group = "track4_checkpoint2_sbi_family_v100"
    for idx, variant in enumerate(variants, start=1):
        cfg_rel = (
            f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
            f"params_{meta['short']}_{variant['strategy_id']}_s{seed}.yaml"
        )
        create_config(meta["base"], cfg_rel, seed=seed, n_train=int(variant["n_train"]))
        rows.append(
            {
                "row_id": f"v100cp2sbi_{idx:04d}",
                "phase": "phaseCP2_track4_v100_sbi_family_block",
                "phase_label": "Track4 Checkpoint 2 V100 SBI family block",
                "launch_mode": "concurrent_4x1n",
                "launch_group": group,
                "family": "track4_checkpoint2_v100_sbi_family_block",
                "track_key": track_key,
                "policy": "strong",
                "nodes": "1",
                "cpus_per_node": "24",
                "gpus_per_node": "1",
                "seed": str(seed),
                "strategy_id": variant["strategy_id"],
                "strategy_description": variant["description"],
                "method_name": variant["method_name"],
                "model_name": variant["model_name"],
                "sample_with": variant["sample_with"],
                "params_file": cfg_rel,
                "tar_path": meta["tar"],
                "data_prefix": meta["prefix"],
                "curr": "0.1",
                "data_access_mode": "shared_split_cache",
                "shared_data_dir": meta["shared"],
                "task_slug": f"{track_key}_{variant['strategy_id']}_s{seed}",
                "n_train": str(variant["n_train"]),
                "eval_limit": str(variant["eval_limit"]),
                "device": "cuda",
                "embedding_dim": "64",
                "embedding_hidden": "256",
                "hidden_features": "128",
                "num_transforms": "5",
                "training_batch_size": "128",
                "learning_rate": "5.0e-4",
                "stop_after_epochs": "15",
                "max_num_epochs": "80",
                "posterior_samples": str(variant["posterior_samples"]),
                "compute_map": variant["compute_map"],
                "notes": "Checkpoint 2 V100 family block: strengthen SNPE on GPU and add one SNLE/SNRE representative on the same HH task contract.",
            }
        )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Checkpoint 2 SBI Family Block ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- leverage the active Falcon V100 allocation for the first serious GPU-backed Checkpoint 2 SBI family block",
        "- promote the promising SNPE branch while also adding one NLE and one NRE representative, as required by the exhaustive plan",
        "- keep the MCMC-based NLE/NRE rows tractable by using a smaller held-out evaluation subset than the direct-sampling SNPE rows",
        "- defer MAP computation to the top posterior family follow-up so the first Falcon block stays focused on family coverage and wall-clock screening",
        "",
        "## Variants",
        *[f"- `{row['strategy_id']}`" for row in rows],
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
