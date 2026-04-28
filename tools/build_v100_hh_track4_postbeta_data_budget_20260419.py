#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import yaml

DATE_TAG = "20260419_v100_hh_track4_postbeta_data_budget"
DATE_LABEL = "2026-04-19"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_postbeta_data_budget_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_postbeta_data_budget_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def create_config(base_rel: str, out_rel: str, updates: Dict[str, str]) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    params = yaml.safe_load(read_text(base_path))

    data_keys = {
        "features_normalize",
        "features_scale_cache_enabled",
        "split_array_cache_enabled",
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "dataloader_pin_memory",
        "features_sub_begin_random",
        "features_sub_begin_random_eval",
        "features_sub_length",
        "global_train_batch_size",
        "train_batch_size",
        "random_seed",
        "Ntrain",
        "Nvalidate",
        "Ntest",
        "target_loss_weight_mode",
        "split_strategy",
    }
    net_keys = {
        "conv_layer_sizes",
        "dense_layer_sizes",
        "conv_layer_kernel",
        "conv_layer_stride",
        "conv_pool_type",
        "conv_pool_kernel",
        "conv_pool_stride",
        "conv_pool_padding",
        "activation_fn",
        "type",
    }
    optimizer_keys = {
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "init_learning_rate",
        "linear_epochs",
        "constant_epochs",
        "final_learning_rate",
    }
    training_keys = {
        "epochs",
        "loss_type",
        "huber_beta",
        "grad_clip_max_norm",
        "reduce_loss_for_logging",
        "use_amp",
        "amp_dtype",
    }
    runconfig_keys = {"save_predictions", "save_diagnostic_plots", "save_checkpoints_epochs", "save_dir"}

    for k, v in updates.items():
        value = yaml.safe_load(v)
        if k == "description":
            params["description"] = value
        elif k in data_keys:
            params.setdefault("data", {})[k] = value
        elif k in net_keys:
            params.setdefault("net", {})[k] = value
        elif k in optimizer_keys:
            if k in {"init_learning_rate", "linear_epochs", "constant_epochs", "final_learning_rate"}:
                params.setdefault("optimizer", {}).setdefault("learning_rate_scheduler", {})[k] = value
            else:
                params.setdefault("optimizer", {})[k] = value
        elif k in training_keys:
            params.setdefault("training", {})[k] = value
        elif k in runconfig_keys:
            params.setdefault("runconfig", {})[k] = value
        else:
            raise KeyError(f"Unhandled config key in postbeta data-budget builder: {k}")

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

    common_arch = {
        "conv_layer_sizes": "[8, 16, 32]",
        "dense_layer_sizes": "[128, 128, 128, 128]",
        "conv_layer_kernel": "3",
        "conv_layer_stride": "1",
        "conv_pool_type": '"avg"',
        "conv_pool_kernel": "2",
        "conv_pool_stride": "2",
        "conv_pool_padding": "0",
    }

    variants = {
        "avgpool_base_4k": {
            "description": "Track4 post-beta data-budget baseline: avg-pool winner at 4096/1024.",
            "Ntrain": "4096",
            "Nvalidate": "1024",
            "split_strategy": "sequential",
        },
        "avgpool_seq_8k": {
            "description": "Track4 post-beta data-budget sequential larger split: 8192/2048.",
            "Ntrain": "8192",
            "Nvalidate": "2048",
            "split_strategy": "sequential",
        },
        "avgpool_rand_8k": {
            "description": "Track4 post-beta data-budget random larger split: 8192/2048.",
            "Ntrain": "8192",
            "Nvalidate": "2048",
            "split_strategy": "random",
        },
    }

    rows = []
    row_num = 0
    for seed in [1011, 1012, 1013, 1014]:
        group = f"track4_seed{seed}_postbeta_data"
        for strategy_id, variant in variants.items():
            row_num += 1
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{strategy_id}_n1_s{seed}.yaml"
            )
            updates = {
                "description": f'"{track_key} {strategy_id} post-beta data-budget {DATE_LABEL}"',
                "learning_rate": "3.0e-4",
                "epochs": "400",
                "init_learning_rate": "1.5e-5",
                "linear_epochs": "40",
                "constant_epochs": "40",
                "final_learning_rate": "1.5e-6",
                "features_normalize": "True",
                "features_scale_cache_enabled": "True",
                "split_array_cache_enabled": "True",
                "dataloader_num_workers": "0",
                "dataloader_persistent_workers": "False",
                "dataloader_prefetch_factor": "null",
                "dataloader_pin_memory": "False",
                "features_sub_begin_random": "True",
                "features_sub_begin_random_eval": "False",
                "features_sub_length": "2000",
                "global_train_batch_size": "128",
                "train_batch_size": "128",
                "random_seed": str(seed),
                "loss_type": "huber",
                "huber_beta": "0.1",
                "grad_clip_max_norm": "1.0",
                "reduce_loss_for_logging": "False",
                "use_amp": "False",
                "amp_dtype": '"float16"',
                "save_predictions": '"test"',
                "save_diagnostic_plots": "True",
                "save_checkpoints_epochs": "10",
                "save_dir": f'"runs/{DATE_TAG}"',
                "Ntrain": variant["Ntrain"],
                "Nvalidate": variant["Nvalidate"],
                "Ntest": "1024",
                "target_loss_weight_mode": "inverse_std",
                "split_strategy": variant["split_strategy"],
                **common_arch,
            }
            create_config(meta["base"], cfg_rel, updates)
            rows.append(
                {
                    "row_id": f"v100t4pdb_{row_num:04d}",
                    "phase": "phaseX_track4_postbeta_data_budget",
                    "phase_label": "Track4 post-beta data budget",
                    "launch_mode": "concurrent_3x1n",
                    "launch_group": group,
                    "family": "track4_postbeta_data_budget",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "2",
                    "cpus_per_node": "24",
                    "gpu_type": "V100",
                    "batch_profile": f"gbs128_{variant['Ntrain']}_{variant['split_strategy']}",
                    "learning_rate": "3.0e-4",
                    "features_sub_length": "2000",
                    "Ntrain": variant["Ntrain"],
                    "global_train_batch_size": "128",
                    "seed": str(seed),
                    "strategy_id": strategy_id,
                    "strategy_description": variant["description"],
                    "params_file": cfg_rel,
                    "tar_path": meta["tar"],
                    "data_prefix": meta["prefix"],
                    "curr": "0.1",
                    "data_access_mode": "direct_tar",
                    "shared_data_dir": meta["shared"],
                    "save_predictions": "test",
                    "save_diagnostic_plots": "True",
                    "save_checkpoints_epochs": "10",
                    "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}",
                    "notes": "Retest larger train budget after the post-beta avg-pool winner.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Post-Beta Data Budget ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- retest larger train/validate budgets after fixing the track4 loss scale and structural baseline",
        "",
        "## Variants",
        "- `avgpool_base_4k`",
        "- `avgpool_seq_8k`",
        "- `avgpool_rand_8k`",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
