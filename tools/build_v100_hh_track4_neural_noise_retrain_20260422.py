#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import yaml

DATE_TAG = "20260422_v100_hh_track4_neural_noise_retrain"
DATE_LABEL = "2026-04-22"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_neural_noise_retrain_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_neural_noise_retrain_notes_{DATE_TAG}.md"


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
        "features_additive_noise_std",
        "global_train_batch_size",
        "train_batch_size",
        "random_seed",
        "Ntrain",
        "Nvalidate",
        "Ntest",
        "target_loss_weight_mode",
    }
    net_keys = {"type", "dropout"}
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
            raise KeyError(f"Unhandled config key in V100 neural noise retrain builder: {k}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return out_rel


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_key = "track4_hh_full"
    meta = {
        "short": "t4v100ntrain",
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

    variants = {
        "effnet_beta005_invvar_e500_noise001": {
            "description": "Winner retrained with low additive observation noise.",
            "type": '"EfficientNet"',
            "dropout": "0.1",
            "target_loss_weight_mode": "inverse_var",
            "loss_type": "huber",
            "huber_beta": "0.05",
            "learning_rate": "1.5e-4",
            "init_learning_rate": "7.5e-6",
            "epochs": "500",
            "linear_epochs": "75",
            "constant_epochs": "75",
            "final_learning_rate": "7.5e-7",
            "features_additive_noise_std": "0.01",
        },
        "effnet_beta003_invvar_e400_noise001": {
            "description": "Nearest confirmed neural neighbor retrained with low additive observation noise.",
            "type": '"EfficientNet"',
            "dropout": "0.1",
            "target_loss_weight_mode": "inverse_var",
            "loss_type": "huber",
            "huber_beta": "0.03",
            "learning_rate": "1.5e-4",
            "init_learning_rate": "7.5e-6",
            "epochs": "400",
            "linear_epochs": "60",
            "constant_epochs": "60",
            "final_learning_rate": "7.5e-7",
            "features_additive_noise_std": "0.01",
        },
    }

    rows = []
    row_num = 0
    for strategy_id, variant in variants.items():
        group = f"track4_v100_neural_noise_retrain_{strategy_id}"
        for seed in [1071, 1072, 1073, 1074]:
            row_num += 1
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{strategy_id}_n1_s{seed}.yaml"
            )
            updates = {
                "description": f'"{track_key} {strategy_id} v100 neural noise retrain {DATE_LABEL}"',
                "type": variant["type"],
                "dropout": variant["dropout"],
                "learning_rate": variant["learning_rate"],
                "epochs": variant["epochs"],
                "init_learning_rate": variant["init_learning_rate"],
                "linear_epochs": variant["linear_epochs"],
                "constant_epochs": variant["constant_epochs"],
                "final_learning_rate": variant["final_learning_rate"],
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
                "features_additive_noise_std": variant["features_additive_noise_std"],
                "global_train_batch_size": "128",
                "train_batch_size": "128",
                "random_seed": str(seed),
                "loss_type": variant["loss_type"],
                "huber_beta": variant["huber_beta"],
                "grad_clip_max_norm": "1.0",
                "reduce_loss_for_logging": "False",
                "use_amp": "False",
                "amp_dtype": '"float16"',
                "save_predictions": '"test"',
                "save_diagnostic_plots": "True",
                "save_checkpoints_epochs": "10",
                "save_dir": f'"runs/{DATE_TAG}"',
                "Ntrain": "4096",
                "Nvalidate": "1024",
                "Ntest": "1024",
                "target_loss_weight_mode": variant["target_loss_weight_mode"],
            }
            create_config(meta["base"], cfg_rel, updates)
            rows.append(
                {
                    "row_id": f"v100t4nr_{row_num:04d}",
                    "phase": "phaseCP4_track4_v100_neural_noise_retrain",
                    "phase_label": "Track4 V100 neural noise retrain",
                    "launch_mode": "concurrent_4x1n",
                    "launch_group": group,
                    "family": "track4_v100_neural_noise_retrain",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "2",
                    "cpus_per_node": "24",
                    "gpu_type": "V100",
                    "batch_profile": "train_eval",
                    "learning_rate": variant["learning_rate"],
                    "features_sub_length": "2000",
                    "features_additive_noise_std": variant["features_additive_noise_std"],
                    "Ntrain": "4096",
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
                    "eval_only_checkpoint": "",
                    "source_run_root": "",
                    "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}",
                    "notes": "Falcon V100 neural retraining probe with low observation noise around the locked EfficientNet winner neighborhood.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Neural Noise Retrain ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- test whether the locked neural winner can retain useful HH recovery when trained with low additive observation noise",
        "- compare the winner against its nearest confirmed neighbor on the same fixed Track 4 HH contract",
        "",
        "## Variants",
        "- `effnet_beta005_invvar_e500_noise001`",
        "- `effnet_beta003_invvar_e400_noise001`",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
