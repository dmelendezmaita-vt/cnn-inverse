#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import yaml

DATE_TAG = "20260422_v100_hh_track4_neural_noise_clean_eval"
DATE_LABEL = "2026-04-22"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_neural_noise_clean_eval_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_neural_noise_clean_eval_notes_{DATE_TAG}.md"

RETRAIN_ROOT = IMPORTANT / "optimization_track_20260422_v100_hh_track4_neural_noise_retrain"
RETRAIN_RUNS = RETRAIN_ROOT / "runs" / "phaseCP4_track4_v100_neural_noise_retrain" / "concurrent_4x1n"


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
            raise KeyError(f"Unhandled config key in V100 neural clean eval builder: {k}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return out_rel


def final_checkpoint(run_root: Path) -> str:
    subdirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No materialized run dir under {run_root}")
    ckpts = sorted((subdirs[0] / "checkpoints").glob("*/*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {subdirs[0]}")
    return str(ckpts[-1])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_key = "track4_hh_full"
    meta = {
        "short": "t4v100nce",
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

    rows = []
    packed_group = "track4_v100_neural_noise_clean_eval_packed"
    row_num = 0
    for strategy_id in ["effnet_beta005_invvar_e500_noise001", "effnet_beta003_invvar_e400_noise001"]:
        for seed in [1071, 1072, 1073, 1074]:
            row_num += 1
            cfg_rel = (
                f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{strategy_id}_clean_n1_s{seed}.yaml"
            )
            updates = {
                "description": f'"{track_key} {strategy_id} clean eval after noise retrain {DATE_LABEL}"',
                "type": '"EfficientNet"',
                "dropout": "0.1",
                "learning_rate": "1.5e-4",
                "epochs": "500",
                "init_learning_rate": "7.5e-6",
                "linear_epochs": "75",
                "constant_epochs": "75",
                "final_learning_rate": "7.5e-7",
                "features_normalize": "True",
                "features_scale_cache_enabled": "True",
                "split_array_cache_enabled": "True",
                "dataloader_num_workers": "0",
                "dataloader_persistent_workers": "False",
                "dataloader_prefetch_factor": "null",
                "dataloader_pin_memory": "False",
                "features_sub_begin_random": "False",
                "features_sub_begin_random_eval": "False",
                "features_sub_length": "2000",
                "features_additive_noise_std": "0.0",
                "global_train_batch_size": "128",
                "train_batch_size": "128",
                "random_seed": str(seed),
                "loss_type": "huber",
                "huber_beta": "0.05" if "beta005" in strategy_id else "0.03",
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
                "target_loss_weight_mode": "inverse_var",
            }
            create_config(meta["base"], cfg_rel, updates)
            run_root = RETRAIN_RUNS / f"v100t4nr_{row_num:04d}"
            rows.append(
                {
                    "row_id": f"v100t4nce_{row_num:04d}",
                    "phase": "phaseCP4_track4_v100_neural_noise_clean_eval",
                    "phase_label": "Track4 V100 neural noise clean eval",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": packed_group,
                    "family": "track4_v100_neural_noise_clean_eval",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "1",
                    "cpus_per_node": "12",
                    "gpu_type": "V100",
                    "batch_profile": "eval_only_gpu",
                    "learning_rate": "1.5e-4",
                    "features_sub_length": "2000",
                    "features_additive_noise_std": "0.0",
                    "Ntrain": "4096",
                    "global_train_batch_size": "128",
                    "seed": str(seed),
                    "strategy_id": f"{strategy_id}_clean_eval",
                    "strategy_description": "Clean evaluation of the completed noise-trained checkpoint.",
                    "params_file": cfg_rel,
                    "tar_path": meta["tar"],
                    "data_prefix": meta["prefix"],
                    "curr": "0.1",
                    "data_access_mode": "direct_tar",
                    "shared_data_dir": meta["shared"],
                    "save_predictions": "test",
                    "save_diagnostic_plots": "True",
                    "save_checkpoints_epochs": "10",
                    "eval_only_use_gpu": "1",
                    "eval_only_checkpoint": final_checkpoint(run_root),
                    "source_run_root": str(run_root),
                    "task_slug": f"{track_key}_{strategy_id}_clean_eval_n1_s{seed}",
                    "notes": "Falcon V100 packed clean eval of the completed noise-trained checkpoint.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Neural Noise Clean Eval ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- evaluate the completed noise-trained EfficientNet checkpoints on clean data",
        "- answer whether low-noise retraining retains clean-data quality rather than only noisy-condition quality",
        "- pack 8 single-GPU eval rows across the 4-node Falcon V100 allocation",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
