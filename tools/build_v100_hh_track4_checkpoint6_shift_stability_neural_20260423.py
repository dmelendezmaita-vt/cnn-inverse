#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import yaml

DATE_TAG = "20260423_v100_hh_track4_checkpoint6_shift_stability_neural"
DATE_LABEL = "2026-04-23"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint6_shift_stability_neural_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint6_shift_stability_neural_notes_{DATE_TAG}.md"

TOP3_ROOT = IMPORTANT / "optimization_track_20260420_a30_hh_track4_top3_confirmation"
TOP3_RUNS = TOP3_ROOT / "runs" / "phaseAP_track4_a30_top3_confirmation" / "concurrent_3x1n"
WINNER_STRATEGY = "effnet_beta005_invvar_e500"


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
        "features_sub_step",
        "features_additive_noise_std",
        "features_multiplicative_noise_std",
        "features_baseline_drift_std",
        "features_mask_fraction",
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
            raise KeyError(f"Unhandled config key in checkpoint6 neural shift builder: {k}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return out_rel


def find_winner_runs() -> dict[int, dict[str, str]]:
    matches: dict[int, dict[str, str]] = {}
    for run_root in sorted(TOP3_RUNS.glob("a30t4top_*/*")):
        params_path = run_root / "params.yaml"
        if not params_path.exists():
            continue
        params = yaml.safe_load(params_path.read_text())
        if WINNER_STRATEGY not in params_path.read_text():
            continue
        seed = int(params["data"]["random_seed"])
        ckpts = sorted((run_root / "checkpoints").glob("*/*.pt"))
        if not ckpts:
            raise RuntimeError(f"No checkpoints found under {run_root}")
        matches[seed] = {
            "run_root": str(run_root),
            "checkpoint": str(ckpts[-1]),
            "params_file": params_path.as_posix(),
        }
    if sorted(matches) != [1071, 1072, 1073, 1074]:
        raise RuntimeError(f"Expected winner checkpoints for seeds 1071-1074, found {sorted(matches)}")
    return matches


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    winner_runs = find_winner_runs()
    track_key = "track4_hh_full"
    meta = {
        "short": "t4v100cp6",
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

    shifts = [
        (
            "clean",
            {
                "features_additive_noise_std": "0.0",
                "features_multiplicative_noise_std": "0.0",
                "features_baseline_drift_std": "0.0",
                "features_mask_fraction": "0.0",
                "features_sub_length": "2000",
                "features_sub_step": "1",
            },
            "Clean reference evaluation.",
            "block1",
        ),
        (
            "mult010",
            {
                "features_additive_noise_std": "0.0",
                "features_multiplicative_noise_std": "0.10",
                "features_baseline_drift_std": "0.0",
                "features_mask_fraction": "0.0",
                "features_sub_length": "2000",
                "features_sub_step": "1",
            },
            "Elementwise multiplicative noise in normalized feature space.",
            "block1",
        ),
        (
            "drift010",
            {
                "features_additive_noise_std": "0.0",
                "features_multiplicative_noise_std": "0.0",
                "features_baseline_drift_std": "0.10",
                "features_mask_fraction": "0.0",
                "features_sub_length": "2000",
                "features_sub_step": "1",
            },
            "Linear baseline drift.",
            "block2",
        ),
        (
            "trunc1000",
            {
                "features_additive_noise_std": "0.0",
                "features_multiplicative_noise_std": "0.0",
                "features_baseline_drift_std": "0.0",
                "features_mask_fraction": "0.0",
                "features_sub_length": "1000",
                "features_sub_step": "1",
            },
            "Time truncation to the first 1000 observation points.",
            "block2",
        ),
        (
            "downstep2",
            {
                "features_additive_noise_std": "0.0",
                "features_multiplicative_noise_std": "0.0",
                "features_baseline_drift_std": "0.0",
                "features_mask_fraction": "0.0",
                "features_sub_length": "2000",
                "features_sub_step": "2",
            },
            "Lower sampling rate via stride-2 temporal subsampling.",
            "block3",
        ),
        (
            "mask20",
            {
                "features_additive_noise_std": "0.0",
                "features_multiplicative_noise_std": "0.0",
                "features_baseline_drift_std": "0.0",
                "features_mask_fraction": "0.20",
                "features_sub_length": "2000",
                "features_sub_step": "1",
            },
            "Partial observability with one contiguous 20% segment masked.",
            "block3",
        ),
    ]

    rows = []
    row_num = 0
    for shift_name, shift_cfg, shift_desc, block_name in shifts:
        group = f"track4_checkpoint6_v100_shift_{block_name}"
        for seed in [1071, 1072, 1073, 1074]:
            row_num += 1
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{WINNER_STRATEGY}_{shift_name}_n1_s{seed}.yaml"
            )
            updates = {
                "description": f'"{track_key} {WINNER_STRATEGY} checkpoint6 {shift_name} {DATE_LABEL}"',
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
                "features_sub_length": shift_cfg["features_sub_length"],
                "features_sub_step": shift_cfg["features_sub_step"],
                "features_additive_noise_std": shift_cfg["features_additive_noise_std"],
                "features_multiplicative_noise_std": shift_cfg["features_multiplicative_noise_std"],
                "features_baseline_drift_std": shift_cfg["features_baseline_drift_std"],
                "features_mask_fraction": shift_cfg["features_mask_fraction"],
                "global_train_batch_size": "128",
                "train_batch_size": "128",
                "random_seed": str(seed),
                "loss_type": "huber",
                "huber_beta": "0.05",
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
            rows.append(
                {
                    "row_id": f"v100cp6n_{row_num:04d}",
                    "phase": "phaseCP6_track4_v100_shift_stability_neural",
                    "phase_label": "Checkpoint6 V100 neural shift stability",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": group,
                    "family": "track4_checkpoint6_shift_stability_neural",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "1",
                    "cpus_per_node": "12",
                    "gpu_type": "V100",
                    "batch_profile": "eval_only_gpu",
                    "learning_rate": "1.5e-4",
                    "features_sub_length": shift_cfg["features_sub_length"],
                    "features_sub_step": shift_cfg["features_sub_step"],
                    "features_additive_noise_std": shift_cfg["features_additive_noise_std"],
                    "features_multiplicative_noise_std": shift_cfg["features_multiplicative_noise_std"],
                    "features_baseline_drift_std": shift_cfg["features_baseline_drift_std"],
                    "features_mask_fraction": shift_cfg["features_mask_fraction"],
                    "Ntrain": "4096",
                    "global_train_batch_size": "128",
                    "seed": str(seed),
                    "strategy_id": f"{WINNER_STRATEGY}_{shift_name}",
                    "strategy_description": f"Eval-only neural checkpoint6 shift family: {shift_desc}",
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
                    "eval_only_checkpoint": winner_runs[seed]["checkpoint"],
                    "source_run_root": winner_runs[seed]["run_root"],
                    "task_slug": f"{track_key}_{WINNER_STRATEGY}_{shift_name}_n1_s{seed}",
                    "notes": "Checkpoint 6 neural shift-stability eval-only block on Falcon V100.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    note_lines = [
        f"# V100 HH Track4 Checkpoint6 Neural Shift Stability ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- evaluate the best confirmed neural baseline under the checkpoint6 shift families without retraining",
        "- characterize whether the neural family ranking is stable outside simple additive-noise testing",
        "",
        "## Shift Families",
        *[f"- `{name}`: {desc}" for name, _cfg, desc, _block in shifts],
    ]
    write_text(NOTES_MD, "\n".join(note_lines) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
