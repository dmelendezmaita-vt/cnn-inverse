#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import yaml

DATE_TAG = "20260423_v100_hh_track4_checkpoint6_drift_mask_boundary_neural"
DATE_LABEL = "2026-04-23"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_checkpoint6_drift_mask_boundary_neural_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_checkpoint6_drift_mask_boundary_neural_notes_{DATE_TAG}.md"

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

    params.setdefault("data", {})
    params["data"]["features_normalize"] = True
    params["data"]["features_scale_cache_enabled"] = True
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["features_sub_length"] = 2000
    params["data"]["features_sub_step"] = 1
    params["data"]["features_additive_noise_std"] = 0.0
    params["data"]["features_multiplicative_noise_std"] = 0.0
    params["data"]["features_baseline_drift_std"] = 0.0
    params["data"]["features_mask_fraction"] = 0.0
    params.setdefault("net", {})
    params["net"]["type"] = "EfficientNet"
    params["net"]["dropout"] = 0.1
    params.setdefault("optimizer", {})
    params["optimizer"]["learning_rate"] = 1.5e-4
    params["optimizer"]["learning_rate_scheduler"] = {
        "init_learning_rate": 7.5e-6,
        "linear_epochs": 75,
        "constant_epochs": 75,
        "final_learning_rate": 7.5e-7,
    }
    params.setdefault("training", {})
    params["training"]["epochs"] = 500
    params["training"]["loss_type"] = "huber"
    params["training"]["huber_beta"] = 0.05
    params["training"]["grad_clip_max_norm"] = 1.0
    params["training"]["reduce_loss_for_logging"] = False
    params["training"]["use_amp"] = False
    params["training"]["amp_dtype"] = "float16"

    data_keys = {
        "features_baseline_drift_std",
        "features_mask_fraction",
        "random_seed",
        "Ntrain",
        "Nvalidate",
        "Ntest",
        "global_train_batch_size",
        "train_batch_size",
        "target_loss_weight_mode",
    }
    for k, v in updates.items():
        value = yaml.safe_load(v)
        if k == "description":
            params["description"] = value
        elif k in data_keys:
            params["data"][k] = value
        else:
            raise KeyError(f"Unhandled config key in drift/mask neural boundary builder: {k}")

    params.setdefault("runconfig", {})
    params["runconfig"]["save_predictions"] = "test"
    params["runconfig"]["save_diagnostic_plots"] = True
    params["runconfig"]["save_checkpoints_epochs"] = 10
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return out_rel


def find_winner_runs() -> dict[int, dict[str, str]]:
    matches: dict[int, dict[str, str]] = {}
    for run_root in sorted(TOP3_RUNS.glob("a30t4top_*/*")):
        params_path = run_root / "params.yaml"
        if not params_path.exists():
            continue
        if WINNER_STRATEGY not in params_path.read_text():
            continue
        params = yaml.safe_load(params_path.read_text())
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
        "short": "t4v100bdm",
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
        ("drift002", {"features_baseline_drift_std": "0.02"}, "Linear baseline drift, std=0.02.", "block1"),
        ("drift003", {"features_baseline_drift_std": "0.03"}, "Linear baseline drift, std=0.03.", "block1"),
        ("drift004", {"features_baseline_drift_std": "0.04"}, "Linear baseline drift, std=0.04.", "block2"),
        ("mask05", {"features_mask_fraction": "0.05"}, "Contiguous missing segment covering 5% of the trace.", "block2"),
        ("mask15", {"features_mask_fraction": "0.15"}, "Contiguous missing segment covering 15% of the trace.", "block3"),
        ("mask25", {"features_mask_fraction": "0.25"}, "Contiguous missing segment covering 25% of the trace.", "block3"),
    ]

    rows = []
    row_num = 0
    for shift_name, shift_cfg, shift_desc, block_name in shifts:
        group = f"track4_checkpoint6_boundary_{block_name}"
        for seed in [1071, 1072, 1073, 1074]:
            row_num += 1
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{WINNER_STRATEGY}_{shift_name}_n1_s{seed}.yaml"
            )
            updates = {
                "description": f'"{track_key} {WINNER_STRATEGY} boundary {shift_name} {DATE_LABEL}"',
                **shift_cfg,
                "random_seed": str(seed),
                "Ntrain": "4096",
                "Nvalidate": "1024",
                "Ntest": "1024",
                "global_train_batch_size": "128",
                "train_batch_size": "128",
                "target_loss_weight_mode": '"inverse_var"',
            }
            create_config(meta["base"], cfg_rel, updates)
            rows.append(
                {
                    "row_id": f"v100bdm_{row_num:04d}",
                    "phase": "phaseCP6_track4_drift_mask_boundary_neural",
                    "phase_label": "Checkpoint6 drift/mask boundary neural",
                    "launch_mode": "concurrent_8x1n",
                    "launch_group": group,
                    "family": "track4_checkpoint6_drift_mask_boundary_neural",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "1",
                    "cpus_per_node": "12",
                    "gpu_type": "V100",
                    "batch_profile": "eval_only_gpu",
                    "learning_rate": "1.5e-4",
                    "features_sub_length": "2000",
                    "features_sub_step": "1",
                    "features_additive_noise_std": "0.0",
                    "features_multiplicative_noise_std": "0.0",
                    "features_baseline_drift_std": shift_cfg.get("features_baseline_drift_std", "0.0"),
                    "features_mask_fraction": shift_cfg.get("features_mask_fraction", "0.0"),
                    "Ntrain": "4096",
                    "global_train_batch_size": "128",
                    "seed": str(seed),
                    "strategy_id": f"{WINNER_STRATEGY}_{shift_name}",
                    "strategy_description": f"Boundary probe: {shift_desc}",
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
                    "notes": "Checkpoint6 boundary-search follow-up around the drift and masking flip points on Falcon.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    note_lines = [
        f"# V100 HH Track4 Checkpoint6 Drift/Mask Boundary Neural ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- refine the boundary where the checkpoint6 neural ranking flips begin",
        "- only run the missing intermediate severities instead of rerunning the full ladder",
        "",
        "## Shift Boundary Points",
        *[f"- `{name}`: {desc}" for name, _cfg, desc, _block in shifts],
    ]
    write_text(NOTES_MD, "\n".join(note_lines) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
