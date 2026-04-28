#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict

DATE_TAG = "20260419_v100_hh_track4_beta_budget_interaction"
DATE_LABEL = "2026-04-19"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_track4_beta_budget_interaction_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_track4_beta_budget_interaction_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def set_key(text: str, key: str, value: str) -> str:
    pattern = rf"(^[ \t]*{re.escape(key)}:)[ \t]*.*$"
    repl = rf"\g<1> {value}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Key '{key}' not found while patching config")
    return new_text


def add_key_under_section(text: str, section: str, key: str, value: str) -> str:
    pattern = rf"(?m)^({re.escape(section)}:[ \t]*\n)"
    repl = rf"\1    {key}: {value}\n"
    new_text, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        raise ValueError(f"Section '{section}' not found while adding key '{key}'")
    return new_text


def create_config(base_rel: str, out_rel: str, updates: Dict[str, str]) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    txt = read_text(base_path)
    allow_add_training = {
        "loss_type",
        "huber_beta",
        "grad_clip_max_norm",
        "reduce_loss_for_logging",
        "use_amp",
        "amp_dtype",
    }
    allow_add_data = {
        "features_scale_cache_enabled",
        "split_array_cache_enabled",
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "dataloader_pin_memory",
    }
    allow_add_runconfig = {"save_diagnostic_plots"}
    for k, v in updates.items():
        try:
            txt = set_key(txt, k, v)
        except ValueError:
            if k in allow_add_training:
                txt = add_key_under_section(txt, "training", k, v)
            elif k in allow_add_data:
                txt = add_key_under_section(txt, "data", k, v)
            elif k in allow_add_runconfig:
                txt = add_key_under_section(txt, "runconfig", k, v)
            else:
                raise
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt)
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
        "ntrain": "4096",
        "nvalidate": "1024",
        "ntest": "1024",
    }

    variants = {
        "base_beta1_g128_e400": {
            "description": "Track4 baseline: beta=1.0, gbs128, epochs400.",
            "huber_beta": "1.0",
            "global_train_batch_size": "128",
            "train_batch_size": "128",
            "epochs": "400",
            "init_learning_rate": "1.5e-5",
            "linear_epochs": "40",
            "constant_epochs": "40",
            "final_learning_rate": "1.5e-6",
        },
        "beta0p1_g128_e400": {
            "description": "Track4 loss-only candidate: beta=0.1, gbs128, epochs400.",
            "huber_beta": "0.1",
            "global_train_batch_size": "128",
            "train_batch_size": "128",
            "epochs": "400",
            "init_learning_rate": "1.5e-5",
            "linear_epochs": "40",
            "constant_epochs": "40",
            "final_learning_rate": "1.5e-6",
        },
        "beta0p1_g64_e400": {
            "description": "Track4 beta-budget interaction: beta=0.1, gbs64, epochs400.",
            "huber_beta": "0.1",
            "global_train_batch_size": "64",
            "train_batch_size": "64",
            "epochs": "400",
            "init_learning_rate": "1.5e-5",
            "linear_epochs": "40",
            "constant_epochs": "40",
            "final_learning_rate": "1.5e-6",
        },
        "beta0p1_g64_e600": {
            "description": "Track4 beta-budget interaction: beta=0.1, gbs64, epochs600.",
            "huber_beta": "0.1",
            "global_train_batch_size": "64",
            "train_batch_size": "64",
            "epochs": "600",
            "init_learning_rate": "1.5e-5",
            "linear_epochs": "60",
            "constant_epochs": "60",
            "final_learning_rate": "1.5e-6",
        },
    }

    rows = []
    row_num = 0
    for seeds in [(999, 1000), (1001, 1002)]:
        for seed in seeds:
            group = f"track4_beta_budget_seed_{seed}"
            for strategy_id, variant in variants.items():
                row_num += 1
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{strategy_id}_n1_s{seed}.yaml"
                )
                updates = {
                    "description": f'"{track_key} {strategy_id} beta-budget interaction {DATE_LABEL}"',
                    "learning_rate": "3.0e-4",
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
                    "global_train_batch_size": variant["global_train_batch_size"],
                    "train_batch_size": variant["train_batch_size"],
                    "random_seed": str(seed),
                    "loss_type": "huber",
                    "huber_beta": variant["huber_beta"],
                    "grad_clip_max_norm": "1.0",
                    "reduce_loss_for_logging": "False",
                    "use_amp": "False",
                    "amp_dtype": '"float16"',
                    "save_predictions": '"test"',
                    "save_diagnostic_plots": "True",
                    "save_checkpoints_epochs": "10",
                    "save_dir": f'"runs/{DATE_TAG}"',
                    "Ntrain": meta["ntrain"],
                    "Nvalidate": meta["nvalidate"],
                    "Ntest": meta["ntest"],
                    "target_loss_weight_mode": "inverse_std",
                }
                create_config(meta["base"], cfg_rel, updates)
                rows.append(
                    {
                        "row_id": f"v100t4bbi_{row_num:04d}",
                        "phase": "phaseR_track4_beta_budget_interaction",
                        "phase_label": "Track4 beta-budget interaction",
                        "launch_mode": "concurrent_4x1n",
                        "launch_group": group,
                        "family": "track4_beta_budget_interaction",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "gpus_per_node": "2",
                        "cpus_per_node": "24",
                        "gpu_type": "V100",
                        "batch_profile": f"gbs{variant['global_train_batch_size']}_e{variant['epochs']}",
                        "learning_rate": "3.0e-4",
                        "features_sub_length": "2000",
                        "Ntrain": meta["ntrain"],
                        "global_train_batch_size": variant["global_train_batch_size"],
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
                        "notes": "Interaction follow-up for the new beta=0.1 track4 candidate.",
                    }
                )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Track4 Beta-Budget Interaction ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- test whether the new `beta=0.1` winner gets stronger or weaker when paired with smaller batch and longer training",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
