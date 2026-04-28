#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict

DATE_TAG = "20260418_v100_hh_1n_factorization"
DATE_LABEL = "2026-04-18"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_1n_factorization_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_1n_factorization_notes_{DATE_TAG}.md"


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

    track_meta = {
        "track3_hh_reduced": {
            "short": "t3",
            "base": "pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            "prefix": "reduced_data",
            "shared": str(
                IMPORTANT
                / "optimization_track_20260411_v100_interactive"
                / "shared_data"
                / "track3_hh_reduced"
                / "reduced_data"
            ),
            "ntrain": "4096",
        },
        "track4_hh_full": {
            "short": "t4",
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
            "ntrain": "8192",
        },
    }

    strategies = {
        "factor_s2": {
            "description": "Shared-data + no-loss-sync + no AMP + full artifacts.",
            "use_amp": "False",
            "save_predictions": '"test"',
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "10",
        },
        "factor_s5": {
            "description": "Shared-data + no-loss-sync + no AMP + lean artifacts.",
            "use_amp": "False",
            "save_predictions": '"none"',
            "save_diagnostic_plots": "False",
            "save_checkpoints_epochs": "null",
        },
    }

    rows = []
    row_num = 0
    for track_key, meta in track_meta.items():
        for strategy_id, strategy in strategies.items():
            group = f"{track_key}_{strategy_id}_1n_factor"
            for seed, repeat_label in [(971, "a"), (972, "a"), (973, "a"), (971, "b")]:
                row_num += 1
                cfg_rel = (
                    f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{strategy_id}_n1_s{seed}_{repeat_label}.yaml"
                )
                updates = {
                    "description": f'"{track_key} {strategy_id} 1n factorization {DATE_LABEL}"',
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
                    "huber_beta": "1.0",
                    "grad_clip_max_norm": "1.0",
                    "reduce_loss_for_logging": "False",
                    "use_amp": strategy["use_amp"],
                    "amp_dtype": '"float16"',
                    "save_predictions": strategy["save_predictions"],
                    "save_diagnostic_plots": strategy["save_diagnostic_plots"],
                    "save_checkpoints_epochs": strategy["save_checkpoints_epochs"],
                    "save_dir": f'"runs/{DATE_TAG}"',
                }
                create_config(meta["base"], cfg_rel, updates)
                rows.append(
                    {
                        "row_id": f"v100fac_{row_num:04d}",
                        "phase": "phaseF1_1n_factorization",
                        "phase_label": "1n factorization grouped confirmation",
                        "launch_mode": "concurrent_4x1n",
                        "launch_group": group,
                        "family": "1n_factorization",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "gpus_per_node": "2",
                        "cpus_per_node": "24",
                        "gpu_type": "V100",
                        "batch_profile": "strong_gbs128",
                        "learning_rate": "3.0e-4",
                        "features_sub_length": "2000",
                        "Ntrain": meta["ntrain"],
                        "global_train_batch_size": "128",
                        "seed": str(seed),
                        "strategy_id": strategy_id,
                        "strategy_description": strategy["description"],
                        "params_file": cfg_rel,
                        "tar_path": meta["tar"],
                        "data_prefix": meta["prefix"],
                        "curr": "0.1",
                        "data_access_mode": "direct_tar",
                        "shared_data_dir": meta["shared"],
                        "save_predictions": strategy["save_predictions"].strip('"'),
                        "save_diagnostic_plots": strategy["save_diagnostic_plots"],
                        "save_checkpoints_epochs": strategy["save_checkpoints_epochs"],
                        "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}_{repeat_label}",
                        "notes": "Grouped 1n factorization package to isolate artifact policy from AMP.",
                    }
                )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH 1n Factorization ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- isolate whether the current 1n defaults are limited by AMP or by artifact policy",
        "- compare S2-style no-AMP full-artifact rows against a new no-AMP lean-artifact candidate",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
