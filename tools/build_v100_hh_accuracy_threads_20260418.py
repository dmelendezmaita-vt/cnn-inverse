#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict

DATE_TAG = "20260418_v100_hh_accuracy_threads"
DATE_LABEL = "2026-04-18"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_accuracy_threads_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_accuracy_threads_notes_{DATE_TAG}.md"


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


def set_key_under_section(text: str, section: str, key: str, value: str) -> str:
    section_pat = re.compile(
        rf"(^[ \t]*{re.escape(section)}:[ \t]*\n)(?P<body>(?:^[ \t]+.*\n)*)",
        flags=re.MULTILINE,
    )
    match = section_pat.search(text)
    if not match:
        raise ValueError(f"Section '{section}' not found while setting key '{key}'")
    body = match.group("body")
    key_pat = re.compile(rf"(^[ \t]+{re.escape(key)}:)[ \t]*.*$", flags=re.MULTILINE)
    new_body, n = key_pat.subn(rf"\1 {value}", body)
    if n == 0:
        raise ValueError(f"Key '{key}' not found under section '{section}'")
    return text[: match.start("body")] + new_body + text[match.end("body") :]


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
        "split_strategy",
    }
    allow_add_runconfig = {"save_diagnostic_plots"}
    for k, v in updates.items():
        if "." in k:
            section, subkey = k.split(".", 1)
            txt = set_key_under_section(txt, section, subkey, v)
            continue
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


def common_updates(seed: int) -> Dict[str, str]:
    return {
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
        "use_amp": "False",
        "amp_dtype": '"float16"',
        "save_predictions": '"test"',
        "save_diagnostic_plots": "True",
        "save_checkpoints_epochs": "10",
        "save_dir": f'"runs/{DATE_TAG}"',
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_meta = {
        "track3_hh_reduced": {
            "short": "t3",
            "base": "src/pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml",
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
            "nvalidate": "1024",
            "ntest": "1024",
        },
        "track4_hh_full": {
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
        },
    }

    # Keep batch size / epochs fixed in the first accuracy wave so we can learn
    # which quality-facing knobs matter before opening the batch/epoch sidecar.
    phase_a_track3_strategies = {
        "acc_base": {
            "description": "Track3 baseline accuracy screen: factor_s5 defaults.",
            "updates": {},
        },
        "acc_uniform": {
            "description": "Track3 accuracy screen: uniform target weighting.",
            "updates": {
                "target_loss_weight_mode": "uniform",
            },
        },
        "acc_adamw_wd0": {
            "description": "Track3 accuracy screen: AdamW with zero weight decay.",
            "updates": {
                "weight_decay": "0.0",
            },
        },
        "acc_adam_lr1e3": {
            "description": "Track3 accuracy screen: Adam lr=1e-3, weight_decay=0.",
            "updates": {
                "optimizer.type": "'Adam'",
                "learning_rate": "1.0e-3",
                "init_learning_rate": "5.0e-5",
                "final_learning_rate": "5.0e-6",
                "weight_decay": "0.0",
            },
        },
    }

    phase_b_track4_strategies = {
        "acc_base": {
            "description": "Track4 full-data confirmation baseline: factor_s5 defaults.",
            "updates": {},
        },
        "acc_uniform": {
            "description": "Track4 full-data confirmation: uniform target weighting.",
            "updates": {
                "target_loss_weight_mode": "uniform",
            },
        },
        "acc_bigsplit": {
            "description": "Track4 full-data confirmation: larger random split budget.",
            "updates": {
                "split_strategy": "random",
                "Ntrain": "8192",
                "Nvalidate": "2048",
                "Ntest": "1024",
            },
        },
        "acc_quality_combo": {
            "description": "Track4 full-data confirmation: larger random split + uniform weighting + Adam low-regularization.",
            "updates": {
                "split_strategy": "random",
                "Ntrain": "8192",
                "Nvalidate": "2048",
                "Ntest": "1024",
                "target_loss_weight_mode": "uniform",
                "optimizer.type": "'Adam'",
                "learning_rate": "1.0e-3",
                "init_learning_rate": "5.0e-5",
                "final_learning_rate": "5.0e-6",
                "weight_decay": "0.0",
                "features_sub_begin_random_eval": "True",
            },
        },
    }

    rows = []
    row_num = 0

    for seed in [971, 972, 973]:
        group = f"track3_seed{seed}_accuracy"
        for strategy_id, strategy in phase_a_track3_strategies.items():
            row_num += 1
            track_key = "track3_hh_reduced"
            meta = track_meta[track_key]
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{strategy_id}_n1_s{seed}.yaml"
            )
            updates = common_updates(seed)
            updates.update(
                {
                    "description": f'"{track_key} {strategy_id} accuracy screen {DATE_LABEL}"',
                    "Ntrain": meta["ntrain"],
                    "Nvalidate": meta["nvalidate"],
                    "Ntest": meta["ntest"],
                }
            )
            updates.update(strategy["updates"])
            create_config(meta["base"], cfg_rel, updates)
            rows.append(
                {
                    "row_id": f"v100acc_{row_num:04d}",
                    "phase": "phaseA_track3_accuracy_screen",
                    "phase_label": "Track3 quick accuracy screen",
                    "launch_mode": "concurrent_4x1n",
                    "launch_group": group,
                    "family": "accuracy_threads",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "2",
                    "cpus_per_node": "24",
                    "gpu_type": "V100",
                    "batch_profile": "strong_gbs128",
                    "learning_rate": updates["learning_rate"],
                    "features_sub_length": "2000",
                    "Ntrain": updates["Ntrain"],
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
                    "save_predictions": "test",
                    "save_diagnostic_plots": "True",
                    "save_checkpoints_epochs": "10",
                    "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}",
                    "notes": "Track3 fast accuracy screen; batch size and epochs held fixed.",
                }
            )

    for seed in [971, 972, 973]:
        group = f"track4_seed{seed}_accuracy"
        for strategy_id, strategy in phase_b_track4_strategies.items():
            row_num += 1
            track_key = "track4_hh_full"
            meta = track_meta[track_key]
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{strategy_id}_n1_s{seed}.yaml"
            )
            updates = common_updates(seed)
            updates.update(
                {
                    "description": f'"{track_key} {strategy_id} accuracy confirmation {DATE_LABEL}"',
                    "Ntrain": meta["ntrain"],
                    "Nvalidate": meta["nvalidate"],
                    "Ntest": meta["ntest"],
                }
            )
            updates.update(strategy["updates"])
            create_config(meta["base"], cfg_rel, updates)
            rows.append(
                {
                    "row_id": f"v100acc_{row_num:04d}",
                    "phase": "phaseB_track4_full_accuracy_confirmation",
                    "phase_label": "Track4 full-data accuracy confirmation",
                    "launch_mode": "concurrent_4x1n",
                    "launch_group": group,
                    "family": "accuracy_threads",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "gpus_per_node": "2",
                    "cpus_per_node": "24",
                    "gpu_type": "V100",
                    "batch_profile": "strong_gbs128",
                    "learning_rate": updates["learning_rate"],
                    "features_sub_length": "2000",
                    "Ntrain": updates["Ntrain"],
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
                    "save_predictions": "test",
                    "save_diagnostic_plots": "True",
                    "save_checkpoints_epochs": "10",
                    "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}",
                    "notes": "Track4 full-data confirmation; batch size and epochs held fixed.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Accuracy Threads ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- phase A: use track3_hh_reduced for a quick quality screen",
        "- phase B: confirm the best full-data quality ideas on track4_hh_full",
        "",
        "## Intentional controls",
        "- keep global_train_batch_size fixed at 128",
        "- keep epochs fixed at 400",
        "- keep the shared-data + cache datapath stack unchanged",
        "",
        "## Track 3 screen",
        "- acc_base: current factor_s5 defaults",
        "- acc_uniform: uniform target weighting",
        "- acc_adamw_wd0: remove weight decay",
        "- acc_adam_lr1e3: Adam + higher learning rate + zero weight decay",
        "",
        "## Track 4 confirmation",
        "- acc_base: current factor_s5 defaults",
        "- acc_uniform: uniform target weighting",
        "- acc_bigsplit: larger random split budget",
        "- acc_quality_combo: larger random split + uniform weighting + Adam low-regularization",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
