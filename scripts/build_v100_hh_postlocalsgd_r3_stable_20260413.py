#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260413_v100_hh_postlocalsgd_r3_stable"
DATE_LABEL = "2026-04-13"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_postlocalsgd_r3_stable_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_postlocalsgd_r3_stable_notes_{DATE_TAG}.md"
SPLIT_ARRAY_CACHE_OPT_IN = os.environ.get("ENABLE_SPLIT_ARRAY_CACHE", "1") != "0"


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


def fmt_lr(v: float) -> str:
    s = f"{v:.1e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def maybe_apply_split_array_cache(updates: Dict[str, str]) -> None:
    if not SPLIT_ARRAY_CACHE_OPT_IN:
        return
    updates["split_array_cache_enabled"] = "True"


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
        "grad_accum_steps",
        "use_no_sync_for_accum",
    }
    allow_add_runconfig = {
        "save_diagnostic_plots",
        "ddp_static_graph",
        "ddp_gradient_as_bucket_view",
        "ddp_bucket_cap_mb",
        "debug",
        "post_local_sgd_enabled",
        "post_local_sgd_period",
        "post_local_sgd_warmup_steps",
        "post_local_sgd_start_local_sgd_iter",
        "post_local_gradient_allreduce",
        "skip_final_distributed_barrier",
    }
    allow_add_data = {
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_multiprocessing_context",
        "dataloader_prefetch_factor",
        "dataloader_pin_memory",
        "features_scale_cache_enabled",
        "split_array_cache_enabled",
    }
    for k, v in updates.items():
        try:
            txt = set_key(txt, k, v)
        except ValueError:
            if k in allow_add_training:
                txt = add_key_under_section(txt, "training", k, v)
            elif k in allow_add_runconfig:
                txt = add_key_under_section(txt, "runconfig", k, v)
            elif k in allow_add_data:
                txt = add_key_under_section(txt, "data", k, v)
            else:
                raise
    write_text(out_path, txt)
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
            "save_predictions": "none",
            "save_diagnostic_plots": "False",
            "save_checkpoints_epochs": "50",
            "ntrain": "4096",
        },
        "track4_hh_full": {
            "short": "t4",
            "base": "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "prefix": "concatenated_data",
            "save_predictions": "test",
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "50",
            "ntrain": "8192",
        },
    }

    variants = [
        ("R3_postlocalsgd_p4_w100", 4, 100),
        ("R3_postlocalsgd_p8_w100", 8, 100),
    ]

    rows: List[Dict[str, str]] = []
    row_num = 0
    for track_key, meta in track_meta.items():
        for nodes in [2, 4]:
            for variant_name, period, warmup_steps in variants:
                for seed in [971, 972, 973]:
                    row_num += 1
                    cfg_rel = (
                        f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                        f"params_{meta['short']}_{variant_name}_n{nodes}_s{seed}.yaml"
                    )
                    updates = {
                        "description": f'"{track_key} {variant_name} phase R3 stable post-local SGD {DATE_LABEL}"',
                        "learning_rate": fmt_lr(3.0e-4),
                        "epochs": "400",
                        "init_learning_rate": fmt_lr(1.5e-5),
                        "linear_epochs": "40",
                        "constant_epochs": "40",
                        "final_learning_rate": fmt_lr(1.5e-6),
                        "features_normalize": "True",
                        "features_scale_cache_enabled": "True",
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
                        "use_amp": "True",
                        "amp_dtype": '"float16"',
                        "grad_accum_steps": "1",
                        "use_no_sync_for_accum": "True",
                        "post_local_sgd_enabled": "True",
                        "post_local_sgd_period": str(period),
                        "post_local_sgd_warmup_steps": str(warmup_steps),
                        "post_local_sgd_start_local_sgd_iter": str(warmup_steps),
                        "post_local_gradient_allreduce": "True",
                        "skip_final_distributed_barrier": "False",
                        "dataloader_num_workers": "0",
                        "dataloader_persistent_workers": "False",
                        "dataloader_prefetch_factor": "null",
                        "save_predictions": f'"{meta["save_predictions"]}"',
                        "save_diagnostic_plots": meta["save_diagnostic_plots"],
                        "save_checkpoints_epochs": meta["save_checkpoints_epochs"],
                        "debug": "False",
                        "save_dir": f'"runs/{DATE_TAG}"',
                    }
                    maybe_apply_split_array_cache(updates)
                    create_config(meta["base"], cfg_rel, updates)
                    rows.append(
                        {
                            "row_id": f"v100r3s_{row_num:04d}",
                            "phase": "phaseR3_postlocalsgd_stable",
                            "launch_mode": "dedicated",
                            "launch_group": "",
                            "family": "postlocalsgd_screen_stable",
                            "track_key": track_key,
                            "policy": "strong",
                            "nodes": str(nodes),
                            "gpus_per_node": "2",
                            "cpus_per_node": "24",
                            "gpu_type": "V100",
                            "batch_profile": "strong_gbs128",
                            "learning_rate": fmt_lr(3.0e-4),
                            "features_sub_length": "2000",
                            "Ntrain": meta["ntrain"],
                            "global_train_batch_size": "128",
                            "seed": str(seed),
                            "strategy_id": variant_name,
                            "strategy_description": f"Stable post-local SGD with period={period} warmup_steps={warmup_steps} and dataloader_num_workers=0",
                            "params_file": cfg_rel,
                            "tar_path": meta["tar"],
                            "data_prefix": meta["prefix"],
                            "curr": "0.1",
                            "data_access_mode": "direct_tar",
                            "shared_data_dir": str(
                                IMPORTANT / "optimization_track_20260411_v100_interactive" / "shared_data" / track_key / meta["prefix"]
                            ),
                            "save_predictions": meta["save_predictions"],
                            "save_diagnostic_plots": meta["save_diagnostic_plots"],
                            "save_checkpoints_epochs": meta["save_checkpoints_epochs"],
                            "task_slug": f"{track_key}_{variant_name}_n{nodes}_s{seed}",
                            "notes": "Phase R3 stable screen with rank-local temp dirs and dataloader_num_workers=0.",
                        }
                    )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Post-Local SGD R3 Stable Matrix ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- config root: `{CONFIG_ROOT}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Design",
        "- Stable follow-on package after the initial R3 pilot exposed stale-step teardown bugs",
        "- Tracks: track3_hh_reduced, track4_hh_full",
        "- Nodes: 2 and 4",
        "- Seeds: 971, 972, 973",
        "- Variants: post-local SGD with period 4 / 8 and warmup 100",
        "- Runtime fix enabled via rank-local temp dirs and dataloader_num_workers=0",
        "- Final distributed barrier kept enabled for safer rank-0 final checkpoint completion",
        "- Checkpoint cadence set to every 50 epochs so late-stage recovery has salvageable artifacts",
        "- Total rows: 24",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
