#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260412_v100_hh_scaling_redesign"
DATE_LABEL = "2026-04-12"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_scaling_redesign_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_scaling_redesign_notes_{DATE_TAG}.md"
SPLIT_ARRAY_CACHE_OPT_IN = os.environ.get("ENABLE_SPLIT_ARRAY_CACHE", "1") != "0"
TRACK3_R5C_DDP_CANDIDATE_DEFAULT = os.environ.get("ENABLE_TRACK3_R5C_DDP_CANDIDATE", "0") != "0"


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


def slug_float(v: float) -> str:
    s = fmt_lr(v)
    return s.replace(".", "p").replace("+", "").replace("-", "m")


def maybe_apply_split_array_cache(updates: Dict[str, str]) -> None:
    if not SPLIT_ARRAY_CACHE_OPT_IN:
        return
    updates.update(
        {
            "split_array_cache_enabled": "True",
            "dataloader_num_workers": "0",
            "dataloader_persistent_workers": "False",
            "dataloader_prefetch_factor": "null",
            "dataloader_pin_memory": "False",
        }
    )


def maybe_apply_track3_r5c_ddp_candidate(
    *,
    track_key: str,
    nodes: int,
    variant_name: str,
    updates: Dict[str, str],
) -> None:
    if not TRACK3_R5C_DDP_CANDIDATE_DEFAULT:
        return
    if track_key != "track3_hh_reduced" or nodes != 4 or variant_name != "R1_accum2_nosync":
        return
    updates.update(
        {
            "ddp_gradient_as_bucket_view": "True",
            "ddp_bucket_cap_mb": "100",
        }
    )


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
    }
    allow_add_data = {
        "features_scale_cache_enabled",
        "split_array_cache_enabled",
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "dataloader_pin_memory",
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
            # carry over the best operational baseline from 20260411
            "save_predictions": "none",
            "save_diagnostic_plots": "False",
            "save_checkpoints_epochs": "null",
        },
        "track4_hh_full": {
            "short": "t4",
            "base": "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "prefix": "concatenated_data",
            # carry over the best operational baseline from 20260411
            "save_predictions": "test",
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "10",
        },
    }

    variants = [
        ("B0_operational_baseline", 1),
        ("R1_accum2_nosync", 2),
        ("R1_accum4_nosync", 4),
        ("R1_accum8_nosync", 8),
    ]

    rows: List[Dict[str, str]] = []
    row_num = 0

    for track_key, meta in track_meta.items():
        for seed in [971]:
            for variant_name, accum_steps in variants:
                row_num += 1
                cfg_rel = (
                    f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{variant_name}_n1_s{seed}.yaml"
                )
                updates = {
                    "description": f'"{track_key} {variant_name} communication-reduction redesign {DATE_LABEL}"',
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
                    "grad_accum_steps": str(accum_steps),
                    "use_no_sync_for_accum": "True",
                    "save_predictions": f'"{meta["save_predictions"]}"',
                    "save_diagnostic_plots": meta["save_diagnostic_plots"],
                    "save_checkpoints_epochs": meta["save_checkpoints_epochs"],
                    "save_dir": f'"runs/{DATE_TAG}"',
                }
                maybe_apply_split_array_cache(updates)
                create_config(meta["base"], cfg_rel, updates)
                rows.append(
                    {
                        "row_id": f"v100red_{row_num:04d}",
                        "phase": "phaseR1_comm_reduction",
                        "launch_mode": "concurrent_4x1n" if track_key == "track3_hh_reduced" else "concurrent_4x1n",
                        "launch_group": f"{track_key}_seed{seed}",
                        "family": "comm_reduction_screen",
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": "1",
                        "gpus_per_node": "2",
                        "cpus_per_node": "24",
                        "gpu_type": "V100",
                        "batch_profile": "strong_gbs128",
                        "learning_rate": fmt_lr(3.0e-4),
                        "features_sub_length": "2000",
                        "Ntrain": "4096" if track_key == "track3_hh_reduced" else "8192",
                        "global_train_batch_size": "128",
                        "seed": str(seed),
                        "strategy_id": variant_name,
                        "strategy_description": f"Communication-reduction screen with grad_accum_steps={accum_steps}",
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
                        "task_slug": f"{track_key}_{variant_name}_n1_s{seed}",
                        "notes": "Phase R1 initial 1-node communication-reduction screen.",
                    }
                )

    # Phase R1b: immediate scaling probe on the lowest-risk redesign candidate only
    scaling_probe_variants = [
        ("B0_operational_baseline", 1),
        ("R1_accum2_nosync", 2),
    ]
    for track_key, meta in track_meta.items():
        for seed in [971]:
            for nodes in [2, 4]:
                for variant_name, accum_steps in scaling_probe_variants:
                    row_num += 1
                    cfg_rel = (
                        f"pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                        f"params_{meta['short']}_{variant_name}_n{nodes}_s{seed}.yaml"
                    )
                    updates = {
                        "description": f'"{track_key} {variant_name} scaling-probe redesign {DATE_LABEL}"',
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
                        "grad_accum_steps": str(accum_steps),
                        "use_no_sync_for_accum": "True",
                        "save_predictions": f'"{meta["save_predictions"]}"',
                        "save_diagnostic_plots": meta["save_diagnostic_plots"],
                        "save_checkpoints_epochs": meta["save_checkpoints_epochs"],
                        "save_dir": f'"runs/{DATE_TAG}"',
                    }
                    maybe_apply_split_array_cache(updates)
                    maybe_apply_track3_r5c_ddp_candidate(
                        track_key=track_key,
                        nodes=nodes,
                        variant_name=variant_name,
                        updates=updates,
                    )
                    create_config(meta["base"], cfg_rel, updates)
                    rows.append(
                        {
                            "row_id": f"v100red_{row_num:04d}",
                            "phase": "phaseR1_scaling_probe",
                            "launch_mode": "dedicated",
                            "launch_group": "",
                            "family": "comm_reduction_scaling_probe",
                            "track_key": track_key,
                            "policy": "strong",
                            "nodes": str(nodes),
                            "gpus_per_node": "2",
                            "cpus_per_node": "24",
                            "gpu_type": "V100",
                            "batch_profile": "strong_gbs128",
                            "learning_rate": fmt_lr(3.0e-4),
                            "features_sub_length": "2000",
                            "Ntrain": "4096" if track_key == "track3_hh_reduced" else "8192",
                            "global_train_batch_size": "128",
                            "seed": str(seed),
                            "strategy_id": variant_name,
                            "strategy_description": f"Scaling probe with grad_accum_steps={accum_steps}",
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
                            "notes": (
                                "Phase R1b immediate multi-node scaling probe."
                                + (
                                    " Track3 4n R1_accum2_nosync rows default to the active R5c candidate"
                                    " (ddp_gradient_as_bucket_view=True, ddp_bucket_cap_mb=100)."
                                    if TRACK3_R5C_DDP_CANDIDATE_DEFAULT
                                    and track_key == "track3_hh_reduced"
                                    and nodes == 4
                                    and variant_name == "R1_accum2_nosync"
                                    else ""
                                )
                            ),
                        }
                    )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Scaling Redesign Matrix ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- config root: `{CONFIG_ROOT}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Design",
        "- Phase R1 initial 1-node communication-reduction screen",
        "- Phase R1b immediate 2-node / 4-node scaling probe using baseline vs accum2 only",
        "- one seed per track for the first batch",
        "- variants: baseline, accum2, accum4, accum8 (1n), then baseline/accum2 (2n/4n)",
        (
            "- Track3 4n R1_accum2_nosync rows default to the active R5c candidate"
            " (`ddp_gradient_as_bucket_view=True`, `ddp_bucket_cap_mb=100`) unless"
            " `ENABLE_TRACK3_R5C_DDP_CANDIDATE=0`."
            if TRACK3_R5C_DDP_CANDIDATE_DEFAULT
            else "- Track3 R5c candidate default is disabled via `ENABLE_TRACK3_R5C_DDP_CANDIDATE=0`."
        ),
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
