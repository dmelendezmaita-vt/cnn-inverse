#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260416_v100_hh_datapath_redesign_r4"
DATE_LABEL = "2026-04-16"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_hh_datapath_redesign_r4_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_hh_datapath_redesign_r4_notes_{DATE_TAG}.md"
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


def create_config(base_rel: str, out_rel: str, updates: Dict[str, str]) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    txt = read_text(base_path)
    allow_add_data = {
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "dataloader_pin_memory",
        "features_scale_cache_enabled",
        "split_array_cache_enabled",
    }
    allow_add_runconfig = {
        "skip_final_distributed_barrier",
        "post_local_gradient_allreduce",
        "post_local_sgd_start_local_sgd_iter",
        "post_local_sgd_warmup_steps",
        "post_local_sgd_period",
        "post_local_sgd_enabled",
        "save_diagnostic_plots",
    }
    for k, v in updates.items():
        try:
            txt = set_key(txt, k, v)
        except ValueError:
            if k in allow_add_data:
                txt = add_key_under_section(txt, "data", k, v)
            elif k in allow_add_runconfig:
                txt = add_key_under_section(txt, "runconfig", k, v)
            else:
                raise
    write_text(out_path, txt)
    return out_rel


def maybe_apply_split_array_cache(updates: Dict[str, str]) -> None:
    if not SPLIT_ARRAY_CACHE_OPT_IN:
        return
    updates["split_array_cache_enabled"] = "True"


def source_shared_dir(track_key: str, prefix: str) -> str:
    return str(
        IMPORTANT
        / "optimization_track_20260411_v100_interactive"
        / "shared_data"
        / track_key
        / prefix
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_meta = {
        "track3_hh_reduced": {
            "short": "t3",
            "prefix": "reduced_data",
            "tar": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            "ntrain": "4096",
            "batch_profile": "strong_gbs128",
            "curr": "0.1",
        },
        "track4_hh_full": {
            "short": "t4",
            "prefix": "concatenated_data",
            "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "ntrain": "8192",
            "batch_profile": "strong_gbs128",
            "curr": "0.1",
        },
    }

    variants = [
        {
            "phase": "phaseR4_datapath_core",
            "family": "datapath_core_verification",
            "launch_mode": "dedicated",
            "track_keys": ["track3_hh_reduced", "track4_hh_full"],
            "nodes": 1,
            "strategy_id": "S4_leanartifacts_amp_shareddata_nolosssync",
            "save_predictions_by_track": {
                "track3_hh_reduced": "none",
                "track4_hh_full": "none",
            },
            "strategy_description": "Best operational-cleanup 1n baseline rerun under the R4 data-path instrumentation path.",
            "base_rel_tpl": (
                "src/pytorch/configs/reruns_20260411_v100_interactive/phaseB_codepath_screen/"
                "S4_leanartifacts_amp_shareddata_nolosssync/{track_key}/"
                "params_{short}_strong_gbs128_n1_lr3p0em4_sub2000_s{seed}.yaml"
            ),
            "notes": "R4 core: repeat the best 1n operational-cleanup recipe under the explicit shared-data path and new loader instrumentation.",
        },
        {
            "phase": "phaseR4_datapath_core",
            "family": "datapath_core_verification",
            "launch_mode": "dedicated",
            "track_keys": ["track3_hh_reduced", "track4_hh_full"],
            "nodes": 4,
            "strategy_id": "R1_accum2_nosync",
            "save_predictions_by_track": {
                "track3_hh_reduced": "none",
                "track4_hh_full": "test",
            },
            "strategy_description": "Current 4n production-safe baseline from R2 rerun under the R4 data-path instrumentation path.",
            "base_rel_tpl": (
                "src/pytorch/configs/reruns_20260413_v100_hh_scaling_replication_r2/{track_key}/"
                "params_{short}_R1_accum2_nosync_n4_s{seed}.yaml"
            ),
            "notes": "R4 core: repeat the current best 4n recipe from R2 to measure what the data-path change does to end-to-end elapsed.",
        },
        {
            "phase": "phaseR4_datapath_sidecar",
            "family": "datapath_track3_sidecar",
            "launch_mode": "dedicated",
            "track_keys": ["track3_hh_reduced"],
            "nodes": 4,
            "strategy_id": "R3_postlocalsgd_p8_w100",
            "save_predictions_by_track": {
                "track3_hh_reduced": "none",
            },
            "strategy_description": "Optional track3-only R3 sidecar under the R4 data path.",
            "base_rel_tpl": (
                "src/pytorch/configs/reruns_20260413_v100_hh_postlocalsgd_r3_stable/{track_key}/"
                "params_{short}_R3_postlocalsgd_p8_w100_n4_s{seed}.yaml"
            ),
            "extra_updates": {
                "skip_final_distributed_barrier": "false",
                "save_checkpoints_epochs": "50",
            },
            "notes": "Optional R4 sidecar: rerun the most interesting track3 R3 candidate under the new data path, preserving recovered runtime defaults.",
        },
    ]

    rows: List[Dict[str, str]] = []
    row_num = 0

    for variant in variants:
        for track_key in variant["track_keys"]:
            meta = track_meta[track_key]
            for seed in [971, 972, 973]:
                row_num += 1
                nodes = int(variant["nodes"])
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                    f"params_{meta['short']}_{variant['strategy_id']}_n{nodes}_s{seed}.yaml"
                )
                base_rel = variant["base_rel_tpl"].format(track_key=track_key, short=meta["short"], seed=seed)
                updates = {
                    "description": f'"{track_key} {variant["strategy_id"]} datapath redesign R4 {DATE_LABEL}"',
                    "save_dir": f'"runs/{DATE_TAG}"',
                    "dataloader_num_workers": "0",
                    "dataloader_persistent_workers": "False",
                    "dataloader_prefetch_factor": "null",
                    "dataloader_pin_memory": "False",
                    "features_scale_cache_enabled": "True",
                }
                maybe_apply_split_array_cache(updates)
                updates.update(variant.get("extra_updates", {}))
                create_config(base_rel, cfg_rel, updates)
                rows.append(
                    {
                        "row_id": f"v100r4_{row_num:04d}",
                        "phase": variant["phase"],
                        "launch_mode": variant["launch_mode"],
                        "launch_group": "",
                        "family": variant["family"],
                        "track_key": track_key,
                        "policy": "strong",
                        "nodes": str(nodes),
                        "gpus_per_node": "2",
                        "cpus_per_node": "24",
                        "gpu_type": "V100",
                        "batch_profile": meta["batch_profile"],
                        "learning_rate": "3.0e-4",
                        "features_sub_length": "2000",
                        "Ntrain": meta["ntrain"],
                        "global_train_batch_size": "128",
                        "seed": str(seed),
                        "strategy_id": variant["strategy_id"],
                        "strategy_description": variant["strategy_description"],
                        "params_file": cfg_rel,
                        "tar_path": meta["tar"],
                        "data_prefix": meta["prefix"],
                        "curr": meta["curr"],
                        "data_access_mode": "direct_tar",
                        "shared_data_dir": source_shared_dir(track_key, meta["prefix"]),
                        "save_predictions": variant["save_predictions_by_track"][track_key],
                        "save_diagnostic_plots": "inherit_from_config",
                        "save_checkpoints_epochs": "inherit_from_config",
                        "task_slug": f"{track_key}_{variant['strategy_id']}_n{nodes}_s{seed}",
                        "notes": variant["notes"],
                    }
                )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    notes = [
        f"# V100 HH Datapath Redesign R4 Matrix ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- config root: `{CONFIG_ROOT}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Design",
        "- Core rows rerun the best 1n operational baseline and current best 4n recipe per track",
        "- Track 3 baseline: S4_leanartifacts_amp_shareddata_nolosssync (1n)",
        "- Track 3 current 4n baseline: R1_accum2_nosync",
        "- Track 4 baseline: S4_leanartifacts_amp_shareddata_nolosssync (1n)",
        "- Track 4 current 4n baseline: R1_accum2_nosync",
        "- Optional sidecar reruns track3 R3_postlocalsgd_p8_w100 (4n) with recovered runtime defaults",
        "- Shared extracted data path is used for all rows via shared_data_dir",
        "- New loader instrumentation in src/pytorch/data.py will expose source kind and timing breakdown",
        "",
        "## Totals",
        "- Core rows: 12",
        "- Optional sidecar rows: 3",
        "- Total rows: 15",
    ]
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
