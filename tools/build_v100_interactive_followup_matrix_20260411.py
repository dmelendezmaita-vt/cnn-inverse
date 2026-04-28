#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

DATE_TAG = "20260411_v100_interactive"
DATE_LABEL = "2026-04-11"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
PLANS = OUT_ROOT / "plans"
SHARED_DATA = OUT_ROOT / "shared_data"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"v100_interactive_followup_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"v100_interactive_followup_notes_{DATE_TAG}.md"
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


def create_config(base_rel: str, out_rel: str, updates: Dict[str, str]) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    txt = read_text(base_path)
    allow_add_training = {"loss_type", "huber_beta", "grad_clip_max_norm", "reduce_loss_for_logging", "use_amp", "amp_dtype"}
    allow_add_runconfig = {"save_diagnostic_plots"}
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


def build_shared_dir(track_key: str, data_prefix: str) -> str:
    return str(SHARED_DATA / track_key / data_prefix)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    PLANS.mkdir(parents=True, exist_ok=True)
    SHARED_DATA.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_meta = {
        "track3_hh_reduced": {
            "short": "t3",
            "base": "src/pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            "prefix": "reduced_data",
        },
        "track4_hh_full": {
            "short": "t4",
            "base": "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "prefix": "concatenated_data",
        },
    }

    strategies = {
        "S0_baseline": {
            "description": "Current large-matrix path: copy_to_node, loss sync enabled, FP32, full artifacts.",
            "data_access_mode": "copy_to_node",
            "shared_data_dir": "",
            "reduce_loss_for_logging": "True",
            "use_amp": "False",
            "amp_dtype": "float16",
            "save_predictions": "test",
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "10",
        },
        "S1_nolosssync": {
            "description": "Disable per-batch loss all-reduce used only for logging.",
            "data_access_mode": "copy_to_node",
            "shared_data_dir": "",
            "reduce_loss_for_logging": "False",
            "use_amp": "False",
            "amp_dtype": "float16",
            "save_predictions": "test",
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "10",
        },
        "S2_shareddata_nolosssync": {
            "description": "Reuse extracted shared dataset and disable loss-sync reduction.",
            "data_access_mode": "direct_tar",
            "shared_data_dir": "AUTO",
            "reduce_loss_for_logging": "False",
            "use_amp": "False",
            "amp_dtype": "float16",
            "save_predictions": "test",
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "10",
        },
        "S3_amp_shareddata_nolosssync": {
            "description": "Enable AMP on top of shared-data and no-loss-sync path.",
            "data_access_mode": "direct_tar",
            "shared_data_dir": "AUTO",
            "reduce_loss_for_logging": "False",
            "use_amp": "True",
            "amp_dtype": "float16",
            "save_predictions": "test",
            "save_diagnostic_plots": "True",
            "save_checkpoints_epochs": "10",
        },
        "S4_leanartifacts_amp_shareddata_nolosssync": {
            "description": "AMP + shared-data + no-loss-sync, with reduced sweep artifacts.",
            "data_access_mode": "direct_tar",
            "shared_data_dir": "AUTO",
            "reduce_loss_for_logging": "False",
            "use_amp": "True",
            "amp_dtype": "float16",
            "save_predictions": "none",
            "save_diagnostic_plots": "False",
            "save_checkpoints_epochs": "null",
        },
    }

    seeds = [971, 972, 973]
    base_lr = 3.0e-4
    base_epochs = 400

    config_cache: Dict[Tuple[str, str, str, int, int, int], str] = {}
    rows: List[Dict[str, str]] = []
    row_id = 0

    def ensure_config(
        *,
        phase: str,
        strategy_id: str,
        track_key: str,
        policy: str,
        nodes: int,
        batch_profile: str,
        seed: int,
        features_sub_length: int,
    ) -> str:
        key = (phase, strategy_id, track_key, policy, nodes, seed, features_sub_length, batch_profile)
        if key in config_cache:
            return config_cache[key]

        meta = track_meta[track_key]
        strategy = strategies[strategy_id]
        if policy == "strong":
            gbs = int(batch_profile.replace("strong_gbs", ""))
            ntrain = 4096 if track_key == "track3_hh_reduced" else 8192
            policy_slug = f"strong_gbs{gbs}"
        else:
            prb = int(batch_profile.replace("weak_prb", ""))
            gbs = prb * nodes * 2
            ntrain_base = 1024 if track_key == "track3_hh_reduced" else 2048
            ntrain = ntrain_base * nodes
            policy_slug = f"weak_prb{prb}"

        cfg_rel = (
            f"src/pytorch/configs/reruns_{DATE_TAG}/{phase}/{strategy_id}/{track_key}/"
            f"params_{meta['short']}_{policy_slug}_n{nodes}_lr{slug_float(base_lr)}_"
            f"sub{features_sub_length}_s{seed}.yaml"
        )

        updates = {
            "description": f'"{track_key} {policy_slug} {strategy_id} interactive follow-up {DATE_LABEL}"',
            "learning_rate": fmt_lr(base_lr),
            "epochs": str(base_epochs),
            "init_learning_rate": fmt_lr(base_lr / 20.0),
            "linear_epochs": "40",
            "constant_epochs": "40",
            "final_learning_rate": fmt_lr(base_lr / 200.0),
            "features_normalize": "True",
            "features_scale_cache_enabled": "True",
            "features_sub_begin_random": "True",
            "features_sub_begin_random_eval": "False",
            "features_sub_length": str(features_sub_length),
            "global_train_batch_size": str(gbs),
            "train_batch_size": str(gbs),
            "Ntrain": str(ntrain),
            "random_seed": str(seed),
            "loss_type": "huber",
            "huber_beta": "1.0",
            "grad_clip_max_norm": "1.0",
            "reduce_loss_for_logging": strategy["reduce_loss_for_logging"],
            "use_amp": strategy["use_amp"],
            "amp_dtype": f'"{strategy["amp_dtype"]}"',
            "save_predictions": f'"{strategy["save_predictions"]}"',
            "save_diagnostic_plots": strategy["save_diagnostic_plots"],
            "save_checkpoints_epochs": strategy["save_checkpoints_epochs"],
            "save_dir": f'"runs/{DATE_TAG}"',
        }
        maybe_apply_split_array_cache(updates)
        create_config(meta["base"], cfg_rel, updates)
        config_cache[key] = cfg_rel
        return cfg_rel

    def add_row(
        *,
        phase: str,
        phase_label: str,
        launch_mode: str,
        launch_group: str,
        track_key: str,
        policy: str,
        nodes: int,
        batch_profile: str,
        features_sub_length: int,
        seed: int,
        strategy_id: str,
        repeat_index: int,
        family: str,
        notes: str,
    ) -> None:
        nonlocal row_id
        meta = track_meta[track_key]
        strategy = strategies[strategy_id]
        cfg_rel = ensure_config(
            phase=phase,
            strategy_id=strategy_id,
            track_key=track_key,
            policy=policy,
            nodes=nodes,
            batch_profile=batch_profile,
            seed=seed,
            features_sub_length=features_sub_length,
        )

        if policy == "strong":
            gbs = int(batch_profile.replace("strong_gbs", ""))
            ntrain = 4096 if track_key == "track3_hh_reduced" else 8192
        else:
            prb = int(batch_profile.replace("weak_prb", ""))
            gbs = prb * nodes * 2
            ntrain = (1024 if track_key == "track3_hh_reduced" else 2048) * nodes

        shared_dir = ""
        if strategy["shared_data_dir"] == "AUTO":
            shared_dir = build_shared_dir(track_key, meta["prefix"])

        row_id += 1
        task_slug = (
            f"{phase}_{strategy_id}_{meta['short']}_{policy}_{batch_profile}_"
            f"n{nodes}_sub{features_sub_length}_s{seed}_r{repeat_index:02d}"
        )
        rows.append(
            {
                "row_id": f"v100if_{row_id:04d}",
                "phase": phase,
                "phase_label": phase_label,
                "launch_mode": launch_mode,
                "launch_group": launch_group,
                "family": family,
                "track_key": track_key,
                "policy": policy,
                "nodes": str(nodes),
                "gpus_per_node": "2",
                "cpus_per_node": "24",
                "gpu_type": "V100",
                "batch_profile": batch_profile,
                "learning_rate": fmt_lr(base_lr),
                "features_sub_length": str(features_sub_length),
                "Ntrain": str(ntrain),
                "global_train_batch_size": str(gbs),
                "seed": str(seed),
                "repeat_index": str(repeat_index),
                "strategy_id": strategy_id,
                "strategy_description": strategy["description"],
                "params_file": cfg_rel,
                "tar_path": meta["tar"],
                "data_prefix": meta["prefix"],
                "curr": "0.1",
                "data_access_mode": strategy["data_access_mode"],
                "shared_data_dir": shared_dir,
                "reduce_loss_for_logging": strategy["reduce_loss_for_logging"],
                "use_amp": strategy["use_amp"],
                "amp_dtype": strategy["amp_dtype"],
                "save_predictions": strategy["save_predictions"],
                "save_diagnostic_plots": strategy["save_diagnostic_plots"],
                "save_checkpoints_epochs": strategy["save_checkpoints_epochs"],
                "task_slug": task_slug,
                "notes": notes,
            }
        )

    # -----------------------------
    # Phase A1: baseline scaling latency
    # -----------------------------
    phase = "phaseA_baseline_latency"
    anchors = [
        ("track3_hh_reduced", "strong", "strong_gbs128", 2000, "strong_anchor"),
        ("track3_hh_reduced", "weak", "weak_prb32", 2000, "weak_anchor"),
        ("track4_hh_full", "strong", "strong_gbs128", 2000, "strong_anchor"),
        ("track4_hh_full", "weak", "weak_prb32", 2000, "weak_anchor"),
    ]
    for track_key, policy, batch_profile, sub_len, family in anchors:
        for nodes in [1, 2, 4]:
            for rep_idx, seed in enumerate(seeds, start=1):
                add_row(
                    phase=phase,
                    phase_label="Phase A baseline latency",
                    launch_mode="dedicated",
                    launch_group="",
                    track_key=track_key,
                    policy=policy,
                    nodes=nodes,
                    batch_profile=batch_profile,
                    features_sub_length=sub_len,
                    seed=seed,
                    strategy_id="S0_baseline",
                    repeat_index=rep_idx,
                    family=family,
                    notes="Baseline end-to-end latency and scaling anchor.",
                )

    # -----------------------------
    # Phase A2: throughput geometry
    # -----------------------------
    phase = "phaseA_throughput_geometry"
    for rep_idx, seed in enumerate(seeds, start=1):
        group_4x1 = f"phaseA_4x1n_seed{seed}"
        for track_key, policy, batch_profile, sub_len, family in anchors:
            add_row(
                phase=phase,
                phase_label="Phase A throughput geometry",
                launch_mode="concurrent_4x1n",
                launch_group=group_4x1,
                track_key=track_key,
                policy=policy,
                nodes=1,
                batch_profile=batch_profile,
                features_sub_length=sub_len,
                seed=seed,
                strategy_id="S0_baseline",
                repeat_index=rep_idx,
                family=family,
                notes="One member of a 4x1-node concurrent throughput block.",
            )

        group_2x2 = f"phaseA_2x2n_seed{seed}"
        for track_key in ["track3_hh_reduced", "track4_hh_full"]:
            add_row(
                phase=phase,
                phase_label="Phase A throughput geometry",
                launch_mode="concurrent_2x2n",
                launch_group=group_2x2,
                track_key=track_key,
                policy="strong",
                nodes=2,
                batch_profile="strong_gbs128",
                features_sub_length=2000,
                seed=seed,
                strategy_id="S0_baseline",
                repeat_index=rep_idx,
                family="strong_anchor",
                notes="One member of a 2x2-node concurrent throughput block.",
            )

    # -----------------------------
    # Phase B: code-path screening on strong anchors
    # -----------------------------
    phase = "phaseB_codepath_screen"
    for track_key in ["track3_hh_reduced", "track4_hh_full"]:
        for nodes in [1, 4]:
            for strategy_id in strategies:
                for rep_idx, seed in enumerate(seeds, start=1):
                    add_row(
                        phase=phase,
                        phase_label="Phase B code-path screening",
                        launch_mode="dedicated",
                        launch_group="",
                        track_key=track_key,
                        policy="strong",
                        nodes=nodes,
                        batch_profile="strong_gbs128",
                        features_sub_length=2000,
                        seed=seed,
                        strategy_id=strategy_id,
                        repeat_index=rep_idx,
                        family="strong_anchor",
                        notes="Screen code-path strategies on strong-scaling anchors.",
                    )

    fieldnames = [
        "row_id",
        "phase",
        "phase_label",
        "launch_mode",
        "launch_group",
        "family",
        "track_key",
        "policy",
        "nodes",
        "gpus_per_node",
        "cpus_per_node",
        "gpu_type",
        "batch_profile",
        "learning_rate",
        "features_sub_length",
        "Ntrain",
        "global_train_batch_size",
        "seed",
        "repeat_index",
        "strategy_id",
        "strategy_description",
        "params_file",
        "tar_path",
        "data_prefix",
        "curr",
        "data_access_mode",
        "shared_data_dir",
        "reduce_loss_for_logging",
        "use_amp",
        "amp_dtype",
        "save_predictions",
        "save_diagnostic_plots",
        "save_checkpoints_epochs",
        "task_slug",
        "notes",
    ]
    with MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    phase_counts = Counter(row["phase"] for row in rows)
    launch_counts = Counter(row["launch_mode"] for row in rows)
    strat_counts = Counter(row["strategy_id"] for row in rows)
    notes = [
        f"# V100 Interactive Follow-up Matrix ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- config root: `{CONFIG_ROOT}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Phase Counts",
    ]
    for k, v in sorted(phase_counts.items()):
        notes.append(f"- `{k}`: `{v}`")
    notes.extend(["", "## Launch Modes"])
    for k, v in sorted(launch_counts.items()):
        notes.append(f"- `{k}`: `{v}`")
    notes.extend(["", "## Strategy Counts"])
    for k, v in sorted(strat_counts.items()):
        notes.append(f"- `{k}`: `{v}`")
    notes.extend(
        [
            "",
            "## Strategy Summary",
            "- `S0_baseline`: current path",
            "- `S1_nolosssync`: disable batch-level logging all-reduce",
            "- `S2_shareddata_nolosssync`: reuse shared extracted dataset",
            "- `S3_amp_shareddata_nolosssync`: add AMP to S2",
            "- `S4_leanartifacts_amp_shareddata_nolosssync`: reduce sweep artifact overhead on top of S3",
        ]
    )
    write_text(NOTES_MD, "\n".join(notes) + "\n")

    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Wrote notes:  {NOTES_MD}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
