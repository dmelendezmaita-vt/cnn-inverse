#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260402"
DATE_LABEL = "2026-04-02"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
SUB_ROOT = OUT_ROOT / "submission"
TABLES = SUB_ROOT / "tables"
NOTES = SUB_ROOT / "notes"
OPT_TABLES = OUT_ROOT / "tables"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}_opttrack"

PLAN_CSV = TABLES / f"optimization_submission_matrix_{DATE_TAG}.csv"
PLAN_MD = NOTES / f"optimization_submission_notes_{DATE_TAG}.md"

BASELINE_SRC_ROOT = IMPORTANT / f"advisor_packet_{DATE_TAG}" / "tables"
BASELINE_COPY = {
    "track3_track4_scalability_training_only.csv": "baseline_track3_track4_scalability_training_only.csv",
    "rerun_metric_summary_20260402.csv": "baseline_rerun_metric_summary_20260402.csv",
    "loss_spike_summary_training_only.csv": "baseline_loss_spike_summary_training_only.csv",
}


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def set_key(text: str, key: str, value: str) -> str:
    # Keep replacement on a single YAML line; do not let whitespace span newlines.
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
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s


def slug_float(v: float) -> str:
    s = fmt_lr(v)
    s = s.replace(".", "p").replace("+", "").replace("-", "m")
    return s


def create_config(base_rel: str, out_rel: str, updates: Dict[str, str]) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    txt = read_text(base_path)
    allow_add_training = {"loss_type", "huber_beta", "grad_clip_max_norm"}
    for k, v in updates.items():
        try:
            txt = set_key(txt, k, v)
        except ValueError:
            if k in allow_add_training:
                txt = add_key_under_section(txt, "training", k, v)
            else:
                raise
    write_text(out_path, txt)
    return out_rel


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    OPT_TABLES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    # Phase 0: baseline snapshot copy
    copied = []
    missing = []
    for src_name, dst_name in BASELINE_COPY.items():
        src = BASELINE_SRC_ROOT / src_name
        dst = OPT_TABLES / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((src_name, dst_name))
        else:
            missing.append(src_name)

    track_meta = {
        "track3_hh_reduced": {
            "short": "t3",
            "base": "src/pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            "prefix": "reduced_data",
            "mem": 128,
        },
        "track4_hh_full": {
            "short": "t4",
            "base": "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
            "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "prefix": "concatenated_data",
            "mem": 192,
        },
    }

    rows: List[Dict[str, str]] = []

    def add_row(
        *,
        phase: str,
        submission_lane: str,
        concurrency_group: str,
        rerun_group: str,
        track_key: str,
        sweep_variable: str,
        sweep_value: str,
        seed: int,
        job_name: str,
        params_file: str,
        tar_path: str,
        data_prefix: str,
        nodes: int,
        gpus_per_node: int,
        cpus_per_node: int,
        mem_gib: int,
        partition: str = "a30_normal_q",
        qos: str = "fal_a30_normal_short",
        time_limit: str = "01:30:00",
        gpu_token: str = "a30",
        gpu: str = "A30",
        cluster: str = "Falcon",
        data_access_mode: str = "copy_to_node",
        report_interval_sec: int = 15,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "phase": phase,
                "submission_lane": submission_lane,
                "concurrency_group": concurrency_group,
                "rerun_group": rerun_group,
                "track_key": track_key,
                "sweep_variable": sweep_variable,
                "sweep_value": sweep_value,
                "seed": str(seed),
                "job_name": job_name,
                "params_file": params_file,
                "tar_path": tar_path,
                "data_prefix": data_prefix,
                "nodes": str(nodes),
                "gpus_per_node": str(gpus_per_node),
                "cpus_per_node": str(cpus_per_node),
                "mem_gib": str(mem_gib),
                "partition": partition,
                "qos": qos,
                "time_limit": time_limit,
                "gpu_token": gpu_token,
                "gpu": gpu,
                "cluster": cluster,
                "data_access_mode": data_access_mode,
                "report_interval_sec": str(report_interval_sec),
                "notes": notes,
            }
        )

    # -----------------------------
    # Phase 1: stability-first optimizer lane
    # -----------------------------
    phase1_lrs = [1.0e-3, 3.0e-4, 1.0e-4]
    phase1_epochs = 400
    phase1_seeds = [901, 902, 903]
    for track_key, meta in track_meta.items():
        for lr in phase1_lrs:
            lr_slug = slug_float(lr)
            for seed in phase1_seeds:
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}_opttrack/{track_key}/phase1_stability/"
                    f"params_{meta['short']}_lr{lr_slug}_ep{phase1_epochs}_s{seed}.yaml"
                )
                create_config(
                    meta["base"],
                    cfg_rel,
                    {
                        "description": f'"{track_key} optimization phase1 stability sweep {DATE_LABEL}"',
                        "learning_rate": fmt_lr(lr),
                        "epochs": str(phase1_epochs),
                        "init_learning_rate": fmt_lr(lr / 20.0),
                        "linear_epochs": "40",
                        "constant_epochs": "40",
                        "final_learning_rate": fmt_lr(lr / 200.0),
                        "random_seed": str(seed),
                    },
                )
                add_row(
                    phase="phase1_stability",
                    submission_lane="phase1_stability",
                    concurrency_group=f"a30_mem{meta['mem']}",
                    rerun_group="opt_phase1_stability_track3_track4",
                    track_key=track_key,
                    sweep_variable="learning_rate_scheduler_epochs",
                    sweep_value=(
                        f"learning_rate={fmt_lr(lr)}|epochs={phase1_epochs}|"
                        f"scheduler=warmup40_const40_cosine"
                    ),
                    seed=seed,
                    job_name=f"trn_opt1_{meta['short']}_lr{lr_slug}_s{seed}",
                    params_file=cfg_rel,
                    tar_path=meta["tar"],
                    data_prefix=meta["prefix"],
                    nodes=1,
                    gpus_per_node=4,
                    cpus_per_node=48,
                    mem_gib=int(meta["mem"]),
                    notes="phase1 stability-first optimizer sweep",
                )

    # Shared base for phases 2-5 (provisional best candidate before phase1 review)
    # learning_rate=3e-4, epochs=400, scheduler enabled.
    base_lr = 3.0e-4
    base_epochs = 400

    # -----------------------------
    # Phase 2: feature/input conditioning lane
    # -----------------------------
    phase2_norm = ["False", "True"]
    phase2_sub_rand = ["True", "False"]
    phase2_sub_len = [1000, 2000, 4000]
    phase2_seeds = [911, 912, 913]
    for track_key, meta in track_meta.items():
        for norm in phase2_norm:
            for sbr in phase2_sub_rand:
                for sub_len in phase2_sub_len:
                    for seed in phase2_seeds:
                        cfg_rel = (
                            f"src/pytorch/configs/reruns_{DATE_TAG}_opttrack/{track_key}/phase2_features/"
                            f"params_{meta['short']}_norm{norm.lower()}_sbr{sbr.lower()}_sub{sub_len}_s{seed}.yaml"
                        )
                        create_config(
                            meta["base"],
                            cfg_rel,
                            {
                                "description": f'"{track_key} optimization phase2 feature conditioning {DATE_LABEL}"',
                                "learning_rate": fmt_lr(base_lr),
                                "epochs": str(base_epochs),
                                "init_learning_rate": fmt_lr(base_lr / 20.0),
                                "linear_epochs": "40",
                                "constant_epochs": "40",
                                "final_learning_rate": fmt_lr(base_lr / 200.0),
                                "features_normalize": norm,
                                "features_sub_begin_random": sbr,
                                "features_sub_length": str(sub_len),
                                "random_seed": str(seed),
                            },
                        )
                        add_row(
                            phase="phase2_features",
                            submission_lane="phase2_features",
                            concurrency_group=f"a30_mem{meta['mem']}",
                            rerun_group="opt_phase2_feature_conditioning_track3_track4",
                            track_key=track_key,
                            sweep_variable="features_normalize_subbegin_sub_length",
                            sweep_value=(
                                f"features_normalize={norm}|features_sub_begin_random={sbr}|"
                                f"features_sub_length={sub_len}"
                            ),
                            seed=seed,
                            job_name=(
                                f"trn_opt2_{meta['short']}_n{norm[0].lower()}_r{sbr[0].lower()}_"
                                f"sub{sub_len}_s{seed}"
                            ),
                            params_file=cfg_rel,
                            tar_path=meta["tar"],
                            data_prefix=meta["prefix"],
                            nodes=1,
                            gpus_per_node=4,
                            cpus_per_node=48,
                            mem_gib=int(meta["mem"]),
                            notes="phase2 feature/input conditioning sweep",
                        )

    # -----------------------------
    # Phase 3: batch/effective-gradient lane
    # -----------------------------
    phase3_batches = [16, 32, 64]
    phase3_seeds = [921, 922, 923]
    for track_key, meta in track_meta.items():
        for gbs in phase3_batches:
            for seed in phase3_seeds:
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}_opttrack/{track_key}/phase3_batch/"
                    f"params_{meta['short']}_gbs{gbs}_s{seed}.yaml"
                )
                create_config(
                    meta["base"],
                    cfg_rel,
                    {
                        "description": f'"{track_key} optimization phase3 batch sweep {DATE_LABEL}"',
                        "learning_rate": fmt_lr(base_lr),
                        "epochs": str(base_epochs),
                        "init_learning_rate": fmt_lr(base_lr / 20.0),
                        "linear_epochs": "40",
                        "constant_epochs": "40",
                        "final_learning_rate": fmt_lr(base_lr / 200.0),
                        "features_normalize": "True",
                        "global_train_batch_size": str(gbs),
                        "train_batch_size": str(gbs),
                        "random_seed": str(seed),
                    },
                )
                add_row(
                    phase="phase3_batch",
                    submission_lane="phase3_batch",
                    concurrency_group=f"a30_mem{meta['mem']}",
                    rerun_group="opt_phase3_batch_gradient_track3_track4",
                    track_key=track_key,
                    sweep_variable="global_train_batch_size",
                    sweep_value=f"global_train_batch_size={gbs}",
                    seed=seed,
                    job_name=f"trn_opt3_{meta['short']}_gbs{gbs}_s{seed}",
                    params_file=cfg_rel,
                    tar_path=meta["tar"],
                    data_prefix=meta["prefix"],
                    nodes=1,
                    gpus_per_node=4,
                    cpus_per_node=48,
                    mem_gib=int(meta["mem"]),
                    notes="phase3 batch/effective-gradient sweep",
                )

    # -----------------------------
    # Phase 4: Ntrain scale lane
    # -----------------------------
    phase4_ntrain = [4096, 8192, 16384]
    phase4_seeds = [931, 932, 933]
    for track_key, meta in track_meta.items():
        for ntrain in phase4_ntrain:
            for seed in phase4_seeds:
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}_opttrack/{track_key}/phase4_ntrain/"
                    f"params_{meta['short']}_ntr{ntrain}_s{seed}.yaml"
                )
                create_config(
                    meta["base"],
                    cfg_rel,
                    {
                        "description": f'"{track_key} optimization phase4 Ntrain scale {DATE_LABEL}"',
                        "learning_rate": fmt_lr(base_lr),
                        "epochs": str(base_epochs),
                        "init_learning_rate": fmt_lr(base_lr / 20.0),
                        "linear_epochs": "40",
                        "constant_epochs": "40",
                        "final_learning_rate": fmt_lr(base_lr / 200.0),
                        "features_normalize": "True",
                        "global_train_batch_size": "32",
                        "train_batch_size": "32",
                        "Ntrain": str(ntrain),
                        "random_seed": str(seed),
                    },
                )
                add_row(
                    phase="phase4_ntrain",
                    submission_lane="phase4_ntrain",
                    concurrency_group=f"a30_mem{meta['mem']}",
                    rerun_group="opt_phase4_ntrain_scale_track3_track4",
                    track_key=track_key,
                    sweep_variable="Ntrain",
                    sweep_value=f"Ntrain={ntrain}",
                    seed=seed,
                    job_name=f"trn_opt4_{meta['short']}_ntr{ntrain}_s{seed}",
                    params_file=cfg_rel,
                    tar_path=meta["tar"],
                    data_prefix=meta["prefix"],
                    nodes=1,
                    gpus_per_node=4,
                    cpus_per_node=48,
                    mem_gib=int(meta["mem"]),
                    notes="phase4 Ntrain scale sweep",
                )

    # -----------------------------
    # Phase 5: robustness lane
    # -----------------------------
    phase5_variants = [
        {
            "name": "mse_clip1p0",
            "updates": {
                "loss_type": "mse",
                "grad_clip_max_norm": "1.0",
            },
        },
        {
            "name": "huber_clip1p0",
            "updates": {
                "loss_type": "huber",
                "huber_beta": "1.0",
                "grad_clip_max_norm": "1.0",
            },
        },
    ]
    phase5_seeds = [941, 942, 943]
    for track_key, meta in track_meta.items():
        for var in phase5_variants:
            for seed in phase5_seeds:
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}_opttrack/{track_key}/phase5_robustness/"
                    f"params_{meta['short']}_{var['name']}_s{seed}.yaml"
                )
                updates = {
                    "description": f'"{track_key} optimization phase5 robustness {var["name"]} {DATE_LABEL}"',
                    "learning_rate": fmt_lr(base_lr),
                    "epochs": str(base_epochs),
                    "init_learning_rate": fmt_lr(base_lr / 20.0),
                    "linear_epochs": "40",
                    "constant_epochs": "40",
                    "final_learning_rate": fmt_lr(base_lr / 200.0),
                    "features_normalize": "True",
                    "features_sub_length": "2000",
                    "global_train_batch_size": "32",
                    "train_batch_size": "32",
                    "random_seed": str(seed),
                }
                updates.update(var["updates"])
                create_config(meta["base"], cfg_rel, updates)
                add_row(
                    phase="phase5_robustness",
                    submission_lane="phase5_robustness",
                    concurrency_group=f"a30_mem{meta['mem']}",
                    rerun_group="opt_phase5_robustness_track3_track4",
                    track_key=track_key,
                    sweep_variable="loss_type_grad_clip",
                    sweep_value=f"variant={var['name']}",
                    seed=seed,
                    job_name=f"trn_opt5_{meta['short']}_{var['name']}_s{seed}",
                    params_file=cfg_rel,
                    tar_path=meta["tar"],
                    data_prefix=meta["prefix"],
                    nodes=1,
                    gpus_per_node=4,
                    cpus_per_node=48,
                    mem_gib=int(meta["mem"]),
                    notes="phase5 robustness sweep",
                )

    fieldnames = [
        "phase",
        "submission_lane",
        "concurrency_group",
        "rerun_group",
        "track_key",
        "sweep_variable",
        "sweep_value",
        "seed",
        "job_name",
        "params_file",
        "tar_path",
        "data_prefix",
        "nodes",
        "gpus_per_node",
        "cpus_per_node",
        "mem_gib",
        "partition",
        "qos",
        "time_limit",
        "gpu_token",
        "gpu",
        "cluster",
        "data_access_mode",
        "report_interval_sec",
        "notes",
    ]

    with PLAN_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    phase_counts = Counter(r["phase"] for r in rows)
    group_counts = Counter(r["rerun_group"] for r in rows)
    track_counts = Counter(r["track_key"] for r in rows)
    lane_counts = Counter(r["submission_lane"] for r in rows)

    lines = [
        f"# Optimization Submission Matrix ({DATE_LABEL})",
        "",
        "Training-only jobs (testing runs excluded).",
        "",
        "## Scope",
        "- This optimization lane is isolated from advisor packets.",
        "- No prior report folders are overwritten.",
        "",
        "## Baseline Snapshots Copied (Phase 0)",
    ]
    if copied:
        for src_name, dst_name in copied:
            lines.append(f"- {src_name} -> {dst_name}")
    if missing:
        for src_name in missing:
            lines.append(f"- MISSING baseline source: {src_name}")

    lines.extend([
        "",
        "## Totals",
        f"- Total jobs: {len(rows)}",
    ])
    for k in sorted(phase_counts):
        lines.append(f"- {k}: {phase_counts[k]} jobs")

    lines.extend([
        "",
        "## By Lane",
    ])
    for k in sorted(lane_counts):
        lines.append(f"- {k}: {lane_counts[k]} jobs")

    lines.extend([
        "",
        "## By Group",
    ])
    for k in sorted(group_counts):
        lines.append(f"- {k}: {group_counts[k]} jobs")

    lines.extend([
        "",
        "## By Track",
    ])
    for k in sorted(track_counts):
        lines.append(f"- {k}: {track_counts[k]} jobs")

    lines.extend([
        "",
        f"CSV matrix: `{PLAN_CSV}`",
    ])

    write_text(PLAN_MD, "\n".join(lines) + "\n")

    print(f"Wrote: {PLAN_CSV}")
    print(f"Wrote: {PLAN_MD}")
    print(f"Total jobs: {len(rows)}")


if __name__ == "__main__":
    main()
