#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260402"
DATE_LABEL = "2026-04-02"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"advisor_packet_{DATE_TAG}" / "phase24_submission"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"pytorch/configs/reruns_{DATE_TAG}_phase24"

PLAN_CSV = TABLES / f"phase24_confidence_topup_matrix_{DATE_TAG}.csv"
PLAN_MD = NOTES / f"phase24_confidence_topup_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def set_key(text: str, key: str, value: str) -> str:
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
    repl = rf"\g<1>{value}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Key '{key}' not found while patching config")
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
    for k, v in updates.items():
        txt = set_key(txt, k, v)
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
            "mem": 128,
        },
        "track4_hh_full": {
            "short": "t4",
            "base": "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
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

    # Phase 4a) Track 2 weak-scaling top-up: add 1 seed to reach n=3 per sweep point.
    track2_base = "pytorch/configs/falcon_seed_repeats/params_dnn_tar_seed301.yaml"
    track2_tar = "/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar"
    track2_prefix = "publication_2020"
    weak_points = [(1, 1000, 32), (2, 2000, 64), (4, 4000, 128)]
    for nodes, ntrain, gbs in weak_points:
        seed = 503
        cfg_rel = (
            f"pytorch/configs/reruns_{DATE_TAG}_phase24/track2_true_weak/"
            f"params_t2_true_weak_a30_{nodes}n_s{seed}.yaml"
        )
        create_config(
            track2_base,
            cfg_rel,
            {
                "description": f'"Track 2 true weak scaling confidence top-up ({nodes} node A30) {DATE_LABEL}"',
                "Ntrain": str(ntrain),
                "global_train_batch_size": str(gbs),
                "train_batch_size": str(gbs),
                "random_seed": str(seed),
            },
        )
        add_row(
            phase="phase4_confidence_topup",
            submission_lane="phase4_confidence_topup",
            concurrency_group="a30_mem64",
            rerun_group="phase2_true_weak_scaling_track2_seed_ext",
            track_key="track2_fhn_scalability",
            sweep_variable="nodes_ntrain_global_batch",
            sweep_value=f"nodes={nodes}|Ntrain={ntrain}|global_batch={gbs}",
            seed=seed,
            job_name=f"trn_p4_t2weak_{nodes}n_s{seed}",
            params_file=cfg_rel,
            tar_path=track2_tar,
            data_prefix=track2_prefix,
            nodes=nodes,
            gpus_per_node=4,
            cpus_per_node=48,
            mem_gib=64,
            notes="phase4 top-up to reach n=3 for true weak scaling confidence",
        )

    # Phase 4b) LR/epoch top-up for Track 3/4: add 1 seed per point to reach n=3.
    lr_epoch_combos = [(5.0e-3, 200), (1.0e-2, 200), (1.0e-2, 300)]
    for track_key, meta in track_meta.items():
        for lr, ep in lr_epoch_combos:
            seed = 527
            lr_slug = slug_float(lr)
            cfg_rel = (
                f"pytorch/configs/reruns_{DATE_TAG}_phase24/{track_key}/phase4_lr/"
                f"params_{meta['short']}_lr{lr_slug}_ep{ep}_s{seed}.yaml"
            )
            create_config(
                meta["base"],
                cfg_rel,
                {
                    "description": f'"{track_key} phase4 lr/epoch confidence top-up {DATE_LABEL}"',
                    "learning_rate": fmt_lr(lr),
                    "epochs": str(ep),
                    "random_seed": str(seed),
                },
            )
            add_row(
                phase="phase4_confidence_topup",
                submission_lane="phase4_confidence_topup",
                concurrency_group=f"a30_mem{meta['mem']}",
                rerun_group="phase2_lr_epoch_seed_ext_track3_track4",
                track_key=track_key,
                sweep_variable="learning_rate_epochs",
                sweep_value=f"learning_rate={fmt_lr(lr)}|epochs={ep}",
                seed=seed,
                job_name=f"trn_p4_{meta['short']}_lr{lr_slug}_ep{ep}_s{seed}",
                params_file=cfg_rel,
                tar_path=meta["tar"],
                data_prefix=meta["prefix"],
                nodes=1,
                gpus_per_node=4,
                cpus_per_node=48,
                mem_gib=int(meta["mem"]),
                notes="phase4 top-up to reach n=3 for lr/epoch confidence",
            )

    # Phase 4c) Input-length top-up for Track 3/4: add 1 seed per point to reach n=3.
    phase4_input_lengths = [500, 1000, 2000]
    for track_key, meta in track_meta.items():
        for sub_len in phase4_input_lengths:
            seed = 528
            cfg_rel = (
                f"pytorch/configs/reruns_{DATE_TAG}_phase24/{track_key}/phase4_input/"
                f"params_{meta['short']}_sub{sub_len}_s{seed}.yaml"
            )
            create_config(
                meta["base"],
                cfg_rel,
                {
                    "description": f'"{track_key} phase4 input-length confidence top-up {DATE_LABEL}"',
                    "features_sub_length": str(sub_len),
                    "random_seed": str(seed),
                },
            )
            add_row(
                phase="phase4_confidence_topup",
                submission_lane="phase4_confidence_topup",
                concurrency_group=f"a30_mem{meta['mem']}",
                rerun_group="phase2_input_length_seed_ext_track3_track4",
                track_key=track_key,
                sweep_variable="features_sub_length",
                sweep_value=f"features_sub_length={sub_len}",
                seed=seed,
                job_name=f"trn_p4_{meta['short']}_sub{sub_len}_s{seed}",
                params_file=cfg_rel,
                tar_path=meta["tar"],
                data_prefix=meta["prefix"],
                nodes=1,
                gpus_per_node=4,
                cpus_per_node=48,
                mem_gib=int(meta["mem"]),
                notes="phase4 top-up to reach n=3 for input-length confidence",
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

    lines = [
        f"# Phase 4 Confidence Top-Up Matrix ({DATE_LABEL})",
        "",
        "Training-only jobs (testing runs excluded).",
        "",
        "## Purpose",
        "- Raise selected Phase-2 sweeps from n=2 to n=3 per sweep point.",
        "- Keep rerun groups unchanged so downstream aggregation reflects improved confidence.",
        "- Use denser periodic reports (`REPORT_INTERVAL=15s`) for improved GPU/utilization trend resolution.",
        "",
        "## Totals",
        f"- Total jobs: {len(rows)}",
    ]

    for k in sorted(phase_counts):
        lines.append(f"- {k}: {phase_counts[k]} jobs")

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
