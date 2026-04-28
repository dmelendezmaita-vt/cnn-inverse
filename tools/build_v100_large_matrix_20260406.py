#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import re
import struct
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

DATE_TAG = "20260406_v100_large"
DATE_LABEL = "2026-04-06 (V100 large matrix)"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
SUB_ROOT = OUT_ROOT / "submission"
TABLES = SUB_ROOT / "tables"
NOTES = SUB_ROOT / "notes"
PLANS = OUT_ROOT / "plans"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}_optfix"

TASK_MATRIX_CSV = TABLES / f"v100_large_task_matrix_{DATE_TAG}.csv"
RUNNER_PLAN_CSV = TABLES / f"v100_large_runner_plan_{DATE_TAG}.csv"
REJECT_CSV = TABLES / f"v100_large_rejected_configs_{DATE_TAG}.csv"
POOL_JSON = NOTES / f"v100_large_pool_sizes_{DATE_TAG}.json"
PLAN_MD = NOTES / f"v100_large_submission_notes_{DATE_TAG}.md"

SBATCH_SCRIPT = REPO / "tools" / "slurm_v100_large_bucket.sbatch"


@dataclass
class TrackMeta:
    key: str
    short: str
    base_cfg: str
    tar_path: str
    data_prefix: str
    mem_gib: int


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


def _resolve_tar_member(tf: tarfile.TarFile, rel_path: str) -> str:
    candidates = [rel_path, f"./{rel_path}", rel_path.lstrip("/")]
    names = set(tf.getnames())
    for c in candidates:
        if c in names:
            return c

    suffix = "/" + rel_path.lstrip("/")
    for n in names:
        if n.endswith(suffix):
            return n
    raise FileNotFoundError(f"Could not resolve member '{rel_path}' in tar")


def _read_array_shape_from_tar_member(
    tar_path: str, member_rel: str, expected_cols: int | None = None
) -> Tuple[int, ...]:
    with tarfile.open(tar_path, "r") as tf:
        member_name = _resolve_tar_member(tf, member_rel)
        member = tf.getmember(member_name)
        f = tf.extractfile(member_name)
        if f is None:
            raise FileNotFoundError(f"Failed to extract member {member_name}")

        with f:
            magic = f.read(6)
            if magic == b"\x93NUMPY":
                major, minor = struct.unpack("BB", f.read(2))
                if major == 1:
                    header_len = struct.unpack("<H", f.read(2))[0]
                elif major in (2, 3):
                    header_len = struct.unpack("<I", f.read(4))[0]
                else:
                    raise ValueError(f"Unsupported NPY version {major}.{minor}")

                header = f.read(header_len).decode("latin1")
                header_dict = ast.literal_eval(header)
                shape = header_dict["shape"]
                if not isinstance(shape, tuple):
                    raise ValueError(f"Invalid shape in header for {member_name}: {shape}")
                return shape

            if expected_cols is None or expected_cols <= 0:
                raise ValueError(
                    f"Member {member_name} is not NPY and expected_cols was not provided"
                )

            total_bytes = int(member.size)
            if total_bytes <= 0:
                raise ValueError(f"Invalid member size for {member_name}: {total_bytes}")

            if total_bytes % (expected_cols * 4) == 0:
                n_rows = total_bytes // (expected_cols * 4)
                return (int(n_rows), int(expected_cols))
            if total_bytes % (expected_cols * 8) == 0:
                n_rows = total_bytes // (expected_cols * 8)
                return (int(n_rows), int(expected_cols))

            raise ValueError(
                f"Could not infer shape for raw member {member_name}: "
                f"size={total_bytes}, expected_cols={expected_cols}"
            )


def infer_pool_sizes(track: TrackMeta) -> Dict[str, int]:
    base_path = REPO / track.base_cfg
    cfg = yaml.safe_load(base_path.read_text())
    data = cfg["data"]

    ntest = int(data["Ntest"])
    nvalidate = int(data["Nvalidate"])
    targets_cols_num = int(data["targets_cols_num"])
    curr = str(data["curr"])
    targets_tpl = data["file_names"]["targets"]
    rel = targets_tpl.format(data_prefix=track.data_prefix, curr=curr)
    shape = _read_array_shape_from_tar_member(track.tar_path, rel, expected_cols=targets_cols_num)
    n_total = int(shape[0])
    n_pool = n_total - ntest
    if n_pool <= 0:
        raise ValueError(f"Computed non-positive pool for {track.key}: total={n_total}, Ntest={ntest}")

    return {
        "n_total": n_total,
        "Ntest": ntest,
        "Nvalidate_default": nvalidate,
        "n_pool": n_pool,
        "ntrain_max_feasible": n_pool - nvalidate,
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    PLANS.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    tracks = {
        "track3_hh_reduced": TrackMeta(
            key="track3_hh_reduced",
            short="t3",
            base_cfg="src/pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml",
            tar_path="/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            data_prefix="reduced_data",
            mem_gib=128,
        ),
        "track4_hh_full": TrackMeta(
            key="track4_hh_full",
            short="t4",
            base_cfg="src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
            tar_path="/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            data_prefix="concatenated_data",
            mem_gib=192,
        ),
    }

    pool_info = {k: infer_pool_sizes(v) for k, v in tracks.items()}
    POOL_JSON.write_text(json.dumps(pool_info, indent=2) + "\n")

    # Large matrix dimensions (2*2*3*4*3*2*3 = 864)
    policies = ["weak", "strong"]
    nodes_list = [1, 2, 4]
    weak_per_rank_batches = [8, 16, 24, 32]
    strong_global_batches = [64, 128, 192, 256]
    learning_rates = [1.0e-4, 3.0e-4, 1.0e-3]
    sub_lengths = [2000, 4000]
    seeds = [971, 972, 973]

    weak_ntrain_base = {
        "track3_hh_reduced": 1024,
        "track4_hh_full": 2048,
    }
    strong_ntrain_fixed = {
        "track3_hh_reduced": 4096,
        "track4_hh_full": 8192,
    }

    task_rows: List[Dict[str, str]] = []
    rejected_rows: List[Dict[str, str]] = []

    task_id = 0

    for track_key, meta in tracks.items():
        for policy in policies:
            for nodes in nodes_list:
                batch_values = weak_per_rank_batches if policy == "weak" else strong_global_batches
                for bval in batch_values:
                    for lr in learning_rates:
                        for sub_len in sub_lengths:
                            for seed in seeds:
                                if policy == "weak":
                                    ntrain = weak_ntrain_base[track_key] * nodes
                                    world_size = nodes * 2
                                    gbs = bval * world_size
                                    batch_profile = f"weak_prb{bval}"
                                else:
                                    ntrain = strong_ntrain_fixed[track_key]
                                    gbs = bval
                                    batch_profile = f"strong_gbs{bval}"

                                nvalidate = pool_info[track_key]["Nvalidate_default"]
                                if ntrain + nvalidate > pool_info[track_key]["n_pool"]:
                                    rejected_rows.append(
                                        {
                                            "track_key": track_key,
                                            "policy": policy,
                                            "nodes": str(nodes),
                                            "batch_profile": batch_profile,
                                            "learning_rate": fmt_lr(lr),
                                            "features_sub_length": str(sub_len),
                                            "seed": str(seed),
                                            "Ntrain": str(ntrain),
                                            "Nvalidate": str(nvalidate),
                                            "n_pool": str(pool_info[track_key]["n_pool"]),
                                            "reason": f"Ntrain+Nvalidate={ntrain+nvalidate} exceeds n_pool={pool_info[track_key]['n_pool']}",
                                        }
                                    )
                                    continue

                                task_id += 1
                                lr_slug = slug_float(lr)
                                cfg_rel = (
                                    f"src/pytorch/configs/reruns_{DATE_TAG}_optfix/{track_key}/"
                                    f"{policy}/n{nodes}/"
                                    f"params_{meta.short}_{policy}_n{nodes}_{batch_profile}_"
                                    f"lr{lr_slug}_sub{sub_len}_s{seed}.yaml"
                                )

                                create_config(
                                    meta.base_cfg,
                                    cfg_rel,
                                    {
                                        "description": f'"{track_key} {policy} large-matrix n{nodes} {batch_profile} lr={fmt_lr(lr)} sub={sub_len} {DATE_LABEL}"',
                                        "learning_rate": fmt_lr(lr),
                                        "epochs": "400",
                                        "init_learning_rate": fmt_lr(lr / 20.0),
                                        "linear_epochs": "40",
                                        "constant_epochs": "40",
                                        "final_learning_rate": fmt_lr(lr / 200.0),
                                        "features_normalize": "True",
                                        "features_sub_begin_random": "True",
                                        "features_sub_begin_random_eval": "False",
                                        "features_sub_length": str(sub_len),
                                        "global_train_batch_size": str(gbs),
                                        "train_batch_size": str(gbs),
                                        "Ntrain": str(ntrain),
                                        "random_seed": str(seed),
                                        "loss_type": "huber",
                                        "huber_beta": "1.0",
                                        "grad_clip_max_norm": "1.0",
                                    },
                                )

                                task_slug = (
                                    f"{meta.short}_{policy}_n{nodes}_{batch_profile}_"
                                    f"lr{lr_slug}_sub{sub_len}_s{seed}"
                                )
                                task_rows.append(
                                    {
                                        "task_id": str(task_id),
                                        "task_slug": task_slug,
                                        "track_key": track_key,
                                        "policy": policy,
                                        "nodes": str(nodes),
                                        "batch_profile": batch_profile,
                                        "learning_rate": fmt_lr(lr),
                                        "features_sub_length": str(sub_len),
                                        "seed": str(seed),
                                        "params_file": cfg_rel,
                                        "Ntrain": str(ntrain),
                                        "Nvalidate": str(nvalidate),
                                        "n_pool": str(pool_info[track_key]["n_pool"]),
                                    }
                                )

    task_fields = [
        "task_id",
        "task_slug",
        "track_key",
        "policy",
        "nodes",
        "batch_profile",
        "learning_rate",
        "features_sub_length",
        "seed",
        "params_file",
        "Ntrain",
        "Nvalidate",
        "n_pool",
    ]

    with TASK_MATRIX_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=task_fields, lineterminator="\n")
        w.writeheader()
        for row in task_rows:
            w.writerow(row)

    reject_fields = [
        "track_key",
        "policy",
        "nodes",
        "batch_profile",
        "learning_rate",
        "features_sub_length",
        "seed",
        "Ntrain",
        "Nvalidate",
        "n_pool",
        "reason",
    ]
    with REJECT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reject_fields, lineterminator="\n")
        w.writeheader()
        for row in rejected_rows:
            w.writerow(row)

    # Runner plan: one Slurm job per node count bucket
    runner_rows: List[Dict[str, str]] = []
    for nodes in [1, 2, 4]:
        runner_rows.append(
            {
                "runner_id": f"v100_large_n{nodes}",
                "job_name": f"v100L_n{nodes}_{DATE_TAG}",
                "nodes": str(nodes),
                "gpus_per_node": "2",
                "cpus_per_node": "24",
                "mem_gib": "350",
                "partition": "v100_normal_q",
                "qos": "fal_v100_normal_base",
                "time_limit": "72:00:00",
                "task_csv": str(TASK_MATRIX_CSV),
                "progress_csv": str(TABLES / f"v100_large_task_progress_n{nodes}_{DATE_TAG}.csv"),
                "summary_json": str(NOTES / f"v100_large_runner_summary_n{nodes}_{DATE_TAG}.json"),
                "results_root": str(OUT_ROOT / "runs"),
                "sbatch_script": str(SBATCH_SCRIPT),
            }
        )

    runner_fields = [
        "runner_id",
        "job_name",
        "nodes",
        "gpus_per_node",
        "cpus_per_node",
        "mem_gib",
        "partition",
        "qos",
        "time_limit",
        "task_csv",
        "progress_csv",
        "summary_json",
        "results_root",
        "sbatch_script",
    ]
    with RUNNER_PLAN_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=runner_fields, lineterminator="\n")
        w.writeheader()
        for row in runner_rows:
            w.writerow(row)

    per_nodes = Counter(int(r["nodes"]) for r in task_rows)
    per_track = Counter(r["track_key"] for r in task_rows)
    per_policy = Counter(r["policy"] for r in task_rows)

    lines = [
        f"# V100 Large Matrix Plan ({DATE_LABEL})",
        "",
        "Large matrix dimensions:",
        "- tracks=2",
        "- policies=2",
        "- nodes=3 (1,2,4)",
        "- batch profiles=4",
        "- learning rates=3",
        "- features_sub_length=2",
        "- seeds=3",
        "- total expected = 864 tasks",
        "",
        "Execution model:",
        "- One Slurm runner job per node bucket (1n / 2n / 4n)",
        "- Each runner executes all tasks for that node count sequentially inside the same allocation",
        "",
        "Isolation:",
        f"- Output root: `data/important_notes/optimization_track_{DATE_TAG}/`",
        f"- Config root: `src/pytorch/configs/reruns_{DATE_TAG}_optfix/`",
        "- Partition/QoS: v100_normal_q + fal_v100_normal_base",
        "",
        f"Task matrix CSV: `{TASK_MATRIX_CSV}`",
        f"Runner plan CSV: `{RUNNER_PLAN_CSV}`",
        f"Rejected CSV: `{REJECT_CSV}`",
        f"Pool JSON: `{POOL_JSON}`",
        "",
        f"Total tasks written: {len(task_rows)}",
        f"Rejected tasks: {len(rejected_rows)}",
        "By nodes:",
    ]
    for n in sorted(per_nodes):
        lines.append(f"- nodes={n}: {per_nodes[n]} tasks")
    lines.append("By track:")
    for k in sorted(per_track):
        lines.append(f"- {k}: {per_track[k]} tasks")
    lines.append("By policy:")
    for k in sorted(per_policy):
        lines.append(f"- {k}: {per_policy[k]} tasks")

    write_text(PLAN_MD, "\n".join(lines) + "\n")

    print(f"Wrote: {TASK_MATRIX_CSV}")
    print(f"Wrote: {RUNNER_PLAN_CSV}")
    print(f"Wrote: {REJECT_CSV}")
    print(f"Wrote: {POOL_JSON}")
    print(f"Wrote: {PLAN_MD}")
    print(f"Total tasks: {len(task_rows)}")
    print(f"Rejected tasks: {len(rejected_rows)}")


if __name__ == "__main__":
    main()
