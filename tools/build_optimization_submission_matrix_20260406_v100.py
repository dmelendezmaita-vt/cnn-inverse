#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import re
import shutil
import struct
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

DATE_TAG = "20260406_v100"
DATE_LABEL = "2026-04-06 (V100 track)"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
SUB_ROOT = OUT_ROOT / "submission"
TABLES = SUB_ROOT / "tables"
NOTES = SUB_ROOT / "notes"
PLANS = OUT_ROOT / "plans"
OPT_TABLES = OUT_ROOT / "tables"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}_optfix"

PLAN_CSV = TABLES / f"optimization_submission_matrix_{DATE_TAG}.csv"
PLAN_MD = NOTES / f"optimization_submission_notes_{DATE_TAG}.md"
REJECT_CSV = TABLES / f"optimization_rejected_configs_{DATE_TAG}.csv"
POOL_JSON = NOTES / f"optimization_track_pool_sizes_{DATE_TAG}.json"

BASELINE_SRC = IMPORTANT / "optimization_track_20260402" / "tables"
BASELINE_COPY = {
    "baseline_track3_track4_scalability_training_only.csv": "baseline_track3_track4_scalability_training_only.csv",
    "baseline_rerun_metric_summary_20260402.csv": "baseline_rerun_metric_summary_20260402.csv",
    "baseline_loss_spike_summary_training_only.csv": "baseline_loss_spike_summary_training_only.csv",
}


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
    candidates = [
        rel_path,
        f"./{rel_path}",
        rel_path.lstrip("/"),
    ]
    names = set(tf.getnames())
    for c in candidates:
        if c in names:
            return c

    # fallback: suffix match for path normalization differences
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

            # Fallback: some ARC tar datasets store raw float arrays with .npy suffix.
            # Infer rows from byte-size + expected target column count.
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
    OPT_TABLES.mkdir(parents=True, exist_ok=True)
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

    # keep baseline snapshots visible in this workspace only
    copied: List[Tuple[str, str]] = []
    missing: List[str] = []
    for src_name, dst_name in BASELINE_COPY.items():
        src = BASELINE_SRC / src_name
        dst = OPT_TABLES / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((src_name, dst_name))
        else:
            missing.append(src_name)

    pool_info = {k: infer_pool_sizes(v) for k, v in tracks.items()}
    POOL_JSON.write_text(json.dumps(pool_info, indent=2) + "\n")

    rows: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

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
        ntrain: int,
        nvalidate: int,
        tar_path: str,
        data_prefix: str,
        nodes: int,
        gpus_per_node: int,
        cpus_per_node: int,
        mem_gib: int,
        partition: str = "v100_normal_q",
        qos: str = "fal_v100_normal_short",
        time_limit: str = "02:00:00",
        gpu_token: str = "v100",
        gpu: str = "V100",
        cluster: str = "Falcon",
        data_access_mode: str = "copy_to_node",
        report_interval_sec: int = 30,
        notes: str = "",
    ) -> None:
        pinfo = pool_info[track_key]
        if ntrain + nvalidate > pinfo["n_pool"]:
            rejected.append(
                {
                    "phase": phase,
                    "track_key": track_key,
                    "job_name": job_name,
                    "params_file": params_file,
                    "sweep_variable": sweep_variable,
                    "sweep_value": sweep_value,
                    "seed": str(seed),
                    "Ntrain": str(ntrain),
                    "Nvalidate": str(nvalidate),
                    "n_pool": str(pinfo["n_pool"]),
                    "reason": f"Ntrain+Nvalidate={ntrain+nvalidate} exceeds n_pool={pinfo['n_pool']}",
                }
            )
            return

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
                "Ntrain": str(ntrain),
                "Nvalidate": str(nvalidate),
                "n_pool": str(pinfo["n_pool"]),
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

    base_lr = 3.0e-4
    base_epochs = 400

    # -----------------------------------------------------------------
    # Phase A: recovery ntrain sweep (fix failed split-capacity jobs)
    # -----------------------------------------------------------------
    phaseA_seeds = [951, 952, 953]
    phaseA_ntrain = {
        "track3_hh_reduced": [2048, 3072, 4096],
        "track4_hh_full": [4096, 8192, 12288],
    }
    for track_key, meta in tracks.items():
        for ntrain in phaseA_ntrain[track_key]:
            for seed in phaseA_seeds:
                cfg_rel = (
                    f"src/pytorch/configs/reruns_{DATE_TAG}_optfix/{track_key}/phaseA_recovery_ntrain/"
                    f"params_{meta.short}_ntr{ntrain}_huber_clip_s{seed}.yaml"
                )
                create_config(
                    meta.base_cfg,
                    cfg_rel,
                    {
                        "description": f'"{track_key} recovery ntrain sweep {DATE_LABEL}"',
                        "learning_rate": fmt_lr(base_lr),
                        "epochs": str(base_epochs),
                        "init_learning_rate": fmt_lr(base_lr / 20.0),
                        "linear_epochs": "40",
                        "constant_epochs": "40",
                        "final_learning_rate": fmt_lr(base_lr / 200.0),
                        "features_normalize": "True",
                        "features_sub_begin_random": "True",
                        "features_sub_begin_random_eval": "False",
                        "features_sub_length": "2000",
                        "global_train_batch_size": "32",
                        "train_batch_size": "32",
                        "Ntrain": str(ntrain),
                        "random_seed": str(seed),
                        "loss_type": "huber",
                        "huber_beta": "1.0",
                        "grad_clip_max_norm": "1.0",
                    },
                )
                add_row(
                    phase="phaseA_recovery_ntrain",
                    submission_lane="phaseA_recovery_ntrain",
                    concurrency_group=f"v100_mem{meta.mem_gib}",
                    rerun_group="optfix_phaseA_recovery_ntrain",
                    track_key=track_key,
                    sweep_variable="Ntrain",
                    sweep_value=f"Ntrain={ntrain}|variant=huber_clip1p0",
                    seed=seed,
                    job_name=f"trn_opt6_{meta.short}_ntr{ntrain}_s{seed}",
                    params_file=cfg_rel,
                    ntrain=ntrain,
                    nvalidate=pool_info[track_key]["Nvalidate_default"],
                    tar_path=meta.tar_path,
                    data_prefix=meta.data_prefix,
                    nodes=1,
                    gpus_per_node=2,
                    cpus_per_node=24,
                    mem_gib=meta.mem_gib,
                    time_limit="02:00:00",
                    report_interval_sec=30,
                    notes="Recovery lane for prior phase4 split-capacity failures",
                )

    # -----------------------------------------------------------------
    # Phase B: scaling improvement matrix (strong+weak, 1/2/4 nodes)
    # -----------------------------------------------------------------
    phaseB_seeds = [961, 962]
    phaseB_nodes = [1, 2, 4]
    phaseB_sub_lengths = [2000, 4000]

    # weak scaling by track: increase Ntrain with nodes; keep per-rank batch constant
    weak_ntrain = {
        "track3_hh_reduced": {1: 1024, 2: 2048, 4: 4096},
        "track4_hh_full": {1: 2048, 2: 4096, 4: 8192},
    }
    weak_per_rank_batch = 32

    # strong scaling by track: fixed Ntrain and global batch
    strong_ntrain = {
        "track3_hh_reduced": 4096,
        "track4_hh_full": 8192,
    }
    strong_global_batch = 128

    for track_key, meta in tracks.items():
        for policy in ["weak", "strong"]:
            for sub_len in phaseB_sub_lengths:
                for nodes in phaseB_nodes:
                    for seed in phaseB_seeds:
                        if policy == "weak":
                            ntrain = weak_ntrain[track_key][nodes]
                            world_size = nodes * 2
                            gbs = weak_per_rank_batch * world_size
                            sweep_value = (
                                f"policy=weak|nodes={nodes}|features_sub_length={sub_len}|"
                                f"Ntrain={ntrain}|global_train_batch_size={gbs}"
                            )
                        else:
                            ntrain = strong_ntrain[track_key]
                            gbs = strong_global_batch
                            sweep_value = (
                                f"policy=strong|nodes={nodes}|features_sub_length={sub_len}|"
                                f"Ntrain={ntrain}|global_train_batch_size={gbs}"
                            )

                        cfg_rel = (
                            f"src/pytorch/configs/reruns_{DATE_TAG}_optfix/{track_key}/phaseB_scaling_improve/"
                            f"params_{meta.short}_{policy}_n{nodes}_sub{sub_len}_gbs{gbs}_s{seed}.yaml"
                        )
                        create_config(
                            meta.base_cfg,
                            cfg_rel,
                            {
                                "description": f'"{track_key} scaling improve {policy} n{nodes} sub{sub_len} {DATE_LABEL}"',
                                "learning_rate": fmt_lr(base_lr),
                                "epochs": str(base_epochs),
                                "init_learning_rate": fmt_lr(base_lr / 20.0),
                                "linear_epochs": "40",
                                "constant_epochs": "40",
                                "final_learning_rate": fmt_lr(base_lr / 200.0),
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

                        add_row(
                            phase="phaseB_scaling_improve",
                            submission_lane="phaseB_scaling_improve",
                            concurrency_group=f"v100_mem{meta.mem_gib}_n{nodes}",
                            rerun_group="optfix_phaseB_scaling_improve",
                            track_key=track_key,
                            sweep_variable="policy_nodes_features_sub_length_Ntrain_global_train_batch_size",
                            sweep_value=sweep_value,
                            seed=seed,
                            job_name=f"trn_opt7_{meta.short}_{policy}_n{nodes}_sub{sub_len}_s{seed}",
                            params_file=cfg_rel,
                            ntrain=ntrain,
                            nvalidate=pool_info[track_key]["Nvalidate_default"],
                            tar_path=meta.tar_path,
                            data_prefix=meta.data_prefix,
                            nodes=nodes,
                            gpus_per_node=2,
                            cpus_per_node=24,
                            mem_gib=meta.mem_gib,
                            time_limit="03:00:00" if nodes >= 2 else "02:00:00",
                            report_interval_sec=30,
                            notes="Scaling-improvement lane (strong+weak) using stable robustness settings",
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
        "Ntrain",
        "Nvalidate",
        "n_pool",
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

    reject_fields = [
        "phase",
        "track_key",
        "job_name",
        "params_file",
        "sweep_variable",
        "sweep_value",
        "seed",
        "Ntrain",
        "Nvalidate",
        "n_pool",
        "reason",
    ]
    with REJECT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reject_fields, lineterminator="\n")
        w.writeheader()
        for row in rejected:
            w.writerow(row)

    phase_counts = Counter(r["phase"] for r in rows)
    lane_counts = Counter(r["submission_lane"] for r in rows)
    track_counts = Counter(r["track_key"] for r in rows)

    lines = [
        f"# Optimization Submission Matrix ({DATE_LABEL})",
        "",
        "Training-only jobs (testing runs excluded).",
        "",
        "## Safety / No-overwrite guarantees",
        "- All new outputs are written under `data/important_notes/optimization_track_20260406_v100/`.",
        "- New configs are written under `src/pytorch/configs/reruns_20260406_v100_optfix/`.",
        "- Existing per-track folders (`first_track_paper_parity`, `second_track_scalability`, `third_track_hh`, `fourth_track_hh_full`) are not edited by this workflow.",
        "- Existing optimization outputs from 2026-04-02 are preserved.",
        "",
        "## Baseline snapshots copied into this workspace",
    ]
    if copied:
        for src_name, dst_name in copied:
            lines.append(f"- {src_name} -> {dst_name}")
    if missing:
        for src_name in missing:
            lines.append(f"- MISSING baseline source: {src_name}")

    lines += [
        "",
        "## Track pool-size feasibility",
    ]
    for tk, info in pool_info.items():
        lines.append(
            f"- {tk}: n_total={info['n_total']}, Ntest={info['Ntest']}, n_pool={info['n_pool']}, "
            f"Nvalidate_default={info['Nvalidate_default']}, ntrain_max_feasible={info['ntrain_max_feasible']}"
        )

    lines += [
        "",
        "## Plan counts",
        f"- Total planned jobs: {len(rows)}",
        f"- Rejected by feasibility guard: {len(rejected)}",
        "- By phase:",
    ]
    for k in sorted(phase_counts):
        lines.append(f"  - {k}: {phase_counts[k]}")

    lines.append("- By track:")
    for k in sorted(track_counts):
        lines.append(f"  - {k}: {track_counts[k]}")

    lines.append("- By lane:")
    for k in sorted(lane_counts):
        lines.append(f"  - {k}: {lane_counts[k]}")

    lines += [
        "",
        "## Lanes",
        "- `phaseA_recovery_ntrain`: replaces previously invalid `Ntrain` settings with feasible values.",
        "- `phaseB_scaling_improve`: strong+weak scaling with stable settings (`huber`, `grad_clip=1.0`, `features_sub_begin_random=True`).",
        "",
        f"Plan CSV: `{PLAN_CSV}`",
        f"Rejected CSV: `{REJECT_CSV}`",
        f"Pool JSON: `{POOL_JSON}`",
    ]

    write_text(PLAN_MD, "\n".join(lines) + "\n")

    print(f"Wrote: {PLAN_CSV}")
    print(f"Wrote: {REJECT_CSV}")
    print(f"Wrote: {POOL_JSON}")
    print(f"Wrote: {PLAN_MD}")
    print(f"Total jobs: {len(rows)}")
    print(f"Rejected jobs: {len(rejected)}")


if __name__ == "__main__":
    main()
