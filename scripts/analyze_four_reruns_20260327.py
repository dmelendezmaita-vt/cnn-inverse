#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATE_TAG = "20260327"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/advisor_packet_{DATE_TAG}/reruns"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
FIGURES = OUT_ROOT / "figures"
JOBS_CSV = TABLES / f"rerun_jobs_registry_{DATE_TAG}.csv"
RUNS_BASE = Path("/projects/neuro-collab/data/runs")


def ensure_dirs() -> None:
    for p in [TABLES, NOTES, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def chunked(items: List[str], size: int = 180) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def query_squeue() -> pd.DataFrame:
    user = os.environ.get("USER", "")
    proc = subprocess.run(
        ["squeue", "-M", "falcon", "-u", user, "-h", "-o", "%i|%T|%M|%j|%V|%S"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        return pd.DataFrame(columns=["job_id", "squeue_state", "squeue_elapsed", "job_name", "squeue_submit", "squeue_start"])
    out = proc.stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        job_id, state, elapsed, job_name, submit, start = parts[:6]
        if not job_id.isdigit():
            continue
        rows.append(
            {
                "job_id": int(job_id),
                "squeue_state": state.strip(),
                "squeue_elapsed": elapsed.strip(),
                "job_name": job_name.strip(),
                "squeue_submit": submit.strip(),
                "squeue_start": start.strip(),
            }
        )
    return pd.DataFrame(rows)


def query_sacct(job_ids: List[int]) -> pd.DataFrame:
    rows = []
    ids = [str(int(x)) for x in sorted(set(job_ids))]
    for ch in chunked(ids, 180):
        proc = subprocess.run(
            [
                "sacct",
                "-M",
                "falcon",
                "-X",
                "-j",
                ",".join(ch),
                "--format=JobIDRaw,State,ElapsedRaw,Elapsed,Submit,Start,End",
                "-n",
                "-P",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            continue
        out = proc.stdout
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 7:
                continue
            jid_raw, state, elapsed_raw, elapsed, submit, start, end = parts[:7]
            jid_raw = jid_raw.strip()
            if not jid_raw.isdigit():
                continue
            state = state.strip().split()[0]
            try:
                elapsed_sec = int(elapsed_raw.strip())
            except Exception:
                elapsed_sec = np.nan
            rows.append(
                {
                    "job_id": int(jid_raw),
                    "sacct_state": state,
                    "sacct_elapsed_sec": elapsed_sec,
                    "sacct_elapsed": elapsed.strip(),
                    "sacct_submit": submit.strip(),
                    "sacct_start": start.strip(),
                    "sacct_end": end.strip(),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "job_id",
                "sacct_state",
                "sacct_elapsed_sec",
                "sacct_elapsed",
                "sacct_submit",
                "sacct_start",
                "sacct_end",
            ]
        )
    return pd.DataFrame(rows).sort_values("job_id").drop_duplicates("job_id", keep="first")


def parse_metric_csv(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "metrics_summary.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    df = clean_df(df)
    if "split" not in df.columns:
        return {}
    test = df[df["split"].str.lower() == "test"]
    if test.empty:
        return {}
    r = test.iloc[0]
    return {
        "test_mse": float(r.get("mse", np.nan)),
        "test_mae": float(r.get("mae", np.nan)),
        "test_r2": float(r.get("r2", np.nan)),
    }


def parse_rank0_runtime(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "run_dnn" / "rank0_info.log"
    if not p.exists():
        p = run_dir / "run_dnn_info.log"
    if not p.exists():
        return {}
    text = p.read_text(errors="ignore")

    def rex(pattern: str) -> float:
        m = re.search(pattern, text)
        if not m:
            return np.nan
        try:
            return float(m.group(1))
        except Exception:
            return np.nan

    return {
        "runtime_train_sec": rex(r"Runtime - train \[sec\]:\s*([0-9eE+\-.]+)"),
        "runtime_eval_sec": rex(r"Runtime - eval \[sec\]:\s*([0-9eE+\-.]+)"),
        "train_samples_total": rex(r"Runtime statistics - train - #samples \(total\):\s*([0-9eE+\-.]+)"),
        "train_samples_per_sec": rex(r"Runtime statistics - train - avg\. samples/sec:\s*([0-9eE+\-.]+)"),
        "per_rank_train_batch_size": rex(r"per_rank_train_batch_size=([0-9]+)"),
    }


def parse_report_gpu(run_dir: Path) -> Dict[str, float]:
    vals = []
    for rpt in sorted(run_dir.glob("report_*.txt")):
        txt = rpt.read_text(errors="ignore")
        m = re.search(r"GPU util avg:\s*([0-9]+)%", txt)
        if m:
            vals.append(float(m.group(1)))
    if not vals:
        return {}
    return {
        "gpu_util_report_mean_pct": float(np.mean(vals)),
        "gpu_util_report_max_pct": float(np.max(vals)),
        "gpu_report_samples": float(len(vals)),
    }


def parse_loss_spikes(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "loss.txt"
    if not p.exists():
        return {}
    try:
        arr = np.loadtxt(p)
    except Exception:
        return {}
    train_loss = arr if arr.ndim == 1 else arr[:, 0]
    train_loss = np.asarray(train_loss, dtype=float)
    if len(train_loss) < 12:
        return {
            "loss_epochs": float(len(train_loss)),
            "loss_max_increase_ratio": np.nan,
            "loss_spike_count_gt25pct": np.nan,
            "loss_has_spike_gt50pct": np.nan,
        }
    t = train_loss[10:]
    prev = np.maximum(t[:-1], 1e-12)
    delta_ratio = (t[1:] - t[:-1]) / prev
    max_inc = float(np.max(delta_ratio)) if len(delta_ratio) else np.nan
    spike25 = int(np.sum(delta_ratio > 0.25)) if len(delta_ratio) else 0
    spike50 = int(max_inc > 0.50) if not math.isnan(max_inc) else 0
    return {
        "loss_epochs": float(len(train_loss)),
        "loss_max_increase_ratio": max_inc,
        "loss_spike_count_gt25pct": float(spike25),
        "loss_has_spike_gt50pct": float(spike50),
    }


def parse_params_file(params_file: str) -> Dict[str, float]:
    p = REPO / params_file
    if not p.exists():
        return {}
    text = p.read_text(errors="ignore")

    def rex(pattern: str) -> float:
        m = re.search(pattern, text, flags=re.MULTILINE)
        if not m:
            return np.nan
        try:
            return float(m.group(1))
        except Exception:
            return np.nan

    return {
        "learning_rate": rex(r"^\s*learning_rate:\s*([0-9eE+\-.]+)\s*$"),
        "epochs": rex(r"^\s*epochs:\s*([0-9]+)\s*$"),
        "global_train_batch_size": rex(r"^\s*global_train_batch_size:\s*([0-9]+)\s*$"),
        "features_sub_length": rex(r"^\s*features_sub_length:\s*([0-9]+)\s*$"),
        "Ntrain": rex(r"^\s*Ntrain:\s*([0-9]+)\s*$"),
    }


def plot_summaries(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return

    # True weak scaling: train runtime and per-GPU throughput.
    weak = metrics[metrics["rerun_group"] == "true_weak_scaling_track2"].copy()
    if not weak.empty:
        g = (
            weak.groupby("nodes")
            .agg(
                runtime_train_sec_mean=("runtime_train_sec", "mean"),
                train_samples_per_sec_mean=("train_samples_per_sec", "mean"),
                gpus_total=("gpus_total", "median"),
            )
            .reset_index()
            .sort_values("nodes")
        )
        g["per_gpu_train_sps"] = g["train_samples_per_sec_mean"] / g["gpus_total"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(g["nodes"], g["runtime_train_sec_mean"], marker="o")
        axes[0].set_title("Track 2 weak-scaling rerun: train runtime")
        axes[0].set_xlabel("Nodes")
        axes[0].set_ylabel("runtime_train_sec (mean)")
        axes[0].grid(alpha=0.3)
        axes[1].plot(g["nodes"], g["per_gpu_train_sps"], marker="o")
        axes[1].set_title("Track 2 weak-scaling rerun: per-GPU throughput")
        axes[1].set_xlabel("Nodes")
        axes[1].set_ylabel("train samples/sec per GPU")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / f"rerun_true_weak_scaling_{DATE_TAG}.png", dpi=180)
        plt.close(fig)

    for group, fname, title, x_key in [
        ("lr_epoch_sweep_track3_track4", f"rerun_lr_epoch_{DATE_TAG}.png", "LR/Epoch Sweep", "sweep_value"),
        ("batch_size_sweep_track3_track4", f"rerun_batch_size_{DATE_TAG}.png", "Batch Size Sweep", "sweep_value"),
        ("input_length_ablation_track3_track4", f"rerun_input_length_{DATE_TAG}.png", "Input Length Ablation", "sweep_value"),
    ]:
        d = metrics[metrics["rerun_group"] == group].copy()
        if d.empty:
            continue
        gg = (
            d.groupby(["track_key", x_key])["test_r2"]
            .mean()
            .reset_index()
            .sort_values([x_key, "track_key"])
        )
        if gg.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.6))
        for track_key in sorted(gg["track_key"].unique()):
            part = gg[gg["track_key"] == track_key]
            ax.plot(part[x_key], part["test_r2"], marker="o", label=track_key)
        ax.set_title(f"{title}: mean test R2 (completed jobs)")
        ax.set_xlabel(x_key)
        ax.set_ylabel("test_r2")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=180)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    if not JOBS_CSV.exists():
        raise SystemExit(f"Missing jobs registry: {JOBS_CSV}")

    jobs = clean_df(pd.read_csv(JOBS_CSV, dtype=str))
    jobs["job_id"] = pd.to_numeric(jobs["job_id"], errors="coerce")
    jobs["nodes"] = pd.to_numeric(jobs["nodes"], errors="coerce")
    jobs["gpus_per_node"] = pd.to_numeric(jobs["gpus_per_node"], errors="coerce")
    jobs["gpus_total"] = jobs["nodes"] * jobs["gpus_per_node"]

    valid_ids = jobs["job_id"].dropna().astype(int).tolist()
    sacct = query_sacct(valid_ids) if valid_ids else pd.DataFrame()
    squeue = query_squeue()

    merged = jobs.merge(sacct, on="job_id", how="left")
    if not squeue.empty:
        merged = merged.merge(
            squeue[["job_id", "squeue_state", "squeue_elapsed", "squeue_submit", "squeue_start"]],
            on="job_id",
            how="left",
        )
    else:
        merged["squeue_state"] = np.nan
        merged["squeue_elapsed"] = np.nan
        merged["squeue_submit"] = np.nan
        merged["squeue_start"] = np.nan

    merged["effective_state"] = merged["sacct_state"].fillna(merged["squeue_state"]).fillna(merged["status"])
    merged["effective_elapsed"] = merged["sacct_elapsed"].fillna(merged["squeue_elapsed"]).fillna(merged["elapsed"])
    merged.to_csv(TABLES / f"rerun_status_snapshot_{DATE_TAG}.csv", index=False)

    completed = merged[merged["effective_state"].str.upper().str.startswith("COMPLETED", na=False)].copy()
    metric_rows = []
    for _, r in completed.iterrows():
        jid = int(r["job_id"])
        run_dir = RUNS_BASE / str(jid)
        rec: Dict[str, float | str] = {
            "job_id": jid,
            "job_name": r["job_name"],
            "rerun_group": r["rerun_group"],
            "track_key": r["track_key"],
            "sweep_variable": r["sweep_variable"],
            "sweep_value": r["sweep_value"],
            "nodes": float(r["nodes"]) if not pd.isna(r["nodes"]) else np.nan,
            "gpus_per_node": float(r["gpus_per_node"]) if not pd.isna(r["gpus_per_node"]) else np.nan,
            "gpus_total": float(r["gpus_total"]) if not pd.isna(r["gpus_total"]) else np.nan,
            "params_file": r["params_file"],
            "run_dir": str(run_dir),
        }
        rec.update(parse_params_file(str(r["params_file"])))
        if run_dir.exists():
            rec.update(parse_metric_csv(run_dir))
            rec.update(parse_rank0_runtime(run_dir))
            rec.update(parse_report_gpu(run_dir))
            rec.update(parse_loss_spikes(run_dir))
        metric_rows.append(rec)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        metrics = pd.DataFrame(
            columns=[
                "job_id",
                "job_name",
                "rerun_group",
                "track_key",
                "sweep_variable",
                "sweep_value",
                "nodes",
                "gpus_per_node",
                "gpus_total",
                "params_file",
                "run_dir",
                "learning_rate",
                "epochs",
                "global_train_batch_size",
                "features_sub_length",
                "Ntrain",
                "test_mse",
                "test_mae",
                "test_r2",
                "runtime_train_sec",
                "runtime_eval_sec",
                "train_samples_total",
                "train_samples_per_sec",
                "per_rank_train_batch_size",
                "gpu_util_report_mean_pct",
                "gpu_util_report_max_pct",
                "gpu_report_samples",
                "loss_epochs",
                "loss_max_increase_ratio",
                "loss_spike_count_gt25pct",
                "loss_has_spike_gt50pct",
            ]
        )
    metrics.to_csv(TABLES / f"rerun_completed_metrics_{DATE_TAG}.csv", index=False)

    # Status summary
    status_summary = (
        merged.groupby(["rerun_group", "effective_state"])
        .size()
        .reset_index(name="count")
        .sort_values(["rerun_group", "count"], ascending=[True, False])
    )
    status_summary.to_csv(TABLES / f"rerun_status_summary_{DATE_TAG}.csv", index=False)

    # Group-level metric summaries (completed only)
    if not metrics.empty:
        metric_summary = (
            metrics.groupby(["rerun_group", "track_key", "sweep_value"])
            .agg(
                n_runs=("job_id", "count"),
                test_mse_mean=("test_mse", "mean"),
                test_mae_mean=("test_mae", "mean"),
                test_r2_mean=("test_r2", "mean"),
                runtime_train_sec_mean=("runtime_train_sec", "mean"),
                train_samples_per_sec_mean=("train_samples_per_sec", "mean"),
                gpu_util_report_mean_pct=("gpu_util_report_mean_pct", "mean"),
                loss_spike_rate_pct=("loss_has_spike_gt50pct", lambda s: float(np.nanmean(s) * 100.0)),
            )
            .reset_index()
            .sort_values(["rerun_group", "track_key", "sweep_value"])
        )
    else:
        metric_summary = pd.DataFrame(
            columns=[
                "rerun_group",
                "track_key",
                "sweep_value",
                "n_runs",
                "test_mse_mean",
                "test_mae_mean",
                "test_r2_mean",
                "runtime_train_sec_mean",
                "train_samples_per_sec_mean",
                "gpu_util_report_mean_pct",
                "loss_spike_rate_pct",
            ]
        )
    metric_summary.to_csv(TABLES / f"rerun_metric_summary_{DATE_TAG}.csv", index=False)

    # Dedicated true weak-scaling table
    weak = metrics[metrics["rerun_group"] == "true_weak_scaling_track2"].copy()
    if not weak.empty:
        weak_summary = (
            weak.groupby("nodes")
            .agg(
                n_runs=("job_id", "count"),
                Ntrain=("Ntrain", "median"),
                global_train_batch_size=("global_train_batch_size", "median"),
                runtime_train_sec_mean=("runtime_train_sec", "mean"),
                train_samples_per_sec_mean=("train_samples_per_sec", "mean"),
                test_mse_mean=("test_mse", "mean"),
                test_mae_mean=("test_mae", "mean"),
                test_r2_mean=("test_r2", "mean"),
                gpus_total=("gpus_total", "median"),
            )
            .reset_index()
            .sort_values("nodes")
        )
        weak_summary["per_gpu_train_sps"] = weak_summary["train_samples_per_sec_mean"] / weak_summary["gpus_total"]
        base = weak_summary.loc[weak_summary["nodes"] == 1, "per_gpu_train_sps"]
        base_val = float(base.iloc[0]) if not base.empty else np.nan
        weak_summary["weak_efficiency_vs_1n"] = weak_summary["per_gpu_train_sps"] / base_val if not np.isnan(base_val) else np.nan
    else:
        weak_summary = pd.DataFrame(
            columns=[
                "nodes",
                "n_runs",
                "Ntrain",
                "global_train_batch_size",
                "runtime_train_sec_mean",
                "train_samples_per_sec_mean",
                "test_mse_mean",
                "test_mae_mean",
                "test_r2_mean",
                "gpus_total",
                "per_gpu_train_sps",
                "weak_efficiency_vs_1n",
            ]
        )
    weak_summary.to_csv(TABLES / f"rerun_true_weak_scaling_summary_{DATE_TAG}.csv", index=False)

    plot_summaries(metrics)

    total = int(merged.shape[0])
    completed_n = int(completed.shape[0])
    running_n = int((merged["effective_state"].str.upper() == "RUNNING").sum())
    pending_n = int((merged["effective_state"].str.upper() == "PENDING").sum())
    failed_n = int(
        merged["effective_state"].str.upper().str.contains("FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY", regex=True).sum()
    )

    lines = [
        f"# Rerun Analytics Snapshot ({DATE_TAG})",
        "",
        f"- Total planned jobs: {total}",
        f"- Completed: {completed_n}",
        f"- Running: {running_n}",
        f"- Pending: {pending_n}",
        f"- Failed/Cancelled/Timeout/OOM: {failed_n}",
        "",
        "## Completion by Rerun Group",
    ]
    for grp, g in merged.groupby("rerun_group"):
        c = int(g["effective_state"].str.upper().str.startswith("COMPLETED").sum())
        lines.append(f"- {grp}: {c}/{int(g.shape[0])} completed")

    lines.extend(
        [
            "",
            "## Analytics Files",
            f"- Status snapshot: `{TABLES / f'rerun_status_snapshot_{DATE_TAG}.csv'}`",
            f"- Completed metrics: `{TABLES / f'rerun_completed_metrics_{DATE_TAG}.csv'}`",
            f"- Group summary: `{TABLES / f'rerun_metric_summary_{DATE_TAG}.csv'}`",
            f"- Weak-scaling summary: `{TABLES / f'rerun_true_weak_scaling_summary_{DATE_TAG}.csv'}`",
        ]
    )

    if not weak_summary.empty:
        best = weak_summary.sort_values("nodes").iloc[-1]
        eff = best.get("weak_efficiency_vs_1n", np.nan)
        if not pd.isna(eff):
            lines.append("")
            lines.append(
                f"- Weak-scaling quick read: at {int(best['nodes'])} nodes, per-GPU throughput retention is {eff:.3f} vs 1-node."
            )

    if failed_n > 0:
        lines.append("")
        lines.append("## Failed Jobs")
        bad = merged[
            merged["effective_state"].str.upper().str.contains("FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY", regex=True)
        ][["job_id", "job_name", "effective_state"]]
        for _, r in bad.iterrows():
            lines.append(f"- {int(r['job_id'])} {r['job_name']}: {r['effective_state']}")

    note_path = NOTES / f"rerun_analytics_snapshot_{DATE_TAG}.md"
    note_path.write_text("\n".join(lines) + "\n")

    run_log = {
        "date_tag": DATE_TAG,
        "jobs_csv": str(JOBS_CSV),
        "total_jobs": total,
        "completed_jobs": completed_n,
        "running_jobs": running_n,
        "pending_jobs": pending_n,
        "failed_jobs": failed_n,
        "metrics_rows": int(metrics.shape[0]),
        "generated_tables_dir": str(TABLES),
        "generated_figures_dir": str(FIGURES),
    }
    (NOTES / f"rerun_analytics_log_{DATE_TAG}.json").write_text(json.dumps(run_log, indent=2))
    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
