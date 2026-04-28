#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analyze_hh_track4_posterior_decision_rules_20260425 import TARGETS_SCALE, postprocess_targets_array  # noqa: E402


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
RUNALL = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
POOL = REPO / "data/important_notes/optimization_track_20260425_tc_hh_track4_pool_sequential_sbi"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_stats(samples: np.ndarray, theta_true: np.ndarray) -> dict[str, float]:
    ks_values = []
    mean_values = []
    for j in range(theta_true.shape[1]):
        ranks = np.sum(samples[:, :, j] < theta_true[:, None, j], axis=1).astype(np.float64)
        u = (ranks + 0.5) / (samples.shape[1] + 1.0)
        u_sorted = np.sort(u)
        emp = (np.arange(len(u_sorted), dtype=np.float64) + 1.0) / len(u_sorted)
        ks_values.append(float(np.max(np.abs(emp - u_sorted))))
        mean_values.append(float(np.mean(u)))
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    coverage = np.mean((theta_true >= q05) & (theta_true <= q95), axis=0)
    return {
        "rank_ks_mean": float(np.mean(ks_values)),
        "rank_ks_max": float(np.max(ks_values)),
        "rank_u_mean": float(np.mean(mean_values)),
        "coverage_90_mean": float(np.mean(coverage)),
    }


def sample_swyft_bank(arr: np.lib.npyio.NpzFile, seed: int, draws: int = 128) -> tuple[np.ndarray, np.ndarray]:
    theta_bank_norm = np.asarray(arr["theta_bank_norm"], dtype=np.float32)
    theta_bank = postprocess_targets_array(theta_bank_norm, TARGETS_SCALE)
    weights = np.asarray(arr["posterior_weights_1d"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    samples = np.empty((weights.shape[0], draws, theta_bank.shape[1]), dtype=np.float32)
    for i in range(weights.shape[0]):
        score = np.sum(np.log(np.clip(weights[i], 1.0e-12, None)), axis=1)
        prob = np.exp(score - np.max(score))
        prob /= np.sum(prob)
        idx = rng.choice(theta_bank.shape[0], size=draws, replace=True, p=prob)
        samples[i] = theta_bank[idx]
    return samples, np.asarray(arr["test_true"], dtype=np.float32)


def discover_run_dirs() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for run_dir in sorted((RUNALL / "runs").glob("*")):
        metrics_file = run_dir / "metrics_summary.json"
        pred_file = run_dir / "predictions_test.npz"
        if not (metrics_file.exists() and pred_file.exists()):
            continue
        metrics = json.loads(metrics_file.read_text())
        framework = str(metrics.get("framework", ""))
        if framework.startswith("bayesflow") or framework.startswith("swyft"):
            rows.append((framework, run_dir))
    for run_dir in sorted((POOL / "runs").glob("*")):
        pred_file = run_dir / "predictions_test.npz"
        if pred_file.exists():
            rows.append(("poolseq_sbi", run_dir))
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    for i, (framework, run_dir) in enumerate(discover_run_dirs(), start=1):
        arr = np.load(run_dir / "predictions_test.npz")
        if "posterior_samples_norm" in arr.files and "test_true" in arr.files:
            samples_norm = np.asarray(arr["posterior_samples_norm"], dtype=np.float32)
            samples = postprocess_targets_array(samples_norm.reshape(-1, samples_norm.shape[-1]), TARGETS_SCALE).reshape(samples_norm.shape)
            theta_true = np.asarray(arr["test_true"], dtype=np.float32)
        elif {"theta_bank_norm", "posterior_weights_1d", "test_true"}.issubset(arr.files):
            samples, theta_true = sample_swyft_bank(arr, 20260426 + i)
        elif {"posterior_samples", "targets"}.issubset(arr.files):
            samples = np.asarray(arr["posterior_samples"], dtype=np.float32)
            theta_true = np.asarray(arr["targets"], dtype=np.float32)
        else:
            continue
        stats = rank_stats(samples, theta_true)
        rows.append({"run_name": run_dir.name, "framework": framework, **stats})
    rows.sort(key=lambda r: (r["rank_ks_mean"], -r["coverage_90_mean"]))
    out_csv = RUNALL / "tables" / "hh_track4_posterior_rank_diagnostics_20260426.csv"
    write_csv(out_csv, rows)
    report = [
        "# HH Track4 Posterior Rank Diagnostics",
        "",
        f"- summary csv: `{out_csv}`",
        "",
        "| run | framework | rank_ks_mean | rank_ks_max | coverage_90_mean |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| `{row['run_name']}` | `{row['framework']}` | {row['rank_ks_mean']:.6f} | {row['rank_ks_max']:.6f} | {row['coverage_90_mean']:.6f} |"
        )
    (RUNALL / "reports" / "hh_track4_posterior_rank_diagnostics_20260426.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
