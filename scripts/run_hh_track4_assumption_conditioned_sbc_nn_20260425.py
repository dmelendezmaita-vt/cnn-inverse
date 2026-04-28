#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hh_track4_assumption_conditioned_compact_hh_utils_20260425 import (
    CURRENT_LABELS,
    CompactHHSimulationConfig,
    bundle_feature_vector,
    sample_prior_thetas,
    simulate_theta_bundle,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SBC-style surrogate posterior validation using nearest-neighbor ABC over the compact HH simulator.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n-bank", type=int, default=512)
    ap.add_argument("--n-query", type=int, default=64)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--downsample-points", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--trace-length", type=int, default=2000)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--pulse-start-frac", type=float, default=0.10)
    ap.add_argument("--pulse-end-frac", type=float, default=0.90)
    ap.add_argument("--g-scale-range", default="0.2,5.0")
    ap.add_argument("--tau-scale-range", default="0.5,2.0")
    ap.add_argument("--gl-scale-range", default="0.1,5.0")
    ap.add_argument("--current-gain-bounds", default="2.0,80.0")
    return ap.parse_args()


def parse_pair(text: str) -> tuple[float, float]:
    left, right = [float(x.strip()) for x in text.split(",")]
    return left, right


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_diagnostics(posterior_samples: np.ndarray, theta_true: np.ndarray) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows = []
    ks = []
    for j in range(theta_true.shape[1]):
        ranks = np.sum(posterior_samples[:, :, j] < theta_true[:, None, j], axis=1).astype(np.float64)
        u = np.sort((ranks + 0.5) / (posterior_samples.shape[1] + 1.0))
        emp = (np.arange(len(u), dtype=np.float64) + 1.0) / len(u)
        ks_j = float(np.max(np.abs(emp - u))) if len(u) else 0.0
        ks.append(ks_j)
        rows.append({"target_index": j + 1, "rank_ks": ks_j, "rank_mean": float(np.mean(u))})
    return rows, {"rank_ks_mean": float(np.mean(ks)), "rank_ks_max": float(np.max(ks))}


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config = CompactHHSimulationConfig(
        trace_length=args.trace_length,
        dt_ms=args.dt_ms,
        pulse_start_frac=args.pulse_start_frac,
        pulse_end_frac=args.pulse_end_frac,
        g_scale_range=parse_pair(args.g_scale_range),
        tau_scale_range=parse_pair(args.tau_scale_range),
        gl_scale_range=parse_pair(args.gl_scale_range),
        max_abs_voltage=5000.0,
    )
    theta_bank = sample_prior_thetas(args.n_bank, args.seed, parse_pair(args.current_gain_bounds))
    bank_features = np.empty((args.n_bank, len(CURRENT_LABELS) * (args.downsample_points + 12)), dtype=np.float32)
    for i in range(args.n_bank):
        bundle, is_ok, _ = simulate_theta_bundle(theta_bank[i], CURRENT_LABELS, config)
        if not is_ok:
            raise RuntimeError("non-finite bank simulation")
        bank_features[i] = bundle_feature_vector(bundle, args.downsample_points, include_summary=True)
    theta_query = sample_prior_thetas(args.n_query, args.seed + 1, parse_pair(args.current_gain_bounds))
    posterior = np.empty((args.n_query, args.k, theta_bank.shape[1]), dtype=np.float32)
    rows = []
    for i in range(args.n_query):
        bundle, is_ok, _ = simulate_theta_bundle(theta_query[i], CURRENT_LABELS, config)
        if not is_ok:
            raise RuntimeError("non-finite query simulation")
        feat = bundle_feature_vector(bundle, args.downsample_points, include_summary=True)
        d = np.mean(np.square(bank_features - feat.reshape(1, -1)), axis=1)
        idx = np.argsort(d)[: args.k]
        posterior[i] = theta_bank[idx]
        rows.append({"query_id": i, "nearest_distance_mean": float(np.mean(d[idx])), "nearest_distance_min": float(np.min(d[idx]))})
    rank_rows, summary = rank_diagnostics(posterior, theta_query)
    write_csv(save_dir / "sbc_nn_query_summary.csv", rows)
    write_csv(save_dir / "sbc_nn_rank_summary.csv", rank_rows)
    manifest = {
        "assumption_conditioned": True,
        "method_family": "sbc_style_nearest_neighbor_abc",
        "n_bank": args.n_bank,
        "n_query": args.n_query,
        "k": args.k,
        "rank_summary": summary,
    }
    (save_dir / "sbc_nn_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# HH Track4 Assumption-Conditioned SBC-Style NN-ABC",
        "",
        f"- rank KS mean: `{summary['rank_ks_mean']:.6f}`",
        f"- rank KS max: `{summary['rank_ks_max']:.6f}`",
        "",
    ]
    (save_dir / "sbc_nn_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
