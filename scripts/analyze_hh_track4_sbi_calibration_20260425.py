#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_prediction_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.name == "predictions_test.npz":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("predictions_test.npz"))


def rank_statistics(targets: np.ndarray, samples: np.ndarray) -> dict[str, object]:
    n_obs, n_samp, n_targets = samples.shape
    ranks = np.sum(samples < targets[:, None, :], axis=1)
    expected_mean = n_samp / 2.0
    expected_var = (n_samp * (n_samp + 2)) / 12.0

    per_target = []
    for j in range(n_targets):
        r = ranks[:, j].astype(np.float64)
        bins = np.bincount(r.astype(np.int64), minlength=n_samp + 1).astype(np.float64)
        expected = np.full(n_samp + 1, n_obs / (n_samp + 1), dtype=np.float64)
        chi2 = float(np.sum((bins - expected) ** 2 / np.where(expected == 0.0, 1.0, expected)))
        per_target.append(
            {
                "target_index": j,
                "rank_mean": float(np.mean(r)),
                "rank_var": float(np.var(r)),
                "expected_rank_mean": float(expected_mean),
                "expected_rank_var": float(expected_var),
                "chi2_uniformity": chi2,
            }
        )
    return {
        "overall_rank_mean": float(np.mean(ranks)),
        "overall_rank_var": float(np.var(ranks)),
        "expected_rank_mean": float(expected_mean),
        "expected_rank_var": float(expected_var),
        "per_target": per_target,
    }


def coverage_bundle(targets: np.ndarray, samples: np.ndarray, levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict[str, object]:
    out = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(samples, alpha, axis=1)
        upper = np.quantile(samples, 1.0 - alpha, axis=1)
        hit = (targets >= lower) & (targets <= upper)
        out[str(level)] = {
            "overall": float(np.mean(hit)),
            "per_target": [float(np.mean(hit[:, j])) for j in range(hit.shape[1])],
            "gap_overall": float(np.mean(hit) - level),
        }
    return out


def recalibrate_scale(targets: np.ndarray, samples: np.ndarray, grid: np.ndarray, levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict[str, object]:
    mean = np.mean(samples, axis=1, keepdims=True)
    centered = samples - mean
    best = None
    for scale in grid:
        scaled = mean + scale * centered
        cov = coverage_bundle(targets, scaled, levels=levels)
        mean_abs_gap = float(np.mean([abs(v["gap_overall"]) for v in cov.values()]))
        if best is None or mean_abs_gap < best["mean_abs_gap"]:
            best = {
                "scale": float(scale),
                "mean_abs_gap": mean_abs_gap,
                "coverage": cov,
            }
    assert best is not None
    return best


def analyze_file(predictions_file: Path) -> dict[str, object]:
    run_dir = predictions_file.parent
    data = np.load(predictions_file)
    targets = np.asarray(data["targets"], dtype=np.float32)
    samples = np.asarray(data["posterior_samples"], dtype=np.float32)
    selected = np.asarray(data["posterior_selected"], dtype=np.float32)
    summary_path = run_dir / "metrics_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    ranks = rank_statistics(targets, samples)
    coverage = coverage_bundle(targets, samples)
    recal = recalibrate_scale(targets, samples, grid=np.linspace(0.5, 2.5, 41))

    return {
        "run_dir": str(run_dir),
        "method": summary.get("method"),
        "density_estimator": summary.get("density_estimator"),
        "feature_mode": summary.get("feature_mode"),
        "seed": summary.get("seed"),
        "n_test": int(targets.shape[0]),
        "posterior_samples": int(samples.shape[1]),
        "selected_rule": summary.get("selected_rule"),
        "selected_mae": float(summary.get("posterior_selected_metrics", {}).get("mae", np.nan)),
        "selected_mse": float(summary.get("posterior_selected_metrics", {}).get("mse", np.nan)),
        "selected_r2": float(summary.get("posterior_selected_metrics", {}).get("r2", np.nan)),
        "overall_rank_mean": ranks["overall_rank_mean"],
        "overall_rank_var": ranks["overall_rank_var"],
        "expected_rank_mean": ranks["expected_rank_mean"],
        "expected_rank_var": ranks["expected_rank_var"],
        "coverage50_overall": coverage["0.5"]["overall"],
        "coverage80_overall": coverage["0.8"]["overall"],
        "coverage90_overall": coverage["0.9"]["overall"],
        "coverage_gap50": coverage["0.5"]["gap_overall"],
        "coverage_gap80": coverage["0.8"]["gap_overall"],
        "coverage_gap90": coverage["0.9"]["gap_overall"],
        "recal_scale": recal["scale"],
        "recal_mean_abs_gap": recal["mean_abs_gap"],
        "recal_coverage50_overall": recal["coverage"]["0.5"]["overall"],
        "recal_coverage80_overall": recal["coverage"]["0.8"]["overall"],
        "recal_coverage90_overall": recal["coverage"]["0.9"]["overall"],
        "rank_details": ranks,
        "coverage_details": coverage,
        "recalibration_details": recal,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {k: v for k, v in row.items() if k not in {"rank_details", "coverage_details", "recalibration_details"}}
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SBI calibration using saved posterior samples.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    files = list(iter_prediction_files([Path(p) for p in args.paths]))
    rows = [analyze_file(p) for p in files]
    write_csv(Path(args.csv_out), rows)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"n_files": len(rows), "csv_out": args.csv_out, "json_out": args.json_out}, indent=2))


if __name__ == "__main__":
    main()
