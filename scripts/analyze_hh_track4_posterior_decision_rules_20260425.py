#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import yaml
import sys

from hh_repo_utils import resolve_hh_dataset_root

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))
from data import _apply_scale_inverse, _apply_targets_transform, _resolve_split_array_cache_dir, preprocess_targets  # type: ignore  # noqa: E402
from run_hh_track4_aligned_multicurrent_classical_20260425 import load_one_current, load_params as classical_load_params  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze posterior decision rules across BayesFlow, Swyft, and SBI run directories.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--run-roots", required=True, help="Comma-separated directories containing completed run subdirectories.")
    ap.add_argument("--params", default="src/pytorch/configs/hh/params_sbi_hh_followup.yaml")
    ap.add_argument("--data-dir", default="concatenated_data.tar.gz")
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - float(np.sum(np.square(diff))) / denom if denom > 0.0 else 0.0
    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def postprocess_targets_array(y_norm: np.ndarray, targets_scale: dict) -> np.ndarray:
    restored = _apply_scale_inverse(y_norm.astype(np.float32, copy=False), targets_scale)
    restored = _apply_targets_transform(restored, targets_scale.get("transform"), inverse=True, array_name="targets")
    return restored.astype(np.float32, copy=False)


def targets_scale(args: argparse.Namespace) -> dict:
    params_path = str((REPO / args.params).resolve() if not Path(args.params).is_absolute() else Path(args.params))
    params = classical_load_params(params_path)
    _, targets = load_one_current(
        params,
        curr="0.1",
        data_dir=args.data_dir,
        data_prefix=args.data_prefix,
        n_train=args.n_train,
        n_validate=args.n_validate,
        n_test=args.n_test,
        feature_mode="raw",
    )
    logger = logging.getLogger("decision_rules_scale")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return preprocess_targets(targets, params, logger=logger)


def sample_medoid(samples: np.ndarray) -> np.ndarray:
    # samples: [N, S, D]
    diffs = samples[:, :, None, :] - samples[:, None, :, :]
    dist = np.sum(np.square(diffs), axis=3)
    medoid_idx = np.argmin(np.sum(dist, axis=2), axis=1)
    return samples[np.arange(samples.shape[0]), medoid_idx]


def gaussian_pseudo_samples(mean: np.ndarray, std: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = np.maximum(std, 1.0e-8)
    noise = rng.normal(size=(mean.shape[0], n_samples, mean.shape[1])).astype(np.float32)
    return (mean[:, None, :] + std[:, None, :] * noise).astype(np.float32, copy=False)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.empty((values.shape[0], values.shape[2]), dtype=np.float32)
    for i in range(values.shape[0]):
        for j in range(values.shape[2]):
            idx = np.argsort(values[i, :, j])
            v = values[i, idx, j]
            w = weights[i, idx, j]
            w = w / np.sum(w)
            cdf = np.cumsum(w)
            out[i, j] = v[np.searchsorted(cdf, 0.5, side="left")]
    return out


def swyft_map_bank(theta_bank_norm: np.ndarray, weights_1d: np.ndarray) -> np.ndarray:
    score = np.sum(np.log(np.clip(weights_1d, 1.0e-12, None)), axis=2)
    idx = np.argmax(score, axis=1)
    return theta_bank_norm[idx]


def save_rule_predictions(run_dir: Path, rule_name: str, pred: np.ndarray, target: np.ndarray) -> Path:
    out = run_dir / f"predictions_{rule_name}.npz"
    np.savez_compressed(out, test_pred=pred.astype(np.float32), test_true=target.astype(np.float32))
    return out


def process_bayesflow_run(run_dir: Path, rows: list[dict[str, object]], targets_scale_obj: dict) -> None:
    arr = np.load(run_dir / "predictions_test.npz")
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    if "posterior_samples_norm" in arr.files:
        samples_norm = np.asarray(arr["posterior_samples_norm"], dtype=np.float32)
        samples = postprocess_targets_array(samples_norm.reshape(-1, samples_norm.shape[-1]), targets_scale_obj).reshape(samples_norm.shape)
    else:
        mean = np.asarray(arr["test_pred"], dtype=np.float32)
        std = np.asarray(arr["posterior_std"], dtype=np.float32)
        samples = gaussian_pseudo_samples(mean, std, n_samples=128, seed=20260425)
    rules = {
        "mean": np.mean(samples, axis=1),
        "median": np.median(samples, axis=1),
        "medoid": sample_medoid(samples),
    }
    for rule_name, pred in rules.items():
        m = metrics(y_true, pred)
        pred_file = save_rule_predictions(run_dir, rule_name, pred, y_true)
        rows.append(
            {
                "run_name": run_dir.name,
                "family": "bayesflow",
                "decision_rule": rule_name,
                **m,
                "prediction_file": str(pred_file),
            }
        )


def process_swyft_run(run_dir: Path, rows: list[dict[str, object]], targets_scale_obj: dict) -> None:
    arr = np.load(run_dir / "predictions_test.npz")
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    if "theta_bank_norm" in arr.files and "posterior_weights_1d" in arr.files:
        theta_bank_norm = np.asarray(arr["theta_bank_norm"], dtype=np.float32)
        weights = np.asarray(arr["posterior_weights_1d"], dtype=np.float32)
        theta_bank = postprocess_targets_array(theta_bank_norm, targets_scale_obj)
        weighted_mean = np.sum(theta_bank[None, :, :] * weights, axis=1)
        repeated_bank = np.broadcast_to(theta_bank[None, :, :], weights.shape)
        weighted_med = weighted_median(repeated_bank, weights)
        map_theta = postprocess_targets_array(swyft_map_bank(theta_bank_norm, weights), targets_scale_obj)
        rules = {
            "mean": weighted_mean,
            "median": weighted_med,
            "map_bank": map_theta,
        }
    else:
        mean = np.asarray(arr["test_pred"], dtype=np.float32)
        std = np.asarray(arr["posterior_std"], dtype=np.float32)
        samples = gaussian_pseudo_samples(mean, std, n_samples=128, seed=20260425)
        rules = {
            "mean": np.mean(samples, axis=1),
            "median": np.median(samples, axis=1),
            "medoid": sample_medoid(samples),
        }
    for rule_name, pred in rules.items():
        m = metrics(y_true, pred)
        pred_file = save_rule_predictions(run_dir, rule_name, pred, y_true)
        rows.append(
            {
                "run_name": run_dir.name,
                "family": "swyft",
                "decision_rule": rule_name,
                **m,
                "prediction_file": str(pred_file),
            }
        )


def process_pool_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
    arr = np.load(run_dir / "predictions_test.npz")
    y_true = np.asarray(arr["targets"], dtype=np.float32)
    rules = {
        "mean": np.asarray(arr["posterior_mean"], dtype=np.float32),
        "median": np.asarray(arr["posterior_median"], dtype=np.float32),
        "selected": np.asarray(arr["posterior_selected"], dtype=np.float32),
        "medoid": sample_medoid(np.asarray(arr["posterior_samples"], dtype=np.float32)),
    }
    for rule_name, pred in rules.items():
        m = metrics(y_true, pred)
        pred_file = save_rule_predictions(run_dir, rule_name, pred, y_true)
        rows.append(
            {
                "run_name": run_dir.name,
                "family": "poolseq_sbi",
                "decision_rule": rule_name,
                **m,
                "prediction_file": str(pred_file),
            }
        )


def discover_runs(roots: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    bayesflow_runs: list[Path] = []
    swyft_runs: list[Path] = []
    pool_runs: list[Path] = []
    for root in roots:
        for run_dir in sorted(root.glob("*")):
            metrics_file = run_dir / "metrics_summary.json"
            pred_file = run_dir / "predictions_test.npz"
            if not (metrics_file.exists() and pred_file.exists()):
                continue
            metrics_obj = json.loads(metrics_file.read_text())
            framework = str(metrics_obj.get("framework", ""))
            method = str(metrics_obj.get("method", ""))
            if framework.startswith("bayesflow"):
                bayesflow_runs.append(run_dir)
            elif framework.startswith("swyft"):
                swyft_runs.append(run_dir)
            elif method in {"snpe", "snle", "snre", "fmpe", "npse"}:
                pool_runs.append(run_dir)
    return bayesflow_runs, swyft_runs, pool_runs


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    roots = [Path(token.strip()) for token in args.run_roots.split(",") if token.strip()]
    if not roots:
        raise SystemExit("Expected at least one run root.")
    scales = targets_scale(args)
    bayesflow_runs, swyft_runs, pool_runs = discover_runs(roots)
    for run_dir in bayesflow_runs:
        process_bayesflow_run(run_dir, rows, scales)
    for run_dir in swyft_runs:
        process_swyft_run(run_dir, rows, scales)
    for run_dir in pool_runs:
        process_pool_run(run_dir, rows)
    rows.sort(key=lambda r: (r["mae"], -r["r2"]))
    out_csv = save_dir / "hh_track4_posterior_decision_rule_comparison_20260425.csv"
    write_csv(out_csv, rows)
    best_rows = {}
    for row in rows:
        best_rows.setdefault(row["run_name"], row)
    report = [
        "# HH Track4 Posterior Decision Rules",
        "",
        f"- summary csv: `{out_csv}`",
        "",
        "| run | family | best rule | MAE | R2 |",
        "|---|---|---|---:|---:|",
    ]
    for run_name, row in sorted(best_rows.items()):
        report.append(f"| `{run_name}` | `{row['family']}` | `{row['decision_rule']}` | {row['mae']:.6f} | {row['r2']:.6f} |")
    (save_dir / "hh_track4_posterior_decision_rule_comparison_20260425.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
