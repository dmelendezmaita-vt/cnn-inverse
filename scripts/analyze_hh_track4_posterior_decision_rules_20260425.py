#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import yaml
import sys

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
sys.path.append(str(REPO / "pytorch"))
from data import _apply_scale_inverse, _apply_targets_transform, _resolve_split_array_cache_dir, preprocess_targets  # type: ignore  # noqa: E402

RUNALL = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
POOL = REPO / "data/important_notes/optimization_track_20260425_tc_hh_track4_pool_sequential_sbi"
PARAMS_FILE = REPO / "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"


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


def targets_scale() -> dict:
    params = load_params(str(PARAMS_FILE))
    params["data"]["data_dir"] = str(
        REPO
        / "data"
        / "important_notes"
        / "optimization_track_20260411_v100_interactive"
        / "shared_data"
        / "track4_hh_full"
        / "concatenated_data"
    )
    params["data"]["data_prefix"] = "concatenated_data"
    params["data"]["curr"] = "0.1"
    params["data"]["Ntrain"] = 4096
    params["data"]["Nvalidate"] = 1024
    params["data"]["Ntest"] = 128
    params["data"]["split_array_cache_enabled"] = True
    cache_dir = _resolve_split_array_cache_dir(params["data"])
    target_train = np.load(cache_dir / "targets_train.npy", mmap_mode="r")
    target_validate = np.load(cache_dir / "targets_validate.npy", mmap_mode="r")
    target_test = np.load(cache_dir / "targets_test.npy", mmap_mode="r")
    targets = {
        "train": np.array(target_train[:4096], copy=True).astype(np.float32, copy=False),
        "validate": np.array(target_validate[:1024], copy=True).astype(np.float32, copy=False),
        "test": np.array(target_test[:128], copy=True).astype(np.float32, copy=False),
    }
    logger = logging.getLogger("decision_rules_scale")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return preprocess_targets(targets, params, logger=logger)


TARGETS_SCALE = targets_scale()


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


def process_bayesflow_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
    arr = np.load(run_dir / "predictions_test.npz")
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    if "posterior_samples_norm" in arr.files:
        samples_norm = np.asarray(arr["posterior_samples_norm"], dtype=np.float32)
        samples = postprocess_targets_array(samples_norm.reshape(-1, samples_norm.shape[-1]), TARGETS_SCALE).reshape(samples_norm.shape)
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


def process_swyft_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
    arr = np.load(run_dir / "predictions_test.npz")
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    if "theta_bank_norm" in arr.files and "posterior_weights_1d" in arr.files:
        theta_bank_norm = np.asarray(arr["theta_bank_norm"], dtype=np.float32)
        weights = np.asarray(arr["posterior_weights_1d"], dtype=np.float32)
        theta_bank = postprocess_targets_array(theta_bank_norm, TARGETS_SCALE)
        weighted_mean = np.sum(theta_bank[None, :, :] * weights, axis=1)
        repeated_bank = np.broadcast_to(theta_bank[None, :, :], weights.shape)
        weighted_med = weighted_median(repeated_bank, weights)
        map_theta = postprocess_targets_array(swyft_map_bank(theta_bank_norm, weights), TARGETS_SCALE)
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


def discover_runall_posterior_runs() -> tuple[list[Path], list[Path]]:
    bayesflow_runs: list[Path] = []
    swyft_runs: list[Path] = []
    for run_dir in sorted((RUNALL / "runs").glob("*")):
        metrics_file = run_dir / "metrics_summary.json"
        pred_file = run_dir / "predictions_test.npz"
        if not (metrics_file.exists() and pred_file.exists()):
            continue
        metrics_obj = json.loads(metrics_file.read_text())
        framework = str(metrics_obj.get("framework", ""))
        if framework.startswith("bayesflow"):
            bayesflow_runs.append(run_dir)
        elif framework.startswith("swyft"):
            swyft_runs.append(run_dir)
    return bayesflow_runs, swyft_runs


def main() -> None:
    rows: list[dict[str, object]] = []
    bayesflow_runs, swyft_runs = discover_runall_posterior_runs()
    pool_runs = [
        "fmpe_mlp_r2_pool4096_final2048",
        "snpe_maf_r3_pool4096_final2048",
    ]
    for run_dir in bayesflow_runs:
        process_bayesflow_run(run_dir, rows)
    for run_dir in swyft_runs:
        process_swyft_run(run_dir, rows)
    for name in pool_runs:
        process_pool_run(POOL / "runs" / name, rows)
    rows.sort(key=lambda r: (r["mae"], -r["r2"]))
    out_csv = RUNALL / "tables" / "hh_track4_posterior_decision_rule_comparison_20260425.csv"
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
    (RUNALL / "reports" / "hh_track4_posterior_decision_rule_comparison_20260425.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
