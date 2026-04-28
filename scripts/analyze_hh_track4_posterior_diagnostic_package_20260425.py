#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
RUNALL_ROOT = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
POOL_ROOT = REPO / "data/important_notes/optimization_track_20260425_tc_hh_track4_pool_sequential_sbi"
OUT_ROOT = RUNALL_ROOT / "reports"
TABLES = RUNALL_ROOT / "tables"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gaussian_pseudo_samples(mean: np.ndarray, std: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = np.maximum(std, 1.0e-8)
    noise = rng.normal(size=(mean.shape[0], n_samples, mean.shape[1])).astype(np.float32)
    return (mean[:, None, :] + std[:, None, :] * noise).astype(np.float32, copy=False)


def interval_coverages(samples: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    q05 = np.quantile(samples, 0.05, axis=1)
    q25 = np.quantile(samples, 0.25, axis=1)
    q75 = np.quantile(samples, 0.75, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    cov50 = (targets >= q25) & (targets <= q75)
    cov90 = (targets >= q05) & (targets <= q95)
    return {
        "coverage50_overall": float(np.mean(cov50)),
        "coverage90_overall": float(np.mean(cov90)),
        "coverage50_per_target": np.mean(cov50, axis=0).astype(float).tolist(),
        "coverage90_per_target": np.mean(cov90, axis=0).astype(float).tolist(),
        "interval50_width_mean": float(np.mean(q75 - q25)),
        "interval90_width_mean": float(np.mean(q95 - q05)),
    }


def rank_uniformity(samples: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    n_samples = samples.shape[1]
    rows = []
    ks_per_target = []
    mean_abs_dev_per_target = []
    for j in range(samples.shape[2]):
        ranks = np.sum(samples[:, :, j] < targets[:, None, j], axis=1).astype(np.float64)
        u = np.sort((ranks + 0.5) / (n_samples + 1.0))
        empirical = (np.arange(u.size, dtype=np.float64) + 1.0) / u.size
        ks = float(np.max(np.abs(empirical - u))) if u.size else 0.0
        mean_abs_dev = float(np.mean(np.abs(u - 0.5))) if u.size else 0.0
        rows.append(
            {
                "target_index": j + 1,
                "rank_ks": ks,
                "rank_mean_abs_dev_from_half": mean_abs_dev,
            }
        )
        ks_per_target.append(ks)
        mean_abs_dev_per_target.append(mean_abs_dev)
    return {
        "rank_ks_max": float(np.max(ks_per_target)) if ks_per_target else 0.0,
        "rank_ks_mean": float(np.mean(ks_per_target)) if ks_per_target else 0.0,
        "rank_mean_abs_dev_mean": float(np.mean(mean_abs_dev_per_target)) if mean_abs_dev_per_target else 0.0,
        "per_target_rows": rows,
    }


def standardized_error(mean: np.ndarray, std: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    std = np.maximum(std, 1.0e-8)
    z = (mean - targets) / std
    return {
        "abs_z_mean": float(np.mean(np.abs(z))),
        "abs_z_median": float(np.median(np.abs(z))),
        "abs_z_per_target": np.mean(np.abs(z), axis=0).astype(float).tolist(),
        "sharpness_mean_std": float(np.mean(std)),
    }


def classifier_proxy(samples: np.ndarray, targets: np.ndarray, seed: int) -> dict[str, float]:
    x_samples = samples.reshape(-1, samples.shape[2]).astype(np.float32, copy=False)
    repeated_targets = np.repeat(targets, samples.shape[1], axis=0).astype(np.float32, copy=False)
    x = np.concatenate([x_samples, repeated_targets], axis=0)
    y = np.concatenate(
        [
            np.zeros(x_samples.shape[0], dtype=np.int64),
            np.ones(repeated_targets.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    class0 = np.flatnonzero(y == 0)
    class1 = np.flatnonzero(y == 1)
    rng.shuffle(class0)
    rng.shuffle(class1)
    split0 = int(0.75 * class0.size)
    split1 = int(0.75 * class1.size)
    train_idx = np.concatenate([class0[:split0], class1[:split1]], axis=0)
    test_idx = np.concatenate([class0[split0:], class1[split1:]], axis=0)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    centroid0 = np.mean(x_train[y_train == 0], axis=0)
    centroid1 = np.mean(x_train[y_train == 1], axis=0)
    score0 = np.sum(np.square(x_test - centroid0.reshape(1, -1)), axis=1)
    score1 = np.sum(np.square(x_test - centroid1.reshape(1, -1)), axis=1)
    pred = (score1 < score0).astype(np.int64)
    decision = (score0 - score1).astype(np.float64)
    accuracy = float(np.mean(pred == y_test))
    order = np.argsort(decision)
    sorted_y = y_test[order]
    n_pos = int(np.sum(sorted_y == 1))
    n_neg = int(np.sum(sorted_y == 0))
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        ranks = np.arange(1, sorted_y.size + 1, dtype=np.float64)
        rank_sum_pos = float(np.sum(ranks[sorted_y == 1]))
        auc = float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return {
        "classifier_proxy_accuracy": accuracy,
        "classifier_proxy_auc": auc,
    }


def load_run_arrays(pred_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    arr = np.load(pred_path)
    if "targets" in arr.files and "posterior_samples" in arr.files:
        targets = np.asarray(arr["targets"], dtype=np.float32)
        samples = np.asarray(arr["posterior_samples"], dtype=np.float32)
        mean = np.asarray(arr["posterior_mean"], dtype=np.float32)
        std = np.asarray(arr["posterior_std"], dtype=np.float32)
        source = "exact_saved_samples"
        return targets, mean, std, samples, source
    targets = np.asarray(arr["test_true"], dtype=np.float32)
    mean = np.asarray(arr["test_pred"], dtype=np.float32)
    std = np.asarray(arr["posterior_std"], dtype=np.float32)
    samples = gaussian_pseudo_samples(mean, std, n_samples=128, seed=seed)
    source = "gaussian_approx_from_mean_std"
    return targets, mean, std, samples, source


def run_specs() -> list[tuple[str, Path, Path]]:
    return [
        (
            "swyft_native_meanstd",
            RUNALL_ROOT / "runs/swyft_native_v100_20260425/metrics_summary.json",
            RUNALL_ROOT / "runs/swyft_native_v100_20260425/predictions_test.npz",
        ),
        (
            "swyft_native_concat_light",
            RUNALL_ROOT / "runs/swyft_native_concat_small_v100_20260425/metrics_summary.json",
            RUNALL_ROOT / "runs/swyft_native_concat_small_v100_20260425/predictions_test.npz",
        ),
        (
            "swyft_tmnre_style_native",
            RUNALL_ROOT / "runs/swyft_tmnre_native_v100_20260425/metrics_summary.json",
            RUNALL_ROOT / "runs/swyft_tmnre_native_v100_20260425/predictions_test.npz",
        ),
        (
            "bayesflow_native_concat",
            RUNALL_ROOT / "runs/bayesflow_native_v100_20260425/metrics_summary.json",
            RUNALL_ROOT / "runs/bayesflow_native_v100_20260425/predictions_test.npz",
        ),
        (
            "bayesflow_native_meanstd",
            RUNALL_ROOT / "runs/bayesflow_native_meanstd_v100_20260425/metrics_summary.json",
            RUNALL_ROOT / "runs/bayesflow_native_meanstd_v100_20260425/predictions_test.npz",
        ),
        (
            "poolseq_fmpe",
            POOL_ROOT / "runs/fmpe_mlp_r2_pool4096_final2048/metrics_summary.json",
            POOL_ROOT / "runs/fmpe_mlp_r2_pool4096_final2048/predictions_test.npz",
        ),
        (
            "poolseq_snpe",
            POOL_ROOT / "runs/snpe_maf_r3_pool4096_final2048/metrics_summary.json",
            POOL_ROOT / "runs/snpe_maf_r3_pool4096_final2048/predictions_test.npz",
        ),
    ]


def extract_test_metrics(metrics_meta: dict) -> dict[str, float]:
    if metrics_meta.get("test_metrics") is not None:
        return metrics_meta["test_metrics"]
    for key in ["posterior_selected_metrics", "posterior_mean_metrics", "posterior_median_metrics", "posterior_map_metrics"]:
        if metrics_meta.get(key) is not None:
            return metrics_meta[key]
    return {key: metrics_meta.get(key) for key in ["mae", "mse", "r2"] if metrics_meta.get(key) is not None}


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_target_rows = []
    json_index = {}
    for idx, (name, metrics_path, pred_path) in enumerate(run_specs()):
        if not metrics_path.exists() or not pred_path.exists():
            continue
        metrics_meta = load_json(metrics_path)
        test_metrics = extract_test_metrics(metrics_meta)
        targets, mean, std, samples, source = load_run_arrays(pred_path, seed=20260425 + idx)
        interval = interval_coverages(samples, targets)
        ranks = rank_uniformity(samples, targets)
        z = standardized_error(mean, std, targets)
        clf = classifier_proxy(samples, targets, seed=20260425 + idx)
        row = {
            "representative": name,
            "posterior_source": source,
            "mae": float(test_metrics.get("mae")),
            "mse": float(test_metrics.get("mse")),
            "r2": float(test_metrics.get("r2")),
            "coverage50_overall": interval["coverage50_overall"],
            "coverage90_overall": interval["coverage90_overall"],
            "interval50_width_mean": interval["interval50_width_mean"],
            "interval90_width_mean": interval["interval90_width_mean"],
            "abs_z_mean": z["abs_z_mean"],
            "abs_z_median": z["abs_z_median"],
            "sharpness_mean_std": z["sharpness_mean_std"],
            "rank_ks_max": ranks["rank_ks_max"],
            "rank_ks_mean": ranks["rank_ks_mean"],
            "rank_mean_abs_dev_mean": ranks["rank_mean_abs_dev_mean"],
            "classifier_proxy_accuracy": clf["classifier_proxy_accuracy"],
            "classifier_proxy_auc": clf["classifier_proxy_auc"],
            "metrics_source": str(metrics_path),
            "predictions_source": str(pred_path),
        }
        summary_rows.append(row)
        for per_target in ranks["per_target_rows"]:
            per_target_rows.append(
                {
                    "representative": name,
                    **per_target,
                    "coverage50_target": float(interval["coverage50_per_target"][per_target["target_index"] - 1]),
                    "coverage90_target": float(interval["coverage90_per_target"][per_target["target_index"] - 1]),
                    "abs_z_target": float(z["abs_z_per_target"][per_target["target_index"] - 1]),
                }
            )
        json_index[name] = {
            "summary": row,
            "interval": interval,
            "ranks": ranks,
            "standardized_error": z,
            "classifier_proxy": clf,
        }

    summary_csv = TABLES / "hh_track4_posterior_diagnostic_package_summary_20260425.csv"
    target_csv = TABLES / "hh_track4_posterior_diagnostic_package_per_target_20260425.csv"
    summary_json = OUT_ROOT / "hh_track4_posterior_diagnostic_package_20260425.json"
    report_md = OUT_ROOT / "hh_track4_posterior_diagnostic_package_20260425.md"

    write_csv(summary_csv, summary_rows)
    write_csv(target_csv, per_target_rows)
    summary_json.write_text(json.dumps(json_index, indent=2) + "\n")

    lines = [
        "# HH Track4 Posterior Diagnostic Package",
        "",
        "This package combines interval coverage, rank-uniformity diagnostics, standardized error summaries, and a classifier-style separability proxy.",
        "",
        f"- summary table: `{summary_csv}`" if summary_rows else "- no runs summarized",
        f"- per-target table: `{target_csv}`" if per_target_rows else "",
        f"- json summary: `{summary_json}`" if summary_rows else "",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| representative | source | MAE | R2 | cov50 | cov90 | abs_z_mean | rank_ks_max | clf_auc |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| `{row['representative']}` | `{row['posterior_source']}` | {row['mae']:.4f} | {row['r2']:.4f} | "
                f"{row['coverage50_overall']:.4f} | {row['coverage90_overall']:.4f} | {row['abs_z_mean']:.4f} | "
                f"{row['rank_ks_max']:.4f} | {row['classifier_proxy_auc']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Classifier proxy uses a nearest-centroid heldout separability check between posterior draws and the corresponding empirical target cloud. It is a practical separability check, not a literal L-C2ST implementation.",
                "",
                "Runs without saved posterior samples are summarized through a Gaussian approximation constructed from the saved posterior mean and posterior standard deviation.",
                "",
            ]
        )
    report_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
