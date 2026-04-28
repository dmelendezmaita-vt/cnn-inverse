#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / "optimization_track_20260421_hh_track4_checkpoint1_analysis"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
FIGS = OUT_ROOT / "figures"
SHARED_DATA_DIR = (
    IMPORTANT
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)
REFERENCE_PARAMS = (
    REPO
    / "src/pytorch/configs/reruns_20260421_v100_hh_track4_classical_data_scaling/track4_hh_full/params_t4_extra_trees_500_n12288_n1_s1119.yaml"
)

MODEL_SPECS = [
    {
        "label": "neural_effnet_beta005_invvar_e500",
        "strategy_id": "effnet_beta005_invvar_e500",
        "registry": IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_top3_confirmation"
        / "tables"
        / "a30_hh_track4_top3_confirmation_registry_20260420_a30_hh_track4_top3_confirmation.csv",
    },
    {
        "label": "extra_trees_500_mse_anchor",
        "strategy_id": "extra_trees_500",
        "registry": IMPORTANT
        / "optimization_track_20260421_v100_hh_track4_classical_stacking"
        / "tables"
        / "v100_hh_track4_classical_stacking_registry_20260421_v100_hh_track4_classical_stacking.csv",
    },
    {
        "label": "random_forest_500_n12288",
        "strategy_id": "random_forest_500_n12288",
        "registry": IMPORTANT
        / "optimization_track_20260421_v100_hh_track4_classical_data_scaling"
        / "tables"
        / "v100_hh_track4_classical_data_scaling_registry_20260421_v100_hh_track4_classical_data_scaling.csv",
    },
    {
        "label": "knn_k11_n12288",
        "strategy_id": "knn_k11_n12288",
        "registry": IMPORTANT
        / "optimization_track_20260421_v100_hh_track4_classical_data_scaling"
        / "tables"
        / "v100_hh_track4_classical_data_scaling_registry_20260421_v100_hh_track4_classical_data_scaling.csv",
    },
]

FRONTIER_CROSSOVER_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260421_v100_hh_track4_frontier_crossover"
    / "tables"
    / "v100_hh_track4_frontier_crossover_registry_20260421_v100_hh_track4_frontier_crossover.csv"
)

NOISE_STD_BY_LABEL = {
    "clean": 0.0,
    "noise001": 0.01,
    "noise002": 0.02,
    "noise005": 0.05,
    "noise010": 0.10,
    "noise020": 0.20,
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_run_dir(run_output_root: str) -> Path:
    root = Path(run_output_root)
    if not root.is_absolute():
        root = REPO / root
    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories under {root}")
    return candidates[-1]


def load_model_predictions(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in load_rows(spec["registry"]) if r["strategy_id"] == spec["strategy_id"] and r["status"] == "COMPLETED"]
    if not rows:
        raise RuntimeError(f"No completed rows for {spec['label']}")

    preds = []
    targets = []
    for row in rows:
        run_dir = resolve_run_dir(row["run_output_root"])
        arr = np.load(run_dir / "predictions.npz")
        if "predictions" in arr.files and "targets" in arr.files:
            preds.append(arr["predictions"])
            targets.append(arr["targets"])
        elif "test_pred" in arr.files and "test_true" in arr.files:
            preds.append(arr["test_pred"])
            targets.append(arr["test_true"])
        else:
            raise KeyError(f"Unsupported predictions.npz schema for {spec['label']}: keys={arr.files}")
    target0 = targets[0]
    for tgt in targets[1:]:
        if not np.allclose(tgt, target0):
            raise RuntimeError(f"Targets do not align across seeds for {spec['label']}")
    pred_mean = np.mean(np.stack(preds, axis=0), axis=0)
    return target0, pred_mean


def parse_frontier_strategy(strategy_id: str) -> tuple[str, str]:
    if strategy_id.endswith("_clean"):
        return strategy_id[: -len("_clean")], "clean"
    m = re.fullmatch(r"(.+)_(noise\d+)", strategy_id)
    if not m:
        raise ValueError(f"Unrecognized frontier crossover strategy_id={strategy_id}")
    return m.group(1), m.group(2)


def frontier_crossover_degradation_rows(target_names: list[str]) -> list[dict[str, float | str | int]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    rows = [r for r in load_rows(FRONTIER_CROSSOVER_REGISTRY) if r["status"] == "COMPLETED"]
    if not rows:
        raise RuntimeError("No completed frontier crossover rows found")

    for row in rows:
        baseline_name, noise_label = parse_frontier_strategy(row["strategy_id"])
        run_dir = resolve_run_dir(row["run_output_root"])
        arr = np.load(run_dir / "predictions.npz")
        preds = arr["predictions"]
        targets = arr["targets"]
        residuals = preds - targets
        abs_err = np.abs(residuals)

        grouped[(baseline_name, noise_label, "__aggregate__")].append(
            {
                "mse": float(np.mean(residuals**2)),
                "mae": float(np.mean(abs_err)),
                "bias": float(np.mean(residuals)),
                "n_samples": float(targets.shape[0]),
            }
        )
        for idx, name in enumerate(target_names):
            grouped[(baseline_name, noise_label, name)].append(
                {
                    "mse": float(np.mean(residuals[:, idx] ** 2)),
                    "mae": float(np.mean(abs_err[:, idx])),
                    "bias": float(np.mean(residuals[:, idx])),
                    "n_samples": float(targets.shape[0]),
                }
            )

    aggregated: list[dict[str, float | str | int]] = []
    clean_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for key in sorted(grouped):
        baseline_name, noise_label, target_name = key
        vals = grouped[key]
        row = {
            "baseline_name": baseline_name,
            "noise_label": noise_label,
            "noise_std": NOISE_STD_BY_LABEL[noise_label],
            "target_name": target_name,
            "n_runs": len(vals),
            "n_samples_mean": int(round(np.mean([v["n_samples"] for v in vals]))),
            "mse": float(np.mean([v["mse"] for v in vals])),
            "mae": float(np.mean([v["mae"] for v in vals])),
            "bias": float(np.mean([v["bias"] for v in vals])),
        }
        aggregated.append(row)
        if noise_label == "clean":
            clean_lookup[(baseline_name, target_name)] = {"mse": row["mse"], "mae": row["mae"]}

    for row in aggregated:
        clean = clean_lookup[(row["baseline_name"], row["target_name"])]
        row["mse_ratio_vs_clean"] = float(row["mse"] / clean["mse"]) if clean["mse"] > 0 else float("nan")
        row["mae_ratio_vs_clean"] = float(row["mae"] / clean["mae"]) if clean["mae"] > 0 else float("nan")

    return aggregated


def load_reference_features(reference_targets: np.ndarray) -> tuple[np.ndarray, list[str]]:
    params = yaml.safe_load(REFERENCE_PARAMS.read_text())
    target_names = list(params["data"]["target_names"])
    cache_root = SHARED_DATA_DIR / ".split_array_cache"
    if not cache_root.exists():
        raise FileNotFoundError(f"Missing split-array cache root: {cache_root}")

    for cache_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        targets_path = cache_dir / "targets_test.npy"
        features_path = cache_dir / "features_test.npy"
        if not targets_path.exists() or not features_path.exists():
            continue
        targets_test = np.load(targets_path, mmap_mode="r")
        if targets_test.shape != reference_targets.shape:
            continue
        if np.allclose(targets_test, reference_targets):
            return np.load(features_path, mmap_mode="r"), target_names

    raise RuntimeError("No cached Track4 test split matched the checkpoint1 target matrix")


def sample_summaries(features_test: np.ndarray) -> dict[str, np.ndarray]:
    n_samples = int(features_test.shape[0])
    amplitude = np.empty(n_samples, dtype=np.float64)
    trace_std = np.empty(n_samples, dtype=np.float64)
    diff_std = np.empty(n_samples, dtype=np.float64)
    mean_abs = np.empty(n_samples, dtype=np.float64)
    for idx in range(n_samples):
        row = np.asarray(features_test[idx, 0, :], dtype=np.float32)
        amplitude[idx] = float(row.max() - row.min())
        trace_std[idx] = float(row.std())
        diff_std[idx] = float(np.diff(row).std())
        mean_abs[idx] = float(np.abs(row).mean())
    return {
        "amplitude": amplitude,
        "trace_std": trace_std,
        "diff_std": diff_std,
        "mean_abs": mean_abs,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    model_targets = None
    model_preds = {}
    for spec in MODEL_SPECS:
        targets, preds = load_model_predictions(spec)
        if model_targets is None:
            model_targets = targets
        elif not np.allclose(model_targets, targets):
            raise RuntimeError(f"Target mismatch for {spec['label']}")
        model_preds[spec["label"]] = preds

    assert model_targets is not None
    features_test, target_names = load_reference_features(model_targets)
    summaries = sample_summaries(features_test)
    sample_mae_by_model = {}
    parameter_rows = []
    residual_rows = []
    bucket_rows = []
    degradation_rows = frontier_crossover_degradation_rows(target_names)

    for label, preds in model_preds.items():
        residuals = preds - model_targets
        abs_err = np.abs(residuals)
        sample_mae = abs_err.mean(axis=1)
        sample_mae_by_model[label] = sample_mae
        for idx, name in enumerate(target_names):
            r = residuals[:, idx]
            ae = abs_err[:, idx]
            t = model_targets[:, idx]
            parameter_rows.append(
                {
                    "model": label,
                    "target_name": name,
                    "mse": float(np.mean(r**2)),
                    "mae": float(np.mean(ae)),
                    "bias": float(np.mean(r)),
                    "residual_std": float(np.std(r)),
                    "target_std": float(np.std(t)),
                }
            )
            for summary_name, summary_values in summaries.items():
                corr = np.corrcoef(ae, summary_values)[0, 1]
                residual_rows.append(
                    {
                        "model": label,
                        "target_name": name,
                        "summary_name": summary_name,
                        "corr_abs_error_vs_summary": float(corr),
                    }
                )

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for idx, ax in enumerate(axes.ravel()):
            ax.scatter(model_targets[:, idx], preds[:, idx], s=4, alpha=0.25)
            lo = float(min(model_targets[:, idx].min(), preds[:, idx].min()))
            hi = float(max(model_targets[:, idx].max(), preds[:, idx].max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            ax.set_title(target_names[idx])
            ax.set_xlabel("true")
            ax.set_ylabel("pred")
        fig.suptitle(f"{label}: true vs pred")
        fig.savefig(FIGS / f"{label}_scatter_true_vs_pred.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for idx, ax in enumerate(axes.ravel()):
            ax.hist(residuals[:, idx], bins=40, alpha=0.8)
            ax.set_title(target_names[idx])
            ax.set_xlabel("residual")
        fig.suptitle(f"{label}: residual histograms")
        fig.savefig(FIGS / f"{label}_residual_histograms.pdf")
        plt.close(fig)

    difficulty = np.mean(np.stack([sample_mae_by_model[k] for k in model_preds], axis=0), axis=0)
    quantiles = np.quantile(difficulty, [0.25, 0.5, 0.75])
    difficulty_bucket = np.digitize(difficulty, quantiles, right=True) + 1
    for label, sample_mae in sample_mae_by_model.items():
        for bucket in [1, 2, 3, 4]:
            mask = difficulty_bucket == bucket
            bucket_rows.append(
                {
                    "model": label,
                    "bucket": bucket,
                    "n_samples": int(mask.sum()),
                    "mean_sample_mae": float(sample_mae[mask].mean()),
                }
            )

    parameter_csv = TABLES / "hh_parameterwise_metrics_checkpoint1_20260421.csv"
    residual_csv = TABLES / "hh_residual_diagnostics_checkpoint1_20260421.csv"
    bucket_csv = TABLES / "hh_difficulty_bucket_metrics_checkpoint1_20260421.csv"
    degradation_csv = TABLES / "hh_clean_vs_noisy_degradation_checkpoint1_20260421.csv"
    write_csv(parameter_csv, parameter_rows, list(parameter_rows[0].keys()))
    write_csv(residual_csv, residual_rows, list(residual_rows[0].keys()))
    write_csv(bucket_csv, bucket_rows, list(bucket_rows[0].keys()))
    write_csv(degradation_csv, degradation_rows, list(degradation_rows[0].keys()))

    degradation_targets = [name for name in target_names]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for idx, ax in enumerate(axes.ravel()):
        target_name = degradation_targets[idx]
        target_rows = [
            r for r in degradation_rows if r["target_name"] == target_name and r["noise_label"] != "clean"
        ]
        for baseline_name in sorted({r["baseline_name"] for r in target_rows}):
            rows = [r for r in target_rows if r["baseline_name"] == baseline_name]
            rows.sort(key=lambda r: r["noise_std"])
            ax.plot(
                [r["noise_std"] for r in rows],
                [r["mae_ratio_vs_clean"] for r in rows],
                marker="o",
                label=baseline_name,
            )
        ax.set_title(target_name)
        ax.set_xlabel("noise std")
        ax.set_ylabel("MAE / clean MAE")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Checkpoint 1 clean-vs-noisy per-target degradation")
    fig.savefig(FIGS / "hh_clean_vs_noisy_degradation_checkpoint1_20260421.pdf")
    plt.close(fig)

    note_lines = [
        "# HH Track4 Checkpoint 1 Analysis (2026-04-21)",
        "",
        "## Models compared",
    ]
    note_lines.extend([f"- `{spec['label']}`" for spec in MODEL_SPECS])
    note_lines.extend(
        [
            "",
            "## Outputs",
            f"- parameter-wise metrics: `{parameter_csv}`",
            f"- residual diagnostics: `{residual_csv}`",
            f"- difficulty-bucket metrics: `{bucket_csv}`",
            f"- clean-vs-noisy degradation: `{degradation_csv}`",
            f"- figures dir: `{FIGS}`",
            "",
            "## Difficulty definition",
            "- per-sample difficulty is the mean sample MAE averaged across the compared frontier models, then bucketed into quartiles",
            "",
            "## Protocol-bucket note",
            "- the current checkpoint1 HH task contract is single-protocol, so no protocol-bucket table was generated here",
        ]
    )
    (NOTES / "hh_track4_checkpoint1_analysis_notes_20260421.md").write_text("\n".join(note_lines) + "\n")

    print(f"Wrote {parameter_csv}")
    print(f"Wrote {residual_csv}")
    print(f"Wrote {bucket_csv}")
    print(f"Wrote {degradation_csv}")


if __name__ == "__main__":
    main()
