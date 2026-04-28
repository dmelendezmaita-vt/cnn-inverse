#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / "optimization_track_20260422_hh_track4_checkpoint5_ambiguity_map"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
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

CP4_OBS_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260422_hh_track4_checkpoint4_observation_richness"
    / "tables"
    / "hh_track4_checkpoint4_observation_richness_registry_20260422_hh_track4_checkpoint4_observation_richness.csv"
)
CP4_PROM_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260422_hh_track4_checkpoint4_representation_promotion"
    / "tables"
    / "hh_track4_checkpoint4_representation_promotion_registry_20260422_hh_track4_checkpoint4_representation_promotion.csv"
)
CP4_NOISE_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260422_hh_track4_checkpoint4_representation_noise_followup"
    / "tables"
    / "hh_track4_checkpoint4_representation_noise_followup_registry_20260422_hh_track4_checkpoint4_representation_noise_followup.csv"
)
SBI_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260422_v100_hh_track4_checkpoint2_snpe_seed_stability"
    / "tables"
    / "v100_hh_track4_checkpoint2_snpe_seed_stability_registry_20260422_v100_hh_track4_checkpoint2_snpe_seed_stability.csv"
)
SBI_PREFIX = "snpe_maf_h192_t8_n8192_"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_run_dir(run_output_root: str) -> Path:
    root = Path(run_output_root)
    if not root.is_absolute():
        root = REPO / root
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories under {root}")
    return candidates[-1]


def completed_row(registry: Path, strategy_id: str) -> dict[str, str]:
    rows = [r for r in load_rows(registry) if r["strategy_id"] == strategy_id and r["status"] == "COMPLETED"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one completed row for {strategy_id} in {registry}, found {len(rows)}")
    return rows[0]


def load_predictions_npz(run_dir: Path, npz_name: str = "predictions.npz") -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(run_dir / npz_name)
    if "predictions" in arr.files and "targets" in arr.files:
        return arr["targets"], arr["predictions"]
    if "test_true" in arr.files and "test_pred" in arr.files:
        return arr["test_true"], arr["test_pred"]
    raise KeyError(f"Unsupported prediction schema at {run_dir / npz_name}: {arr.files}")


def per_target_metrics(targets: np.ndarray, preds: np.ndarray, target_names: list[str]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    residuals = preds - targets
    for idx, name in enumerate(target_names):
        target = targets[:, idx]
        residual = residuals[:, idx]
        mse = float(np.mean(residual**2))
        mae = float(np.mean(np.abs(residual)))
        target_std = float(np.std(target))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        r2 = float(1.0 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else float("nan")
        rows.append(
            {
                "target_index": idx,
                "target_name": name,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": mae,
                "bias": float(np.mean(residual)),
                "target_std": target_std,
                "r2": r2,
            }
        )
    return rows


def load_reference_features(reference_targets: np.ndarray) -> tuple[np.ndarray, list[str]]:
    params = yaml.safe_load(REFERENCE_PARAMS.read_text())
    target_names = list(params["data"]["target_names"])
    cache_root = SHARED_DATA_DIR / ".split_array_cache"
    for cache_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        targets_path = cache_dir / "targets_test.npy"
        features_path = cache_dir / "features_test.npy"
        if not targets_path.exists() or not features_path.exists():
            continue
        cached_targets = np.load(targets_path, mmap_mode="r")
        if cached_targets.shape == reference_targets.shape and np.allclose(cached_targets, reference_targets):
            return np.load(features_path, mmap_mode="r"), target_names
    raise RuntimeError("No split-array cache matched the checkpoint5 target matrix")


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


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def aggregate_sbi_calibration(target_names: list[str]) -> list[dict[str, float | str]]:
    rows = [r for r in load_rows(SBI_REGISTRY) if r["status"] == "COMPLETED" and r["strategy_id"].startswith(SBI_PREFIX)]
    if not rows:
        raise RuntimeError("No completed stable SNPE rows found for checkpoint5 calibration")

    grouped: dict[str, list[dict[str, float]]] = {name: [] for name in target_names}
    for row in rows:
        run_dir = resolve_run_dir(row["run_output_root"])
        arr = np.load(run_dir / "predictions_test.npz")
        targets = arr["targets"]
        mean = arr["posterior_mean"]
        std = arr["posterior_std"]
        samples = arr["posterior_samples"]
        for idx, name in enumerate(target_names):
            target = targets[:, idx]
            pred = mean[:, idx]
            residual = pred - target
            q25, q75 = np.quantile(samples[:, :, idx], [0.25, 0.75], axis=1)
            q10, q90 = np.quantile(samples[:, :, idx], [0.10, 0.90], axis=1)
            q05, q95 = np.quantile(samples[:, :, idx], [0.05, 0.95], axis=1)
            grouped[name].append(
                {
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mean_posterior_std": float(np.mean(std[:, idx])),
                    "width50": float(np.mean(q75 - q25)),
                    "width80": float(np.mean(q90 - q10)),
                    "width90": float(np.mean(q95 - q05)),
                    "coverage50": float(np.mean((target >= q25) & (target <= q75))),
                    "coverage80": float(np.mean((target >= q10) & (target <= q90))),
                    "coverage90": float(np.mean((target >= q05) & (target <= q95))),
                }
            )

    out_rows = []
    for name in target_names:
        vals = grouped[name]
        out_rows.append(
            {
                "target_name": name,
                "n_seeds": len(vals),
                "mae": float(np.mean([v["mae"] for v in vals])),
                "rmse": float(np.mean([v["rmse"] for v in vals])),
                "mean_posterior_std": float(np.mean([v["mean_posterior_std"] for v in vals])),
                "width50": float(np.mean([v["width50"] for v in vals])),
                "width80": float(np.mean([v["width80"] for v in vals])),
                "width90": float(np.mean([v["width90"] for v in vals])),
                "coverage50": float(np.mean([v["coverage50"] for v in vals])),
                "coverage80": float(np.mean([v["coverage80"] for v in vals])),
                "coverage90": float(np.mean([v["coverage90"] for v in vals])),
            }
        )
    return out_rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    raw_targets, raw_preds = load_predictions_npz(resolve_run_dir(completed_row(CP4_OBS_REGISTRY, "extra_trees_500_leaf3__raw")["run_output_root"]))
    rich_targets, rich_preds = load_predictions_npz(
        resolve_run_dir(completed_row(CP4_OBS_REGISTRY, "extra_trees_500_leaf3__raw_plus_fft256_summary12")["run_output_root"])
    )
    rf_targets, rf_preds = load_predictions_npz(
        resolve_run_dir(completed_row(CP4_PROM_REGISTRY, "random_forest_500__raw_plus_fft256_summary12")["run_output_root"])
    )
    knn_clean_targets, knn_clean_preds = load_predictions_npz(
        resolve_run_dir(completed_row(CP4_NOISE_REGISTRY, "knn_k11__clean")["run_output_root"])
    )
    knn_noise_targets, knn_noise_preds = load_predictions_npz(
        resolve_run_dir(completed_row(CP4_NOISE_REGISTRY, "knn_k11__noise020")["run_output_root"])
    )

    if not (np.allclose(raw_targets, rich_targets) and np.allclose(raw_targets, rf_targets)):
        raise RuntimeError("Checkpoint5 classical targets are not aligned")

    features_test, target_names = load_reference_features(raw_targets)
    summaries = sample_summaries(features_test)

    raw_metrics = {r["target_name"]: r for r in per_target_metrics(raw_targets, raw_preds, target_names)}
    rich_metrics = {r["target_name"]: r for r in per_target_metrics(rich_targets, rich_preds, target_names)}
    rf_metrics = {r["target_name"]: r for r in per_target_metrics(rf_targets, rf_preds, target_names)}
    knn_clean_metrics = {r["target_name"]: r for r in per_target_metrics(knn_clean_targets, knn_clean_preds, target_names)}
    knn_noise_metrics = {r["target_name"]: r for r in per_target_metrics(knn_noise_targets, knn_noise_preds, target_names)}
    sbi_rows = aggregate_sbi_calibration(target_names)
    sbi_by_target = {r["target_name"]: r for r in sbi_rows}

    raw_abs = np.abs(raw_preds - raw_targets)
    rich_abs = np.abs(rich_preds - rich_targets)
    rf_abs = np.abs(rf_preds - rf_targets)

    local_ambiguity_rows = []
    parameter_rows = []
    for idx, name in enumerate(target_names):
        raw_row = raw_metrics[name]
        rich_row = rich_metrics[name]
        rf_row = rf_metrics[name]
        knn_clean_row = knn_clean_metrics[name]
        knn_noise_row = knn_noise_metrics[name]
        sbi_row = sbi_by_target[name]

        rich_gain_mae_pct = float((raw_row["mae"] - rich_row["mae"]) / raw_row["mae"]) if raw_row["mae"] else float("nan")
        rich_gain_mse_pct = float((raw_row["mse"] - rich_row["mse"]) / raw_row["mse"]) if raw_row["mse"] else float("nan")
        rich_gain_r2 = float(rich_row["r2"] - raw_row["r2"])
        noise_ratio = float(knn_noise_row["mae"] / knn_clean_row["mae"]) if knn_clean_row["mae"] else float("nan")
        rf_nrmse = float(rf_row["rmse"] / rf_row["target_std"]) if rf_row["target_std"] else float("nan")
        sbi_std_ratio = float(sbi_row["mean_posterior_std"] / rf_row["target_std"]) if rf_row["target_std"] else float("nan")

        easy_flag = rf_nrmse < 0.50 and rf_row["r2"] > 0.50
        obs_help_flag = rich_gain_mae_pct > 0.10 or rich_gain_r2 > 0.05
        ambiguity_flag = rf_row["r2"] < 0.15 and sbi_row["coverage90"] < 0.65 and sbi_std_ratio > 0.50
        if easy_flag:
            status = "easy"
        elif ambiguity_flag:
            status = "ambiguity_dominated"
        elif obs_help_flag:
            status = "observation_helped"
        else:
            status = "hard_residual"

        parameter_rows.append(
            {
                "target_name": name,
                "raw_et_leaf3_mae": raw_row["mae"],
                "raw_et_leaf3_r2": raw_row["r2"],
                "rich_et_leaf3_mae": rich_row["mae"],
                "rich_et_leaf3_r2": rich_row["r2"],
                "rich_gain_mae_pct": rich_gain_mae_pct,
                "rich_gain_mse_pct": rich_gain_mse_pct,
                "rich_gain_r2_delta": rich_gain_r2,
                "promoted_rf_mae": rf_row["mae"],
                "promoted_rf_r2": rf_row["r2"],
                "promoted_rf_nrmse": rf_nrmse,
                "knn_noise020_mae_ratio_vs_clean": noise_ratio,
                "snpe_maf_mae": sbi_row["mae"],
                "snpe_maf_coverage90": sbi_row["coverage90"],
                "snpe_maf_std_ratio": sbi_std_ratio,
                "easy_flag": easy_flag,
                "observation_help_flag": obs_help_flag,
                "ambiguity_flag": ambiguity_flag,
                "heuristic_status": status,
            }
        )

        for summary_name, summary_values in summaries.items():
            local_ambiguity_rows.append(
                {
                    "target_name": name,
                    "summary_name": summary_name,
                    "corr_raw_abs_error": corr(raw_abs[:, idx], summary_values),
                    "corr_rich_abs_error": corr(rich_abs[:, idx], summary_values),
                    "corr_promoted_rf_abs_error": corr(rf_abs[:, idx], summary_values),
                }
            )

    parameter_csv = TABLES / "parameter_status.csv"
    ambiguity_csv = TABLES / "local_ambiguity.csv"
    sbi_csv = TABLES / "sbi_calibration.csv"
    write_csv(parameter_csv, parameter_rows, list(parameter_rows[0].keys()))
    write_csv(ambiguity_csv, local_ambiguity_rows, list(local_ambiguity_rows[0].keys()))
    write_csv(sbi_csv, sbi_rows, list(sbi_rows[0].keys()))

    easy = [r["target_name"] for r in parameter_rows if r["easy_flag"]]
    helped = sorted(parameter_rows, key=lambda r: r["rich_gain_mae_pct"], reverse=True)
    ambiguity = [r["target_name"] for r in parameter_rows if r["ambiguity_flag"]]

    note_lines = [
        "# HH Track4 Checkpoint 5 Ambiguity Map (2026-04-22)",
        "",
        "## Purpose",
        "- answer the checkpoint5 gate using completed checkpoint4 classical representation packages plus the stable SNPE seed-stability artifacts",
        "- distinguish easy parameters, parameters helped by richer observations, and parameters that still look ambiguity-dominated",
        "",
        "## Inputs",
        f"- raw observation anchor: `{CP4_OBS_REGISTRY.name}` strategy `extra_trees_500_leaf3__raw`",
        f"- richer observation anchor: `{CP4_OBS_REGISTRY.name}` strategy `extra_trees_500_leaf3__raw_plus_fft256_summary12`",
        f"- promoted clean classical winner: `{CP4_PROM_REGISTRY.name}` strategy `random_forest_500__raw_plus_fft256_summary12`",
        f"- robustness reference: `{CP4_NOISE_REGISTRY.name}` strategies `knn_k11__clean` and `knn_k11__noise020`",
        f"- probabilistic reference: `{SBI_REGISTRY.name}` prefix `{SBI_PREFIX}`",
        "",
        "## Outputs",
        f"- parameter status: `{parameter_csv}`",
        f"- local ambiguity correlations: `{ambiguity_csv}`",
        f"- SBI calibration: `{sbi_csv}`",
        "",
        "## Heuristic Read",
        f"- easy parameters: `{easy if easy else 'none'}`",
        f"- strongest richer-observation gains: `{[(r['target_name'], round(float(r['rich_gain_mae_pct']), 3)) for r in helped[:3]]}`",
        f"- ambiguity-dominated candidates: `{ambiguity if ambiguity else 'none'}`",
        "",
        "## Caveat",
        "- this is a practical identifiability / calibration package, not a structural profile-likelihood analysis; direct simulator likelihood work is still blocked by missing HH simulator metadata",
    ]
    (NOTES / "hh_track4_checkpoint5_ambiguity_map_notes_20260422.md").write_text("\n".join(note_lines) + "\n")

    print(f"Wrote {parameter_csv}")
    print(f"Wrote {ambiguity_csv}")
    print(f"Wrote {sbi_csv}")


if __name__ == "__main__":
    main()
