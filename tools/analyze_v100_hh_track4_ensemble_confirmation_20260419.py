#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

DATE_TAG = "20260419_v100_hh_track4_ensemble_confirmation"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_ensemble_confirmation_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_track4_ensemble_confirmation_registry_{DATE_TAG}.csv"
SUMMARY_MD = NOTES / f"track4_ensemble_confirmation_interpretation_{DATE_TAG}.md"


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / ss_tot


def metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_predictions(run_output_root: Path) -> tuple[np.ndarray, np.ndarray]:
    pred_path = next(run_output_root.glob("*/predictions.npz"))
    arr = np.load(pred_path)
    return arr["test_true"], arr["test_pred"]


def main() -> None:
    matrix_rows = {row["row_id"]: row for row in load_rows(MATRIX_CSV)}
    if REGISTRY_CSV.exists():
        registry_rows = [row for row in load_rows(REGISTRY_CSV) if row.get("status") == "COMPLETED"]
    else:
        registry_rows = []
        for row_id, row in matrix_rows.items():
            run_root = OUT_ROOT / "runs" / row["phase"] / row["launch_mode"] / row_id
            if list(run_root.glob("*/metrics_summary.json")):
                registry_rows.append(
                    {
                        "row_id": row_id,
                        "run_output_root": str(run_root),
                        "status": "COMPLETED",
                    }
                )

    per_strategy: dict[str, dict[str, list[float]]] = {}
    per_seed: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for row in registry_rows:
        run_output_root = Path(row["run_output_root"])
        y_true, y_pred = load_predictions(run_output_root)
        strategy = matrix_rows[row["row_id"]]["strategy_id"]
        seed = matrix_rows[row["row_id"]]["seed"]

        stats = per_strategy.setdefault(strategy, {"r2": [], "mse": [], "mae": []})
        stats["r2"].append(metric_r2(y_true, y_pred))
        stats["mse"].append(metric_mse(y_true, y_pred))
        stats["mae"].append(metric_mae(y_true, y_pred))

        per_seed.setdefault(seed, {})[strategy] = (y_true, y_pred)

    blends = {
        "blend_eff55_conv45": 0.55,
        "blend_eff50_conv50": 0.50,
        "blend_eff70_conv30": 0.70,
    }
    blend_stats: dict[str, dict[str, list[float]]] = {
        name: {"r2": [], "mse": [], "mae": []} for name in blends
    }
    for seed, seed_rows in sorted(per_seed.items()):
        if "avgpool_baseline" not in seed_rows or "effnet_beta005_invstd" not in seed_rows:
            continue
        y_true, conv_pred = seed_rows["avgpool_baseline"]
        _, eff_pred = seed_rows["effnet_beta005_invstd"]
        for name, eff_weight in blends.items():
            pred = eff_weight * eff_pred + (1.0 - eff_weight) * conv_pred
            blend_stats[name]["r2"].append(metric_r2(y_true, pred))
            blend_stats[name]["mse"].append(metric_mse(y_true, pred))
            blend_stats[name]["mae"].append(metric_mae(y_true, pred))

    lines = [
        f"# Track4 Ensemble Confirmation Interpretation ({DATE_TAG})",
        "",
        "## Single Models",
        "",
        "| variant | mean test_r2 | mean test_mse | mean test_mae |",
        "| --- | ---: | ---: | ---: |",
    ]
    for strategy, stats in sorted(per_strategy.items()):
        lines.append(
            f"| `{strategy}` | `{mean(stats['r2']):.6f}` | `{mean(stats['mse']):,.3f}` | `{mean(stats['mae']):.3f}` |"
        )

    if any(v["r2"] for v in blend_stats.values()):
        lines.extend(
            [
                "",
                "## Fixed Blends",
                "",
                "| variant | mean test_r2 | mean test_mse | mean test_mae |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for strategy, stats in sorted(blend_stats.items()):
            if not stats["r2"]:
                continue
            lines.append(
                f"| `{strategy}` | `{mean(stats['r2']):.6f}` | `{mean(stats['mse']):,.3f}` | `{mean(stats['mae']):.3f}` |"
            )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
