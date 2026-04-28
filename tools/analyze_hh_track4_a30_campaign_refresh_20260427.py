#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


COMPARABILITY_CLASS = "direct_track4_a30_campaign"
OUR_OUTPUT_PREFIX = "hh_track4_a30_"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Regenerate A30-only HH Track4 comparison tables, diagnostics, branch registry, "
            "and reporting synthesis from a single campaign root."
        )
    )
    ap.add_argument("--campaign-root", required=True, help="Campaign root containing tables/, notes/, and runs/.")
    return ap.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def campaign_tag(campaign_root: Path) -> str:
    name = campaign_root.name
    if name.startswith("optimization_track_"):
        return name[len("optimization_track_") :]
    return name


def relpath(path: Path | None, campaign_root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(campaign_root))
    except ValueError:
        return str(path)


def is_within_root(path: Path, campaign_root: Path) -> bool:
    try:
        path.resolve().relative_to(campaign_root.resolve())
        return True
    except ValueError:
        return False


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def metric_value(obj: dict, key: str, split: str = "test") -> float | None:
    value = obj.get(key)
    if isinstance(value, dict):
        for candidate in (split, "test", "validate", "train", "value"):
            if candidate in value:
                parsed = safe_float(value[candidate])
                if parsed is not None:
                    return parsed
        return None
    return safe_float(value)


def discover_input_tables(campaign_root: Path) -> tuple[list[Path], list[Path]]:
    tables_dir = campaign_root / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"Missing tables directory under {campaign_root}")
    matrix_paths = sorted(p for p in tables_dir.glob("*.csv") if "_matrix_" in p.name and not p.name.startswith(OUR_OUTPUT_PREFIX))
    registry_paths = sorted(
        p for p in tables_dir.glob("*.csv") if "_registry_" in p.name and not p.name.startswith(OUR_OUTPUT_PREFIX)
    )
    return matrix_paths, registry_paths


def merge_input_rows(matrix_paths: list[Path], registry_paths: list[Path]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for path in matrix_paths:
        for row in read_csv_rows(path):
            row_id = row.get("row_id", "").strip()
            if not row_id:
                continue
            target = merged.setdefault(row_id, {"row_id": row_id})
            target.update(row)
            target["matrix_source"] = str(path)
    for path in registry_paths:
        for row in read_csv_rows(path):
            row_id = row.get("row_id", "").strip()
            if not row_id:
                continue
            target = merged.setdefault(row_id, {"row_id": row_id})
            target.update(row)
            target["registry_source"] = str(path)
    return [merged[row_id] for row_id in sorted(merged)]


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def locate_run_root(campaign_root: Path, row: dict[str, str]) -> Path | None:
    candidates: list[Path] = []
    run_output_root = row.get("run_output_root", "").strip()
    if run_output_root:
        candidate = Path(run_output_root)
        if is_within_root(candidate, campaign_root):
            candidates.append(candidate)
    row_id = row.get("row_id", "").strip()
    if row_id:
        candidates.extend(sorted((campaign_root / "runs").glob(f"*/*/{row_id}")))
    candidates = [path for path in dedupe_paths(candidates) if path.exists()]
    if not candidates:
        return None
    for candidate in candidates:
        if (candidate / "metrics_summary.json").exists() or (candidate / "predictions.npz").exists():
            return candidate
        metric_paths = sorted(candidate.rglob("metrics_summary.json"), key=lambda p: (len(p.parts), str(p)))
        if metric_paths:
            return metric_paths[0].parent
        pred_paths = sorted(candidate.rglob("predictions.npz"), key=lambda p: (len(p.parts), str(p)))
        if pred_paths:
            return pred_paths[0].parent
    return candidates[0]


def infer_noise_from_strategy(strategy_id: str) -> float | None:
    if not strategy_id:
        return None
    match = re.search(r"(?:^|_)(clean|noise(\d+))(?:_|$)", strategy_id)
    if not match:
        return None
    if match.group(1) == "clean":
        return 0.0
    digits = match.group(2)
    if not digits:
        return None
    return float(int(digits)) / 100.0


def infer_noise_label(noise_std: float | None) -> str:
    if noise_std is None:
        return "unspecified"
    if abs(noise_std) < 1.0e-12:
        return "clean"
    return f"noise{int(round(noise_std * 100.0)):03d}"


def strategy_base(strategy_id: str) -> str:
    if not strategy_id:
        return ""
    return re.sub(r"_(?:clean|noise\d+)$", "", strategy_id)


def load_metrics_per_target_rows(path: Path) -> list[dict[str, object]]:
    obj = load_json(path)
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        test_rows = obj.get("per_target", {}).get("test", [])
        if isinstance(test_rows, list):
            return [row for row in test_rows if isinstance(row, dict)]
    return []


def load_target_names(artifact_root: Path, per_target_rows: list[dict[str, object]]) -> list[str]:
    if per_target_rows:
        names = [str(row.get("target_name", f"target_{idx}")) for idx, row in enumerate(per_target_rows)]
        if names:
            return names
    target_names_path = artifact_root / "target_names.json"
    if target_names_path.exists():
        obj = load_json(target_names_path)
        names = obj.get("target_names", [])
        if isinstance(names, list):
            return [str(name) for name in names]
    return []


def load_prediction_pair(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.load(path)
    known_pairs = [
        ("test_pred", "test_true"),
        ("test_prediction", "test_true"),
        ("predictions", "targets"),
        ("test_pred", "targets"),
    ]
    for pred_key, true_key in known_pairs:
        if pred_key in arr.files and true_key in arr.files:
            pred = np.asarray(arr[pred_key], dtype=np.float32)
            true = np.asarray(arr[true_key], dtype=np.float32)
            if pred.shape == true.shape and pred.ndim == 2:
                return pred, true
    return None


def compute_per_target_rows(pred: np.ndarray, true: np.ndarray, target_names: list[str]) -> list[dict[str, object]]:
    diff = pred - true
    abs_err = np.abs(diff)
    out: list[dict[str, object]] = []
    for idx in range(true.shape[1]):
        y_true = true[:, idx]
        y_pred = pred[:, idx]
        denom = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = 1.0 - float(np.sum(np.square(y_true - y_pred))) / denom if denom > 0.0 else 0.0
        range_true = float(np.max(y_true) - np.min(y_true))
        mae = float(np.mean(abs_err[:, idx]))
        out.append(
            {
                "target_index": idx,
                "target_name": target_names[idx] if idx < len(target_names) else f"target_{idx}",
                "mae": mae,
                "rmse": float(np.sqrt(np.mean(np.square(diff[:, idx])))),
                "r2": r2,
                "nmae_range": mae / range_true if range_true > 0.0 else math.nan,
                "bias": float(np.mean(diff[:, idx])),
            }
        )
    return out


def merge_per_target_rows(
    existing_rows: list[dict[str, object]],
    derived_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[int, dict[str, object]] = {}
    for idx, row in enumerate(derived_rows):
        target_index = int(safe_float(row.get("target_index")) or idx)
        merged[target_index] = dict(row)
    for idx, row in enumerate(existing_rows):
        target_index = int(safe_float(row.get("target_index")) or idx)
        base = merged.setdefault(target_index, {"target_index": target_index})
        for key, value in row.items():
            if value not in (None, ""):
                base[key] = value
    return [merged[idx] for idx in sorted(merged)]


def compute_diagnostic_row(
    row: dict[str, object],
    pred: np.ndarray,
    true: np.ndarray,
    per_target_rows: list[dict[str, object]],
    campaign_root: Path,
) -> dict[str, object]:
    diff = pred - true
    abs_err = np.abs(diff)
    l2_err = np.linalg.norm(diff, axis=1)
    std_true = np.std(true, axis=0)
    std_pred = np.std(pred, axis=0)
    spread_ratio = std_pred / np.clip(std_true, 1.0e-12, None)
    target_mae = np.array([safe_float(target.get("mae")) or math.nan for target in per_target_rows], dtype=np.float64)
    target_r2 = np.array([safe_float(target.get("r2")) or math.nan for target in per_target_rows], dtype=np.float64)
    target_nmae = np.array(
        [safe_float(target.get("nmae_range")) or math.nan for target in per_target_rows],
        dtype=np.float64,
    )
    return {
        "row_id": row["row_id"],
        "phase": row["phase"],
        "phase_label": row["phase_label"],
        "family": row["family"],
        "strategy_id": row["strategy_id"],
        "strategy_base": row["strategy_base"],
        "baseline_name": row["baseline_name"],
        "seed": row["seed"],
        "noise_std": row["noise_std"],
        "noise_label": row["noise_label"],
        "mae": row["mae"],
        "r2": row["r2"],
        "mean_abs_error": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
        "max_abs_error": float(np.max(abs_err)),
        "median_l2_error": float(np.median(l2_err)),
        "p95_l2_error": float(np.quantile(l2_err, 0.95)),
        "bias_abs_mean": float(np.mean(np.abs(np.mean(diff, axis=0)))),
        "spread_ratio_mean": float(np.mean(spread_ratio)),
        "worst_target_mae": float(np.nanmax(target_mae)),
        "worst_target_r2": float(np.nanmin(target_r2)),
        "worst_target_nmae_range": float(np.nanmax(target_nmae)),
        "artifact_dir": row["artifact_dir"],
        "predictions_path": row["predictions_path"],
    }


def build_row_records(campaign_root: Path, merged_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    row_records: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    per_target_rows_all: list[dict[str, object]] = []

    for merged in merged_rows:
        record: dict[str, object] = {
            "row_id": merged.get("row_id", ""),
            "phase": merged.get("phase", merged.get("family", "")),
            "phase_label": merged.get("phase_label", merged.get("family", "")),
            "family": merged.get("family", merged.get("phase", "")),
            "strategy_id": merged.get("strategy_id", merged.get("baseline_name", "")),
            "baseline_name": merged.get("baseline_name", ""),
            "seed": safe_float(merged.get("seed")),
            "status": merged.get("status", "unknown"),
            "return_code": safe_float(merged.get("return_code")),
            "launch_mode": merged.get("launch_mode", ""),
            "launch_group": merged.get("launch_group", ""),
            "policy": merged.get("policy", ""),
            "notes": merged.get("notes", ""),
            "matrix_source": relpath(Path(merged["matrix_source"]), campaign_root) if merged.get("matrix_source") else "",
            "registry_source": relpath(Path(merged["registry_source"]), campaign_root) if merged.get("registry_source") else "",
            "artifact_dir": "",
            "metrics_summary_path": "",
            "predictions_path": "",
            "mae": None,
            "mse": None,
            "r2": None,
            "runtime_sec": safe_float(merged.get("elapsed_sec")),
            "n_test": None,
        }
        record["strategy_base"] = strategy_base(str(record["strategy_id"]))

        artifact_root = locate_run_root(campaign_root, merged)
        if artifact_root is not None:
            record["artifact_dir"] = relpath(artifact_root, campaign_root)

        metrics_summary_path: Path | None = None
        metrics_per_target_path: Path | None = None
        predictions_path: Path | None = None
        metrics_summary_obj: dict = {}

        if artifact_root is not None:
            candidate = artifact_root / "metrics_summary.json"
            if candidate.exists():
                metrics_summary_path = candidate
            candidate = artifact_root / "metrics_per_target.json"
            if candidate.exists():
                metrics_per_target_path = candidate
            candidate = artifact_root / "predictions.npz"
            if candidate.exists():
                predictions_path = candidate

        if metrics_summary_path is not None:
            record["metrics_summary_path"] = relpath(metrics_summary_path, campaign_root)
            metrics_summary_obj = load_json(metrics_summary_path)
            record["mae"] = metric_value(metrics_summary_obj, "mae")
            record["mse"] = metric_value(metrics_summary_obj, "mse")
            record["r2"] = metric_value(metrics_summary_obj, "r2")
            record["runtime_sec"] = (
                safe_float(metrics_summary_obj.get("metadata", {}).get("runtime_sec")) or record["runtime_sec"]
            )
            record["n_test"] = safe_float(metrics_summary_obj.get("metadata", {}).get("n_test"))

        noise_std = safe_float(metrics_summary_obj.get("metadata", {}).get("features_additive_noise_std"))
        if noise_std is None:
            noise_std = infer_noise_from_strategy(str(record["strategy_id"]))
        record["noise_std"] = noise_std
        record["noise_label"] = infer_noise_label(noise_std)

        if predictions_path is not None:
            record["predictions_path"] = relpath(predictions_path, campaign_root)

        per_target_rows = load_metrics_per_target_rows(metrics_per_target_path) if metrics_per_target_path else []
        if predictions_path is not None:
            prediction_pair = load_prediction_pair(predictions_path)
            if prediction_pair is not None:
                pred, true = prediction_pair
                target_names = load_target_names(artifact_root or predictions_path.parent, per_target_rows)
                derived_per_target_rows = compute_per_target_rows(pred, true, target_names)
                per_target_rows = merge_per_target_rows(per_target_rows, derived_per_target_rows) if per_target_rows else derived_per_target_rows
                diagnostics_rows.append(compute_diagnostic_row(record, pred, true, per_target_rows, campaign_root))

        for target in per_target_rows:
            per_target_rows_all.append(
                {
                    "row_id": record["row_id"],
                    "phase": record["phase"],
                    "phase_label": record["phase_label"],
                    "family": record["family"],
                    "strategy_id": record["strategy_id"],
                    "strategy_base": record["strategy_base"],
                    "baseline_name": record["baseline_name"],
                    "seed": record["seed"],
                    "noise_std": record["noise_std"],
                    "noise_label": record["noise_label"],
                    "target_index": safe_float(target.get("target_index")),
                    "target_name": target.get("target_name", ""),
                    "mae": safe_float(target.get("mae")),
                    "rmse": safe_float(target.get("rmse")),
                    "r2": safe_float(target.get("r2")),
                    "nmae_range": safe_float(target.get("nmae_range")),
                    "bias": safe_float(target.get("bias")),
                    "mape_percent": safe_float(target.get("mape_percent")),
                    "source": relpath(metrics_per_target_path, campaign_root) if metrics_per_target_path else record["predictions_path"],
                }
            )

        row_records.append(record)

    return row_records, diagnostics_rows, per_target_rows_all


def numeric_sort_key(value: object, default: float = float("inf")) -> float:
    parsed = safe_float(value)
    return parsed if parsed is not None else default


def build_run_comparison_rows(row_records: list[dict[str, object]]) -> list[dict[str, object]]:
    completed = [row for row in row_records if safe_float(row.get("mae")) is not None]
    completed.sort(
        key=lambda row: (
            numeric_sort_key(row.get("noise_std"), default=-1.0),
            numeric_sort_key(row.get("mae")),
            -numeric_sort_key(row.get("r2"), default=-float("inf")),
            numeric_sort_key(row.get("runtime_sec")),
            str(row.get("row_id", "")),
        )
    )
    return completed


def build_strategy_comparison_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        key = (
            str(row["phase"]),
            str(row["phase_label"]),
            str(row["family"]),
            str(row["strategy_base"]),
            str(row["baseline_name"]),
            str(row["noise_label"]),
        )
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    for key, rows in grouped.items():
        maes = np.array([float(row["mae"]) for row in rows], dtype=np.float64)
        r2s = np.array([float(row["r2"]) for row in rows], dtype=np.float64)
        runtimes = np.array(
            [float(value) for value in (safe_float(row.get("runtime_sec")) for row in rows) if value is not None],
            dtype=np.float64,
        )
        best_row = min(rows, key=lambda row: (float(row["mae"]), -float(row["r2"]), numeric_sort_key(row.get("runtime_sec"))))
        out.append(
            {
                "phase": key[0],
                "phase_label": key[1],
                "family": key[2],
                "strategy_base": key[3],
                "baseline_name": key[4],
                "noise_label": key[5],
                "noise_std": rows[0]["noise_std"],
                "completed_runs": len(rows),
                "seed_count": len({row["seed"] for row in rows}),
                "mean_mae": float(np.mean(maes)),
                "std_mae": float(np.std(maes)),
                "mean_r2": float(np.mean(r2s)),
                "std_r2": float(np.std(r2s)),
                "mean_runtime_sec": float(np.mean(runtimes)) if runtimes.size else math.nan,
                "min_mae": float(np.min(maes)),
                "max_mae": float(np.max(maes)),
                "best_row_id": best_row["row_id"],
                "best_strategy_id": best_row["strategy_id"],
                "source_row": best_row["artifact_dir"],
            }
        )
    out.sort(key=lambda row: (numeric_sort_key(row["noise_std"], default=-1.0), float(row["mean_mae"]), -float(row["mean_r2"])))
    return out


def build_noise_robustness_rows(strategy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in strategy_rows:
        key = (str(row["phase"]), str(row["phase_label"]), str(row["family"]), str(row["strategy_base"]))
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    for key, rows in grouped.items():
        clean_rows = [row for row in rows if safe_float(row["noise_std"]) == 0.0]
        noisy_rows = [row for row in rows if safe_float(row["noise_std"]) not in (None, 0.0)]
        clean_best = min(clean_rows, key=lambda row: float(row["mean_mae"])) if clean_rows else None
        worst_noisy = max(noisy_rows, key=lambda row: float(row["mean_mae"])) if noisy_rows else None
        best_noisy = min(noisy_rows, key=lambda row: float(row["mean_mae"])) if noisy_rows else None
        out.append(
            {
                "phase": key[0],
                "phase_label": key[1],
                "family": key[2],
                "strategy_base": key[3],
                "baseline_name": clean_best["baseline_name"] if clean_best else (best_noisy["baseline_name"] if best_noisy else ""),
                "observed_noise_levels": ",".join(sorted({str(row["noise_label"]) for row in rows})),
                "clean_mean_mae": clean_best["mean_mae"] if clean_best else math.nan,
                "clean_mean_r2": clean_best["mean_r2"] if clean_best else math.nan,
                "best_noisy_label": best_noisy["noise_label"] if best_noisy else "",
                "best_noisy_mean_mae": best_noisy["mean_mae"] if best_noisy else math.nan,
                "worst_noisy_label": worst_noisy["noise_label"] if worst_noisy else "",
                "worst_noisy_mean_mae": worst_noisy["mean_mae"] if worst_noisy else math.nan,
                "mae_delta_worst_vs_clean": (
                    float(worst_noisy["mean_mae"]) - float(clean_best["mean_mae"]) if clean_best and worst_noisy else math.nan
                ),
                "r2_delta_worst_vs_clean": (
                    float(worst_noisy["mean_r2"]) - float(clean_best["mean_r2"]) if clean_best and worst_noisy else math.nan
                ),
            }
        )
    out.sort(
        key=lambda row: (
            numeric_sort_key(row["clean_mean_mae"]),
            numeric_sort_key(row["mae_delta_worst_vs_clean"]),
            str(row["strategy_base"]),
        )
    )
    return out


def build_branch_registry_rows(row_records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in row_records:
        key = (str(row["phase"]), str(row["phase_label"]), str(row["family"]))
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    for key, rows in grouped.items():
        completed = [row for row in rows if safe_float(row.get("mae")) is not None]
        failed = [row for row in rows if str(row.get("status", "")).upper() == "FAILED"]
        if len(completed) == len(rows) and rows:
            status = "completed"
        elif completed:
            status = "partial"
        elif failed:
            status = "failed"
        else:
            status = "registered"
        best = min(completed, key=lambda row: (float(row["mae"]), -float(row["r2"]))) if completed else None
        out.append(
            {
                "branch": key[0],
                "phase_label": key[1],
                "family": key[2],
                "status": status,
                "total_rows": len(rows),
                "completed_rows": len(completed),
                "failed_rows": len(failed),
                "best_strategy_id": best["strategy_id"] if best else "",
                "best_strategy_base": best["strategy_base"] if best else "",
                "best_row_id": best["row_id"] if best else "",
                "best_mae": best["mae"] if best else math.nan,
                "best_r2": best["r2"] if best else math.nan,
                "noise_labels": ",".join(sorted({str(row.get("noise_label", "")) for row in rows if row.get("noise_label", "")})),
                "evidence": best["artifact_dir"] if best else "",
                "matrix_source": rows[0].get("matrix_source", ""),
                "registry_source": rows[0].get("registry_source", ""),
            }
        )
    out.sort(key=lambda row: (row["status"] != "completed", numeric_sort_key(row["best_mae"]), str(row["branch"])))
    return out


def build_key_results_rows(
    campaign_root: Path,
    run_rows: list[dict[str, object]],
    strategy_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    diagnostics_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(section: str, name: str, metric_name: str, metric_value: object, source: str, note: str) -> None:
        rows.append(
            {
                "section": section,
                "name": name,
                "comparability_class": COMPARABILITY_CLASS,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "source": source,
                "note": note,
            }
        )

    if run_rows:
        best_clean = min(
            (row for row in run_rows if safe_float(row["noise_std"]) == 0.0),
            default=None,
            key=lambda row: (float(row["mae"]), -float(row["r2"])),
        )
        if best_clean is not None:
            add("best_clean_run", str(best_clean["strategy_id"]), "mae", best_clean["mae"], str(best_clean["artifact_dir"]), str(best_clean["phase_label"]))
            add("best_clean_run", str(best_clean["strategy_id"]), "r2", best_clean["r2"], str(best_clean["artifact_dir"]), str(best_clean["phase_label"]))
        best_any = min(run_rows, key=lambda row: (float(row["mae"]), -float(row["r2"])))
        add("best_overall_run", str(best_any["strategy_id"]), "mae", best_any["mae"], str(best_any["artifact_dir"]), str(best_any["phase_label"]))
        add("best_overall_run", str(best_any["strategy_id"]), "runtime_sec", best_any["runtime_sec"], str(best_any["artifact_dir"]), str(best_any["phase_label"]))
        fastest = min(
            (row for row in run_rows if safe_float(row.get("runtime_sec")) is not None),
            default=None,
            key=lambda row: float(row["runtime_sec"]),
        )
        if fastest is not None:
            add("systems_efficiency", str(fastest["strategy_id"]), "runtime_sec", fastest["runtime_sec"], str(fastest["artifact_dir"]), "fastest completed row")

    if robustness_rows:
        comparable = [row for row in robustness_rows if safe_float(row.get("mae_delta_worst_vs_clean")) is not None]
        if comparable:
            most_stable = min(comparable, key=lambda row: float(row["mae_delta_worst_vs_clean"]))
            add(
                "noise_robustness",
                str(most_stable["strategy_base"]),
                "mae_delta_worst_vs_clean",
                most_stable["mae_delta_worst_vs_clean"],
                str(campaign_root / "tables"),
                str(most_stable["phase_label"]),
            )

    for branch in branch_rows:
        if safe_float(branch.get("best_mae")) is None:
            continue
        add("best_by_branch", str(branch["branch"]), "best_mae", branch["best_mae"], str(branch["evidence"]), str(branch["status"]))
        add("best_by_branch", str(branch["branch"]), "best_r2", branch["best_r2"], str(branch["evidence"]), str(branch["status"]))

    if diagnostics_rows:
        best_tail = min(diagnostics_rows, key=lambda row: float(row["p95_abs_error"]))
        add("diagnostics", str(best_tail["strategy_id"]), "p95_abs_error", best_tail["p95_abs_error"], str(best_tail["artifact_dir"]), "lower is better")
        best_target = min(diagnostics_rows, key=lambda row: float(row["worst_target_nmae_range"]))
        add(
            "diagnostics",
            str(best_target["strategy_id"]),
            "worst_target_nmae_range",
            best_target["worst_target_nmae_range"],
            str(best_target["artifact_dir"]),
            "lower is better",
        )

    return rows


def build_limitations_rows() -> list[dict[str, object]]:
    return [
        {
            "limitation_id": "L1",
            "scope": "coverage",
            "statement": "The refresh reads only matrix, registry, metrics, and prediction artifacts present under the supplied campaign root, so missing or pruned rerun rows remain outside scope.",
            "severity": "high",
        },
        {
            "limitation_id": "L2",
            "scope": "comparability",
            "statement": "These A30-only summaries compare direct point-estimation reruns across local branches, noise settings, and schedules, and they should not be merged numerically with posterior-family analyses from separate campaign roots.",
            "severity": "high",
        },
        {
            "limitation_id": "L3",
            "scope": "diagnostics",
            "statement": "The diagnostics are deterministic error-shape diagnostics derived from point predictions and per-target metrics, because posterior-sample calibration artifacts are not part of the A30 direct-output contract.",
            "severity": "medium",
        },
        {
            "limitation_id": "L4",
            "scope": "noise",
            "statement": "Noise robustness summaries depend on campaign-local strategy naming and metrics metadata, so rows without explicit clean or noise tags remain visible but cannot contribute to clean-vs-noisy deltas.",
            "severity": "medium",
        },
        {
            "limitation_id": "L5",
            "scope": "runtime",
            "statement": "Runtime columns prefer campaign-local training metadata, then registry elapsed time, which means they reflect the rerun's local execution contract rather than a globally standardized benchmark protocol.",
            "severity": "medium",
        },
    ]


def build_branch_registry_report(branch_rows: list[dict[str, object]], campaign_tag_text: str) -> str:
    lines = [
        f"# HH Track4 A30 Branch Registry ({campaign_tag_text})",
        "",
        "| branch | phase_label | status | completed_rows | total_rows | best_strategy_id | best_mae |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in branch_rows:
        best_mae = safe_float(row.get("best_mae"))
        mae_text = f"{best_mae:.6f}" if best_mae is not None else ""
        lines.append(
            f"| `{row['branch']}` | `{row['phase_label']}` | `{row['status']}` | {int(row['completed_rows'])} | {int(row['total_rows'])} | `{row['best_strategy_id']}` | {mae_text} |"
        )
    return "\n".join(lines) + "\n"


def build_reporting_synthesis(
    campaign_root: Path,
    tag: str,
    matrix_paths: list[Path],
    registry_paths: list[Path],
    run_rows: list[dict[str, object]],
    strategy_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    diagnostics_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    limitation_rows: list[dict[str, object]],
) -> str:
    lines = [
        f"# HH Track4 A30 Reporting Synthesis ({tag})",
        "",
        "## Scope",
        "",
        (
            "This refresh was regenerated entirely from the supplied campaign root, and it only uses campaign-local "
            "matrix tables, registry tables, per-row metrics summaries, per-target metrics, and point-prediction arrays."
        ),
        "",
        "## Campaign Coverage",
        "",
        f"- campaign root: `{campaign_root}`",
        f"- discovered matrix tables: {len(matrix_paths)}",
        f"- discovered registry tables: {len(registry_paths)}",
        f"- completed runs with usable metrics: {len(run_rows)}",
        f"- strategy aggregates: {len(strategy_rows)}",
        f"- branch aggregates: {len(branch_rows)}",
        "",
        "## Core Findings",
        "",
        "| result space | row | metric | note |",
        "|---|---|---:|---|",
    ]

    clean_rows = [row for row in run_rows if safe_float(row["noise_std"]) == 0.0]
    noisy_rows = [row for row in run_rows if safe_float(row["noise_std"]) not in (None, 0.0)]
    if clean_rows:
        best_clean = min(clean_rows, key=lambda row: (float(row["mae"]), -float(row["r2"])))
        lines.append(
            f"| best clean rerun | `{best_clean['strategy_id']}` | `MAE {float(best_clean['mae']):.6f}` | `{best_clean['phase_label']}` |"
        )
    if noisy_rows:
        best_noisy = min(noisy_rows, key=lambda row: (float(row["mae"]), -float(row["r2"])))
        lines.append(
            f"| best noisy rerun | `{best_noisy['strategy_id']}` | `MAE {float(best_noisy['mae']):.6f}` | `{best_noisy['noise_label']}` under `{best_noisy['phase_label']}` |"
        )
    if diagnostics_rows:
        best_diag = min(diagnostics_rows, key=lambda row: float(row["p95_abs_error"]))
        lines.append(
            f"| lowest tail error | `{best_diag['strategy_id']}` | `p95_abs_error {float(best_diag['p95_abs_error']):.6f}` | `{best_diag['phase_label']}` |"
        )
    fastest_rows = [row for row in run_rows if safe_float(row.get("runtime_sec")) is not None]
    if fastest_rows:
        fastest = min(fastest_rows, key=lambda row: float(row["runtime_sec"]))
        lines.append(
            f"| fastest completed rerun | `{fastest['strategy_id']}` | `runtime {float(fastest['runtime_sec']):.6f} sec` | `{fastest['phase_label']}` |"
        )

    lines += [
        "",
        "## Branch Status",
        "",
        "| branch | status | completed_rows | total_rows | best_strategy_id | best_mae |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in branch_rows:
        mae = safe_float(row["best_mae"])
        mae_text = f"{mae:.6f}" if mae is not None else ""
        lines.append(
            f"| `{row['branch']}` | `{row['status']}` | {int(row['completed_rows'])} | {int(row['total_rows'])} | `{row['best_strategy_id']}` | {mae_text} |"
        )

    lines += [
        "",
        "## Strategy Comparison",
        "",
        "| phase | strategy_base | noise_label | mean_mae | std_mae | mean_r2 | completed_runs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in strategy_rows[:20]:
        lines.append(
            f"| `{row['phase']}` | `{row['strategy_base']}` | `{row['noise_label']}` | {float(row['mean_mae']):.6f} | {float(row['std_mae']):.6f} | {float(row['mean_r2']):.6f} | {int(row['completed_runs'])} |"
        )

    if robustness_rows:
        lines += [
            "",
            "## Noise Robustness",
            "",
            "| strategy_base | clean_mean_mae | worst_noisy_label | worst_noisy_mean_mae | mae_delta_worst_vs_clean |",
            "|---|---:|---|---:|---:|",
        ]
        comparable = [row for row in robustness_rows if safe_float(row["clean_mean_mae"]) is not None]
        for row in comparable[:20]:
            lines.append(
                f"| `{row['strategy_base']}` | {float(row['clean_mean_mae']):.6f} | `{row['worst_noisy_label']}` | {float(row['worst_noisy_mean_mae']):.6f} | {float(row['mae_delta_worst_vs_clean']):.6f} |"
            )

    if diagnostics_rows:
        lines += [
            "",
            "## Diagnostics",
            "",
            "| row | p95_abs_error | worst_target_nmae_range | bias_abs_mean | spread_ratio_mean |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in sorted(diagnostics_rows, key=lambda item: float(item["p95_abs_error"]))[:20]:
            lines.append(
                f"| `{row['strategy_id']}` | {float(row['p95_abs_error']):.6f} | {float(row['worst_target_nmae_range']):.6f} | {float(row['bias_abs_mean']):.6f} | {float(row['spread_ratio_mean']):.6f} |"
            )

    lines += [
        "",
        "## Limitations",
        "",
        "| id | scope | severity | statement |",
        "|---|---|---|---|",
    ]
    for row in limitation_rows:
        lines.append(f"| `{row['limitation_id']}` | `{row['scope']}` | `{row['severity']}` | {row['statement']} |")

    lines += [
        "",
        "## Generated Artifacts",
        "",
        f"- `tables/{OUR_OUTPUT_PREFIX}run_comparison_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}strategy_comparison_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}noise_robustness_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}prediction_diagnostics_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}target_diagnostics_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}literature_branch_registry_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}reporting_key_results_{tag}.csv`",
        f"- `tables/{OUR_OUTPUT_PREFIX}reporting_limitations_{tag}.csv`",
        f"- `reports/{OUR_OUTPUT_PREFIX}literature_branch_registry_{tag}.md`",
        f"- `reports/{OUR_OUTPUT_PREFIX}reporting_synthesis_{tag}.md`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root does not exist: {campaign_root}")

    tag = campaign_tag(campaign_root)
    matrix_paths, registry_paths = discover_input_tables(campaign_root)
    merged_rows = merge_input_rows(matrix_paths, registry_paths)
    row_records, diagnostics_rows, target_rows = build_row_records(campaign_root, merged_rows)
    run_rows = build_run_comparison_rows(row_records)
    strategy_rows = build_strategy_comparison_rows(run_rows)
    robustness_rows = build_noise_robustness_rows(strategy_rows)
    branch_rows = build_branch_registry_rows(row_records)
    key_results_rows = build_key_results_rows(
        campaign_root,
        run_rows,
        strategy_rows,
        robustness_rows,
        diagnostics_rows,
        branch_rows,
    )
    limitation_rows = build_limitations_rows()

    tables_dir = campaign_root / "tables"
    reports_dir = campaign_root / "reports"

    run_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}run_comparison_{tag}.csv"
    strategy_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}strategy_comparison_{tag}.csv"
    robustness_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}noise_robustness_{tag}.csv"
    diagnostics_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}prediction_diagnostics_{tag}.csv"
    target_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}target_diagnostics_{tag}.csv"
    branch_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}literature_branch_registry_{tag}.csv"
    key_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}reporting_key_results_{tag}.csv"
    limitations_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}reporting_limitations_{tag}.csv"
    branch_md = reports_dir / f"{OUR_OUTPUT_PREFIX}literature_branch_registry_{tag}.md"
    report_md = reports_dir / f"{OUR_OUTPUT_PREFIX}reporting_synthesis_{tag}.md"

    write_csv_rows(run_csv, run_rows)
    write_csv_rows(strategy_csv, strategy_rows)
    write_csv_rows(robustness_csv, robustness_rows)
    write_csv_rows(diagnostics_csv, diagnostics_rows)
    write_csv_rows(target_csv, target_rows)
    write_csv_rows(branch_csv, branch_rows)
    write_csv_rows(key_csv, key_results_rows)
    write_csv_rows(limitations_csv, limitation_rows)
    write_text(branch_md, build_branch_registry_report(branch_rows, tag))
    write_text(
        report_md,
        build_reporting_synthesis(
            campaign_root,
            tag,
            matrix_paths,
            registry_paths,
            run_rows,
            strategy_rows,
            robustness_rows,
            diagnostics_rows,
            branch_rows,
            limitation_rows,
        ),
    )

    print(run_csv)
    print(strategy_csv)
    print(robustness_csv)
    print(diagnostics_csv)
    print(target_csv)
    print(branch_csv)
    print(key_csv)
    print(limitations_csv)
    print(branch_md)
    print(report_md)


if __name__ == "__main__":
    main()
