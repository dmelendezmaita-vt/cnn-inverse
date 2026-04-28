#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wasserstein_distance


def iter_prediction_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.name == "predictions.npz":
            yield path
        return
    yield from sorted(path.rglob("predictions.npz"))


def load_target_names(run_dir: Path, n_targets: int) -> list[str]:
    params_path = run_dir / "params.yaml"
    if params_path.exists():
        try:
            import yaml

            params = yaml.safe_load(params_path.read_text())
            names = params.get("data", {}).get("target_names")
            if isinstance(names, list) and len(names) == n_targets:
                return [str(x) for x in names]
        except Exception:
            pass
    return [f"target_{i}" for i in range(n_targets)]


def safe_log10(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, 1.0e-12, None))


def analyze_file(predictions_file: Path) -> list[dict[str, object]]:
    run_dir = predictions_file.parent
    data = np.load(predictions_file)
    rows: list[dict[str, object]] = []

    split_names = sorted({key[:-5] for key in data.files if key.endswith("_true")})
    for split in split_names:
        true_key = f"{split}_true"
        pred_key = f"{split}_pred"
        if true_key not in data or pred_key not in data:
            continue
        y_true = np.asarray(data[true_key])
        y_pred = np.asarray(data[pred_key])
        if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
            continue

        target_names = load_target_names(run_dir, y_true.shape[1])
        for i in range(y_true.shape[1]):
            true_col = y_true[:, i]
            pred_col = y_pred[:, i]
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "predictions_file": str(predictions_file),
                    "split": split,
                    "target_index": i,
                    "target_name": target_names[i],
                    "wasserstein_raw": float(wasserstein_distance(true_col, pred_col)),
                    "wasserstein_log10": float(
                        wasserstein_distance(safe_log10(true_col), safe_log10(pred_col))
                    ),
                    "true_mean": float(np.mean(true_col)),
                    "pred_mean": float(np.mean(pred_col)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_files": len({row["predictions_file"] for row in rows}),
        "n_rows": len(rows),
        "splits": sorted({str(row["split"]) for row in rows}),
    }
    per_split = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        per_split[split] = {
            "mean_wasserstein_raw": float(np.mean([row["wasserstein_raw"] for row in split_rows])),
            "mean_wasserstein_log10": float(np.mean([row["wasserstein_log10"] for row in split_rows])),
        }
    summary["per_split"] = per_split
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-target Wasserstein diagnostics from predictions.npz files.")
    ap.add_argument("path", help="Run directory, package directory, or predictions.npz file")
    ap.add_argument("--csv-out", default=None, help="Optional CSV output path")
    ap.add_argument("--json-out", default=None, help="Optional JSON summary output path")
    args = ap.parse_args()

    source = Path(args.path)
    files = list(iter_prediction_files(source))
    rows: list[dict[str, object]] = []
    for file in files:
        rows.extend(analyze_file(file))

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))

    if args.csv_out:
        write_csv(Path(args.csv_out), rows)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
