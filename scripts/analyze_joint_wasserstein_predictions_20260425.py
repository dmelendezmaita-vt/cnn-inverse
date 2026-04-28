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


def sample_unit_directions(dim: int, n_proj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, dim))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return dirs / norms


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, *, n_proj: int, seed: int) -> float:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected two 2D arrays with matching second dimension, got {x.shape} and {y.shape}")
    dirs = sample_unit_directions(x.shape[1], n_proj, seed)
    vals = []
    for d in dirs:
        px = x @ d
        py = y @ d
        vals.append(float(wasserstein_distance(px, py)))
    return float(np.mean(vals))


def analyze_file(predictions_file: Path, *, n_proj: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = predictions_file.parent
    data = np.load(predictions_file)
    per_target: list[dict[str, object]] = []
    per_split: list[dict[str, object]] = []

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
            per_target.append(
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

        per_split.append(
            {
                "run_dir": str(run_dir),
                "predictions_file": str(predictions_file),
                "split": split,
                "n_targets": y_true.shape[1],
                "sliced_wasserstein_raw": sliced_wasserstein_distance(y_true, y_pred, n_proj=n_proj, seed=seed),
                "sliced_wasserstein_log10": sliced_wasserstein_distance(
                    safe_log10(y_true), safe_log10(y_pred), n_proj=n_proj, seed=seed
                ),
            }
        )
    return per_target, per_split


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


def summarize(per_target: list[dict[str, object]], per_split: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {
        "n_files": len({row["predictions_file"] for row in per_target}),
        "n_target_rows": len(per_target),
        "n_split_rows": len(per_split),
        "splits": sorted({str(row["split"]) for row in per_split}),
    }
    split_summary = {}
    for split in sorted({str(row["split"]) for row in per_split}):
        split_target = [row for row in per_target if row["split"] == split]
        split_joint = [row for row in per_split if row["split"] == split]
        split_summary[split] = {
            "mean_wasserstein_raw": float(np.mean([row["wasserstein_raw"] for row in split_target])),
            "mean_wasserstein_log10": float(np.mean([row["wasserstein_log10"] for row in split_target])),
            "mean_sliced_wasserstein_raw": float(np.mean([row["sliced_wasserstein_raw"] for row in split_joint])),
            "mean_sliced_wasserstein_log10": float(np.mean([row["sliced_wasserstein_log10"] for row in split_joint])),
        }
    out["per_split"] = split_summary
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-target and joint sliced-Wasserstein diagnostics from predictions.npz files.")
    ap.add_argument("path", help="Run directory, package directory, or predictions.npz file")
    ap.add_argument("--csv-target-out")
    ap.add_argument("--csv-split-out")
    ap.add_argument("--json-out")
    ap.add_argument("--n-proj", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260425)
    args = ap.parse_args()

    source = Path(args.path)
    files = list(iter_prediction_files(source))
    per_target: list[dict[str, object]] = []
    per_split: list[dict[str, object]] = []
    for file in files:
        target_rows, split_rows = analyze_file(file, n_proj=args.n_proj, seed=args.seed)
        per_target.extend(target_rows)
        per_split.extend(split_rows)

    summary = summarize(per_target, per_split)
    print(json.dumps(summary, indent=2))

    if args.csv_target_out:
        write_csv(Path(args.csv_target_out), per_target)
    if args.csv_split_out:
        write_csv(Path(args.csv_split_out), per_split)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
