#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
SCRIPT_DIR = REPO / "scripts"
CLASSICAL_HELPER = SCRIPT_DIR / "run_hh_classical_baseline_20260420.py"


def import_classical_helper():
    spec = importlib.util.spec_from_file_location("hh_classical_baseline_20260420", CLASSICAL_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helper script: {CLASSICAL_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CPU-side standardized efficiency microbenchmark for HH Track4 classical representatives."
    )
    ap.add_argument("--params", required=True)
    ap.add_argument("--baseline", required=True, choices=["random_forest_500", "knn_k11"])
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12", choices=["raw_plus_fft256_summary12"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-prefix", default=None)
    ap.add_argument("--curr", default=None)
    ap.add_argument("--summary-csv", default=None)
    return ap.parse_args()


def peak_rss_mb() -> float | None:
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # Linux reports kilobytes; macOS reports bytes. This repo runs on Linux cluster nodes.
    return rss / 1024.0 if rss > 0 else None


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_params(params: dict) -> dict:
    params = json.loads(json.dumps(params))
    params.setdefault("data", {})
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["split_array_cache_dir"] = str(
        REPO
        / "data"
        / "important_notes"
        / "optimization_track_20260411_v100_interactive"
        / "shared_data"
        / "track4_hh_full"
        / "concatenated_data"
        / ".split_array_cache"
        / "459f596906888c8d"
    )
    params["data"]["features_scale_cache_enabled"] = True
    params["data"]["features_normalize"] = True
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["features_sub_length"] = 2000
    params["data"]["features_sub_step"] = 1
    params["data"]["features_additive_noise_std"] = 0.0
    params["data"]["features_multiplicative_noise_std"] = 0.0
    params["data"]["features_baseline_drift_std"] = 0.0
    params["data"]["features_mask_fraction"] = 0.0
    return params


def inverse_targets(arr: np.ndarray, targets_scale: dict, classical) -> np.ndarray:
    out = np.array(arr, copy=True)
    out = classical._apply_scale_inverse(out, targets_scale)
    transform_cfg = targets_scale.get("transform") if isinstance(targets_scale, dict) else None
    return classical._apply_targets_transform(out, transform_cfg, inverse=True, array_name="targets")


def write_summary_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    classical = import_classical_helper()
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as exc:  # pragma: no cover - handled at runtime on cluster
        raise SystemExit(
            "run_hh_track4_checkpoint7_standardized_efficiency_20260423.py requires scikit-learn "
            "inside the benchmark environment."
        ) from exc

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hh_track4_cp7_standardized_efficiency")

    params = prepare_params(load_params(args.params))
    if args.data_dir is not None:
        params["data"]["data_dir"] = args.data_dir
    if args.data_prefix is not None:
        params["data"]["data_prefix"] = args.data_prefix
    if args.curr is not None:
        params["data"]["curr"] = args.curr

    helper_args = SimpleNamespace(
        data_dir=args.data_dir,
        data_prefix=args.data_prefix,
        curr=args.curr,
        feature_mode=args.feature_mode,
    )

    t_total0 = time.perf_counter()
    t_load0 = time.perf_counter()
    x_train, _x_validate, x_eval, y_train, _y_validate, y_eval, targets_scale = classical.load_cached_splits(
        params,
        helper_args,
        logger,
    )
    feature_load_sec = time.perf_counter() - t_load0

    family, reducer, estimator = classical.build_estimator(args.baseline)
    logger.info(
        "baseline=%s family=%s repeat=%d feature_mode=%s x_train=%s x_eval=%s",
        args.baseline,
        family,
        args.repeat,
        args.feature_mode,
        x_train.shape,
        x_eval.shape,
    )

    t_fit0 = time.perf_counter()
    if reducer is not None:
        x_train_fit = reducer.fit_transform(x_train)
        x_eval_fit = reducer.transform(x_eval)
    else:
        x_train_fit = x_train
        x_eval_fit = x_eval
    estimator.fit(x_train_fit, y_train)
    fit_sec = time.perf_counter() - t_fit0

    t_eval0 = time.perf_counter()
    y_pred = estimator.predict(x_eval_fit)
    eval_sec = time.perf_counter() - t_eval0
    total_sec = time.perf_counter() - t_total0

    y_pred_post = inverse_targets(y_pred, targets_scale, classical)
    y_eval_post = inverse_targets(y_eval, targets_scale, classical)
    mae = float(mean_absolute_error(y_eval_post, y_pred_post))
    mse = float(mean_squared_error(y_eval_post, y_pred_post))
    r2 = float(r2_score(y_eval_post, y_pred_post))

    n_train = int(x_train.shape[0])
    n_eval = int(x_eval.shape[0])
    metrics = {
        "baseline": args.baseline,
        "repeat": int(args.repeat),
        "feature_mode": args.feature_mode,
        "feature_load_sec": float(feature_load_sec),
        "fit_sec": float(fit_sec),
        "eval_sec": float(eval_sec),
        "total_sec": float(total_sec),
        "peak_rss_mb": peak_rss_mb(),
        "n_train": n_train,
        "n_eval": n_eval,
        "sec_per_1k_train": float(fit_sec / max(1.0, n_train / 1000.0)),
        "ms_per_eval_example": float(1000.0 * eval_sec / max(1, n_eval)),
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "params": str(Path(args.params)),
        "helper_source": str(CLASSICAL_HELPER),
    }

    per_target = []
    for idx in range(y_eval_post.shape[1]):
        per_target.append(
            {
                "target_index": idx,
                "mae": float(mean_absolute_error(y_eval_post[:, idx], y_pred_post[:, idx])),
                "mse": float(mean_squared_error(y_eval_post[:, idx], y_pred_post[:, idx])),
                "r2": float(r2_score(y_eval_post[:, idx], y_pred_post[:, idx])),
            }
        )

    (save_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (save_dir / "metrics_per_target.json").write_text(json.dumps(per_target, indent=2) + "\n")
    (save_dir / "params_snapshot.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    np.savez_compressed(save_dir / "predictions.npz", targets=y_eval_post, predictions=y_pred_post)
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), metrics)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
