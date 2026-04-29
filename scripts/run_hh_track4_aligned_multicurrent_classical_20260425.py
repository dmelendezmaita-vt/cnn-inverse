#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from shared_data_utils import ensure_shared_data

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required") from exc

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_DATA_DIR = "concatenated_data.tar.gz"
sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))

from data import (  # noqa: E402
    _apply_scale_inverse,
    _apply_targets_transform,
    _resolve_split_array_cache_dir,
    load_data,
    preprocess_targets,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aligned multi-current deterministic baselines for HH Track4.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--baseline", required=True, choices=["ridge", "mlp"])
    ap.add_argument("--aggregation", required=True, choices=["concat", "meanstd"])
    ap.add_argument(
        "--feature-mode",
        default="raw_plus_fft256_summary12",
        choices=["raw", "summary12", "raw_plus_summary12", "fft256_summary12", "raw_plus_fft256_summary12"],
    )
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--data-dir", default=DEFAULT_SHARED_DATA_DIR)
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-validate", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--svd-components", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_local_data_dir(data_dir: str, data_prefix: str) -> str:
    path = Path(str(data_dir)).expanduser()
    if not path.is_absolute():
        path = (REPO / path).resolve()
    text = str(path)
    is_tar = path.is_file() and text.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"))
    if not is_tar:
        return str(path)

    prefix = str(data_prefix or path.stem).strip("/.")
    prepared_root = REPO / ".prepared_data"
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared_dir = prepared_root / prefix
    ensure_shared_data(path, prepared_dir)
    return str(prepared_dir)


def crop_trace_window(arr: np.ndarray, count: int, sub_length: int | None, sub_step: int | None = None) -> np.ndarray:
    out = np.asarray(arr[:count])
    if sub_length and sub_length < out.shape[-1]:
        out = out[..., :sub_length]
    if sub_step and sub_step > 1:
        out = out[..., ::sub_step]
    return out.astype(np.float32, copy=False)


def flatten_time_window(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(arr.shape[0], -1)


def compute_summary12(arr: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr, dtype=np.float32).reshape(arr.shape[0], -1)
    diffs = np.diff(traces, axis=1)
    summary = np.column_stack(
        [
            np.mean(traces, axis=1),
            np.std(traces, axis=1),
            np.min(traces, axis=1),
            np.max(traces, axis=1),
            np.median(traces, axis=1),
            np.quantile(traces, 0.10, axis=1),
            np.quantile(traces, 0.90, axis=1),
            np.sqrt(np.mean(np.square(traces), axis=1)),
            np.mean(np.abs(diffs), axis=1),
            np.max(np.abs(diffs), axis=1),
            np.mean(traces > 0.0, axis=1),
            np.mean(traces > -20.0, axis=1),
        ]
    )
    return summary.astype(np.float32, copy=False)


def compute_fft256(arr: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr, dtype=np.float32).reshape(arr.shape[0], -1)
    centered = traces - np.mean(traces, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    magnitudes = np.log1p(np.abs(spectrum[:, 1:257]))
    return magnitudes.astype(np.float32, copy=False)


def materialize_feature_mode(flat_block: np.ndarray, feature_mode: str) -> np.ndarray:
    raw = np.asarray(flat_block, dtype=np.float32)
    summary = None
    fft = None

    if feature_mode == "raw":
        return raw
    if feature_mode in {"summary12", "raw_plus_summary12", "fft256_summary12", "raw_plus_fft256_summary12"}:
        summary = compute_summary12(raw)
    if feature_mode in {"fft256_summary12", "raw_plus_fft256_summary12"}:
        fft = compute_fft256(raw)

    if feature_mode == "summary12":
        return summary
    if feature_mode == "raw_plus_summary12":
        return np.concatenate([raw, summary], axis=1).astype(np.float32, copy=False)
    if feature_mode == "fft256_summary12":
        return np.concatenate([fft, summary], axis=1).astype(np.float32, copy=False)
    if feature_mode == "raw_plus_fft256_summary12":
        return np.concatenate([raw, fft, summary], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature_mode={feature_mode}")


def normalize_features(x_train: np.ndarray, x_validate: np.ndarray, x_test: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        return x_train, x_validate, x_test
    shift = np.mean(x_train, axis=0, keepdims=True).astype(np.float32, copy=False)
    mult = np.std(x_train, axis=0, keepdims=True).astype(np.float32, copy=False)
    mult = np.where(mult == 0.0, 1.0, mult)
    return (x_train - shift) / mult, (x_validate - shift) / mult, (x_test - shift) / mult


def load_one_current(
    params: dict,
    *,
    curr: str,
    data_dir: str,
    data_prefix: str,
    n_train: int,
    n_validate: int,
    n_test: int,
    feature_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    params = json.loads(json.dumps(params))
    params["data"]["data_dir"] = resolve_local_data_dir(data_dir, data_prefix)
    params["data"]["data_prefix"] = data_prefix
    params["data"]["curr"] = curr
    params["data"]["Ntrain"] = int(n_train)
    params["data"]["Nvalidate"] = int(n_validate)
    params["data"]["Ntest"] = int(n_test)
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    cache_dir = _resolve_split_array_cache_dir(params["data"])
    if cache_dir is None:
        raise SystemExit(f"Unable to resolve split cache directory for curr={curr}.")
    required = [
        cache_dir / "features_train.npy",
        cache_dir / "features_validate.npy",
        cache_dir / "features_test.npy",
        cache_dir / "targets_train.npy",
        cache_dir / "targets_validate.npy",
        cache_dir / "targets_test.npy",
    ]
    if not all(path.exists() for path in required):
        logger = logging.getLogger("aligned_multicurrent_classical")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        load_data(params, logger)

    feature_train = np.load(cache_dir / "features_train.npy", mmap_mode="r")
    feature_validate = np.load(cache_dir / "features_validate.npy", mmap_mode="r")
    feature_test = np.load(cache_dir / "features_test.npy", mmap_mode="r")
    target_train = np.load(cache_dir / "targets_train.npy", mmap_mode="r")
    target_validate = np.load(cache_dir / "targets_validate.npy", mmap_mode="r")
    target_test = np.load(cache_dir / "targets_test.npy", mmap_mode="r")

    sub_length = params["data"].get("features_sub_length")
    sub_step = int(params["data"].get("features_sub_step", 0) or 0)
    train_window = crop_trace_window(feature_train, n_train, sub_length, sub_step=sub_step)
    validate_window = crop_trace_window(feature_validate, n_validate, sub_length, sub_step=sub_step)
    test_window = crop_trace_window(feature_test, n_test, sub_length, sub_step=sub_step)

    x_train_raw = flatten_time_window(train_window)
    x_validate_raw = flatten_time_window(validate_window)
    x_test_raw = flatten_time_window(test_window)
    x_train_raw, x_validate_raw, x_test_raw = normalize_features(
        x_train_raw,
        x_validate_raw,
        x_test_raw,
        enabled=bool(params["data"].get("features_normalize", False)),
    )
    features = {
        "train": materialize_feature_mode(x_train_raw, feature_mode),
        "validate": materialize_feature_mode(x_validate_raw, feature_mode),
        "test": materialize_feature_mode(x_test_raw, feature_mode),
    }
    targets = {
        "train": np.array(target_train[:n_train], copy=True).astype(np.float32, copy=False),
        "validate": np.array(target_validate[:n_validate], copy=True).astype(np.float32, copy=False),
        "test": np.array(target_test[:n_test], copy=True).astype(np.float32, copy=False),
    }
    return features, targets


def aggregate_currents(per_current_features: list[np.ndarray], aggregation: str) -> np.ndarray:
    if aggregation == "concat":
        return np.concatenate(per_current_features, axis=1).astype(np.float32, copy=False)
    stacked = np.stack(per_current_features, axis=1).astype(np.float32, copy=False)
    mean = np.mean(stacked, axis=1)
    std = np.std(stacked, axis=1)
    return np.concatenate([mean, std], axis=1).astype(np.float32, copy=False)


def build_model(args: argparse.Namespace):
    if args.baseline == "ridge":
        return Ridge(alpha=1.0, random_state=args.seed)
    if args.baseline == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=1.0e-4,
            batch_size=128,
            learning_rate_init=1.0e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=args.seed,
            verbose=False,
        )
    raise ValueError(args.baseline)


def postprocess_targets_array(y_norm: np.ndarray, targets_scale: dict) -> np.ndarray:
    restored = _apply_scale_inverse(y_norm.astype(np.float32, copy=False), targets_scale)
    restored = _apply_targets_transform(restored, targets_scale.get("transform"), inverse=True, array_name="targets")
    return restored.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("aligned_multicurrent_classical")

    params = load_params(args.params)
    if args.data_dir is not None:
        params["data"]["data_dir"] = args.data_dir
    if args.data_prefix is not None:
        params["data"]["data_prefix"] = args.data_prefix

    currents = [x.strip() for x in args.currents.split(",") if x.strip()]
    n_train = int(args.n_train or params["data"]["Ntrain"])
    n_validate = int(args.n_validate or params["data"].get("Nvalidate", 1024))
    n_test = int(args.n_test or params["data"].get("Ntest", 1024))
    seed = int(args.seed if args.seed is not None else params["data"].get("random_seed", 0))

    features_by_split = {"train": [], "validate": [], "test": []}
    targets_ref = None
    for curr in currents:
        features_curr, targets_curr = load_one_current(
            params,
            curr=curr,
            data_dir=args.data_dir,
            data_prefix=args.data_prefix,
            n_train=n_train,
            n_validate=n_validate,
            n_test=n_test,
            feature_mode=args.feature_mode,
        )
        for split in features_by_split:
            features_by_split[split].append(features_curr[split])
        if targets_ref is None:
            targets_ref = targets_curr
        else:
            same_frac = float(np.mean(np.all(np.isclose(targets_ref["train"][:256], targets_curr["train"][:256]), axis=1)))
            logger.info("alignment check curr=%s rowwise_equal_frac_train_first256=%.6f", curr, same_frac)

    assert targets_ref is not None
    targets_scale = preprocess_targets(targets_ref, params, logger)
    y_train = targets_ref["train"]
    y_validate = targets_ref["validate"]
    y_test = targets_ref["test"]

    x_train = aggregate_currents(features_by_split["train"], args.aggregation)
    x_validate = aggregate_currents(features_by_split["validate"], args.aggregation)
    x_test = aggregate_currents(features_by_split["test"], args.aggregation)
    logger.info(
        "aligned multicurrent features built: aggregation=%s feature_mode=%s x_train=%s x_validate=%s x_test=%s",
        args.aggregation,
        args.feature_mode,
        x_train.shape,
        x_validate.shape,
        x_test.shape,
    )

    svd_model = None
    if args.svd_components and args.svd_components > 0 and args.svd_components < x_train.shape[1]:
        svd_model = TruncatedSVD(n_components=args.svd_components, random_state=seed)
        x_train = svd_model.fit_transform(x_train).astype(np.float32, copy=False)
        x_validate = svd_model.transform(x_validate).astype(np.float32, copy=False)
        x_test = svd_model.transform(x_test).astype(np.float32, copy=False)
        logger.info("applied TruncatedSVD n_components=%d", args.svd_components)

    model = build_model(args)
    started = time.time()
    if args.baseline == "mlp":
        model.fit(np.concatenate([x_train, x_validate], axis=0), np.concatenate([y_train, y_validate], axis=0))
    else:
        model.fit(x_train, y_train)
    train_elapsed = time.time() - started

    y_train_pred = np.asarray(model.predict(x_train), dtype=np.float32)
    y_validate_pred = np.asarray(model.predict(x_validate), dtype=np.float32)
    y_test_pred = np.asarray(model.predict(x_test), dtype=np.float32)

    y_train_post = postprocess_targets_array(y_train, targets_scale)
    y_validate_post = postprocess_targets_array(y_validate, targets_scale)
    y_test_post = postprocess_targets_array(y_test, targets_scale)
    y_train_pred_post = postprocess_targets_array(y_train_pred, targets_scale)
    y_validate_pred_post = postprocess_targets_array(y_validate_pred, targets_scale)
    y_test_pred_post = postprocess_targets_array(y_test_pred, targets_scale)

    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    metrics_rows = [
        {"split": "train", **metrics(y_train_post, y_train_pred_post)},
        {"split": "validate", **metrics(y_validate_post, y_validate_pred_post)},
        {"split": "test", **metrics(y_test_post, y_test_pred_post)},
    ]
    with (save_dir / "metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    np.savez_compressed(
        save_dir / "predictions.npz",
        train_true=y_train_post,
        train_pred=y_train_pred_post,
        validate_true=y_validate_post,
        validate_pred=y_validate_pred_post,
        test_true=y_test_post,
        test_pred=y_test_pred_post,
    )

    summary = {
        "baseline": args.baseline,
        "aggregation": args.aggregation,
        "feature_mode": args.feature_mode,
        "currents": currents,
        "n_train": n_train,
        "n_validate": n_validate,
        "n_test": n_test,
        "svd_components": args.svd_components,
        "train_elapsed_sec": train_elapsed,
        "metrics_test": metrics_rows[-1],
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (save_dir / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
