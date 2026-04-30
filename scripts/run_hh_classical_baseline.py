#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
except ImportError as exc:  # pragma: no cover - handled at runtime on cluster
    raise SystemExit(
        "run_hh_classical_baseline_20260420.py requires scikit-learn inside the cluster environment."
    ) from exc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hh_repo_utils import resolve_hh_dataset_root  # noqa: E402

from data import (  # noqa: E402
    _apply_scale_inverse,
    _apply_targets_transform,
    _load_features_scale_cache,
    _resolve_features_scale_cache_path,
    _resolve_split_array_cache_dir,
    _save_features_scale_cache,
    load_data,
    preprocess_features,
    preprocess_targets,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run classical non-neural baselines on the HH inverse task.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument(
        "--feature-mode",
        default="raw",
        choices=[
            "raw",
            "summary12",
            "raw_plus_summary12",
            "fft256_summary12",
            "raw_plus_fft256_summary12",
        ],
    )
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-prefix", default=None)
    ap.add_argument("--curr", default=None)
    ap.add_argument("--features-additive-noise-std", type=float, default=None)
    ap.add_argument("--features-multiplicative-noise-std", type=float, default=None)
    ap.add_argument("--features-baseline-drift-std", type=float, default=None)
    ap.add_argument("--features-mask-fraction", type=float, default=None)
    ap.add_argument("--random-seed", type=int, default=None)
    return ap.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_local_data_dir(params: dict, logger: logging.Logger) -> dict:
    params = json.loads(json.dumps(params))
    data_cfg = params["data"]
    resolved = resolve_hh_dataset_root(
        data_cfg["data_dir"],
        data_cfg.get("data_prefix"),
        curr=str(data_cfg.get("curr")) if data_cfg.get("curr") is not None else None,
    )
    logger.info("Resolved HH dataset root to %s", resolved)
    data_cfg["data_dir"] = str(resolved)
    return params


def crop_trace_window(arr: np.ndarray, sub_length: int | None, sub_step: int | None = None) -> np.ndarray:
    out = np.asarray(arr)
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


def materialize_features(arr: np.ndarray, sub_length: int | None, feature_mode: str, sub_step: int | None = None) -> np.ndarray:
    window = crop_trace_window(arr, sub_length, sub_step=sub_step)
    raw = flatten_time_window(window)
    summary = None
    fft = None

    if feature_mode == "raw":
        return raw
    if feature_mode in {"summary12", "raw_plus_summary12", "fft256_summary12", "raw_plus_fft256_summary12"}:
        summary = compute_summary12(window)
    if feature_mode in {"fft256_summary12", "raw_plus_fft256_summary12"}:
        fft = compute_fft256(window)

    if feature_mode == "summary12":
        return summary
    if feature_mode == "raw_plus_summary12":
        return np.concatenate([raw, summary], axis=1).astype(np.float32, copy=False)
    if feature_mode == "fft256_summary12":
        return np.concatenate([fft, summary], axis=1).astype(np.float32, copy=False)
    if feature_mode == "raw_plus_fft256_summary12":
        return np.concatenate([raw, fft, summary], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature_mode={feature_mode}")


def normalize_test_features(
    raw_test: np.ndarray,
    scale: dict,
    sub_length: int | None,
    sub_step: int | None,
    noise_std: float,
    multiplicative_noise_std: float,
    baseline_drift_std: float,
    mask_fraction: float,
    seed: int,
    feature_mode: str,
) -> np.ndarray:
    out = np.array(raw_test, copy=True)
    out = (out - scale["shift"]) * (1.0 / scale["mult"])
    out = apply_eval_shift(
        out,
        additive_noise_std=noise_std,
        multiplicative_noise_std=multiplicative_noise_std,
        baseline_drift_std=baseline_drift_std,
        mask_fraction=mask_fraction,
        seed=seed,
    )
    return materialize_features(out, sub_length, feature_mode, sub_step=sub_step)


def feature_cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "train": cache_dir / "features_train.npy",
        "validate": cache_dir / "features_validate.npy",
        "test": cache_dir / "features_test.npy",
    }


def target_cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "train": cache_dir / "targets_train.npy",
        "validate": cache_dir / "targets_validate.npy",
        "test": cache_dir / "targets_test.npy",
    }


def load_feature_scale_from_cache_or_train(
    feature_train_raw: np.ndarray,
    n_train: int,
    data_params: dict,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    cache_path = _resolve_features_scale_cache_path(data_params, array_name="features")
    if cache_path is not None and cache_path.exists():
        scale = _load_features_scale_cache(cache_path)
        logger.info("features scale cache hit = %s", cache_path)
        return scale

    train_view = feature_train_raw[:n_train]
    scale = {
        "shift": np.zeros((1, train_view.shape[1], 1), dtype=train_view.dtype),
        "mult": np.ones((1, train_view.shape[1], 1), dtype=train_view.dtype),
    }
    if data_params.get("features_normalize", False):
        scale = {
            "shift": np.mean(train_view, axis=(0, 2), keepdims=True).astype(np.float32, copy=False),
            "mult": np.std(train_view, axis=(0, 2), keepdims=True).astype(np.float32, copy=False),
        }
        scale["mult"] = np.where(scale["mult"] == 0.0, 1.0, scale["mult"])
    if cache_path is not None:
        try:
            _save_features_scale_cache(cache_path, scale)
            logger.info("features scale cache write = %s", cache_path)
        except Exception as exc:  # pragma: no cover - best effort cache write
            logger.warning("features scale cache write failed at %s: %s", cache_path, exc)
    return scale


def materialize_scaled_block(
    arr: np.ndarray,
    count: int,
    sub_length: int | None,
    scale: dict[str, np.ndarray],
    *,
    additive_noise_std: float = 0.0,
    multiplicative_noise_std: float = 0.0,
    baseline_drift_std: float = 0.0,
    mask_fraction: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    if count <= 0:
        width = sub_length if sub_length is not None else arr.shape[-1]
        return np.empty((0, arr.shape[1], width), dtype=np.float32)
    width = sub_length if sub_length is not None else arr.shape[-1]
    block = np.array(arr[:count, :, :width], copy=True).astype(np.float32, copy=False)
    block = (block - scale["shift"]) * (1.0 / scale["mult"])
    block = apply_eval_shift(
        block,
        additive_noise_std=additive_noise_std,
        multiplicative_noise_std=multiplicative_noise_std,
        baseline_drift_std=baseline_drift_std,
        mask_fraction=mask_fraction,
        seed=seed,
    )
    return block.astype(np.float32, copy=False)


def apply_eval_shift(
    arr: np.ndarray,
    *,
    additive_noise_std: float = 0.0,
    multiplicative_noise_std: float = 0.0,
    baseline_drift_std: float = 0.0,
    mask_fraction: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    out = np.array(arr, copy=True).astype(np.float32, copy=False)
    if (
        additive_noise_std <= 0.0
        and multiplicative_noise_std <= 0.0
        and baseline_drift_std <= 0.0
        and mask_fraction <= 0.0
    ):
        return out
    rng = np.random.default_rng(seed)
    if additive_noise_std > 0.0:
        out = out + rng.normal(loc=0.0, scale=additive_noise_std, size=out.shape).astype(np.float32, copy=False)
    if multiplicative_noise_std > 0.0:
        out = out * (
            1.0 + rng.normal(loc=0.0, scale=multiplicative_noise_std, size=out.shape).astype(np.float32, copy=False)
        )
    if baseline_drift_std > 0.0:
        drift_axis = np.linspace(-1.0, 1.0, out.shape[-1], dtype=np.float32).reshape(1, 1, -1)
        drift_coeff = rng.normal(loc=0.0, scale=baseline_drift_std, size=(out.shape[0], out.shape[1], 1)).astype(
            np.float32,
            copy=False,
        )
        out = out + drift_coeff * drift_axis
    if mask_fraction > 0.0:
        seg_len = max(1, int(round(mask_fraction * out.shape[-1])))
        seg_len = min(seg_len, out.shape[-1])
        for i in range(out.shape[0]):
            start = int(rng.integers(0, out.shape[-1] - seg_len + 1))
            out[i, :, start:start + seg_len] = 0.0
    return out


def load_cached_splits(
    params: dict,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    params = json.loads(json.dumps(params))
    if args.data_dir is not None:
        params["data"]["data_dir"] = args.data_dir
    if args.data_prefix is not None:
        params["data"]["data_prefix"] = args.data_prefix
    if args.curr is not None:
        params["data"]["curr"] = args.curr
    if args.features_additive_noise_std is not None:
        params["data"]["features_additive_noise_std"] = float(args.features_additive_noise_std)
    if args.features_multiplicative_noise_std is not None:
        params["data"]["features_multiplicative_noise_std"] = float(args.features_multiplicative_noise_std)
    if args.features_baseline_drift_std is not None:
        params["data"]["features_baseline_drift_std"] = float(args.features_baseline_drift_std)
    if args.features_mask_fraction is not None:
        params["data"]["features_mask_fraction"] = float(args.features_mask_fraction)
    if args.random_seed is not None:
        params["data"]["random_seed"] = int(args.random_seed)
    params = resolve_local_data_dir(params, logger)

    params["data"]["split_array_cache_enabled"] = True
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False

    cache_dir = _resolve_split_array_cache_dir(params["data"])
    if cache_dir is None:
        raise SystemExit("Unable to resolve split-array cache for the requested HH config.")
    if not cache_dir.exists():
        logger.info("Split-array cache missing at %s, building it now.", cache_dir)
        load_data(params, logger)
    logger.info("Using split-array cache %s", cache_dir)

    feature_files = feature_cache_files(cache_dir)
    target_files = target_cache_files(cache_dir)
    if not all(path.exists() for path in [*feature_files.values(), *target_files.values()]):
        logger.info("Split-array cache incomplete at %s, rebuilding it now.", cache_dir)
        load_data(params, logger)
    if not all(path.exists() for path in [*feature_files.values(), *target_files.values()]):
        raise SystemExit(f"Incomplete split-array cache at {cache_dir}")

    feature_train_raw = np.load(feature_files["train"], mmap_mode="r")
    feature_validate_raw = np.load(feature_files["validate"], mmap_mode="r")
    feature_test_raw = np.load(feature_files["test"], mmap_mode="r")
    target_train_raw = np.load(target_files["train"], mmap_mode="r")
    target_validate_raw = np.load(target_files["validate"], mmap_mode="r")
    target_test_raw = np.load(target_files["test"], mmap_mode="r")

    n_train = int(params["data"]["Ntrain"])
    n_validate = int(params["data"]["Nvalidate"])
    n_test = int(params["data"]["Ntest"])
    sub_length = params["data"].get("features_sub_length")
    noise_std = float(params["data"].get("features_additive_noise_std", 0.0) or 0.0)
    multiplicative_noise_std = float(params["data"].get("features_multiplicative_noise_std", 0.0) or 0.0)
    baseline_drift_std = float(params["data"].get("features_baseline_drift_std", 0.0) or 0.0)
    mask_fraction = float(params["data"].get("features_mask_fraction", 0.0) or 0.0)
    seed = int(params["data"]["random_seed"])

    feature_scale = load_feature_scale_from_cache_or_train(feature_train_raw, n_train, params["data"], logger)
    sub_step = int(params["data"].get("features_sub_step", 0) or 0)

    train_block = materialize_scaled_block(feature_train_raw, n_train, sub_length, feature_scale)
    validate_block = materialize_scaled_block(feature_validate_raw, n_validate, sub_length, feature_scale)
    test_block = materialize_scaled_block(
        feature_test_raw,
        n_test,
        sub_length,
        feature_scale,
        additive_noise_std=noise_std,
        multiplicative_noise_std=multiplicative_noise_std,
        baseline_drift_std=baseline_drift_std,
        mask_fraction=mask_fraction,
        seed=seed,
    )

    x_train = materialize_features(train_block, None, args.feature_mode, sub_step=sub_step)
    x_validate = materialize_features(validate_block, None, args.feature_mode, sub_step=sub_step)
    x_test = materialize_features(test_block, None, args.feature_mode, sub_step=sub_step)

    targets = {
        "train": np.array(target_train_raw[:n_train], copy=True).astype(np.float32, copy=False),
        "validate": np.array(target_validate_raw[:n_validate], copy=True).astype(np.float32, copy=False),
        "test": np.array(target_test_raw[:n_test], copy=True).astype(np.float32, copy=False),
    }
    targets_scale = preprocess_targets(targets, params, logger)
    return x_train, x_validate, x_test, targets["train"], targets["validate"], targets["test"], targets_scale


def build_estimator(baseline: str):
    if baseline == "ridge_svd128_alpha1":
        svd = TruncatedSVD(n_components=128, random_state=0)
        model = Ridge(alpha=1.0, random_state=0)
        return ("svd_ridge", svd, model)
    if baseline.startswith("knn_k"):
        n_neighbors = int(baseline.split("_k", 1)[1])
        return ("knn", None, KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", metric="euclidean", n_jobs=-1))
    if baseline.startswith("extra_trees_"):
        parts = baseline.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unsupported ExtraTrees baseline='{baseline}'")
        n_estimators = int(parts[2])
        min_samples_leaf = 1
        max_depth = None
        max_features = 1.0
        for part in parts[3:]:
            if part.startswith("leaf"):
                min_samples_leaf = int(part[4:])
            elif part.startswith("depth"):
                max_depth = int(part[5:])
            elif part == "sqrt":
                max_features = "sqrt"
            else:
                raise ValueError(f"Unsupported ExtraTrees modifier '{part}' in baseline='{baseline}'")
        return (
            "extra_trees",
            None,
            ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=0,
                n_jobs=-1,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_features=max_features,
            ),
        )
    if baseline.startswith("random_forest_"):
        parts = baseline.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unsupported RandomForest baseline='{baseline}'")
        n_estimators = int(parts[2])
        min_samples_leaf = 1
        max_depth = None
        max_features = 1.0
        for part in parts[3:]:
            if part.startswith("leaf"):
                min_samples_leaf = int(part[4:])
            elif part.startswith("depth"):
                max_depth = int(part[5:])
            elif part == "sqrt":
                max_features = "sqrt"
            else:
                raise ValueError(f"Unsupported RandomForest modifier '{part}' in baseline='{baseline}'")
        return (
            "random_forest",
            None,
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=0,
                n_jobs=-1,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_features=max_features,
            ),
        )
    if baseline == "hist_gbrt_300_depth6":
        base = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            max_depth=6,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        return ("hist_gbrt", None, MultiOutputRegressor(base, n_jobs=-1))
    if baseline == "svd128_mlp_512_256":
        reducer = TruncatedSVD(n_components=128, random_state=0)
        model = MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=1.0e-4,
            learning_rate_init=1.0e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=0,
        )
        return ("svd_mlp", reducer, model)
    if baseline == "svd128_svr_rbf":
        reducer = TruncatedSVD(n_components=128, random_state=0)
        model = MultiOutputRegressor(
            SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"),
            n_jobs=-1,
        )
        return ("svd_svr", reducer, model)
    raise ValueError(f"Unsupported baseline={baseline}")


def fit_predict_simple_baseline(
    baseline: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    _family, reducer, estimator = build_estimator(baseline)
    if reducer is not None:
        x_train_fit = reducer.fit_transform(x_train)
        x_eval_fit = reducer.transform(x_eval)
    else:
        x_train_fit = x_train
        x_eval_fit = x_eval
    estimator.fit(x_train_fit, y_train)
    return estimator.predict(x_eval_fit)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hh_classical_baseline")

    params = load_params(args.params)
    if args.data_dir is not None:
        params["data"]["data_dir"] = args.data_dir
    if args.data_prefix is not None:
        params["data"]["data_prefix"] = args.data_prefix
    if args.curr is not None:
        params["data"]["curr"] = args.curr
    params = resolve_local_data_dir(params, logger)

    params["data"]["split_array_cache_enabled"] = True
    params["data"]["features_scale_cache_enabled"] = True
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False

    t0 = time.time()
    feature_mode = args.feature_mode
    try:
        x_train, x_validate, x_test, y_train, y_validate, y_test, targets_scale = load_cached_splits(params, args, logger)
    except Exception as exc:
        logger.warning("Split-array cache path unavailable (%s); falling back to load_data", exc)
        params["data"]["split_array_cache_enabled"] = False
        params["data"]["features_scale_cache_enabled"] = False
        features, targets, _, _ = load_data(params, logger)
        raw_test = np.array(features["test"], copy=True)
        features_scale = preprocess_features(features, params, logger)
        targets_scale = preprocess_targets(targets, params, logger)

        sub_length = params["data"].get("features_sub_length")
        sub_step = int(params["data"].get("features_sub_step", 0) or 0)
        noise_std = float(params["data"].get("features_additive_noise_std", 0.0) or 0.0)
        multiplicative_noise_std = float(params["data"].get("features_multiplicative_noise_std", 0.0) or 0.0)
        baseline_drift_std = float(params["data"].get("features_baseline_drift_std", 0.0) or 0.0)
        mask_fraction = float(params["data"].get("features_mask_fraction", 0.0) or 0.0)
        x_train = materialize_features(features["train"], sub_length, feature_mode, sub_step=sub_step)
        if noise_std > 0.0:
            x_test = normalize_test_features(
                raw_test,
                features_scale,
                sub_length,
                sub_step=sub_step,
                noise_std=noise_std,
                multiplicative_noise_std=multiplicative_noise_std,
                baseline_drift_std=baseline_drift_std,
                mask_fraction=mask_fraction,
                seed=int(params["data"]["random_seed"]),
                feature_mode=feature_mode,
            )
        else:
            shifted_test = apply_eval_shift(
                features["test"],
                additive_noise_std=0.0,
                multiplicative_noise_std=multiplicative_noise_std,
                baseline_drift_std=baseline_drift_std,
                mask_fraction=mask_fraction,
                seed=int(params["data"]["random_seed"]),
            )
            x_test = materialize_features(shifted_test, sub_length, feature_mode, sub_step=sub_step)
        y_train = np.asarray(targets["train"])
        y_validate = np.asarray(targets["validate"])
        y_test = np.asarray(targets["test"])
        x_validate = materialize_features(features["validate"], sub_length, feature_mode, sub_step=sub_step)

    sub_length = params["data"].get("features_sub_length")
    sub_step = int(params["data"].get("features_sub_step", 0) or 0)
    noise_std = float(params["data"].get("features_additive_noise_std", 0.0) or 0.0)
    multiplicative_noise_std = float(params["data"].get("features_multiplicative_noise_std", 0.0) or 0.0)
    baseline_drift_std = float(params["data"].get("features_baseline_drift_std", 0.0) or 0.0)
    mask_fraction = float(params["data"].get("features_mask_fraction", 0.0) or 0.0)

    if args.baseline == "blend_et500_rf500_knn11_mean":
        base_names = ["extra_trees_500", "random_forest_500", "knn_k11"]
        logger.info("baseline=%s family=blend_mean x_train=%s x_test=%s", args.baseline, x_train.shape, x_test.shape)
        preds = [fit_predict_simple_baseline(name, x_train, y_train, x_test) for name in base_names]
        y_pred = np.mean(preds, axis=0)
    elif args.baseline == "stack_et500_rf500_knn11_ridge":
        base_names = ["extra_trees_500", "random_forest_500", "knn_k11"]
        logger.info("baseline=%s family=stack_ridge x_train=%s x_validate=%s x_test=%s", args.baseline, x_train.shape, x_validate.shape, x_test.shape)
        validate_preds = []
        test_preds = []
        for name in base_names:
            validate_preds.append(fit_predict_simple_baseline(name, x_train, y_train, x_validate))
            test_preds.append(fit_predict_simple_baseline(name, x_train, y_train, x_test))
        meta_x_validate = np.concatenate(validate_preds, axis=1)
        meta_x_test = np.concatenate(test_preds, axis=1)
        meta_model = Ridge(alpha=1.0, random_state=0)
        meta_model.fit(meta_x_validate, y_validate)
        y_pred = meta_model.predict(meta_x_test)
    else:
        family, reducer, estimator = build_estimator(args.baseline)
        logger.info("baseline=%s family=%s x_train=%s x_test=%s", args.baseline, family, x_train.shape, x_test.shape)

        if reducer is not None:
            x_train_fit = reducer.fit_transform(x_train)
            x_test_fit = reducer.transform(x_test)
        else:
            x_train_fit = x_train
            x_test_fit = x_test

        estimator.fit(x_train_fit, y_train)
        y_pred = estimator.predict(x_test_fit)

    y_pred_post = np.array(y_pred, copy=True)
    y_test_post = np.array(y_test, copy=True)
    y_pred_post = _apply_scale_inverse(y_pred_post, targets_scale)
    y_test_post = _apply_scale_inverse(y_test_post, targets_scale)
    transform_cfg = targets_scale.get("transform") if isinstance(targets_scale, dict) else None
    y_pred_post = _apply_targets_transform(y_pred_post, transform_cfg, inverse=True, array_name="targets")
    y_test_post = _apply_targets_transform(y_test_post, transform_cfg, inverse=True, array_name="targets")

    mse = float(mean_squared_error(y_test_post, y_pred_post))
    mae = float(mean_absolute_error(y_test_post, y_pred_post))
    r2 = float(r2_score(y_test_post, y_pred_post))

    per_target = []
    for idx in range(y_test_post.shape[1]):
        per_target.append(
            {
                "target_index": idx,
                "mse": float(mean_squared_error(y_test_post[:, idx], y_pred_post[:, idx])),
                "mae": float(mean_absolute_error(y_test_post[:, idx], y_pred_post[:, idx])),
                "r2": float(r2_score(y_test_post[:, idx], y_pred_post[:, idx])),
            }
        )

    runtime_sec = float(time.time() - t0)
    metrics_summary = {
        "mse": {"test": mse},
        "mae": {"test": mae},
        "r2": {"test": r2},
        "metadata": {
            "baseline": args.baseline,
            "feature_mode": feature_mode,
            "runtime_sec": runtime_sec,
            "features_sub_length": int(sub_length) if sub_length else None,
            "feature_dim": int(x_train.shape[1]),
            "features_additive_noise_std": noise_std,
            "features_multiplicative_noise_std": multiplicative_noise_std,
            "features_baseline_drift_std": baseline_drift_std,
            "features_mask_fraction": mask_fraction,
            "features_sub_step": sub_step if sub_step > 0 else None,
            "random_seed": int(params["data"]["random_seed"]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
        },
    }

    (save_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    (save_dir / "metrics_per_target.json").write_text(json.dumps(per_target, indent=2) + "\n")
    np.savez_compressed(save_dir / "predictions.npz", targets=y_test_post, predictions=y_pred_post)
    (save_dir / "params_snapshot.yaml").write_text(yaml.safe_dump(params, sort_keys=False))

    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
