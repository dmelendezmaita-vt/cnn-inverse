#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

def _ensure_tensorflow_io_stub() -> None:
    try:
        import tensorflow as tf  # type: ignore
        gfile = getattr(getattr(tf, "io", None), "gfile", None)
        if hasattr(gfile, "join") and hasattr(gfile, "makedirs") and hasattr(gfile, "GFile"):
            return
    except Exception:
        pass
    tf_stub = types.ModuleType("tensorflow")
    def _gfile_join(*parts):
        cleaned = []
        for idx, part in enumerate(parts):
            text = str(part)
            if idx == 0:
                cleaned.append(text.rstrip("/"))
            else:
                cleaned.append(text.strip("/"))
        return "/".join(part for part in cleaned if part != "")

    def _gfile_makedirs(path):
        os.makedirs(path, exist_ok=True)

    def _gfile_open(path, mode="rb"):
        return open(path, mode)

    gfile_stub = types.SimpleNamespace(join=_gfile_join, makedirs=_gfile_makedirs, GFile=_gfile_open)
    tf_stub.io = types.SimpleNamespace(gfile=gfile_stub)
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = tf_stub

_ensure_tensorflow_io_stub()

from sbi.inference import FMPE, NPSE, SNLE, SNPE, SNRE
from sbi.neural_nets.factory import classifier_nn, likelihood_nn, posterior_flow_nn, posterior_nn, posterior_score_nn
from sbi.utils import BoxUniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "src"))

from data import (  # noqa: E402
    _apply_scale_inverse,
    _apply_targets_transform,
    _resolve_split_array_cache_dir,
    load_data,
    preprocess_targets,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the first HH SBI baseline block for Checkpoint 2.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--method", default="snpe", choices=["snpe", "snle", "snre", "fmpe", "npse"])
    ap.add_argument("--density-estimator", default="maf")
    ap.add_argument("--sample-with", default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-prefix", default=None)
    ap.add_argument("--curr", default=None)
    ap.add_argument("--device", default="auto")
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--embedding-hidden", type=int, default=256)
    ap.add_argument("--hidden-features", type=int, default=128)
    ap.add_argument("--num-transforms", type=int, default=5)
    ap.add_argument("--training-batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=5.0e-4)
    ap.add_argument("--stop-after-epochs", type=int, default=15)
    ap.add_argument("--max-num-epochs", type=int, default=80)
    ap.add_argument("--clip-max-norm", type=float, default=5.0)
    ap.add_argument("--train-additive-noise-std", type=float, default=0.0)
    ap.add_argument("--train-multiplicative-noise-std", type=float, default=0.0)
    ap.add_argument("--train-baseline-drift-std", type=float, default=0.0)
    ap.add_argument("--train-mask-fraction", type=float, default=0.0)
    ap.add_argument("--eval-additive-noise-std", type=float, default=0.0)
    ap.add_argument("--eval-multiplicative-noise-std", type=float, default=0.0)
    ap.add_argument("--eval-baseline-drift-std", type=float, default=0.0)
    ap.add_argument("--eval-mask-fraction", type=float, default=0.0)
    ap.add_argument("--posterior-samples", type=int, default=256)
    ap.add_argument("--compute-map", action="store_true")
    ap.add_argument("--map-num-iter", type=int, default=100)
    ap.add_argument("--map-num-to-optimize", type=int, default=50)
    ap.add_argument("--map-learning-rate", type=float, default=5.0e-2)
    ap.add_argument("--map-num-init-samples", type=int, default=500)
    ap.add_argument(
        "--decision-rule",
        default="mean",
        choices=["mean", "median", "map", "validate_calibrated"],
    )
    ap.add_argument(
        "--selection-split",
        default="validate",
        choices=["validate"],
    )
    ap.add_argument("--selection-cov-lambda", type=float, default=1.0)
    ap.add_argument("--selection-candidate-rules", default="mean,median")
    return ap.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def hh_raw_prior_bounds(n_targets: int) -> tuple[np.ndarray, np.ndarray] | None:
    if n_targets != 6:
        return None
    # The current six-target Hodgkin-Huxley setup reuses the same three
    # conductance ranges for the second current condition.
    low = np.asarray([0.05, 0.2, 5.0, 0.05, 0.2, 5.0], dtype=np.float32)
    high = np.asarray([10000.0, 100.0, 1000.0, 10000.0, 100.0, 1000.0], dtype=np.float32)
    return low, high


def materialize_features(arr: np.ndarray, count: int, sub_length: int | None) -> np.ndarray:
    block = np.array(arr[:count], copy=True)
    if sub_length and sub_length < block.shape[-1]:
        block = block[..., :sub_length]
    return block.reshape(block.shape[0], -1).astype(np.float32, copy=False)


def crop_trace_window(arr: np.ndarray, count: int, sub_length: int | None, sub_step: int | None = None) -> np.ndarray:
    block = np.array(arr[:count], copy=True)
    if sub_length and sub_length < block.shape[-1]:
        block = block[..., :sub_length]
    if sub_step and sub_step > 1:
        block = block[..., ::sub_step]
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


def compute_summary12(arr2d: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr2d, dtype=np.float32)
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


def compute_fft256(arr2d: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr2d, dtype=np.float32)
    centered = traces - np.mean(traces, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    magnitudes = np.log1p(np.abs(spectrum[:, 1:257]))
    return magnitudes.astype(np.float32, copy=False)


def materialize_feature_mode(flat_block: np.ndarray, feature_mode: str) -> np.ndarray:
    raw = np.asarray(flat_block, dtype=np.float32, copy=False)
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


def normalize_features(
    x_train: np.ndarray,
    x_validate: np.ndarray,
    x_test: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shift = np.zeros((1, x_train.shape[1]), dtype=np.float32)
    mult = np.ones((1, x_train.shape[1]), dtype=np.float32)
    if enabled:
        shift = np.mean(x_train, axis=0, keepdims=True).astype(np.float32, copy=False)
        mult = np.std(x_train, axis=0, keepdims=True).astype(np.float32, copy=False)
        mult = np.where(mult == 0.0, 1.0, mult)
        x_train = (x_train - shift) / mult
        x_validate = (x_validate - shift) / mult
        x_test = (x_test - shift) / mult
    return x_train, x_validate, x_test, {"shift": shift, "mult": mult}


def load_cached_splits(params: dict, args: argparse.Namespace, logger: logging.Logger) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, np.ndarray, np.ndarray]:
    params = json.loads(json.dumps(params))
    if args.data_dir is not None:
        params["data"]["data_dir"] = args.data_dir
    if args.data_prefix is not None:
        params["data"]["data_prefix"] = args.data_prefix
    if args.curr is not None:
        params["data"]["curr"] = args.curr

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

    cached_train = int(feature_train_raw.shape[0])
    cached_validate = int(feature_validate_raw.shape[0])
    cached_test = int(feature_test_raw.shape[0])
    n_train = int(args.n_train or params["data"]["Ntrain"])
    n_validate = int(params["data"].get("Nvalidate", cached_validate) or 0)
    n_test = int(args.eval_limit or params["data"].get("Ntest", cached_test) or cached_test)

    if n_train > cached_train:
        raise SystemExit(f"Requested n_train={n_train} exceeds cached train size {cached_train}")
    if n_validate > cached_validate:
        raise SystemExit(f"Requested n_validate={n_validate} exceeds cached validate size {cached_validate}")
    if n_test > cached_test:
        raise SystemExit(f"Requested n_test={n_test} exceeds cached test size {cached_test}")

    sub_length = params["data"].get("features_sub_length")
    sub_step = int(params["data"].get("features_sub_step", 0) or 0)
    base_seed = int(args.seed if args.seed is not None else params["data"].get("random_seed", 0))
    train_additive_noise_std = float(args.train_additive_noise_std)
    train_multiplicative_noise_std = float(args.train_multiplicative_noise_std)
    train_baseline_drift_std = float(args.train_baseline_drift_std)
    train_mask_fraction = float(args.train_mask_fraction)
    eval_additive_noise_std = float(args.eval_additive_noise_std)
    eval_multiplicative_noise_std = float(args.eval_multiplicative_noise_std)
    eval_baseline_drift_std = float(args.eval_baseline_drift_std)
    eval_mask_fraction = float(args.eval_mask_fraction)
    train_window = crop_trace_window(feature_train_raw, n_train, sub_length, sub_step=sub_step)
    validate_window = crop_trace_window(feature_validate_raw, n_validate, sub_length, sub_step=sub_step)
    test_window = crop_trace_window(feature_test_raw, n_test, sub_length, sub_step=sub_step)
    x_train_raw = train_window.reshape(train_window.shape[0], -1).astype(np.float32, copy=False)
    x_validate_raw = validate_window.reshape(validate_window.shape[0], -1).astype(np.float32, copy=False)
    x_test_raw = test_window.reshape(test_window.shape[0], -1).astype(np.float32, copy=False)
    x_train_raw, x_validate_raw, x_test_raw, feature_scale = normalize_features(
        x_train_raw,
        x_validate_raw,
        x_test_raw,
        enabled=bool(params["data"].get("features_normalize", False)),
    )
    x_train_raw = apply_eval_shift(
        x_train_raw.reshape(train_window.shape[0], train_window.shape[1], -1),
        additive_noise_std=train_additive_noise_std,
        multiplicative_noise_std=train_multiplicative_noise_std,
        baseline_drift_std=train_baseline_drift_std,
        mask_fraction=train_mask_fraction,
        seed=base_seed + 11,
    ).reshape(x_train_raw.shape[0], -1)
    x_validate_raw = apply_eval_shift(
        x_validate_raw.reshape(validate_window.shape[0], validate_window.shape[1], -1),
        additive_noise_std=eval_additive_noise_std,
        multiplicative_noise_std=eval_multiplicative_noise_std,
        baseline_drift_std=eval_baseline_drift_std,
        mask_fraction=eval_mask_fraction,
        seed=base_seed + 101,
    ).reshape(x_validate_raw.shape[0], -1)
    x_test_raw = apply_eval_shift(
        x_test_raw.reshape(test_window.shape[0], test_window.shape[1], -1),
        additive_noise_std=eval_additive_noise_std,
        multiplicative_noise_std=eval_multiplicative_noise_std,
        baseline_drift_std=eval_baseline_drift_std,
        mask_fraction=eval_mask_fraction,
        seed=base_seed + 202,
    ).reshape(x_test_raw.shape[0], -1)
    x_train = materialize_feature_mode(x_train_raw, args.feature_mode)
    x_validate = materialize_feature_mode(x_validate_raw, args.feature_mode)
    x_test = materialize_feature_mode(x_test_raw, args.feature_mode)

    targets = {
        "train": np.array(target_train_raw[:n_train], copy=True).astype(np.float32, copy=False),
        "validate": np.array(target_validate_raw[:n_validate], copy=True).astype(np.float32, copy=False),
        "test": np.array(target_test_raw[:n_test], copy=True).astype(np.float32, copy=False),
    }
    targets_scale = preprocess_targets(targets, params, logger)

    raw_bounds = hh_raw_prior_bounds(targets["train"].shape[1])
    if raw_bounds is not None:
        raw_low, raw_high = raw_bounds
        bounds = np.stack([raw_low, raw_high], axis=0).astype(np.float32, copy=False)
        bounds = _apply_targets_transform(bounds, targets_scale.get("transform"), inverse=False, array_name="hh_prior_bounds")
        prior_low = ((bounds[0:1] - targets_scale["shift"]) / targets_scale["mult"]).reshape(-1).astype(np.float32, copy=False)
        prior_high = ((bounds[1:2] - targets_scale["shift"]) / targets_scale["mult"]).reshape(-1).astype(np.float32, copy=False)
    else:
        all_targets = np.concatenate([targets["train"], targets["validate"], targets["test"]], axis=0)
        prior_low = np.min(all_targets, axis=0).astype(np.float32, copy=False)
        prior_high = np.max(all_targets, axis=0).astype(np.float32, copy=False)
        span = np.maximum(prior_high - prior_low, 1.0e-6).astype(np.float32, copy=False)
        prior_low = prior_low - 1.0e-4 * span
        prior_high = prior_high + 1.0e-4 * span

    features = {"train": x_train, "validate": x_validate, "test": x_test}
    logger.info(
        "Prepared HH SBI splits: feature_mode=%s x_train=%s x_validate=%s x_test=%s y_train=%s",
        args.feature_mode,
        x_train.shape,
        x_validate.shape,
        x_test.shape,
        targets["train"].shape,
    )
    return features, targets, targets_scale, prior_low, prior_high


def make_embedding_net(input_dim: int, embedding_hidden: int, embedding_dim: int) -> torch.nn.Module:
    if embedding_dim <= 0:
        return torch.nn.Identity()
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, embedding_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(embedding_hidden, embedding_dim),
        torch.nn.GELU(),
    )


def build_sbi_components(args: argparse.Namespace, feature_dim: int, theta_dim: int, prior, device: str):
    embedding_net = make_embedding_net(feature_dim, args.embedding_hidden, args.embedding_dim)
    model_kind = args.density_estimator

    if args.method == "snpe":
        builder = posterior_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            embedding_net=embedding_net,
        )
        inference = SNPE(
            prior=prior,
            density_estimator=builder,
            device=device,
            logging_level="INFO",
            show_progress_bars=False,
        )
        sample_with = args.sample_with or "direct"
        return inference, sample_with

    if args.method == "snle":
        # SNLE models p(x|theta); in sbi's likelihood builders the embedding net is
        # applied to the conditioning variable theta, not the 2000-step observation.
        builder = likelihood_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            embedding_net=torch.nn.Identity(),
        )
        inference = SNLE(
            prior=prior,
            density_estimator=builder,
            device=device,
            logging_level="INFO",
            show_progress_bars=False,
        )
        sample_with = args.sample_with or "mcmc"
        return inference, sample_with

    if args.method == "snre":
        builder = classifier_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            embedding_net_x=embedding_net,
        )
        inference = SNRE(
            prior=prior,
            classifier=builder,
            device=device,
            logging_level="INFO",
            show_progress_bars=False,
        )
        sample_with = args.sample_with or "mcmc"
        return inference, sample_with

    if args.method == "fmpe":
        builder = posterior_flow_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_layers=args.num_transforms,
            embedding_net=embedding_net,
        )
        inference = FMPE(
            prior=prior,
            density_estimator=builder,
            device=device,
            logging_level="INFO",
            show_progress_bars=False,
        )
        sample_with = args.sample_with or "ode"
        return inference, sample_with

    if args.method == "npse":
        builder = posterior_score_nn(
            model=model_kind,
            sde_type="ve",
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_layers=args.num_transforms,
            embedding_net=embedding_net,
        )
        inference = NPSE(
            prior=prior,
            score_estimator=builder,
            sde_type="ve",
            device=device,
            logging_level="INFO",
            show_progress_bars=False,
        )
        sample_with = args.sample_with or "sde"
        return inference, sample_with

    raise ValueError(f"Unsupported method={args.method}")


def invert_target_array(arr: np.ndarray, targets_scale: dict) -> np.ndarray:
    orig_shape = arr.shape
    flat = np.array(arr, copy=True).reshape(-1, orig_shape[-1])
    flat = _apply_scale_inverse(flat, targets_scale)
    flat = _apply_targets_transform(flat, targets_scale.get("transform"), inverse=True, array_name="targets")
    return flat.reshape(orig_shape)


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_coverage(y_true: np.ndarray, samples: np.ndarray, level: float) -> dict[str, object]:
    alpha = (1.0 - level) / 2.0
    lower = np.quantile(samples, alpha, axis=1)
    upper = np.quantile(samples, 1.0 - alpha, axis=1)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "level": level,
        "overall": float(np.mean(covered)),
        "per_target": [float(x) for x in np.mean(covered, axis=0)],
    }


def parse_candidate_rules(raw: str) -> list[str]:
    rules = [tok.strip() for tok in raw.split(",") if tok.strip()]
    allowed = {"mean", "median", "map"}
    if not rules:
        raise ValueError("selection_candidate_rules must contain at least one candidate rule.")
    unknown = [rule for rule in rules if rule not in allowed]
    if unknown:
        raise ValueError(f"Unsupported selection candidate rules: {unknown}")
    return rules


def compute_coverage_bundle(y_true: np.ndarray, samples: np.ndarray, levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict[str, dict[str, object]]:
    return {
        str(level): compute_coverage(y_true, samples, level)
        for level in levels
    }


def mean_coverage_gap(coverage_bundle: dict[str, dict[str, object]]) -> float:
    gaps = []
    for level_key, payload in coverage_bundle.items():
        level = float(level_key)
        gaps.append(abs(float(payload["overall"]) - level))
    return float(np.mean(gaps))


def select_point_prediction(
    rule: str,
    *,
    samples: np.ndarray,
    posterior_map: np.ndarray | None = None,
) -> np.ndarray:
    if rule == "mean":
        return np.mean(samples, axis=1)
    if rule == "median":
        return np.median(samples, axis=1)
    if rule == "map":
        if posterior_map is None:
            raise ValueError("Requested MAP decision rule without posterior_map.")
        return posterior_map
    raise ValueError(f"Unsupported decision rule: {rule}")


def posterior_samples_for_test(
    posterior,
    x_test: np.ndarray,
    num_samples: int,
    device: str,
    logger: logging.Logger,
) -> np.ndarray:
    out = []
    total = int(x_test.shape[0])
    with torch.inference_mode():
        for idx in range(total):
            if idx == 0 or (idx + 1) % 64 == 0 or (idx + 1) == total:
                logger.info("Sampling posterior for test observation %d/%d", idx + 1, total)
            x_i = torch.as_tensor(x_test[idx], dtype=torch.float32, device=device)
            draws = posterior.sample((num_samples,), x=x_i, show_progress_bars=False)
            out.append(draws.detach().cpu().numpy())
    return np.stack(out, axis=0)


def posterior_map_for_test(
    posterior,
    x_test: np.ndarray,
    device: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> np.ndarray:
    out = []
    total = int(x_test.shape[0])
    for idx in range(total):
        if idx == 0 or (idx + 1) % 64 == 0 or (idx + 1) == total:
            logger.info("Computing posterior MAP for test observation %d/%d", idx + 1, total)
        x_i = torch.as_tensor(x_test[idx], dtype=torch.float32, device=device)
        posterior_i = posterior.set_default_x(x_i)
        value = posterior_i.map(
            num_iter=args.map_num_iter,
            num_to_optimize=args.map_num_to_optimize,
            learning_rate=args.map_learning_rate,
            num_init_samples=args.map_num_init_samples,
            show_progress_bars=False,
        )
        value_np = np.asarray(value.detach().cpu().numpy()).reshape(-1)
        out.append(value_np)
    return np.stack(out, axis=0)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hh_sbi_baseline")

    params = load_params(args.params)
    seed = int(args.seed if args.seed is not None else params["data"].get("random_seed", 0))
    seed_everything(seed)

    device = select_device(args.device)
    logger.info("Running HH SBI baseline with method=%s density_estimator=%s device=%s seed=%d", args.method, args.density_estimator, device, seed)
    track_cuda_memory = device.startswith("cuda") and torch.cuda.is_available()
    if track_cuda_memory:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    if "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))

    t0 = time.time()
    features, targets, targets_scale, prior_low, prior_high = load_cached_splits(params, args, logger)

    x_train = torch.as_tensor(features["train"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(targets["train"], dtype=torch.float32, device=device)
    x_validate = features["validate"]
    y_validate_norm = targets["validate"]
    x_test = features["test"]
    y_test_norm = targets["test"]

    prior = BoxUniform(
        low=torch.as_tensor(prior_low, dtype=torch.float32),
        high=torch.as_tensor(prior_high, dtype=torch.float32),
        device=device,
    )
    inference, sample_with = build_sbi_components(args, x_train.shape[1], y_train.shape[1], prior, device)
    inference.append_simulations(y_train, x_train, data_device=device)

    t_train_start = time.time()
    estimator = inference.train(
        training_batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=0.1,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_num_epochs,
        clip_max_norm=args.clip_max_norm,
        show_train_summary=True,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    training_runtime_sec = time.time() - t_train_start

    posterior = inference.build_posterior(estimator, sample_with=sample_with)
    validation_samples_norm = None
    validation_coverage = None
    validation_rule_scores = None
    selected_rule = args.decision_rule
    selected_rule_source = "cli"
    selected_rule_score = None
    selected_metrics = None
    selection_candidate_rules = parse_candidate_rules(args.selection_candidate_rules)
    validation_map_norm = None
    validation_map = None
    if args.decision_rule == "map" and not args.compute_map:
        raise SystemExit("decision_rule=map requires --compute-map.")
    if "map" in selection_candidate_rules and not args.compute_map:
        raise SystemExit("selection_candidate_rules includes map but --compute-map is disabled.")

    t_eval_start = time.time()
    if args.decision_rule == "validate_calibrated":
        validation_samples_norm = posterior_samples_for_test(
            posterior,
            x_test=x_validate,
            num_samples=args.posterior_samples,
            device=device,
            logger=logger,
        )
        validation_coverage = compute_coverage_bundle(y_validate_norm, validation_samples_norm)
        coverage_gap = mean_coverage_gap(validation_coverage)
        if "map" in selection_candidate_rules:
            validation_map_norm = posterior_map_for_test(
                posterior,
                x_test=x_validate,
                device=device,
                args=args,
                logger=logger,
            )
            validation_map = invert_target_array(validation_map_norm, targets_scale)

        validation_rule_scores = {}
        for rule in selection_candidate_rules:
            pred_norm = select_point_prediction(rule, samples=validation_samples_norm, posterior_map=validation_map_norm)
            mae_norm = float(np.mean(np.abs(y_validate_norm - pred_norm)))
            score = float(mae_norm + args.selection_cov_lambda * coverage_gap)
            validation_rule_scores[rule] = {
                "normalized_mae": mae_norm,
                "mean_coverage_gap": coverage_gap,
                "score": score,
            }
        selected_rule = min(
            selection_candidate_rules,
            key=lambda rule: (
                validation_rule_scores[rule]["score"],
                validation_rule_scores[rule]["normalized_mae"],
                rule,
            ),
        )
        selected_rule_source = args.selection_split
        selected_rule_score = float(validation_rule_scores[selected_rule]["score"])

    samples_norm = posterior_samples_for_test(
        posterior,
        x_test=x_test,
        num_samples=args.posterior_samples,
        device=device,
        logger=logger,
    )
    evaluation_runtime_sec = time.time() - t_eval_start

    y_test = invert_target_array(y_test_norm, targets_scale)
    samples = invert_target_array(samples_norm, targets_scale)
    posterior_mean = np.mean(samples, axis=1)
    posterior_median = np.median(samples, axis=1)
    posterior_std = np.std(samples, axis=1)
    posterior_map = None
    map_runtime_sec = 0.0
    map_metrics = None
    if args.compute_map:
        t_map_start = time.time()
        posterior_map_norm = posterior_map_for_test(
            posterior,
            x_test=x_test,
            device=device,
            args=args,
            logger=logger,
        )
        map_runtime_sec = time.time() - t_map_start
        posterior_map = invert_target_array(posterior_map_norm, targets_scale)
        map_metrics = compute_point_metrics(y_test, posterior_map)

    mean_metrics = compute_point_metrics(y_test, posterior_mean)
    median_metrics = compute_point_metrics(y_test, posterior_median)
    coverage = compute_coverage_bundle(y_test, samples)

    if selected_rule == "mean":
        selected_metrics = mean_metrics
        selected_prediction = posterior_mean
    elif selected_rule == "median":
        selected_metrics = median_metrics
        selected_prediction = posterior_median
    elif selected_rule == "map":
        if posterior_map is None or map_metrics is None:
            raise SystemExit("Selected MAP decision rule without MAP predictions.")
        selected_metrics = map_metrics
        selected_prediction = posterior_map
    else:
        raise SystemExit(f"Unsupported selected_rule={selected_rule}")

    gpu_memory = None
    if track_cuda_memory:
        torch.cuda.synchronize()
        device_idx = torch.cuda.current_device()
        gpu_memory = {
            "device_index": int(device_idx),
            "device_name": torch.cuda.get_device_name(device_idx),
            "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device_idx) / (1024.0 * 1024.0)),
            "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device_idx) / (1024.0 * 1024.0)),
            "memory_allocated_mb": float(torch.cuda.memory_allocated(device_idx) / (1024.0 * 1024.0)),
            "memory_reserved_mb": float(torch.cuda.memory_reserved(device_idx) / (1024.0 * 1024.0)),
        }

    metrics_summary = {
        "method": args.method,
        "density_estimator": args.density_estimator,
        "device": device,
        "feature_mode": args.feature_mode,
        "sample_with": sample_with,
        "decision_rule": args.decision_rule,
        "selection_split": args.selection_split,
        "selection_cov_lambda": float(args.selection_cov_lambda),
        "selection_candidate_rules": selection_candidate_rules,
        "selected_rule": selected_rule,
        "selected_rule_source": selected_rule_source,
        "selected_rule_score": selected_rule_score,
        "seed": seed,
        "n_train": int(features["train"].shape[0]),
        "n_validate": int(features["validate"].shape[0]),
        "n_test": int(features["test"].shape[0]),
        "feature_dim": int(features["train"].shape[1]),
        "embedding_dim": int(args.embedding_dim),
        "embedding_hidden": int(args.embedding_hidden),
        "hidden_features": int(args.hidden_features),
        "num_transforms": int(args.num_transforms),
        "training_batch_size": int(args.training_batch_size),
        "learning_rate": float(args.learning_rate),
        "stop_after_epochs": int(args.stop_after_epochs),
        "max_num_epochs": int(args.max_num_epochs),
        "train_additive_noise_std": float(args.train_additive_noise_std),
        "train_multiplicative_noise_std": float(args.train_multiplicative_noise_std),
        "train_baseline_drift_std": float(args.train_baseline_drift_std),
        "train_mask_fraction": float(args.train_mask_fraction),
        "eval_additive_noise_std": float(args.eval_additive_noise_std),
        "eval_multiplicative_noise_std": float(args.eval_multiplicative_noise_std),
        "eval_baseline_drift_std": float(args.eval_baseline_drift_std),
        "eval_mask_fraction": float(args.eval_mask_fraction),
        "posterior_samples": int(args.posterior_samples),
        "training_runtime_sec": float(training_runtime_sec),
        "evaluation_runtime_sec": float(evaluation_runtime_sec),
        "map_runtime_sec": float(map_runtime_sec),
        "total_runtime_sec": float(time.time() - t0),
        "prior_low": [float(x) for x in prior_low],
        "prior_high": [float(x) for x in prior_high],
        "posterior_mean_metrics": mean_metrics,
        "posterior_median_metrics": median_metrics,
        "posterior_map_metrics": map_metrics,
        "posterior_selected_metrics": selected_metrics,
        "posterior_concentration": {
            "overall_mean_std": float(np.mean(posterior_std)),
            "per_target_mean_std": [float(x) for x in np.mean(posterior_std, axis=0)],
        },
        "marginal_coverage": coverage,
        "validation_rule_scores": validation_rule_scores,
        "validation_marginal_coverage": validation_coverage,
        "gpu_memory": gpu_memory,
    }

    (save_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    (save_dir / "params_snapshot.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    payload = {
        "targets": y_test.astype(np.float32, copy=False),
        "posterior_mean": posterior_mean.astype(np.float32, copy=False),
        "posterior_median": posterior_median.astype(np.float32, copy=False),
        "posterior_std": posterior_std.astype(np.float32, copy=False),
        "posterior_samples": samples.astype(np.float32, copy=False),
        "posterior_selected": selected_prediction.astype(np.float32, copy=False),
    }
    if posterior_map is not None:
        payload["posterior_map"] = posterior_map.astype(np.float32, copy=False)
    with (save_dir / "predictions_test.npz").open("wb") as f:
        np.savez(f, **payload)

    logger.info("Wrote metrics summary to %s", save_dir / "metrics_summary.json")
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
