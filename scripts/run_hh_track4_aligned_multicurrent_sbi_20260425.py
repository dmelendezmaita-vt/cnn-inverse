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

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
DEFAULT_SHARED_DATA_DIR = str(
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)
sys.path.append(str(REPO / "pytorch"))

from data import _apply_scale_inverse, _apply_targets_transform, _resolve_split_array_cache_dir, preprocess_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aligned multi-current SBI baselines for HH Track4.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--method", required=True, choices=["snpe", "fmpe", "npse", "snre", "snle"])
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
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--density-estimator", default="maf")
    ap.add_argument("--training-batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=5.0e-4)
    ap.add_argument("--stop-after-epochs", type=int, default=20)
    ap.add_argument("--max-num-epochs", type=int, default=100)
    ap.add_argument("--embedding-hidden", type=int, default=256)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--hidden-features", type=int, default=128)
    ap.add_argument("--num-transforms", type=int, default=6)
    ap.add_argument("--posterior-samples", type=int, default=16)
    ap.add_argument("--compute-map", action="store_true")
    ap.add_argument("--sample-with")
    ap.add_argument("--decision-rule", default="mean", choices=["mean", "median"])
    return ap.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA but no CUDA device is available.")
    return requested


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
    params["data"]["data_dir"] = data_dir
    params["data"]["data_prefix"] = data_prefix
    params["data"]["curr"] = curr
    params["data"]["Ntrain"] = int(n_train)
    params["data"]["Nvalidate"] = int(n_validate)
    params["data"]["Ntest"] = int(n_test)
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    cache_dir = _resolve_split_array_cache_dir(params["data"])
    if cache_dir is None or not cache_dir.exists():
        raise SystemExit(f"Missing split cache for curr={curr}: {cache_dir}")

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
        x_train_raw, x_validate_raw, x_test_raw, enabled=bool(params["data"].get("features_normalize", False))
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
    return np.concatenate([np.mean(stacked, axis=1), np.std(stacked, axis=1)], axis=1).astype(np.float32, copy=False)


def load_aligned_multicurrent(params: dict, args: argparse.Namespace, logger: logging.Logger):
    currents = [x.strip() for x in args.currents.split(",") if x.strip()]
    features_by_split = {"train": [], "validate": [], "test": []}
    targets_ref = None
    for curr in currents:
        features_curr, targets_curr = load_one_current(
            params,
            curr=curr,
            data_dir=args.data_dir,
            data_prefix=args.data_prefix,
            n_train=args.n_train,
            n_validate=args.n_validate,
            n_test=args.n_test,
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
    x_train = aggregate_currents(features_by_split["train"], args.aggregation)
    x_validate = aggregate_currents(features_by_split["validate"], args.aggregation)
    x_test = aggregate_currents(features_by_split["test"], args.aggregation)

    targets_scale = preprocess_targets(targets_ref, params, logger)
    return currents, {"train": x_train, "validate": x_validate, "test": x_test}, targets_ref, targets_scale


def hh_raw_prior_bounds(theta_dim: int) -> tuple[np.ndarray, np.ndarray]:
    low_triplet = np.array([0.05, 0.2, 5.0], dtype=np.float32)
    high_triplet = np.array([10000.0, 100.0, 1000.0], dtype=np.float32)
    repeats = theta_dim // 3
    low = np.tile(low_triplet, repeats)
    high = np.tile(high_triplet, repeats)
    return low[:theta_dim].astype(np.float32), high[:theta_dim].astype(np.float32)


def invert_target_array(arr: np.ndarray, targets_scale: dict) -> np.ndarray:
    orig_shape = arr.shape
    flat = np.array(arr, copy=True).reshape(-1, orig_shape[-1])
    flat = _apply_scale_inverse(flat.astype(np.float32, copy=False), targets_scale)
    flat = _apply_targets_transform(flat, targets_scale.get("transform"), inverse=True, array_name="targets")
    return flat.reshape(orig_shape).astype(np.float32, copy=False)


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_coverage_bundle(y_true: np.ndarray, samples: np.ndarray, levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict[str, dict[str, object]]:
    out = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(samples, alpha, axis=1)
        upper = np.quantile(samples, 1.0 - alpha, axis=1)
        covered = (y_true >= lower) & (y_true <= upper)
        out[str(level)] = {
            "level": level,
            "overall": float(np.mean(covered)),
            "per_target": [float(x) for x in np.mean(covered, axis=0)],
        }
    return out


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
    if args.method == "fmpe" and model_kind == "maf":
        model_kind = "mlp"
    if args.method == "snre" and model_kind == "maf":
        model_kind = "resnet"
    if args.method == "snpe":
        builder = posterior_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            embedding_net=embedding_net,
        )
        return SNPE(prior=prior, density_estimator=builder, device=device, logging_level="INFO", show_progress_bars=False), args.sample_with or "direct"
    if args.method == "snle":
        builder = likelihood_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            embedding_net=torch.nn.Identity(),
        )
        return SNLE(prior=prior, density_estimator=builder, device=device, logging_level="INFO", show_progress_bars=False), args.sample_with or "mcmc"
    if args.method == "snre":
        builder = classifier_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            embedding_net_x=embedding_net,
        )
        return SNRE(prior=prior, classifier=builder, device=device, logging_level="INFO", show_progress_bars=False), args.sample_with or "mcmc"
    if args.method == "fmpe":
        builder = posterior_flow_nn(
            model=model_kind,
            z_score_theta="none",
            z_score_x="none",
            hidden_features=args.hidden_features,
            num_layers=args.num_transforms,
            embedding_net=embedding_net,
        )
        return FMPE(prior=prior, density_estimator=builder, device=device, logging_level="INFO", show_progress_bars=False), args.sample_with or "ode"
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
        return NPSE(prior=prior, score_estimator=builder, sde_type="ve", device=device, logging_level="INFO", show_progress_bars=False), args.sample_with or "sde"
    raise ValueError(args.method)


def posterior_samples_for_test(posterior, x_test: np.ndarray, num_samples: int, device: str, logger: logging.Logger) -> np.ndarray:
    out = []
    total = int(x_test.shape[0])
    with torch.inference_mode():
        for idx in range(total):
            if idx == 0 or (idx + 1) % 32 == 0 or (idx + 1) == total:
                logger.info("Sampling posterior for test observation %d/%d", idx + 1, total)
            x_i = torch.as_tensor(x_test[idx], dtype=torch.float32, device=device)
            draws = posterior.sample((num_samples,), x=x_i, show_progress_bars=False)
            out.append(draws.detach().cpu().numpy())
    return np.stack(out, axis=0)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("aligned_multicurrent_sbi")

    params = load_params(args.params)
    seed_everything(args.seed)
    device = select_device(args.device)
    logger.info("Running aligned multicurrent SBI method=%s aggregation=%s device=%s", args.method, args.aggregation, device)
    if "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))

    t0 = time.time()
    currents, features, targets, targets_scale = load_aligned_multicurrent(params, args, logger)
    y_train_norm = targets["train"]
    y_validate_norm = targets["validate"]
    y_test_norm = targets["test"]

    raw_low, raw_high = hh_raw_prior_bounds(y_train_norm.shape[1])
    bounds = np.stack([raw_low, raw_high], axis=0).astype(np.float32)
    bounds = _apply_targets_transform(bounds, targets_scale.get("transform"), inverse=False, array_name="hh_prior_bounds")
    prior_low = ((bounds[0:1] - targets_scale["shift"]) / targets_scale["mult"]).reshape(-1).astype(np.float32)
    prior_high = ((bounds[1:2] - targets_scale["shift"]) / targets_scale["mult"]).reshape(-1).astype(np.float32)

    x_train = torch.as_tensor(features["train"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train_norm, dtype=torch.float32, device=device)
    prior = BoxUniform(low=torch.as_tensor(prior_low, dtype=torch.float32), high=torch.as_tensor(prior_high, dtype=torch.float32), device=device)
    inference, sample_with = build_sbi_components(args, x_train.shape[1], y_train.shape[1], prior, device)
    inference.append_simulations(y_train, x_train, data_device=device)

    t_train = time.time()
    estimator = inference.train(
        training_batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=0.1,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_num_epochs,
        show_train_summary=True,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    training_runtime_sec = time.time() - t_train

    posterior = inference.build_posterior(estimator, sample_with=sample_with)
    t_eval = time.time()
    samples_norm = posterior_samples_for_test(posterior, features["test"], args.posterior_samples, device, logger)
    evaluation_runtime_sec = time.time() - t_eval

    y_test = invert_target_array(y_test_norm, targets_scale)
    samples = invert_target_array(samples_norm, targets_scale)
    posterior_mean = np.mean(samples, axis=1)
    posterior_median = np.median(samples, axis=1)
    posterior_std = np.std(samples, axis=1)
    if args.decision_rule == "mean":
        selected = posterior_mean
    else:
        selected = posterior_median

    mean_metrics = compute_point_metrics(y_test, posterior_mean)
    median_metrics = compute_point_metrics(y_test, posterior_median)
    selected_metrics = mean_metrics if args.decision_rule == "mean" else median_metrics
    coverage = compute_coverage_bundle(y_test, samples)

    metrics_summary = {
        "method": args.method,
        "density_estimator": args.density_estimator,
        "device": device,
        "feature_mode": args.feature_mode,
        "aggregation": args.aggregation,
        "currents": currents,
        "sample_with": sample_with,
        "decision_rule": args.decision_rule,
        "seed": args.seed,
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
        "posterior_samples": int(args.posterior_samples),
        "training_runtime_sec": float(training_runtime_sec),
        "evaluation_runtime_sec": float(evaluation_runtime_sec),
        "total_runtime_sec": float(time.time() - t0),
        "posterior_mean_metrics": mean_metrics,
        "posterior_median_metrics": median_metrics,
        "posterior_selected_metrics": selected_metrics,
        "posterior_concentration": {
            "overall_mean_std": float(np.mean(posterior_std)),
            "per_target_mean_std": [float(x) for x in np.mean(posterior_std, axis=0)],
        },
        "marginal_coverage": coverage,
    }

    (save_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    (save_dir / "params_snapshot.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    np.savez(
        save_dir / "predictions_test.npz",
        targets=y_test.astype(np.float32, copy=False),
        posterior_mean=posterior_mean.astype(np.float32, copy=False),
        posterior_median=posterior_median.astype(np.float32, copy=False),
        posterior_std=posterior_std.astype(np.float32, copy=False),
        posterior_samples=samples.astype(np.float32, copy=False),
        posterior_selected=selected.astype(np.float32, copy=False),
    )
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
