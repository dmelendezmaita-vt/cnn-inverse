#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import types

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

from sbi.utils import BoxUniform

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
sys.path.insert(0, str(REPO / "scripts"))

import run_hh_sbi_baseline_20260422 as baseline  # noqa: E402

DEFAULT_SHARED_DATA_DIR = str(
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pool-based sequential SBI over cached HH Track4 split arrays.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--method", default="snpe", choices=["snpe", "snle", "snre", "fmpe", "npse"])
    ap.add_argument("--density-estimator", default="maf")
    ap.add_argument("--sample-with", default=None)
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--final-train-size", type=int, default=None)
    ap.add_argument("--n-validate", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=1024)
    ap.add_argument("--data-dir", default=DEFAULT_SHARED_DATA_DIR)
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--curr", default="0.1")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--feature-mode",
        default="raw_plus_fft256_summary12",
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
    ap.add_argument("--posterior-samples", type=int, default=8)
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
    ap.add_argument("--selection-split", default="validate", choices=["validate"])
    ap.add_argument("--selection-cov-lambda", type=float, default=1.0)
    ap.add_argument("--selection-candidate-rules", default="mean,median")
    ap.add_argument("--rounds", type=int, default=3, choices=[2, 3])
    ap.add_argument("--round-train-sizes", default=None, help="Comma-separated cumulative sizes, for example 512,1024,2048.")
    ap.add_argument("--initial-train-size", type=int, default=None)
    ap.add_argument("--candidate-pool-size", type=int, default=1024)
    ap.add_argument("--candidate-posterior-samples", type=int, default=4)
    return ap.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise SystemExit("Expected a non-empty comma-separated integer list.")
    return values


def apply_param_overrides(
    params: dict,
    *,
    args: argparse.Namespace,
    pool_size: int,
    n_validate: int,
    n_test: int,
) -> dict:
    effective = json.loads(json.dumps(params))
    effective.setdefault("data", {})
    effective["data"]["data_dir"] = args.data_dir
    effective["data"]["data_prefix"] = args.data_prefix
    effective["data"]["curr"] = args.curr
    effective["data"]["Ntrain"] = int(pool_size)
    effective["data"]["Nvalidate"] = int(n_validate)
    effective["data"]["Ntest"] = int(n_test)
    effective["data"]["split_array_cache_enabled"] = True
    effective["data"]["features_sub_begin_random"] = False
    effective["data"]["features_sub_begin_random_eval"] = False
    return effective


def even_cumulative_sizes(total: int, rounds: int) -> list[int]:
    if total < rounds:
        raise SystemExit(f"final_train_size={total} is too small for rounds={rounds}")
    base = total // rounds
    rem = total % rounds
    sizes = []
    running = 0
    for idx in range(rounds):
        running += base + (1 if idx < rem else 0)
        sizes.append(running)
    if sizes[-1] != total:
        raise AssertionError("Even cumulative schedule did not sum to total.")
    return sizes


def default_round_train_sizes(final_train_size: int, rounds: int) -> list[int]:
    if rounds == 2:
        sizes = [final_train_size // 2, final_train_size]
    else:
        sizes = [final_train_size // 4, final_train_size // 2, final_train_size]
    if any(size <= 0 for size in sizes) or any(curr <= prev for prev, curr in zip(sizes, sizes[1:])):
        return even_cumulative_sizes(final_train_size, rounds)
    sizes[-1] = final_train_size
    return sizes


def schedule_from_initial(initial_train_size: int, final_train_size: int, rounds: int) -> list[int]:
    if initial_train_size <= 0:
        raise SystemExit(f"initial_train_size must be positive, got {initial_train_size}")
    if initial_train_size >= final_train_size:
        raise SystemExit(
            f"initial_train_size={initial_train_size} must be smaller than final_train_size={final_train_size}"
        )
    if rounds == 2:
        return [initial_train_size, final_train_size]
    remaining = final_train_size - initial_train_size
    second_add = remaining // 2
    second = initial_train_size + second_add
    if second <= initial_train_size or second >= final_train_size:
        return even_cumulative_sizes(final_train_size, rounds)
    return [initial_train_size, second, final_train_size]


def resolve_round_train_sizes(args: argparse.Namespace, final_train_size: int) -> list[int]:
    if args.round_train_sizes:
        sizes = parse_int_list(args.round_train_sizes)
        if len(sizes) not in {2, 3}:
            raise SystemExit("round_train_sizes must define exactly 2 or 3 cumulative round sizes.")
        if sizes[-1] != final_train_size:
            raise SystemExit(
                f"round_train_sizes must end at final_train_size={final_train_size}, got {sizes[-1]}"
            )
        if any(size <= 0 for size in sizes):
            raise SystemExit("round_train_sizes must be positive.")
        if any(curr <= prev for prev, curr in zip(sizes, sizes[1:])):
            raise SystemExit("round_train_sizes must be strictly increasing.")
        return sizes
    if args.initial_train_size is not None:
        return schedule_from_initial(args.initial_train_size, final_train_size, args.rounds)
    return default_round_train_sizes(final_train_size, args.rounds)


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def resolve_effective_density_estimator(args: argparse.Namespace) -> str:
    if args.method != "fmpe":
        return args.density_estimator
    allowed = {"mlp", "ada_mlp", "transformer", "transformer_cross_attn"}
    if args.density_estimator in allowed:
        return args.density_estimator
    return "mlp"


def train_round(
    *,
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    device: str,
) -> tuple[object, str, float]:
    prior = BoxUniform(
        low=torch.as_tensor(prior_low, dtype=torch.float32),
        high=torch.as_tensor(prior_high, dtype=torch.float32),
        device=device,
    )
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    build_args = SimpleNamespace(**vars(args))
    build_args.density_estimator = resolve_effective_density_estimator(args)
    inference, sample_with = baseline.build_sbi_components(
        build_args,
        x_train.shape[1],
        y_train.shape[1],
        prior,
        device,
    )
    inference.append_simulations(y_train_t, x_train_t, data_device=device)
    t0 = time.time()
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
    training_runtime_sec = time.time() - t0
    posterior = inference.build_posterior(estimator, sample_with=sample_with)
    return posterior, sample_with, training_runtime_sec


def score_candidate_pool(
    *,
    posterior,
    x_pool: np.ndarray,
    num_samples: int,
    device: str,
    logger: logging.Logger,
) -> np.ndarray:
    samples = baseline.posterior_samples_for_test(
        posterior,
        x_test=x_pool,
        num_samples=num_samples,
        device=device,
        logger=logger,
    )
    return np.mean(np.std(samples, axis=1), axis=1).astype(np.float32, copy=False)


def acquire_next_batch(
    *,
    round_index: int,
    posterior,
    train_features: np.ndarray,
    remaining_indices: np.ndarray,
    batch_size: int,
    candidate_pool_size: int,
    candidate_posterior_samples: int,
    device: str,
    logger: logging.Logger,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    if batch_size <= 0:
        raise SystemExit(f"Requested non-positive acquisition batch_size={batch_size}")
    remaining_count = int(remaining_indices.shape[0])
    if batch_size > remaining_count:
        raise SystemExit(f"Requested batch_size={batch_size} exceeds remaining pool size {remaining_count}")

    eval_size = min(remaining_count, max(candidate_pool_size, batch_size))
    if eval_size == remaining_count:
        candidate_indices = np.array(remaining_indices, copy=True)
    else:
        candidate_indices = np.sort(rng.choice(remaining_indices, size=eval_size, replace=False))

    logger.info(
        "Round %d acquisition: scoring %d candidates to select %d observations from remaining pool of %d",
        round_index,
        eval_size,
        batch_size,
        remaining_count,
    )
    t0 = time.time()
    candidate_scores = score_candidate_pool(
        posterior=posterior,
        x_pool=train_features[candidate_indices],
        num_samples=candidate_posterior_samples,
        device=device,
        logger=logger,
    )
    acquisition_runtime_sec = time.time() - t0

    order = np.argsort(-candidate_scores, kind="mergesort")
    guided_count = min(batch_size, candidate_indices.shape[0])
    guided_indices = candidate_indices[order[:guided_count]]
    guided_scores = candidate_scores[order[:guided_count]]

    random_fill_count = batch_size - guided_count
    if random_fill_count > 0:
        fill_source = np.setdiff1d(remaining_indices, guided_indices, assume_unique=False)
        fill_indices = rng.choice(fill_source, size=random_fill_count, replace=False)
        selected_indices = np.concatenate([guided_indices, fill_indices.astype(np.int64, copy=False)])
    else:
        selected_indices = guided_indices

    metadata = {
        "candidate_eval_size": int(eval_size),
        "guided_selected_count": int(guided_count),
        "random_fill_count": int(random_fill_count),
        "candidate_posterior_samples": int(candidate_posterior_samples),
        "acquisition_runtime_sec": float(acquisition_runtime_sec),
        "candidate_score_summary": summarize_scores(candidate_scores),
        "selected_score_summary": summarize_scores(guided_scores),
    }
    return selected_indices.astype(np.int64, copy=False), metadata


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hh_track4_pool_sequential_sbi")

    params = baseline.load_params(args.params)
    seed = int(args.seed if args.seed is not None else params["data"].get("random_seed", 0))
    args.seed = seed
    baseline.seed_everything(seed)

    device = baseline.select_device(args.device)
    effective_density_estimator = resolve_effective_density_estimator(args)
    logger.info(
        "Running HH Track4 pool sequential SBI with method=%s density_estimator=%s device=%s seed=%d",
        args.method,
        effective_density_estimator,
        device,
        seed,
    )
    if args.method == "fmpe" and effective_density_estimator != args.density_estimator:
        logger.info(
            "FMPE remapping density_estimator=%s to supported vector-field architecture=%s",
            args.density_estimator,
            effective_density_estimator,
        )

    track_cuda_memory = device.startswith("cuda") and torch.cuda.is_available()
    if track_cuda_memory:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    if "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))

    t0 = time.time()
    pool_size = int(args.pool_size if args.pool_size is not None else params["data"].get("Ntrain", 0))
    if pool_size <= 0:
        raise SystemExit(f"Resolved invalid pool_size={pool_size}")
    n_validate = int(args.n_validate if args.n_validate is not None else params["data"].get("Nvalidate", 0))
    n_test = int(args.n_test)
    final_train_size = int(args.final_train_size if args.final_train_size is not None else pool_size)
    if final_train_size <= 0:
        raise SystemExit(f"Resolved invalid final_train_size={final_train_size}")
    if final_train_size > pool_size:
        raise SystemExit(f"final_train_size={final_train_size} exceeds pool_size={pool_size}")
    if args.candidate_pool_size <= 0:
        raise SystemExit(f"candidate_pool_size must be positive, got {args.candidate_pool_size}")
    if args.candidate_posterior_samples <= 0:
        raise SystemExit(
            f"candidate_posterior_samples must be positive, got {args.candidate_posterior_samples}"
        )
    if args.posterior_samples <= 0:
        raise SystemExit(f"posterior_samples must be positive, got {args.posterior_samples}")
    if n_validate <= 0 and args.decision_rule == "validate_calibrated":
        raise SystemExit("decision_rule=validate_calibrated requires n_validate > 0.")

    round_train_sizes = resolve_round_train_sizes(args, final_train_size)
    if round_train_sizes[-1] != final_train_size:
        raise SystemExit("Internal error: round_train_sizes did not end at final_train_size.")
    if round_train_sizes[-1] > pool_size:
        raise SystemExit(
            f"Final round size {round_train_sizes[-1]} exceeds pool_size={pool_size}"
        )

    params_effective = apply_param_overrides(
        params,
        args=args,
        pool_size=pool_size,
        n_validate=n_validate,
        n_test=n_test,
    )
    cache_args = SimpleNamespace(**vars(args))
    cache_args.n_train = pool_size
    cache_args.eval_limit = n_test
    features, targets, targets_scale, prior_low, prior_high = baseline.load_cached_splits(
        params_effective,
        cache_args,
        logger,
    )

    train_features = features["train"]
    train_targets = targets["train"]
    if train_features.shape[0] != pool_size or train_targets.shape[0] != pool_size:
        raise SystemExit(
            f"Loaded pool shape mismatch: features={train_features.shape} targets={train_targets.shape} pool_size={pool_size}"
        )

    rng = np.random.default_rng(seed)
    pool_order = rng.permutation(pool_size).astype(np.int64, copy=False)
    selected_pool_indices = np.array(pool_order[:round_train_sizes[0]], copy=True)
    selection_round = np.full(pool_size, -1, dtype=np.int16)
    selection_round[selected_pool_indices] = 1
    round_history: list[dict[str, object]] = []
    posterior = None
    sample_with = None
    training_runtime_sec = 0.0

    logger.info(
        "Sequential schedule over fixed HH pool: pool_size=%d final_train_size=%d round_train_sizes=%s",
        pool_size,
        final_train_size,
        round_train_sizes,
    )

    for round_index, round_train_size in enumerate(round_train_sizes, start=1):
        if selected_pool_indices.shape[0] != round_train_size:
            raise SystemExit(
                f"Round {round_index} expected {round_train_size} selected points, found {selected_pool_indices.shape[0]}"
            )
        previous_size = 0 if round_index == 1 else round_train_sizes[round_index - 2]
        logger.info(
            "Training round %d/%d on %d selected examples (%d newly available this round)",
            round_index,
            len(round_train_sizes),
            round_train_size,
            round_train_size - previous_size,
        )
        posterior, sample_with, round_training_runtime_sec = train_round(
            args=args,
            x_train=train_features[selected_pool_indices],
            y_train=train_targets[selected_pool_indices],
            prior_low=prior_low,
            prior_high=prior_high,
            device=device,
        )
        training_runtime_sec += round_training_runtime_sec

        remaining_before = int(np.sum(selection_round < 0))
        round_record: dict[str, object] = {
            "round_index": int(round_index),
            "train_size": int(round_train_size),
            "new_examples_this_round": int(round_train_size - previous_size),
            "training_runtime_sec": float(round_training_runtime_sec),
            "remaining_pool_size_before_acquisition": remaining_before,
            "sample_with": sample_with,
        }

        if round_index < len(round_train_sizes):
            next_round_size = round_train_sizes[round_index]
            batch_size = next_round_size - round_train_size
            remaining_indices = np.flatnonzero(selection_round < 0).astype(np.int64, copy=False)
            acquired_indices, acquisition_meta = acquire_next_batch(
                round_index=round_index,
                posterior=posterior,
                train_features=train_features,
                remaining_indices=remaining_indices,
                batch_size=batch_size,
                candidate_pool_size=args.candidate_pool_size,
                candidate_posterior_samples=args.candidate_posterior_samples,
                device=device,
                logger=logger,
                rng=rng,
            )
            selection_round[acquired_indices] = round_index + 1
            selected_pool_indices = np.concatenate([selected_pool_indices, acquired_indices])
            round_record["acquisition_batch_size"] = int(batch_size)
            round_record["remaining_pool_size_after_acquisition"] = int(np.sum(selection_round < 0))
            round_record.update(acquisition_meta)
        else:
            round_record["acquisition_batch_size"] = 0
            round_record["remaining_pool_size_after_acquisition"] = remaining_before
            round_record["candidate_eval_size"] = 0
            round_record["guided_selected_count"] = 0
            round_record["random_fill_count"] = 0
            round_record["candidate_posterior_samples"] = int(args.candidate_posterior_samples)
            round_record["acquisition_runtime_sec"] = 0.0
            round_record["candidate_score_summary"] = None
            round_record["selected_score_summary"] = None
        round_history.append(round_record)

    if posterior is None or sample_with is None:
        raise SystemExit("Sequential SBI finished without a trained posterior.")
    if np.unique(selected_pool_indices).size != selected_pool_indices.size:
        raise SystemExit("Sequential selection produced duplicate pool indices.")
    if selected_pool_indices.shape[0] != final_train_size:
        raise SystemExit(
            f"Sequential selection ended with {selected_pool_indices.shape[0]} indices, expected {final_train_size}"
        )

    x_validate = features["validate"]
    y_validate_norm = targets["validate"]
    x_test = features["test"]
    y_test_norm = targets["test"]

    validation_samples_norm = None
    validation_coverage = None
    validation_rule_scores = None
    selected_rule = args.decision_rule
    selected_rule_source = "cli"
    selected_rule_score = None
    selected_metrics = None
    selection_candidate_rules = baseline.parse_candidate_rules(args.selection_candidate_rules)
    validation_map_norm = None
    validation_map = None
    if args.decision_rule == "map" and not args.compute_map:
        raise SystemExit("decision_rule=map requires --compute-map.")
    if "map" in selection_candidate_rules and not args.compute_map:
        raise SystemExit("selection_candidate_rules includes map but --compute-map is disabled.")

    t_eval_start = time.time()
    if args.decision_rule == "validate_calibrated":
        validation_samples_norm = baseline.posterior_samples_for_test(
            posterior,
            x_test=x_validate,
            num_samples=args.posterior_samples,
            device=device,
            logger=logger,
        )
        validation_coverage = baseline.compute_coverage_bundle(y_validate_norm, validation_samples_norm)
        coverage_gap = baseline.mean_coverage_gap(validation_coverage)
        if "map" in selection_candidate_rules:
            validation_map_norm = baseline.posterior_map_for_test(
                posterior,
                x_test=x_validate,
                device=device,
                args=args,
                logger=logger,
            )
            validation_map = baseline.invert_target_array(validation_map_norm, targets_scale)

        validation_rule_scores = {}
        for rule in selection_candidate_rules:
            pred_norm = baseline.select_point_prediction(
                rule,
                samples=validation_samples_norm,
                posterior_map=validation_map_norm,
            )
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

    samples_norm = baseline.posterior_samples_for_test(
        posterior,
        x_test=x_test,
        num_samples=args.posterior_samples,
        device=device,
        logger=logger,
    )
    evaluation_runtime_sec = time.time() - t_eval_start

    y_test = baseline.invert_target_array(y_test_norm, targets_scale)
    samples = baseline.invert_target_array(samples_norm, targets_scale)
    posterior_mean = np.mean(samples, axis=1)
    posterior_median = np.median(samples, axis=1)
    posterior_std = np.std(samples, axis=1)
    posterior_map = None
    map_runtime_sec = 0.0
    map_metrics = None
    if args.compute_map:
        t_map_start = time.time()
        posterior_map_norm = baseline.posterior_map_for_test(
            posterior,
            x_test=x_test,
            device=device,
            args=args,
            logger=logger,
        )
        map_runtime_sec = time.time() - t_map_start
        posterior_map = baseline.invert_target_array(posterior_map_norm, targets_scale)
        map_metrics = baseline.compute_point_metrics(y_test, posterior_map)

    mean_metrics = baseline.compute_point_metrics(y_test, posterior_mean)
    median_metrics = baseline.compute_point_metrics(y_test, posterior_median)
    coverage = baseline.compute_coverage_bundle(y_test, samples)

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

    sequential_summary = {
        "pool_size": int(pool_size),
        "final_train_size": int(final_train_size),
        "rounds": int(len(round_train_sizes)),
        "round_train_sizes": [int(x) for x in round_train_sizes],
        "initial_train_size": int(round_train_sizes[0]),
        "candidate_pool_size": int(args.candidate_pool_size),
        "candidate_posterior_samples": int(args.candidate_posterior_samples),
        "acquisition_score": "mean_normalized_posterior_std",
        "selection_round_counts": {
            str(round_index): int(np.sum(selection_round == round_index))
            for round_index in range(1, len(round_train_sizes) + 1)
        },
        "unused_pool_size": int(np.sum(selection_round < 0)),
        "round_history": round_history,
    }

    metrics_summary = {
        "method": args.method,
        "density_estimator": args.density_estimator,
        "effective_density_estimator": effective_density_estimator,
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
        "n_train": int(final_train_size),
        "pool_train_size": int(pool_size),
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
        "sequential": sequential_summary,
    }

    (save_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    (save_dir / "params_snapshot.yaml").write_text(yaml.safe_dump(params_effective, sort_keys=False))
    with (save_dir / "predictions_test.npz").open("wb") as f:
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
        np.savez(f, **payload)
    with (save_dir / "pool_selection.npz").open("wb") as f:
        np.savez(
            f,
            selected_pool_indices=selected_pool_indices.astype(np.int64, copy=False),
            selection_round=selection_round.astype(np.int16, copy=False),
            pool_order=pool_order.astype(np.int64, copy=False),
            round_train_sizes=np.asarray(round_train_sizes, dtype=np.int64),
        )

    logger.info("Wrote metrics summary to %s", save_dir / "metrics_summary.json")
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
