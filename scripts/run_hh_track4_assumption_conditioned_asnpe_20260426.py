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

os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import tensorflow as tf  # type: ignore
    if not hasattr(tf, "TensorShape"):
        raise ImportError("TensorFlow stub required")
except Exception:
    tf_stub = types.ModuleType("tensorflow")

    class TensorShape(tuple):
        def as_list(self):
            return list(self)

    class DType:
        pass

    class TypeSpec:
        pass

    class SparseTensor:
        pass

    class RaggedTensor:
        pass

    tf_stub.TensorShape = TensorShape
    tf_stub.DType = DType
    tf_stub.TypeSpec = TypeSpec
    tf_stub.SparseTensor = SparseTensor
    tf_stub.RaggedTensor = RaggedTensor
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = tf_stub

import bayesflow as bf

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch").resolve()
sys.path.append(str(REPO / "scripts"))

from hh_track4_assumption_conditioned_compact_hh_utils_20260425 import (  # noqa: E402
    CURRENT_LABELS,
    CompactHHSimulationConfig,
    bundle_feature_vector,
    mapped_params_from_theta,
    sample_prior_thetas,
    simulate_theta_batch,
    unpack_theta,
    write_csv,
)
from hh_track4_assumption_conditioned_surrogate_contract_20260425 import (  # noqa: E402
    DATE_TAG,
    DEFAULT_CURRENT_GAIN_BOUNDS,
    DEFAULT_DT_MS,
    DEFAULT_G_SCALE_RANGE,
    DEFAULT_GL_SCALE_RANGE,
    DEFAULT_PULSE_END_FRAC,
    DEFAULT_PULSE_START_FRAC,
    DEFAULT_TAU_SCALE_RANGE,
    DEFAULT_TRACE_LENGTH,
    RUNS,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assumption-conditioned ASNPE-style loop over the compact HH surrogate.")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--bundle-label", default="")
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--ensemble-size", type=int, default=2)
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--initial-sims", type=int, default=128)
    ap.add_argument("--round-sims", type=int, default=32)
    ap.add_argument("--candidate-pool", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=5.0e-4)
    ap.add_argument("--posterior-samples", type=int, default=256)
    ap.add_argument("--sample-batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--trace-length", type=int, default=DEFAULT_TRACE_LENGTH)
    ap.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    ap.add_argument("--pulse-start-frac", type=float, default=DEFAULT_PULSE_START_FRAC)
    ap.add_argument("--pulse-end-frac", type=float, default=DEFAULT_PULSE_END_FRAC)
    ap.add_argument("--current-gain-bounds", default=f"{DEFAULT_CURRENT_GAIN_BOUNDS[0]},{DEFAULT_CURRENT_GAIN_BOUNDS[1]}")
    return ap.parse_args()


def parse_pair(text: str) -> tuple[float, float]:
    left, right = [float(x.strip()) for x in text.split(",")]
    if left <= 0.0 or right <= left:
        raise SystemExit(f"Expected increasing positive pair, got {text!r}")
    return left, right


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    label = args.bundle_label.strip() or "default"
    return RUNS / "asnpe" / f"asnpe_{label}_{DATE_TAG}"


def build_config(args: argparse.Namespace) -> CompactHHSimulationConfig:
    return CompactHHSimulationConfig(
        trace_length=args.trace_length,
        dt_ms=args.dt_ms,
        pulse_start_frac=args.pulse_start_frac,
        pulse_end_frac=args.pulse_end_frac,
        g_scale_range=DEFAULT_G_SCALE_RANGE,
        tau_scale_range=DEFAULT_TAU_SCALE_RANGE,
        gl_scale_range=DEFAULT_GL_SCALE_RANGE,
        max_abs_voltage=5000.0,
    )


def simulate_dataset(thetas: np.ndarray, config: CompactHHSimulationConfig, workers: int) -> tuple[np.ndarray, np.ndarray]:
    bundles, valid, _max_abs = simulate_theta_batch(thetas, currents=CURRENT_LABELS, config=config, workers=workers, chunksize=1)
    rows = []
    kept_thetas = []
    feature_dim: int | None = None
    for theta, bundle, ok in zip(thetas, bundles, valid):
        if feature_dim is None and bundle.ndim == 2:
            try:
                feature_dim = int(
                    bundle_feature_vector(np.nan_to_num(bundle, nan=0.0, posinf=0.0, neginf=0.0), downsample_points=128, include_summary=True).shape[0]
                )
            except Exception:
                feature_dim = None
        if not ok:
            continue
        rows.append(bundle_feature_vector(bundle, downsample_points=128, include_summary=True))
        kept_thetas.append(theta)
    theta_dim = int(np.asarray(thetas).shape[1]) if np.asarray(thetas).ndim == 2 else 0
    if not rows:
        if feature_dim is None:
            feature_dim = len(CURRENT_LABELS) * (128 + 12)
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, theta_dim), dtype=np.float32),
        )
    return np.asarray(rows, dtype=np.float32), np.asarray(kept_thetas, dtype=np.float32)


def build_workflow(save_dir: Path, learning_rate: float) -> bf.workflows.BasicWorkflow:
    return bf.workflows.BasicWorkflow(
        simulator=None,
        inference_network="coupling_flow",
        summary_network=None,
        initial_learning_rate=learning_rate,
        checkpoint_filepath=str(save_dir / "checkpoints"),
        save_weights_only=False,
        save_best_only=True,
        inference_variables=["theta"],
        inference_conditions=["x"],
        standardize="inference_variables",
    )


def fit_ensemble(
    x_train: np.ndarray,
    theta_train: np.ndarray,
    ensemble_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    member_root: Path,
) -> list[bf.workflows.BasicWorkflow]:
    workflows: list[bf.workflows.BasicWorkflow] = []
    for member_idx in range(ensemble_size):
        member_dir = member_root / f"member_{member_idx:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        workflow = build_workflow(member_dir, learning_rate)
        workflow.fit_offline(
            data={"theta": theta_train.astype(np.float32, copy=False), "x": x_train.astype(np.float32, copy=False)},
            validation_data=None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        workflows.append(workflow)
    return workflows


def proposal_sample(workflows: list[bf.workflows.BasicWorkflow], x_obs: np.ndarray, n_candidates: int) -> np.ndarray:
    n_per = max(1, n_candidates // len(workflows))
    samples = []
    for workflow in workflows:
        draws = workflow.sample(
            num_samples=n_per,
            conditions={"x": x_obs.astype(np.float32, copy=False)},
            batch_size=1,
            split=False,
        )["theta"][0]
        samples.append(np.asarray(draws, dtype=np.float32))
    out = np.concatenate(samples, axis=0)
    if out.shape[0] >= n_candidates:
        return out[:n_candidates].astype(np.float32, copy=False)
    extra = workflows[0].sample(
        num_samples=n_candidates - out.shape[0],
        conditions={"x": x_obs.astype(np.float32, copy=False)},
        batch_size=1,
        split=False,
    )["theta"][0]
    return np.concatenate([out, np.asarray(extra, dtype=np.float32)], axis=0).astype(np.float32, copy=False)


def acquisition_scores(workflows: list[bf.workflows.BasicWorkflow], candidate_theta: np.ndarray, x_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repeated_x = np.repeat(x_obs.astype(np.float32, copy=False), candidate_theta.shape[0], axis=0)
    log_probs = []
    for workflow in workflows:
        lp = workflow.approximator.log_prob({"theta": candidate_theta.astype(np.float32, copy=False), "x": repeated_x})
        log_probs.append(np.asarray(lp, dtype=np.float64).reshape(-1))
    log_probs = np.stack(log_probs, axis=1)
    log_mean = np.logaddexp.reduce(log_probs, axis=1) - np.log(log_probs.shape[1])
    proposal_density = np.exp(log_mean - np.max(log_mean))
    density_var = np.var(np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True)), axis=1)
    score = proposal_density * density_var
    return score.astype(np.float64), log_mean.astype(np.float64), density_var.astype(np.float64)


def posterior_summary(workflows: list[bf.workflows.BasicWorkflow], x_obs: np.ndarray, posterior_samples: int, sample_batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    draws = []
    n_per = max(1, posterior_samples // len(workflows))
    for workflow in workflows:
        arr = workflow.sample(
            num_samples=n_per,
            conditions={"x": x_obs.astype(np.float32, copy=False)},
            batch_size=1,
            split=False,
            sample_batch_size=sample_batch_size,
        )["theta"][0]
        draws.append(np.asarray(arr, dtype=np.float32))
    samples = np.concatenate(draws, axis=0)
    if samples.shape[0] > posterior_samples:
        samples = samples[:posterior_samples]
    mean = np.mean(samples, axis=0).astype(np.float32)
    return mean, samples.astype(np.float32)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("asnpe")

    current_gain_bounds = parse_pair(args.current_gain_bounds)
    config = build_config(args)

    theta_true = sample_prior_thetas(1, args.seed, current_gain_bounds)[0].astype(np.float32)
    x_obs_bundle, valid, _ = simulate_theta_batch(theta_true.reshape(1, -1), currents=CURRENT_LABELS, config=config, workers=1)
    if not valid[0]:
        raise SystemExit("True observation simulation invalid under surrogate.")
    x_obs = bundle_feature_vector(x_obs_bundle[0], downsample_points=128, include_summary=True).reshape(1, -1).astype(np.float32)

    theta_train = sample_prior_thetas(args.initial_sims, args.seed + 1, current_gain_bounds)
    x_train, theta_train = simulate_dataset(theta_train, config, args.workers)
    round_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    total_start = time.time()
    for round_idx in range(args.n_rounds):
        round_dir = save_dir / f"round_{round_idx:02d}"
        workflows = fit_ensemble(
            x_train=x_train,
            theta_train=theta_train,
            ensemble_size=args.ensemble_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            member_root=round_dir / "ensemble",
        )
        posterior_mean, posterior_samples = posterior_summary(
            workflows,
            x_obs=x_obs,
            posterior_samples=args.posterior_samples,
            sample_batch_size=args.sample_batch_size,
        )

        if round_idx == args.n_rounds - 1:
            raw_true, gain_true = unpack_theta(theta_true)
            raw_mean, gain_mean = unpack_theta(posterior_mean)
            round_rows.append(
                {
                    "round_index": round_idx,
                    "dataset_size": int(theta_train.shape[0]),
                    "selected_count": 0,
                    "mean_candidate_score": 0.0,
                    "posterior_mean_abs_log10_error": float(np.mean(np.abs(theta_true[:6] - posterior_mean[:6]))),
                    "posterior_current_gain_abs_log10_error": float(abs(theta_true[6] - posterior_mean[6])),
                    "posterior_mean_relative_error_raw": float(np.mean(np.abs(raw_true - raw_mean) / np.maximum(np.abs(raw_true), 1.0e-12))),
                    "posterior_current_gain_relative_error": float(abs(gain_true - gain_mean) / max(abs(gain_true), 1.0e-12)),
                }
            )
            final_workflows = workflows
            final_posterior_samples = posterior_samples
            final_posterior_mean = posterior_mean
            break

        candidate_theta = proposal_sample(workflows, x_obs, args.candidate_pool)
        score, log_mean, density_var = acquisition_scores(workflows, candidate_theta, x_obs)
        top_idx = np.argsort(score)[::-1][: args.round_sims]
        selected_theta = candidate_theta[top_idx]
        selected_x, selected_theta_valid = simulate_dataset(selected_theta, config, args.workers)
        x_train = np.concatenate([x_train, selected_x], axis=0).astype(np.float32, copy=False)
        theta_train = np.concatenate([theta_train, selected_theta_valid], axis=0).astype(np.float32, copy=False)

        round_rows.append(
            {
                "round_index": round_idx,
                "dataset_size": int(theta_train.shape[0]),
                "selected_count": int(selected_theta_valid.shape[0]),
                "mean_candidate_score": float(np.mean(score)),
                "max_candidate_score": float(np.max(score)),
                "posterior_mean_abs_log10_error": float(np.mean(np.abs(theta_true[:6] - posterior_mean[:6]))),
                "posterior_current_gain_abs_log10_error": float(abs(theta_true[6] - posterior_mean[6])),
            }
        )
        for rank, idx in enumerate(top_idx):
            candidate_rows.append(
                {
                    "round_index": round_idx,
                    "candidate_rank": rank,
                    "selected": 1,
                    "score": float(score[idx]),
                    "log_mean_density": float(log_mean[idx]),
                    "density_variance": float(density_var[idx]),
                    **{f"theta_{j+1}": float(candidate_theta[idx, j]) for j in range(candidate_theta.shape[1])},
                }
            )
    runtime_sec = time.time() - total_start

    summary = {
        "assumption_conditioned": True,
        "method_family": "asnpe_style_active_posterior_estimation",
        "bayesian_nde_approximation": {
            "type": "deep_ensemble",
            "ensemble_size": args.ensemble_size,
            "note": "p(phi|D) is approximated by an independently trained BayesFlow ensemble, and acquisition uses proposal-density-weighted disagreement across component posterior densities at x0.",
        },
        "n_rounds": args.n_rounds,
        "initial_sims": args.initial_sims,
        "round_sims": args.round_sims,
        "candidate_pool": args.candidate_pool,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "posterior_samples": int(final_posterior_samples.shape[0]),
        "runtime_sec": float(runtime_sec),
        "posterior_mean_abs_log10_error": float(np.mean(np.abs(theta_true[:6] - final_posterior_mean[:6]))),
        "posterior_current_gain_abs_log10_error": float(abs(theta_true[6] - final_posterior_mean[6])),
        "theta_true": theta_true.astype(float).tolist(),
        "theta_posterior_mean": final_posterior_mean.astype(float).tolist(),
        "mapped_true": mapped_params_from_theta(theta_true, g_scale_range=DEFAULT_G_SCALE_RANGE, tau_scale_range=DEFAULT_TAU_SCALE_RANGE, gl_scale_range=DEFAULT_GL_SCALE_RANGE),
        "mapped_posterior_mean": mapped_params_from_theta(final_posterior_mean, g_scale_range=DEFAULT_G_SCALE_RANGE, tau_scale_range=DEFAULT_TAU_SCALE_RANGE, gl_scale_range=DEFAULT_GL_SCALE_RANGE),
    }

    write_csv(save_dir / "asnpe_round_history.csv", round_rows)
    if candidate_rows:
        write_csv(save_dir / "asnpe_candidate_scores.csv", candidate_rows)
    np.savez_compressed(
        save_dir / "asnpe_posterior_samples.npz",
        theta_true=theta_true.astype(np.float32),
        posterior_mean=final_posterior_mean.astype(np.float32),
        posterior_samples=final_posterior_samples.astype(np.float32),
        x_obs=x_obs.astype(np.float32),
    )
    (save_dir / "asnpe_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
