#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hh_track4_assumption_conditioned_compact_hh_utils_20260425 import (  # noqa: E402
    CURRENT_LABELS,
    CompactHHSimulationConfig,
    bundle_feature_vector,
    mapped_params_from_theta,
    sample_prior_thetas,
    simulate_theta_batch,
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


@dataclass(frozen=True)
class GBConfig:
    trace_length: int
    dt_ms: float
    pulse_start_frac: float
    pulse_end_frac: float
    g_scale_range: tuple[float, float]
    tau_scale_range: tuple[float, float]
    gl_scale_range: tuple[float, float]
    max_abs_voltage: float


class BetaPosteriorMLP(nn.Module):
    def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dim, theta_dim)
        self.logstd_head = nn.Linear(dim, theta_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.logstd_head(h), min=-5.0, max=3.0)
        return mean, log_std


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assumption-conditioned generalized-Bayes NPE adaptation over the compact HH surrogate.")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--bundle-label", default="")
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--n-train-obs", type=int, default=8)
    ap.add_argument("--n-test-obs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1.0e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-hidden-layers", type=int, default=2)
    ap.add_argument("--weight-clip", type=float, default=100.0)
    ap.add_argument("--betas", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--trace-length", type=int, default=DEFAULT_TRACE_LENGTH)
    ap.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    ap.add_argument("--pulse-start-frac", type=float, default=DEFAULT_PULSE_START_FRAC)
    ap.add_argument("--pulse-end-frac", type=float, default=DEFAULT_PULSE_END_FRAC)
    ap.add_argument("--current-gain-bounds", default=f"{DEFAULT_CURRENT_GAIN_BOUNDS[0]},{DEFAULT_CURRENT_GAIN_BOUNDS[1]}")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def parse_pair(text: str) -> tuple[float, float]:
    left, right = [float(x.strip()) for x in text.split(",")]
    if left <= 0.0 or right <= left:
        raise SystemExit(f"Expected increasing positive pair, got {text!r}")
    return left, right


def parse_betas(text: str) -> np.ndarray:
    vals = np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=np.float32)
    if vals.size == 0 or np.any(vals <= 0.0):
        raise SystemExit("Expected positive beta values.")
    return vals


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    label = args.bundle_label.strip() or "default"
    return RUNS / "generalized_bayes_npe" / f"generalized_bayes_npe_{label}_{DATE_TAG}"


def select_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def simulate_feature_bank(thetas: np.ndarray, config: CompactHHSimulationConfig, workers: int) -> tuple[np.ndarray, np.ndarray]:
    bundles, valid, _ = simulate_theta_batch(thetas, currents=CURRENT_LABELS, config=config, workers=workers, chunksize=1)
    feature_rows = []
    kept_thetas = []
    for theta, bundle, ok in zip(thetas, bundles, valid):
        if not ok:
            continue
        feature_rows.append(bundle_feature_vector(bundle, downsample_points=128, include_summary=True))
        kept_thetas.append(theta)
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(kept_thetas, dtype=np.float32)


def discrepancy_matrix(obs_features: np.ndarray, bank_features: np.ndarray) -> np.ndarray:
    diff = obs_features[:, None, :] - bank_features[None, :, :]
    return np.mean(np.square(diff), axis=2).astype(np.float32, copy=False)


def snis_weights(discrepancies: np.ndarray, beta: float, weight_clip: float) -> np.ndarray:
    shifted = discrepancies - np.min(discrepancies)
    weights = np.exp(-beta * shifted.astype(np.float64))
    weights = np.minimum(weights, weight_clip)
    weights_sum = np.sum(weights)
    if weights_sum <= 0.0 or not np.isfinite(weights_sum):
        weights = np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
    else:
        weights /= weights_sum
    return weights.astype(np.float32, copy=False)


def build_training_arrays(
    obs_train_features: np.ndarray,
    bank_features: np.ndarray,
    bank_thetas: np.ndarray,
    betas: np.ndarray,
    weight_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    discrepancies = discrepancy_matrix(obs_train_features, bank_features)
    x_rows = []
    theta_rows = []
    weight_rows = []
    for obs_idx in range(obs_train_features.shape[0]):
        for beta in betas:
            weights = snis_weights(discrepancies[obs_idx], float(beta), weight_clip)
            cond = np.concatenate([np.repeat(obs_train_features[obs_idx : obs_idx + 1], bank_features.shape[0], axis=0), np.full((bank_features.shape[0], 1), beta, dtype=np.float32)], axis=1)
            x_rows.append(cond)
            theta_rows.append(bank_thetas.astype(np.float32, copy=False))
            weight_rows.append(weights.reshape(-1, 1))
    x_train = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    theta_train = np.concatenate(theta_rows, axis=0).astype(np.float32, copy=False)
    weights_train = np.concatenate(weight_rows, axis=0).astype(np.float32, copy=False)
    return x_train, theta_train, weights_train


def weighted_gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * log_std)
    nll = 0.5 * (((target - mean) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi))
    nll = nll.mean(dim=1, keepdim=True)
    return torch.mean(weights * nll)


def train_model(model: BetaPosteriorMLP, x_train: np.ndarray, theta_train: np.ndarray, weights_train: np.ndarray, device: torch.device, epochs: int, batch_size: int, learning_rate: float) -> list[float]:
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(theta_train, dtype=torch.float32),
        torch.tensor(weights_train, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.to(device)
    model.train()
    for _ in range(epochs):
        running = 0.0
        count = 0
        for xb, yb, wb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad()
            mean, log_std = model(xb)
            loss = weighted_gaussian_nll(mean, log_std, yb, wb)
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1
        losses.append(running / max(count, 1))
    return losses


def posterior_summary(model: BetaPosteriorMLP, obs_features: np.ndarray, betas: np.ndarray, theta_true: np.ndarray, device: torch.device, draws: int = 1024) -> tuple[list[dict[str, object]], dict[tuple[int, float], np.ndarray]]:
    model.eval()
    rows: list[dict[str, object]] = []
    samples_map: dict[tuple[int, float], np.ndarray] = {}
    with torch.no_grad():
        for obs_idx in range(obs_features.shape[0]):
            for beta in betas:
                cond = np.concatenate([obs_features[obs_idx], np.asarray([beta], dtype=np.float32)], axis=0)[None, :]
                mean, log_std = model(torch.tensor(cond, dtype=torch.float32, device=device))
                mean_np = mean.cpu().numpy()[0]
                std_np = np.exp(log_std.cpu().numpy()[0])
                rng = np.random.default_rng(20260426 + obs_idx * 97 + int(beta * 100))
                draws_np = mean_np[None, :] + rng.normal(size=(draws, mean_np.shape[0])).astype(np.float32) * std_np[None, :]
                samples_map[(obs_idx, float(beta))] = draws_np.astype(np.float32)
                rows.append(
                    {
                        "observation_index": obs_idx,
                        "beta": float(beta),
                        "mean_abs_log10_error_params": float(np.mean(np.abs(theta_true[obs_idx, :6] - mean_np[:6]))),
                        "abs_log10_error_current_gain": float(abs(theta_true[obs_idx, 6] - mean_np[6])),
                        "posterior_std_norm_mean": float(np.mean(std_np)),
                        "posterior_std_norm_max": float(np.max(std_np)),
                    }
                )
    return rows, samples_map


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    config = build_config(args)
    betas = parse_betas(args.betas)
    current_gain_bounds = parse_pair(args.current_gain_bounds)

    t0 = time.time()
    bank_theta = sample_prior_thetas(args.n_base, args.seed + 1, current_gain_bounds)
    bank_features, bank_theta = simulate_feature_bank(bank_theta, config, args.workers)

    obs_train_theta = sample_prior_thetas(args.n_train_obs, args.seed + 101, current_gain_bounds)
    obs_train_features, obs_train_theta = simulate_feature_bank(obs_train_theta, config, args.workers)
    obs_test_theta = sample_prior_thetas(args.n_test_obs, args.seed + 202, current_gain_bounds)
    obs_test_features, obs_test_theta = simulate_feature_bank(obs_test_theta, config, args.workers)

    x_train, theta_train, weights_train = build_training_arrays(obs_train_features, bank_features, bank_theta, betas, args.weight_clip)

    model = BetaPosteriorMLP(input_dim=x_train.shape[1], theta_dim=theta_train.shape[1], hidden_dim=args.hidden_dim, num_hidden_layers=args.num_hidden_layers)
    losses = train_model(model, x_train, theta_train, weights_train, device, args.epochs, args.batch_size, args.learning_rate)

    train_rows, _ = posterior_summary(model, obs_train_features, betas, obs_train_theta, device, draws=512)
    test_rows, samples_map = posterior_summary(model, obs_test_features, betas, obs_test_theta, device, draws=1024)

    runtime_sec = time.time() - t0
    write_csv(save_dir / "generalized_bayes_npe_train_summary.csv", train_rows)
    write_csv(save_dir / "generalized_bayes_npe_test_summary.csv", test_rows)

    beta_summary_rows = []
    for beta in betas:
        beta_rows = [row for row in test_rows if abs(row["beta"] - float(beta)) < 1.0e-8]
        beta_summary_rows.append(
            {
                "beta": float(beta),
                "mean_abs_log10_error_params_mean": float(np.mean([row["mean_abs_log10_error_params"] for row in beta_rows])),
                "abs_log10_error_current_gain_mean": float(np.mean([row["abs_log10_error_current_gain"] for row in beta_rows])),
                "posterior_std_norm_mean_mean": float(np.mean([row["posterior_std_norm_mean"] for row in beta_rows])),
            }
        )
    write_csv(save_dir / "generalized_bayes_npe_beta_summary.csv", beta_summary_rows)

    manifest = {
        "assumption_conditioned": True,
        "provenance_recovered": False,
        "method_family": "generalized_bayes_npe_local_beta_conditioned",
        "adaptation_note": "This is a bounded assumption-conditioned adaptation of beta-conditioned generalized-Bayes NPE, using SNIS-style tempered Gibbs weights over a prior simulation bank for each observed surrogate bundle.",
        "beta_values": betas.astype(float).tolist(),
        "n_base": int(bank_theta.shape[0]),
        "n_train_obs": int(obs_train_theta.shape[0]),
        "n_test_obs": int(obs_test_theta.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
        "weight_clip": args.weight_clip,
        "runtime_sec": float(runtime_sec),
        "training_loss_final": float(losses[-1]),
        "beta_summary": beta_summary_rows,
        "mapped_example_true": mapped_params_from_theta(obs_test_theta[0], g_scale_range=DEFAULT_G_SCALE_RANGE, tau_scale_range=DEFAULT_TAU_SCALE_RANGE, gl_scale_range=DEFAULT_GL_SCALE_RANGE),
    }
    (save_dir / "generalized_bayes_npe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# HH Track4 Generalized-Bayes NPE Adaptation",
        "",
        "- assumption-conditioned adaptation, not a provenance-faithful run",
        "- fixed-base SNIS-style reweighting over a surrogate simulation bank",
        "",
        "| beta | mean_abs_log10_error_params_mean | abs_log10_error_current_gain_mean | posterior_std_norm_mean_mean |",
        "|---:|---:|---:|---:|",
    ]
    for row in beta_summary_rows:
        report.append(
            f"| {row['beta']:.2f} | {row['mean_abs_log10_error_params_mean']:.6f} | {row['abs_log10_error_current_gain_mean']:.6f} | {row['posterior_std_norm_mean_mean']:.6f} |"
        )
    (save_dir / "generalized_bayes_npe_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
