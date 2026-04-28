#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hh_track4_assumption_conditioned_surrogate_contract_20260425 import (
    ACTIVE_CURRENT_GRID,
    DATE_TAG,
    DEFAULT_CURRENT_GAIN_BOUNDS,
    DEFAULT_DOWNSAMPLE_POINTS,
    RUNS_ACTIVE_SEQUENTIAL,
)
from hh_track4_assumption_conditioned_compact_hh_utils_20260425 import (
    CompactHHSimulationConfig,
    mapped_params_from_theta,
    sample_prior_thetas,
    sample_unit_directions,
    trace_feature_matrix,
    unpack_theta,
    write_csv,
)
from run_hh_track4_assumption_conditioned_compact_hh_sandbox_20260424 import simulate_compact_hh


ACQUISITION_POLICIES = ("uncertainty", "disagreement", "wasserstein")
ACQUISITION_POLICY_DESCRIPTIONS = {
    "uncertainty": "mean marginal posterior predictive standard deviation over the surrogate trace feature bank",
    "disagreement": "expected pairwise Euclidean disagreement between posterior predictive feature draws",
    "wasserstein": "mean pairwise projected 1-Wasserstein disagreement across posterior predictive feature draws",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Assumption-conditioned simulator-active current-selection loop over the compact HH surrogate."
    )
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--bundle-label", default="")
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--n-particles", type=int, default=128)
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--initial-currents", default="0.30")
    ap.add_argument("--candidate-currents", default=",".join(ACTIVE_CURRENT_GRID))
    ap.add_argument("--downsample-points", type=int, default=DEFAULT_DOWNSAMPLE_POINTS)
    ap.add_argument("--temperature", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--trace-length", type=int, default=2000)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--pulse-start-frac", type=float, default=0.10)
    ap.add_argument("--pulse-end-frac", type=float, default=0.90)
    ap.add_argument("--g-scale-range", default="0.2,5.0")
    ap.add_argument("--tau-scale-range", default="0.5,2.0")
    ap.add_argument("--gl-scale-range", default="0.1,5.0")
    ap.add_argument("--current-gain-bounds", default=f"{DEFAULT_CURRENT_GAIN_BOUNDS[0]},{DEFAULT_CURRENT_GAIN_BOUNDS[1]}")
    ap.add_argument("--acquisition-policy", choices=ACQUISITION_POLICIES, default="disagreement")
    ap.add_argument("--wasserstein-projections", type=int, default=16)
    return ap.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise SystemExit("Expected a non-empty comma-separated float list.")
    return values


def parse_pair(text: str) -> tuple[float, float]:
    left, right = [float(x.strip()) for x in text.split(",")]
    if left <= 0.0 or right <= left:
        raise SystemExit(f"Expected positive increasing pair, got {text}")
    return left, right


def sanitize_path_fragment(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def ordered_unique_current_labels(currents: list[float]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in currents:
        label = f"{value:.2f}"
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    if not ordered:
        raise SystemExit("At least one current label is required.")
    return ordered


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    label = sanitize_path_fragment(args.bundle_label) if args.bundle_label else "default"
    stem = f"active_sequential_{args.acquisition_policy}_{label}_{DATE_TAG}"
    return RUNS_ACTIVE_SEQUENTIAL / stem


def config_from_args(args: argparse.Namespace) -> CompactHHSimulationConfig:
    return CompactHHSimulationConfig(
        trace_length=args.trace_length,
        dt_ms=args.dt_ms,
        pulse_start_frac=args.pulse_start_frac,
        pulse_end_frac=args.pulse_end_frac,
        g_scale_range=parse_pair(args.g_scale_range),
        tau_scale_range=parse_pair(args.tau_scale_range),
        gl_scale_range=parse_pair(args.gl_scale_range),
        max_abs_voltage=5000.0,
    )


def simulate_theta_current(theta: np.ndarray, current_label: float, config: CompactHHSimulationConfig) -> np.ndarray:
    raw_params, current_gain = unpack_theta(theta)
    return simulate_compact_hh(
        raw_params,
        current_gain * float(current_label),
        config.trace_length,
        config.dt_ms,
        pulse_start_frac=config.pulse_start_frac,
        pulse_end_frac=config.pulse_end_frac,
        g_scale_range=config.g_scale_range,
        tau_scale_range=config.tau_scale_range,
        gl_scale_range=config.gl_scale_range,
    ).astype(np.float32, copy=False)


def feature_of_trace(trace: np.ndarray, downsample_points: int) -> np.ndarray:
    return trace_feature_matrix(trace.reshape(1, -1), downsample_points, include_summary=True)[0].astype(np.float32, copy=False)


def posterior_weights(
    particle_features: dict[tuple[int, str], np.ndarray],
    observed_features: dict[str, np.ndarray],
    n_particles: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    discrepancies = np.zeros(n_particles, dtype=np.float64)
    observed_currents = sorted(observed_features.keys(), key=float)
    for particle_idx in range(n_particles):
        vals = []
        for current_label in observed_currents:
            diff = particle_features[(particle_idx, current_label)] - observed_features[current_label]
            vals.append(float(np.mean(np.square(diff))))
        discrepancies[particle_idx] = float(np.mean(vals)) if vals else 0.0
    shifted = discrepancies - float(np.min(discrepancies))
    weights = np.exp(-temperature * shifted)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0 or not np.isfinite(weights_sum):
        weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    else:
        weights /= weights_sum
    return weights.astype(np.float64, copy=False), discrepancies


def weighted_predictive_uncertainty(features: np.ndarray, weights: np.ndarray) -> float:
    mean = np.sum(features * weights.reshape(-1, 1), axis=0)
    variance = np.sum(weights.reshape(-1, 1) * np.square(features - mean.reshape(1, -1)), axis=0)
    return float(np.mean(np.sqrt(np.maximum(variance, 0.0))))


def weighted_predictive_disagreement(features: np.ndarray, weights: np.ndarray) -> float:
    sq_norm = np.sum(np.square(features), axis=1)
    pairwise_sq = np.maximum(sq_norm.reshape(-1, 1) + sq_norm.reshape(1, -1) - 2.0 * (features @ features.T), 0.0)
    pairwise_dist = np.sqrt(pairwise_sq, dtype=np.float64)
    weight_matrix = weights.reshape(-1, 1) * weights.reshape(1, -1)
    return float(np.sum(weight_matrix * pairwise_dist))


def weighted_projected_wasserstein(features: np.ndarray, weights: np.ndarray, directions: np.ndarray) -> float:
    projected = features @ directions.T
    weight_matrix = weights.reshape(-1, 1) * weights.reshape(1, -1)
    values = []
    for projection_idx in range(projected.shape[1]):
        delta = np.abs(projected[:, projection_idx].reshape(-1, 1) - projected[:, projection_idx].reshape(1, -1))
        values.append(float(np.sum(weight_matrix * delta)))
    return float(np.mean(values))


def candidate_policy_scores(
    features: np.ndarray,
    weights: np.ndarray,
    wasserstein_directions: np.ndarray,
) -> dict[str, float]:
    feature_bank = np.asarray(features, dtype=np.float64)
    return {
        "uncertainty": weighted_predictive_uncertainty(feature_bank, weights),
        "disagreement": weighted_predictive_disagreement(feature_bank, weights),
        "wasserstein": weighted_projected_wasserstein(feature_bank, weights, wasserstein_directions),
    }


def projection_seed(base_seed: int, episode_id: int, round_idx: int) -> int:
    return int(base_seed + 10007 * episode_id + 379 * round_idx)


def weighted_logtheta_mean(theta_particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(theta_particles * weights.reshape(-1, 1), axis=0).astype(np.float64, copy=False)


def summarize_episode(
    episode_id: int,
    theta_true: np.ndarray,
    theta_particles: np.ndarray,
    weights: np.ndarray,
    selected_currents: list[str],
    prior_std: np.ndarray,
    acquisition_policy: str,
) -> dict[str, object]:
    posterior_mean = weighted_logtheta_mean(theta_particles, weights)
    raw_true, current_gain_true = unpack_theta(theta_true)
    raw_mean, current_gain_mean = unpack_theta(posterior_mean)
    weight_ess = float(1.0 / np.sum(np.square(weights)))
    posterior_std = np.sqrt(np.sum(weights.reshape(-1, 1) * np.square(theta_particles - posterior_mean.reshape(1, -1)), axis=0))
    contraction = posterior_std / np.maximum(prior_std, 1.0e-12)
    return {
        "episode_id": episode_id,
        "acquisition_policy": acquisition_policy,
        "selected_currents": ",".join(selected_currents),
        "ess": weight_ess,
        "mean_abs_log10_error_params": float(np.mean(np.abs(theta_true[:6] - posterior_mean[:6]))),
        "abs_log10_error_current_gain": float(abs(theta_true[6] - posterior_mean[6])),
        "mean_relative_error_params": float(np.mean(np.abs(raw_true - raw_mean) / np.maximum(np.abs(raw_true), 1.0e-12))),
        "relative_error_current_gain": float(abs(current_gain_true - current_gain_mean) / max(abs(current_gain_true), 1.0e-12)),
        "posterior_contraction_mean": float(np.mean(contraction)),
        "posterior_contraction_max": float(np.max(contraction)),
        **{f"true_raw_param_{i+1}": float(raw_true[i]) for i in range(6)},
        **{f"posterior_mean_raw_param_{i+1}": float(raw_mean[i]) for i in range(6)},
        "true_current_gain": float(current_gain_true),
        "posterior_mean_current_gain": float(current_gain_mean),
    }


def validate_args(args: argparse.Namespace, candidate_currents: list[str], initial_currents: list[str]) -> None:
    if args.n_particles <= 0:
        raise SystemExit(f"Expected positive --n-particles, got {args.n_particles}")
    if args.n_episodes <= 0:
        raise SystemExit(f"Expected positive --n-episodes, got {args.n_episodes}")
    if args.n_rounds <= 0:
        raise SystemExit(f"Expected positive --n-rounds, got {args.n_rounds}")
    if args.wasserstein_projections <= 0:
        raise SystemExit(f"Expected positive --wasserstein-projections, got {args.wasserstein_projections}")
    if not candidate_currents:
        raise SystemExit("Expected at least one candidate current.")
    if not initial_currents:
        raise SystemExit("Expected at least one initial current.")


def main() -> None:
    args = parse_args()
    candidate_currents = ordered_unique_current_labels(parse_float_list(args.candidate_currents))
    initial_currents = ordered_unique_current_labels(parse_float_list(args.initial_currents))
    validate_args(args, candidate_currents, initial_currents)

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_args(args)
    current_gain_bounds = parse_pair(args.current_gain_bounds)
    t0 = time.time()

    episode_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for episode_id in range(args.n_episodes):
        theta_true = sample_prior_thetas(1, args.seed + 97 * episode_id, current_gain_bounds)[0]
        theta_particles = sample_prior_thetas(args.n_particles, args.seed + 1009 * episode_id + 1, current_gain_bounds)
        prior_std = np.std(theta_particles, axis=0)
        feature_cache: dict[tuple[int, str], np.ndarray] = {}
        observed_features: dict[str, np.ndarray] = {}
        selected_currents = list(initial_currents)

        for current_label in selected_currents:
            trace = simulate_theta_current(theta_true, float(current_label), config)
            observed_features[current_label] = feature_of_trace(trace, args.downsample_points)

        for particle_idx in range(args.n_particles):
            for current_label in selected_currents:
                trace = simulate_theta_current(theta_particles[particle_idx], float(current_label), config)
                feature_cache[(particle_idx, current_label)] = feature_of_trace(trace, args.downsample_points)

        n_acquisitions = max(0, args.n_rounds - len(selected_currents))
        for round_idx in range(n_acquisitions + 1):
            weights, discrepancies = posterior_weights(feature_cache, observed_features, args.n_particles, args.temperature)
            posterior_mean = weighted_logtheta_mean(theta_particles, weights)
            round_record = {
                "episode_id": episode_id,
                "round_index": round_idx,
                "acquisition_policy": args.acquisition_policy,
                "observed_currents": ",".join(sorted(observed_features.keys(), key=float)),
                "ess": float(1.0 / np.sum(np.square(weights))),
                "mean_discrepancy": float(np.mean(discrepancies)),
                "best_discrepancy": float(np.min(discrepancies)),
                "posterior_mean_log10_current_gain": float(posterior_mean[6]),
            }
            if round_idx == n_acquisitions:
                round_record.update(
                    {
                        "candidate_count": 0,
                        "selected_current": "",
                        "selected_score": 0.0,
                        "selected_uncertainty_score": 0.0,
                        "selected_disagreement_score": 0.0,
                        "selected_wasserstein_score": 0.0,
                    }
                )
                round_rows.append(round_record)
                episode_rows.append(
                    summarize_episode(
                        episode_id,
                        theta_true,
                        theta_particles,
                        weights,
                        selected_currents,
                        prior_std,
                        args.acquisition_policy,
                    )
                )
                break

            remaining_currents = [current for current in candidate_currents if current not in observed_features]
            if not remaining_currents:
                raise RuntimeError("No candidate current remained for acquisition.")

            feature_dim = next(iter(observed_features.values())).shape[0]
            wasserstein_directions = sample_unit_directions(
                feature_dim,
                args.wasserstein_projections,
                seed=projection_seed(args.seed, episode_id, round_idx),
            )
            score_rows = []
            best_current = ""
            best_policy_score = None
            best_score_map: dict[str, float] | None = None

            for current_label in remaining_currents:
                features = np.empty((args.n_particles, feature_dim), dtype=np.float32)
                for particle_idx in range(args.n_particles):
                    key = (particle_idx, current_label)
                    if key not in feature_cache:
                        trace = simulate_theta_current(theta_particles[particle_idx], float(current_label), config)
                        feature_cache[key] = feature_of_trace(trace, args.downsample_points)
                    features[particle_idx] = feature_cache[key]

                score_map = candidate_policy_scores(features, weights, wasserstein_directions)
                policy_score = float(score_map[args.acquisition_policy])
                score_row = {
                    "episode_id": episode_id,
                    "round_index": round_idx,
                    "acquisition_policy": args.acquisition_policy,
                    "candidate_current": current_label,
                    "score": policy_score,
                    "uncertainty_score": float(score_map["uncertainty"]),
                    "disagreement_score": float(score_map["disagreement"]),
                    "wasserstein_score": float(score_map["wasserstein"]),
                    "selected": False,
                }
                score_rows.append(score_row)

                if (
                    best_policy_score is None
                    or policy_score > best_policy_score
                    or (np.isclose(policy_score, best_policy_score) and float(current_label) < float(best_current))
                ):
                    best_policy_score = policy_score
                    best_current = current_label
                    best_score_map = score_map

            if best_policy_score is None or best_score_map is None:
                raise RuntimeError("Failed to score any candidate current for acquisition.")

            for row in score_rows:
                row["selected"] = row["candidate_current"] == best_current
            candidate_rows.extend(score_rows)

            round_record.update(
                {
                    "candidate_count": len(remaining_currents),
                    "selected_current": best_current,
                    "selected_score": float(best_policy_score),
                    "selected_uncertainty_score": float(best_score_map["uncertainty"]),
                    "selected_disagreement_score": float(best_score_map["disagreement"]),
                    "selected_wasserstein_score": float(best_score_map["wasserstein"]),
                }
            )
            round_rows.append(round_record)

            trace = simulate_theta_current(theta_true, float(best_current), config)
            observed_features[best_current] = feature_of_trace(trace, args.downsample_points)
            selected_currents.append(best_current)

    episode_rows.sort(key=lambda row: int(row["episode_id"]))
    round_rows.sort(key=lambda row: (int(row["episode_id"]), int(row["round_index"])))
    candidate_rows.sort(key=lambda row: (int(row["episode_id"]), int(row["round_index"]), -float(row["score"]), float(row["candidate_current"])))

    write_csv(save_dir / "active_sequential_episode_summary.csv", episode_rows)
    write_csv(save_dir / "active_sequential_round_history.csv", round_rows)
    write_csv(save_dir / "active_sequential_candidate_scores.csv", candidate_rows)

    mapped_example = mapped_params_from_theta(
        theta_particles[0],
        g_scale_range=config.g_scale_range,
        tau_scale_range=config.tau_scale_range,
        gl_scale_range=config.gl_scale_range,
    )
    manifest = {
        "assumption_conditioned": True,
        "provenance_recovered": False,
        "bundle_integrated": True,
        "method_family": "simulator_active_surrogate_current_selection",
        "acquisition_policy": args.acquisition_policy,
        "acquisition_rule": ACQUISITION_POLICY_DESCRIPTIONS[args.acquisition_policy],
        "available_acquisition_policies": {name: ACQUISITION_POLICY_DESCRIPTIONS[name] for name in ACQUISITION_POLICIES},
        "literature_alignment": [
            "square-pulse current clamp surrogate",
            "compact HH particle bank",
            "sequential current acquisition under posterior predictive scoring",
            ACQUISITION_POLICY_DESCRIPTIONS[args.acquisition_policy],
        ],
        "save_dir": str(save_dir),
        "bundle_label": args.bundle_label,
        "n_episodes": args.n_episodes,
        "n_particles": args.n_particles,
        "n_rounds": args.n_rounds,
        "initial_currents": initial_currents,
        "candidate_currents": candidate_currents,
        "temperature": args.temperature,
        "trace_length": args.trace_length,
        "dt_ms": args.dt_ms,
        "pulse_start_frac": args.pulse_start_frac,
        "pulse_end_frac": args.pulse_end_frac,
        "downsample_points": args.downsample_points,
        "wasserstein_projections": args.wasserstein_projections,
        "g_scale_range": config.g_scale_range,
        "tau_scale_range": config.tau_scale_range,
        "gl_scale_range": config.gl_scale_range,
        "current_gain_bounds": current_gain_bounds,
        "runtime_sec": float(time.time() - t0),
        "aggregate_metrics": {
            "mean_abs_log10_error_params_mean": float(np.mean([row["mean_abs_log10_error_params"] for row in episode_rows])),
            "mean_relative_error_params_mean": float(np.mean([row["mean_relative_error_params"] for row in episode_rows])),
            "posterior_contraction_mean_mean": float(np.mean([row["posterior_contraction_mean"] for row in episode_rows])),
            "ess_mean": float(np.mean([row["ess"] for row in episode_rows])),
        },
        "example_mapped_params_first_particle": mapped_example,
    }
    (save_dir / "active_sequential_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report_lines = [
        "# HH Track4 Assumption-Conditioned Active Sequential Design",
        "",
        f"Workspace-clock generated: `{DATE_TAG}`",
        "",
        "## Scope",
        "",
        "This run is an assumption-conditioned simulator-active experiment over the compact HH surrogate. It does not recover Track4 provenance. It demonstrates the workflow infrastructure needed for simulator-active current selection.",
        "",
        "## Aggregate Metrics",
        "",
        f"- mean abs log10 parameter error: `{manifest['aggregate_metrics']['mean_abs_log10_error_params_mean']:.6f}`",
        f"- mean relative parameter error: `{manifest['aggregate_metrics']['mean_relative_error_params_mean']:.6f}`",
        f"- mean posterior contraction: `{manifest['aggregate_metrics']['posterior_contraction_mean_mean']:.6f}`",
        f"- mean effective sample size: `{manifest['aggregate_metrics']['ess_mean']:.6f}`",
        "",
        "## Acquisition Policy",
        "",
        f"- selected policy: `{args.acquisition_policy}`",
        f"- policy definition: {ACQUISITION_POLICY_DESCRIPTIONS[args.acquisition_policy]}",
        f"- available policies: `{', '.join(ACQUISITION_POLICIES)}`",
        f"- Wasserstein projection count: `{args.wasserstein_projections}`",
        "",
        "## Guardrail",
        "",
        "This is a surrogate workflow artifact. It is not evidence that the same optimal current schedule would be preferred under the unrecovered Track4 simulator.",
        "",
    ]
    (save_dir / "active_sequential_report.md").write_text("\n".join(report_lines))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
