#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_hh_track4_assumption_conditioned_compact_hh_sandbox_20260424 import (  # noqa: E402
    map_surrogate_params,
    simulate_compact_hh,
    summary12,
)


CURRENT_LABELS = ("0.1", "0.2", "0.3", "0.4", "0.5")
RAW_PARAM_LOW = np.asarray([0.05, 0.2, 5.0, 0.05, 0.2, 5.0], dtype=np.float64)
RAW_PARAM_HIGH = np.asarray([10000.0, 100.0, 1000.0, 10000.0, 100.0, 1000.0], dtype=np.float64)


@dataclass(frozen=True)
class CompactHHSimulationConfig:
    trace_length: int
    dt_ms: float
    pulse_start_frac: float
    pulse_end_frac: float
    g_scale_range: tuple[float, float]
    tau_scale_range: tuple[float, float]
    gl_scale_range: tuple[float, float]
    max_abs_voltage: float


@dataclass(frozen=True)
class SimulateThetaTask:
    theta: tuple[float, ...]
    currents: tuple[str, ...]
    config: CompactHHSimulationConfig


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def theta_bounds(current_gain_bounds: tuple[float, float]) -> np.ndarray:
    low = np.concatenate([np.log10(RAW_PARAM_LOW), [math.log10(current_gain_bounds[0])]])
    high = np.concatenate([np.log10(RAW_PARAM_HIGH), [math.log10(current_gain_bounds[1])]])
    return np.stack([low, high], axis=1).astype(np.float64, copy=False)


def sample_prior_thetas(n: int, seed: int, current_gain_bounds: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bounds = theta_bounds(current_gain_bounds)
    u = rng.uniform(size=(n, bounds.shape[0]))
    return (bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])).astype(np.float64, copy=False)


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, float]:
    theta = np.asarray(theta, dtype=np.float64)
    raw_params = np.power(10.0, theta[:6], dtype=np.float64)
    current_gain = float(10.0 ** theta[6])
    return raw_params, current_gain


def pack_theta(raw_params: np.ndarray, current_gain: float) -> np.ndarray:
    raw_params = np.asarray(raw_params, dtype=np.float64)
    return np.concatenate([np.log10(raw_params), [math.log10(current_gain)]], axis=0).astype(np.float64, copy=False)


def clip_theta(theta: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    return np.clip(theta, bounds[:, 0], bounds[:, 1]).astype(np.float64, copy=False)


def mapped_params_from_theta(
    theta: np.ndarray,
    *,
    g_scale_range: tuple[float, float],
    tau_scale_range: tuple[float, float],
    gl_scale_range: tuple[float, float],
) -> dict[str, float]:
    raw_params, _ = unpack_theta(theta)
    mapped = map_surrogate_params(
        raw_params,
        g_scale_range=g_scale_range,
        tau_scale_range=tau_scale_range,
        gl_scale_range=gl_scale_range,
    )
    return {key: float(value) for key, value in mapped.items()}


def _downsample_index(length: int, n_points: int) -> np.ndarray:
    if n_points <= 0 or n_points >= length:
        return np.arange(length, dtype=np.int64)
    return np.round(np.linspace(0, length - 1, num=n_points)).astype(np.int64)


def standardized_trace_block(traces: np.ndarray, downsample_points: int) -> np.ndarray:
    traces = np.asarray(traces, dtype=np.float32)
    if traces.ndim == 1:
        traces = traces.reshape(1, -1)
    centered = traces - np.mean(traces, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    scale = np.where(scale <= 1.0e-6, 1.0, scale)
    standardized = centered / scale
    index = _downsample_index(standardized.shape[1], downsample_points)
    return standardized[:, index].astype(np.float32, copy=False)


def trace_feature_matrix(traces: np.ndarray, downsample_points: int, *, include_summary: bool = True) -> np.ndarray:
    traces = np.asarray(traces, dtype=np.float32)
    if traces.ndim == 1:
        traces = traces.reshape(1, -1)
    parts: list[np.ndarray] = []
    coarse = standardized_trace_block(traces, downsample_points)
    if coarse.shape[1] > 0:
        parts.append(coarse)
    if include_summary:
        parts.append(summary12(traces))
    if not parts:
        raise ValueError("trace_feature_matrix requires at least one feature block")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def bundle_feature_vector(bundle: np.ndarray, downsample_points: int, *, include_summary: bool = True) -> np.ndarray:
    bundle = np.asarray(bundle, dtype=np.float32)
    if bundle.ndim != 2:
        raise ValueError(f"Expected bundle shape (n_currents, trace_length), got {bundle.shape}")
    rows = [trace_feature_matrix(bundle[i : i + 1], downsample_points, include_summary=include_summary)[0] for i in range(bundle.shape[0])]
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def observed_feature_matrices(
    observed_bundle: dict[str, np.ndarray],
    *,
    currents: tuple[str, ...] = CURRENT_LABELS,
    downsample_points: int,
    include_summary: bool = True,
) -> dict[str, np.ndarray]:
    return {
        curr: trace_feature_matrix(observed_bundle[curr], downsample_points, include_summary=include_summary)
        for curr in currents
    }


def sample_unit_directions(dim: int, n_proj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_proj, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (directions / norms).astype(np.float64, copy=False)


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, *, n_proj: int, seed: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected matching 2D arrays, got {x.shape} and {y.shape}")
    directions = sample_unit_directions(x.shape[1], n_proj, seed)
    values = []
    for direction in directions:
        values.append(float(wasserstein_distance(x @ direction, y @ direction)))
    return float(np.mean(values))


def per_current_sliced_wasserstein(
    observed_features: dict[str, np.ndarray],
    simulated_bundle: np.ndarray,
    *,
    currents: tuple[str, ...] = CURRENT_LABELS,
    downsample_points: int,
    include_summary: bool = True,
    n_proj: int,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    simulated_bundle = np.asarray(simulated_bundle, dtype=np.float32)
    rows: list[dict[str, object]] = []
    values = []
    for current_idx, curr in enumerate(currents):
        simulated_features = trace_feature_matrix(
            simulated_bundle[current_idx : current_idx + 1],
            downsample_points,
            include_summary=include_summary,
        )
        distance = sliced_wasserstein_distance(
            observed_features[curr],
            simulated_features,
            n_proj=n_proj,
            seed=seed + 1009 * current_idx,
        )
        rows.append(
            {
                "current_label": curr,
                "sliced_wasserstein": float(distance),
                "observed_feature_rows": int(observed_features[curr].shape[0]),
                "feature_dim": int(observed_features[curr].shape[1]),
            }
        )
        values.append(float(distance))
    return float(np.mean(values)), rows


def simulate_theta_bundle(theta: np.ndarray, currents: tuple[str, ...], config: CompactHHSimulationConfig) -> tuple[np.ndarray, bool, float]:
    invalid_bundle = np.full((len(currents), config.trace_length), np.nan, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float64)
    safe_log10_min = math.log10(np.finfo(np.float64).tiny)
    safe_log10_max = math.log10(np.finfo(np.float64).max)
    if theta.ndim != 1 or theta.size < 7 or not np.isfinite(theta).all():
        return invalid_bundle, False, float("inf")
    if np.any(theta < safe_log10_min) or np.any(theta > safe_log10_max):
        return invalid_bundle, False, float("inf")
    try:
        raw_params, current_gain = unpack_theta(theta)
        traces = []
        for curr in currents:
            trace = simulate_compact_hh(
                raw_params,
                current_gain * float(curr),
                config.trace_length,
                config.dt_ms,
                pulse_start_frac=config.pulse_start_frac,
                pulse_end_frac=config.pulse_end_frac,
                g_scale_range=config.g_scale_range,
                tau_scale_range=config.tau_scale_range,
                gl_scale_range=config.gl_scale_range,
            )
            traces.append(trace)
    except (ArithmeticError, ValueError):
        return invalid_bundle, False, float("inf")
    bundle = np.stack(traces, axis=0).astype(np.float32, copy=False)
    is_finite = bool(np.isfinite(bundle).all())
    if not is_finite:
        return bundle, False, float("inf")
    max_abs_voltage = float(np.max(np.abs(bundle)))
    is_valid = max_abs_voltage <= config.max_abs_voltage
    return bundle, is_valid, max_abs_voltage


def simulate_theta_task(task: SimulateThetaTask) -> tuple[np.ndarray, bool, float]:
    theta = np.asarray(task.theta, dtype=np.float64)
    return simulate_theta_bundle(theta, task.currents, task.config)


def simulate_theta_batch(
    thetas: np.ndarray,
    *,
    currents: tuple[str, ...],
    config: CompactHHSimulationConfig,
    workers: int,
    chunksize: int = 1,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    tasks = [
        SimulateThetaTask(tuple(np.asarray(theta, dtype=np.float64).tolist()), currents, config)
        for theta in np.asarray(thetas, dtype=np.float64)
    ]
    if workers == 1:
        outputs = [simulate_theta_task(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            outputs = pool.map(simulate_theta_task, tasks, chunksize=max(1, chunksize))
    bundles = [bundle for bundle, _, _ in outputs]
    valid = np.asarray([flag for _, flag, _ in outputs], dtype=bool)
    max_abs = np.asarray([value for _, _, value in outputs], dtype=np.float64)
    return bundles, valid, max_abs
