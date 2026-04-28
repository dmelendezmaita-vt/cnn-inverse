#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
TRACK4_FEATURES = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
    / "y"
)
SIMULATION_INVALID_VOLTAGE_ABS = 5000.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run an assumption-conditioned compact HH sandbox against Track4 traces."
    )
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n-observed", type=int, default=256)
    ap.add_argument("--n-param-samples", type=int, default=128)
    ap.add_argument("--trace-length", type=int, default=2000)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--current-gains", default="20,40,80")
    ap.add_argument("--pulse-start-frac", type=float, default=0.10)
    ap.add_argument("--pulse-end-frac", type=float, default=0.90)
    ap.add_argument("--g-scale-range", default="0.2,5.0")
    ap.add_argument("--tau-scale-range", default="0.5,2.0")
    ap.add_argument("--gl-scale-range", default="0.1,5.0")
    ap.add_argument("--seed", type=int, default=3201)
    ap.add_argument("--workers", type=int, default=max(1, min(18, os.cpu_count() or 1)))
    return ap.parse_args()


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1.0e-12)
    return out


def safe_div_scalar(num: float, den: float) -> float:
    return 0.0 if abs(den) <= 1.0e-12 else num / den


def exp_scalar_clipped(x: float, clip: float = 80.0) -> float:
    if x > clip:
        x = clip
    elif x < -clip:
        x = -clip
    return math.exp(x)


def alpha_n(v: np.ndarray) -> np.ndarray:
    x = v + 55.0
    return safe_div(0.01 * x, 1.0 - np.exp(-x / 10.0))


def beta_n(v: np.ndarray) -> np.ndarray:
    return 0.125 * np.exp(-(v + 65.0) / 80.0)


def alpha_m(v: np.ndarray) -> np.ndarray:
    x = v + 40.0
    return safe_div(0.1 * x, 1.0 - np.exp(-x / 10.0))


def beta_m(v: np.ndarray) -> np.ndarray:
    return 4.0 * np.exp(-(v + 65.0) / 18.0)


def alpha_h(v: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(-(v + 65.0) / 20.0)


def beta_h(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))


def alpha_n_scalar(v: float) -> float:
    x = v + 55.0
    return safe_div_scalar(0.01 * x, 1.0 - exp_scalar_clipped(-x / 10.0))


def beta_n_scalar(v: float) -> float:
    return 0.125 * exp_scalar_clipped(-(v + 65.0) / 80.0)


def alpha_m_scalar(v: float) -> float:
    x = v + 40.0
    return safe_div_scalar(0.1 * x, 1.0 - exp_scalar_clipped(-x / 10.0))


def beta_m_scalar(v: float) -> float:
    return 4.0 * exp_scalar_clipped(-(v + 65.0) / 18.0)


def alpha_h_scalar(v: float) -> float:
    return 0.07 * exp_scalar_clipped(-(v + 65.0) / 20.0)


def beta_h_scalar(v: float) -> float:
    return 1.0 / (1.0 + exp_scalar_clipped(-(v + 35.0) / 10.0))


def make_square_pulse(length: int, amplitude: float, start_frac: float, end_frac: float) -> np.ndarray:
    pulse = np.zeros(length, dtype=np.float32)
    lo = int(start_frac * length)
    hi = int(end_frac * length)
    lo = max(0, min(lo, length - 1))
    hi = max(lo + 1, min(hi, length))
    pulse[lo:hi] = amplitude
    return pulse


def map_raw_to_positive_interval(raw_value: float, raw_lo: float, raw_hi: float, out_lo: float, out_hi: float) -> float:
    raw_log = math.log10(raw_value)
    lo_log = math.log10(raw_lo)
    hi_log = math.log10(raw_hi)
    t = (raw_log - lo_log) / (hi_log - lo_log)
    out_log = math.log10(out_lo) + t * (math.log10(out_hi) - math.log10(out_lo))
    return float(10 ** out_log)


def parse_pair(text: str) -> tuple[float, float]:
    left, right = [float(x.strip()) for x in text.split(",")]
    if left <= 0.0 or right <= 0.0 or left >= right:
        raise ValueError(f"Expected positive increasing pair, got {text}")
    return left, right


def map_surrogate_params(
    raw_params: np.ndarray,
    *,
    g_scale_range: tuple[float, float],
    tau_scale_range: tuple[float, float],
    gl_scale_range: tuple[float, float],
) -> dict[str, float]:
    return {
        "g_na": 120.0 * map_raw_to_positive_interval(raw_params[0], 0.05, 10000.0, g_scale_range[0], g_scale_range[1]),
        "tau_m_scale": map_raw_to_positive_interval(raw_params[1], 0.2, 100.0, tau_scale_range[0], tau_scale_range[1]),
        "tau_h_scale": map_raw_to_positive_interval(raw_params[2], 5.0, 1000.0, tau_scale_range[0], tau_scale_range[1]),
        "g_k": 36.0 * map_raw_to_positive_interval(raw_params[3], 0.05, 10000.0, g_scale_range[0], g_scale_range[1]),
        "tau_n_scale": map_raw_to_positive_interval(raw_params[4], 0.2, 100.0, tau_scale_range[0], tau_scale_range[1]),
        "g_l": 0.3 * map_raw_to_positive_interval(raw_params[5], 5.0, 1000.0, gl_scale_range[0], gl_scale_range[1]),
    }


def simulate_compact_hh(
    raw_params: np.ndarray,
    current_amplitude: float,
    length: int,
    dt_ms: float,
    *,
    pulse_start_frac: float,
    pulse_end_frac: float,
    g_scale_range: tuple[float, float],
    tau_scale_range: tuple[float, float],
    gl_scale_range: tuple[float, float],
) -> np.ndarray:
    params = map_surrogate_params(
        raw_params,
        g_scale_range=g_scale_range,
        tau_scale_range=tau_scale_range,
        gl_scale_range=gl_scale_range,
    )
    g_na = params["g_na"]
    g_k = params["g_k"]
    g_l = params["g_l"]
    tau_m_scale = params["tau_m_scale"]
    tau_h_scale = params["tau_h_scale"]
    tau_n_scale = params["tau_n_scale"]

    e_na = 50.0
    e_k = -77.0
    e_l = -54.387
    c_m = 1.0

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (g_na, g_k, g_l, tau_m_scale, tau_h_scale, tau_n_scale, dt_ms)
    ) or not math.isfinite(current_amplitude):
        return np.full(length, np.nan, dtype=np.float32)

    current = make_square_pulse(length, current_amplitude, pulse_start_frac, pulse_end_frac)
    v = np.empty(length, dtype=np.float32)
    v[0] = -65.0

    a_m0 = alpha_m_scalar(float(v[0]))
    b_m0 = beta_m_scalar(float(v[0]))
    a_h0 = alpha_h_scalar(float(v[0]))
    b_h0 = beta_h_scalar(float(v[0]))
    a_n0 = alpha_n_scalar(float(v[0]))
    b_n0 = beta_n_scalar(float(v[0]))
    if not all(math.isfinite(value) for value in (a_m0, b_m0, a_h0, b_h0, a_n0, b_n0)):
        v.fill(np.nan)
        return v
    m = a_m0 / (a_m0 + b_m0)
    h = a_h0 / (a_h0 + b_h0)
    n = a_n0 / (a_n0 + b_n0)

    dt = dt_ms
    for i in range(1, length):
        vv = float(v[i - 1])
        if not math.isfinite(vv) or abs(vv) > SIMULATION_INVALID_VOLTAGE_ABS:
            v[i:] = np.nan
            break
        a_m = alpha_m_scalar(vv)
        b_m = beta_m_scalar(vv)
        a_h = alpha_h_scalar(vv)
        b_h = beta_h_scalar(vv)
        a_n = alpha_n_scalar(vv)
        b_n = beta_n_scalar(vv)
        if not all(math.isfinite(value) for value in (a_m, b_m, a_h, b_h, a_n, b_n)):
            v[i:] = np.nan
            break

        dm = (a_m * (1.0 - m) - b_m * m) / tau_m_scale
        dh = (a_h * (1.0 - h) - b_h * h) / tau_h_scale
        dn = (a_n * (1.0 - n) - b_n * n) / tau_n_scale
        if not all(math.isfinite(value) for value in (dm, dh, dn)):
            v[i:] = np.nan
            break
        m += dt * dm
        h += dt * dh
        n += dt * dn
        m = min(max(m, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        n = min(max(n, 0.0), 1.0)

        i_na = g_na * (m ** 3) * h * (vv - e_na)
        i_k = g_k * (n ** 4) * (vv - e_k)
        i_l = g_l * (vv - e_l)
        if not all(math.isfinite(value) for value in (i_na, i_k, i_l)):
            v[i:] = np.nan
            break
        dv = (current[i - 1] - i_na - i_k - i_l) / c_m
        if not math.isfinite(dv):
            v[i:] = np.nan
            break
        next_v = vv + dt * dv
        if not math.isfinite(next_v) or abs(next_v) > SIMULATION_INVALID_VOLTAGE_ABS:
            v[i:] = np.nan
            break
        v[i] = next_v
    return v


def summary12(traces: np.ndarray) -> np.ndarray:
    diffs = np.diff(traces, axis=1)
    return np.column_stack(
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
    ).astype(np.float32, copy=False)


def summary_overlap_score(obs_summary: np.ndarray, sim_summary: np.ndarray) -> tuple[float, float]:
    obs_lo = np.quantile(obs_summary, 0.05, axis=0)
    obs_hi = np.quantile(obs_summary, 0.95, axis=0)
    obs_med = np.median(obs_summary, axis=0)

    sim_lo = np.quantile(sim_summary, 0.05, axis=0)
    sim_hi = np.quantile(sim_summary, 0.95, axis=0)
    sim_med = np.median(sim_summary, axis=0)

    in_interval = ((obs_med >= sim_lo) & (obs_med <= sim_hi)).astype(np.float32)
    obs_iqr = np.maximum(np.quantile(obs_summary, 0.75, axis=0) - np.quantile(obs_summary, 0.25, axis=0), 1.0e-6)
    median_abs_z = np.mean(np.abs(obs_med - sim_med) / obs_iqr)
    return float(np.mean(in_interval)), float(median_abs_z)


def nearest_trace_rmse(obs_trace: np.ndarray, sim_traces: np.ndarray) -> float:
    obs = obs_trace.astype(np.float32, copy=False)
    obs_center = obs - np.mean(obs)
    obs_scale = max(float(np.std(obs_center)), 1.0e-6)
    obs_norm = obs_center / obs_scale

    sims = sim_traces.astype(np.float32, copy=False)
    sims_center = sims - np.mean(sims, axis=1, keepdims=True)
    sims_scale = np.maximum(np.std(sims_center, axis=1, keepdims=True), 1.0e-6)
    sims_norm = sims_center / sims_scale
    rmse = np.sqrt(np.mean((sims_norm - obs_norm.reshape(1, -1)) ** 2, axis=1))
    return float(np.min(rmse))


def load_observed(curr: str, n_observed: int, trace_length: int, seed: int) -> np.ndarray:
    path = TRACK4_FEATURES / f"concatenated_data_{curr}_curr.npy"
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(15000, 400000))
    rng = np.random.default_rng(seed + int(float(curr) * 1000))
    idx = rng.choice(arr.shape[0], size=n_observed, replace=False)
    return np.asarray(arr[idx, :trace_length], dtype=np.float32)


@dataclass(frozen=True)
class Task:
    current_label: str
    current_gain: float
    raw_params: np.ndarray
    trace_length: int
    dt_ms: float
    pulse_start_frac: float
    pulse_end_frac: float
    g_scale_range: tuple[float, float]
    tau_scale_range: tuple[float, float]
    gl_scale_range: tuple[float, float]


def run_task(task: Task) -> np.ndarray:
    current_amp = task.current_gain * float(task.current_label)
    return simulate_compact_hh(
        task.raw_params,
        current_amp,
        task.trace_length,
        task.dt_ms,
        pulse_start_frac=task.pulse_start_frac,
        pulse_end_frac=task.pulse_end_frac,
        g_scale_range=task.g_scale_range,
        tau_scale_range=task.tau_scale_range,
        gl_scale_range=task.gl_scale_range,
    )


def raw_param_draws(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = np.asarray([0.05, 0.2, 5.0, 0.05, 0.2, 5.0], dtype=np.float64)
    high = np.asarray([10000.0, 100.0, 1000.0, 10000.0, 100.0, 1000.0], dtype=np.float64)
    u = rng.uniform(size=(n, 6))
    draws = 10 ** (np.log10(low) + u * (np.log10(high) - np.log10(low)))
    return draws.astype(np.float64, copy=False)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    currents = ["0.1", "0.2", "0.3", "0.4", "0.5"]
    current_gains = [float(x.strip()) for x in args.current_gains.split(",") if x.strip()]
    g_scale_range = parse_pair(args.g_scale_range)
    tau_scale_range = parse_pair(args.tau_scale_range)
    gl_scale_range = parse_pair(args.gl_scale_range)
    observed = {curr: load_observed(curr, args.n_observed, args.trace_length, args.seed) for curr in currents}
    observed_summary = {curr: summary12(traces) for curr, traces in observed.items()}

    exemplar_rows = {curr: observed[curr][0] for curr in currents}

    rng_draws = raw_param_draws(args.n_param_samples, args.seed)

    tasks: list[Task] = []
    task_keys: list[tuple[float, str, int]] = []
    for gain in current_gains:
        for i, raw_params in enumerate(rng_draws):
            for curr in currents:
                tasks.append(Task(
                    curr,
                    gain,
                    raw_params,
                    args.trace_length,
                    args.dt_ms,
                    args.pulse_start_frac,
                    args.pulse_end_frac,
                    g_scale_range,
                    tau_scale_range,
                    gl_scale_range,
                ))
                task_keys.append((gain, curr, i))

    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        simulated = pool.map(run_task, tasks, chunksize=8)

    sim_by_gain_curr: dict[tuple[float, str], list[np.ndarray]] = {}
    for key, trace in zip(task_keys, simulated):
        gain, curr, _ = key
        sim_by_gain_curr.setdefault((gain, curr), []).append(trace)

    rows = []
    summary_json: dict[str, object] = {
        "assumption_conditioned": True,
        "model_family": "compact_single_compartment_hh_sandbox",
        "trace_length": args.trace_length,
        "dt_ms": args.dt_ms,
        "current_gains": current_gains,
        "n_observed_per_current": args.n_observed,
        "n_param_samples": args.n_param_samples,
        "workers": args.workers,
        "pulse_start_frac": args.pulse_start_frac,
        "pulse_end_frac": args.pulse_end_frac,
        "g_scale_range": g_scale_range,
        "tau_scale_range": tau_scale_range,
        "gl_scale_range": gl_scale_range,
        "currents": currents,
        "per_gain": {},
    }

    best_gain = None
    best_score = None
    for gain in current_gains:
        gain_records = []
        interval_scores = []
        median_zs = []
        rmses = []
        for curr in currents:
            sim_traces = np.asarray(sim_by_gain_curr[(gain, curr)], dtype=np.float32)
            sim_summary = summary12(sim_traces)
            interval_score, median_abs_z = summary_overlap_score(observed_summary[curr], sim_summary)
            exemplar_rmse = nearest_trace_rmse(exemplar_rows[curr], sim_traces)
            interval_scores.append(interval_score)
            median_zs.append(median_abs_z)
            rmses.append(exemplar_rmse)
            row = {
                "current_gain": gain,
                "current_label": curr,
                "summary_interval_coverage": interval_score,
                "summary_median_abs_z": median_abs_z,
                "exemplar_best_trace_rmse_z": exemplar_rmse,
                "observed_mean": float(np.mean(observed[curr])),
                "observed_std": float(np.std(observed[curr])),
                "observed_min": float(np.min(observed[curr])),
                "observed_max": float(np.max(observed[curr])),
                "simulated_mean": float(np.mean(sim_traces)),
                "simulated_std": float(np.std(sim_traces)),
                "simulated_min": float(np.min(sim_traces)),
                "simulated_max": float(np.max(sim_traces)),
            }
            rows.append(row)
            gain_records.append(row)

        overall_interval = float(np.mean(interval_scores))
        overall_median_abs_z = float(np.mean(median_zs))
        overall_rmse = float(np.mean(rmses))
        compatibility_score = float(overall_interval - 0.25 * overall_median_abs_z - 0.10 * overall_rmse)
        summary_json["per_gain"][str(gain)] = {
            "overall_summary_interval_coverage": overall_interval,
            "overall_summary_median_abs_z": overall_median_abs_z,
            "overall_exemplar_best_trace_rmse_z": overall_rmse,
            "compatibility_score": compatibility_score,
            "per_current_rows": gain_records,
        }
        if best_score is None or compatibility_score > best_score:
            best_score = compatibility_score
            best_gain = gain

    summary_json["best_gain"] = best_gain
    summary_json["best_gain_metrics"] = summary_json["per_gain"][str(best_gain)]

    fieldnames = list(rows[0].keys())
    with (save_dir / "assumption_conditioned_compact_hh_summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    (save_dir / "assumption_conditioned_compact_hh_summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")
    (save_dir / "assumption_conditioned_compact_hh_run_manifest.json").write_text(
        json.dumps(
            {
                "assumption_conditioned": True,
                "not_dataset_faithful": True,
                "purpose": "compact HH sandbox compatibility sweep",
                "currents": currents,
                "current_gains": current_gains,
                "n_observed": args.n_observed,
                "n_param_samples": args.n_param_samples,
                "trace_length": args.trace_length,
                "dt_ms": args.dt_ms,
                "workers": args.workers,
                "seed": args.seed,
                "pulse_start_frac": args.pulse_start_frac,
                "pulse_end_frac": args.pulse_end_frac,
                "g_scale_range": g_scale_range,
                "tau_scale_range": tau_scale_range,
                "gl_scale_range": gl_scale_range,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
