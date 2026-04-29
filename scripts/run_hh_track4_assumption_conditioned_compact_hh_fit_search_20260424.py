#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from hh_repo_utils import resolve_hh_feature_root

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_hh_track4_assumption_conditioned_compact_hh_sandbox_20260424 import (  # noqa: E402
    map_surrogate_params,
    parse_pair,
    simulate_compact_hh,
    summary12,
)


RAW_PARAM_BOUNDS = (
    (0.05, 10000.0),
    (0.2, 100.0),
    (5.0, 1000.0),
    (0.05, 10000.0),
    (0.2, 100.0),
    (5.0, 1000.0),
)
SUMMARY_SCALE_FLOOR = np.asarray(
    [5.0, 5.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 0.25, 2.0, 0.05, 0.05],
    dtype=np.float64,
)
CURRENT_LABELS = ("0.1", "0.2", "0.3", "0.4", "0.5")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fit an assumption-conditioned compact HH sandbox to small Track4 exemplar bundles."
    )
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n-exemplars", type=int, default=3)
    ap.add_argument("--exemplar-indices", default="")
    ap.add_argument("--trace-length", type=int, default=2000)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--pulse-start-frac", type=float, default=0.10)
    ap.add_argument("--pulse-end-frac", type=float, default=0.90)
    ap.add_argument("--g-scale-range", default="0.2,5.0")
    ap.add_argument("--tau-scale-range", default="0.5,2.0")
    ap.add_argument("--gl-scale-range", default="0.1,5.0")
    ap.add_argument("--current-gain-bounds", default="2.0,80.0")
    ap.add_argument("--data-dir", default="concatenated_data.tar.gz")
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--maxiter", type=int, default=12)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--mutation-min", type=float, default=0.5)
    ap.add_argument("--mutation-max", type=float, default=1.0)
    ap.add_argument("--recombination", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3201)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--max-abs-voltage", type=float, default=5000.0)
    ap.add_argument("--summary-weight", type=float, default=0.35)
    ap.add_argument("--range-weight", type=float, default=0.20)
    ap.add_argument("--mean-weight", type=float, default=0.10)
    ap.add_argument("--std-weight", type=float, default=0.10)
    ap.add_argument("--trace-weight", type=float, default=1.0)
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(text: str) -> list[int]:
    if not text.strip():
        return []
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if any(value < 0 or value >= 15000 for value in values):
        raise ValueError("exemplar indices must be between 0 and 14999")
    if len(values) != len(set(values)):
        raise ValueError("exemplar indices must be unique")
    return values


def choose_exemplar_indices(n_exemplars: int, exemplar_indices: str, seed: int) -> list[int]:
    explicit = parse_indices(exemplar_indices)
    if explicit:
        return explicit
    rng = np.random.default_rng(seed)
    picked = rng.choice(15000, size=n_exemplars, replace=False)
    return sorted(int(x) for x in picked)


def load_trace_bundle(indices: list[int], trace_length: int, *, data_dir: str, data_prefix: str) -> dict[str, np.ndarray]:
    feature_root = resolve_hh_feature_root(data_dir, data_prefix=data_prefix)
    bundle: dict[str, np.ndarray] = {}
    for curr in CURRENT_LABELS:
        path = feature_root / f"{data_prefix}_{curr}_curr.npy"
        arr = np.memmap(path, dtype=np.float32, mode="r", shape=(15000, 400000))
        bundle[curr] = np.asarray(arr[indices, :trace_length], dtype=np.float32)
    return bundle


def zscore_trace_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs_center = observed - float(np.mean(observed))
    sim_center = simulated - float(np.mean(simulated))
    obs_scale = max(float(np.std(obs_center)), 1.0e-6)
    sim_scale = max(float(np.std(sim_center)), 1.0e-6)
    obs_norm = obs_center / obs_scale
    sim_norm = sim_center / sim_scale
    return float(np.sqrt(np.mean(np.square(obs_norm - sim_norm))))


def trace_metrics(observed: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    obs_summary = summary12(observed.reshape(1, -1))[0].astype(np.float64, copy=False)
    sim_summary = summary12(simulated.reshape(1, -1))[0].astype(np.float64, copy=False)
    summary_scale = np.maximum(np.abs(obs_summary), SUMMARY_SCALE_FLOOR)
    obs_std = max(float(np.std(observed)), 1.0)
    obs_range = max(float(np.max(observed) - np.min(observed)), 1.0)
    return {
        "trace_rmse_z": zscore_trace_rmse(observed, simulated),
        "summary_rel_l1": float(np.mean(np.abs(sim_summary - obs_summary) / summary_scale)),
        "mean_rel_error": float(abs(float(np.mean(simulated)) - float(np.mean(observed))) / obs_std),
        "std_rel_error": float(abs(float(np.std(simulated)) - float(np.std(observed))) / obs_std),
        "range_rel_error": float(
            abs((float(np.max(simulated)) - float(np.min(simulated))) - (float(np.max(observed)) - float(np.min(observed))))
            / obs_range
        ),
        "observed_mean": float(np.mean(observed)),
        "observed_std": float(np.std(observed)),
        "observed_min": float(np.min(observed)),
        "observed_max": float(np.max(observed)),
        "simulated_mean": float(np.mean(simulated)),
        "simulated_std": float(np.std(simulated)),
        "simulated_min": float(np.min(simulated)),
        "simulated_max": float(np.max(simulated)),
    }


def unpack_log_params(theta: np.ndarray) -> tuple[np.ndarray, float]:
    raw_params = np.power(10.0, theta[:6], dtype=np.float64)
    current_gain = float(10.0 ** theta[6])
    return raw_params, current_gain


def evaluate_candidate(
    theta: np.ndarray,
    observed_bundle: dict[str, np.ndarray],
    trace_length: int,
    dt_ms: float,
    pulse_start_frac: float,
    pulse_end_frac: float,
    g_scale_range: tuple[float, float],
    tau_scale_range: tuple[float, float],
    gl_scale_range: tuple[float, float],
    max_abs_voltage: float,
    summary_weight: float,
    range_weight: float,
    mean_weight: float,
    std_weight: float,
    trace_weight: float,
) -> float:
    raw_params, current_gain = unpack_log_params(np.asarray(theta, dtype=np.float64))
    total = 0.0
    count = 0
    for curr, observed in observed_bundle.items():
        current_amplitude = current_gain * float(curr)
        simulated = simulate_compact_hh(
            raw_params,
            current_amplitude,
            trace_length,
            dt_ms,
            pulse_start_frac=pulse_start_frac,
            pulse_end_frac=pulse_end_frac,
            g_scale_range=g_scale_range,
            tau_scale_range=tau_scale_range,
            gl_scale_range=gl_scale_range,
        )
        if not np.isfinite(simulated).all():
            return 1.0e6
        sim_abs_max = float(np.max(np.abs(simulated)))
        if sim_abs_max > max_abs_voltage:
            return 1.0e5 + sim_abs_max
        metrics = trace_metrics(observed, simulated)
        total += (
            trace_weight * metrics["trace_rmse_z"]
            + summary_weight * metrics["summary_rel_l1"]
            + range_weight * metrics["range_rel_error"]
            + mean_weight * metrics["mean_rel_error"]
            + std_weight * metrics["std_rel_error"]
        )
        count += 1
    return total / max(count, 1)


def build_bounds(current_gain_bounds: tuple[float, float]) -> list[tuple[float, float]]:
    bounds = [(math.log10(lo), math.log10(hi)) for lo, hi in RAW_PARAM_BOUNDS]
    bounds.append((math.log10(current_gain_bounds[0]), math.log10(current_gain_bounds[1])))
    return bounds


def summarize_best_candidate(
    theta: np.ndarray,
    observed_bundle: dict[str, np.ndarray],
    trace_length: int,
    dt_ms: float,
    pulse_start_frac: float,
    pulse_end_frac: float,
    g_scale_range: tuple[float, float],
    tau_scale_range: tuple[float, float],
    gl_scale_range: tuple[float, float],
    max_abs_voltage: float,
    summary_weight: float,
    range_weight: float,
    mean_weight: float,
    std_weight: float,
    trace_weight: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw_params, current_gain = unpack_log_params(theta)
    current_rows: list[dict[str, object]] = []
    objective_components = []
    for curr, observed in observed_bundle.items():
        simulated = simulate_compact_hh(
            raw_params,
            current_gain * float(curr),
            trace_length,
            dt_ms,
            pulse_start_frac=pulse_start_frac,
            pulse_end_frac=pulse_end_frac,
            g_scale_range=g_scale_range,
            tau_scale_range=tau_scale_range,
            gl_scale_range=gl_scale_range,
        )
        metrics = trace_metrics(observed, simulated)
        objective_component = (
            trace_weight * metrics["trace_rmse_z"]
            + summary_weight * metrics["summary_rel_l1"]
            + range_weight * metrics["range_rel_error"]
            + mean_weight * metrics["mean_rel_error"]
            + std_weight * metrics["std_rel_error"]
        )
        objective_components.append(objective_component)
        row = {
            "current_label": curr,
            "objective_component": float(objective_component),
            "trace_rmse_z": metrics["trace_rmse_z"],
            "summary_rel_l1": metrics["summary_rel_l1"],
            "range_rel_error": metrics["range_rel_error"],
            "mean_rel_error": metrics["mean_rel_error"],
            "std_rel_error": metrics["std_rel_error"],
            "observed_mean": metrics["observed_mean"],
            "observed_std": metrics["observed_std"],
            "observed_min": metrics["observed_min"],
            "observed_max": metrics["observed_max"],
            "simulated_mean": metrics["simulated_mean"],
            "simulated_std": metrics["simulated_std"],
            "simulated_min": metrics["simulated_min"],
            "simulated_max": metrics["simulated_max"],
            "simulated_abs_max": float(np.max(np.abs(simulated))),
            "voltage_limit_exceeded": float(np.max(np.abs(simulated))) > max_abs_voltage,
        }
        current_rows.append(row)
    mapped = map_surrogate_params(
        raw_params,
        g_scale_range=g_scale_range,
        tau_scale_range=tau_scale_range,
        gl_scale_range=gl_scale_range,
    )
    summary_row = {
        "objective": float(np.mean(objective_components)),
        "current_gain": current_gain,
        "mean_trace_rmse_z": float(np.mean([row["trace_rmse_z"] for row in current_rows])),
        "mean_summary_rel_l1": float(np.mean([row["summary_rel_l1"] for row in current_rows])),
        "mean_range_rel_error": float(np.mean([row["range_rel_error"] for row in current_rows])),
        "mean_mean_rel_error": float(np.mean([row["mean_rel_error"] for row in current_rows])),
        "mean_std_rel_error": float(np.mean([row["std_rel_error"] for row in current_rows])),
        "max_abs_sim_voltage": float(max(row["simulated_abs_max"] for row in current_rows)),
        "raw_param_1": float(raw_params[0]),
        "raw_param_2": float(raw_params[1]),
        "raw_param_3": float(raw_params[2]),
        "raw_param_4": float(raw_params[3]),
        "raw_param_5": float(raw_params[4]),
        "raw_param_6": float(raw_params[5]),
        "mapped_g_na": float(mapped["g_na"]),
        "mapped_tau_m_scale": float(mapped["tau_m_scale"]),
        "mapped_tau_h_scale": float(mapped["tau_h_scale"]),
        "mapped_g_k": float(mapped["g_k"]),
        "mapped_tau_n_scale": float(mapped["tau_n_scale"]),
        "mapped_g_l": float(mapped["g_l"]),
    }
    return summary_row, current_rows


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    g_scale_range = parse_pair(args.g_scale_range)
    tau_scale_range = parse_pair(args.tau_scale_range)
    gl_scale_range = parse_pair(args.gl_scale_range)
    current_gain_bounds = parse_pair(args.current_gain_bounds)
    exemplar_indices = choose_exemplar_indices(args.n_exemplars, args.exemplar_indices, args.seed)
    observed_traces = load_trace_bundle(
        exemplar_indices,
        args.trace_length,
        data_dir=args.data_dir,
        data_prefix=args.data_prefix,
    )

    bounds = build_bounds(current_gain_bounds)
    summary_rows: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []
    per_exemplar_json: list[dict[str, object]] = []

    for exemplar_rank, exemplar_index in enumerate(exemplar_indices):
        observed_bundle = {
            curr: observed_traces[curr][exemplar_rank]
            for curr in CURRENT_LABELS
        }
        start_time = time.time()
        result = differential_evolution(
            evaluate_candidate,
            bounds=bounds,
            args=(
                observed_bundle,
                args.trace_length,
                args.dt_ms,
                args.pulse_start_frac,
                args.pulse_end_frac,
                g_scale_range,
                tau_scale_range,
                gl_scale_range,
                args.max_abs_voltage,
                args.summary_weight,
                args.range_weight,
                args.mean_weight,
                args.std_weight,
                args.trace_weight,
            ),
            seed=args.seed + exemplar_rank,
            strategy="best1bin",
            maxiter=args.maxiter,
            popsize=args.popsize,
            tol=args.tol,
            mutation=(args.mutation_min, args.mutation_max),
            recombination=args.recombination,
            workers=args.workers,
            updating="deferred" if args.workers != 1 else "immediate",
            polish=args.polish,
            init="latinhypercube",
            disp=False,
        )
        elapsed = time.time() - start_time

        summary_row, exemplar_current_rows = summarize_best_candidate(
            result.x,
            observed_bundle,
            args.trace_length,
            args.dt_ms,
            args.pulse_start_frac,
            args.pulse_end_frac,
            g_scale_range,
            tau_scale_range,
            gl_scale_range,
            args.max_abs_voltage,
            args.summary_weight,
            args.range_weight,
            args.mean_weight,
            args.std_weight,
            args.trace_weight,
        )
        summary_row.update(
            {
                "exemplar_rank": exemplar_rank,
                "exemplar_index": exemplar_index,
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "elapsed_sec": float(elapsed),
            }
        )
        summary_rows.append(summary_row)

        for row in exemplar_current_rows:
            row.update({"exemplar_rank": exemplar_rank, "exemplar_index": exemplar_index})
            current_rows.append(row)

        exemplar_json = {
            "assumption_conditioned": True,
            "not_dataset_faithful": True,
            "model_family": "compact_single_compartment_hh_sandbox_fit_search",
            "exemplar_rank": exemplar_rank,
            "exemplar_index": exemplar_index,
            "optimizer": {
                "name": "scipy.optimize.differential_evolution",
                "success": bool(result.success),
                "message": str(result.message),
                "fun": float(result.fun),
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "elapsed_sec": float(elapsed),
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "tol": args.tol,
                "mutation": [args.mutation_min, args.mutation_max],
                "recombination": args.recombination,
                "workers": args.workers,
                "polish": bool(args.polish),
            },
            "fit_summary": summary_row,
            "per_current_metrics": exemplar_current_rows,
        }
        per_exemplar_json.append(exemplar_json)
        exemplar_path = save_dir / f"assumption_conditioned_compact_hh_fit_result_exemplar_{exemplar_rank:02d}.json"
        exemplar_path.write_text(json.dumps(exemplar_json, indent=2) + "\n")

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    current_fieldnames = list(current_rows[0].keys()) if current_rows else []
    if summary_rows:
        write_csv(
            save_dir / "assumption_conditioned_compact_hh_fit_search_summary.csv",
            summary_rows,
            summary_fieldnames,
        )
    if current_rows:
        write_csv(
            save_dir / "assumption_conditioned_compact_hh_fit_search_current_metrics.csv",
            current_rows,
            current_fieldnames,
        )

    aggregate = {
        "assumption_conditioned": True,
        "not_dataset_faithful": True,
        "model_family": "compact_single_compartment_hh_sandbox_fit_search",
        "trace_length": args.trace_length,
        "dt_ms": args.dt_ms,
        "currents": list(CURRENT_LABELS),
        "exemplar_indices": exemplar_indices,
        "n_exemplars": len(exemplar_indices),
        "optimizer": "scipy.optimize.differential_evolution",
        "workers": args.workers,
        "seed": args.seed,
        "current_gain_bounds": current_gain_bounds,
        "g_scale_range": g_scale_range,
        "tau_scale_range": tau_scale_range,
        "gl_scale_range": gl_scale_range,
        "pulse_start_frac": args.pulse_start_frac,
        "pulse_end_frac": args.pulse_end_frac,
        "summary_weight": args.summary_weight,
        "range_weight": args.range_weight,
        "mean_weight": args.mean_weight,
        "std_weight": args.std_weight,
        "max_abs_voltage": args.max_abs_voltage,
        "per_exemplar_results": per_exemplar_json,
    }
    if summary_rows:
        best_row = min(summary_rows, key=lambda row: float(row["objective"]))
        aggregate["aggregate_metrics"] = {
            "mean_objective": float(np.mean([row["objective"] for row in summary_rows])),
            "mean_trace_rmse_z": float(np.mean([row["mean_trace_rmse_z"] for row in summary_rows])),
            "mean_summary_rel_l1": float(np.mean([row["mean_summary_rel_l1"] for row in summary_rows])),
            "best_exemplar_index": int(best_row["exemplar_index"]),
            "best_objective": float(best_row["objective"]),
        }
    (save_dir / "assumption_conditioned_compact_hh_fit_search_manifest.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    print(json.dumps(aggregate.get("aggregate_metrics", {}), indent=2))


if __name__ == "__main__":
    main()
