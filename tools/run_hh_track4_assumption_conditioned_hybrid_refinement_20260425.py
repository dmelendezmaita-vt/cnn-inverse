#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from run_hh_track4_assumption_conditioned_compact_hh_fit_search_20260424 import (
    CURRENT_LABELS,
    RAW_PARAM_BOUNDS,
    build_bounds,
    choose_exemplar_indices,
    evaluate_candidate,
    load_trace_bundle,
    parse_pair,
    summarize_best_candidate,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assumption-conditioned hybrid refinement seeded from external predictions or a prior midpoint.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--prediction-file", default="")
    ap.add_argument("--prediction-field", default="test_pred")
    ap.add_argument("--prediction-row-index", type=int, default=0)
    ap.add_argument("--init-mode", choices=["prediction", "midpoint"], default="midpoint")
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
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--maxiter", type=int, default=120)
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


def prior_midpoint_log_theta(current_gain_bounds: tuple[float, float]) -> np.ndarray:
    vals = []
    for lo, hi in RAW_PARAM_BOUNDS:
        vals.append(0.5 * (math.log10(lo) + math.log10(hi)))
    vals.append(0.5 * (math.log10(current_gain_bounds[0]) + math.log10(current_gain_bounds[1])))
    return np.asarray(vals, dtype=np.float64)


def prediction_init_log_theta(prediction_file: str, field: str, row_index: int, current_gain_bounds: tuple[float, float]) -> np.ndarray:
    data = np.load(prediction_file)
    if field not in data:
        raise SystemExit(f"prediction field {field} not found in {prediction_file}")
    arr = np.asarray(data[field])
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise SystemExit(f"prediction field must be shaped [N,6], got {arr.shape}")
    raw_params = np.asarray(arr[row_index], dtype=np.float64)
    theta = [math.log10(max(float(x), 1.0e-12)) for x in raw_params]
    theta.append(0.5 * (math.log10(current_gain_bounds[0]) + math.log10(current_gain_bounds[1])))
    return np.asarray(theta, dtype=np.float64)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    g_scale_range = parse_pair(args.g_scale_range)
    tau_scale_range = parse_pair(args.tau_scale_range)
    gl_scale_range = parse_pair(args.gl_scale_range)
    current_gain_bounds = parse_pair(args.current_gain_bounds)
    exemplar_indices = choose_exemplar_indices(args.n_exemplars, args.exemplar_indices, args.seed)
    observed_traces = load_trace_bundle(exemplar_indices, args.trace_length)
    observed_bundle = {curr: observed_traces[curr][0] for curr in CURRENT_LABELS}

    if args.init_mode == "prediction":
        if not args.prediction_file:
            raise SystemExit("prediction mode requires --prediction-file")
        x0 = prediction_init_log_theta(args.prediction_file, args.prediction_field, args.prediction_row_index, current_gain_bounds)
    else:
        x0 = prior_midpoint_log_theta(current_gain_bounds)

    bounds = build_bounds(current_gain_bounds)
    objective = lambda theta: evaluate_candidate(  # noqa: E731
        theta,
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
    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": args.maxiter, "disp": False},
    )

    summary_row, current_rows = summarize_best_candidate(
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
            "init_mode": args.init_mode,
            "prediction_file": args.prediction_file,
            "prediction_field": args.prediction_field,
            "prediction_row_index": args.prediction_row_index,
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(getattr(result, "nfev", -1)),
        }
    )
    write_csv(save_dir / "hybrid_refinement_summary.csv", [summary_row], list(summary_row.keys()))
    write_csv(save_dir / "hybrid_refinement_current_metrics.csv", current_rows, list(current_rows[0].keys()))
    manifest = {
        "assumption_conditioned": True,
        "method_family": "hybrid_amortized_init_plus_local_refinement",
        "summary": summary_row,
        "per_current": current_rows,
    }
    (save_dir / "hybrid_refinement_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary_row, indent=2))


if __name__ == "__main__":
    main()
