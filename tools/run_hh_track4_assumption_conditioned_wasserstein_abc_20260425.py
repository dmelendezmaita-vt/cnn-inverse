#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from run_hh_track4_assumption_conditioned_compact_hh_fit_search_20260424 import (
    CURRENT_LABELS,
    RAW_PARAM_BOUNDS,
    choose_exemplar_indices,
    load_trace_bundle,
    parse_pair,
)
from run_hh_track4_assumption_conditioned_compact_hh_sandbox_20260424 import (
    map_surrogate_params,
    simulate_compact_hh,
    summary12,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assumption-conditioned sliced-Wasserstein ABC over the compact HH sandbox.")
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
    ap.add_argument("--n-particles", type=int, default=256)
    ap.add_argument("--keep-top", type=int, default=24)
    ap.add_argument("--n-projections", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260425)
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi), size=size)


def sliced_wasserstein(obs: np.ndarray, sim: np.ndarray, n_proj: int, rng: np.random.Generator) -> float:
    dim = obs.shape[1]
    dirs = rng.normal(size=(n_proj, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = []
    for d in dirs:
        vals.append(wasserstein_distance(obs @ d, sim @ d))
    return float(np.mean(vals))


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    g_scale_range = parse_pair(args.g_scale_range)
    tau_scale_range = parse_pair(args.tau_scale_range)
    gl_scale_range = parse_pair(args.gl_scale_range)
    current_gain_bounds = parse_pair(args.current_gain_bounds)
    exemplar_indices = choose_exemplar_indices(args.n_exemplars, args.exemplar_indices, args.seed)
    observed_traces = load_trace_bundle(exemplar_indices, args.trace_length)

    observed_summary = []
    for curr in CURRENT_LABELS:
        observed_summary.append(summary12(observed_traces[curr].astype(np.float32)))
    obs = np.concatenate(observed_summary, axis=1).astype(np.float64, copy=False)

    rows = []
    for particle_id in range(args.n_particles):
        raw_params = np.array(
            [
                sample_log_uniform(rng, lo, hi, 1)[0]
                for (lo, hi) in RAW_PARAM_BOUNDS
            ],
            dtype=np.float64,
        )
        current_gain = sample_log_uniform(rng, current_gain_bounds[0], current_gain_bounds[1], 1)[0]
        sim_summary_parts = []
        for curr in CURRENT_LABELS:
            traces = []
            for _ in exemplar_indices:
                traces.append(
                    simulate_compact_hh(
                        raw_params,
                        current_gain * float(curr),
                        args.trace_length,
                        args.dt_ms,
                        pulse_start_frac=args.pulse_start_frac,
                        pulse_end_frac=args.pulse_end_frac,
                        g_scale_range=g_scale_range,
                        tau_scale_range=tau_scale_range,
                        gl_scale_range=gl_scale_range,
                    )
                )
            sim_summary_parts.append(summary12(np.asarray(traces, dtype=np.float32)))
        sim = np.concatenate(sim_summary_parts, axis=1).astype(np.float64, copy=False)
        distance = sliced_wasserstein(obs, sim, args.n_projections, rng)
        rows.append(
            {
                "particle_id": particle_id,
                "distance": distance,
                "current_gain": float(current_gain),
                **{f"raw_param_{i+1}": float(raw_params[i]) for i in range(6)},
                **{f"mapped_{k}": float(v) for k, v in map_surrogate_params(raw_params, g_scale_range=g_scale_range, tau_scale_range=tau_scale_range, gl_scale_range=gl_scale_range).items()},
            }
        )

    rows.sort(key=lambda r: float(r["distance"]))
    top_rows = rows[: args.keep_top]
    write_csv(save_dir / "wasserstein_abc_particles.csv", rows, list(rows[0].keys()))
    write_csv(save_dir / "wasserstein_abc_top_particles.csv", top_rows, list(top_rows[0].keys()))
    manifest = {
        "assumption_conditioned": True,
        "method_family": "sliced_wasserstein_abc",
        "n_particles": args.n_particles,
        "keep_top": args.keep_top,
        "best_distance": float(top_rows[0]["distance"]),
        "median_top_distance": float(np.median([r["distance"] for r in top_rows])),
        "top_particles": top_rows,
    }
    (save_dir / "wasserstein_abc_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
