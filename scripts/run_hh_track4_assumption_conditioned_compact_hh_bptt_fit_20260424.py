#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


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
CURRENT_LABELS = ("0.1", "0.2", "0.3", "0.4", "0.5")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Assumption-conditioned compact HH BPTT fit on small Track4 exemplar bundles.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-exemplars", type=int, default=3)
    ap.add_argument("--exemplar-indices", default="")
    ap.add_argument("--trace-length", type=int, default=2000)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--pulse-start-frac", type=float, default=0.25)
    ap.add_argument("--pulse-end-frac", type=float, default=0.75)
    ap.add_argument("--g-scale-range", default="0.2,5.0")
    ap.add_argument("--tau-scale-range", default="0.5,2.0")
    ap.add_argument("--gl-scale-range", default="0.1,5.0")
    ap.add_argument("--current-gain-bounds", default="2.0,20.0")
    ap.add_argument("--seed", type=int, default=3301)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=3.0e-2)
    ap.add_argument("--summary-weight", type=float, default=0.35)
    ap.add_argument("--range-weight", type=float, default=0.20)
    ap.add_argument("--mean-weight", type=float, default=0.10)
    ap.add_argument("--std-weight", type=float, default=0.10)
    ap.add_argument("--trace-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--max-abs-voltage", type=float, default=5000.0)
    ap.add_argument("--progress-every", type=int, default=25)
    return ap.parse_args()


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
    def _map(raw_value: float, raw_lo: float, raw_hi: float, out_lo: float, out_hi: float) -> float:
        raw_log = math.log10(raw_value)
        lo_log = math.log10(raw_lo)
        hi_log = math.log10(raw_hi)
        t = (raw_log - lo_log) / (hi_log - lo_log)
        out_log = math.log10(out_lo) + t * (math.log10(out_hi) - math.log10(out_lo))
        return float(10 ** out_log)

    return {
        "g_na": 120.0 * _map(raw_params[0], 0.05, 10000.0, g_scale_range[0], g_scale_range[1]),
        "tau_m_scale": _map(raw_params[1], 0.2, 100.0, tau_scale_range[0], tau_scale_range[1]),
        "tau_h_scale": _map(raw_params[2], 5.0, 1000.0, tau_scale_range[0], tau_scale_range[1]),
        "g_k": 36.0 * _map(raw_params[3], 0.05, 10000.0, g_scale_range[0], g_scale_range[1]),
        "tau_n_scale": _map(raw_params[4], 0.2, 100.0, tau_scale_range[0], tau_scale_range[1]),
        "g_l": 0.3 * _map(raw_params[5], 5.0, 1000.0, gl_scale_range[0], gl_scale_range[1]),
    }


def choose_exemplar_indices(n_exemplars: int, explicit: str, seed: int) -> list[int]:
    if explicit.strip():
        out = [int(x.strip()) for x in explicit.split(",") if x.strip()]
        return out
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(15000, size=n_exemplars, replace=False))


def load_trace_bundle(indices: list[int], trace_length: int) -> dict[str, np.ndarray]:
    bundle = {}
    for curr in CURRENT_LABELS:
        path = TRACK4_FEATURES / f"concatenated_data_{curr}_curr.npy"
        arr = np.memmap(path, dtype=np.float32, mode="r", shape=(15000, 400000))
        bundle[curr] = np.asarray(arr[indices, :trace_length], dtype=np.float32)
    return bundle


def safe_div_scalar_tensor(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    eps = torch.full_like(den, 1.0e-12)
    return torch.where(torch.abs(den) <= eps, torch.zeros_like(num), num / den)


def alpha_n(v: torch.Tensor) -> torch.Tensor:
    x = v + 55.0
    return safe_div_scalar_tensor(0.01 * x, 1.0 - torch.exp(-x / 10.0))


def beta_n(v: torch.Tensor) -> torch.Tensor:
    return 0.125 * torch.exp(-(v + 65.0) / 80.0)


def alpha_m(v: torch.Tensor) -> torch.Tensor:
    x = v + 40.0
    return safe_div_scalar_tensor(0.1 * x, 1.0 - torch.exp(-x / 10.0))


def beta_m(v: torch.Tensor) -> torch.Tensor:
    return 4.0 * torch.exp(-(v + 65.0) / 18.0)


def alpha_h(v: torch.Tensor) -> torch.Tensor:
    return 0.07 * torch.exp(-(v + 65.0) / 20.0)


def beta_h(v: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-(v + 35.0) / 10.0))


def make_square_pulse(length: int, amplitude: torch.Tensor, start_frac: float, end_frac: float, device: torch.device) -> torch.Tensor:
    pulse = torch.zeros(length, dtype=torch.float32, device=device)
    lo = int(start_frac * length)
    hi = int(end_frac * length)
    lo = max(0, min(lo, length - 1))
    hi = max(lo + 1, min(hi, length))
    pulse[lo:hi] = amplitude
    return pulse


def map_log_param(log_raw: torch.Tensor, raw_lo: float, raw_hi: float, out_lo: float, out_hi: float) -> torch.Tensor:
    lo_log = math.log10(raw_lo)
    hi_log = math.log10(raw_hi)
    t = (log_raw - lo_log) / (hi_log - lo_log)
    out_log = math.log10(out_lo) + t * (math.log10(out_hi) - math.log10(out_lo))
    return torch.pow(torch.tensor(10.0, device=log_raw.device), out_log)


def simulate_bundle(
    log_raw_params: torch.Tensor,
    log_current_gain: torch.Tensor,
    current_labels: list[float],
    trace_length: int,
    dt_ms: float,
    pulse_start_frac: float,
    pulse_end_frac: float,
    g_scale_range: tuple[float, float],
    tau_scale_range: tuple[float, float],
    gl_scale_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    g_na = 120.0 * map_log_param(log_raw_params[0], 0.05, 10000.0, g_scale_range[0], g_scale_range[1])
    tau_m_scale = map_log_param(log_raw_params[1], 0.2, 100.0, tau_scale_range[0], tau_scale_range[1])
    tau_h_scale = map_log_param(log_raw_params[2], 5.0, 1000.0, tau_scale_range[0], tau_scale_range[1])
    g_k = 36.0 * map_log_param(log_raw_params[3], 0.05, 10000.0, g_scale_range[0], g_scale_range[1])
    tau_n_scale = map_log_param(log_raw_params[4], 0.2, 100.0, tau_scale_range[0], tau_scale_range[1])
    g_l = 0.3 * map_log_param(log_raw_params[5], 5.0, 1000.0, gl_scale_range[0], gl_scale_range[1])
    current_gain = torch.pow(torch.tensor(10.0, device=device), log_current_gain)

    e_na = 50.0
    e_k = -77.0
    e_l = -54.387
    c_m = 1.0

    all_traces = []
    for curr in current_labels:
        current = make_square_pulse(trace_length, current_gain * curr, pulse_start_frac, pulse_end_frac, device)
        v = torch.empty(trace_length, dtype=torch.float32, device=device)
        v[0] = -65.0
        m = alpha_m(v[0]) / (alpha_m(v[0]) + beta_m(v[0]))
        h = alpha_h(v[0]) / (alpha_h(v[0]) + beta_h(v[0]))
        n = alpha_n(v[0]) / (alpha_n(v[0]) + beta_n(v[0]))
        for i in range(1, trace_length):
            vv = v[i - 1]
            a_m = alpha_m(vv)
            b_m = beta_m(vv)
            a_h = alpha_h(vv)
            b_h = beta_h(vv)
            a_n = alpha_n(vv)
            b_n = beta_n(vv)
            m = torch.clamp(m + dt_ms * ((a_m * (1.0 - m) - b_m * m) / tau_m_scale), 0.0, 1.0)
            h = torch.clamp(h + dt_ms * ((a_h * (1.0 - h) - b_h * h) / tau_h_scale), 0.0, 1.0)
            n = torch.clamp(n + dt_ms * ((a_n * (1.0 - n) - b_n * n) / tau_n_scale), 0.0, 1.0)
            i_na = g_na * (m ** 3) * h * (vv - e_na)
            i_k = g_k * (n ** 4) * (vv - e_k)
            i_l = g_l * (vv - e_l)
            v[i] = vv + dt_ms * ((current[i - 1] - i_na - i_k - i_l) / c_m)
        all_traces.append(v)
    return torch.stack(all_traces, dim=0)


def summary12_torch(traces: torch.Tensor) -> torch.Tensor:
    diffs = traces[:, 1:] - traces[:, :-1]
    q10 = torch.quantile(traces, 0.10, dim=1)
    q90 = torch.quantile(traces, 0.90, dim=1)
    return torch.stack(
        [
            traces.mean(dim=1),
            traces.std(dim=1, unbiased=False),
            traces.min(dim=1).values,
            traces.max(dim=1).values,
            traces.median(dim=1).values,
            q10,
            q90,
            torch.sqrt(torch.mean(traces.square(), dim=1)),
            torch.mean(torch.abs(diffs), dim=1),
            torch.max(torch.abs(diffs), dim=1).values,
            torch.mean((traces > 0.0).float(), dim=1),
            torch.mean((traces > -20.0).float(), dim=1),
        ],
        dim=1,
    )


def compute_loss(
    observed: torch.Tensor,
    simulated: torch.Tensor,
    trace_weight: float,
    summary_weight: float,
    range_weight: float,
    mean_weight: float,
    std_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs_center = observed - observed.mean(dim=1, keepdim=True)
    sim_center = simulated - simulated.mean(dim=1, keepdim=True)
    obs_scale = torch.clamp(obs_center.std(dim=1, keepdim=True, unbiased=False), min=1.0e-6)
    sim_scale = torch.clamp(sim_center.std(dim=1, keepdim=True, unbiased=False), min=1.0e-6)
    trace_rmse = torch.sqrt(torch.mean(((obs_center / obs_scale) - (sim_center / sim_scale)) ** 2, dim=1)).mean()

    obs_summary = summary12_torch(observed)
    sim_summary = summary12_torch(simulated)
    summary_scale_floor = torch.tensor(
        [5.0, 5.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 0.25, 2.0, 0.05, 0.05],
        dtype=torch.float32,
        device=observed.device,
    )
    summary_scale = torch.maximum(obs_summary.abs(), summary_scale_floor)
    summary_rel_l1 = torch.mean(torch.abs(sim_summary - obs_summary) / summary_scale)

    obs_mean = observed.mean(dim=1)
    sim_mean = simulated.mean(dim=1)
    obs_std = torch.clamp(observed.std(dim=1, unbiased=False), min=1.0)
    sim_std = simulated.std(dim=1, unbiased=False)
    mean_rel = torch.mean(torch.abs(sim_mean - obs_mean) / obs_std)
    std_rel = torch.mean(torch.abs(sim_std - obs_std) / obs_std)
    obs_range = torch.clamp(observed.max(dim=1).values - observed.min(dim=1).values, min=1.0)
    sim_range = simulated.max(dim=1).values - simulated.min(dim=1).values
    range_rel = torch.mean(torch.abs(sim_range - obs_range) / obs_range)

    loss = (
        trace_weight * trace_rmse
        + summary_weight * summary_rel_l1
        + range_weight * range_rel
        + mean_weight * mean_rel
        + std_weight * std_rel
    )
    metrics = {
        "trace_rmse_z": float(trace_rmse.detach().cpu()),
        "summary_rel_l1": float(summary_rel_l1.detach().cpu()),
        "range_rel_error": float(range_rel.detach().cpu()),
        "mean_rel_error": float(mean_rel.detach().cpu()),
        "std_rel_error": float(std_rel.detach().cpu()),
    }
    return loss, metrics


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)

    exemplar_indices = choose_exemplar_indices(args.n_exemplars, args.exemplar_indices, args.seed)
    raw_bundle = load_trace_bundle(exemplar_indices, args.trace_length)
    observed = np.concatenate([raw_bundle[curr] for curr in CURRENT_LABELS], axis=0)
    observed_t = torch.as_tensor(observed, dtype=torch.float32, device=device)
    current_labels = [float(curr) for curr in CURRENT_LABELS for _ in exemplar_indices]

    g_scale_range = parse_pair(args.g_scale_range)
    tau_scale_range = parse_pair(args.tau_scale_range)
    gl_scale_range = parse_pair(args.gl_scale_range)
    current_gain_bounds = parse_pair(args.current_gain_bounds)

    param_bounds = torch.tensor(
        [
            [math.log10(0.05), math.log10(10000.0)],
            [math.log10(0.2), math.log10(100.0)],
            [math.log10(5.0), math.log10(1000.0)],
            [math.log10(0.05), math.log10(10000.0)],
            [math.log10(0.2), math.log10(100.0)],
            [math.log10(5.0), math.log10(1000.0)],
        ],
        dtype=torch.float32,
        device=device,
    )
    gain_bounds = torch.tensor([math.log10(current_gain_bounds[0]), math.log10(current_gain_bounds[1])], dtype=torch.float32, device=device)

    log_raw_params = torch.nn.Parameter(param_bounds.mean(dim=1).clone())
    log_current_gain = torch.nn.Parameter(gain_bounds.mean().clone())
    optimizer = torch.optim.Adam([log_raw_params, log_current_gain], lr=args.learning_rate)

    history = []
    best = None
    start = time.time()
    live_path = save_dir / "assumption_conditioned_compact_hh_bptt_fit_live.json"
    history_path = save_dir / "assumption_conditioned_compact_hh_bptt_fit_history.csv"
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        clamped_params = torch.maximum(torch.minimum(log_raw_params, param_bounds[:, 1]), param_bounds[:, 0])
        clamped_gain = torch.clamp(log_current_gain, gain_bounds[0], gain_bounds[1])
        simulated = simulate_bundle(
            clamped_params,
            clamped_gain,
            current_labels,
            args.trace_length,
            args.dt_ms,
            args.pulse_start_frac,
            args.pulse_end_frac,
            g_scale_range,
            tau_scale_range,
            gl_scale_range,
            device,
        )
        if torch.max(torch.abs(simulated)).item() > args.max_abs_voltage:
            loss = torch.tensor(float(args.max_abs_voltage), device=device, requires_grad=True)
            metrics = {
                "trace_rmse_z": float("nan"),
                "summary_rel_l1": float("nan"),
                "range_rel_error": float("nan"),
                "mean_rel_error": float("nan"),
                "std_rel_error": float("nan"),
            }
        else:
            loss, metrics = compute_loss(
                observed_t,
                simulated,
                args.trace_weight,
                args.summary_weight,
                args.range_weight,
                args.mean_weight,
                args.std_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_raw_params, log_current_gain], args.grad_clip)
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach().cpu()), **metrics}
        history.append(record)
        if best is None or record["loss"] < best["loss"]:
            best = {
                "step": step,
                "loss": record["loss"],
                "log_raw_params": clamped_params.detach().cpu().tolist(),
                "log_current_gain": float(clamped_gain.detach().cpu()),
                **metrics,
            }
        if step == 0 or (step + 1) % args.progress_every == 0 or (step + 1) == args.steps:
            live_payload = {
                "step": step,
                "steps_total": args.steps,
                "elapsed_sec": time.time() - start,
                "current": record,
                "best": best,
            }
            live_path.write_text(json.dumps(live_payload, indent=2) + "\n")
            with history_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()), lineterminator="\n")
                writer.writeheader()
                writer.writerows(history)

    final_log_raw_params = torch.tensor(best["log_raw_params"], dtype=torch.float32, device=device)
    final_log_current_gain = torch.tensor(best["log_current_gain"], dtype=torch.float32, device=device)
    simulated = simulate_bundle(
        final_log_raw_params,
        final_log_current_gain,
        current_labels,
        args.trace_length,
        args.dt_ms,
        args.pulse_start_frac,
        args.pulse_end_frac,
        g_scale_range,
        tau_scale_range,
        gl_scale_range,
        device,
    ).detach().cpu().numpy()

    raw_params = np.power(10.0, np.asarray(best["log_raw_params"], dtype=np.float64))
    current_gain = float(10.0 ** best["log_current_gain"])
    summary_rows = []
    for i, curr in enumerate(CURRENT_LABELS):
        obs = raw_bundle[curr]
        sim = simulated[i * len(exemplar_indices):(i + 1) * len(exemplar_indices)]
        trace_rmse_vals = []
        for o, s in zip(obs, sim):
            oc = o - o.mean()
            sc = s - s.mean()
            os = max(float(oc.std()), 1.0e-6)
            ss = max(float(sc.std()), 1.0e-6)
            trace_rmse_vals.append(float(np.sqrt(np.mean(((oc / os) - (sc / ss)) ** 2))))
        summary_rows.append(
            {
                "current_label": curr,
                "mean_trace_rmse_z": float(np.mean(trace_rmse_vals)),
                "observed_min": float(obs.min()),
                "observed_max": float(obs.max()),
                "simulated_min": float(sim.min()),
                "simulated_max": float(sim.max()),
            }
        )

    mapped = map_surrogate_params(raw_params, g_scale_range=g_scale_range, tau_scale_range=tau_scale_range, gl_scale_range=gl_scale_range)
    out = {
        "assumption_conditioned": True,
        "not_dataset_faithful": True,
        "optimizer_family": "adam_bptt",
        "device": str(device),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "trace_weight": args.trace_weight,
        "elapsed_sec": time.time() - start,
        "exemplar_indices": exemplar_indices,
        "best": {
            **best,
            "raw_params": raw_params.tolist(),
            "current_gain": current_gain,
            "mapped": {k: float(v) for k, v in mapped.items()},
        },
        "per_current": summary_rows,
        "gpu_memory": None
        if device.type != "cuda"
        else {
            "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
            "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)),
        },
    }

    (save_dir / "assumption_conditioned_compact_hh_bptt_fit_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    with (save_dir / "assumption_conditioned_compact_hh_bptt_fit_current_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(out["best"], indent=2))


if __name__ == "__main__":
    main()
