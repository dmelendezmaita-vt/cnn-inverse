#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import swyft
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from hh_track4_assumption_conditioned_compact_hh_utils_20260425 import CURRENT_LABELS  # noqa: E402
from run_hh_track4_aligned_multicurrent_swyft_20260425 import (  # noqa: E402
    DEFAULT_STRUCTURED_HIDDEN_FEATURES,
    build_aligned_dataset,
    build_swyft_model,
    postprocess_targets_array,
)
from run_hh_track4_assumption_conditioned_compact_hh_sandbox_20260424 import simulate_compact_hh  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SBC-style surrogate validation for a trained Swyft posterior.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--stage1-run-dir", required=True)
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12")
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--aggregation", default="concat", choices=["concat", "meanstd"])
    ap.add_argument(
        "--data-dir",
        default="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track4_hh_full/concatenated_data",
    )
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--n-query", type=int, default=64)
    ap.add_argument("--posterior-draws", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--inference-batch-size", type=int, default=64)
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--fixed-current-gain", type=float, default=2.0)
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_accelerator(requested: str) -> tuple[str, int]:
    if requested == "cpu":
        return "cpu", 1
    if not torch.cuda.is_available():
        raise SystemExit("Requested GPU but CUDA is unavailable.")
    return "gpu", 1


def compute_summary12(arr2d: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr2d, dtype=np.float32)
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


def compute_fft256(arr2d: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr2d, dtype=np.float32)
    centered = traces - np.mean(traces, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    magnitudes = np.log1p(np.abs(spectrum[:, 1:257]))
    return magnitudes.astype(np.float32, copy=False)


def find_checkpoint(stage1_run_dir: Path) -> Path:
    ckpts = sorted((stage1_run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"No checkpoint under {stage1_run_dir}")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def featurize_bundle(bundle: np.ndarray, feature_mode: str, aggregation: str) -> np.ndarray:
    per_current = []
    for i in range(bundle.shape[0]):
        raw = bundle[i].astype(np.float32)[None, :]
        if feature_mode == "raw":
            feat = raw
        elif feature_mode == "summary12":
            feat = compute_summary12(raw)
        elif feature_mode == "raw_plus_summary12":
            feat = np.concatenate([raw, compute_summary12(raw)], axis=1)
        elif feature_mode == "fft256_summary12":
            feat = np.concatenate([compute_fft256(raw), compute_summary12(raw)], axis=1)
        elif feature_mode == "raw_plus_fft256_summary12":
            feat = np.concatenate([raw, compute_fft256(raw), compute_summary12(raw)], axis=1)
        else:
            raise ValueError(feature_mode)
        per_current.append(feat[0])
    per_current = [np.asarray(x, dtype=np.float32) for x in per_current]
    if aggregation == "concat":
        return np.concatenate(per_current, axis=0).astype(np.float32)
    stack = np.stack(per_current, axis=0).astype(np.float32)
    return np.concatenate([np.mean(stack, axis=0), np.std(stack, axis=0)], axis=0).astype(np.float32)


def rank_summary(posterior_samples: np.ndarray, theta_true: np.ndarray) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows = []
    ks = []
    for j in range(theta_true.shape[1]):
        ranks = np.sum(posterior_samples[:, :, j] < theta_true[:, None, j], axis=1).astype(np.float64)
        u = np.sort((ranks + 0.5) / (posterior_samples.shape[1] + 1.0))
        emp = (np.arange(len(u), dtype=np.float64) + 1.0) / len(u)
        ks_j = float(np.max(np.abs(emp - u))) if len(u) else 0.0
        rows.append({"target_index": j + 1, "rank_ks": ks_j, "rank_mean": float(np.mean(u))})
        ks.append(ks_j)
    return rows, {"rank_ks_mean": float(np.mean(ks)), "rank_ks_max": float(np.max(ks))}


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("swyft_surrogate_sbc")
    seed_everything(args.seed)
    accelerator, devices = select_accelerator(args.device)
    currents, targets_scale, x_train, x_validate, _x_test, y_train, y_validate, _y_test, dataset_meta = build_aligned_dataset(args, logger)
    stage1_run_dir = Path(args.stage1_run_dir)
    summary = json.loads((stage1_run_dir / "metrics_summary.json").read_text())
    ckpt = find_checkpoint(stage1_run_dir)
    model_variant = str(summary.get("model_variant", "flat"))
    structured_hidden = int(summary.get("structured_hidden_features", DEFAULT_STRUCTURED_HIDDEN_FEATURES) or DEFAULT_STRUCTURED_HIDDEN_FEATURES)
    include_pairwise = bool(summary.get("pairwise_marginals_enabled", True))
    model = build_swyft_model(
        num_features=x_train.shape[1],
        num_params=y_train.shape[1],
        learning_rate=1.0e-3,
        weight_decay=1.0e-2,
        hidden_features=int(summary.get("hidden_features", 256)),
        num_blocks=int(summary.get("num_blocks", 3)),
        dropout=float(summary.get("dropout", 0.1)),
        include_pairwise=include_pairwise,
        model_variant=model_variant,
        dataset_meta=dataset_meta,
        structured_hidden_features=structured_hidden,
    )
    state = torch.load(ckpt, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    trainer = swyft.SwyftTrainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        default_root_dir=str(save_dir / "infer"),
    )
    theta_bank_norm = np.concatenate([y_train, y_validate], axis=0).astype(np.float32)
    theta_bank = postprocess_targets_array(theta_bank_norm, targets_scale)
    bank = swyft.Samples({"z": theta_bank_norm.astype(np.float32)})
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(theta_bank.shape[0], size=args.n_query, replace=False)
    posterior_draws = np.empty((args.n_query, args.posterior_draws, theta_bank.shape[1]), dtype=np.float32)
    theta_true = theta_bank[chosen].astype(np.float32)
    rows = []
    for qi, bank_idx in enumerate(chosen):
        bundle = []
        for curr in currents:
            trace = simulate_compact_hh(theta_true[qi], args.fixed_current_gain * float(curr), 1000, 0.025, pulse_start_frac=0.10, pulse_end_frac=0.90, g_scale_range=(0.2, 5.0), tau_scale_range=(0.5, 2.0), gl_scale_range=(0.1, 5.0))
            bundle.append(trace.astype(np.float32))
        x = featurize_bundle(np.stack(bundle, axis=0), args.feature_mode, args.aggregation)
        obs = swyft.Sample(x=x.astype(np.float32))
        outputs = trainer.infer(model, obs, bank, batch_size=args.inference_batch_size)
        lrs_1d = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        score = np.zeros(theta_bank_norm.shape[0], dtype=np.float64)
        for j in range(theta_bank_norm.shape[1]):
            _, w = swyft.get_weighted_samples(lrs_1d, f"theta[{j}]")
            score += np.log(np.clip(np.asarray(w, dtype=np.float64).reshape(-1), 1.0e-12, None))
        prob = np.exp(score - np.max(score))
        prob /= np.sum(prob)
        idx = rng.choice(theta_bank.shape[0], size=args.posterior_draws, replace=True, p=prob)
        posterior_draws[qi] = theta_bank[idx]
        rows.append({"query_id": int(qi), "true_bank_index": int(bank_idx), "max_prob": float(np.max(prob))})
    rank_rows, summary_rank = rank_summary(posterior_draws, theta_true)
    with (save_dir / "swyft_surrogate_sbc_query_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (save_dir / "swyft_surrogate_sbc_rank_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rank_rows)
    manifest = {
        "assumption_conditioned": True,
        "framework": "swyft_surrogate_sbc",
        "stage1_run_dir": str(stage1_run_dir),
        "n_query": args.n_query,
        "posterior_draws": args.posterior_draws,
        "fixed_current_gain": args.fixed_current_gain,
        "rank_summary": summary_rank,
    }
    (save_dir / "swyft_surrogate_sbc_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
