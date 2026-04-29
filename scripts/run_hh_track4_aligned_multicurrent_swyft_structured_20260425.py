#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import swyft
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))

from data import preprocess_targets  # type: ignore  # noqa: E402
from run_hh_track4_aligned_multicurrent_classical_20260425 import (  # noqa: E402
    load_one_current,
    load_params,
    postprocess_targets_array,
)
from run_hh_track4_aligned_multicurrent_swyft_20260425 import (  # noqa: E402
    HHNativeSwyftModule,
    infer_posterior_means,
    metrics,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Structured Swyft aligned multi-current variant.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12")
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--structured-mode", default="currfusion_chunkstats", choices=["currfusion_chunkstats", "currfusion_chunkstats_deltas"])
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument(
        "--data-dir",
        default="concatenated_data.tar.gz",
    )
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=1.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-2)
    ap.add_argument("--hidden-features", type=int, default=192)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--inference-batch-size", type=int, default=128)
    ap.add_argument("--disable-pairwise", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunk_stats(arr: np.ndarray, n_chunks: int) -> np.ndarray:
    n, d = arr.shape
    edges = np.linspace(0, d, num=n_chunks + 1, dtype=np.int64)
    parts = []
    for i in range(n_chunks):
        lo, hi = int(edges[i]), int(edges[i + 1])
        block = arr[:, lo:hi]
        if block.shape[1] == 0:
            block = arr[:, :1]
        parts.append(np.mean(block, axis=1, keepdims=True))
        parts.append(np.std(block, axis=1, keepdims=True))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def structured_fusion(features_list: list[np.ndarray], mode: str, n_chunks: int) -> np.ndarray:
    per_current = [chunk_stats(arr.astype(np.float32, copy=False), n_chunks) for arr in features_list]
    stacked = np.stack(per_current, axis=1).astype(np.float32, copy=False)
    flat = stacked.reshape(stacked.shape[0], -1)
    pooled = np.concatenate([np.mean(stacked, axis=1), np.std(stacked, axis=1)], axis=1).astype(np.float32, copy=False)
    if mode == "currfusion_chunkstats":
        return np.concatenate([flat, pooled], axis=1).astype(np.float32, copy=False)
    deltas = stacked[:, 1:, :] - stacked[:, :-1, :]
    return np.concatenate([flat, pooled, deltas.reshape(deltas.shape[0], -1)], axis=1).astype(np.float32, copy=False)


def build_dataset(args: argparse.Namespace, logger: logging.Logger):
    params = load_params(args.params)
    currents = [x.strip() for x in args.currents.split(",") if x.strip()]
    features_by_split = {"train": [], "validate": [], "test": []}
    targets_ref = None
    for curr in currents:
        features_curr, targets_curr = load_one_current(
            params,
            curr=curr,
            data_dir=args.data_dir,
            data_prefix=args.data_prefix,
            n_train=args.n_train,
            n_validate=args.n_validate,
            n_test=args.n_test,
            feature_mode=args.feature_mode,
        )
        for split in features_by_split:
            features_by_split[split].append(features_curr[split])
        if targets_ref is None:
            targets_ref = targets_curr
    assert targets_ref is not None
    targets_scale = preprocess_targets(targets_ref, params, logger)
    x_train = structured_fusion(features_by_split["train"], args.structured_mode, args.n_chunks)
    x_validate = structured_fusion(features_by_split["validate"], args.structured_mode, args.n_chunks)
    x_test = structured_fusion(features_by_split["test"], args.structured_mode, args.n_chunks)
    return currents, targets_scale, x_train, x_validate, x_test, targets_ref["train"], targets_ref["validate"], targets_ref["test"]


def select_accelerator(requested: str) -> tuple[str, int]:
    if requested == "cpu":
        return "cpu", 1
    if requested == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("Requested GPU but torch.cuda.is_available() is False in the Swyft env.")
        return "gpu", 1
    return ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("swyft_structured")
    seed_everything(args.seed)
    accelerator, devices = select_accelerator(args.device)
    currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test = build_dataset(args, logger)
    train_samples = swyft.Samples({"x": x_train.astype(np.float32, copy=False), "z": y_train.astype(np.float32, copy=False)})
    validate_samples = swyft.Samples({"x": x_validate.astype(np.float32, copy=False), "z": y_validate.astype(np.float32, copy=False)})
    train_loader = train_samples.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validate_loader = validate_samples.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = HHNativeSwyftModule(
        num_features=x_train.shape[1],
        num_params=y_train.shape[1],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_features=args.hidden_features,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        include_pairwise=not args.disable_pairwise,
    )
    trainer = swyft.SwyftTrainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        default_root_dir=str(save_dir),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validate_loader)
    theta_bank = np.concatenate([y_train, y_validate], axis=0).astype(np.float32, copy=False)
    pred_norm, std_norm, q_norm, weights_1d, _ = infer_posterior_means(
        trainer=trainer,
        model=model,
        x_test=x_test,
        theta_bank=theta_bank,
        inference_batch_size=args.inference_batch_size,
        logger=logger,
    )
    y_true = postprocess_targets_array(y_test, targets_scale)
    y_pred = postprocess_targets_array(pred_norm, targets_scale)
    std_post = np.abs(postprocess_targets_array(pred_norm + std_norm, targets_scale) - y_pred)
    q05 = postprocess_targets_array(q_norm[:, 0, :], targets_scale)
    q95 = postprocess_targets_array(q_norm[:, 1, :], targets_scale)
    metrics_test = metrics(y_true, y_pred)
    np.savez_compressed(
        save_dir / "predictions_test.npz",
        test_true=y_true,
        test_pred=y_pred,
        test_pred_norm=pred_norm,
        posterior_std_norm=std_norm,
        posterior_std=std_post,
        posterior_q05=q05,
        posterior_q95=q95,
        posterior_weights_1d=weights_1d,
    )
    summary = {
        "framework": "swyft_native_structured",
        "structured_mode": args.structured_mode,
        "n_chunks": args.n_chunks,
        "feature_mode": args.feature_mode,
        "currents": currents,
        "feature_dim": int(x_train.shape[1]),
        "pairwise_marginals_enabled": bool(not args.disable_pairwise),
        "test_metrics": metrics_test,
        "posterior_concentration": {
            "overall_mean_std": float(np.mean(std_post)),
            "per_target_mean_std": np.mean(std_post, axis=0).astype(float).tolist(),
        },
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

