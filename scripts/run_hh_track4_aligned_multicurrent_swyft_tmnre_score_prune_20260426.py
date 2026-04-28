#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import swyft
import torch

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
sys.path.insert(0, str(REPO / "scripts"))

from run_hh_track4_aligned_multicurrent_swyft_20260425 import (  # noqa: E402
    DEFAULT_STRUCTURED_HIDDEN_FEATURES,
    build_aligned_dataset,
    build_swyft_model,
    infer_posterior_means,
    postprocess_targets_array,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Non-rectangular score-pruned TMNRE follow-up for native Swyft.")
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
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=1.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-2)
    ap.add_argument("--hidden-features", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--inference-batch-size", type=int, default=64)
    ap.add_argument("--score-observations", type=int, default=128)
    ap.add_argument("--keep-fracs", default="0.5,0.25")
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - float(np.sum(np.square(diff))) / denom if denom > 0.0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def parse_keep_fracs(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(not (0.0 < x <= 1.0) for x in vals):
        raise SystemExit("keep-fracs must be comma-separated values in (0,1].")
    return vals


def select_accelerator(requested: str) -> tuple[str, int]:
    if requested == "cpu":
        return "cpu", 1
    if not torch.cuda.is_available():
        raise SystemExit("Requested GPU but CUDA is unavailable.")
    return "gpu", 1


def find_checkpoint(stage1_run_dir: Path) -> Path:
    ckpts = sorted((stage1_run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"No checkpoint under {stage1_run_dir}")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def load_stage1_summary(stage1_run_dir: Path) -> dict[str, object]:
    p = stage1_run_dir / "metrics_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fit_stage(
    save_dir: Path,
    stage_label: str,
    args: argparse.Namespace,
    accelerator: str,
    devices: int,
    dataset_meta: dict[str, object],
    model_variant: str,
    structured_hidden_features: int,
    include_pairwise: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validate: np.ndarray,
    y_validate: np.ndarray,
):
    train_samples = swyft.Samples({"x": x_train.astype(np.float32), "z": y_train.astype(np.float32)})
    validate_samples = swyft.Samples({"x": x_validate.astype(np.float32), "z": y_validate.astype(np.float32)})
    train_loader = train_samples.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validate_loader = validate_samples.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_swyft_model(
        num_features=x_train.shape[1],
        num_params=y_train.shape[1],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_features=args.hidden_features,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        include_pairwise=include_pairwise,
        model_variant=model_variant,
        dataset_meta=dataset_meta,
        structured_hidden_features=structured_hidden_features,
    )
    trainer = swyft.SwyftTrainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        default_root_dir=str(save_dir / stage_label),
    )
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validate_loader)
    return model, trainer, time.time() - t0


def aggregate_bank_scores(
    trainer: swyft.SwyftTrainer,
    model,
    x_validate: np.ndarray,
    theta_bank: np.ndarray,
    inference_batch_size: int,
    num_obs: int,
    logger: logging.Logger,
) -> np.ndarray:
    bank = swyft.Samples({"z": theta_bank.astype(np.float32, copy=False)})
    num_obs = min(num_obs, x_validate.shape[0])
    scores = np.zeros(theta_bank.shape[0], dtype=np.float64)
    for i in range(num_obs):
        if i == 0 or (i + 1) % 16 == 0 or i + 1 == num_obs:
            logger.info("Score-prune stage score inference for validation observation %d/%d", i + 1, num_obs)
        obs = swyft.Sample(x=np.asarray(x_validate[i], dtype=np.float32))
        outputs = trainer.infer(model, obs, bank, batch_size=inference_batch_size)
        lrs_1d = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        for j in range(theta_bank.shape[1]):
            _, w = swyft.get_weighted_samples(lrs_1d, f"theta[{j}]")
            scores += np.log(np.clip(np.asarray(w, dtype=np.float64).reshape(-1), 1.0e-12, None))
    scores /= max(num_obs * theta_bank.shape[1], 1)
    return scores


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("swyft_tmnre_score_prune")
    seed_everything(args.seed)
    accelerator, devices = select_accelerator(args.device)
    keep_fracs = parse_keep_fracs(args.keep_fracs)
    currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test, dataset_meta = build_aligned_dataset(args, logger)

    stage1_run_dir = Path(args.stage1_run_dir)
    stage1_summary = load_stage1_summary(stage1_run_dir)
    ckpt = find_checkpoint(stage1_run_dir)
    model_variant = str(stage1_summary.get("model_variant", "flat"))
    structured_hidden = int(stage1_summary.get("structured_hidden_features", DEFAULT_STRUCTURED_HIDDEN_FEATURES) or DEFAULT_STRUCTURED_HIDDEN_FEATURES)
    include_pairwise = bool(stage1_summary.get("pairwise_marginals_enabled", True))

    stage1_model = build_swyft_model(
        num_features=x_train.shape[1],
        num_params=y_train.shape[1],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_features=args.hidden_features,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        include_pairwise=include_pairwise,
        model_variant=model_variant,
        dataset_meta=dataset_meta,
        structured_hidden_features=structured_hidden,
    )
    state = torch.load(ckpt, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    stage1_model.load_state_dict(state_dict)
    stage1_trainer = swyft.SwyftTrainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        default_root_dir=str(save_dir / "stage1_infer"),
    )

    current_x_train = x_train.astype(np.float32)
    current_y_train = y_train.astype(np.float32)
    current_x_validate = x_validate.astype(np.float32)
    current_y_validate = y_validate.astype(np.float32)
    current_theta_bank = np.concatenate([y_train, y_validate], axis=0).astype(np.float32)
    current_model = stage1_model
    current_trainer = stage1_trainer
    stage_rows = []
    training_runtime_sec = 0.0

    for idx, keep_frac in enumerate(keep_fracs, start=1):
        score = aggregate_bank_scores(
            trainer=current_trainer,
            model=current_model,
            x_validate=current_x_validate,
            theta_bank=current_theta_bank,
            inference_batch_size=args.inference_batch_size,
            num_obs=args.score_observations,
            logger=logger,
        )
        keep_n = max(1, int(round(keep_frac * current_theta_bank.shape[0])))
        top_idx = np.argsort(score)[-keep_n:]
        mask = np.zeros(current_theta_bank.shape[0], dtype=bool)
        mask[top_idx] = True
        train_mask = mask[: current_y_train.shape[0]]
        validate_mask = mask[current_y_train.shape[0] :]
        if not np.any(train_mask) or not np.any(validate_mask):
            raise SystemExit("Score pruning produced an empty train or validate split.")
        n_train_in = current_x_train.shape[0]
        n_validate_in = current_x_validate.shape[0]
        n_bank_in = current_theta_bank.shape[0]
        current_x_train = current_x_train[train_mask]
        current_y_train = current_y_train[train_mask]
        current_x_validate = current_x_validate[validate_mask]
        current_y_validate = current_y_validate[validate_mask]
        current_theta_bank = current_theta_bank[mask]
        current_model, current_trainer, stage_sec = fit_stage(
            save_dir,
            f"score_stage{idx+1}_fit",
            args,
            accelerator,
            devices,
            dataset_meta,
            model_variant,
            structured_hidden,
            include_pairwise,
            current_x_train,
            current_y_train,
            current_x_validate,
            current_y_validate,
        )
        training_runtime_sec += stage_sec
        stage_rows.append({
            "stage_index": idx,
            "keep_frac_requested": keep_frac,
            "n_train_in": int(n_train_in),
            "n_validate_in": int(n_validate_in),
            "n_bank_in": int(n_bank_in),
            "n_train_out": int(current_x_train.shape[0]),
            "n_validate_out": int(current_x_validate.shape[0]),
            "n_bank_out": int(current_theta_bank.shape[0]),
            "score_min": float(np.min(score)),
            "score_max": float(np.max(score)),
            "training_runtime_sec": float(stage_sec),
        })

    pred_norm, std_norm, q_norm, weights_1d, eval_sec = infer_posterior_means(
        trainer=current_trainer,
        model=current_model,
        x_test=x_test,
        theta_bank=current_theta_bank,
        inference_batch_size=args.inference_batch_size,
        logger=logger,
    )
    y_true = postprocess_targets_array(y_test, targets_scale)
    y_pred = postprocess_targets_array(pred_norm, targets_scale)
    std_post = np.abs(postprocess_targets_array(pred_norm + std_norm, targets_scale) - y_pred)
    q05 = postprocess_targets_array(q_norm[:, 0, :], targets_scale)
    q95 = postprocess_targets_array(q_norm[:, 1, :], targets_scale)
    m = metrics(y_true, y_pred)
    np.savez_compressed(
        save_dir / "predictions_test.npz",
        x_test=x_test.astype(np.float32),
        test_true=y_true,
        test_true_norm=y_test.astype(np.float32),
        test_pred=y_pred,
        test_pred_norm=pred_norm,
        posterior_std_norm=std_norm,
        posterior_std=std_post,
        posterior_q05=q05,
        posterior_q95=q95,
        theta_bank_norm=current_theta_bank.astype(np.float32),
        posterior_weights_1d=weights_1d.astype(np.float32),
    )
    with (save_dir / "score_prune_schedule.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(stage_rows)
    summary = {
        "framework": "swyft_tmnre_score_prune",
        "stage1_checkpoint": str(ckpt),
        "feature_mode": args.feature_mode,
        "aggregation": args.aggregation,
        "currents": currents,
        "keep_fracs": keep_fracs,
        "model_variant": model_variant,
        "structured_hidden_features": structured_hidden if model_variant != "flat" else None,
        "training_runtime_sec": float(training_runtime_sec),
        "evaluation_runtime_sec": float(eval_sec),
        "total_runtime_sec": float(training_runtime_sec + eval_sec),
        "score_schedule": stage_rows,
        "test_metrics": m,
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
