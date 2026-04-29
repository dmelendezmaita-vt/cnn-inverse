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
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))

from data import preprocess_targets  # type: ignore  # noqa: E402
from run_hh_track4_aligned_multicurrent_classical_20260425 import (  # noqa: E402
    aggregate_currents,
    load_one_current,
    load_params,
    postprocess_targets_array,
)


PAIRWISE_MARGINALS = tuple((i, j) for i in range(6) for j in range(i + 1, 6))
MODEL_VARIANTS = ("flat", "aligned_currents_pool")
DEFAULT_STRUCTURED_HIDDEN_FEATURES = 96


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Native Swyft aligned multi-current HH baseline.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12")
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--aggregation", default="meanstd", choices=["concat", "meanstd"])
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
    ap.add_argument("--hidden-features", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--inference-batch-size", type=int, default=128)
    ap.add_argument("--disable-pairwise", action="store_true")
    ap.add_argument("--model-variant", default="flat", choices=MODEL_VARIANTS)
    ap.add_argument("--structured-hidden-features", type=int, default=DEFAULT_STRUCTURED_HIDDEN_FEATURES)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mean_and_std(samples: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        return mean, std
    weights = weights / total
    mean = float(np.sum(weights * samples))
    var = float(np.sum(weights * np.square(samples - mean)))
    std = float(np.sqrt(max(var, 0.0)))
    return mean, std


def weighted_quantile(samples: np.ndarray, weights: np.ndarray, q: float) -> float:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return float(np.quantile(samples, q))
    order = np.argsort(samples)
    samples = samples[order]
    weights = weights[order] / total
    cdf = np.cumsum(weights)
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), samples.size - 1)
    return float(samples[idx])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def parse_current_values(currents: list[str]) -> np.ndarray:
    values = np.asarray([float(x) for x in currents], dtype=np.float32)
    if values.size == 0:
        raise SystemExit("At least one current value is required.")
    if values.size == 1:
        return np.zeros(1, dtype=np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros_like(values)
    return ((values - lo) / span * 2.0 - 1.0).astype(np.float32, copy=False)


def build_aligned_dataset(args: argparse.Namespace, logger: logging.Logger):
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
        else:
            same_frac = float(
                np.mean(np.all(np.isclose(targets_ref["train"][:256], targets_curr["train"][:256]), axis=1))
            )
            logger.info("alignment check curr=%s rowwise_equal_frac_train_first256=%.6f", curr, same_frac)

    assert targets_ref is not None
    targets_scale = preprocess_targets(targets_ref, params, logger)
    y_train = targets_ref["train"]
    y_validate = targets_ref["validate"]
    y_test = targets_ref["test"]

    x_train = aggregate_currents(features_by_split["train"], args.aggregation)
    x_validate = aggregate_currents(features_by_split["validate"], args.aggregation)
    x_test = aggregate_currents(features_by_split["test"], args.aggregation)
    per_current_feature_dim = int(features_by_split["train"][0].shape[1])
    dataset_meta = {
        "num_currents": int(len(currents)),
        "per_current_feature_dim": per_current_feature_dim,
        "concat_feature_dim": int(per_current_feature_dim * len(currents)),
        "aggregation": args.aggregation,
        "normalized_current_values": parse_current_values(currents).astype(float).tolist(),
    }
    logger.info(
        "native swyft features built: aggregation=%s feature_mode=%s x_train=%s x_validate=%s x_test=%s",
        args.aggregation,
        args.feature_mode,
        x_train.shape,
        x_validate.shape,
        x_test.shape,
    )
    return currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test, dataset_meta


class AlignedCurrentStructuredEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_currents: int,
        per_current_feature_dim: int,
        latent_features: int,
        output_features: int,
        dropout: float,
        current_values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> None:
        super().__init__()
        if num_currents <= 0:
            raise ValueError(f"num_currents must be positive, got {num_currents}")
        if per_current_feature_dim <= 0:
            raise ValueError(f"per_current_feature_dim must be positive, got {per_current_feature_dim}")
        if latent_features <= 0 or output_features <= 0:
            raise ValueError("structured encoder feature sizes must be positive")
        current_values_arr = np.asarray(current_values, dtype=np.float32).reshape(-1)
        if current_values_arr.shape[0] != num_currents:
            raise ValueError(
                f"Expected {num_currents} current values for aligned encoder, got {current_values_arr.shape[0]}"
            )

        self.num_currents = num_currents
        self.per_current_feature_dim = per_current_feature_dim
        self.output_features = output_features

        self.register_buffer(
            "current_values",
            torch.from_numpy(current_values_arr).view(1, num_currents, 1),
            persistent=False,
        )
        if num_currents > 1:
            delta_values = np.diff(current_values_arr).astype(np.float32, copy=False)
        else:
            delta_values = np.zeros(1, dtype=np.float32)
        self.register_buffer(
            "delta_values",
            torch.from_numpy(delta_values).view(1, max(num_currents - 1, 1), 1),
            persistent=False,
        )

        self.input_norm = nn.LayerNorm(per_current_feature_dim)
        self.current_encoder = nn.Sequential(
            nn.Linear(per_current_feature_dim + 1, latent_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_features, latent_features),
            nn.GELU(),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(latent_features + 1, latent_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_features, latent_features),
            nn.GELU(),
        )
        self.summary_head = nn.Sequential(
            nn.LayerNorm(latent_features * 7),
            nn.Linear(latent_features * 7, output_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError(f"Expected 2D flattened observation tensor, got shape={tuple(x.shape)}")
        expected_dim = self.num_currents * self.per_current_feature_dim
        if x.shape[1] != expected_dim:
            raise RuntimeError(f"Expected flattened dim={expected_dim}, got {x.shape[1]}")

        batch_size = x.shape[0]
        blocks = x.reshape(batch_size, self.num_currents, self.per_current_feature_dim)
        blocks = self.input_norm(blocks)
        current_tags = self.current_values.expand(batch_size, -1, -1)
        encoded = self.current_encoder(torch.cat([blocks, current_tags], dim=-1))

        mean_pool = encoded.mean(dim=1)
        std_pool = encoded.std(dim=1, unbiased=False)
        max_pool = encoded.max(dim=1).values
        first_state = encoded[:, 0, :]
        last_state = encoded[:, -1, :]

        if self.num_currents > 1:
            delta_tags = self.delta_values.expand(batch_size, -1, -1)
            delta_encoded = self.delta_encoder(torch.cat([encoded[:, 1:, :] - encoded[:, :-1, :], delta_tags], dim=-1))
            delta_mean = delta_encoded.mean(dim=1)
            delta_absmax = delta_encoded.abs().amax(dim=1)
        else:
            delta_mean = torch.zeros_like(mean_pool)
            delta_absmax = torch.zeros_like(mean_pool)

        summary = torch.cat(
            [mean_pool, std_pool, max_pool, first_state, last_state, delta_mean, delta_absmax],
            dim=1,
        )
        return self.summary_head(summary)


class HHNativeSwyftModule(swyft.SwyftModule):
    def __init__(
        self,
        *,
        num_features: int,
        num_params: int,
        learning_rate: float,
        weight_decay: float,
        hidden_features: int,
        num_blocks: int,
        dropout: float,
        include_pairwise: bool,
        model_variant: str = "flat",
        num_currents: int | None = None,
        per_current_feature_dim: int | None = None,
        current_values: list[float] | tuple[float, ...] | np.ndarray | None = None,
        structured_hidden_features: int = DEFAULT_STRUCTURED_HIDDEN_FEATURES,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_variant = model_variant
        self.input_num_features = num_features
        self.encoded_num_features = num_features
        self.structured_encoder = None

        if model_variant == "aligned_currents_pool":
            if num_currents is None or per_current_feature_dim is None or current_values is None:
                raise ValueError("Aligned current pooling requires num_currents, per_current_feature_dim, and current_values.")
            expected_dim = int(num_currents) * int(per_current_feature_dim)
            if expected_dim != num_features:
                raise ValueError(f"Aligned current pooling expects flattened concat dim={expected_dim}, got {num_features}.")
            self.structured_encoder = AlignedCurrentStructuredEncoder(
                num_currents=int(num_currents),
                per_current_feature_dim=int(per_current_feature_dim),
                latent_features=int(structured_hidden_features),
                output_features=int(hidden_features),
                dropout=dropout,
                current_values=current_values,
            )
            self.encoded_num_features = int(hidden_features)
        elif model_variant != "flat":
            raise ValueError(f"Unsupported model_variant={model_variant}")

        self.logratios_1d = swyft.LogRatioEstimator_1dim(
            num_features=self.encoded_num_features,
            num_params=num_params,
            varnames="theta",
            dropout=dropout,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            use_batch_norm=True,
            Lmax=0,
        )
        self.logratios_2d = None
        if include_pairwise:
            self.logratios_2d = swyft.LogRatioEstimator_Ndim(
                num_features=self.encoded_num_features,
                marginals=PAIRWISE_MARGINALS,
                varnames="theta",
                dropout=dropout,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                Lmax=0,
            )

    def encode_observation(self, x: torch.Tensor) -> torch.Tensor:
        if self.structured_encoder is None:
            return x
        return self.structured_encoder(x)

    def forward(self, A, B):
        x = self.encode_observation(A["x"])
        out_1d = self.logratios_1d(x, B["z"])
        if self.logratios_2d is None:
            return out_1d
        out_2d = self.logratios_2d(x, B["z"])
        return out_1d, out_2d


def build_swyft_model(
    *,
    num_features: int,
    num_params: int,
    learning_rate: float,
    weight_decay: float,
    hidden_features: int,
    num_blocks: int,
    dropout: float,
    include_pairwise: bool,
    model_variant: str,
    dataset_meta: dict[str, object],
    structured_hidden_features: int = DEFAULT_STRUCTURED_HIDDEN_FEATURES,
) -> HHNativeSwyftModule:
    if model_variant != "flat":
        if dataset_meta.get("aggregation") != "concat":
            raise SystemExit(f"Model variant {model_variant} requires --aggregation concat to preserve aligned current structure.")
        if int(dataset_meta["num_currents"]) * int(dataset_meta["per_current_feature_dim"]) != num_features:
            raise SystemExit(
                f"Model variant {model_variant} requires concat-preserved aligned current features, got input dim={num_features}."
            )
    return HHNativeSwyftModule(
        num_features=num_features,
        num_params=num_params,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        dropout=dropout,
        include_pairwise=include_pairwise,
        model_variant=model_variant,
        num_currents=int(dataset_meta["num_currents"]),
        per_current_feature_dim=int(dataset_meta["per_current_feature_dim"]),
        current_values=list(dataset_meta["normalized_current_values"]),
        structured_hidden_features=int(structured_hidden_features),
    )


def select_accelerator(requested: str) -> tuple[str, int]:
    if requested == "cpu":
        return "cpu", 1
    if requested == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("Requested GPU but torch.cuda.is_available() is False in the Swyft env.")
        return "gpu", 1
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def infer_posterior_means(
    *,
    trainer: swyft.SwyftTrainer,
    model: HHNativeSwyftModule,
    x_test: np.ndarray,
    theta_bank: np.ndarray,
    inference_batch_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    num_test = x_test.shape[0]
    num_params = theta_bank.shape[1]
    num_bank = theta_bank.shape[0]
    predictions = np.empty((num_test, num_params), dtype=np.float32)
    posterior_std = np.empty((num_test, num_params), dtype=np.float32)
    posterior_q05 = np.empty((num_test, num_params), dtype=np.float32)
    posterior_q95 = np.empty((num_test, num_params), dtype=np.float32)
    posterior_weights = np.empty((num_test, num_bank, num_params), dtype=np.float32)

    bank = swyft.Samples({"z": theta_bank.astype(np.float32, copy=False)})
    started = time.time()
    for idx in range(num_test):
        if idx == 0 or (idx + 1) % 16 == 0 or idx + 1 == num_test:
            logger.info("Swyft posterior inference for test observation %d/%d", idx + 1, num_test)
        obs = swyft.Sample(x=np.asarray(x_test[idx], dtype=np.float32))
        outputs = trainer.infer(model, obs, bank, batch_size=inference_batch_size)
        lrs_1d = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        for j in range(num_params):
            params_j, weights_j = swyft.get_weighted_samples(lrs_1d, f"theta[{j}]")
            posterior_weights[idx, :, j] = np.asarray(weights_j, dtype=np.float32).reshape(-1)
            mean_j, std_j = weighted_mean_and_std(params_j, weights_j)
            predictions[idx, j] = mean_j
            posterior_std[idx, j] = std_j
            posterior_q05[idx, j] = weighted_quantile(params_j, weights_j, 0.05)
            posterior_q95[idx, j] = weighted_quantile(params_j, weights_j, 0.95)
    return (
        predictions,
        posterior_std,
        np.stack([posterior_q05, posterior_q95], axis=1),
        posterior_weights,
        time.time() - started,
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("aligned_multicurrent_swyft_native")

    seed_everything(args.seed)
    accelerator, devices = select_accelerator(args.device)
    currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test, dataset_meta = build_aligned_dataset(
        args, logger
    )
    train_samples = swyft.Samples({"x": x_train.astype(np.float32, copy=False), "z": y_train.astype(np.float32, copy=False)})
    validate_samples = swyft.Samples(
        {"x": x_validate.astype(np.float32, copy=False), "z": y_validate.astype(np.float32, copy=False)}
    )
    train_loader = train_samples.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validate_loader = validate_samples.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_swyft_model(
        num_features=x_train.shape[1],
        num_params=y_train.shape[1],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_features=args.hidden_features,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        include_pairwise=not args.disable_pairwise,
        model_variant=args.model_variant,
        dataset_meta=dataset_meta,
        structured_hidden_features=args.structured_hidden_features,
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

    started = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validate_loader)
    training_runtime_sec = time.time() - started

    theta_bank = np.concatenate([y_train, y_validate], axis=0).astype(np.float32, copy=False)
    y_pred_norm, posterior_std_norm, posterior_quantiles_norm, posterior_weights_1d, evaluation_runtime_sec = infer_posterior_means(
        trainer=trainer,
        model=model,
        x_test=x_test,
        theta_bank=theta_bank,
        inference_batch_size=args.inference_batch_size,
        logger=logger,
    )

    y_test_post = postprocess_targets_array(y_test, targets_scale)
    y_pred_post = postprocess_targets_array(y_pred_norm, targets_scale)
    posterior_std_post = np.abs(postprocess_targets_array(y_pred_norm + posterior_std_norm, targets_scale) - y_pred_post)
    posterior_q05_post = postprocess_targets_array(posterior_quantiles_norm[:, 0, :], targets_scale)
    posterior_q95_post = postprocess_targets_array(posterior_quantiles_norm[:, 1, :], targets_scale)

    metrics_test = metrics(y_test_post, y_pred_post)
    total_runtime_sec = training_runtime_sec + evaluation_runtime_sec

    with (save_dir / "metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "mse", "mae", "r2"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"split": "test", **metrics_test})

    np.savez_compressed(
        save_dir / "predictions_test.npz",
        x_test=x_test.astype(np.float32, copy=False),
        test_true=y_test_post,
        test_true_norm=y_test.astype(np.float32, copy=False),
        test_pred=y_pred_post,
        test_pred_norm=y_pred_norm,
        posterior_std_norm=posterior_std_norm,
        posterior_std=posterior_std_post,
        posterior_q05=posterior_q05_post,
        posterior_q95=posterior_q95_post,
        theta_bank_norm=theta_bank.astype(np.float32, copy=False),
        posterior_weights_1d=posterior_weights_1d.astype(np.float32, copy=False),
    )

    summary = {
        "framework": "swyft_native",
        "feature_mode": args.feature_mode,
        "aggregation": args.aggregation,
        "currents": currents,
        "seed": args.seed,
        "device": accelerator,
        "n_train": int(x_train.shape[0]),
        "n_validate": int(x_validate.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "posterior_bank_size": int(theta_bank.shape[0]),
        "training_runtime_sec": float(training_runtime_sec),
        "evaluation_runtime_sec": float(evaluation_runtime_sec),
        "total_runtime_sec": float(total_runtime_sec),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "hidden_features": int(args.hidden_features),
        "num_blocks": int(args.num_blocks),
        "dropout": float(args.dropout),
        "model_variant": args.model_variant,
        "structured_hidden_features": int(args.structured_hidden_features) if args.model_variant != "flat" else None,
        "logratio_feature_dim": int(model.encoded_num_features),
        "num_currents": int(dataset_meta["num_currents"]),
        "per_current_feature_dim": int(dataset_meta["per_current_feature_dim"]),
        "normalized_current_values": dataset_meta["normalized_current_values"],
        "pairwise_marginals_enabled": bool(not args.disable_pairwise),
        "pairwise_marginals": [list(pair) for pair in PAIRWISE_MARGINALS] if not args.disable_pairwise else [],
        "test_metrics": metrics_test,
        "posterior_concentration": {
            "overall_mean_std": float(np.mean(posterior_std_post)),
            "per_target_mean_std": np.mean(posterior_std_post, axis=0).astype(float).tolist(),
        },
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
