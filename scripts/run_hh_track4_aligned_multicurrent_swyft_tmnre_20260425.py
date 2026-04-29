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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))

from run_hh_track4_aligned_multicurrent_swyft_20260425 import (  # noqa: E402
    DEFAULT_STRUCTURED_HIDDEN_FEATURES,
    MODEL_VARIANTS,
    HHNativeSwyftModule,
    build_aligned_dataset,
    build_swyft_model,
    infer_posterior_means,
    postprocess_targets_array,
)


DEFAULT_BOUND_QUANTILES = "0.25:0.75,0.15:0.85,0.10:0.90,0.05:0.95,0.0:1.0"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explicit TMNRE-style truncation follow-up for native Swyft.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument(
        "--stage1-run-dir",
        default="",
    )
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
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--learning-rate", type=float, default=1.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-2)
    ap.add_argument("--hidden-features", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--inference-batch-size", type=int, default=128)
    ap.add_argument("--bound-threshold", type=float, default=1.0e-3)
    ap.add_argument("--bound-observations", type=int, default=128)
    ap.add_argument("--min-retained-frac", type=float, default=0.25)
    ap.add_argument("--max-retained-frac", type=float, default=1.0)
    ap.add_argument("--bound-quantiles", default=DEFAULT_BOUND_QUANTILES)
    ap.add_argument("--schedule-matrix", default="")
    ap.add_argument("--schedule-matrix-file", default="")
    ap.add_argument("--model-variant", default="auto", choices=("auto", *MODEL_VARIANTS))
    ap.add_argument("--structured-hidden-features", type=int, default=None)
    ap.add_argument("--disable-pairwise", action="store_true")
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_accelerator(requested: str) -> tuple[str, int]:
    if requested == "cpu":
        return "cpu", 1
    if not torch.cuda.is_available():
        raise SystemExit("Requested GPU but torch.cuda.is_available() is False in the Swyft env.")
    return "gpu", 1


def maybe_load_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse JSON from {path}: {exc}") from exc


def summary_value_or_default(summary: dict[str, object], key: str, default: object) -> object:
    value = summary.get(key, default)
    return default if value is None else value


def parse_quantile_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(f"Invalid quantile pair '{token}', expected q_low:q_high.")
        q_low_text, q_high_text = token.split(":", 1)
        q_low = float(q_low_text)
        q_high = float(q_high_text)
        if not (0.0 <= q_low <= q_high <= 1.0):
            raise SystemExit(f"Quantile pair must satisfy 0 <= q_low <= q_high <= 1, got {token}.")
        pairs.append((q_low, q_high))
    if not pairs:
        raise SystemExit("At least one quantile pair is required.")
    return pairs


def normalize_quantile_pairs(value: object, default_pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if value is None:
        return list(default_pairs)
    if isinstance(value, str):
        return parse_quantile_pairs(value)
    if isinstance(value, (list, tuple)):
        pairs: list[tuple[float, float]] = []
        for item in value:
            if isinstance(item, str):
                pairs.extend(parse_quantile_pairs(item))
            elif isinstance(item, dict):
                if "q_low" not in item or "q_high" not in item:
                    raise SystemExit(f"Quantile dict entries must contain q_low and q_high, got {item}.")
                q_low = float(item["q_low"])
                q_high = float(item["q_high"])
                if not (0.0 <= q_low <= q_high <= 1.0):
                    raise SystemExit(f"Invalid quantile dict entry: {item}")
                pairs.append((q_low, q_high))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                q_low = float(item[0])
                q_high = float(item[1])
                if not (0.0 <= q_low <= q_high <= 1.0):
                    raise SystemExit(f"Invalid quantile pair: {item}")
                pairs.append((q_low, q_high))
            else:
                raise SystemExit(f"Unsupported quantile pair entry: {item}")
        if pairs:
            return pairs
    raise SystemExit(f"Unsupported quantile specification: {value}")


def normalize_schedule_row(
    row: object,
    *,
    idx: int,
    args: argparse.Namespace,
    default_pairs: list[tuple[float, float]],
) -> dict[str, object]:
    stage_name = f"stage{idx + 1}"
    if isinstance(row, dict):
        bound_threshold = float(row.get("bound_threshold", args.bound_threshold))
        bound_observations = int(row.get("bound_observations", args.bound_observations))
        min_retained_frac = float(row.get("min_retained_frac", args.min_retained_frac))
        max_retained_frac = float(row.get("max_retained_frac", args.max_retained_frac))
        quantiles_value = row.get("quantiles")
        if quantiles_value is None and "q_low" in row and "q_high" in row:
            quantiles_value = [(row["q_low"], row["q_high"])]
        quantile_pairs = normalize_quantile_pairs(quantiles_value, default_pairs)
        if "name" in row:
            stage_name = str(row["name"])
    elif isinstance(row, (list, tuple)):
        if len(row) == 6:
            bound_threshold, bound_observations, min_retained_frac, max_retained_frac, q_low, q_high = row
            quantile_pairs = normalize_quantile_pairs([(q_low, q_high)], default_pairs)
        elif len(row) == 7:
            stage_name, bound_threshold, bound_observations, min_retained_frac, max_retained_frac, q_low, q_high = row
            stage_name = str(stage_name)
            quantile_pairs = normalize_quantile_pairs([(q_low, q_high)], default_pairs)
        else:
            raise SystemExit(
                "Schedule matrix row lists must have 6 columns "
                "[bound_threshold, bound_observations, min_retained_frac, max_retained_frac, q_low, q_high] "
                "or 7 columns with a leading stage name."
            )
        bound_threshold = float(bound_threshold)
        bound_observations = int(bound_observations)
        min_retained_frac = float(min_retained_frac)
        max_retained_frac = float(max_retained_frac)
    else:
        raise SystemExit(f"Unsupported schedule matrix row type: {type(row).__name__}")

    if bound_threshold <= 0.0:
        raise SystemExit(f"Schedule stage {stage_name} must use a positive bound_threshold, got {bound_threshold}.")
    if bound_observations <= 0:
        raise SystemExit(f"Schedule stage {stage_name} must use positive bound_observations, got {bound_observations}.")
    if not (0.0 < min_retained_frac <= max_retained_frac <= 1.0):
        raise SystemExit(
            f"Schedule stage {stage_name} must satisfy 0 < min_retained_frac <= max_retained_frac <= 1, "
            f"got min={min_retained_frac} max={max_retained_frac}."
        )
    return {
        "name": stage_name,
        "bound_threshold": bound_threshold,
        "bound_observations": bound_observations,
        "min_retained_frac": min_retained_frac,
        "max_retained_frac": max_retained_frac,
        "quantile_pairs": quantile_pairs,
    }


def load_schedule_matrix(args: argparse.Namespace) -> list[dict[str, object]]:
    default_pairs = parse_quantile_pairs(args.bound_quantiles)
    raw: object | None = None
    if args.schedule_matrix_file:
        raw = maybe_load_json(Path(args.schedule_matrix_file))
    elif args.schedule_matrix:
        schedule_path = Path(args.schedule_matrix)
        if schedule_path.exists():
            raw = maybe_load_json(schedule_path)
        else:
            try:
                raw = json.loads(args.schedule_matrix)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Could not parse --schedule-matrix JSON: {exc}") from exc

    if raw is None:
        return [normalize_schedule_row({}, idx=0, args=args, default_pairs=default_pairs)]

    if isinstance(raw, dict):
        rows = raw.get("stages")
        if rows is None:
            raise SystemExit("Schedule matrix JSON objects must contain a top-level 'stages' array.")
    elif isinstance(raw, list):
        rows = raw
    else:
        raise SystemExit("Schedule matrix must be a JSON list or a JSON object with a 'stages' array.")

    if not rows:
        raise SystemExit("Schedule matrix must contain at least one stage.")
    return [normalize_schedule_row(row, idx=idx, args=args, default_pairs=default_pairs) for idx, row in enumerate(rows)]


def find_checkpoint(stage1_run_dir: Path) -> Path:
    candidates = sorted((stage1_run_dir / "checkpoints").glob("*.ckpt"))
    if not candidates:
        raise SystemExit(f"No stage1 checkpoint found under {stage1_run_dir}/checkpoints")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_stage1_summary(
    *,
    stage1_summary: dict[str, object],
    args: argparse.Namespace,
    currents: list[str],
    x_train: np.ndarray,
) -> None:
    if not stage1_summary:
        return
    if "feature_mode" in stage1_summary and stage1_summary["feature_mode"] != args.feature_mode:
        raise SystemExit(
            f"Stage1 checkpoint feature_mode={stage1_summary['feature_mode']} does not match requested feature_mode={args.feature_mode}."
        )
    if "aggregation" in stage1_summary and stage1_summary["aggregation"] != args.aggregation:
        raise SystemExit(
            f"Stage1 checkpoint aggregation={stage1_summary['aggregation']} does not match requested aggregation={args.aggregation}."
        )
    if "currents" in stage1_summary and list(stage1_summary["currents"]) != list(currents):
        raise SystemExit(
            f"Stage1 checkpoint currents={stage1_summary['currents']} do not match requested currents={currents}."
        )
    if "feature_dim" in stage1_summary and int(stage1_summary["feature_dim"]) != int(x_train.shape[1]):
        raise SystemExit(
            f"Stage1 checkpoint feature_dim={stage1_summary['feature_dim']} does not match current input dim={x_train.shape[1]}."
        )


def resolve_stage1_model_config(
    args: argparse.Namespace,
    stage1_summary: dict[str, object],
) -> dict[str, object]:
    model_variant = str(stage1_summary.get("model_variant", "flat"))
    if not stage1_summary and args.model_variant != "auto":
        model_variant = args.model_variant
    structured_hidden_features = int(
        summary_value_or_default(stage1_summary, "structured_hidden_features", DEFAULT_STRUCTURED_HIDDEN_FEATURES)
    )
    if not stage1_summary and args.structured_hidden_features is not None:
        structured_hidden_features = int(args.structured_hidden_features)
    return {
        "learning_rate": float(stage1_summary.get("learning_rate", args.learning_rate)),
        "weight_decay": float(stage1_summary.get("weight_decay", args.weight_decay)),
        "hidden_features": int(stage1_summary.get("hidden_features", args.hidden_features)),
        "num_blocks": int(stage1_summary.get("num_blocks", args.num_blocks)),
        "dropout": float(stage1_summary.get("dropout", args.dropout)),
        "include_pairwise": bool(stage1_summary.get("pairwise_marginals_enabled", True)),
        "model_variant": model_variant,
        "structured_hidden_features": structured_hidden_features,
    }


def resolve_followup_model_config(
    args: argparse.Namespace,
    stage1_summary: dict[str, object],
) -> dict[str, object]:
    stage1_variant = str(stage1_summary.get("model_variant", "flat"))
    model_variant = stage1_variant if args.model_variant == "auto" else args.model_variant
    stage1_pairwise = bool(stage1_summary.get("pairwise_marginals_enabled", True))
    structured_hidden_features = (
        int(args.structured_hidden_features)
        if args.structured_hidden_features is not None
        else int(summary_value_or_default(stage1_summary, "structured_hidden_features", DEFAULT_STRUCTURED_HIDDEN_FEATURES))
    )
    return {
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "hidden_features": int(args.hidden_features),
        "num_blocks": int(args.num_blocks),
        "dropout": float(args.dropout),
        "include_pairwise": bool(stage1_pairwise and not args.disable_pairwise),
        "model_variant": model_variant,
        "structured_hidden_features": structured_hidden_features,
    }


def load_stage1_model(
    *,
    x_dim: int,
    z_dim: int,
    checkpoint_path: Path,
    dataset_meta: dict[str, object],
    model_config: dict[str, object],
) -> HHNativeSwyftModule:
    model = build_swyft_model(
        num_features=x_dim,
        num_params=z_dim,
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        hidden_features=int(model_config["hidden_features"]),
        num_blocks=int(model_config["num_blocks"]),
        dropout=float(model_config["dropout"]),
        include_pairwise=bool(model_config["include_pairwise"]),
        model_variant=str(model_config["model_variant"]),
        dataset_meta=dataset_meta,
        structured_hidden_features=int(model_config["structured_hidden_features"]),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    return model


def collect_rect_bounds(
    *,
    trainer: swyft.SwyftTrainer,
    model: HHNativeSwyftModule,
    x_validate: np.ndarray,
    theta_bank: np.ndarray,
    inference_batch_size: int,
    bound_threshold: float,
    bound_observations: int,
    logger: logging.Logger,
    stage_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    bank = swyft.Samples({"z": theta_bank.astype(np.float32, copy=False)})
    num_obs = min(bound_observations, x_validate.shape[0])
    if num_obs <= 0:
        raise SystemExit(f"TMNRE schedule stage {stage_name} has no validation observations available for bound estimation.")

    lower_rows = []
    upper_rows = []
    for idx in range(num_obs):
        if idx == 0 or (idx + 1) % 16 == 0 or idx + 1 == num_obs:
            logger.info("TMNRE stage %s bound inference for validation observation %d/%d", stage_name, idx + 1, num_obs)
        obs = swyft.Sample(x=np.asarray(x_validate[idx], dtype=np.float32))
        outputs = trainer.infer(model, obs, bank, batch_size=inference_batch_size)
        lrs_1d = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        rect = swyft.get_rect_bounds(lrs_1d, threshold=bound_threshold)
        bounds = rect.bounds.detach().cpu().numpy()
        lower_rows.append(bounds[:, 0, 0])
        upper_rows.append(bounds[:, 0, 1])
    return np.asarray(lower_rows, dtype=np.float32), np.asarray(upper_rows, dtype=np.float32)


def choose_bounded_global_bounds(
    *,
    lower_rows: np.ndarray,
    upper_rows: np.ndarray,
    theta_bank: np.ndarray,
    quantile_pairs: list[tuple[float, float]],
    min_retained_frac: float,
    max_retained_frac: float,
) -> dict[str, object]:
    full_lower = np.min(theta_bank, axis=0)
    full_upper = np.max(theta_bank, axis=0)
    best_candidate: dict[str, object] | None = None
    best_distance = float("inf")
    candidate_summaries = []

    for q_low, q_high in quantile_pairs:
        global_lower = np.quantile(lower_rows, q_low, axis=0).astype(np.float32)
        global_upper = np.quantile(upper_rows, q_high, axis=0).astype(np.float32)
        global_lower = np.maximum(global_lower, full_lower)
        global_upper = np.minimum(global_upper, full_upper)
        global_upper = np.maximum(global_upper, global_lower)
        mask = np.all((theta_bank >= global_lower) & (theta_bank <= global_upper), axis=1)
        retained_frac = float(np.mean(mask))
        if retained_frac < min_retained_frac:
            status = "below_min_retained_frac"
            distance = min_retained_frac - retained_frac
        elif retained_frac > max_retained_frac:
            status = "above_max_retained_frac"
            distance = retained_frac - max_retained_frac
        else:
            status = "within_bounds"
            distance = 0.0

        candidate = {
            "q_low": float(q_low),
            "q_high": float(q_high),
            "lower": global_lower,
            "upper": global_upper,
            "mask": mask,
            "retained_frac": retained_frac,
            "status": status,
        }
        candidate_summaries.append(
            {
                "q_low": float(q_low),
                "q_high": float(q_high),
                "retained_frac": retained_frac,
                "status": status,
            }
        )
        if distance < best_distance:
            best_candidate = candidate
            best_distance = distance
        if distance == 0.0:
            break

    assert best_candidate is not None
    best_candidate["candidate_summaries"] = candidate_summaries
    return best_candidate


def derive_global_bounds(
    *,
    trainer: swyft.SwyftTrainer,
    model: HHNativeSwyftModule,
    x_validate: np.ndarray,
    theta_bank: np.ndarray,
    inference_batch_size: int,
    stage_spec: dict[str, object],
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, object]]:
    lower_rows, upper_rows = collect_rect_bounds(
        trainer=trainer,
        model=model,
        x_validate=x_validate,
        theta_bank=theta_bank,
        inference_batch_size=inference_batch_size,
        bound_threshold=float(stage_spec["bound_threshold"]),
        bound_observations=int(stage_spec["bound_observations"]),
        logger=logger,
        stage_name=str(stage_spec["name"]),
    )
    chosen = choose_bounded_global_bounds(
        lower_rows=lower_rows,
        upper_rows=upper_rows,
        theta_bank=theta_bank,
        quantile_pairs=list(stage_spec["quantile_pairs"]),
        min_retained_frac=float(stage_spec["min_retained_frac"]),
        max_retained_frac=float(stage_spec["max_retained_frac"]),
    )
    meta = {
        "name": str(stage_spec["name"]),
        "bound_threshold": float(stage_spec["bound_threshold"]),
        "bound_observations": int(lower_rows.shape[0]),
        "min_retained_frac": float(stage_spec["min_retained_frac"]),
        "max_retained_frac": float(stage_spec["max_retained_frac"]),
        "q_low": float(chosen["q_low"]),
        "q_high": float(chosen["q_high"]),
        "retained_frac": float(chosen["retained_frac"]),
        "selection_status": str(chosen["status"]),
        "global_lower": np.asarray(chosen["lower"], dtype=np.float32).astype(float).tolist(),
        "global_upper": np.asarray(chosen["upper"], dtype=np.float32).astype(float).tolist(),
        "candidate_quantiles": chosen["candidate_summaries"],
    }
    return np.asarray(chosen["mask"], dtype=bool), meta


def fit_stage_model(
    *,
    save_dir: Path,
    stage_label: str,
    accelerator: str,
    devices: int,
    args: argparse.Namespace,
    dataset_meta: dict[str, object],
    model_config: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validate: np.ndarray,
    y_validate: np.ndarray,
) -> tuple[HHNativeSwyftModule, swyft.SwyftTrainer, float]:
    train_samples = swyft.Samples({"x": x_train.astype(np.float32, copy=False), "z": y_train.astype(np.float32, copy=False)})
    validate_samples = swyft.Samples(
        {"x": x_validate.astype(np.float32, copy=False), "z": y_validate.astype(np.float32, copy=False)}
    )
    train_loader = train_samples.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validate_loader = validate_samples.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_swyft_model(
        num_features=x_train.shape[1],
        num_params=y_train.shape[1],
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        hidden_features=int(model_config["hidden_features"]),
        num_blocks=int(model_config["num_blocks"]),
        dropout=float(model_config["dropout"]),
        include_pairwise=bool(model_config["include_pairwise"]),
        model_variant=str(model_config["model_variant"]),
        dataset_meta=dataset_meta,
        structured_hidden_features=int(model_config["structured_hidden_features"]),
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
    started = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validate_loader)
    return model, trainer, time.time() - started


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("aligned_multicurrent_swyft_tmnre")

    seed_everything(args.seed)
    accelerator, devices = select_accelerator(args.device)
    schedule_matrix = load_schedule_matrix(args)
    currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test, dataset_meta = build_aligned_dataset(
        args, logger
    )

    stage1_run_dir = Path(args.stage1_run_dir)
    stage1_summary_raw = maybe_load_json(stage1_run_dir / "metrics_summary.json")
    if stage1_summary_raw and not isinstance(stage1_summary_raw, dict):
        raise SystemExit(f"Expected a JSON object in {stage1_run_dir / 'metrics_summary.json'}, got {type(stage1_summary_raw).__name__}.")
    stage1_summary = stage1_summary_raw if isinstance(stage1_summary_raw, dict) else {}
    validate_stage1_summary(stage1_summary=stage1_summary, args=args, currents=currents, x_train=x_train)
    stage1_model_config = resolve_stage1_model_config(args, stage1_summary)
    followup_model_config = resolve_followup_model_config(args, stage1_summary)
    checkpoint_path = find_checkpoint(stage1_run_dir)
    stage1_model = load_stage1_model(
        x_dim=x_train.shape[1],
        z_dim=y_train.shape[1],
        checkpoint_path=checkpoint_path,
        dataset_meta=dataset_meta,
        model_config=stage1_model_config,
    )
    stage1_trainer = swyft.SwyftTrainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        default_root_dir=str(save_dir / "stage1_infer"),
    )

    current_x_train = x_train.astype(np.float32, copy=False)
    current_y_train = y_train.astype(np.float32, copy=False)
    current_x_validate = x_validate.astype(np.float32, copy=False)
    current_y_validate = y_validate.astype(np.float32, copy=False)
    current_theta_bank = np.concatenate([y_train, y_validate], axis=0).astype(np.float32, copy=False)
    current_model = stage1_model
    current_trainer = stage1_trainer
    training_runtime_sec = 0.0
    stage_summaries = []

    for stage_idx, stage_spec in enumerate(schedule_matrix, start=1):
        n_train_in = int(current_x_train.shape[0])
        n_validate_in = int(current_x_validate.shape[0])
        n_bank_in = int(current_theta_bank.shape[0])
        bank_mask, bounds_meta = derive_global_bounds(
            trainer=current_trainer,
            model=current_model,
            x_validate=current_x_validate,
            theta_bank=current_theta_bank,
            inference_batch_size=args.inference_batch_size,
            stage_spec=stage_spec,
            logger=logger,
        )
        lower = np.asarray(bounds_meta["global_lower"], dtype=np.float32)
        upper = np.asarray(bounds_meta["global_upper"], dtype=np.float32)
        train_mask = np.all((current_y_train >= lower) & (current_y_train <= upper), axis=1)
        validate_mask = np.all((current_y_validate >= lower) & (current_y_validate <= upper), axis=1)
        if not np.any(train_mask) or not np.any(validate_mask):
            raise SystemExit(f"TMNRE schedule stage {bounds_meta['name']} produced an empty train or validate split.")

        current_x_train = current_x_train[train_mask].astype(np.float32, copy=False)
        current_y_train = current_y_train[train_mask].astype(np.float32, copy=False)
        current_x_validate = current_x_validate[validate_mask].astype(np.float32, copy=False)
        current_y_validate = current_y_validate[validate_mask].astype(np.float32, copy=False)
        current_theta_bank = current_theta_bank[bank_mask].astype(np.float32, copy=False)

        logger.info(
            "TMNRE schedule stage %d/%d name=%s retained train=%d/%d validate=%d/%d bank=%d/%d selected=%s q=(%.2f, %.2f)",
            stage_idx,
            len(schedule_matrix),
            bounds_meta["name"],
            current_x_train.shape[0],
            n_train_in,
            current_x_validate.shape[0],
            n_validate_in,
            current_theta_bank.shape[0],
            n_bank_in,
            bounds_meta["selection_status"],
            bounds_meta["q_low"],
            bounds_meta["q_high"],
        )

        fit_stage_label = f"stage{stage_idx + 1}_fit"
        current_model, current_trainer, stage_training_runtime = fit_stage_model(
            save_dir=save_dir,
            stage_label=fit_stage_label,
            accelerator=accelerator,
            devices=devices,
            args=args,
            dataset_meta=dataset_meta,
            model_config=followup_model_config,
            x_train=current_x_train,
            y_train=current_y_train,
            x_validate=current_x_validate,
            y_validate=current_y_validate,
        )
        training_runtime_sec += stage_training_runtime
        stage_summaries.append(
            {
                **bounds_meta,
                "fit_stage_label": fit_stage_label,
                "n_train_in": n_train_in,
                "n_validate_in": n_validate_in,
                "n_bank_in": n_bank_in,
                "n_train_out": int(current_x_train.shape[0]),
                "n_validate_out": int(current_x_validate.shape[0]),
                "n_bank_out": int(current_theta_bank.shape[0]),
                "training_runtime_sec": float(stage_training_runtime),
            }
        )

    y_pred_norm, posterior_std_norm, posterior_quantiles_norm, posterior_weights_1d, evaluation_runtime_sec = infer_posterior_means(
        trainer=current_trainer,
        model=current_model,
        x_test=x_test,
        theta_bank=current_theta_bank,
        inference_batch_size=args.inference_batch_size,
        logger=logger,
    )

    y_test_post = postprocess_targets_array(y_test, targets_scale)
    y_pred_post = postprocess_targets_array(y_pred_norm, targets_scale)
    posterior_std_post = np.abs(postprocess_targets_array(y_pred_norm + posterior_std_norm, targets_scale) - y_pred_post)
    posterior_q05_post = postprocess_targets_array(posterior_quantiles_norm[:, 0, :], targets_scale)
    posterior_q95_post = postprocess_targets_array(posterior_quantiles_norm[:, 1, :], targets_scale)
    metrics_test = metrics(y_test_post, y_pred_post)

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
        theta_bank_norm=current_theta_bank.astype(np.float32, copy=False),
        posterior_weights_1d=posterior_weights_1d.astype(np.float32, copy=False),
    )

    total_runtime_sec = training_runtime_sec + evaluation_runtime_sec
    summary = {
        "framework": "swyft_tmnre_style_native",
        "stage1_checkpoint": str(checkpoint_path),
        "feature_mode": args.feature_mode,
        "aggregation": args.aggregation,
        "currents": currents,
        "seed": args.seed,
        "device": accelerator,
        "n_train_full": int(x_train.shape[0]),
        "n_validate_full": int(x_validate.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_train_truncated": int(current_x_train.shape[0]),
        "n_validate_truncated": int(current_x_validate.shape[0]),
        "posterior_bank_size_truncated": int(current_theta_bank.shape[0]),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "hidden_features": int(args.hidden_features),
        "num_blocks": int(args.num_blocks),
        "dropout": float(args.dropout),
        "stage1_model_variant": str(stage1_model_config["model_variant"]),
        "model_variant": str(followup_model_config["model_variant"]),
        "structured_hidden_features": (
            int(followup_model_config["structured_hidden_features"])
            if str(followup_model_config["model_variant"]) != "flat"
            else None
        ),
        "logratio_feature_dim": int(current_model.encoded_num_features),
        "num_currents": int(dataset_meta["num_currents"]),
        "per_current_feature_dim": int(dataset_meta["per_current_feature_dim"]),
        "normalized_current_values": dataset_meta["normalized_current_values"],
        "pairwise_marginals_enabled": bool(followup_model_config["include_pairwise"]),
        "training_runtime_sec": float(training_runtime_sec),
        "evaluation_runtime_sec": float(evaluation_runtime_sec),
        "total_runtime_sec": float(total_runtime_sec),
        "bounds": stage_summaries[-1],
        "bound_schedule_requested": schedule_matrix,
        "bound_schedule": stage_summaries,
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
