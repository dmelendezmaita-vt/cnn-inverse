#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.machinery
import json
import logging
import os
import random
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

REPO = Path(__file__).resolve().parents[1]
for entry in ("", str(REPO)):
    while entry in sys.path:
        sys.path.remove(entry)


def _ensure_tensorflow_keras_stub() -> None:
    try:
        import tensorflow as tf  # type: ignore
        if hasattr(tf, "TensorShape"):
            return
    except Exception:
        pass

    tf_stub = types.ModuleType("tensorflow")

    class TensorShape(tuple):
        def as_list(self):
            return list(self)

    class DType:
        pass

    class TypeSpec:
        pass

    class SparseTensor:
        pass

    class RaggedTensor:
        pass

    tf_stub.TensorShape = TensorShape
    tf_stub.DType = DType
    tf_stub.TypeSpec = TypeSpec
    tf_stub.SparseTensor = SparseTensor
    tf_stub.RaggedTensor = RaggedTensor
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = tf_stub


_ensure_tensorflow_keras_stub()

import bayesflow as bf
import keras
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(REPO / "src"))

from data import preprocess_targets  # type: ignore  # noqa: E402
from run_hh_track4_aligned_multicurrent_classical_20260425 import (  # noqa: E402
    aggregate_currents,
    load_one_current,
    load_params,
    postprocess_targets_array,
)


FLAT_AGGREGATIONS = {"concat", "meanstd"}
STACK_AGGREGATIONS = {"stack"}
FUSION_AGGREGATIONS = {"per_current_fusion"}
STRUCTURED_AGGREGATIONS = STACK_AGGREGATIONS | FUSION_AGGREGATIONS


@dataclass(frozen=True)
class DatasetBundle:
    currents: list[str]
    targets_scale: dict
    observation_train: dict[str, np.ndarray]
    observation_validate: dict[str, np.ndarray]
    observation_test: dict[str, np.ndarray]
    observation_export_test: np.ndarray
    observation_export_train_shape: tuple[int, ...]
    observation_metadata: dict[str, object]
    y_train: np.ndarray
    y_validate: np.ndarray
    y_test: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Native BayesFlow aligned multi-current HH baseline.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12")
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--aggregation", default="concat", choices=["concat", "meanstd", "stack", "per_current_fusion"])
    ap.add_argument(
        "--data-dir",
        default="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track4_hh_full/concatenated_data",
    )
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=5.0e-4)
    ap.add_argument("--posterior-samples", type=int, default=256)
    ap.add_argument("--sample-batch-size", type=int, default=64)
    ap.add_argument("--diagnostic-samples", type=int, default=128)
    ap.add_argument("--inference-network", default="coupling_flow")
    ap.add_argument(
        "--summary-network",
        default="auto",
        choices=[
            "auto",
            "none",
            "deep_set",
            "set_transformer",
            "time_series_network",
            "time_series_transformer",
            "fusion_mlp",
        ],
    )
    ap.add_argument("--summary-dim", type=int, default=96)
    ap.add_argument("--summary-dropout", type=float, default=0.05)
    ap.add_argument("--fusion-backbone-dim", type=int, default=48)
    ap.add_argument("--fusion-encoder-widths", default="256,128")
    ap.add_argument("--fusion-head-widths", default="256,128")
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


def parse_currents(currents_str: str) -> list[str]:
    currents = [x.strip() for x in currents_str.split(",") if x.strip()]
    if not currents:
        raise ValueError("At least one current must be provided.")
    return currents


def parse_widths(spec: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    widths = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    if not widths and not allow_empty:
        raise ValueError(f"Width specification must contain at least one integer, got {spec!r}.")
    return widths


def current_feature_key(curr: str) -> str:
    return "x_curr_" + curr.replace("-", "m").replace(".", "p")


def stack_currents_with_context(per_current_features: list[np.ndarray], current_values: np.ndarray) -> np.ndarray:
    stacked = np.stack(per_current_features, axis=1).astype(np.float32, copy=False)
    current_channel = np.broadcast_to(
        current_values.reshape(1, current_values.shape[0], 1),
        (stacked.shape[0], stacked.shape[1], 1),
    ).astype(np.float32, copy=False)
    return np.concatenate([stacked, current_channel], axis=2).astype(np.float32, copy=False)


def resolve_summary_network_name(args: argparse.Namespace) -> str:
    if args.summary_network != "auto":
        return args.summary_network
    if args.aggregation in FLAT_AGGREGATIONS:
        return "none"
    if args.aggregation in STACK_AGGREGATIONS:
        return "deep_set"
    if args.aggregation in FUSION_AGGREGATIONS:
        return "fusion_mlp"
    raise ValueError(f"Unsupported aggregation={args.aggregation!r}")


def validate_observation_config(args: argparse.Namespace, resolved_summary_network: str) -> None:
    if args.aggregation in FLAT_AGGREGATIONS and resolved_summary_network != "none":
        raise ValueError(
            f"aggregation={args.aggregation!r} produces flat conditions, so summary_network must resolve to 'none'."
        )
    if args.aggregation in STACK_AGGREGATIONS and resolved_summary_network not in {
        "deep_set",
        "set_transformer",
        "time_series_network",
        "time_series_transformer",
    }:
        raise ValueError(
            f"aggregation={args.aggregation!r} requires a structured summary network, got {resolved_summary_network!r}."
        )
    if args.aggregation in FUSION_AGGREGATIONS and resolved_summary_network != "fusion_mlp":
        raise ValueError(
            f"aggregation={args.aggregation!r} currently supports only summary_network='fusion_mlp' or 'auto'."
        )


def build_observation_splits(
    currents: list[str],
    features_by_split: dict[str, list[np.ndarray]],
    aggregation: str,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, tuple[int, ...], dict[str, object]]:
    current_values = np.asarray([float(curr) for curr in currents], dtype=np.float32)
    current_keys = [current_feature_key(curr) for curr in currents]

    observation_splits: dict[str, dict[str, np.ndarray]] = {}
    export_arrays: dict[str, np.ndarray] = {}
    export_train_shape: tuple[int, ...] | None = None

    for split, per_current in features_by_split.items():
        if aggregation in FLAT_AGGREGATIONS:
            x_split = aggregate_currents(per_current, aggregation)
            observation_splits[split] = {"x": x_split}
            export_arrays[split] = x_split
        elif aggregation in STACK_AGGREGATIONS:
            x_split = stack_currents_with_context(per_current, current_values)
            observation_splits[split] = {"x": x_split}
            export_arrays[split] = x_split
        elif aggregation in FUSION_AGGREGATIONS:
            observation_splits[split] = {
                key: np.asarray(feature, dtype=np.float32)
                for key, feature in zip(current_keys, per_current)
            }
            export_arrays[split] = stack_currents_with_context(per_current, current_values)
        else:
            raise ValueError(f"Unsupported aggregation={aggregation!r}")

        if split == "train":
            export_train_shape = tuple(export_arrays[split].shape)

    assert export_train_shape is not None
    observation_metadata = {
        "observation_keys": list(observation_splits["train"].keys()),
        "current_feature_keys": current_keys,
        "current_values": current_values.astype(float).tolist(),
        "per_current_feature_dim": int(features_by_split["train"][0].shape[1]),
        "observation_export_shape_test": list(export_arrays["test"].shape),
    }
    return observation_splits, export_arrays["test"], export_train_shape, observation_metadata


def build_aligned_dataset(args: argparse.Namespace, logger: logging.Logger) -> DatasetBundle:
    params = load_params(args.params)
    currents = parse_currents(args.currents)

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

    observation_splits, observation_export_test, observation_export_train_shape, observation_metadata = build_observation_splits(
        currents,
        features_by_split,
        args.aggregation,
    )
    logger.info(
        "native bayesflow observations built: aggregation=%s feature_mode=%s train_keys=%s export_train_shape=%s export_test_shape=%s",
        args.aggregation,
        args.feature_mode,
        sorted(observation_splits["train"].keys()),
        observation_export_train_shape,
        observation_export_test.shape,
    )
    return DatasetBundle(
        currents=currents,
        targets_scale=targets_scale,
        observation_train=observation_splits["train"],
        observation_validate=observation_splits["validate"],
        observation_test=observation_splits["test"],
        observation_export_test=observation_export_test,
        observation_export_train_shape=observation_export_train_shape,
        observation_metadata=observation_metadata,
        y_train=y_train,
        y_validate=y_validate,
        y_test=y_test,
    )


def build_fusion_adapter(currents: list[str]) -> bf.adapters.Adapter:
    current_keys = [current_feature_key(curr) for curr in currents]
    return (
        bf.adapters.Adapter()
        .convert_dtype(from_dtype="float64", to_dtype="float32")
        .concatenate(["theta"], into="inference_variables")
        .group(current_keys, into="summary_variables")
    )


def build_fusion_summary_network(args: argparse.Namespace, currents: list[str]) -> bf.networks.FusionNetwork:
    encoder_widths = parse_widths(args.fusion_encoder_widths)
    head_widths = parse_widths(args.fusion_head_widths, allow_empty=True)
    activation = "mish"

    backbones = {}
    for curr in currents:
        key = current_feature_key(curr)
        encoder_layers: list[keras.layers.Layer] = [keras.layers.Flatten(name=f"{key}_flatten")]
        if encoder_widths:
            encoder_layers.append(
                bf.networks.MLP(
                    widths=encoder_widths,
                    activation=activation,
                    residual=True,
                    dropout=args.summary_dropout,
                    norm="layer",
                    name=f"{key}_encoder_mlp",
                )
            )
        encoder_layers.append(
            keras.layers.Dense(args.fusion_backbone_dim, activation="linear", name=f"{key}_encoder_out")
        )
        backbones[key] = keras.Sequential(encoder_layers, name=f"{key}_encoder")

    head_layers: list[keras.layers.Layer] = []
    if head_widths:
        head_layers.append(
            bf.networks.MLP(
                widths=head_widths,
                activation=activation,
                residual=True,
                dropout=args.summary_dropout,
                norm="layer",
                name="fusion_head_mlp",
            )
        )
    head_layers.append(keras.layers.Dense(args.summary_dim, activation="linear", name="fusion_summary_out"))
    head = keras.Sequential(head_layers, name="fusion_head")
    return bf.networks.FusionNetwork(backbones=backbones, head=head, name="per_current_fusion_summary")


def build_structured_summary_network(args: argparse.Namespace, resolved_summary_network: str) -> bf.networks.SummaryNetwork | None:
    if resolved_summary_network == "none":
        return None
    if resolved_summary_network == "deep_set":
        return bf.networks.DeepSet(
            summary_dim=args.summary_dim,
            activation="mish",
            dropout=args.summary_dropout,
            mlp_widths_equivariant=(128, 128),
            mlp_widths_invariant_inner=(128, 64),
            mlp_widths_invariant_outer=(128, 64),
            mlp_widths_invariant_last=(128, 128),
        )
    if resolved_summary_network == "set_transformer":
        return bf.networks.SetTransformer(
            summary_dim=args.summary_dim,
            embed_dims=(64, 64),
            num_heads=(4, 4),
            mlp_depths=(2, 2),
            mlp_widths=(128, 128),
            dropout=args.summary_dropout,
        )
    if resolved_summary_network == "time_series_network":
        return bf.networks.TimeSeriesNetwork(
            summary_dim=args.summary_dim,
            filters=(64, 64),
            kernel_sizes=(3, 3),
            strides=(1, 1),
            recurrent_dim=128,
            dropout=args.summary_dropout,
        )
    if resolved_summary_network == "time_series_transformer":
        return bf.networks.TimeSeriesTransformer(
            summary_dim=args.summary_dim,
            embed_dims=(64, 64),
            num_heads=(4, 4),
            mlp_depths=(2, 2),
            mlp_widths=(128, 128),
            dropout=args.summary_dropout,
            time_embedding=None,
        )
    raise ValueError(f"Unsupported summary_network={resolved_summary_network!r}")


def build_workflow(
    args: argparse.Namespace,
    save_dir: Path,
    currents: list[str],
    resolved_summary_network: str,
) -> bf.workflows.BasicWorkflow:
    validate_observation_config(args, resolved_summary_network)
    adapter = None
    summary_network = None
    summary_variables = None
    if args.aggregation in FUSION_AGGREGATIONS:
        adapter = build_fusion_adapter(currents)
        summary_network = build_fusion_summary_network(args, currents)
        summary_variables = [current_feature_key(curr) for curr in currents]
    else:
        summary_network = build_structured_summary_network(args, resolved_summary_network)
        if summary_network is not None:
            summary_variables = ["x"]
    return bf.workflows.BasicWorkflow(
        simulator=None,
        adapter=adapter,
        inference_network=args.inference_network,
        summary_network=summary_network,
        initial_learning_rate=args.learning_rate,
        checkpoint_filepath=str(save_dir / "checkpoints"),
        save_weights_only=False,
        save_best_only=True,
        inference_variables=["theta"],
        inference_conditions=["x"] if summary_network is None else None,
        summary_variables=summary_variables,
        standardize="inference_variables",
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("aligned_multicurrent_bayesflow_native")

    seed_everything(args.seed)
    resolved_summary_network = resolve_summary_network_name(args)
    dataset = build_aligned_dataset(args, logger)
    workflow = build_workflow(args, save_dir, dataset.currents, resolved_summary_network)

    train_data = {
        "theta": dataset.y_train.astype(np.float32, copy=False),
        **{
            key: value.astype(np.float32, copy=False)
            for key, value in dataset.observation_train.items()
        },
    }
    validate_data = {
        "theta": dataset.y_validate.astype(np.float32, copy=False),
        **{
            key: value.astype(np.float32, copy=False)
            for key, value in dataset.observation_validate.items()
        },
    }

    started = time.time()
    history = workflow.fit_offline(
        data=train_data,
        validation_data=validate_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    training_runtime_sec = time.time() - started

    started = time.time()
    posterior_samples = workflow.sample(
        num_samples=args.posterior_samples,
        conditions={key: value.astype(np.float32, copy=False) for key, value in dataset.observation_test.items()},
        batch_size=args.sample_batch_size,
        split=False,
    )
    evaluation_runtime_sec = time.time() - started

    theta_samples = np.asarray(posterior_samples["theta"], dtype=np.float32)
    y_pred_norm = np.mean(theta_samples, axis=1).astype(np.float32, copy=False)
    posterior_std_norm = np.std(theta_samples, axis=1).astype(np.float32, copy=False)
    posterior_q05_norm = np.quantile(theta_samples, 0.05, axis=1).astype(np.float32, copy=False)
    posterior_q95_norm = np.quantile(theta_samples, 0.95, axis=1).astype(np.float32, copy=False)

    y_test_post = postprocess_targets_array(dataset.y_test, dataset.targets_scale)
    y_pred_post = postprocess_targets_array(y_pred_norm, dataset.targets_scale)
    posterior_std_post = np.abs(
        postprocess_targets_array(y_pred_norm + posterior_std_norm, dataset.targets_scale) - y_pred_post
    )
    posterior_q05_post = postprocess_targets_array(posterior_q05_norm, dataset.targets_scale)
    posterior_q95_post = postprocess_targets_array(posterior_q95_norm, dataset.targets_scale)

    metrics_test = metrics(y_test_post, y_pred_post)
    total_runtime_sec = training_runtime_sec + evaluation_runtime_sec

    diagnostics = workflow.compute_default_diagnostics(
        {
            "theta": dataset.y_test.astype(np.float32, copy=False),
            **{
                key: value.astype(np.float32, copy=False)
                for key, value in dataset.observation_test.items()
            },
        },
        num_samples=args.diagnostic_samples,
        as_data_frame=True,
    )
    diagnostics_path = save_dir / "diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    with (save_dir / "metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "mse", "mae", "r2"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"split": "test", **metrics_test})

    np.savez_compressed(
        save_dir / "predictions_test.npz",
        x_test=dataset.observation_export_test.astype(np.float32, copy=False),
        x_test_current_values=np.asarray(dataset.observation_metadata["current_values"], dtype=np.float32),
        test_true=y_test_post,
        test_true_norm=dataset.y_test.astype(np.float32, copy=False),
        test_pred=y_pred_post,
        test_pred_norm=y_pred_norm,
        posterior_std_norm=posterior_std_norm,
        posterior_std=posterior_std_post,
        posterior_q05=posterior_q05_post,
        posterior_q95=posterior_q95_post,
        posterior_samples_norm=theta_samples,
    )

    summary = {
        "framework": "bayesflow_native",
        "backend": os.environ.get("KERAS_BACKEND", "torch"),
        "feature_mode": args.feature_mode,
        "aggregation": args.aggregation,
        "currents": dataset.currents,
        "seed": args.seed,
        "summary_network_requested": args.summary_network,
        "summary_network_resolved": resolved_summary_network,
        "summary_dim": int(args.summary_dim) if resolved_summary_network != "none" else None,
        "summary_dropout": float(args.summary_dropout),
        "fusion_backbone_dim": int(args.fusion_backbone_dim) if args.aggregation in FUSION_AGGREGATIONS else None,
        "fusion_encoder_widths": list(parse_widths(args.fusion_encoder_widths))
        if args.aggregation in FUSION_AGGREGATIONS
        else None,
        "fusion_head_widths": list(parse_widths(args.fusion_head_widths))
        if args.aggregation in FUSION_AGGREGATIONS
        else None,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "n_train": int(dataset.y_train.shape[0]),
        "n_validate": int(dataset.y_validate.shape[0]),
        "n_test": int(dataset.y_test.shape[0]),
        "feature_dim": int(np.prod(dataset.observation_export_train_shape[1:])),
        "feature_shape": list(dataset.observation_export_train_shape[1:]),
        "observation_keys": dataset.observation_metadata["observation_keys"],
        "current_feature_keys": dataset.observation_metadata["current_feature_keys"],
        "per_current_feature_dim": int(dataset.observation_metadata["per_current_feature_dim"]),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "posterior_samples": int(args.posterior_samples),
        "training_runtime_sec": float(training_runtime_sec),
        "evaluation_runtime_sec": float(evaluation_runtime_sec),
        "total_runtime_sec": float(total_runtime_sec),
        "train_history_keys": list(history.history.keys()),
        "train_final_loss": float(history.history["loss"][-1]) if "loss" in history.history else None,
        "train_final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "test_metrics": metrics_test,
        "posterior_concentration": {
            "overall_mean_std": float(np.mean(posterior_std_post)),
            "per_target_mean_std": np.mean(posterior_std_post, axis=0).astype(float).tolist(),
        },
        "diagnostics_path": str(diagnostics_path),
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
