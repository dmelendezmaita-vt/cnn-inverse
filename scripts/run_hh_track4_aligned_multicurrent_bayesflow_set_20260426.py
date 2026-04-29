#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

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

    class DType: ...
    class TypeSpec: ...
    class SparseTensor: ...
    class RaggedTensor: ...

    tf_stub.TensorShape = TensorShape
    tf_stub.DType = DType
    tf_stub.TypeSpec = TypeSpec
    tf_stub.SparseTensor = SparseTensor
    tf_stub.RaggedTensor = RaggedTensor
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = tf_stub


_ensure_tensorflow_keras_stub()

import bayesflow as bf

sys.path.insert(0, str(REPO / "src" / "pytorch"))
sys.path.insert(0, str(REPO / "src"))

from data import preprocess_targets  # type: ignore  # noqa: E402
from run_hh_track4_aligned_multicurrent_classical_20260425 import (  # noqa: E402
    load_one_current,
    load_params,
    postprocess_targets_array,
)
from run_hh_track4_aligned_multicurrent_bayesflow_20260425 import metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Set-based BayesFlow aligned multi-current posterior family.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12")
    ap.add_argument("--summary-network", default="set_transformer", choices=["deep_set", "set_transformer"])
    ap.add_argument(
        "--data-dir",
        default="concatenated_data.tar.gz",
    )
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=5.0e-4)
    ap.add_argument("--posterior-samples", type=int, default=128)
    ap.add_argument("--sample-batch-size", type=int, default=32)
    ap.add_argument("--diagnostic-samples", type=int, default=64)
    ap.add_argument("--summary-dim", type=int, default=64)
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            features_by_split[split].append(features_curr[split].astype(np.float32, copy=False))
        if targets_ref is None:
            targets_ref = targets_curr
    assert targets_ref is not None
    targets_scale = preprocess_targets(targets_ref, params, logger)
    x_train = np.stack(features_by_split["train"], axis=1).astype(np.float32, copy=False)
    x_validate = np.stack(features_by_split["validate"], axis=1).astype(np.float32, copy=False)
    x_test = np.stack(features_by_split["test"], axis=1).astype(np.float32, copy=False)
    return currents, targets_scale, x_train, x_validate, x_test, targets_ref["train"], targets_ref["validate"], targets_ref["test"]


def build_workflow(args: argparse.Namespace, save_dir: Path) -> bf.workflows.BasicWorkflow:
    if args.summary_network == "deep_set":
        summary_network = bf.networks.DeepSet(summary_dim=args.summary_dim)
    else:
        summary_network = bf.networks.SetTransformer(summary_dim=args.summary_dim)
    return bf.workflows.BasicWorkflow(
        simulator=None,
        inference_network="coupling_flow",
        summary_network=summary_network,
        initial_learning_rate=args.learning_rate,
        checkpoint_filepath=str(save_dir / "checkpoints"),
        save_weights_only=False,
        save_best_only=True,
        inference_variables=["theta"],
        summary_variables=["x"],
        standardize="inference_variables",
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("bayesflow_set")
    seed_everything(args.seed)
    currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test = build_dataset(args, logger)
    workflow = build_workflow(args, save_dir)
    history = workflow.fit_offline(
        data={"theta": y_train.astype(np.float32, copy=False), "x": x_train.astype(np.float32, copy=False)},
        validation_data={"theta": y_validate.astype(np.float32, copy=False), "x": x_validate.astype(np.float32, copy=False)},
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    posterior_samples = workflow.sample(
        num_samples=args.posterior_samples,
        conditions={"x": x_test.astype(np.float32, copy=False)},
        batch_size=args.sample_batch_size,
        split=False,
    )
    theta_samples = np.asarray(posterior_samples["theta"], dtype=np.float32)
    pred_norm = np.mean(theta_samples, axis=1).astype(np.float32)
    std_norm = np.std(theta_samples, axis=1).astype(np.float32)
    q05_norm = np.quantile(theta_samples, 0.05, axis=1).astype(np.float32)
    q95_norm = np.quantile(theta_samples, 0.95, axis=1).astype(np.float32)
    y_true = postprocess_targets_array(y_test, targets_scale)
    y_pred = postprocess_targets_array(pred_norm, targets_scale)
    std_post = np.abs(postprocess_targets_array(pred_norm + std_norm, targets_scale) - y_pred)
    q05 = postprocess_targets_array(q05_norm, targets_scale)
    q95 = postprocess_targets_array(q95_norm, targets_scale)
    diagnostics = workflow.compute_default_diagnostics(
        {"theta": y_test.astype(np.float32, copy=False), "x": x_test.astype(np.float32, copy=False)},
        num_samples=args.diagnostic_samples,
        as_data_frame=True,
    )
    diagnostics.to_csv(save_dir / "diagnostics.csv", index=False)
    np.savez_compressed(
        save_dir / "predictions_test.npz",
        x_test=x_test.astype(np.float32),
        test_true=y_true,
        test_pred=y_pred,
        test_pred_norm=pred_norm,
        posterior_std_norm=std_norm,
        posterior_std=std_post,
        posterior_q05=q05,
        posterior_q95=q95,
        posterior_samples_norm=theta_samples,
    )
    summary = {
        "framework": "bayesflow_set_posterior",
        "summary_network": args.summary_network,
        "summary_dim": args.summary_dim,
        "currents": currents,
        "feature_mode": args.feature_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "posterior_samples": args.posterior_samples,
        "train_final_loss": float(history.history["loss"][-1]) if "loss" in history.history else None,
        "train_final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "test_metrics": metrics(y_true, y_pred),
        "posterior_concentration": {
            "overall_mean_std": float(np.mean(std_post)),
            "per_target_mean_std": np.mean(std_post, axis=0).astype(float).tolist(),
        },
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
