#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "src"))

from data import preprocess_targets  # type: ignore  # noqa: E402
from run_hh_track4_aligned_multicurrent_classical_20260425 import (  # noqa: E402
    load_one_current,
    load_params,
    postprocess_targets_array,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Temporal CNN aligned multi-current HH baseline.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--currents", default="0.1,0.2,0.3,0.4,0.5")
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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("Requested GPU but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - float(np.sum(np.square(diff))) / denom if denom > 0.0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def compute_summary12(arr: np.ndarray) -> np.ndarray:
    traces = np.asarray(arr, dtype=np.float32)
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


def load_raw_aligned_dataset(args: argparse.Namespace, logger: logging.Logger):
    params = load_params(args.params)
    currents = [x.strip() for x in args.currents.split(",") if x.strip()]
    raw_by_split = {"train": [], "validate": [], "test": []}
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
            feature_mode="raw",
        )
        for split in raw_by_split:
            raw_by_split[split].append(features_curr[split])
        if targets_ref is None:
            targets_ref = targets_curr
    assert targets_ref is not None
    targets_scale = preprocess_targets(targets_ref, params, logger)
    x_train = np.stack(raw_by_split["train"], axis=1).astype(np.float32, copy=False)
    x_validate = np.stack(raw_by_split["validate"], axis=1).astype(np.float32, copy=False)
    x_test = np.stack(raw_by_split["test"], axis=1).astype(np.float32, copy=False)
    return currents, targets_scale, x_train, x_validate, x_test, targets_ref["train"], targets_ref["validate"], targets_ref["test"]


class TemporalCNNRegressor(nn.Module):
    def __init__(self, num_currents: int, trace_len: int, out_dim: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(num_currents, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(num_currents * 12, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 16 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )
        self.trace_len = trace_len

    def forward(self, x):
        summary = compute_summary12_torch(x)
        z = self.temporal(x).flatten(start_dim=1)
        s = self.summary_proj(summary)
        return self.head(torch.cat([z, s], dim=1))


def compute_summary12_torch(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, T]
    mean = x.mean(dim=2)
    std = x.std(dim=2, unbiased=False)
    minv = x.min(dim=2).values
    maxv = x.max(dim=2).values
    med = x.median(dim=2).values
    q10 = torch.quantile(x, 0.10, dim=2)
    q90 = torch.quantile(x, 0.90, dim=2)
    rms = torch.sqrt(torch.mean(x * x, dim=2))
    diffs = x[:, :, 1:] - x[:, :, :-1]
    mad = torch.mean(torch.abs(diffs), dim=2)
    maxdiff = torch.max(torch.abs(diffs), dim=2).values
    frac_pos = torch.mean((x > 0.0).float(), dim=2)
    frac_gt20 = torch.mean((x > -20.0).float(), dim=2)
    return torch.cat([mean, std, minv, maxv, med, q10, q90, rms, mad, maxdiff, frac_pos, frac_gt20], dim=1)


def train_model(model, train_loader, val_loader, device, epochs, lr, wd):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = loss_fn(model(xb), yb)
                val_loss += float(loss.item()) * xb.shape[0]
                count += xb.shape[0]
        val_loss /= max(count, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("temporal_cnn")
    seed_everything(args.seed)
    device = select_device(args.device)
    currents, targets_scale, x_train, x_validate, x_test, y_train, y_validate, y_test = load_raw_aligned_dataset(args, logger)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validate), torch.from_numpy(y_validate.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = TemporalCNNRegressor(num_currents=x_train.shape[1], trace_len=x_train.shape[2], out_dim=y_train.shape[1])
    t0 = time.time()
    best_val = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.weight_decay)
    train_sec = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_test).to(device)).cpu().numpy().astype(np.float32)
    eval_sec = 0.0
    y_true = postprocess_targets_array(y_test, targets_scale)
    y_pred = postprocess_targets_array(pred_norm, targets_scale)
    m = metrics(y_true, y_pred)
    np.savez_compressed(save_dir / "predictions_test.npz", test_true=y_true, test_pred=y_pred, test_pred_norm=pred_norm)
    summary = {
        "framework": "temporal_cnn_aligned",
        "currents": currents,
        "trace_len": int(x_train.shape[2]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_loss": float(best_val),
        "training_runtime_sec": float(train_sec),
        "evaluation_runtime_sec": float(eval_sec),
        "total_runtime_sec": float(train_sec + eval_sec),
        "test_metrics": m,
    }
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

