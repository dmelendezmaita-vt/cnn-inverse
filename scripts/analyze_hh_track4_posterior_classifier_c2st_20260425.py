#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analyze_hh_track4_posterior_decision_rules_20260425 import TARGETS_SCALE, postprocess_targets_array


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
RUNALL = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
POOL = REPO / "data/important_notes/optimization_track_20260425_tc_hh_track4_pool_sequential_sbi"


class SmallMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = int(np.sum(y_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[y_sorted == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_classifier(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    xtr = torch.tensor(x[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    xte = torch.tensor(x[te], dtype=torch.float32)
    yte = y[te]
    ds = TensorDataset(xtr, ytr)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    model = SmallMLP(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(25):
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        score = model(xte).cpu().numpy()
    pred = (score >= 0.0).astype(np.int64)
    acc = float(np.mean(pred == yte))
    auc = auc_from_scores(yte.astype(np.int64), score.astype(np.float64))
    return acc, auc


def load_posterior_samples(run_name: str, pred_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(pred_path)
    if "posterior_samples" in arr.files:
        return np.asarray(arr["targets"], dtype=np.float32), np.asarray(arr["posterior_samples"], dtype=np.float32)
    if "posterior_samples_norm" in arr.files:
        samples_norm = np.asarray(arr["posterior_samples_norm"], dtype=np.float32)
        samples = postprocess_targets_array(samples_norm.reshape(-1, samples_norm.shape[-1]), TARGETS_SCALE).reshape(samples_norm.shape)
        return np.asarray(arr["test_true"], dtype=np.float32), samples
    if "test_true" in arr.files and "test_pred" in arr.files and "posterior_std" in arr.files:
        mean = np.asarray(arr["test_pred"], dtype=np.float32)
        std = np.asarray(arr["posterior_std"], dtype=np.float32)
        rng = np.random.default_rng(20260425)
        noise = rng.normal(size=(mean.shape[0], 128, mean.shape[1])).astype(np.float32)
        samples = mean[:, None, :] + np.maximum(std, 1.0e-8)[:, None, :] * noise
        return np.asarray(arr["test_true"], dtype=np.float32), samples.astype(np.float32, copy=False)
    if "theta_bank_norm" in arr.files and "posterior_weights_1d" in arr.files:
        theta_bank_norm = np.asarray(arr["theta_bank_norm"], dtype=np.float32)
        weights = np.asarray(arr["posterior_weights_1d"], dtype=np.float32)
        theta_bank = postprocess_targets_array(theta_bank_norm, TARGETS_SCALE)
        rng = np.random.default_rng(20260425)
        sample_count = 64
        samples = np.empty((weights.shape[0], sample_count, theta_bank.shape[1]), dtype=np.float32)
        for i in range(weights.shape[0]):
            score = np.sum(np.log(np.clip(weights[i], 1.0e-12, None)), axis=1)
            prob = np.exp(score - np.max(score))
            prob /= np.sum(prob)
            idx = rng.choice(theta_bank.shape[0], size=sample_count, replace=True, p=prob)
            samples[i] = theta_bank[idx]
        return np.asarray(arr["test_true"], dtype=np.float32), samples
    raise ValueError(f"{run_name} does not contain usable posterior samples")


def main() -> None:
    specs = {
        "bayesflow_meanstd": RUNALL / "runs/bayesflow_native_meanstd_v100_20260425/predictions_test.npz",
        "bayesflow_currfusion": RUNALL / "runs/bayesflow_native_structured_currfusion_v100_20260425/predictions_test.npz",
        "swyft_aligned_pool": RUNALL / "runs/swyft_native_aligned_currents_pool_v100_20260425/predictions_test.npz",
        "swyft_tmnre_structured": RUNALL / "runs/swyft_tmnre_structured_stage1_v100_20260425/predictions_test.npz",
        "poolseq_fmpe": POOL / "runs/fmpe_mlp_r2_pool4096_final2048/predictions_test.npz",
    }
    rows = []
    for idx, (name, path) in enumerate(specs.items()):
        targets, samples = load_posterior_samples(name, path)
        post = samples.reshape(-1, samples.shape[-1]).astype(np.float32)
        target_rep = np.repeat(targets, samples.shape[1], axis=0).astype(np.float32)
        x = np.concatenate([post, target_rep], axis=0)
        y = np.concatenate([np.zeros(post.shape[0], dtype=np.int64), np.ones(target_rep.shape[0], dtype=np.int64)], axis=0)
        acc, auc = train_classifier(x, y, 20260425 + idx)
        rows.append({"run_name": name, "classifier_accuracy": acc, "classifier_auc": auc, "source": str(path)})
    rows.sort(key=lambda r: r["classifier_auc"])
    out_csv = RUNALL / "tables" / "hh_track4_posterior_classifier_c2st_20260425.csv"
    write_csv(out_csv, rows)
    report = [
        "# HH Track4 Posterior Classifier Diagnostic",
        "",
        f"- summary csv: `{out_csv}`",
        "",
        "| run | accuracy | auc |",
        "|---|---:|---:|",
    ]
    for row in rows:
        report.append(f"| `{row['run_name']}` | {row['classifier_accuracy']:.6f} | {row['classifier_auc']:.6f} |")
    (RUNALL / "reports" / "hh_track4_posterior_classifier_c2st_20260425.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
