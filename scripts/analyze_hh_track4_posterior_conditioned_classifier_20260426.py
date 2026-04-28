#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analyze_hh_track4_posterior_decision_rules_20260425 import TARGETS_SCALE, postprocess_targets_array  # noqa: E402


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
RUNALL = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"


class SmallMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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


def random_project(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(x.shape[1], out_dim)).astype(np.float32)
    proj /= np.sqrt(np.sum(np.square(proj), axis=0, keepdims=True))
    return (x @ proj).astype(np.float32)


def train_classifier(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    xtr = torch.tensor(x[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    xte = torch.tensor(x[te], dtype=torch.float32)
    yte = y[te]
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True)
    torch.manual_seed(seed)
    model = SmallMLP(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(30):
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    with torch.no_grad():
        score = model(xte).numpy()
    pred = (score >= 0.0).astype(np.int64)
    acc = float(np.mean(pred == yte))
    auc = auc_from_scores(yte.astype(np.int64), score.astype(np.float64))
    return acc, auc


def discover_run_dirs() -> list[Path]:
    run_dirs: list[Path] = []
    for run_dir in sorted((RUNALL / "runs").glob("*")):
        metrics_file = run_dir / "metrics_summary.json"
        pred_file = run_dir / "predictions_test.npz"
        if not (metrics_file.exists() and pred_file.exists()):
            continue
        metrics = json.loads(metrics_file.read_text())
        framework = str(metrics.get("framework", ""))
        if not (framework.startswith("bayesflow") or framework.startswith("swyft")):
            continue
        run_dirs.append(run_dir)
    return run_dirs


def flatten_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)


def samples_from_bayesflow(arr: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = flatten_x(np.asarray(arr["x_test"], dtype=np.float32))
    samples_norm = np.asarray(arr["posterior_samples_norm"], dtype=np.float32)
    samples = postprocess_targets_array(samples_norm.reshape(-1, samples_norm.shape[-1]), TARGETS_SCALE).reshape(samples_norm.shape)
    target = np.asarray(arr["test_true"], dtype=np.float32)
    return x, samples, target


def samples_from_swyft(arr: np.lib.npyio.NpzFile, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = flatten_x(np.asarray(arr["x_test"], dtype=np.float32))
    theta_bank_norm = np.asarray(arr["theta_bank_norm"], dtype=np.float32)
    theta_bank = postprocess_targets_array(theta_bank_norm, TARGETS_SCALE)
    weights = np.asarray(arr["posterior_weights_1d"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    draws = 64
    samples = np.empty((weights.shape[0], draws, theta_bank.shape[1]), dtype=np.float32)
    for i in range(weights.shape[0]):
        score = np.sum(np.log(np.clip(weights[i], 1.0e-12, None)), axis=1)
        prob = np.exp(score - np.max(score))
        prob /= np.sum(prob)
        idx = rng.choice(theta_bank.shape[0], size=draws, replace=True, p=prob)
        samples[i] = theta_bank[idx]
    target = np.asarray(arr["test_true"], dtype=np.float32)
    return x, samples, target


def evaluate_conditioned(run_name: str, x: np.ndarray, samples: np.ndarray, target: np.ndarray, seed: int) -> dict[str, object]:
    x_proj = random_project(x, out_dim=128, seed=seed)
    rep_x = np.repeat(x_proj, samples.shape[1], axis=0)
    post_theta = samples.reshape(-1, samples.shape[2]).astype(np.float32)
    target_theta = np.repeat(target, samples.shape[1], axis=0).astype(np.float32)
    post_input = np.concatenate([rep_x, post_theta], axis=1)
    target_input = np.concatenate([rep_x, target_theta], axis=1)
    xx = np.concatenate([post_input, target_input], axis=0)
    yy = np.concatenate(
        [np.zeros(post_input.shape[0], dtype=np.int64), np.ones(target_input.shape[0], dtype=np.int64)],
        axis=0,
    )
    acc, auc = train_classifier(xx, yy, seed)
    return {"run_name": run_name, "classifier_accuracy": acc, "classifier_auc": auc}


def try_evaluate_run(run_dir: Path, seed: int) -> dict[str, object] | None:
    arr = np.load(run_dir / "predictions_test.npz")
    if "posterior_samples_norm" in arr.files and "x_test" in arr.files:
        x, samples, target = samples_from_bayesflow(arr)
    elif {"theta_bank_norm", "posterior_weights_1d", "x_test", "test_true"}.issubset(arr.files):
        x, samples, target = samples_from_swyft(arr, seed)
    else:
        return None
    return evaluate_conditioned(run_dir.name, x, samples, target, seed)


def main() -> None:
    rows: list[dict[str, object]] = []
    for i, run_dir in enumerate(discover_run_dirs(), start=1):
        row = try_evaluate_run(run_dir, 20260426 + i)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r["classifier_auc"])
    out_csv = RUNALL / "tables" / "hh_track4_posterior_conditioned_classifier_20260426.csv"
    write_csv(out_csv, rows)
    report = [
        "# HH Track4 Posterior Conditioned Classifier",
        "",
        f"- summary csv: `{out_csv}`",
        "",
        "| run | accuracy | auc |",
        "|---|---:|---:|",
    ]
    for row in rows:
        report.append(f"| `{row['run_name']}` | {row['classifier_accuracy']:.6f} | {row['classifier_auc']:.6f} |")
    (RUNALL / "reports" / "hh_track4_posterior_conditioned_classifier_20260426.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
