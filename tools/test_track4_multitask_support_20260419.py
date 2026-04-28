#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "utils"))

from nets import create_network  # noqa: E402
from run_dnn import UncertaintyWeightedMultiTaskLoss  # noqa: E402


class _Logger:
    def info(self, *args, **kwargs):
        pass


def _base_params():
    return {
        "data": {
            "num_features": [1, 256],
            "num_targets": 6,
        },
        "net": {
            "activation_fn": "gelu",
            "conv_layer_sizes": [2, 4],
            "dense_layer_sizes": [16, 16],
            "conv_layer_kernel": 3,
            "conv_layer_stride": 1,
            "conv_pool_type": "avg",
            "conv_pool_kernel": 2,
            "conv_pool_stride": 2,
            "conv_pool_padding": 0,
            "dropout": False,
        },
    }


def main() -> None:
    logger = _Logger()

    params = _base_params()
    params["net"] = dict(params["net"], type="ConvMultiHead", target_head_strategy="per_target", head_dense_layer_sizes=[8])
    net = create_network(params, logger)
    x = torch.randn(3, 1, 256)
    y = net(x)
    if tuple(y.shape) != (3, 6):
        raise AssertionError(f"per_target head shape mismatch: {tuple(y.shape)}")

    params = _base_params()
    params["net"] = dict(
        params["net"],
        type="ConvMultiHead",
        target_head_strategy="grouped",
        head_dense_layer_sizes=[8],
        target_head_groups=[[0, 1], [2, 3], [4, 5]],
    )
    net = create_network(params, logger)
    y = net(x)
    if tuple(y.shape) != (3, 6):
        raise AssertionError(f"grouped head shape mismatch: {tuple(y.shape)}")

    params = _base_params()
    params["net"] = dict(params["net"], type="ConvNet", learn_task_uncertainty=True)
    net = create_network(params, logger)
    if not hasattr(net, "log_task_vars"):
        raise AssertionError("expected learn_task_uncertainty to attach log_task_vars")
    pred = torch.randn(3, 6, requires_grad=True)
    target = torch.randn(3, 6)
    loss_fn = UncertaintyWeightedMultiTaskLoss(net.log_task_vars, base_loss="huber", beta=0.1)
    loss = loss_fn(pred, target)
    loss.backward()
    if net.log_task_vars.grad is None:
        raise AssertionError("expected uncertainty loss to backprop into log_task_vars")

    print("TRACK4_MULTITASK_SUPPORT_TEST_OK")


if __name__ == "__main__":
    main()
