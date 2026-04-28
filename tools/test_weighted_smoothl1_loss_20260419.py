#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

import torch

RUN_DNN = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/src/pytorch/run_dnn.py")


def load_weighted_smoothl1_loss():
    module = ast.parse(RUN_DNN.read_text(), filename=str(RUN_DNN))
    target = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "WeightedSmoothL1Loss":
            target = node
            break
    if target is None:
        raise AssertionError("WeightedSmoothL1Loss class not found in run_dnn.py")

    wrapper = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(wrapper)
    namespace = {"torch": torch}
    exec(compile(wrapper, str(RUN_DNN), "exec"), namespace)
    return namespace["WeightedSmoothL1Loss"]


def main() -> None:
    WeightedSmoothL1Loss = load_weighted_smoothl1_loss()

    pred = torch.tensor([[0.0, 2.0], [4.0, 8.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0], [5.0, 10.0]], dtype=torch.float32)
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    loss_fn = WeightedSmoothL1Loss(weights, beta=1.0)
    got = loss_fn(pred, target).item()

    elem = torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0, reduction="none")
    expected = (elem * weights.view(1, -1)).mean().item()

    if abs(got - expected) > 1.0e-7:
        raise AssertionError(f"WeightedSmoothL1Loss mismatch: got={got} expected={expected}")

    print("WEIGHTED_SMOOTHL1_LOSS_TEST_OK")


if __name__ == "__main__":
    main()
