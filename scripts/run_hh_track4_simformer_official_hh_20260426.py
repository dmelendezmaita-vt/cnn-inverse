#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
import types

import numpy as np


SIMFORMER_ROOT = Path("/home/dmm96/.cache/simformer")
PYTHONPATHS = [SIMFORMER_ROOT / "src" / "probjax", SIMFORMER_ROOT / "src" / "scoresbibm"]
for path in PYTHONPATHS:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import jax.random as jrandom

if not hasattr(jrandom, "PRNGKeyArray"):
    jrandom.PRNGKeyArray = type(jrandom.PRNGKey(0))  # type: ignore[attr-defined]
if not hasattr(jax, "linear_util"):
    from jax._src import linear_util as _linear_util  # type: ignore

    jax.linear_util = _linear_util  # type: ignore[attr-defined]
from jax import core as _jax_core
from jax._src import core as _jax_core_internal  # type: ignore

_jax_core_public = vars(_jax_core)
_jax_core_private = vars(_jax_core_internal)

for _name in (
    "AbstractValue",
    "Atom",
    "CallPrimitive",
    "ClosedJaxpr",
    "DropVar",
    "Jaxpr",
    "JaxprEqn",
    "Literal",
    "Primitive",
    "ShapedArray",
    "Var",
    "eval_jaxpr",
):
    if _name not in _jax_core_public and _name in _jax_core_private:
        _jax_core_public[_name] = _jax_core_private[_name]
if "new_sublevel" not in _jax_core_public:
    _jax_core_public["new_sublevel"] = contextlib.nullcontext
_pjit_mod = importlib.import_module("jax.experimental.pjit")
if not hasattr(_pjit_mod, "pjit_p"):
    _pjit_mod.pjit_p = object()


def ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    class _Tensor:
        pass
    torch_stub.cat = lambda xs, dim=0: np.concatenate([np.asarray(x) for x in xs], axis=dim)
    torch_stub.Tensor = _Tensor
    sys.modules["torch"] = torch_stub

if "IPython" not in sys.modules:
    ipy_mod = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    def _display(*args, **kwargs):
        return None

    class _SVG:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    display_mod.display = _display
    display_mod.SVG = _SVG
    ipy_mod.display = display_mod
    sys.modules["IPython"] = ipy_mod
    sys.modules["IPython.display"] = display_mod

if "flax" not in sys.modules:
    flax_mod = types.ModuleType("flax")
    flax_core_mod = types.ModuleType("flax.core")
    flax_errors_mod = types.ModuleType("flax.errors")
    flax_linen_mod = types.ModuleType("flax.linen")

    class _FlaxScope:
        def push(self, *args, **kwargs):
            return self

        def put_variable(self, *args, **kwargs):
            return None

    class _FlaxModule:
        scope = _FlaxScope()
        variables = {}

        def __post_init__(self):
            return None

        def is_initializing(self):
            return False

        def make_rng(self, *_args, **_kwargs):
            return None

        def has_rng(self, *_args, **_kwargs):
            return False

    def _compact(fn):
        return fn

    class _InvalidRngError(Exception):
        pass

    flax_mod.__path__ = []
    flax_core_mod.Scope = _FlaxScope
    flax_errors_mod.InvalidRngError = _InvalidRngError
    flax_linen_mod.Module = _FlaxModule
    flax_linen_mod.compact = _compact
    flax_mod.core = flax_core_mod
    flax_mod.errors = flax_errors_mod
    flax_mod.linen = flax_linen_mod
    sys.modules["flax"] = flax_mod
    sys.modules["flax.core"] = flax_core_mod
    sys.modules["flax.errors"] = flax_errors_mod
    sys.modules["flax.linen"] = flax_linen_mod

SCORESBIBM_ROOT = SIMFORMER_ROOT / "src" / "scoresbibm" / "scoresbibm"
ensure_package("scoresbibm", SCORESBIBM_ROOT)
ensure_package("scoresbibm.tasks", SCORESBIBM_ROOT / "tasks")
ensure_package("scoresbibm.methods", SCORESBIBM_ROOT / "methods")
ensure_package("scoresbibm.utils", SCORESBIBM_ROOT / "utils")

load_module("scoresbibm.tasks.base_task", SCORESBIBM_ROOT / "tasks" / "base_task.py")
load_module("scoresbibm.methods.guidance", SCORESBIBM_ROOT / "methods" / "guidance.py")
load_module("scoresbibm.methods.models", SCORESBIBM_ROOT / "methods" / "models.py")
load_module("scoresbibm.utils.condition_masks", SCORESBIBM_ROOT / "utils" / "condition_masks.py")
load_module("scoresbibm.utils.edge_masks", SCORESBIBM_ROOT / "utils" / "edge_masks.py")
load_module("scoresbibm.methods.sde", SCORESBIBM_ROOT / "methods" / "sde.py")
load_module("scoresbibm.methods.neural_nets", SCORESBIBM_ROOT / "methods" / "neural_nets.py")
load_module("scoresbibm.tasks.all_conditional_tasks", SCORESBIBM_ROOT / "tasks" / "all_conditional_tasks.py")
hhtask_mod = load_module("scoresbibm.tasks.hhtask", SCORESBIBM_ROOT / "tasks" / "hhtask.py")
score_transformer_mod = load_module("scoresbibm.methods.score_transformer", SCORESBIBM_ROOT / "methods" / "score_transformer.py")

HHTask = hhtask_mod.HHTask
train_transformer_model = score_transformer_mod.train_transformer_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reduced official Simformer score-transformer run on the upstream HH task.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--max-number-steps", type=int, default=500)
    ap.add_argument("--min-number-steps", type=int, default=200)
    ap.add_argument("--training-batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1.0e-3)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--num-eval-observations", type=int, default=5)
    ap.add_argument("--posterior-steps", type=int, default=100)
    return ap.parse_args()


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def make_cfg(args: argparse.Namespace) -> DotDict:
    train = DotDict(
        max_number_steps=args.max_number_steps,
        min_number_steps=args.min_number_steps,
        total_number_steps_scaling=1.0,
        training_batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=1.0e-5,
        clip_max_norm=10.0,
        validation_fraction=0.05,
        val_repeat=2,
        val_every=1,
        stop_early_count=3,
        rebalance_loss=False,
        z_score_data=False,
        condition_mask_fn=DotDict(name="posterior"),
        edge_mask_fn=DotDict(name="none"),
    )
    model = DotDict(
        token_dim=40,
        condition_token_dim=10,
        condition_token_init_scale=0.1,
        condition_token_init_mean=0.0,
        condition_mode="concat",
        time_embedding_dim=128,
        num_heads=4,
        num_layers=4,
        attn_size=10,
        widening_factor=3,
        num_hidden_layers=1,
        skip_connection_attn=True,
        skip_connection_mlp=True,
        layer_norm=True,
        use_output_scale_fn=True,
    )
    posterior = DotDict(sampling_method="sde", num_steps=args.posterior_steps, method="euler_maruyama")
    sde = DotDict(name="vesde", sigma_max=15.0, sigma_min=1.0e-4, T_max=1.0, T_min=1.0e-5, scale_min=1.0e-3)
    return DotDict(name="score_transformer", backend="jax", device="cpu", train=train, model=model, posterior=posterior, sde=sde)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    rng = set_seed(args.seed)
    task = HHTask(backend="jax")
    rng, rng_data, rng_train, rng_eval = jax.random.split(rng, 4)

    data = task.get_data(args.num_simulations, rng=rng_data)
    cfg = make_cfg(args)

    t0 = time.time()
    model = train_transformer_model(task, data, cfg, rng_train)
    model.set_default_sampling_kwargs(**cfg.posterior)
    train_runtime = time.time() - t0

    eval_data = task.get_data(args.num_eval_observations, rng=rng_eval)
    theta_true = np.asarray(eval_data["theta"])
    x_obs = np.asarray(eval_data["x"])
    node_id = np.asarray(task.get_node_id())
    theta_dim = task.get_theta_dim()
    x_dim = task.get_x_dim()
    condition_mask = np.array([False] * theta_dim + [True] * x_dim, dtype=bool)

    t1 = time.time()
    samples = model.sample_batched(
        num_samples=args.num_samples,
        x_o=jnp.asarray(x_obs),
        node_id=jnp.asarray(node_id),
        condition_mask=jnp.asarray(condition_mask),
        rng=jax.random.PRNGKey(args.seed + 99),
        num_steps=args.posterior_steps,
        sampling_method="sde",
    )
    sample_runtime = time.time() - t1
    samples = np.asarray(samples)
    posterior_mean = np.mean(samples, axis=1)
    posterior_std = np.std(samples, axis=1)
    diff = posterior_mean - theta_true
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(np.square(theta_true - np.mean(theta_true))))
    r2 = 1.0 - float(np.sum(np.square(diff))) / denom if denom > 0.0 else 0.0

    row_per_obs = []
    for i in range(theta_true.shape[0]):
        row = {
            "observation_index": i,
            "mae_theta": float(np.mean(np.abs(theta_true[i] - posterior_mean[i]))),
            "std_theta_mean": float(np.mean(posterior_std[i])),
        }
        row_per_obs.append(row)
    write_csv(save_dir / "simformer_hh_eval_per_observation.csv", row_per_obs)

    np.savez_compressed(
        save_dir / "simformer_hh_predictions.npz",
        theta_true=theta_true.astype(np.float32),
        posterior_mean=posterior_mean.astype(np.float32),
        posterior_std=posterior_std.astype(np.float32),
        posterior_samples=samples.astype(np.float32),
        x_obs=x_obs.astype(np.float32),
    )

    summary = {
        "method_family": "simformer_official_score_transformer_hh_task",
        "source_repo": "https://github.com/mackelab/simformer",
        "source_paper": "https://arxiv.org/abs/2404.09636",
        "task": "official_upstream_hh",
        "assumption_conditioned": False,
        "direct_track4_comparable": False,
        "limitation_note": "This is the official Simformer HH benchmark task from the upstream repository, run under a reduced training budget, not a direct port to the local Track4 assumption-conditioned workflow.",
        "num_simulations": args.num_simulations,
        "max_number_steps": args.max_number_steps,
        "min_number_steps": args.min_number_steps,
        "training_batch_size": args.training_batch_size,
        "posterior_steps": args.posterior_steps,
        "num_eval_observations": args.num_eval_observations,
        "num_samples": args.num_samples,
        "train_runtime_sec": train_runtime,
        "sample_runtime_sec": sample_runtime,
        "test_metrics": {"mse": mse, "mae": mae, "r2": r2},
    }
    (save_dir / "simformer_hh_metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Simformer Official HH Task",
        "",
        "- official upstream codepath, reduced local budget",
        "- not directly comparable to the local Track4 fixed-array workflow",
        "",
        f"- source repo: `{summary['source_repo']}`",
        f"- source paper: `{summary['source_paper']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| `mae` | {mae:.6f} |",
        f"| `mse` | {mse:.6f} |",
        f"| `r2` | {r2:.6f} |",
        f"| `train_runtime_sec` | {train_runtime:.6f} |",
        f"| `sample_runtime_sec` | {sample_runtime:.6f} |",
    ]
    (save_dir / "simformer_hh_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
