#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "src"))

from data import _apply_scale_inverse, _apply_targets_transform, load_data, preprocess_features, preprocess_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--study-spec", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-prefix", default=None)
    ap.add_argument("--curr", default=None)
    return ap.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def crop_and_flatten(arr: np.ndarray, sub_length: int | None) -> np.ndarray:
    out = np.asarray(arr)
    if sub_length and sub_length < out.shape[-1]:
        out = out[..., :sub_length]
    return out.reshape(out.shape[0], -1)


def parse_study_spec(spec: str) -> tuple[str, str]:
    family, metric = spec.rsplit("_", 1)
    return family, metric


def build_estimator(family: str, trial: optuna.Trial):
    n_estimators = trial.suggest_int("n_estimators", 200, 1000, step=100)
    max_features = trial.suggest_categorical("max_features", [1.0, "sqrt", 0.5])
    max_depth = trial.suggest_categorical("max_depth", [None, 20, 30])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
            n_jobs=-1,
        )
    raise ValueError(f"unsupported family={family}")


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hh_tabular_optuna")

    params = load_params(args.params)
    if args.data_dir is not None:
        params["data"]["data_dir"] = args.data_dir
    if args.data_prefix is not None:
        params["data"]["data_prefix"] = args.data_prefix
    if args.curr is not None:
        params["data"]["curr"] = args.curr

    # Load the largest train budget once; trials choose a prefix length.
    params["data"]["Ntrain"] = 12288
    params["data"]["Nvalidate"] = 1024
    params["data"]["Ntest"] = 1024
    params["data"]["features_scale_cache_enabled"] = False
    params["data"]["split_array_cache_enabled"] = False
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False

    t0 = time.time()
    features, targets, _, _ = load_data(params, logger)
    preprocess_features(features, params, logger)
    targets_scale = preprocess_targets(targets, params, logger)

    sub_length = params["data"].get("features_sub_length")
    x_train_full = crop_and_flatten(features["train"], sub_length)
    x_validate = crop_and_flatten(features["validate"], sub_length)
    y_train_full = np.asarray(targets["train"])
    y_validate = np.asarray(targets["validate"])

    family, metric = parse_study_spec(args.study_spec)

    def objective(trial: optuna.Trial) -> float:
        n_train = trial.suggest_categorical("n_train", [4096, 8192, 12288])
        x_train = x_train_full[:n_train]
        y_train = y_train_full[:n_train]
        model = build_estimator(family, trial)
        model.fit(x_train, y_train)
        preds = model.predict(x_validate)

        preds_post = _apply_scale_inverse(np.array(preds, copy=True), targets_scale)
        validate_post = _apply_scale_inverse(np.array(y_validate, copy=True), targets_scale)
        transform_cfg = targets_scale.get("transform") if isinstance(targets_scale, dict) else None
        preds_post = _apply_targets_transform(preds_post, transform_cfg, inverse=True, array_name="targets")
        validate_post = _apply_targets_transform(validate_post, transform_cfg, inverse=True, array_name="targets")

        mse = float(mean_squared_error(validate_post, preds_post))
        mae = float(mean_absolute_error(validate_post, preds_post))
        r2 = float(r2_score(validate_post, preds_post))
        trial.set_user_attr("mse", mse)
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("r2", r2)

        if metric == "mse":
            return mse
        if metric == "mae":
            return mae
        if metric == "r2":
            return r2
        raise ValueError(f"unsupported metric={metric}")

    direction = "maximize" if metric == "r2" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    trials_rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
        }
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update({f"attr_{k}": v for k, v in trial.user_attrs.items()})
        trials_rows.append(row)

    runtime_sec = time.time() - t0
    summary = {
        "study_spec": args.study_spec,
        "direction": direction,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "runtime_sec": runtime_sec,
    }
    (save_dir / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (save_dir / "study_trials.json").write_text(json.dumps(trials_rows, indent=2) + "\n")
    (save_dir / "params_snapshot.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
