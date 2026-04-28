#!/usr/bin/env python3
from __future__ import annotations

import io
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from shared_data_utils import ensure_shared_data

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pytorch.data import (  # noqa: E402
    _resolve_features_scale_cache_path,
    load_data,
    preprocess_features,
)


def _load_params(params_file: str | Path, repo_root: Path | None = None) -> dict:
    repo_root = repo_root or REPO
    path = Path(params_file)
    if not path.is_absolute():
        path = repo_root / path
    return yaml.safe_load(path.read_text())


def _make_logger(name: str) -> tuple[logging.Logger, io.StringIO, logging.StreamHandler]:
    stream = io.StringIO()
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream, handler


def warm_feature_scale_cache(
    params_file: str | Path,
    *,
    repo_root: Path | None = None,
    data_dir_override: str | Path | None = None,
    shared_data_dir: str | Path | None = None,
    tar_path: str | Path | None = None,
    force: bool = False,
) -> Dict[str, object]:
    repo_root = repo_root or REPO
    params = _load_params(params_file, repo_root=repo_root)
    data_cfg = params.setdefault("data", {})

    if shared_data_dir:
        shared_dir_path = Path(shared_data_dir)
        if tar_path and (not shared_dir_path.is_dir() or not any(shared_dir_path.iterdir())):
            ensure_shared_data(str(tar_path), str(shared_dir_path))
        if shared_dir_path.is_dir():
            data_cfg["data_dir"] = str(shared_dir_path)
    elif data_dir_override is not None:
        data_cfg["data_dir"] = str(data_dir_override)

    if not bool(data_cfg.get("features_scale_cache_enabled", False)):
        return {
            "status": "disabled",
            "params_file": str(params_file),
            "data_dir": str(data_cfg.get("data_dir")),
            "cache_path": None,
        }

    cache_path = _resolve_features_scale_cache_path(data_cfg, array_name="features")
    if cache_path is None:
        return {
            "status": "unsupported",
            "params_file": str(params_file),
            "data_dir": str(data_cfg.get("data_dir")),
            "cache_path": None,
        }

    if cache_path.exists() and force:
        cache_path.unlink()

    if cache_path.exists() and not force:
        return {
            "status": "exists",
            "params_file": str(params_file),
            "data_dir": str(data_cfg.get("data_dir")),
            "cache_path": str(cache_path),
        }

    params.setdefault("runconfig", {})
    params["runconfig"].setdefault("save_dir", "runs/tmp_scale_cache_warm")

    logger, stream, handler = _make_logger(f"warm_feature_scale_cache.{cache_path.name}")
    try:
        features, targets, features_noise, targets_noise = load_data(params, logger)
        preprocess_features(features, params, logger)
    finally:
        logger.removeHandler(handler)

    return {
        "status": "warmed",
        "params_file": str(params_file),
        "data_dir": str(data_cfg.get("data_dir")),
        "cache_path": str(cache_path),
        "log": stream.getvalue(),
    }


def warm_feature_scale_cache_for_rows(
    rows: Iterable[Dict[str, str]],
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> List[Dict[str, object]]:
    repo_root = repo_root or REPO
    results: List[Dict[str, object]] = []
    seen_cache_paths: set[str] = set()

    for row in rows:
        params = _load_params(row["params_file"], repo_root=repo_root)
        data_cfg = params.setdefault("data", {})
        shared_data_dir = row.get("shared_data_dir") or None
        tar_path = row.get("tar_path") or None
        if shared_data_dir:
            shared_dir_path = Path(shared_data_dir)
            if tar_path and (not shared_dir_path.is_dir() or not any(shared_dir_path.iterdir())):
                ensure_shared_data(str(tar_path), str(shared_dir_path))
            if shared_dir_path.is_dir():
                data_cfg["data_dir"] = str(shared_dir_path)

        cache_path = _resolve_features_scale_cache_path(data_cfg, array_name="features")
        cache_path_str = str(cache_path) if cache_path is not None else None
        if cache_path_str and cache_path_str in seen_cache_paths:
            result = {
                "status": "duplicate",
                "params_file": str(row["params_file"]),
                "data_dir": str(data_cfg.get("data_dir")),
                "cache_path": cache_path_str,
            }
            results.append(result)
            continue

        result = warm_feature_scale_cache(
            row["params_file"],
            repo_root=repo_root,
            shared_data_dir=shared_data_dir,
            tar_path=tar_path,
            force=force,
        )
        cache_path = result.get("cache_path")
        if cache_path:
            seen_cache_paths.add(str(cache_path))
        results.append(result)

    return results


def summarize_feature_scale_cache_results(results: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(results)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    cache_paths = sorted(
        {
            str(row["cache_path"])
            for row in rows
            if row.get("cache_path")
        }
    )
    return {
        "status_counts": dict(status_counts),
        "unique_cache_paths": cache_paths,
        "n_unique_cache_paths": len(cache_paths),
    }
