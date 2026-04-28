#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List
import io
import logging
import shutil
import sys

import yaml

from shared_data_utils import ensure_shared_data

REPO = Path(__file__).resolve().parents[1]

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pytorch.data import _resolve_split_array_cache_dir, load_data  # noqa: E402


def _load_params(params_file: str | Path, repo_root: Path | None = None) -> dict:
    repo_root = repo_root or REPO
    path = Path(params_file)
    if not path.is_absolute():
        path = repo_root / path
    return yaml.safe_load(path.read_text())


def _resolve_params_path(params_file: str | Path, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or REPO
    path = Path(params_file)
    if not path.is_absolute():
        path = repo_root / path
    return path


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


def warm_split_array_cache(
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

    if not bool(data_cfg.get("split_array_cache_enabled", False)):
        return {
            "status": "disabled",
            "params_file": str(params_file),
            "data_dir": str(data_cfg.get("data_dir")),
            "cache_dir": None,
        }

    cache_dir = _resolve_split_array_cache_dir(data_cfg)
    if cache_dir is None:
        return {
            "status": "unsupported",
            "params_file": str(params_file),
            "data_dir": str(data_cfg.get("data_dir")),
            "cache_dir": None,
        }

    if cache_dir.exists() and force:
        shutil.rmtree(cache_dir)

    cached_files = sorted(str(p) for p in cache_dir.rglob("*.npy")) if cache_dir.exists() else []
    if cached_files and not force:
        return {
            "status": "exists",
            "params_file": str(params_file),
            "data_dir": str(data_cfg.get("data_dir")),
            "cache_dir": str(cache_dir),
            "cache_file_count": len(cached_files),
        }

    params.setdefault("runconfig", {})
    params["runconfig"].setdefault("save_dir", "runs/tmp_split_array_cache_warm")

    logger, stream, handler = _make_logger(f"warm_split_array_cache.{cache_dir.name}")
    try:
        load_data(params, logger)
    finally:
        logger.removeHandler(handler)

    cached_files = sorted(str(p) for p in cache_dir.rglob("*.npy")) if cache_dir.exists() else []
    return {
        "status": "warmed",
        "params_file": str(params_file),
        "data_dir": str(data_cfg.get("data_dir")),
        "cache_dir": str(cache_dir),
        "cache_file_count": len(cached_files),
        "log": stream.getvalue(),
    }


def warm_split_array_cache_for_rows(
    rows: Iterable[Dict[str, str]],
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> List[Dict[str, object]]:
    repo_root = repo_root or REPO
    results: List[Dict[str, object]] = []
    seen_cache_dirs: set[str] = set()

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

        cache_dir = _resolve_split_array_cache_dir(data_cfg)
        cache_dir_str = str(cache_dir) if cache_dir is not None else None
        if cache_dir_str and cache_dir_str in seen_cache_dirs:
            result = {
                "status": "duplicate",
                "params_file": str(row["params_file"]),
                "data_dir": str(data_cfg.get("data_dir")),
                "cache_dir": cache_dir_str,
                "cache_file_count": len(list(cache_dir.rglob("*.npy"))) if cache_dir.exists() else 0,
            }
            results.append(result)
            continue

        result = warm_split_array_cache(
            row["params_file"],
            repo_root=repo_root,
            shared_data_dir=shared_data_dir,
            tar_path=tar_path,
            force=force,
        )
        cache_dir = result.get("cache_dir")
        if cache_dir:
            seen_cache_dirs.add(str(cache_dir))
        results.append(result)

    return results


def summarize_split_array_cache_results(results: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(results)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    cache_dirs = sorted(
        {
            str(row["cache_dir"])
            for row in rows
            if row.get("cache_dir")
        }
    )
    return {
        "status_counts": dict(status_counts),
        "unique_cache_dirs": cache_dirs,
        "n_unique_cache_dirs": len(cache_dirs),
    }


def materialize_split_array_cache_params(
    params_file: str | Path,
    output_dir: str | Path,
    *,
    repo_root: Path | None = None,
) -> str:
    repo_root = repo_root or REPO
    source_path = _resolve_params_path(params_file, repo_root=repo_root)
    params = _load_params(source_path, repo_root=repo_root)
    params = deepcopy(params)
    params.setdefault("data", {})
    params["data"]["split_array_cache_enabled"] = True
    # Older HH configs can still carry the multi-worker loader settings that
    # stalled in the original R4 investigation. Keep split-cache opt-in aligned
    # with the known-safe single-process dataloader path.
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    if isinstance(params.get("description"), str) and "split-array-cache opt-in" not in params["description"]:
        params["description"] = f'{params["description"]} [split-array-cache opt-in]'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    output_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return str(output_path)


def prepare_rows_for_split_array_cache(
    rows: Iterable[Dict[str, str]],
    output_dir: str | Path,
    *,
    repo_root: Path | None = None,
    enable: bool = False,
) -> List[Dict[str, str]]:
    repo_root = repo_root or REPO
    prepared: List[Dict[str, str]] = []
    for row in rows:
        row_copy = dict(row)
        if enable:
            params = _load_params(row_copy["params_file"], repo_root=repo_root)
            if not bool(params.get("data", {}).get("split_array_cache_enabled", False)):
                row_copy["params_file"] = materialize_split_array_cache_params(
                    row_copy["params_file"],
                    output_dir,
                    repo_root=repo_root,
                )
        prepared.append(row_copy)
    return prepared
