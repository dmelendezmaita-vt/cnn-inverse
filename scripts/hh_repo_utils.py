#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from shared_data_utils import ensure_shared_data


REPO_ROOT = Path(__file__).resolve().parents[1]


def is_tar_archive(path: Path) -> bool:
    text = str(path)
    return path.is_file() and text.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"))


def derive_data_prefix(path: Path, explicit_prefix: str | None = None) -> str:
    if explicit_prefix:
        return str(explicit_prefix).strip("/.")
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def resolve_hh_dataset_root(data_dir: str | Path, data_prefix: str | None = None) -> Path:
    path = Path(str(data_dir)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if not is_tar_archive(path):
        return path

    prefix = derive_data_prefix(path, explicit_prefix=data_prefix)
    prepared_root = REPO_ROOT / ".prepared_data"
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared_dir = prepared_root / prefix
    ensure_shared_data(path, prepared_dir)
    return prepared_dir


def resolve_hh_feature_root(data_dir: str | Path, data_prefix: str | None = None) -> Path:
    dataset_root = resolve_hh_dataset_root(data_dir, data_prefix=data_prefix)
    return dataset_root / "y"
