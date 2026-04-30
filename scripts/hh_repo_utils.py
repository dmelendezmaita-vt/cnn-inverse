#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
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


def _stage_mode(explicit_stage_mode: str | None = None) -> str:
    if explicit_stage_mode:
        return explicit_stage_mode
    return str(os.environ.get("NC_HH_STAGE_MODE", "none"))


def _tmpfs_stage_root() -> Path:
    explicit = os.environ.get("NC_HH_STAGE_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    user = os.environ.get("USER", "user")
    return Path("/dev/shm") / user / "nc_hh_stage"


def _nvme_stage_root() -> Path:
    explicit = os.environ.get("NC_HH_STAGE_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    alloc = os.environ.get("ALLOC_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
    user = os.environ.get("USER", "user")
    return Path(f"/localscratch/{alloc}/{user}/nc_hh_stage")


def _stage_root_for_mode(mode: str) -> Path:
    if mode.startswith("shm_"):
        return _tmpfs_stage_root()
    if mode.startswith("nvme_"):
        return _nvme_stage_root()
    raise ValueError(f"Unsupported stage-root mode: {mode!r}")


def _prepared_root() -> Path:
    explicit = os.environ.get("NC_HH_PREPARED_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    return REPO_ROOT / ".prepared_data"


def _curr_tag(curr: str) -> str:
    return curr.replace("-", "m").replace(".", "p")


def _copy_single_current(source_dir: Path, dest_dir: Path, curr: str, data_prefix: str) -> None:
    (dest_dir / "y").mkdir(parents=True, exist_ok=True)
    (dest_dir / "support_stats").mkdir(parents=True, exist_ok=True)
    (dest_dir / "generated_params").mkdir(parents=True, exist_ok=True)
    file_specs = [
        source_dir / "y" / f"{data_prefix}_{curr}_curr.npy",
        source_dir / "support_stats" / f"{data_prefix}_{curr}_curr.npy",
        source_dir / "support_stats" / f"{data_prefix}_{curr}_curr.npy.lz4",
        source_dir / "generated_params" / f"{data_prefix}_{curr}_curr.npy",
    ]
    for src in file_specs:
        if src.exists():
            rel_parent = src.parent.relative_to(source_dir)
            target_parent = dest_dir / rel_parent
            target_parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", str(src), str(target_parent / src.name)], check=True)


def _copy_full_raw_tree(source_dir: Path, dest_dir: Path) -> None:
    for name in ("y", "support_stats", "generated_params"):
        src = source_dir / name
        if src.exists():
            subprocess.run(["cp", "-a", str(src), str(dest_dir / name)], check=True)


def _extract_tar_to_stage(tar_path: Path, dest_dir: Path) -> None:
    tar_copy = dest_dir.parent / f"{dest_dir.name}.tar.gz"
    subprocess.run(["cp", "-f", str(tar_path), str(tar_copy)], check=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(tar_copy), "-C", str(dest_dir)], check=True)


def _measure_tree_bytes(root: Path) -> dict[str, int]:
    logical = 0
    count = 0
    page_size = os.sysconf("SC_PAGESIZE")
    page_rounded = 0
    for path in root.rglob("*"):
        if path.is_file():
            size = path.stat().st_size
            logical += size
            page_rounded += ((size + page_size - 1) // page_size) * page_size
            count += 1
    return {
        "file_count": count,
        "logical_bytes": logical,
        "page_rounded_bytes": page_rounded,
    }


def stage_hh_dataset_root(
    source_dir: Path,
    data_prefix: str,
    *,
    curr: str | None = None,
    stage_mode: str | None = None,
) -> Path:
    mode = _stage_mode(stage_mode)
    if mode in {"", "none", "prepared_shared"}:
        return source_dir
    if mode not in {"shm_full_copy", "shm_curr_copy", "nvme_full_extract"}:
        raise ValueError(f"Unsupported HH stage mode: {mode!r}")
    if mode == "shm_curr_copy" and not curr:
        mode = "shm_full_copy"

    stage_root = _stage_root_for_mode(mode)
    stage_root.mkdir(parents=True, exist_ok=True)
    if mode == "shm_full_copy":
        tag = f"{data_prefix}__full_v1"
    elif mode == "shm_curr_copy":
        tag = f"{data_prefix}__curr_{_curr_tag(str(curr))}__v1"
    else:
        tag = f"{data_prefix}__nvme_full_v1"
    dest_dir = stage_root / tag
    manifest_path = dest_dir / ".stage_manifest.json"
    lock_path = stage_root / f".{tag}.lock"

    with lock_path.open("w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            return dest_dir
        tmp_dir = stage_root / f".{tag}.tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if mode == "shm_full_copy":
            _copy_full_raw_tree(source_dir, tmp_dir)
        elif mode == "shm_curr_copy":
            _copy_single_current(source_dir, tmp_dir, str(curr), data_prefix)
        else:
            _extract_tar_to_stage(source_dir, tmp_dir)
        payload = {
            "mode": mode,
            "source_dir": str(source_dir),
            "dest_dir": str(dest_dir),
            "data_prefix": data_prefix,
            "curr": curr,
            "created_at_epoch_sec": time.time(),
            "copy_elapsed_sec": time.time() - started,
            **_measure_tree_bytes(tmp_dir),
        }
        (tmp_dir / ".stage_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        os.replace(tmp_dir, dest_dir)
        return dest_dir


def resolve_hh_dataset_root(
    data_dir: str | Path,
    data_prefix: str | None = None,
    *,
    curr: str | None = None,
    stage_mode: str | None = None,
) -> Path:
    path = Path(str(data_dir)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    prefix = derive_data_prefix(path, explicit_prefix=data_prefix)
    if not is_tar_archive(path):
        return stage_hh_dataset_root(path, prefix, curr=curr, stage_mode=stage_mode)
    if _stage_mode(stage_mode) == "nvme_full_extract":
        return stage_hh_dataset_root(path, prefix, curr=curr, stage_mode="nvme_full_extract")

    prepared_root = _prepared_root()
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared_dir = prepared_root / prefix
    ensure_shared_data(path, prepared_dir)
    return stage_hh_dataset_root(prepared_dir, prefix, curr=curr, stage_mode=stage_mode)


def resolve_hh_feature_root(
    data_dir: str | Path,
    data_prefix: str | None = None,
    *,
    curr: str | None = None,
    stage_mode: str | None = None,
) -> Path:
    dataset_root = resolve_hh_dataset_root(data_dir, data_prefix=data_prefix, curr=curr, stage_mode=stage_mode)
    return dataset_root / "y"
