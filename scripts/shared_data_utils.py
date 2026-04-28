#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path


def _dir_populated(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def ensure_shared_data(
    tar_path: str | Path,
    shared_dir: str | Path,
    *,
    wait_timeout_s: float = 1800.0,
    poll_interval_s: float = 1.0,
) -> str:
    tar_path = Path(tar_path)
    target = Path(shared_dir)

    if _dir_populated(target):
        return "exists"

    if not tar_path.is_file():
        raise FileNotFoundError(f"Tar archive not found: {tar_path}")

    if target.exists() and not target.is_dir():
        raise ValueError(f"Shared data target exists and is not a directory: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.extract.lock"
    start = time.time()
    lock_fd = None

    while True:
        if _dir_populated(target):
            return "exists"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if time.time() - start > wait_timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for shared-data lock {lock_path} while preparing {target}"
                )
            time.sleep(poll_interval_s)

    try:
        if _dir_populated(target):
            return "exists"

        with tempfile.TemporaryDirectory(prefix=f"{target.name}_extract_", dir=str(target.parent)) as tmp:
            tmp_path = Path(tmp)
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(tmp_path)

            top_level = list(tmp_path.iterdir())
            dirs = [p for p in top_level if p.is_dir()]

            if target.exists():
                shutil.rmtree(target)

            if len(dirs) == 1 and len(top_level) == 1:
                shutil.move(str(dirs[0]), str(target))
            else:
                target.mkdir(parents=True, exist_ok=True)
                for item in top_level:
                    shutil.move(str(item), str(target / item.name))

        return "prepared"
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-path", required=True)
    ap.add_argument("--shared-dir", required=True)
    ap.add_argument("--wait-timeout-s", type=float, default=1800.0)
    ap.add_argument("--poll-interval-s", type=float, default=1.0)
    args = ap.parse_args()

    status = ensure_shared_data(
        args.tar_path,
        args.shared_dir,
        wait_timeout_s=args.wait_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    print(
        json.dumps(
            {
                "status": status,
                "tar_path": str(args.tar_path),
                "shared_dir": str(args.shared_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
