#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from shared_data_utils import ensure_shared_data


SCRIPT = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/scripts/shared_data_utils.py")


def create_fixture(tmp: Path) -> tuple[Path, Path]:
    payload = tmp / "fixture_root"
    (payload / "y").mkdir(parents=True)
    (payload / "generated_params").mkdir(parents=True)
    (payload / "support_stats").mkdir(parents=True)

    np.save(payload / "y" / "fixture_0.1_curr.npy", np.arange(24, dtype=np.float32).reshape(3, 8))
    np.save(payload / "generated_params" / "fixture_0.1_curr.npy", np.arange(18, dtype=np.float32).reshape(3, 6))
    np.save(payload / "support_stats" / "fixture_0.1_curr.npy", np.arange(6, dtype=np.float32).reshape(3, 2))

    tar_path = tmp / "fixture.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(payload, arcname=payload.name)
    return tar_path, payload


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="shared_data_utils_test_") as td:
        tmp = Path(td)
        tar_path, _payload = create_fixture(tmp)

        shared_dir = tmp / "shared" / "fixture_root"
        status = ensure_shared_data(tar_path, shared_dir)
        assert status == "prepared", status
        assert (shared_dir / "y" / "fixture_0.1_curr.npy").is_file()
        assert (shared_dir / "generated_params" / "fixture_0.1_curr.npy").is_file()
        assert (shared_dir / "support_stats" / "fixture_0.1_curr.npy").is_file()

        status = ensure_shared_data(tar_path, shared_dir)
        assert status == "exists", status

        cli_shared_dir = tmp / "shared_cli" / "fixture_root"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--tar-path", str(tar_path), "--shared-dir", str(cli_shared_dir)],
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip())
        assert payload["status"] == "prepared", payload
        assert (cli_shared_dir / "y" / "fixture_0.1_curr.npy").is_file()

    print("SHARED_DATA_UTILS_TEST_OK")


if __name__ == "__main__":
    main()
