#!/usr/bin/env python3
from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pytorch.data import load_data
from test_data_tar_singlefile_split_20260416 import make_fixture, make_params


def make_logger(name: str):
    stream = io.StringIO()
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream, handler


def run_once(data_dir: Path, *, cache_enabled: bool):
    params = make_params(data_dir)
    params["data"]["split_array_cache_enabled"] = cache_enabled
    logger, stream, handler = make_logger(f"test_split_array_cache.{cache_enabled}")
    try:
        features, targets, features_noise, targets_noise = load_data(params, logger)
    finally:
        logger.removeHandler(handler)
    return (features, targets, features_noise, targets_noise), stream.getvalue()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_split_array_cache_") as td:
        data_dir = Path(td) / "fixture_root"
        make_fixture(data_dir)

        (features1, targets1, _fn1, _tn1), logs1 = run_once(data_dir, cache_enabled=True)
        cache_root = data_dir / ".split_array_cache"
        files = sorted(cache_root.rglob("*.npy"))
        assert len(files) == 6
        assert "split array cache write" in logs1

        (features2, targets2, _fn2, _tn2), logs2 = run_once(data_dir, cache_enabled=True)
        assert "split array cache hit" in logs2

        np.testing.assert_allclose(features1["train"], features2["train"], rtol=0, atol=0)
        np.testing.assert_allclose(features1["validate"], features2["validate"], rtol=0, atol=0)
        np.testing.assert_allclose(features1["test"], features2["test"], rtol=0, atol=0)
        np.testing.assert_allclose(targets1["train"], targets2["train"], rtol=0, atol=0)
        np.testing.assert_allclose(targets1["validate"], targets2["validate"], rtol=0, atol=0)
        np.testing.assert_allclose(targets1["test"], targets2["test"], rtol=0, atol=0)

    print("SPLIT_ARRAY_CACHE_TEST_OK")


if __name__ == "__main__":
    main()
