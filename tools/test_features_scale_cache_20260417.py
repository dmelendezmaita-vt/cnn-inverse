#!/usr/bin/env python3
from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pytorch.data import load_data, preprocess_features


def make_fixture(root: Path, prefix: str = "fixture", curr: str = "0.1") -> None:
    (root / "y").mkdir(parents=True)
    (root / "generated_params").mkdir(parents=True)
    (root / "support_stats").mkdir(parents=True)

    features = np.arange(32, dtype=np.float32).reshape(4, 8)
    targets = np.arange(24, dtype=np.float32).reshape(4, 6) + 1.0
    stats = np.arange(8, dtype=np.float32).reshape(4, 2)

    np.save(root / "y" / f"{prefix}_{curr}_curr.npy", features)
    np.save(root / "generated_params" / f"{prefix}_{curr}_curr.npy", targets)
    np.save(root / "support_stats" / f"{prefix}_{curr}_curr.npy", stats)


def make_params(data_dir: Path) -> dict:
    return {
        "data": {
            "data_dir": str(data_dir),
            "data_prefix": "fixture",
            "curr": "0.1",
            "dataset_layout": "tar_singlefile_split",
            "use_mmap": True,
            "features_cols_num": 8,
            "targets_cols_num": 6,
            "file_names": {
                "features": "y/{data_prefix}_{curr}_curr.npy",
                "features_stats": "support_stats/{data_prefix}_{curr}_curr.npy",
                "targets": "generated_params/{data_prefix}_{curr}_curr.npy",
            },
            "features_type": "TIME",
            "features_use_channels_range": [0, 1],
            "features_normalize": True,
            "features_scale_cache_enabled": True,
            "features_sub_length": 8,
            "features_sub_begin_random": False,
            "features_sub_begin_random_eval": False,
            "targets_type": "ODE",
            "targets_normalize": True,
            "Ntrain": 2,
            "Nvalidate": 1,
            "Ntest": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "random_seed": 123,
        }
    }


def make_logger(name: str) -> tuple[logging.Logger, io.StringIO, logging.StreamHandler]:
    stream = io.StringIO()
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream, handler


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_features_scale_cache_") as td:
        data_dir = Path(td) / "fixture_root"
        make_fixture(data_dir)
        params = make_params(data_dir)

        load_logger, _, load_handler = make_logger("test_features_scale_cache.load")
        try:
            features, _targets, _features_noise, _targets_noise = load_data(params, load_logger)
        finally:
            load_logger.removeHandler(load_handler)

        logger1, stream1, handler1 = make_logger("test_features_scale_cache.first")
        try:
            scale1 = preprocess_features(features, params, logger1)
        finally:
            logger1.removeHandler(handler1)

        cache_dir = data_dir / ".scale_cache"
        cache_files = sorted(cache_dir.glob("features_scale_*.npz"))
        assert len(cache_files) == 1
        assert "scale cache write" in stream1.getvalue()

        logger2, stream2, handler2 = make_logger("test_features_scale_cache.second")
        try:
            scale2 = preprocess_features(features, params, logger2)
        finally:
            logger2.removeHandler(handler2)

        assert "scale cache hit" in stream2.getvalue()
        np.testing.assert_allclose(scale1["shift"], scale2["shift"], rtol=0, atol=0)
        np.testing.assert_allclose(scale1["mult"], scale2["mult"], rtol=0, atol=0)

    print("FEATURES_SCALE_CACHE_TEST_OK")


if __name__ == "__main__":
    main()
