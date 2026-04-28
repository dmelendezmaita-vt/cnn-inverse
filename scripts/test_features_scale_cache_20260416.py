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

from pytorch.data import (
    _resolve_features_scale_cache_path,
    load_data,
    preprocess_features,
)


def make_fixture(root: Path, prefix: str = "fixture", curr: str = "0.1") -> None:
    (root / "y").mkdir(parents=True)
    (root / "generated_params").mkdir(parents=True)
    (root / "support_stats").mkdir(parents=True)

    features = np.arange(32, dtype=np.float32).reshape(4, 8)
    targets = (np.arange(24, dtype=np.float32).reshape(4, 6) + 1.0)
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
            "features_additive_noise_std": 0,
            "targets_type": "ODE",
            "targets_normalize": True,
            "Ntrain": 2,
            "Nvalidate": 1,
            "Ntest": 1,
            "train_shuffle_buffer_size": 4,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "random_seed": 123,
        }
    }


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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_features_scale_cache_") as td:
        root = Path(td) / "fixture_root"
        make_fixture(root)
        params = make_params(root)

        cache_path = _resolve_features_scale_cache_path(params["data"], array_name="features")
        assert cache_path is not None
        assert not cache_path.exists()

        load_logger, _, load_handler = make_logger("test_features_scale_cache.load1")
        try:
            features, _, _, _ = load_data(params, load_logger)
        finally:
            load_logger.removeHandler(load_handler)

        pre_logger, pre_stream, pre_handler = make_logger("test_features_scale_cache.pre1")
        try:
            scale_1 = preprocess_features(features, params, pre_logger)
        finally:
            pre_logger.removeHandler(pre_handler)

        assert cache_path.exists()
        assert "scale cache write" in pre_stream.getvalue()

        load_logger, _, load_handler = make_logger("test_features_scale_cache.load2")
        try:
            features_2, _, _, _ = load_data(params, load_logger)
        finally:
            load_logger.removeHandler(load_handler)

        # Distort train features so a recompute would obviously differ.
        features_2["train"] = np.zeros_like(np.asarray(features_2["train"])) + 999.0

        pre_logger, pre_stream, pre_handler = make_logger("test_features_scale_cache.pre2")
        try:
            scale_2 = preprocess_features(features_2, params, pre_logger)
        finally:
            pre_logger.removeHandler(pre_handler)

        assert "scale cache hit" in pre_stream.getvalue()
        np.testing.assert_allclose(scale_1["shift"], scale_2["shift"])
        np.testing.assert_allclose(scale_1["mult"], scale_2["mult"])

    print("FEATURES_SCALE_CACHE_TEST_OK")


if __name__ == "__main__":
    main()
