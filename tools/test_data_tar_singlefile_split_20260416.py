#!/usr/bin/env python3
from __future__ import annotations

import io
import logging
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pytorch.data import load_data
from pytorch.data import (
    IndexedArray,
    create_dataloader,
    dictarray_uses_indexed_arrays,
    make_features_preprocess_transform,
    make_targets_preprocess_transform,
    preprocess_features,
    preprocess_targets,
)
from utils import Mode


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


def run_case(data_dir: Path, *, lazy_split_views: bool = False) -> tuple[dict, str]:
    params = make_params(data_dir)
    if lazy_split_views:
        params["data"]["lazy_split_views"] = True
    logger, stream, handler = make_logger(f"test_data_tar_singlefile_split.{data_dir.name}.lazy={lazy_split_views}")
    try:
        features, targets, features_noise, targets_noise = load_data(params, logger)
    finally:
        logger.removeHandler(handler)

    assert features_noise["train"] is None
    assert targets_noise["train"] is None
    assert features["train"].shape == (2, 1, 8)
    assert features["validate"].shape == (1, 1, 8)
    assert features["test"].shape == (1, 1, 8)
    assert targets["train"].shape == (2, 6)
    assert targets["validate"].shape == (1, 6)
    assert targets["test"].shape == (1, 6)
    if lazy_split_views:
        assert isinstance(features["train"], IndexedArray)
        assert isinstance(targets["train"], IndexedArray)
        assert dictarray_uses_indexed_arrays(features)
        assert dictarray_uses_indexed_arrays(targets)
    return params, stream.getvalue()


def run_lazy_vs_eager_equivalence(data_dir: Path) -> None:
    eager_params = make_params(data_dir)
    lazy_params = make_params(data_dir)
    lazy_params["data"]["lazy_split_views"] = True

    load_logger_eager, _, load_handler_eager = make_logger("test_data_tar_singlefile_split.eager.load")
    load_logger_lazy, lazy_stream, load_handler_lazy = make_logger("test_data_tar_singlefile_split.lazy.load")
    try:
        eager_features, eager_targets, eager_features_noise, eager_targets_noise = load_data(eager_params, load_logger_eager)
        lazy_features, lazy_targets, lazy_features_noise, lazy_targets_noise = load_data(lazy_params, load_logger_lazy)
    finally:
        load_logger_eager.removeHandler(load_handler_eager)
        load_logger_lazy.removeHandler(load_handler_lazy)

    pre_logger_eager, _, pre_handler_eager = make_logger("test_data_tar_singlefile_split.eager.pre")
    pre_logger_lazy, lazy_pre_stream, pre_handler_lazy = make_logger("test_data_tar_singlefile_split.lazy.pre")
    try:
        eager_feature_scale = preprocess_features(eager_features, eager_params, pre_logger_eager)
        eager_target_scale = preprocess_targets(eager_targets, eager_params, pre_logger_eager)
        lazy_feature_scale = preprocess_features(lazy_features, lazy_params, pre_logger_lazy)
        lazy_target_scale = preprocess_targets(lazy_targets, lazy_params, pre_logger_lazy)
    finally:
        pre_logger_eager.removeHandler(pre_handler_eager)
        pre_logger_lazy.removeHandler(pre_handler_lazy)

    assert "deferring feature scaling/materialization to the dataloader" in lazy_pre_stream.getvalue()
    assert "deferring target scaling/materialization to the dataloader" in lazy_pre_stream.getvalue()
    assert isinstance(lazy_features["train"], IndexedArray)
    assert isinstance(lazy_targets["train"], IndexedArray)

    eager_dl_logger, _, eager_dl_handler = make_logger("test_data_tar_singlefile_split.eager.dl")
    lazy_dl_logger, _, lazy_dl_handler = make_logger("test_data_tar_singlefile_split.lazy.dl")
    try:
        eager_dl = create_dataloader(
            eager_params,
            eager_dl_logger,
            Mode.EVAL,
            features=eager_features["train"],
            targets=eager_targets["train"],
            features_noise=eager_features_noise["train"],
            targets_noise=eager_targets_noise["train"],
            features_transform_fn=None,
            targets_transform_fn=None,
            item_return_order="yx",
            distributed=False,
            rank=0,
            world_size=1,
        )
        lazy_dl = create_dataloader(
            lazy_params,
            lazy_dl_logger,
            Mode.EVAL,
            features=lazy_features["train"],
            targets=lazy_targets["train"],
            features_noise=lazy_features_noise["train"],
            targets_noise=lazy_targets_noise["train"],
            features_transform_fn=make_features_preprocess_transform(lazy_feature_scale, lazy_params),
            targets_transform_fn=make_targets_preprocess_transform(lazy_target_scale),
            item_return_order="yx",
            distributed=False,
            rank=0,
            world_size=1,
        )
    finally:
        eager_dl_logger.removeHandler(eager_dl_handler)
        lazy_dl_logger.removeHandler(lazy_dl_handler)

    eager_x, eager_y = next(iter(eager_dl))
    lazy_x, lazy_y = next(iter(lazy_dl))
    np.testing.assert_allclose(lazy_x.numpy(), eager_x.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(lazy_y.numpy(), eager_y.numpy(), rtol=1e-6, atol=1e-6)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_data_tar_singlefile_split_") as td:
        tmp = Path(td)
        extracted = tmp / "fixture_root"
        make_fixture(extracted)

        tar_path = tmp / "fixture.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(extracted, arcname=extracted.name)

        _params_dir, logs_dir = run_case(extracted)
        assert "data_source=filesystem_dir" in logs_dir
        assert "use_mmap_requested=True" in logs_dir
        assert "use_mmap_effective=True" in logs_dir
        assert "data load timings (sec): total=" in logs_dir

        _params_tar, logs_tar = run_case(tar_path)
        assert "data_source=tar_archive" in logs_tar
        assert "use_mmap_requested=True" in logs_tar
        assert "use_mmap_effective=False" in logs_tar
        assert "data split summary:" in logs_tar

        _params_lazy, logs_lazy = run_case(extracted, lazy_split_views=True)
        assert "data_source=filesystem_dir" in logs_lazy
        assert "data split summary:" in logs_lazy
        run_lazy_vs_eager_equivalence(extracted)

    print("DATA_TAR_SINGLEFILE_SPLIT_TEST_OK")


if __name__ == "__main__":
    main()
