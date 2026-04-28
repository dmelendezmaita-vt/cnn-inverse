#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from features_scale_cache_utils import (
    summarize_feature_scale_cache_results,
    warm_feature_scale_cache,
)
from test_features_scale_cache_20260417 import make_fixture, make_params


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_features_scale_cache_utils_") as td:
        root = Path(td)
        data_dir = root / "fixture_root"
        make_fixture(data_dir)
        params = make_params(data_dir)

        params_file = root / "params.yaml"
        params_file.write_text(yaml.safe_dump(params, sort_keys=False))

        first = warm_feature_scale_cache(params_file)
        assert first["status"] == "warmed"
        assert first["cache_path"] is not None
        assert Path(first["cache_path"]).exists()

        second = warm_feature_scale_cache(params_file)
        assert second["status"] == "exists"
        assert second["cache_path"] == first["cache_path"]

        third = warm_feature_scale_cache(params_file, force=True)
        assert third["status"] == "warmed"
        assert third["cache_path"] == first["cache_path"]

        summary = summarize_feature_scale_cache_results([first, second, third])
        assert summary["status_counts"]["warmed"] == 2
        assert summary["status_counts"]["exists"] == 1
        assert summary["n_unique_cache_paths"] == 1

    print("FEATURES_SCALE_CACHE_UTILS_TEST_OK")


if __name__ == "__main__":
    main()
