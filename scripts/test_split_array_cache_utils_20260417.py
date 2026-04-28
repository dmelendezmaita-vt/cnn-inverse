#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pytorch.data import _resolve_split_array_cache_dir
from split_array_cache_utils import (
    materialize_split_array_cache_params,
    prepare_rows_for_split_array_cache,
    summarize_split_array_cache_results,
    warm_split_array_cache,
)
from test_data_tar_singlefile_split_20260416 import make_fixture, make_params


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_split_array_cache_utils_") as td:
        root = Path(td)
        data_dir = root / "fixture_root"
        make_fixture(data_dir)
        params = make_params(data_dir)
        params["data"]["split_array_cache_enabled"] = True

        params_file = root / "params.yaml"
        params_file.write_text(yaml.safe_dump(params, sort_keys=False))

        first = warm_split_array_cache(params_file)
        assert first["status"] == "warmed"
        assert first["cache_dir"] is not None
        assert Path(first["cache_dir"]).exists()
        assert first["cache_file_count"] == 6

        second = warm_split_array_cache(params_file)
        assert second["status"] == "exists"
        assert second["cache_dir"] == first["cache_dir"]
        assert second["cache_file_count"] == 6

        third = warm_split_array_cache(params_file, force=True)
        assert third["status"] == "warmed"
        assert third["cache_dir"] == first["cache_dir"]
        assert third["cache_file_count"] == 6

        override_dir = root / "overrides"
        override_path = materialize_split_array_cache_params(params_file, override_dir)
        override_loaded = yaml.safe_load(Path(override_path).read_text())
        assert override_loaded["data"]["split_array_cache_enabled"] is True
        assert override_loaded["data"]["dataloader_num_workers"] == 0
        assert override_loaded["data"]["dataloader_persistent_workers"] is False
        assert override_loaded["data"]["dataloader_prefetch_factor"] is None
        assert override_loaded["data"]["dataloader_pin_memory"] is False

        params_no_split = make_params(data_dir)
        params_no_split["data"]["split_array_cache_enabled"] = False
        params_no_split_file = root / "params_no_split.yaml"
        params_no_split_file.write_text(yaml.safe_dump(params_no_split, sort_keys=False))

        prepared = prepare_rows_for_split_array_cache(
            [{"row_id": "row1", "params_file": str(params_no_split_file)}],
            override_dir / "rows",
            enable=True,
        )
        assert prepared[0]["params_file"] != str(params_no_split_file)
        prepared_loaded = yaml.safe_load(Path(prepared[0]["params_file"]).read_text())
        assert prepared_loaded["data"]["split_array_cache_enabled"] is True
        assert prepared_loaded["data"]["dataloader_num_workers"] == 0

        same_a = make_params(data_dir)
        same_b = make_params(data_dir)
        same_a["data"]["split_array_cache_enabled"] = True
        same_b["data"]["split_array_cache_enabled"] = True
        same_a["data"]["random_seed"] = 971
        same_b["data"]["random_seed"] = 973
        cache_a = _resolve_split_array_cache_dir(same_a["data"])
        cache_b = _resolve_split_array_cache_dir(same_b["data"])
        assert cache_a == cache_b

        seeded = make_params(data_dir)
        seeded["data"]["split_array_cache_enabled"] = True
        seeded["data"]["split_strategy"] = "random"
        seeded["data"]["random_seed"] = 971
        cache_seeded = _resolve_split_array_cache_dir(seeded["data"])
        assert cache_seeded != cache_a

        summary = summarize_split_array_cache_results([first, second, third])
        assert summary["status_counts"]["warmed"] == 2
        assert summary["status_counts"]["exists"] == 1
        assert summary["n_unique_cache_dirs"] == 1

    print("SPLIT_ARRAY_CACHE_UTILS_TEST_OK")


if __name__ == "__main__":
    main()
