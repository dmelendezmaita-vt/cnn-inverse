#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    builder_paths = [
        REPO / "tools/build_v100_interactive_followup_matrix_20260411.py",
        REPO / "tools/build_v100_hh_scaling_redesign_matrix_20260412.py",
        REPO / "tools/build_v100_hh_scaling_replication_r2_20260413.py",
        REPO / "tools/build_v100_hh_postlocalsgd_r3_20260413.py",
        REPO / "tools/build_v100_hh_postlocalsgd_r3_stable_20260413.py",
        REPO / "tools/build_v100_hh_datapath_redesign_r4_20260416.py",
        REPO / "tools/build_a100_hh_datapath_redesign_r4_20260416.py",
    ]

    old = os.environ.get("ENABLE_SPLIT_ARRAY_CACHE")
    try:
        for idx, path in enumerate(builder_paths):
            if "ENABLE_SPLIT_ARRAY_CACHE" in os.environ:
                os.environ.pop("ENABLE_SPLIT_ARRAY_CACHE")
            mod = load_module(f"builder_default_{idx}", path)
            updates = {}
            helper = getattr(mod, "maybe_apply_split_array_cache")
            helper(updates)
            assert updates.get("split_array_cache_enabled") == "True", path
            if "dataloader_num_workers" in updates:
                assert updates["dataloader_num_workers"] == "0", path
                assert updates["dataloader_persistent_workers"] == "False", path
                assert updates["dataloader_prefetch_factor"] == "null", path
                assert updates["dataloader_pin_memory"] == "False", path

            os.environ["ENABLE_SPLIT_ARRAY_CACHE"] = "0"
            mod_disabled = load_module(f"builder_disabled_{idx}", path)
            updates_disabled = {}
            helper_disabled = getattr(mod_disabled, "maybe_apply_split_array_cache")
            helper_disabled(updates_disabled)
            assert updates_disabled == {}, path
    finally:
        if old is None:
            os.environ.pop("ENABLE_SPLIT_ARRAY_CACHE", None)
        else:
            os.environ["ENABLE_SPLIT_ARRAY_CACHE"] = old

    print("SPLIT_ARRAY_CACHE_BUILDER_DEFAULT_TEST_OK")


if __name__ == "__main__":
    main()
