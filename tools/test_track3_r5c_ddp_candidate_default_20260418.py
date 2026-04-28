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
        REPO / "tools/build_v100_hh_scaling_redesign_matrix_20260412.py",
        REPO / "tools/build_v100_hh_scaling_replication_r2_20260413.py",
    ]

    old = os.environ.get("ENABLE_TRACK3_R5C_DDP_CANDIDATE")
    try:
        for idx, path in enumerate(builder_paths):
            os.environ.pop("ENABLE_TRACK3_R5C_DDP_CANDIDATE", None)
            mod = load_module(f"track3_r5c_default_{idx}", path)
            updates_default = {}
            mod.maybe_apply_track3_r5c_ddp_candidate(
                track_key="track3_hh_reduced",
                nodes=4,
                variant_name="R1_accum2_nosync",
                updates=updates_default,
            )
            assert updates_default == {}, path

            os.environ["ENABLE_TRACK3_R5C_DDP_CANDIDATE"] = "1"
            mod = load_module(f"track3_r5c_enabled_{idx}", path)
            updates = {}
            mod.maybe_apply_track3_r5c_ddp_candidate(
                track_key="track3_hh_reduced",
                nodes=4,
                variant_name="R1_accum2_nosync",
                updates=updates,
            )
            assert updates.get("ddp_gradient_as_bucket_view") == "True", path
            assert updates.get("ddp_bucket_cap_mb") == "100", path

            updates_other = {}
            mod.maybe_apply_track3_r5c_ddp_candidate(
                track_key="track4_hh_full",
                nodes=4,
                variant_name="R1_accum2_nosync",
                updates=updates_other,
            )
            assert updates_other == {}, path

            updates_nodes = {}
            mod.maybe_apply_track3_r5c_ddp_candidate(
                track_key="track3_hh_reduced",
                nodes=2,
                variant_name="R1_accum2_nosync",
                updates=updates_nodes,
            )
            assert updates_nodes == {}, path

            os.environ["ENABLE_TRACK3_R5C_DDP_CANDIDATE"] = "0"
            mod_disabled = load_module(f"track3_r5c_disabled_{idx}", path)
            updates_disabled = {}
            mod_disabled.maybe_apply_track3_r5c_ddp_candidate(
                track_key="track3_hh_reduced",
                nodes=4,
                variant_name="R1_accum2_nosync",
                updates=updates_disabled,
            )
            assert updates_disabled == {}, path
    finally:
        if old is None:
            os.environ.pop("ENABLE_TRACK3_R5C_DDP_CANDIDATE", None)
        else:
            os.environ["ENABLE_TRACK3_R5C_DDP_CANDIDATE"] = old

    print("TRACK3_R5C_DDP_CANDIDATE_DEFAULT_TEST_OK")


if __name__ == "__main__":
    main()
