#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hh_track4_assumption_conditioned_surrogate_bundle_integration_utils_20260425 import (  # noqa: E402
    forwarded_flag_value,
    resolve_bundle_save_dir,
    run_bundle_delegate,
)


TARGET_SCRIPT = SCRIPT_DIR / "run_hh_track4_assumption_conditioned_wasserstein_abc_20260425.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description="Bundle-integrated wrapper for the assumption-conditioned Wasserstein ABC runner.",
        epilog="All unrecognized arguments are forwarded to the underlying Wasserstein ABC runner.",
    )
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--bundle-label", default="")
    return ap.parse_known_args()


def main() -> None:
    args, forwarded = parse_args()
    bundle_label = args.bundle_label or "default"
    save_dir, used_default = resolve_bundle_save_dir(
        method_stem="wasserstein_abc_surrogate_bundle",
        cli_save_dir=args.save_dir,
        forwarded_args=forwarded,
        bundle_label=bundle_label,
    )
    run_bundle_delegate(
        target_script=TARGET_SCRIPT,
        wrapper_name=Path(__file__).name,
        method_family="sliced_wasserstein_abc",
        bundle_label=bundle_label,
        save_dir=save_dir,
        forwarded_args=forwarded,
        used_default_save_dir=used_default,
        extra_metadata={
            "n_particles_hint": forwarded_flag_value(forwarded, "--n-particles"),
            "keep_top_hint": forwarded_flag_value(forwarded, "--keep-top"),
            "n_projections_hint": forwarded_flag_value(forwarded, "--n-projections"),
        },
    )


if __name__ == "__main__":
    main()
