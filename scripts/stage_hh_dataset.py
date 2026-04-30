#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hh_repo_utils import resolve_hh_dataset_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage the HH dataset to a reusable node-local location.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-prefix", default=None)
    ap.add_argument("--curr", default=None)
    ap.add_argument(
        "--stage-mode",
        required=True,
        choices=["none", "prepared_shared", "shm_curr_copy", "shm_full_copy", "nvme_full_extract"],
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_hh_dataset_root(
        args.data_dir,
        args.data_prefix,
        curr=args.curr,
        stage_mode=args.stage_mode,
    )
    payload = {
        "resolved_data_dir": str(resolved),
        "stage_mode": args.stage_mode,
        "curr": args.curr,
        "data_prefix": args.data_prefix,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
