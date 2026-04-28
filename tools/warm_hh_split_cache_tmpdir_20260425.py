#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_DATA_DIR = str(
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)
sys.path.append(str(REPO / "src"))
sys.path.append(str(REPO / "tools"))

from data import _resolve_split_array_cache_dir  # noqa: E402
from split_array_cache_utils import warm_split_array_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage HH split caches into local TMPDIR-backed scratch.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--data-dir", default=DEFAULT_SHARED_DATA_DIR)
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--currents", default="0.1")
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--n-validate", type=int, required=True)
    ap.add_argument("--n-test", type=int, required=True)
    ap.add_argument("--ensure-source-cache", action="store_true")
    ap.add_argument("--report-json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    params = yaml.safe_load(Path(args.params).read_text())
    params.setdefault("data", {})
    params["data"]["data_dir"] = args.data_dir
    params["data"]["data_prefix"] = args.data_prefix
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["Ntrain"] = int(args.n_train)
    params["data"]["Nvalidate"] = int(args.n_validate)
    params["data"]["Ntest"] = int(args.n_test)

    os.environ.setdefault("NC_STAGE_SPLIT_CACHE_TO_TMPDIR", "1")
    if not os.environ.get("NC_LOCAL_STAGE_ROOT") and os.environ.get("TMPDIR"):
        os.environ["NC_LOCAL_STAGE_ROOT"] = os.environ["TMPDIR"]

    rows = []
    for curr in [x.strip() for x in args.currents.split(",") if x.strip()]:
        params["data"]["curr"] = curr
        source_status = None
        if args.ensure_source_cache:
            with tempfile.TemporaryDirectory(prefix=f"warm_split_cache_{curr.replace('.', 'p')}") as tmp_dir:
                params_path = Path(tmp_dir) / "params.yaml"
                params_path.write_text(yaml.safe_dump(params, sort_keys=False))
                old_stage_flag = os.environ.get("NC_STAGE_SPLIT_CACHE_TO_TMPDIR")
                try:
                    os.environ["NC_STAGE_SPLIT_CACHE_TO_TMPDIR"] = "0"
                    source_status = warm_split_array_cache(
                        params_path,
                        repo_root=REPO,
                        shared_data_dir=args.data_dir,
                        force=False,
                    )
                finally:
                    if old_stage_flag is None:
                        os.environ.pop("NC_STAGE_SPLIT_CACHE_TO_TMPDIR", None)
                    else:
                        os.environ["NC_STAGE_SPLIT_CACHE_TO_TMPDIR"] = old_stage_flag
        started = time.time()
        staged_dir = _resolve_split_array_cache_dir(params["data"])
        elapsed = time.time() - started
        row = {
            "curr": curr,
            "cache_dir": str(staged_dir),
            "elapsed_sec": elapsed,
            "exists": bool(staged_dir and Path(staged_dir).exists()),
            "tmpdir": os.environ.get("TMPDIR"),
            "local_stage_root": os.environ.get("NC_LOCAL_STAGE_ROOT"),
        }
        if source_status is not None:
            row["source_cache_status"] = source_status.get("status")
            row["source_cache_dir"] = source_status.get("cache_dir")
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
