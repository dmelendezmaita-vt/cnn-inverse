#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from split_array_cache_utils import warm_split_array_cache

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"
SHARED_DATA_DIR = REPO / "data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track4_hh_full/concatenated_data"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Warm larger HH Track4 split-array caches.")
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1201)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    params = yaml.safe_load(BASE_CONFIG.read_text())
    params.setdefault("data", {})
    params["data"]["data_dir"] = str(SHARED_DATA_DIR)
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["random_seed"] = int(args.seed)
    params["data"]["Ntrain"] = int(args.n_train)
    params["data"]["Nvalidate"] = 1024
    params["data"]["Ntest"] = 1024
    tmp_dir = REPO / "data/important_notes/optimization_track_20260422_hh_track4_split_array_cache_warmup" / "params"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    params_path = tmp_dir / f"params_track4_hh_full_n{args.n_train}_s{args.seed}.yaml"
    params_path.write_text(yaml.safe_dump(params, sort_keys=False))

    result = warm_split_array_cache(
        params_path,
        repo_root=REPO,
        shared_data_dir=SHARED_DATA_DIR,
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
