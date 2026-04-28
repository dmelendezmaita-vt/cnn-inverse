#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from split_array_cache_utils import warm_split_array_cache_for_rows


def load_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_matches(row, args):
    if args.row_ids and row["row_id"] not in args.row_ids:
        return False
    if args.phases and row["phase"] not in args.phases:
        return False
    if args.strategies and row["strategy_id"] not in args.strategies:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-csv", required=True)
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    matrix_rows = [row for row in load_csv(Path(args.matrix_csv)) if row_matches(row, args)]
    if args.limit > 0:
        matrix_rows = matrix_rows[: args.limit]

    results = warm_split_array_cache_for_rows(matrix_rows, force=args.force)
    print(
        json.dumps(
            {
                "matrix_csv": str(args.matrix_csv),
                "selected_rows": len(matrix_rows),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
