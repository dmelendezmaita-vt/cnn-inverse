#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


TERMINAL = {"COMPLETED", "FAILED", "BLOCKED", "SHOWN_USELESS"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Wait for a registry row to finish, then optionally launch a fallback backlog.")
    ap.add_argument("--registry-csv", required=True)
    ap.add_argument("--row-id", required=True)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--fallback-cmd", default="")
    return ap.parse_args()


def read_status(path: Path, row_id: str) -> str | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_id") == row_id:
                return row.get("status") or None
    return None


def main() -> None:
    args = parse_args()
    registry = Path(args.registry_csv)
    while True:
        status = read_status(registry, args.row_id)
        if status in TERMINAL:
            break
        time.sleep(args.poll_sec)
    if status == "FAILED" and args.fallback_cmd:
        proc = subprocess.run(args.fallback_cmd, shell=True, check=False)
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
