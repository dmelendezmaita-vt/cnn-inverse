#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Wait for posterior artifacts, then refresh downstream diagnostics and queues.")
    ap.add_argument("--wait-file", action="append", required=True)
    ap.add_argument("--command", action="append", required=True)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    wait_files = [Path(x) for x in args.wait_file]
    while True:
        if all(path.exists() for path in wait_files):
            break
        time.sleep(args.poll_sec)
    for cmd in args.command:
        proc = subprocess.run(cmd, shell=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
