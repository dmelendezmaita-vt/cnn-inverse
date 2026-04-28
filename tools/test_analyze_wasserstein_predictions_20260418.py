#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "analyze_wasserstein_predictions_20260418.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        run_dir.mkdir(parents=True)
        np.savez_compressed(
            run_dir / "predictions.npz",
            test_true=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float),
            test_pred=np.array([[1.0, 12.0], [2.0, 18.0], [4.0, 33.0]], dtype=float),
        )
        proc = subprocess.run(
            ["python", str(SCRIPT), str(run_dir)],
            text=True,
            capture_output=True,
            check=True,
        )
        summary = json.loads(proc.stdout)
        assert summary["n_files"] == 1
        assert summary["n_rows"] == 2
        assert "test" in summary["splits"]
        assert summary["per_split"]["test"]["mean_wasserstein_raw"] >= 0.0
        assert summary["per_split"]["test"]["mean_wasserstein_log10"] >= 0.0
        print("ANALYZE_WASSERSTEIN_PREDICTIONS_TEST_OK")


if __name__ == "__main__":
    main()
