from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.modeling.arimax import run_arimax_pipeline
from src.modeling.predictive_nowcast import run_predictive_nowcast_pipeline


def main() -> None:
    predictive_artifacts = run_predictive_nowcast_pipeline()
    arimax_artifacts = run_arimax_pipeline()

    print("Training pipeline complete.")
    print("Predictive winner:", predictive_artifacts["winner"])
    print("Predictive test metrics:", predictive_artifacts["winner_test_metrics"])
    print("Simulation model:", "ARIMAX")
    print("ARIMAX test metrics:", arimax_artifacts.test_metrics)


if __name__ == "__main__":
    main()
