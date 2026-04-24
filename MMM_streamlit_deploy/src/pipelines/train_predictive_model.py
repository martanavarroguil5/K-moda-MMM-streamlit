from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.modeling.predictive_nowcast import run_predictive_nowcast_pipeline


def main() -> None:
    artifacts = run_predictive_nowcast_pipeline()
    print("Predictive pipeline complete.")
    print("Winning specification:", artifacts["winner"])
    print("Winner test metrics:", artifacts["winner_test_metrics"])


if __name__ == "__main__":
    main()
