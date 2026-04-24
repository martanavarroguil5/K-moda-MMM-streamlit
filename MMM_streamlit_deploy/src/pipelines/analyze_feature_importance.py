from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.modeling.feature_importance import run_random_forest_feature_importance


def main() -> None:
    artifacts = run_random_forest_feature_importance()
    print("Random Forest feature importance complete.")
    print("Recommended spec:", artifacts.recommended_spec)
    print("Test metrics:", artifacts.metrics)
    print(artifacts.spec_summary.to_string(index=False))
    print(artifacts.feature_importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
