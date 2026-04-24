from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.modeling.multicollinearity import run_multicollinearity_review


def main() -> None:
    artifacts = run_multicollinearity_review()
    print("Multicollinearity review complete.")
    print("Dropped weight reference:", artifacts.dropped_weight_reference)
    print("Top raw-media VIF:")
    print(artifacts.raw_vif.head(10).to_string(index=False))
    print("Top weighted-media VIF:")
    print(artifacts.weight_vif.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
