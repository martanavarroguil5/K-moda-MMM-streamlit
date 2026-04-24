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


def main() -> None:
    artifacts = run_arimax_pipeline()
    print("ARIMAX training complete.")
    print("Selected order:", artifacts.selected_order)
    print("Train metrics:", artifacts.train_metrics)
    print("Test metrics:", artifacts.test_metrics)
    print(artifacts.coefficients[artifacts.coefficients["is_business_variable"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
