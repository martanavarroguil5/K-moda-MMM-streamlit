from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.modeling.model_spec_review import run_model_spec_review


def main() -> None:
    artifacts = run_model_spec_review()
    print("Model specification review complete.")
    print("Recommended spec:", artifacts.recommended_spec)
    print(artifacts.summary.to_string(index=False))


if __name__ == "__main__":
    main()
