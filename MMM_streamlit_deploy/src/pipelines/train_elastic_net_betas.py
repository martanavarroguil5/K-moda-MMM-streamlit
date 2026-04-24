from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.modeling.beta_workflow import run_beta_training_pipeline


def main() -> None:
    artifacts = run_beta_training_pipeline()
    print("Elastic Net beta pipeline complete.")
    print("Spec:", artifacts["model_results"]["winner"])
    print("Test metrics:", artifacts["model_results"]["test_metrics"])
    print("Stable media betas:", artifacts["model_results"]["stable_media_features"])


if __name__ == "__main__":
    main()
