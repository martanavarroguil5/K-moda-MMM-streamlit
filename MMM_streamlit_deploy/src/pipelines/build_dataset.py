from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.common.config import CONFIG
from src.data.weekly_aggregations import ensure_dirs
from src.features.dataset_builder import build_model_dataset


def main() -> None:
    ensure_dirs()
    dataset, checks = build_model_dataset()
    print("Causal dataset built:", dataset.shape)
    print("Diagnostic dataset path:", str(CONFIG.diagnostic_dataset_file))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
