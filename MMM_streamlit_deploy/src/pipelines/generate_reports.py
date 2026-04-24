from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.validation.workflow import generate_visual_reports


def main() -> None:
    generate_visual_reports()
    print("Visual reports generated.")


if __name__ == "__main__":
    main()
