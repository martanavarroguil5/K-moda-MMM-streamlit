from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(ROOT))

from src.simulation.workflow import run_budget_simulation


def main() -> None:
    scenarios_df, mroi_df = run_budget_simulation()
    print(
        scenarios_df[
            [
                "scenario",
                "predicted_sales_2024",
                "predicted_gross_profit_2024",
                "delta_vs_historical_sales",
                "delta_vs_historical_profit",
            ]
        ].to_string(index=False)
    )
    print()
    print(mroi_df.to_string(index=False))


if __name__ == "__main__":
    main()
