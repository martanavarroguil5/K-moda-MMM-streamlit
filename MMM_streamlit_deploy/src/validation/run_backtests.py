from __future__ import annotations

import pandas as pd

from src.common.config import CONFIG


def load_backtests() -> pd.DataFrame:
    if not CONFIG.backtest_results_file.exists():
        raise FileNotFoundError(
            "Backtest results not found. Run `python3 -m src.modeling.train_mmm` first."
        )
    return pd.read_csv(CONFIG.backtest_results_file)


def summarize_backtests(backtests: pd.DataFrame) -> pd.DataFrame:
    summary = (
        backtests.groupby("spec", as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            max_negative_media_coefficients=("negative_media_coefficients", "max"),
        )
        .sort_values("mean_mape")
    )
    return summary


def main() -> None:
    backtests = load_backtests()
    summary = summarize_backtests(backtests)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
