from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def unique_weeks(df: pd.DataFrame, date_col: str = "semana_inicio") -> List[pd.Timestamp]:
    return sorted(pd.to_datetime(df[date_col]).drop_duplicates().tolist())


def expanding_year_splits(
    df: pd.DataFrame,
    date_col: str = "semana_inicio",
    min_train_year: int = 2021,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    weeks = pd.to_datetime(df[date_col])
    years = sorted(weeks.dt.year.unique().tolist())
    splits = []
    for year in years:
        if year <= min_train_year:
            continue
        train_mask = weeks.dt.year < year
        valid_mask = weeks.dt.year == year
        if train_mask.any() and valid_mask.any():
            splits.append((train_mask.to_numpy(), valid_mask.to_numpy(), f"validate_{year}"))
    return splits


def panel_time_cv_indices(
    df: pd.DataFrame,
    date_col: str = "semana_inicio",
    n_splits: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    weeks = np.array(unique_weeks(df, date_col))
    if len(weeks) <= n_splits + 1:
        raise ValueError("Not enough unique weeks to build time-based CV splits.")
    fold_edges = np.linspace(0, len(weeks), n_splits + 2, dtype=int)
    indices = []
    for split_idx in range(1, len(fold_edges) - 1):
        train_weeks = weeks[: fold_edges[split_idx]]
        valid_weeks = weeks[fold_edges[split_idx] : fold_edges[split_idx + 1]]
        train_mask = df[date_col].isin(train_weeks).to_numpy()
        valid_mask = df[date_col].isin(valid_weeks).to_numpy()
        if train_mask.any() and valid_mask.any():
            indices.append((np.where(train_mask)[0], np.where(valid_mask)[0]))
    return indices


def quarter_label(series: Iterable[pd.Timestamp]) -> pd.Series:
    dates = pd.to_datetime(pd.Series(series))
    quarter = ((dates.dt.month - 1) // 3) + 1
    return dates.dt.year.astype(str) + "-Q" + quarter.astype(str)

