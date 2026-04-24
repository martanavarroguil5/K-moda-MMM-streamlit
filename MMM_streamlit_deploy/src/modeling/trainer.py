from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.modeling.specs import ELASTIC_NET_ALPHAS, ELASTIC_NET_L1, RANDOM_STATE, TRAFFIC_COLUMNS
from src.validation.backtesting import panel_time_cv_indices


def ensure_prerequisites() -> None:
    if not CONFIG.model_dataset_file.exists():
        raise FileNotFoundError(
            f"Model dataset not found at {CONFIG.model_dataset_file}. Run src/pipelines/build_dataset.py first."
        )
    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.reports_figures_dir.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.model_dataset_file)
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"])
    city_dummies = pd.get_dummies(df["ciudad"], prefix="city", drop_first=True, dtype=float)
    df = pd.concat([df, city_dummies], axis=1)
    return df.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)


def media_columns(df: pd.DataFrame) -> List[str]:
    return sorted([column for column in df.columns if column.startswith("media_")])


def budget_share_columns(df: pd.DataFrame) -> List[str]:
    return sorted([column for column in df.columns if column.startswith("budget_share_pct_")])


def city_dummy_columns(df: pd.DataFrame) -> List[str]:
    return sorted([column for column in df.columns if column.startswith("city_")])


def build_feature_frame(
    df: pd.DataFrame,
    control_columns: List[str],
    city_columns: List[str],
    media_feature_columns: List[str] | None = None,
    include_traffic: bool = False,
) -> pd.DataFrame:
    columns = control_columns + city_columns
    if include_traffic:
        columns += TRAFFIC_COLUMNS
    if media_feature_columns:
        columns += media_feature_columns
    return df[columns].astype(float)


def fit_elastic_net(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "ventas_netas",
    positive: bool = False,
) -> Tuple[StandardScaler, ElasticNetCV]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[feature_columns])
    y_train = df_train[target_col].to_numpy(dtype=float)
    cv_indices = panel_time_cv_indices(df_train[["semana_inicio"]].copy(), n_splits=3)
    model = ElasticNetCV(
        alphas=ELASTIC_NET_ALPHAS,
        l1_ratio=ELASTIC_NET_L1,
        cv=cv_indices,
        random_state=RANDOM_STATE,
        max_iter=50000,
        selection="cyclic",
        positive=positive,
    )
    model.fit(x_train, y_train)
    return scaler, model


def fit_elastic_net_with_params(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    alpha: float,
    l1_ratio: float,
    target_col: str = "ventas_netas",
    positive: bool = False,
) -> Tuple[StandardScaler, ElasticNet]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[feature_columns])
    y_train = df_train[target_col].to_numpy(dtype=float)
    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        random_state=RANDOM_STATE,
        max_iter=200000,
        selection="cyclic",
        positive=positive,
    )
    model.fit(x_train, y_train)
    return scaler, model


def predict_with_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: StandardScaler,
    model: ElasticNetCV | ElasticNet,
) -> np.ndarray:
    x = scaler.transform(df[feature_columns])
    return model.predict(x)


def original_scale_coefficients(
    scaler: StandardScaler,
    model: ElasticNetCV | ElasticNet,
    feature_columns: List[str],
) -> pd.Series:
    raw_coef = model.coef_ / scaler.scale_
    return pd.Series(raw_coef, index=feature_columns, dtype=float)


def standardized_coefficients(
    model: ElasticNetCV | ElasticNet,
    feature_columns: List[str],
) -> pd.Series:
    return pd.Series(model.coef_, index=feature_columns, dtype=float)


def original_scale_intercept(
    scaler: StandardScaler,
    model: ElasticNetCV | ElasticNet,
    feature_columns: List[str],
) -> float:
    coef = original_scale_coefficients(scaler, model, feature_columns)
    return float(model.intercept_ - np.sum(scaler.mean_ * coef.to_numpy()))
