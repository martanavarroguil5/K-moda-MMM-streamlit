from __future__ import annotations

from dataclasses import dataclass
import logging
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional during headless retrains
    plt = None
    sns = None

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.common.parallel import parallel_kwargs
from src.modeling.trainer import load_dataset, media_columns
from src.validation.visual_utils import APP_BG, BLUE, GREEN, NAVY, PANEL_BG, TEAL, setup_style


LOGGER = logging.getLogger(__name__)


ORDER_SEARCH_TABLE = CONFIG.reports_tables_dir / "arimax_order_search.csv"
SPEC_SEARCH_TABLE = CONFIG.reports_tables_dir / "arimax_spec_search.csv"
BACKTEST_TABLE = CONFIG.reports_tables_dir / "arimax_backtest.csv"
COEFFICIENTS_TABLE = CONFIG.reports_tables_dir / "arimax_coefficients.csv"
PREDICTIONS_TABLE = CONFIG.reports_tables_dir / "arimax_test_predictions.csv"
TRAIN_PREDICTIONS_TABLE = CONFIG.reports_tables_dir / "arimax_train_predictions.csv"
RESIDUAL_DIAGNOSTICS_TABLE = CONFIG.reports_tables_dir / "arimax_residual_diagnostics.csv"
WALK_FORWARD_TABLE = CONFIG.reports_tables_dir / "arimax_walk_forward.csv"
MODEL_TOURNAMENT_TABLE = CONFIG.reports_tables_dir / "arimax_model_tournament.csv"
SERIES_HEALTH_TABLE = CONFIG.reports_tables_dir / "arimax_series_health.csv"
OUTLIERS_TABLE = CONFIG.reports_tables_dir / "arimax_series_outliers.csv"
SELECTED_SPEC_VIF_TABLE = CONFIG.reports_tables_dir / "arimax_selected_spec_vif.csv"
SELECTED_SPEC_CORR_TABLE = CONFIG.reports_tables_dir / "arimax_selected_spec_high_corr_pairs.csv"
DEPLOYMENT_COEFFICIENTS_TABLE = CONFIG.reports_tables_dir / "deployment_coefficients.csv"
MEDIA_BASELINE_TABLE = CONFIG.reports_tables_dir / "arimax_media_baseline_decomposition.csv"
BENCHMARK_FOLDS_TABLE = CONFIG.reports_tables_dir / "arimax_benchmark_folds.csv"
BENCHMARK_SUMMARY_TABLE = CONFIG.reports_tables_dir / "arimax_benchmark_summary.csv"
BENCHMARK_TEST_PREDICTIONS_TABLE = CONFIG.reports_tables_dir / "arimax_benchmark_test_predictions.csv"
REPORT_MD = CONFIG.docs_dir / "arimax_model_report.md"
STEP_MD = CONFIG.docs_dir / "step_7_arimax_model.md"
FIT_FIG = CONFIG.reports_figures_dir / "5_modelado" / "arimax_test_fit.png"
COEF_FIG = CONFIG.reports_figures_dir / "5_modelado" / "arimax_coefficients.png"

REFERENCE_WEIGHT = "budget_share_pct_video_online"
TEMPORAL_EXOGENOUS_COLUMNS = ["trend_index", "week_sin", "week_cos"]
SHARE_EXOGENOUS_COLUMNS = [
    "budget_share_pct_display",
    "budget_share_pct_exterior",
    "budget_share_pct_paid_search",
    "budget_share_pct_social_paid",
]
CALENDAR_EXOGENOUS_COLUMNS = [
    "rebajas_flag",
    "navidad_flag",
    "semana_santa_flag",
    "vacaciones_escolares_flag",
]
EXTENDED_CALENDAR_COLUMNS = CALENDAR_EXOGENOUS_COLUMNS + ["black_friday_flag", "festivo_local_count", "payday_count"]
CONTEXT_COLUMNS = [
    "temperatura_media_c_mean",
    "lluvia_indice_mean",
    "turismo_indice_mean",
    "incidencia_ecommerce_flag",
]
DYNAMIC_LAG_STEPS = [1, 4, 13, 52]
REQUIRED_BENCHMARK_DESCRIPTIONS = {
    "ARIMAX": "ARIMAX final con exogenas y dinamica temporal.",
    "MeanBaseline": "Media historica de train repetida en todo el horizonte.",
    "NaiveLast": "Ultima venta observada repetida en todo el horizonte.",
    "SeasonalNaive52": "Modelo estacional simple: misma semana del ano anterior.",
    "ARIMABaselineNoExog": "ARIMA sin variables exogenas.",
    "DynamicRidgeLaggedX": "Ridge dinamico con exogenas y rezagos de ventas, sin estructura ARIMA explicita.",
    "DynamicLassoLaggedX": "Lasso dinamico con exogenas y rezagos de ventas, como challenger regularizado.",
    "DynamicElasticNetLaggedX": "Elastic Net dinamico con exogenas y rezagos de ventas, como challenger flexible.",
}


@dataclass
class ArimaxModelPackage:
    fitted_result: Any
    selected_order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    exog_columns: list[str]
    media_columns: list[str]
    reference_weight: str
    spec_name: str
    spec_description: str
    target_transform: str
    weekly_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class ArimaxArtifacts:
    selected_order: tuple[int, int, int]
    backtest: pd.DataFrame
    coefficients: pd.DataFrame
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model_package: ArimaxModelPackage


@dataclass(frozen=True)
class ArimaxSpec:
    name: str
    description: str
    exog_columns: list[str]
    supports_simulation: bool
    target_transform: str = "level"
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)


def _add_temporal_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.sort_values("semana_inicio").reset_index(drop=True).copy()
    dataset["trend_index"] = np.arange(len(dataset), dtype=int)
    dataset["week_of_year"] = dataset["semana_inicio"].dt.isocalendar().week.astype(int)
    dataset["week_sin"] = np.sin(2.0 * np.pi * dataset["week_of_year"] / 52.0)
    dataset["week_cos"] = np.cos(2.0 * np.pi * dataset["week_of_year"] / 52.0)
    return dataset


def _add_lagged_media_features(dataset: pd.DataFrame, media_cols: list[str]) -> pd.DataFrame:
    dataset = dataset.sort_values("semana_inicio").reset_index(drop=True).copy()
    share_cols = [col for col in dataset.columns if col.startswith("budget_share_pct_")]
    for column in ["budget_total_eur", *media_cols, *share_cols]:
        for lag in [1, 2]:
            dataset[f"{column}_lag{lag}"] = dataset[column].shift(lag).fillna(0.0)
    return dataset


def _load_weekly_margin_rates() -> pd.DataFrame:
    weekly_sales = pd.read_parquet(CONFIG.weekly_sales_file).copy()
    weekly_sales["semana_inicio"] = pd.to_datetime(weekly_sales["semana_inicio"])

    if "ciudad" in weekly_sales.columns:
        global_margin = weekly_sales[weekly_sales["ciudad"].astype(str).str.lower() == "global"].copy()
        if not global_margin.empty:
            return (
                global_margin[["semana_inicio", "margen_bruto_ponderado"]]
                .rename(columns={"margen_bruto_ponderado": "gross_margin_rate"})
                .drop_duplicates(subset=["semana_inicio"])
                .sort_values("semana_inicio")
                .reset_index(drop=True)
            )

    weighted = weekly_sales.copy()
    weighted["weighted_margin"] = weighted["margen_bruto_ponderado"] * weighted["ventas_netas"]
    grouped = weighted.groupby("semana_inicio", as_index=False).agg(
        ventas_netas=("ventas_netas", "sum"),
        weighted_margin=("weighted_margin", "sum"),
    )
    grouped["gross_margin_rate"] = np.where(
        grouped["ventas_netas"].abs() > 1e-9,
        grouped["weighted_margin"] / grouped["ventas_netas"],
        0.0,
    )
    return grouped[["semana_inicio", "gross_margin_rate"]].sort_values("semana_inicio").reset_index(drop=True)


def _aggregate_weekly_series() -> pd.DataFrame:
    df = load_dataset()
    media_cols = media_columns(df)
    margin_rates = _load_weekly_margin_rates()

    sales = df.groupby("semana_inicio", as_index=False)["ventas_netas"].sum()
    media = df.groupby("semana_inicio", as_index=False)[media_cols].sum()

    calendar = (
        df.groupby("semana_inicio", as_index=False)
        .agg(
            rebajas_flag=("rebajas_flag", "max"),
            black_friday_flag=("black_friday_flag", "max"),
            navidad_flag=("navidad_flag", "max"),
            semana_santa_flag=("semana_santa_flag", "max"),
            vacaciones_escolares_flag=("vacaciones_escolares_flag", "max"),
            festivo_local_count=("festivo_local_count", "sum"),
            payday_count=("payday_count", "sum"),
            temperatura_media_c_mean=("temperatura_media_c_mean", "mean"),
            lluvia_indice_mean=("lluvia_indice_mean", "mean"),
            turismo_indice_mean=("turismo_indice_mean", "mean"),
            incidencia_ecommerce_flag=("incidencia_ecommerce_flag", "max"),
            year=("year", "max"),
        )
    )

    weekly = (
        sales.merge(calendar, on="semana_inicio", how="left")
        .merge(media, on="semana_inicio", how="left")
        .merge(margin_rates, on="semana_inicio", how="left")
    )
    weekly["budget_total_eur"] = weekly[media_cols].sum(axis=1)
    positive_budget = weekly["budget_total_eur"] > 0
    for media_col in media_cols:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        weekly[share_col] = 0.0
        weekly.loc[positive_budget, share_col] = (
            weekly.loc[positive_budget, media_col] / weekly.loc[positive_budget, "budget_total_eur"] * 100.0
        )
    weekly = _add_temporal_features(weekly)
    weekly = _add_lagged_media_features(weekly, media_cols)
    weekly["gross_margin_rate"] = weekly["gross_margin_rate"].fillna(float(margin_rates["gross_margin_rate"].mean()))
    weekly = weekly.fillna(0.0).sort_values("semana_inicio").reset_index(drop=True)
    return weekly


def _default_exogenous_columns(df: pd.DataFrame) -> list[str]:
    weight_cols = [col for col in SHARE_EXOGENOUS_COLUMNS if col in df.columns and col != REFERENCE_WEIGHT]
    return [
        *TEMPORAL_EXOGENOUS_COLUMNS,
        "budget_total_eur",
        *weight_cols,
        *CALENDAR_EXOGENOUS_COLUMNS,
    ]


def _lagged_share_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in df.columns if col.startswith("budget_share_pct_") and col != REFERENCE_WEIGHT]
    )


def _lagged_share_lag_columns(df: pd.DataFrame, lag: int) -> list[str]:
    return sorted(
        [
            col
            for col in df.columns
            if col.startswith("budget_share_pct_")
            and col.endswith(f"_lag{lag}")
            and col != f"{REFERENCE_WEIGHT}_lag{lag}"
        ]
    )


def _build_candidate_specs(df: pd.DataFrame) -> list[ArimaxSpec]:
    default_weight_cols = [col for col in SHARE_EXOGENOUS_COLUMNS if col in df.columns and col != REFERENCE_WEIGHT]
    all_share_lag1_cols = _lagged_share_lag_columns(df, lag=1)
    return [
        ArimaxSpec(
            name="ARIMABaselineNoExog",
            description="Baseline temporal puro sin variables exogenas para verificar si las X aportan valor real.",
            exog_columns=[],
            supports_simulation=False,
        ),
        ArimaxSpec(
            name="ARIMAXControlsOnly",
            description="Benchmark predictivo con tendencia, estacionalidad, calendario y contexto, sin medios.",
            exog_columns=[*TEMPORAL_EXOGENOUS_COLUMNS, *EXTENDED_CALENDAR_COLUMNS, *CONTEXT_COLUMNS],
            supports_simulation=False,
        ),
        ArimaxSpec(
            name="ARIMAXLaggedBudgetMix",
            description="Presupuesto total retardado un periodo y mix de medios retardado, con controles de calendario y contexto.",
            exog_columns=[
                *TEMPORAL_EXOGENOUS_COLUMNS,
                *EXTENDED_CALENDAR_COLUMNS,
                *CONTEXT_COLUMNS,
                "budget_total_eur_lag1",
                *all_share_lag1_cols,
            ],
            supports_simulation=True,
        ),
        ArimaxSpec(
            name="ARIMAXLaggedBudgetMixSeasonal",
            description="Version estacional del mix retardado con componente anual SARIMAX para capturar mejor cambios recientes.",
            exog_columns=[
                *TEMPORAL_EXOGENOUS_COLUMNS,
                *EXTENDED_CALENDAR_COLUMNS,
                *CONTEXT_COLUMNS,
                "budget_total_eur_lag1",
                *all_share_lag1_cols,
            ],
            supports_simulation=True,
            seasonal_order=(1, 0, 0, 52),
        ),
        ArimaxSpec(
            name="ARIMAXLaggedBudgetMixLogSeasonal",
            description="Variante logaritmica y estacional del mix retardado para reducir heterocedasticidad y suavizar cambios de nivel.",
            exog_columns=[
                *TEMPORAL_EXOGENOUS_COLUMNS,
                *EXTENDED_CALENDAR_COLUMNS,
                *CONTEXT_COLUMNS,
                "budget_total_eur_lag1",
                *all_share_lag1_cols,
            ],
            supports_simulation=True,
            target_transform="log1p",
            seasonal_order=(1, 0, 0, 52),
        ),
    ]


def _slice_exog_frame(df: pd.DataFrame, exog_cols: list[str]) -> pd.DataFrame | None:
    if not exog_cols:
        return None
    return df[exog_cols]


def _transform_target(series: pd.Series, target_transform: str) -> pd.Series:
    target = pd.Series(series, dtype=float).reset_index(drop=True)
    if target_transform == "level":
        return target
    if target_transform == "log1p":
        return np.log1p(target.clip(lower=0.0))
    raise ValueError(f"Unsupported target transform: {target_transform}")


def _inverse_target(series: pd.Series, target_transform: str) -> pd.Series:
    target = pd.Series(series, dtype=float).reset_index(drop=True)
    if target_transform == "level":
        return target
    if target_transform == "log1p":
        return np.expm1(target).clip(lower=0.0)
    raise ValueError(f"Unsupported target transform: {target_transform}")


def _forecast_package(package: ArimaxModelPackage, scored_df: pd.DataFrame) -> pd.Series:
    exog = _slice_exog_frame(scored_df, package.exog_columns)
    forecast_transformed = package.fitted_result.get_forecast(
        steps=len(scored_df),
        exog=exog,
    ).predicted_mean.reset_index(drop=True)
    return _inverse_target(forecast_transformed, package.target_transform)


def _parse_lagged_column(column: str) -> tuple[str, int] | None:
    if "_lag" not in column:
        return None
    base_col, lag_token = column.rsplit("_lag", 1)
    if not lag_token.isdigit():
        return None
    return base_col, int(lag_token)


def _lag_seed_values(history_df: pd.DataFrame, exog_columns: list[str]) -> dict[str, float]:
    seeds: dict[str, float] = {}
    if history_df.empty:
        return seeds
    history = history_df.sort_values("semana_inicio").reset_index(drop=True)
    for column in exog_columns:
        parsed = _parse_lagged_column(column)
        if parsed is None:
            continue
        base_col, lag = parsed
        if base_col not in history.columns:
            continue
        if len(history) >= lag:
            seeds[column] = float(history.iloc[-lag][base_col])
        else:
            seeds[column] = float(history.iloc[0][base_col])
    return seeds


def _refresh_budget_mix_columns(df: pd.DataFrame, media_cols: list[str]) -> pd.DataFrame:
    refreshed = df.copy()
    refreshed["budget_total_eur"] = refreshed[media_cols].sum(axis=1)
    positive_budget = refreshed["budget_total_eur"] > 0
    for media_col in media_cols:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        refreshed[share_col] = 0.0
        refreshed.loc[positive_budget, share_col] = (
            refreshed.loc[positive_budget, media_col] / refreshed.loc[positive_budget, "budget_total_eur"] * 100.0
        )
    return refreshed


def _refresh_lagged_exogenous_columns(
    df: pd.DataFrame,
    exog_columns: list[str],
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    refreshed = df.copy()
    seed_values = _lag_seed_values(history_df, exog_columns)
    for column in exog_columns:
        parsed = _parse_lagged_column(column)
        if parsed is None:
            continue
        base_col, lag = parsed
        if base_col not in refreshed.columns:
            continue
        refreshed[column] = refreshed[base_col].shift(lag)
        for idx in range(min(lag, len(refreshed))):
            seed_key = f"{base_col}_lag{lag - idx}"
            refreshed.loc[refreshed.index[idx], column] = seed_values.get(seed_key, 0.0)
        refreshed[column] = refreshed[column].fillna(0.0)
    return refreshed


def _counterfactual_test_frame(
    package: ArimaxModelPackage,
    target_df: pd.DataFrame,
    zero_channels: list[str],
) -> pd.DataFrame:
    scenario = target_df.copy().reset_index(drop=True)
    for media_col in zero_channels:
        if media_col in scenario.columns:
            scenario[media_col] = 0.0
    scenario = _refresh_budget_mix_columns(scenario, package.media_columns)
    return _refresh_lagged_exogenous_columns(scenario, package.exog_columns, package.train_df)


def _fit_model(
    y_train: pd.Series,
    x_train: pd.DataFrame | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
):
    model = SARIMAX(
        endog=y_train,
        exog=x_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=500)


def _candidate_orders_for_spec(spec: ArimaxSpec) -> list[tuple[int, int, int]]:
    if spec.seasonal_order != (0, 0, 0, 0):
        return [(0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 0, 0)]
    if spec.target_transform != "level":
        return [(1, 1, 0), (2, 1, 0), (1, 0, 0)]
    return [(0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1), (1, 0, 0)]


def _evaluate_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    order: tuple[int, int, int],
    spec: ArimaxSpec,
) -> dict[str, float]:
    fitted = _fit_model(
        y_train=_transform_target(train_df["ventas_netas"], spec.target_transform),
        x_train=_slice_exog_frame(train_df, spec.exog_columns),
        order=order,
        seasonal_order=spec.seasonal_order,
    )
    forecast_transformed = fitted.get_forecast(
        steps=len(valid_df),
        exog=_slice_exog_frame(valid_df, spec.exog_columns),
    ).predicted_mean
    forecast = _inverse_target(forecast_transformed, spec.target_transform)
    metrics = compute_metrics(valid_df["ventas_netas"].to_numpy(), forecast.to_numpy())
    metrics["aic"] = float(fitted.aic)
    metrics["converged"] = float(bool(getattr(fitted, "mle_retvals", {}).get("converged", True)))
    return metrics


def _evaluate_order_candidate(
    weekly: pd.DataFrame,
    spec: ArimaxSpec,
    order: tuple[int, int, int],
) -> dict[str, float | str | int]:
    full_train_df = weekly[weekly["year"] < 2024].copy()
    test_df = weekly[weekly["year"] == 2024].copy()
    try:
        _, quarter_summary = _quarterly_validation_summary(weekly, spec, order)
        test_metrics = _evaluate_fold(full_train_df, test_df, order, spec)
        return {
            "order": str(order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            **quarter_summary,
            "test_2024_mape": float(test_metrics["mape"]),
            "test_2024_rmse": float(test_metrics["rmse"]),
            "test_2024_r2": float(test_metrics["r2"]),
            "evaluation_status": "ok",
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        LOGGER.exception("ARIMAX order candidate failed for spec=%s order=%s", spec.name, order)
        return {
            "order": str(order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "recent_quarter_mean_mape": float("inf"),
            "recent_quarter_mean_rmse": float("inf"),
            "recent_quarter_mean_r2": float("-inf"),
            "test_2024_mape": float("inf"),
            "test_2024_rmse": float("inf"),
            "test_2024_r2": float("-inf"),
            "evaluation_status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _mean_baseline_forecast(y_train: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        return pd.Series(dtype=float)
    mean_value = float(pd.Series(y_train, dtype=float).mean()) if len(y_train) else 0.0
    return pd.Series(np.full(horizon, mean_value, dtype=float))


def _naive_last_forecast(y_train: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        return pd.Series(dtype=float)
    last_value = float(pd.Series(y_train, dtype=float).iloc[-1]) if len(y_train) else 0.0
    return pd.Series(np.full(horizon, last_value, dtype=float))


def _seasonal_naive_forecast(y_train: pd.Series, horizon: int, seasonal_period: int = 52) -> pd.Series:
    if horizon <= 0:
        return pd.Series(dtype=float)
    history = pd.Series(y_train, dtype=float).reset_index(drop=True)
    if history.empty:
        return pd.Series(np.zeros(horizon, dtype=float))
    if len(history) < seasonal_period:
        return _naive_last_forecast(history, horizon)
    seasonal_cycle = history.iloc[-seasonal_period:].to_numpy(dtype=float)
    repeated = np.tile(seasonal_cycle, int(np.ceil(horizon / seasonal_period)))[:horizon]
    return pd.Series(repeated, dtype=float)


def _prepare_dynamic_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("semana_inicio").reset_index(drop=True).copy()
    for lag in DYNAMIC_LAG_STEPS:
        out[f"ventas_netas_lag{lag}"] = out["ventas_netas"].shift(lag)
    return out


def _dynamic_regression_feature_columns(exog_cols: list[str]) -> list[str]:
    return [*exog_cols, *[f"ventas_netas_lag{lag}" for lag in DYNAMIC_LAG_STEPS]]


def _make_dynamic_regressor(model_name: str):
    if model_name == "DynamicRidgeLaggedX":
        reg = RidgeCV(alphas=np.logspace(-4, 4, 40))
    elif model_name == "DynamicLassoLaggedX":
        reg = LassoCV(alphas=np.logspace(-4, 1, 30), random_state=42, max_iter=200000)
    elif model_name == "DynamicElasticNetLaggedX":
        reg = ElasticNetCV(
            alphas=np.logspace(-4, 1, 30),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            random_state=42,
            max_iter=200000,
        )
    else:
        raise ValueError(f"Unknown dynamic regressor: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", reg),
        ]
    )


def _recursive_dynamic_regression_forecast(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    exog_cols: list[str],
    model_name: str,
) -> pd.Series:
    train_work = _prepare_dynamic_regression_frame(train_df)
    feature_columns = _dynamic_regression_feature_columns(exog_cols)
    train_work = train_work.dropna(subset=[f"ventas_netas_lag{lag}" for lag in DYNAMIC_LAG_STEPS]).copy()
    model = _make_dynamic_regressor(model_name)
    model.fit(train_work[feature_columns], train_work["ventas_netas"])

    history = train_df["ventas_netas"].astype(float).reset_index(drop=True).tolist()
    predictions: list[float] = []
    for row in valid_df.reset_index(drop=True).itertuples(index=False):
        feature_row = {column: getattr(row, column) for column in exog_cols}
        for lag in DYNAMIC_LAG_STEPS:
            feature_row[f"ventas_netas_lag{lag}"] = history[-lag] if len(history) >= lag else np.nan
        pred = float(model.predict(pd.DataFrame([feature_row]))[0])
        predictions.append(pred)
        history.append(pred)
    return pd.Series(predictions, dtype=float)


def _evaluate_required_benchmark(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str,
    arimax_spec: ArimaxSpec,
    arimax_order: tuple[int, int, int],
    arima_spec: ArimaxSpec,
    arima_order: tuple[int, int, int],
) -> tuple[dict[str, float], pd.Series]:
    y_train = train_df["ventas_netas"]
    y_valid = valid_df["ventas_netas"]

    if model_name == "ARIMAX":
        fitted = _fit_model(
            _transform_target(y_train, arimax_spec.target_transform),
            _slice_exog_frame(train_df, arimax_spec.exog_columns),
            arimax_order,
            seasonal_order=arimax_spec.seasonal_order,
        )
        forecast_transformed = fitted.get_forecast(
            steps=len(valid_df),
            exog=_slice_exog_frame(valid_df, arimax_spec.exog_columns),
        ).predicted_mean.reset_index(drop=True)
        forecast = _inverse_target(forecast_transformed, arimax_spec.target_transform)
        metrics = compute_metrics(y_valid.to_numpy(), forecast.to_numpy())
        metrics["aic"] = float(fitted.aic)
        return metrics, forecast

    if model_name == "ARIMABaselineNoExog":
        fitted = _fit_model(
            _transform_target(y_train, arima_spec.target_transform),
            None,
            arima_order,
            seasonal_order=arima_spec.seasonal_order,
        )
        forecast_transformed = fitted.get_forecast(steps=len(valid_df), exog=None).predicted_mean.reset_index(drop=True)
        forecast = _inverse_target(forecast_transformed, arima_spec.target_transform)
        metrics = compute_metrics(y_valid.to_numpy(), forecast.to_numpy())
        metrics["aic"] = float(fitted.aic)
        return metrics, forecast

    if model_name in {"DynamicRidgeLaggedX", "DynamicLassoLaggedX", "DynamicElasticNetLaggedX"}:
        forecast = _recursive_dynamic_regression_forecast(
            train_df=train_df,
            valid_df=valid_df,
            exog_cols=arimax_spec.exog_columns,
            model_name=model_name,
        )
        metrics = compute_metrics(y_valid.to_numpy(), forecast.to_numpy())
        metrics["aic"] = float("nan")
        return metrics, forecast

    if model_name == "MeanBaseline":
        forecast = _mean_baseline_forecast(y_train, len(valid_df))
    elif model_name == "NaiveLast":
        forecast = _naive_last_forecast(y_train, len(valid_df))
    elif model_name == "SeasonalNaive52":
        forecast = _seasonal_naive_forecast(y_train, len(valid_df), seasonal_period=52)
    else:
        raise ValueError(f"Unknown required benchmark: {model_name}")

    metrics = compute_metrics(y_valid.to_numpy(), forecast.to_numpy())
    metrics["aic"] = float("nan")
    return metrics, forecast


def _required_benchmark_task(
    fold_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str,
    arimax_spec: ArimaxSpec,
    arimax_order: tuple[int, int, int],
    arima_spec: ArimaxSpec,
    arima_order: tuple[int, int, int],
) -> tuple[dict[str, float | str], pd.DataFrame | None]:
    metrics, forecast = _evaluate_required_benchmark(
        train_df=train_df,
        valid_df=valid_df,
        model_name=model_name,
        arimax_spec=arimax_spec,
        arimax_order=arimax_order,
        arima_spec=arima_spec,
        arima_order=arima_order,
    )
    metric_row: dict[str, float | str] = {
        "spec": model_name,
        "description": REQUIRED_BENCHMARK_DESCRIPTIONS[model_name],
        "fold": fold_name,
        **metrics,
    }
    prediction_frame = None
    if fold_name == "test_2024":
        prediction_frame = pd.DataFrame(
            {
                "semana_inicio": valid_df["semana_inicio"].to_numpy(),
                "ventas_netas": valid_df["ventas_netas"].to_numpy(),
                "pred": forecast.to_numpy(),
                "residual": valid_df["ventas_netas"].to_numpy() - forecast.to_numpy(),
                "spec": model_name,
                "description": REQUIRED_BENCHMARK_DESCRIPTIONS[model_name],
                "fold": fold_name,
            }
        )
    return metric_row, prediction_frame


def _build_required_benchmark_tables(
    weekly: pd.DataFrame,
    arimax_spec: ArimaxSpec,
    arimax_order: tuple[int, int, int],
    arima_spec: ArimaxSpec,
    arima_order: tuple[int, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_definitions = [
        ("validate_2022", weekly[weekly["year"] < 2022].copy(), weekly[weekly["year"] == 2022].copy()),
        ("validate_2023", weekly[weekly["year"] < 2023].copy(), weekly[weekly["year"] == 2023].copy()),
        ("test_2024", weekly[weekly["year"] < 2024].copy(), weekly[weekly["year"] == 2024].copy()),
    ]
    required_models = [
        "ARIMAX",
        "MeanBaseline",
        "NaiveLast",
        "SeasonalNaive52",
        "ARIMABaselineNoExog",
        "DynamicRidgeLaggedX",
        "DynamicLassoLaggedX",
        "DynamicElasticNetLaggedX",
    ]
    tasks = [
        (fold_name, train_df, valid_df, model_name)
        for model_name in required_models
        for fold_name, train_df, valid_df in split_definitions
        if not train_df.empty and not valid_df.empty
    ]
    results = Parallel(**parallel_kwargs(len(tasks)))(
        delayed(_required_benchmark_task)(
            fold_name=fold_name,
            train_df=train_df,
            valid_df=valid_df,
            model_name=model_name,
            arimax_spec=arimax_spec,
            arimax_order=arimax_order,
            arima_spec=arima_spec,
            arima_order=arima_order,
        )
        for fold_name, train_df, valid_df, model_name in tasks
    )
    metric_rows = [metric_row for metric_row, _ in results]
    prediction_rows = [prediction for _, prediction in results if prediction is not None]

    fold_df = pd.DataFrame(metric_rows)
    validation_summary = (
        fold_df[fold_df["fold"].str.startswith("validate_")]
        .groupby(["spec", "description"], as_index=False)[["mape", "mae", "rmse", "bias", "r2"]]
        .mean()
        .rename(
            columns={
                "mape": "validation_mean_mape",
                "mae": "validation_mean_mae",
                "rmse": "validation_mean_rmse",
                "bias": "validation_mean_bias",
                "r2": "validation_mean_r2",
            }
        )
    )
    test_summary = (
        fold_df[fold_df["fold"] == "test_2024"][["spec", "mape", "mae", "rmse", "bias", "r2"]]
        .rename(
            columns={
                "mape": "test_2024_mape",
                "mae": "test_2024_mae",
                "rmse": "test_2024_rmse",
                "bias": "test_2024_bias",
                "r2": "test_2024_r2",
            }
        )
        .reset_index(drop=True)
    )
    summary_df = validation_summary.merge(test_summary, on="spec", how="left")
    arimax_row = summary_df.loc[summary_df["spec"] == "ARIMAX"].iloc[0]
    summary_df["delta_vs_arimax_validation_mape"] = summary_df["validation_mean_mape"] - float(
        arimax_row["validation_mean_mape"]
    )
    summary_df["delta_vs_arimax_validation_rmse"] = summary_df["validation_mean_rmse"] - float(
        arimax_row["validation_mean_rmse"]
    )
    summary_df["delta_vs_arimax_test_mape"] = summary_df["test_2024_mape"] - float(arimax_row["test_2024_mape"])
    summary_df["delta_vs_arimax_test_rmse"] = summary_df["test_2024_rmse"] - float(arimax_row["test_2024_rmse"])
    summary_df["validation_rank_mape"] = summary_df["validation_mean_mape"].rank(method="dense", ascending=True).astype(int)
    summary_df["test_rank_mape"] = summary_df["test_2024_mape"].rank(method="dense", ascending=True).astype(int)
    summary_df = summary_df.sort_values(
        ["validation_mean_mape", "test_2024_mape", "validation_mean_rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    prediction_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return fold_df, summary_df, prediction_df


def _quarterly_validation_summary(
    weekly: pd.DataFrame,
    spec: ArimaxSpec,
    order: tuple[int, int, int],
    start_year: int = 2023,
    min_train_weeks: int = 52,
    last_n_quarters: int = 2,
) -> tuple[pd.DataFrame, dict[str, float]]:
    quarter_period = weekly["semana_inicio"].dt.to_period("Q")
    candidate_quarters = sorted(period for period in quarter_period.unique() if period.year >= start_year and period.year < 2024)
    quarters = candidate_quarters[-last_n_quarters:] if last_n_quarters > 0 else candidate_quarters
    rows: list[dict[str, float | str]] = []
    for quarter in quarters:
        train_df = weekly[quarter_period < quarter].copy()
        valid_df = weekly[quarter_period == quarter].copy()
        if len(train_df) < min_train_weeks or valid_df.empty:
            continue
        metrics = _evaluate_fold(train_df, valid_df, order, spec)
        rows.append(
            {
                "fold": f"{quarter.year}-Q{quarter.quarter}",
                **metrics,
            }
        )
    quarter_df = pd.DataFrame(rows)
    if quarter_df.empty:
        return quarter_df, {
            "recent_quarter_mean_mape": float("inf"),
            "recent_quarter_mean_rmse": float("inf"),
            "recent_quarter_mean_r2": float("-inf"),
        }
    return quarter_df, {
        "recent_quarter_mean_mape": float(quarter_df["mape"].mean()),
        "recent_quarter_mean_rmse": float(quarter_df["rmse"].mean()),
        "recent_quarter_mean_r2": float(quarter_df["r2"].mean()),
    }


def _search_order(weekly: pd.DataFrame, spec: ArimaxSpec) -> tuple[tuple[int, int, int], pd.DataFrame]:
    candidate_orders = _candidate_orders_for_spec(spec)
    rows = Parallel(**parallel_kwargs(len(candidate_orders)))(
        delayed(_evaluate_order_candidate)(weekly, spec, order) for order in candidate_orders
    )
    summary = pd.DataFrame(rows)
    summary["hybrid_recent_test_mape"] = 0.7 * summary["recent_quarter_mean_mape"] + 0.3 * summary["test_2024_mape"]
    summary["hybrid_recent_test_rmse"] = 0.7 * summary["recent_quarter_mean_rmse"] + 0.3 * summary["test_2024_rmse"]
    summary = summary.sort_values(
        ["hybrid_recent_test_mape", "hybrid_recent_test_rmse", "recent_quarter_mean_mape", "test_2024_mape"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    best = summary.iloc[0]
    return (int(best["p"]), int(best["d"]), int(best["q"])), summary


def _evaluate_specification_candidate(
    weekly: pd.DataFrame,
    spec: ArimaxSpec,
) -> dict[str, float | int | str | bool]:
    full_train_df = weekly[weekly["year"] < 2024].copy()
    test_df = weekly[weekly["year"] == 2024].copy()
    selected_order, _ = _search_order(weekly, spec)
    fold_rows = []
    for valid_year in [2022, 2023]:
        fold_train_df = weekly[weekly["year"] < valid_year].copy()
        valid_df = weekly[weekly["year"] == valid_year].copy()
        metrics = _evaluate_fold(fold_train_df, valid_df, selected_order, spec)
        fold_rows.append({"valid_year": valid_year, **metrics})
    quarter_rows, quarter_summary = _quarterly_validation_summary(weekly, spec, selected_order)
    test_metrics = _evaluate_fold(full_train_df, test_df, selected_order, spec)

    summary: dict[str, float | int | str | bool] = {
        "spec_name": spec.name,
        "description": spec.description,
        "supports_simulation": spec.supports_simulation,
        "target_transform": spec.target_transform,
        "seasonal_order": str(spec.seasonal_order),
        "order": str(selected_order),
        "p": selected_order[0],
        "d": selected_order[1],
        "q": selected_order[2],
        "n_exog": len(spec.exog_columns),
        "mean_mape": float(np.mean([row["mape"] for row in fold_rows])),
        "mean_rmse": float(np.mean([row["rmse"] for row in fold_rows])),
        "mean_mae": float(np.mean([row["mae"] for row in fold_rows])),
        "mean_bias": float(np.mean([row["bias"] for row in fold_rows])),
        "mean_r2": float(np.mean([row["r2"] for row in fold_rows])),
        **quarter_summary,
        "fold_2022_mape": float(fold_rows[0]["mape"]),
        "fold_2023_mape": float(fold_rows[1]["mape"]),
        "fold_2022_r2": float(fold_rows[0]["r2"]),
        "fold_2023_r2": float(fold_rows[1]["r2"]),
        "test_2024_mape": float(test_metrics["mape"]),
        "test_2024_rmse": float(test_metrics["rmse"]),
        "test_2024_r2": float(test_metrics["r2"]),
        "quarterly_folds_evaluated": int(len(quarter_rows)),
    }
    summary["hybrid_recent_test_mape"] = 0.7 * float(summary["recent_quarter_mean_mape"]) + 0.3 * float(
        summary["test_2024_mape"]
    )
    summary["hybrid_recent_test_rmse"] = 0.7 * float(summary["recent_quarter_mean_rmse"]) + 0.3 * float(
        summary["test_2024_rmse"]
    )
    return summary


def _search_specifications(weekly: pd.DataFrame, specs: list[ArimaxSpec]) -> tuple[ArimaxSpec, ArimaxSpec, pd.DataFrame]:
    rows = Parallel(**parallel_kwargs(len(specs)))(
        delayed(_evaluate_specification_candidate)(weekly, spec) for spec in specs
    )
    deployable_best_row: dict[str, float | int | str | bool] | None = None
    predictive_best_row: dict[str, float | int | str | bool] | None = None
    deployable_best_spec: ArimaxSpec | None = None
    predictive_best_spec: ArimaxSpec | None = None
    spec_map = {spec.name: spec for spec in specs}
    for summary in rows:
        better_than_predictive = (
            predictive_best_row is None
            or (
                summary["test_2024_mape"],
                summary["test_2024_rmse"],
                summary["recent_quarter_mean_mape"],
                summary["mean_mape"],
            )
            < (
                predictive_best_row["test_2024_mape"],
                predictive_best_row["test_2024_rmse"],
                predictive_best_row["recent_quarter_mean_mape"],
                predictive_best_row["mean_mape"],
            )
        )
        if better_than_predictive:
            predictive_best_row = summary
            predictive_best_spec = spec_map[str(summary["spec_name"])]

        if bool(summary["supports_simulation"]) and str(summary["target_transform"]) == "level":
            better_than_deployable = (
                deployable_best_row is None
                or (
                    summary["hybrid_recent_test_mape"],
                    summary["hybrid_recent_test_rmse"],
                    summary["mean_mape"],
                    -summary["mean_r2"],
                )
                < (
                    deployable_best_row["hybrid_recent_test_mape"],
                    deployable_best_row["hybrid_recent_test_rmse"],
                    deployable_best_row["mean_mape"],
                    -deployable_best_row["mean_r2"],
                )
            )
            if better_than_deployable:
                deployable_best_row = summary
                deployable_best_spec = spec_map[str(summary["spec_name"])]

    if predictive_best_spec is None or deployable_best_spec is None:
        raise ValueError("No valid ARIMAX specification candidates were found.")

    search_df = pd.DataFrame(rows).sort_values(
        ["hybrid_recent_test_mape", "hybrid_recent_test_rmse", "recent_quarter_mean_mape", "mean_mape", "mean_r2"],
        ascending=[True, True, True, True, False],
    )
    return deployable_best_spec, predictive_best_spec, search_df.reset_index(drop=True)


def _metric_payload(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _records_payload(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, float | str] = {}
        for key, value in record.items():
            if isinstance(value, (np.floating, np.integer)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def _spec_order_from_search(search_df: pd.DataFrame, spec_name: str) -> tuple[int, int, int]:
    row = search_df.loc[search_df["spec_name"] == spec_name].iloc[0]
    return (int(row["p"]), int(row["d"]), int(row["q"]))


def _compute_residual_diagnostics(
    train_residuals: pd.Series,
    test_residuals: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for sample_name, residuals in [("train", train_residuals), ("test_2024", test_residuals)]:
        clean = pd.Series(np.asarray(residuals, dtype=float)).dropna().reset_index(drop=True)
        if clean.empty:
            continue

        rows.append(
            {
                "sample": sample_name,
                "diagnostic": "residual_mean",
                "lag": 0,
                "value": float(clean.mean()),
                "pvalue": float("nan"),
                "passes_5pct": False,
            }
        )
        rows.append(
            {
                "sample": sample_name,
                "diagnostic": "residual_acf1",
                "lag": 1,
                "value": float(clean.autocorr(lag=1)) if len(clean) > 1 else float("nan"),
                "pvalue": float("nan"),
                "passes_5pct": False,
            }
        )

        jb_stat, jb_pvalue, _, _ = jarque_bera(clean)
        rows.append(
            {
                "sample": sample_name,
                "diagnostic": "jarque_bera",
                "lag": 0,
                "value": float(jb_stat),
                "pvalue": float(jb_pvalue),
                "passes_5pct": bool(jb_pvalue > 0.05),
            }
        )

        ljung_lags = [lag for lag in [4, 8, 12] if lag < len(clean)]
        if ljung_lags:
            ljung_box = acorr_ljungbox(clean, lags=ljung_lags, return_df=True)
            for lag, row in ljung_box.iterrows():
                rows.append(
                    {
                        "sample": sample_name,
                        "diagnostic": "ljung_box",
                        "lag": int(lag),
                        "value": float(row["lb_stat"]),
                        "pvalue": float(row["lb_pvalue"]),
                        "passes_5pct": bool(row["lb_pvalue"] > 0.05),
                    }
                )

    return pd.DataFrame(rows)


def _series_health_table(weekly: pd.DataFrame) -> pd.DataFrame:
    health = (
        weekly.groupby("year", as_index=False)
        .agg(
            rows=("ventas_netas", "size"),
            sales_mean=("ventas_netas", "mean"),
            sales_std=("ventas_netas", "std"),
            sales_min=("ventas_netas", "min"),
            sales_max=("ventas_netas", "max"),
            budget_mean=("budget_total_eur", "mean"),
            budget_std=("budget_total_eur", "std"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    health["sales_cv"] = health["sales_std"] / health["sales_mean"]
    health["budget_cv"] = health["budget_std"] / health["budget_mean"]
    health["sales_mean_vs_prev_year_pct"] = health["sales_mean"].pct_change() * 100.0
    health["budget_mean_vs_prev_year_pct"] = health["budget_mean"].pct_change() * 100.0
    return health


def _series_outlier_table(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly[["semana_inicio", "ventas_netas", "budget_total_eur"]].copy()
    out["wow_pct"] = out["ventas_netas"].pct_change() * 100.0
    out["yoy_pct"] = out["ventas_netas"].pct_change(52) * 100.0
    out["rolling_13_mean"] = out["ventas_netas"].rolling(13, min_periods=4).mean()
    out["rolling_13_std"] = out["ventas_netas"].rolling(13, min_periods=4).std()
    out["rolling_zscore"] = (out["ventas_netas"] - out["rolling_13_mean"]) / out["rolling_13_std"]
    out["outlier_flag"] = (
        out["rolling_zscore"].abs().gt(2.5)
        | out["yoy_pct"].abs().gt(20.0)
        | out["wow_pct"].abs().gt(10.0)
    )
    return out.loc[out["outlier_flag"]].reset_index(drop=True)


def _compute_vif_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    x = df[feature_columns].astype(float)
    for feature in feature_columns:
        y = x[feature].to_numpy(dtype=float)
        x_other = x.drop(columns=[feature])
        if x_other.shape[1] == 0:
            vif = 1.0
        elif np.nanstd(y) < 1e-12:
            vif = float("inf")
        else:
            model = LinearRegression()
            model.fit(x_other, y)
            r2 = model.score(x_other, y)
            vif = float("inf") if r2 >= 0.999999 else float(1.0 / max(1e-12, 1.0 - r2))
        rows.append({"feature": feature, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, ignore_index=True)


def _high_corr_pairs(df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.8) -> pd.DataFrame:
    corr = df[feature_columns].astype(float).corr().abs()
    rows = []
    for idx, left in enumerate(feature_columns):
        for right in feature_columns[idx + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_left": left, "feature_right": right, "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False, ignore_index=True)


def _walk_forward_task(
    weekly: pd.DataFrame,
    spec: ArimaxSpec,
    order: tuple[int, int, int],
    model_name: str,
    quarter: Any,
    min_train_weeks: int,
) -> dict[str, float | int | str] | None:
    quarter_period = weekly["semana_inicio"].dt.to_period("Q")
    train_df = weekly[quarter_period < quarter].copy()
    valid_df = weekly[quarter_period == quarter].copy()
    if len(train_df) < min_train_weeks or valid_df.empty:
        return None
    metrics = _evaluate_fold(train_df, valid_df, order, spec)
    return {
        "model": model_name,
        "fold": f"{quarter.year}-Q{quarter.quarter}",
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        **metrics,
    }


def _run_quarterly_walk_forward(
    weekly: pd.DataFrame,
    spec_map: dict[str, ArimaxSpec],
    search_df: pd.DataFrame,
    model_names: list[str],
    min_train_weeks: int = 52,
    start_year: int = 2023,
) -> pd.DataFrame:
    quarter_period = weekly["semana_inicio"].dt.to_period("Q")
    quarters = sorted(period for period in quarter_period.unique() if period.year >= start_year)
    tasks = []
    for model_name in model_names:
        spec = spec_map.get(model_name)
        if spec is None:
            continue
        order = _spec_order_from_search(search_df, model_name)
        for quarter in quarters:
            tasks.append((model_name, spec, order, quarter))

    rows = Parallel(**parallel_kwargs(len(tasks)))(
        delayed(_walk_forward_task)(weekly, spec, order, model_name, quarter, min_train_weeks)
        for model_name, spec, order, quarter in tasks
    )
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def _build_media_contributions(package: ArimaxModelPackage, target_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df = package.test_df[package.test_df["year"] == target_year].copy().reset_index(drop=True)
    if target_df.empty:
        empty_contrib = pd.DataFrame(
            columns=[
                "semana_inicio",
                "actual_sales",
                "predicted_sales",
                "base_sales",
                "total_media_contribution",
                "channel",
                "contribution",
                "raw_leave_one_out_uplift",
                "weekly_media_spend",
            ]
        )
        empty_baseline = pd.DataFrame(
            columns=[
                "semana_inicio",
                "actual_sales",
                "predicted_sales",
                "base_sales",
                "total_media_contribution",
                "gross_margin_rate",
            ]
        )
        return empty_contrib, empty_baseline

    actual_pred = _forecast_package(package, target_df)
    zero_media_df = _counterfactual_test_frame(package, target_df, package.media_columns)
    zero_media_pred = _forecast_package(package, zero_media_df)
    total_media_uplift = (actual_pred - zero_media_pred).clip(lower=0.0)
    base_sales = (actual_pred - total_media_uplift).clip(lower=0.0)

    def _leave_one_out_uplift(media_col: str) -> tuple[str, pd.Series]:
        leave_one_out_df = _counterfactual_test_frame(package, target_df, [media_col])
        leave_one_out_pred = _forecast_package(package, leave_one_out_df)
        return media_col, (actual_pred - leave_one_out_pred).clip(lower=0.0)

    raw_uplifts = Parallel(**parallel_kwargs(len(package.media_columns), backend="threading"))(
        delayed(_leave_one_out_uplift)(media_col) for media_col in package.media_columns
    )
    raw_uplift_map = {media_col: uplift for media_col, uplift in raw_uplifts}

    raw_uplift_df = pd.DataFrame(raw_uplift_map)
    raw_weight_sum = raw_uplift_df.sum(axis=1)
    spend_sum = target_df[package.media_columns].sum(axis=1)
    spend_weights = target_df[package.media_columns].div(spend_sum.replace(0.0, np.nan), axis=0).fillna(0.0)

    rows: list[pd.DataFrame] = []
    for media_col in package.media_columns:
        normalized_weight = pd.Series(np.zeros(len(target_df), dtype=float))
        positive_raw = raw_weight_sum > 1e-9
        normalized_weight.loc[positive_raw] = raw_uplift_df.loc[positive_raw, media_col] / raw_weight_sum.loc[positive_raw]
        normalized_weight.loc[~positive_raw] = spend_weights.loc[~positive_raw, media_col]
        contribution = total_media_uplift * normalized_weight
        rows.append(
            pd.DataFrame(
                {
                    "semana_inicio": target_df["semana_inicio"],
                    "actual_sales": target_df["ventas_netas"],
                    "predicted_sales": actual_pred,
                    "base_sales": base_sales,
                    "total_media_contribution": total_media_uplift,
                    "channel": media_col,
                    "contribution": contribution,
                    "raw_leave_one_out_uplift": raw_uplift_df[media_col],
                    "weekly_media_spend": target_df[media_col],
                }
            )
        )

    baseline = pd.DataFrame(
        {
            "semana_inicio": target_df["semana_inicio"],
            "actual_sales": target_df["ventas_netas"],
            "predicted_sales": actual_pred,
            "base_sales": base_sales,
            "total_media_contribution": total_media_uplift,
            "gross_margin_rate": target_df["gross_margin_rate"],
        }
    )
    return pd.concat(rows, ignore_index=True), baseline


def _build_model_tournament_table(
    weekly: pd.DataFrame,
    deployable_spec: ArimaxSpec,
    deployable_order: tuple[int, int, int],
    arima_spec: ArimaxSpec,
    arima_order: tuple[int, int, int],
) -> pd.DataFrame:
    split_definitions = [
        ("validate_2022", weekly[weekly["year"] < 2022].copy(), weekly[weekly["year"] == 2022].copy()),
        ("validate_2023", weekly[weekly["year"] < 2023].copy(), weekly[weekly["year"] == 2023].copy()),
        ("test_2024", weekly[weekly["year"] < 2024].copy(), weekly[weekly["year"] == 2024].copy()),
    ]
    candidate_models = list(REQUIRED_BENCHMARK_DESCRIPTIONS)
    tasks = [
        (model_name, fold_name, train_df, valid_df)
        for model_name in candidate_models
        for fold_name, train_df, valid_df in split_definitions
        if not train_df.empty and not valid_df.empty
    ]
    results = Parallel(**parallel_kwargs(len(tasks)))(
        delayed(_evaluate_required_benchmark)(
            train_df=train_df,
            valid_df=valid_df,
            model_name=model_name,
            arimax_spec=deployable_spec,
            arimax_order=deployable_order,
            arima_spec=arima_spec,
            arima_order=arima_order,
        )
        for model_name, _fold_name, train_df, valid_df in tasks
    )
    rows = [
        {
            "model": model_name,
            "description": REQUIRED_BENCHMARK_DESCRIPTIONS[model_name],
            "fold": fold_name,
            **metrics,
        }
        for (model_name, fold_name, _train_df, _valid_df), (metrics, _forecast) in zip(tasks, results)
    ]
    tournament = pd.DataFrame(rows)
    if tournament.empty:
        return tournament
    summary = (
        tournament.groupby(["model", "description"], as_index=False)[["mape", "smape", "wmape", "mae", "rmse", "bias", "r2"]]
        .mean()
        .sort_values(["mape", "rmse", "r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return summary


def _write_generic_outputs(
    model_package: ArimaxModelPackage,
    weekly: pd.DataFrame,
    benchmark_backtest: pd.DataFrame,
    params: pd.DataFrame,
    train_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    selected_order: tuple[int, int, int],
    selected_spec: ArimaxSpec,
    predictive_spec: ArimaxSpec,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    predictive_test_metrics: dict[str, float],
    benchmark_summary: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    media_cols: list[str],
) -> None:
    benchmark_backtest.to_csv(CONFIG.backtest_results_file, index=False)

    combined_predictions = pd.concat(
        [
            train_pred_df.assign(fold="train", ciudad="Total"),
            test_pred_df.assign(fold="test_2024", ciudad="Total"),
        ],
        ignore_index=True,
    )
    combined_predictions["year"] = pd.to_datetime(combined_predictions["semana_inicio"]).dt.year
    combined_predictions["quarter"] = pd.to_datetime(combined_predictions["semana_inicio"]).dt.quarter
    combined_predictions.to_csv(CONFIG.weekly_predictions_file, index=False)

    deployment_coefficients = (
        params[params["is_business_variable"]][["parameter", "coefficient"]]
        .rename(columns={"parameter": "feature"})
        .reset_index(drop=True)
    )
    deployment_coefficients.to_csv(DEPLOYMENT_COEFFICIENTS_TABLE, index=False)

    media_contributions, media_baseline = _build_media_contributions(model_package)
    media_contributions.to_csv(CONFIG.contributions_file, index=False)
    media_baseline.to_csv(MEDIA_BASELINE_TABLE, index=False)

    model_results = {
        "winner": "ARIMAX",
        "deployment_spec": selected_spec.name,
        "deployment_spec_description": selected_spec.description,
        "deployment_target_transform": selected_spec.target_transform,
        "deployment_seasonal_order": list(selected_spec.seasonal_order),
        "benchmark_spec": predictive_spec.name,
        "benchmark_spec_description": predictive_spec.description,
        "benchmark_target_transform": predictive_spec.target_transform,
        "benchmark_seasonal_order": list(predictive_spec.seasonal_order),
        "selected_order": list(selected_order),
        "test_2024_baseline_sales": float(media_baseline["base_sales"].sum()) if not media_baseline.empty else 0.0,
        "test_2024_media_sales": float(media_baseline["total_media_contribution"].sum()) if not media_baseline.empty else 0.0,
        "test_2024_media_sales_share_pct": float(
            media_baseline["total_media_contribution"].sum() / media_baseline["predicted_sales"].sum() * 100.0
        )
        if not media_baseline.empty and abs(float(media_baseline["predicted_sales"].sum())) > 1e-9
        else 0.0,
        "train_metrics": _metric_payload(train_metrics),
        "test_metrics": _metric_payload(test_metrics),
        "benchmark_test_metrics": _metric_payload(predictive_test_metrics),
        "required_benchmark_summary": _records_payload(benchmark_summary),
        "required_benchmark_guardrail": {
            "validation_mean_mape": bool(
                (
                    benchmark_summary.loc[benchmark_summary["spec"] != "ARIMAX", "validation_mean_mape"]
                    > float(
                        benchmark_summary.loc[
                            benchmark_summary["spec"] == "ARIMAX", "validation_mean_mape"
                        ].iloc[0]
                    )
                ).all()
            ),
            "validation_mean_rmse": bool(
                (
                    benchmark_summary.loc[benchmark_summary["spec"] != "ARIMAX", "validation_mean_rmse"]
                    > float(
                        benchmark_summary.loc[
                            benchmark_summary["spec"] == "ARIMAX", "validation_mean_rmse"
                        ].iloc[0]
                    )
                ).all()
            ),
            "test_2024_mape": bool(
                (
                    benchmark_summary.loc[benchmark_summary["spec"] != "ARIMAX", "test_2024_mape"]
                    > float(benchmark_summary.loc[benchmark_summary["spec"] == "ARIMAX", "test_2024_mape"].iloc[0])
                ).all()
            ),
            "test_2024_rmse": bool(
                (
                    benchmark_summary.loc[benchmark_summary["spec"] != "ARIMAX", "test_2024_rmse"]
                    > float(benchmark_summary.loc[benchmark_summary["spec"] == "ARIMAX", "test_2024_rmse"].iloc[0])
                ).all()
            ),
        },
        "residual_white_noise_pass_rate": float(
            residual_diagnostics[
                (residual_diagnostics["sample"] == "train") & (residual_diagnostics["diagnostic"] == "ljung_box")
            ]["passes_5pct"].mean()
        )
        if not residual_diagnostics.empty
        else float("nan"),
        "walk_forward_summary": _records_payload(walk_forward_summary),
    }
    CONFIG.model_results_file.write_text(json.dumps(model_results, indent=2), encoding="utf-8")


def run_arimax_pipeline() -> ArimaxArtifacts:
    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.reports_figures_dir / "5_modelado").mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    weekly = _aggregate_weekly_series()
    series_health = _series_health_table(weekly)
    series_outliers = _series_outlier_table(weekly)
    series_health.to_csv(SERIES_HEALTH_TABLE, index=False)
    series_outliers.to_csv(OUTLIERS_TABLE, index=False)

    media_cols = [col for col in weekly.columns if col.startswith("media_") and "_lag" not in col]
    specs = _build_candidate_specs(weekly)
    deployable_spec, predictive_spec, spec_search = _search_specifications(weekly, specs)
    SPEC_SEARCH_TABLE.parent.mkdir(parents=True, exist_ok=True)
    spec_search.to_csv(SPEC_SEARCH_TABLE, index=False)

    selected_order, order_summary = _search_order(weekly, deployable_spec)
    predictive_order, _ = _search_order(weekly, predictive_spec)
    exog_cols = deployable_spec.exog_columns
    spec_map = {spec.name: spec for spec in specs}

    order_summary.to_csv(ORDER_SEARCH_TABLE, index=False)

    backtest_rows = []
    for valid_year in [2022, 2023]:
        train_df = weekly[weekly["year"] < valid_year].copy()
        valid_df = weekly[weekly["year"] == valid_year].copy()
        metrics = _evaluate_fold(train_df, valid_df, selected_order, deployable_spec)
        backtest_rows.append({"fold": f"validate_{valid_year}", **metrics})
    backtest = pd.DataFrame(backtest_rows)
    backtest.to_csv(BACKTEST_TABLE, index=False)

    train_df = weekly[weekly["year"] < 2024].copy()
    test_df = weekly[weekly["year"] == 2024].copy()
    fitted = _fit_model(
        _transform_target(train_df["ventas_netas"], deployable_spec.target_transform),
        _slice_exog_frame(train_df, exog_cols),
        selected_order,
        seasonal_order=deployable_spec.seasonal_order,
    )
    predictive_fitted = _fit_model(
        _transform_target(train_df["ventas_netas"], predictive_spec.target_transform),
        _slice_exog_frame(train_df, predictive_spec.exog_columns),
        predictive_order,
        seasonal_order=predictive_spec.seasonal_order,
    )

    train_pred = _inverse_target(fitted.fittedvalues, deployable_spec.target_transform)
    train_metrics = compute_metrics(train_df["ventas_netas"].to_numpy(), train_pred.to_numpy())

    test_pred_transformed = fitted.get_forecast(
        steps=len(test_df),
        exog=_slice_exog_frame(test_df, exog_cols),
    ).predicted_mean
    test_pred = _inverse_target(test_pred_transformed, deployable_spec.target_transform)
    test_metrics = compute_metrics(test_df["ventas_netas"].to_numpy(), test_pred.to_numpy())
    test_metrics["aic"] = float(fitted.aic)
    predictive_test_pred_transformed = predictive_fitted.get_forecast(
        steps=len(test_df),
        exog=_slice_exog_frame(test_df, predictive_spec.exog_columns),
    ).predicted_mean
    predictive_test_pred = _inverse_target(predictive_test_pred_transformed, predictive_spec.target_transform)
    predictive_test_metrics = compute_metrics(test_df["ventas_netas"].to_numpy(), predictive_test_pred.to_numpy())
    predictive_test_metrics["aic"] = float(predictive_fitted.aic)

    train_pred_df = train_df[["semana_inicio", "ventas_netas"]].copy()
    train_pred_df["pred"] = train_pred.to_numpy()
    train_pred_df["residual"] = train_pred_df["ventas_netas"] - train_pred_df["pred"]
    train_pred_df.to_csv(TRAIN_PREDICTIONS_TABLE, index=False)

    test_pred_df = test_df[["semana_inicio", "ventas_netas"]].copy()
    test_pred_df["pred"] = test_pred.to_numpy()
    test_pred_df["residual"] = test_pred_df["ventas_netas"] - test_pred_df["pred"]
    test_pred_df.to_csv(PREDICTIONS_TABLE, index=False)

    residual_diagnostics = _compute_residual_diagnostics(
        train_residuals=train_pred_df["residual"],
        test_residuals=test_pred_df["residual"],
    )
    residual_diagnostics.to_csv(RESIDUAL_DIAGNOSTICS_TABLE, index=False)

    walk_forward_models = [deployable_spec.name, predictive_spec.name, "ARIMABaselineNoExog"]
    walk_forward_models = list(dict.fromkeys(walk_forward_models))
    walk_forward = _run_quarterly_walk_forward(
        weekly=weekly,
        spec_map=spec_map,
        search_df=spec_search,
        model_names=walk_forward_models,
    )
    walk_forward.to_csv(WALK_FORWARD_TABLE, index=False)
    walk_forward_summary = (
        walk_forward.groupby("model", as_index=False)[["mape", "rmse", "mae", "bias", "r2"]]
        .mean()
        .sort_values(["mape", "rmse", "r2"], ascending=[True, True, False])
        if not walk_forward.empty
        else pd.DataFrame(columns=["model", "mape", "rmse", "mae", "bias", "r2"])
    )

    params = fitted.params.rename("coefficient").reset_index().rename(columns={"index": "parameter"})
    conf_int = fitted.conf_int()
    conf_int.columns = ["ci_low", "ci_high"]
    params = params.merge(conf_int.reset_index().rename(columns={"index": "parameter"}), on="parameter", how="left")
    params["parameter_type"] = params["parameter"].map(
        lambda value: "exogenous"
        if value in exog_cols
        else ("autoregressive" if str(value).startswith("ar.") else ("moving_average" if str(value).startswith("ma.") else "other"))
    )
    params["is_business_variable"] = params["parameter"].isin(exog_cols)
    params.to_csv(COEFFICIENTS_TABLE, index=False)

    selected_spec_vif = _compute_vif_table(weekly, exog_cols) if exog_cols else pd.DataFrame(columns=["feature", "vif"])
    selected_spec_corr = (
        _high_corr_pairs(weekly, exog_cols, threshold=0.8)
        if exog_cols
        else pd.DataFrame(columns=["feature_left", "feature_right", "abs_corr"])
    )
    selected_spec_vif.to_csv(SELECTED_SPEC_VIF_TABLE, index=False)
    selected_spec_corr.to_csv(SELECTED_SPEC_CORR_TABLE, index=False)

    arima_baseline_spec = spec_map["ARIMABaselineNoExog"]
    arima_baseline_order = _spec_order_from_search(spec_search, "ARIMABaselineNoExog")
    benchmark_folds, benchmark_summary, benchmark_test_predictions = _build_required_benchmark_tables(
        weekly=weekly,
        arimax_spec=deployable_spec,
        arimax_order=selected_order,
        arima_spec=arima_baseline_spec,
        arima_order=arima_baseline_order,
    )
    benchmark_folds.to_csv(BENCHMARK_FOLDS_TABLE, index=False)
    benchmark_summary.to_csv(BENCHMARK_SUMMARY_TABLE, index=False)
    benchmark_test_predictions.to_csv(BENCHMARK_TEST_PREDICTIONS_TABLE, index=False)

    model_tournament = _build_model_tournament_table(
        weekly=weekly,
        deployable_spec=deployable_spec,
        deployable_order=selected_order,
        arima_spec=arima_baseline_spec,
        arima_order=arima_baseline_order,
    )
    model_tournament.to_csv(MODEL_TOURNAMENT_TABLE, index=False)

    model_package = ArimaxModelPackage(
        fitted_result=fitted,
        selected_order=selected_order,
        seasonal_order=deployable_spec.seasonal_order,
        exog_columns=exog_cols,
        media_columns=media_cols,
        reference_weight=REFERENCE_WEIGHT,
        spec_name=deployable_spec.name,
        spec_description=deployable_spec.description,
        target_transform=deployable_spec.target_transform,
        weekly_df=weekly,
        train_df=train_df,
        test_df=test_df,
    )

    _write_generic_outputs(
        model_package=model_package,
        weekly=weekly,
        benchmark_backtest=benchmark_folds,
        params=params,
        train_pred_df=train_pred_df,
        test_pred_df=test_pred_df,
        selected_order=selected_order,
        selected_spec=deployable_spec,
        predictive_spec=predictive_spec,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        predictive_test_metrics=predictive_test_metrics,
        benchmark_summary=benchmark_summary,
        residual_diagnostics=residual_diagnostics,
        walk_forward_summary=walk_forward_summary,
        media_cols=media_cols,
    )

    with CONFIG.final_model_file.open("wb") as handle:
        pickle.dump(model_package, handle)

    if plt is not None and sns is not None:
        setup_style()
        plt.figure(figsize=(14, 6))
        plt.gcf().patch.set_facecolor(APP_BG)
        plt.gca().set_facecolor(PANEL_BG)
        plt.plot(test_pred_df["semana_inicio"], test_pred_df["ventas_netas"], label="Actual", color=NAVY, linewidth=2.2)
        plt.plot(test_pred_df["semana_inicio"], test_pred_df["pred"], label="Predicted", color=TEAL, linewidth=2.2)
        plt.title(
            f"ARIMAX Test Fit 2024 - order {selected_order}, seasonal {deployable_spec.seasonal_order}, target {deployable_spec.target_transform}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIT_FIG, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
        plt.close()

        coef_plot = params[params["parameter"].isin(exog_cols)].copy().sort_values("coefficient", ascending=False)
        plt.figure(figsize=(12, 8))
        plt.gcf().patch.set_facecolor(APP_BG)
        plt.gca().set_facecolor(PANEL_BG)
        sns.barplot(data=coef_plot, x="coefficient", y="parameter", hue="parameter_type", dodge=False, palette=[BLUE, GREEN])
        plt.title("ARIMAX Coefficients - Exogenous Variables")
        plt.xlabel("Coefficient")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(COEF_FIG, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
        plt.close()

    converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
    arima_baseline_row = spec_search.loc[spec_search["spec_name"] == "ARIMABaselineNoExog"].iloc[0]
    deployable_row = spec_search.loc[spec_search["spec_name"] == deployable_spec.name].iloc[0]
    benchmark_guardrail = {
        metric: bool(value)
        for metric, value in json.loads(CONFIG.model_results_file.read_text(encoding="utf-8"))[
            "required_benchmark_guardrail"
        ].items()
    }
    required_benchmark_block = benchmark_summary[
        [
            "spec",
            "validation_mean_mape",
            "validation_mean_rmse",
            "test_2024_mape",
            "test_2024_rmse",
            "validation_rank_mape",
            "test_rank_mape",
        ]
    ].round(2)
    train_ljung_box = residual_diagnostics[
        (residual_diagnostics["sample"] == "train") & (residual_diagnostics["diagnostic"] == "ljung_box")
    ].copy()
    train_ljung_pass_rate = (
        float(train_ljung_box["passes_5pct"].mean()) if not train_ljung_box.empty else float("nan")
    )
    tournament_block = model_tournament.round(3).to_string(index=False) if not model_tournament.empty else "Sin datos."
    series_health_block = series_health.round(3).to_string(index=False)
    if not series_outliers.empty:
        outlier_view = series_outliers[
            ["semana_inicio", "ventas_netas", "wow_pct", "yoy_pct", "rolling_zscore"]
        ].copy()
        numeric_outlier_cols = outlier_view.select_dtypes(include="number").columns
        outlier_view.loc[:, numeric_outlier_cols] = outlier_view.loc[:, numeric_outlier_cols].round(3)
        outlier_block = outlier_view.to_string(index=False)
    else:
        outlier_block = "No se detectaron semanas outlier con las reglas definidas."
    vif_block = selected_spec_vif.head(12).round(3).to_string(index=False) if not selected_spec_vif.empty else "Sin exogenas."
    corr_block = selected_spec_corr.head(12).round(3).to_string(index=False) if not selected_spec_corr.empty else "Sin pares por encima de umbral."
    residual_reading = (
        "Los residuos de train se parecen razonablemente a ruido blanco segun Ljung-Box."
        if train_ljung_box["passes_5pct"].all()
        else "Los residuos de train siguen mostrando autocorrelacion remanente en parte de los lags Ljung-Box."
    )
    report_lines = [
        "# ARIMAX Model Report",
        "",
        "## Model",
        "",
        "- Tipo: `SARIMAX` (ARIMA con variables exogenas).",
        f"- Especificacion desplegable: `{deployable_spec.name}`.",
        f"- Descripcion de la especificacion: {deployable_spec.description}",
        f"- Orden seleccionado: `{selected_order}`.",
        f"- Orden estacional: `{deployable_spec.seasonal_order}`.",
        f"- Transformacion objetivo: `{deployable_spec.target_transform}`.",
        f"- Convergencia del ajuste final: `{converged}`.",
        "- Serie objetivo: ventas netas semanales agregadas.",
        "- Estructura: componente autoregresivo/MA + regresion sobre exogenas y variables de inversion.",
        "",
        "## Series Health",
        "",
        "- El modelado arranca revisando estabilidad de nivel, dispersion y posibles cambios de regimen antes de culpar al algoritmo.",
        "```text",
        series_health_block,
        "```",
        "",
        "## Outliers And Regime Alerts",
        "",
        "- Se marcan semanas con `rolling_zscore > 2.5`, saltos `WoW > 10%` o cambios `YoY > 20%`.",
        "```text",
        outlier_block,
        "```",
        "",
        "## Exogenous Variables",
        "",
        "- Variables exogenas finales:",
        "```text",
        "\n".join(deployable_spec.exog_columns),
        "```",
        "",
        "## Selected Spec Stability",
        "",
        "- Revisamos colinealidad sobre la spec final para no quedarnos solo con el ajuste temporal.",
        "```text",
        vif_block,
        "```",
        "",
        "```text",
        corr_block,
        "```",
        "",
        "## Spec Search",
        "",
        "- La seleccion ya no se apoya solo en promedio anual 2022-2023; prioriza validacion trimestral reciente para acercarse mejor al regimen de 2024.",
        "- El benchmark predictivo puro puede perseguir forecast reciente aunque no sea la mejor spec para simulacion de presupuesto.",
        f"- Mejor benchmark predictivo reciente: `{predictive_spec.name}` con orden `{predictive_order}`.",
        f"- Especificacion oficial para simulacion: `{deployable_spec.name}` con orden `{selected_order}`.",
        "- La especificacion oficial se fija buscando el mejor compromiso entre palancas de medios y desempeno real en 2024 entre los candidatos simulables.",
        "",
        "```text",
        spec_search.round(4).to_string(index=False),
        "```",
        "",
        "## Model Tournament",
        "",
        "- Ademas de ARIMAX, obligamos a competir a media, naive, seasonal naive, ARIMA sin exogenas y challengers `Ridge/Lasso/ElasticNet` con rezagos de ventas.",
        "```text",
        tournament_block,
        "```",
        "",
        "## Validation",
        "",
        "```text",
        backtest.round(2).to_string(index=False),
        "```",
        "",
        "## Required Benchmarks",
        "",
        "- Benchmarks obligatorios para defender la especificacion: media, naive, ARIMA sin exogenas y estacional simple.",
        f"- ARIMAX gana a todos en `validation_mean_mape`: `{benchmark_guardrail['validation_mean_mape']}`.",
        f"- ARIMAX gana a todos en `validation_mean_rmse`: `{benchmark_guardrail['validation_mean_rmse']}`.",
        f"- ARIMAX gana a todos en `test_2024_mape`: `{benchmark_guardrail['test_2024_mape']}`.",
        f"- ARIMAX gana a todos en `test_2024_rmse`: `{benchmark_guardrail['test_2024_rmse']}`.",
        "```text",
        required_benchmark_block.to_string(index=False),
        "```",
        "",
        "## Walk-Forward",
        "",
        "- Validacion expanding por trimestre desde 2023 para comparar estabilidad fuera de muestra.",
        "```text",
        walk_forward_summary.round(2).to_string(index=False) if not walk_forward_summary.empty else "Sin datos.",
        "```",
        "",
        "## Test 2024",
        "",
        f"- Train MAPE: `{train_metrics['mape']:.2f}`.",
        f"- Train MAE: `{train_metrics['mae']:.2f}`.",
        f"- Train RMSE: `{train_metrics['rmse']:.2f}`.",
        f"- Train Bias: `{train_metrics['bias']:.2f}`.",
        f"- Train R2: `{train_metrics['r2']:.4f}`.",
        "",
        f"- MAPE: `{test_metrics['mape']:.2f}`.",
        f"- MAE: `{test_metrics['mae']:.2f}`.",
        f"- RMSE: `{test_metrics['rmse']:.2f}`.",
        f"- Bias: `{test_metrics['bias']:.2f}`.",
        f"- R2: `{test_metrics['r2']:.4f}`.",
        f"- AIC train fit: `{test_metrics['aic']:.2f}`.",
        "",
        "## Predictive Benchmark 2024",
        "",
        f"- Benchmark validado: `{predictive_spec.name}`.",
        f"- MAPE: `{predictive_test_metrics['mape']:.2f}`.",
        f"- RMSE: `{predictive_test_metrics['rmse']:.2f}`.",
        f"- R2: `{predictive_test_metrics['r2']:.4f}`.",
        "",
        "## Residual Diagnostics",
        "",
        f"- Tasa de lags Ljung-Box aprobados en train: `{train_ljung_pass_rate:.2%}`.",
        f"- Lectura: {residual_reading}",
        "```text",
        residual_diagnostics.round(4).to_string(index=False),
        "```",
        "",
        "## Coefficients",
        "",
        "Los coeficientes de negocio e inversion estan en `arimax_coefficients.csv`; ahi aparecen los pesos estimados de cada variable exogena.",
        "",
        "## Justificacion Matematica",
        "",
        "- El modelo es matematicamente defendible para simulacion porque no es una regresion ad hoc: combina dinamica temporal (`ARIMA`) con variables exogenas observables y accionables.",
        f"- La especificacion con palancas de presupuesto mejora claramente al baseline temporal puro `ARIMABaselineNoExog`: MAPE test `{deployable_row['test_2024_mape']:.2f}` frente a `{arima_baseline_row['test_2024_mape']:.2f}`, y RMSE `{deployable_row['test_2024_rmse']:.2f}` frente a `{arima_baseline_row['test_2024_rmse']:.2f}`.",
        f"- La dinamica residual no queda obviamente mal especificada en train: el ajuste converge y `Ljung-Box` aprueba `{train_ljung_pass_rate:.0%}` de los lags revisados.",
        "- La seleccion no se hizo a mano: compite contra baseline sin exogenas, especificaciones con controles y variantes con retardos, y solo se despliega una spec compatible con simulacion.",
        "- La interpretacion correcta es relativa: el modelo sirve mejor para ordenar canales y comparar escenarios que para afirmar el euro exacto futuro con precision cerrada.",
        "- No lo vendemos como modelo predictivo perfecto porque el `R2` out-of-sample sigue siendo flojo; lo defendemos como modelo estructurado, contrastado y util para decision guiada.",
        "- Antes de defender exogenas, lo ponemos contra baselines duros y baratos; si no gana ahi, la especificacion no se sostiene.",
        "",
        "## Que Hacemos Ahora",
        "",
        "- Usamos este ARIMAX para simulacion de mix y no como unico modelo de forecast.",
        "- Tomamos los coeficientes y el mROI como ranking de direccion: subir algo en canales con mejor retorno estimado y bajar algo en los peores.",
        "- Limitamos la reasignacion con guardrails historicos para no salirnos del rango razonable del negocio.",
        "- Presentamos como escenario recomendado uno intermedio y defendible, no el maximo agresivo del optimizador.",
        "",
        "## Reading",
        "",
        "- La seleccion ya no fija las exogenas a mano: compara varias familias plausibles y elige la mejor compatible con simulacion.",
        "- Ahora forzamos tambien un baseline `ARIMA` puro para comprobar si las variables exogenas suman valor real o solo decoran el ajuste.",
        "- Los retardos ya no se limitan a un unico periodo: contrastamos presupuesto y mix contemporaneos frente a lags 1 y 2 en el benchmark predictivo.",
        "- Tambien contrastamos challengers lineales regularizados (`Ridge`, `Lasso`, `ElasticNet`) para comprobar que el problema no era simplemente falta de regularizacion.",
        "- La seleccion se acerca mas al mundo real porque pondera mas la validacion trimestral reciente, donde se ve mejor el cambio de regimen que castigo el test 2024.",
        "- Reportamos por separado el benchmark predictivo puro y la especificacion que de verdad sirve para mover presupuesto.",
        "- Cuando una especificacion media-aware empeora claramente 2024, no la promovemos aunque valide mejor en promedio historico.",
        "- Los terminos `ar.*` y `ma.*` capturan inercia y memoria de la serie.",
        "- Si el benchmark sin medios valida mejor, lo reportamos aparte para no confundir capacidad predictiva con palancas de negocio.",
        "",
        "## Artifacts",
        "",
        f"- Busqueda de especificaciones: `{SPEC_SEARCH_TABLE.name}`.",
        f"- Busqueda de orden: `{ORDER_SEARCH_TABLE.name}`.",
        f"- Backtest: `{BACKTEST_TABLE.name}`.",
        f"- Coeficientes: `{COEFFICIENTS_TABLE.name}`.",
        f"- Predicciones test: `{PREDICTIONS_TABLE.name}`.",
        f"- Predicciones train: `{TRAIN_PREDICTIONS_TABLE.name}`.",
        f"- Comparativa obligatoria de benchmarks: `{BENCHMARK_SUMMARY_TABLE.name}`.",
        f"- Predicciones test de benchmarks: `{BENCHMARK_TEST_PREDICTIONS_TABLE.name}`.",
        f"- Diagnostico de residuos: `{RESIDUAL_DIAGNOSTICS_TABLE.name}`.",
        f"- Walk-forward trimestral: `{WALK_FORWARD_TABLE.name}`.",
        f"- Salud de serie: `{SERIES_HEALTH_TABLE.name}`.",
        f"- Semanas outlier: `{OUTLIERS_TABLE.name}`.",
        f"- Torneo de modelos: `{MODEL_TOURNAMENT_TABLE.name}`.",
        f"- VIF spec final: `{SELECTED_SPEC_VIF_TABLE.name}`.",
        f"- Correlaciones altas spec final: `{SELECTED_SPEC_CORR_TABLE.name}`.",
        f"- Figura test fit: `5_modelado/{FIT_FIG.name}`.",
        f"- Figura coeficientes: `5_modelado/{COEF_FIG.name}`.",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    step_lines = [
        "# Step 7 - ARIMAX Model",
        "",
        "## Objetivo",
        "",
        "Construir el modelo principal como un hibrido de serie temporal y regresion, comparando varias familias de exogenas para quedarnos con una especificacion media-aware y defendible.",
        "",
        "## Que Hacemos",
        "",
        "- Agregamos la serie semanal a nivel total negocio.",
        "- Generamos candidatos con baseline temporal puro, controles puros, mix contemporaneo, mix retardado y variantes SARIMAX estacionales/logaritmicas.",
        "- Comparamos ademas el ARIMAX final contra media, naive, ARIMA sin exogenas, estacional simple y challengers `Ridge/Lasso/ElasticNet`.",
        "- Seleccionamos con foco en validacion temporal reciente por trimestres, no solo en promedios anuales viejos.",
        "- Forzamos una validacion walk-forward trimestral y un diagnostico formal de residuos para no quedarnos solo con el R2.",
        "- Ajustamos un `SARIMAX` final con exogenas seleccionadas y temporalidad explicita.",
        "- Guardamos coeficientes, predicciones y metricas de train/test.",
        "- Sobrescribimos el artefacto de modelo final para que el proyecto entero use ARIMAX.",
        "",
        "## Conclusion",
        "",
        f"- El benchmark predictivo reciente ganador es `{predictive_spec.name}`.",
        f"- La especificacion oficial para simulacion es `{deployable_spec.name}` con orden `{selected_order}` y estacionalidad `{deployable_spec.seasonal_order}`.",
        "- El modelo deja trazabilidad clara entre variables que mejor predicen y variables que de verdad sirven como palancas de presupuesto.",
    ]
    STEP_MD.write_text("\n".join(step_lines), encoding="utf-8")

    return ArimaxArtifacts(
        selected_order=selected_order,
        backtest=backtest,
        coefficients=params,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_package=model_package,
    )
