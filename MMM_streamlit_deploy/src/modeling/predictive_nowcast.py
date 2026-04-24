from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.data.weekly_aggregations import GLOBAL_PANEL_LABEL
from src.common.metrics import compute_metrics
from src.common.parallel import parallel_kwargs
from src.modeling.specs import CONTROL_COLUMNS, TRAFFIC_COLUMNS
from src.modeling.selection import naive_predictions
from src.modeling.trainer import ensure_prerequisites, load_dataset
from src.validation.backtesting import expanding_year_splits, panel_time_cv_indices


BASE_TEMPORAL_COLUMNS = [
    "trend_index",
    "week_sin",
    "week_cos",
    "rebajas_flag",
    "black_friday_flag",
    "navidad_flag",
    "semana_santa_flag",
    "vacaciones_escolares_flag",
]

LAG_BASE_COLUMNS = [
    "ventas_netas",
    "pedidos_total",
    "ticket_medio_neto",
    "visitas_tienda_sum",
    "sesiones_web_sum",
    "tasa_conversion_tienda_mean",
    "tasa_conversion_web_mean",
    "budget_total_eur",
]

ROLLING_BASE_COLUMNS = [
    "ventas_netas",
    "pedidos_total",
    "ticket_medio_neto",
    "visitas_tienda_sum",
    "sesiones_web_sum",
]

PARTIAL_WEEK_CUTOFFS = {
    "d3": 2,
    "d4": 3,
}


@dataclass
class PredictiveSpec:
    name: str
    model_type: str
    feature_columns: List[str]
    description: str


@dataclass
class PredictiveModelPackage:
    model: Pipeline
    feature_columns: List[str]
    spec_name: str
    description: str


def _lagged_group_rolling_mean(values: pd.Series, groups: pd.Series, window: int, min_periods: int) -> pd.Series:
    return values.groupby(groups).transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())


def add_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ciudad", "semana_inicio"]).reset_index(drop=True).copy()
    out["naive_pred"] = naive_predictions(out)
    grouped = out.groupby("ciudad", group_keys=False)

    for base_col in LAG_BASE_COLUMNS:
        if base_col not in out.columns:
            continue
        for lag in range(1, 5):
            out[f"{base_col}_lag_{lag}"] = grouped[base_col].shift(lag)

    for base_col in ROLLING_BASE_COLUMNS:
        if base_col not in out.columns:
            continue
        out[f"{base_col}_roll4"] = (
            grouped[base_col].shift(1).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    for suffix in PARTIAL_WEEK_CUTOFFS:
        share_source = out[f"pedidos_total_{suffix}"].replace(0.0, np.nan) / out["pedidos_total"].replace(0.0, np.nan)
        out[f"pedidos_partial_share_{suffix}_roll8"] = _lagged_group_rolling_mean(share_source, out["ciudad"], 8, 4)
        out[f"pedidos_partial_share_{suffix}_roll8"] = out[f"pedidos_partial_share_{suffix}_roll8"].clip(0.05, 0.98)
        out[f"pedidos_total_{suffix}_projected"] = (
            out[f"pedidos_total_{suffix}"] / out[f"pedidos_partial_share_{suffix}_roll8"].replace(0.0, np.nan)
        )

        visits_share = out[f"visitas_tienda_sum_{suffix}"].replace(0.0, np.nan) / out["visitas_tienda_sum"].replace(0.0, np.nan)
        out[f"visitas_partial_share_{suffix}_roll8"] = _lagged_group_rolling_mean(visits_share, out["ciudad"], 8, 4)
        out[f"visitas_partial_share_{suffix}_roll8"] = out[f"visitas_partial_share_{suffix}_roll8"].clip(0.05, 0.98)
        out[f"visitas_tienda_sum_{suffix}_projected"] = (
            out[f"visitas_tienda_sum_{suffix}"] / out[f"visitas_partial_share_{suffix}_roll8"].replace(0.0, np.nan)
        )

        sessions_share = out[f"sesiones_web_sum_{suffix}"].replace(0.0, np.nan) / out["sesiones_web_sum"].replace(0.0, np.nan)
        out[f"sesiones_partial_share_{suffix}_roll8"] = _lagged_group_rolling_mean(sessions_share, out["ciudad"], 8, 4)
        out[f"sesiones_partial_share_{suffix}_roll8"] = out[f"sesiones_partial_share_{suffix}_roll8"].clip(0.05, 0.98)
        out[f"sesiones_web_sum_{suffix}_projected"] = (
            out[f"sesiones_web_sum_{suffix}"] / out[f"sesiones_partial_share_{suffix}_roll8"].replace(0.0, np.nan)
        )

        out[f"pedidos_total_{suffix}_surprise"] = (
            out[f"pedidos_total_{suffix}"] / out["pedidos_total_roll4"].replace(0.0, np.nan)
        )
        out[f"ventas_proxy_{suffix}"] = out[f"pedidos_total_{suffix}_projected"] * out["ticket_medio_neto_roll4"]

    numeric_cols = [col for col in out.columns if col not in {"semana_inicio", "ciudad"}]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def city_columns(df: pd.DataFrame) -> List[str]:
    return sorted([column for column in df.columns if column.startswith("city_")])


def week_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series) - pd.to_timedelta(pd.to_datetime(series).dt.weekday, unit="D")


def build_partial_week_features() -> pd.DataFrame:
    orders = pd.read_csv(CONFIG.orders_file, usecols=["fecha_pedido"], parse_dates=["fecha_pedido"])
    traffic = pd.read_csv(
        CONFIG.traffic_file,
        usecols=["fecha", "visitas_tienda", "sesiones_web"],
        parse_dates=["fecha"],
    )

    partial_frames: list[pd.DataFrame] = []
    for suffix, cutoff in PARTIAL_WEEK_CUTOFFS.items():
        order_slice = orders[orders["fecha_pedido"].dt.weekday <= cutoff].copy()
        order_slice["semana_inicio"] = week_start(order_slice["fecha_pedido"])
        order_weekly = (
            order_slice.groupby(["semana_inicio"], as_index=False)
            .agg(**{f"pedidos_total_{suffix}": ("fecha_pedido", "size")})
        )
        order_weekly["ciudad"] = GLOBAL_PANEL_LABEL

        traffic_slice = traffic[traffic["fecha"].dt.weekday <= cutoff].copy()
        traffic_slice["semana_inicio"] = week_start(traffic_slice["fecha"])
        traffic_weekly = (
            traffic_slice.groupby(["semana_inicio"], as_index=False)
            .agg(
                **{
                    f"visitas_tienda_sum_{suffix}": ("visitas_tienda", "sum"),
                    f"sesiones_web_sum_{suffix}": ("sesiones_web", "sum"),
                }
            )
        )
        traffic_weekly["ciudad"] = GLOBAL_PANEL_LABEL
        partial_frames.append(order_weekly.merge(traffic_weekly, on=["semana_inicio", "ciudad"], how="outer"))

    merged = partial_frames[0]
    for frame in partial_frames[1:]:
        merged = merged.merge(frame, on=["semana_inicio", "ciudad"], how="outer")
    return merged.fillna(0.0)


def load_predictive_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.diagnostic_dataset_file)
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"])
    partial_features = build_partial_week_features()
    df = df.merge(partial_features, on=["semana_inicio", "ciudad"], how="left")
    partial_columns = [column for column in partial_features.columns if column not in {"semana_inicio", "ciudad"}]
    df[partial_columns] = df[partial_columns].fillna(0.0)
    city_dummies = pd.get_dummies(df["ciudad"], prefix="city", drop_first=True, dtype=float)
    df = pd.concat([df, city_dummies], axis=1)
    active_cities = set(load_dataset()["ciudad"].unique())
    df = df[df["ciudad"].isin(active_cities)].copy()
    return df.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)


def build_specs(df: pd.DataFrame) -> List[PredictiveSpec]:
    city_cols = city_columns(df)
    media_cols = sorted([column for column in df.columns if column.startswith("media_")])
    share_cols = sorted([column for column in df.columns if column.startswith("budget_share_pct_")])
    extra_control_cols = [column for column in CONTROL_COLUMNS if column not in BASE_TEMPORAL_COLUMNS]

    historical_features = (
        BASE_TEMPORAL_COLUMNS
        + [f"ventas_netas_lag_{lag}" for lag in range(1, 5)]
        + ["ventas_netas_roll4"]
        + city_cols
    )
    historical_controls_features = historical_features + extra_control_cols

    lagged_operational_features = (
        historical_controls_features
        + [f"pedidos_total_lag_{lag}" for lag in range(1, 5)]
        + [f"ticket_medio_neto_lag_{lag}" for lag in range(1, 5)]
        + ["pedidos_total_roll4", "ticket_medio_neto_roll4"]
    )

    traffic_nowcast_features = (
        lagged_operational_features
        + TRAFFIC_COLUMNS
        + [f"visitas_tienda_sum_lag_{lag}" for lag in range(1, 5)]
        + [f"sesiones_web_sum_lag_{lag}" for lag in range(1, 5)]
        + [f"tasa_conversion_tienda_mean_lag_{lag}" for lag in range(1, 5)]
        + [f"tasa_conversion_web_mean_lag_{lag}" for lag in range(1, 5)]
        + ["visitas_tienda_sum_roll4", "sesiones_web_sum_roll4"]
    )

    safe_partial_d3_features = (
        lagged_operational_features
        + ["pedidos_total_d3", "visitas_tienda_sum_d3", "sesiones_web_sum_d3"]
        + media_cols
        + share_cols
        + [f"budget_total_eur_lag_{lag}" for lag in range(1, 5)]
    )
    safe_partial_d4_features = (
        lagged_operational_features
        + ["pedidos_total_d4", "visitas_tienda_sum_d4", "sesiones_web_sum_d4"]
        + media_cols
        + share_cols
        + [f"budget_total_eur_lag_{lag}" for lag in range(1, 5)]
    )
    safe_partial_d3_minimal_features = (
        historical_controls_features
        + [f"pedidos_total_lag_{lag}" for lag in range(1, 5)]
        + ["pedidos_total_roll4", "pedidos_total_d3", "visitas_tienda_sum_d3", "sesiones_web_sum_d3"]
    )
    safe_partial_d4_minimal_features = (
        lagged_operational_features
        + ["pedidos_total_d4", "visitas_tienda_sum_d4", "sesiones_web_sum_d4"]
    )
    projected_partial_d4_features = (
        lagged_operational_features
        + [
            "pedidos_total_d4",
            "pedidos_total_d4_projected",
            "pedidos_total_d4_surprise",
            "visitas_tienda_sum_d4_projected",
            "sesiones_web_sum_d4_projected",
            "ventas_proxy_d4",
        ]
        + media_cols
        + share_cols
        + [f"budget_total_eur_lag_{lag}" for lag in range(1, 5)]
    )
    projected_partial_d4_minimal_features = (
        historical_controls_features
        + [f"pedidos_total_lag_{lag}" for lag in range(1, 5)]
        + [f"ticket_medio_neto_lag_{lag}" for lag in range(1, 5)]
        + [
            "pedidos_total_roll4",
            "ticket_medio_neto_roll4",
            "pedidos_total_d4",
            "pedidos_total_d4_projected",
            "pedidos_total_d4_surprise",
            "ventas_proxy_d4",
        ]
    )

    return [
        PredictiveSpec(
            name="HistoricalLagElasticNet",
            model_type="elasticnet",
            feature_columns=historical_features,
            description="Benchmark historico limpio con lags de ventas y estacionalidad, sin variables downstream.",
        ),
        PredictiveSpec(
            name="LaggedOperationalRidge",
            model_type="ridge",
            feature_columns=lagged_operational_features,
            description="Nowcast conservador con pedidos y ticket solo en rezagos, sin variables contemporaneas.",
        ),
        PredictiveSpec(
            name="PartialWeekD3Ridge",
            model_type="ridge",
            feature_columns=safe_partial_d3_features,
            description="Nowcast de media semana con pedidos y trafico hasta el miercoles, mas rezagos y medios.",
        ),
        PredictiveSpec(
            name="PartialWeekD4Ridge",
            model_type="ridge",
            feature_columns=safe_partial_d4_features,
            description="Nowcast de media semana larga con pedidos y trafico hasta el jueves, mas rezagos y medios.",
        ),
        PredictiveSpec(
            name="PartialWeekD3MinimalRidge",
            model_type="ridge",
            feature_columns=safe_partial_d3_minimal_features,
            description="Nowcast parsimonioso hasta el miercoles con pedidos parciales, trafico parcial e historia.",
        ),
        PredictiveSpec(
            name="PartialWeekD4MinimalRidge",
            model_type="ridge",
            feature_columns=safe_partial_d4_minimal_features,
            description="Nowcast parsimonioso hasta el jueves con pedidos parciales, trafico parcial e historia.",
        ),
        PredictiveSpec(
            name="ProjectedPartialWeekD4Ridge",
            model_type="ridge",
            feature_columns=projected_partial_d4_features,
            description="Nowcast hasta el jueves usando proyeccion de semana completa desde pedidos y trafico parciales.",
        ),
        PredictiveSpec(
            name="ProjectedPartialWeekD4MinimalRidge",
            model_type="ridge",
            feature_columns=projected_partial_d4_minimal_features,
            description="Nowcast parsimonioso hasta el jueves con proyeccion de pedidos y ticket historico.",
        ),
    ]


def _make_regressor(model_type: str, cv_indices: List[tuple[np.ndarray, np.ndarray]]):
    if model_type == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 40), cv=cv_indices)
    if model_type == "lasso":
        return LassoCV(alphas=np.logspace(-4, 1, 30), cv=cv_indices, random_state=42, max_iter=200000)
    if model_type == "elasticnet":
        return ElasticNetCV(
            alphas=np.logspace(-4, 1, 30),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=cv_indices,
            random_state=42,
            max_iter=200000,
            selection="cyclic",
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_predictive_model(df_train: pd.DataFrame, spec: PredictiveSpec) -> Pipeline:
    cv_indices = panel_time_cv_indices(df_train[["semana_inicio"]].copy(), n_splits=3)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", _make_regressor(spec.model_type, cv_indices)),
        ]
    )
    pipeline.fit(df_train[spec.feature_columns], df_train["ventas_netas"])
    return pipeline


def evaluate_spec(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    spec: PredictiveSpec,
) -> tuple[Dict[str, float], pd.DataFrame]:
    train_work = train_df[spec.feature_columns + ["semana_inicio", "ciudad", "ventas_netas"]].dropna().copy()
    valid_work = valid_df[spec.feature_columns + ["semana_inicio", "ciudad", "ventas_netas"]].dropna().copy()

    model = fit_predictive_model(train_work, spec)
    scored = valid_work[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    scored["pred"] = model.predict(valid_work[spec.feature_columns])
    metrics = compute_metrics(scored["ventas_netas"], scored["pred"])
    metrics["rows"] = int(len(scored))

    reg = model.named_steps["reg"]
    metrics["alpha"] = float(getattr(reg, "alpha_", np.nan))
    if hasattr(reg, "l1_ratio_"):
        metrics["l1_ratio"] = float(reg.l1_ratio_)
    return metrics, scored


def evaluate_naive(valid_df: pd.DataFrame) -> tuple[Dict[str, float], pd.DataFrame]:
    scored = valid_df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    scored["pred"] = valid_df["naive_pred"].to_numpy()
    metrics = compute_metrics(scored["ventas_netas"], scored["pred"])
    metrics["rows"] = int(len(scored))
    metrics["alpha"] = float("nan")
    return metrics, scored


def _evaluate_spec_task(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    spec: PredictiveSpec,
    fold_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    metrics, scored = evaluate_spec(train_df, valid_df, spec)
    metrics["fold"] = fold_name
    metrics["spec"] = spec.name
    return metrics, scored.assign(fold=fold_name, spec=spec.name)


def choose_winner(backtest_results: pd.DataFrame) -> str:
    modeled = backtest_results[backtest_results["spec"] != "NaiveRolling4"].copy()
    ranking = (
        modeled.groupby("spec")
        .agg(mean_mape=("mape", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"))
        .reset_index()
        .sort_values(["mean_mape", "mean_rmse", "mean_r2"], ascending=[True, True, False])
    )
    return str(ranking.iloc[0]["spec"])


def fit_full_package(df_train: pd.DataFrame, spec: PredictiveSpec) -> PredictiveModelPackage:
    train_work = df_train[spec.feature_columns + ["semana_inicio", "ventas_netas"]].dropna().copy()
    model = fit_predictive_model(train_work, spec)
    return PredictiveModelPackage(
        model=model,
        feature_columns=spec.feature_columns,
        spec_name=spec.name,
        description=spec.description,
    )


def coefficient_table(package: PredictiveModelPackage) -> pd.DataFrame:
    scaler = package.model.named_steps["scaler"]
    reg = package.model.named_steps["reg"]
    raw_coef = np.asarray(reg.coef_, dtype=float) / scaler.scale_
    coef = pd.Series(raw_coef, index=package.feature_columns, dtype=float)
    table = pd.DataFrame({"feature": coef.index, "coefficient": coef.values})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_test_fit(scored: pd.DataFrame) -> None:
    weekly = scored.groupby("semana_inicio", as_index=False)[["ventas_netas", "pred"]].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(weekly["semana_inicio"], weekly["ventas_netas"], label="Actual")
    plt.plot(weekly["semana_inicio"], weekly["pred"], label="Predicted")
    plt.title("Predictive Test 2024: Actual vs Predicted Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG.reports_figures_dir / "predictive_test_actual_vs_predicted.png", dpi=160)
    plt.close()


def _dataframe_code_block(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def _safe_metric_payload(metrics: Dict[str, float]) -> Dict[str, float | None]:
    payload: Dict[str, float | None] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)) and np.isnan(value):
            payload[key] = None
        elif isinstance(value, (np.floating, float)):
            payload[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            payload[key] = int(value)
        else:
            payload[key] = value
    return payload


def _winner_usage_note(winner_spec: PredictiveSpec) -> str:
    description = winner_spec.description.lower()
    if "pedidos de la semana actual" in description or "pedidos actuales" in description:
        return (
            "El ganador usa `pedidos_total` contemporaneo como senal operativa de nowcast; "
            "se mantiene separado del MMM causal y no debe leerse como atribucion de medios."
        )
    if "ticket medio de la semana actual" in description:
        return (
            "El ganador usa `ticket_medio_neto` contemporaneo como senal operativa de nowcast; "
            "se mantiene separado del MMM causal y no debe leerse como atribucion de medios."
        )
    return "Se evita mezclar este benchmark predictivo con el MMM causal de medios."


def write_validation_markdown(
    backtest_results: pd.DataFrame,
    test_results: pd.DataFrame,
    winner: str,
    winner_spec: PredictiveSpec,
) -> None:
    winner_backtest = backtest_results[backtest_results["spec"] == winner].copy()
    winner_mean_r2 = float(winner_backtest["r2"].mean()) if not winner_backtest.empty else float("nan")
    winner_mean_mape = float(winner_backtest["mape"].mean()) if not winner_backtest.empty else float("nan")
    lines = [
        "# Predictive Validation Results",
        "",
        f"- Modelo predictivo ganador: `{winner}`",
        "- Split temporal: train con anios antiguos, validacion con bloques posteriores y test final en 2024.",
        f"- R2 medio en backtest del ganador: `{winner_mean_r2:.4f}`.",
        f"- MAPE medio en backtest del ganador: `{winner_mean_mape:.4f}`.",
        f"- {_winner_usage_note(winner_spec)}",
        "",
        "## Backtesting En Train/Validation",
        "",
        _dataframe_code_block(backtest_results.round(4)),
        "",
        "## Test 2024",
        "",
        _dataframe_code_block(test_results.round(4)),
        "",
        "## Lectura",
        "",
        "- `NaiveRolling4` sigue siendo un baseline duro con solo ventas pasadas.",
        "- `HistoricalLagElasticNet` muestra un benchmark historico limpio sin variables demasiado cercanas a la venta.",
        "- Este artefacto queda como contraste metodologico y no como sustituto del MMM explicativo.",
    ]
    CONFIG.predictive_validation_results_md.write_text("\n".join(lines), encoding="utf-8")


def write_model_report(
    winner: str,
    winner_spec: PredictiveSpec,
    backtest_results: pd.DataFrame,
    test_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    historical_metrics: Dict[str, float],
    coefficients: pd.DataFrame,
) -> None:
    top_coef = coefficients.head(10).copy()
    winner_backtest = backtest_results[backtest_results["spec"] == winner].copy()
    winner_mean_r2 = float(winner_backtest["r2"].mean()) if not winner_backtest.empty else float("nan")
    winner_mean_mape = float(winner_backtest["mape"].mean()) if not winner_backtest.empty else float("nan")
    lines = [
        "# Predictive Model Report",
        "",
        f"- Modelo ganador: `{winner}`",
        f"- Descripcion: {winner_spec.description}",
        "- Objetivo: predecir `ventas_netas` semanales agregadas con validacion temporal estricta.",
        f"- Naturaleza del modelo: benchmark/nowcast predictivo. {_winner_usage_note(winner_spec)}",
        "",
        "## Backtesting Medio Del Ganador",
        "",
        f"- R2 medio validacion: `{winner_mean_r2:.4f}`",
        f"- MAPE medio validacion: `{winner_mean_mape:.4f}`",
        "",
        "## Resultado Test 2024",
        "",
        f"- MAPE ganador: `{test_metrics['mape']:.4f}`",
        f"- MAE ganador: `{test_metrics['mae']:.2f}`",
        f"- RMSE ganador: `{test_metrics['rmse']:.2f}`",
        f"- Bias ganador: `{test_metrics['bias']:.2f}`",
        f"- R2 ganador: `{test_metrics['r2']:.6f}`",
        "",
        "## Comparativa Clave",
        "",
        f"- Baseline `NaiveRolling4`: MAPE `{baseline_metrics['mape']:.4f}`, R2 `{baseline_metrics['r2']:.6f}`.",
        f"- Forecast historico estricto: MAPE `{historical_metrics['mape']:.4f}`, R2 `{historical_metrics['r2']:.6f}`.",
        f"- Benchmark limpio ganador: MAPE `{test_metrics['mape']:.4f}`, R2 `{test_metrics['r2']:.6f}`.",
        "",
        "## Por Que Mejora",
        "",
        f"- {winner_spec.description}",
        f"- {_winner_usage_note(winner_spec)}",
        "- Su papel es operativo o metodologico segun la especificacion ganadora; no sustituye la lectura causal del MMM.",
        "",
        "## Coeficientes Mas Influyentes",
        "",
        _dataframe_code_block(top_coef.round(6)),
    ]
    CONFIG.predictive_model_report_md.write_text("\n".join(lines), encoding="utf-8")


def run_predictive_nowcast_pipeline() -> dict:
    ensure_prerequisites()
    df = add_predictive_features(load_predictive_dataset())
    specs = build_specs(df)
    spec_map = {spec.name: spec for spec in specs}

    train_pool = df[df["year"] < 2024].copy()
    test_df = df[df["year"] == 2024].copy()

    backtest_rows = []
    prediction_frames = []
    for train_mask, valid_mask, fold_name in expanding_year_splits(train_pool, min_train_year=2021):
        fold_train = train_pool.loc[train_mask].copy()
        fold_valid = train_pool.loc[valid_mask].copy()

        naive_metrics, naive_scored = evaluate_naive(fold_valid)
        naive_metrics["fold"] = fold_name
        naive_metrics["spec"] = "NaiveRolling4"
        backtest_rows.append(naive_metrics)
        prediction_frames.append(naive_scored.assign(fold=fold_name, spec="NaiveRolling4"))

        fold_results = Parallel(**parallel_kwargs(len(specs)))(
            delayed(_evaluate_spec_task)(fold_train, fold_valid, spec, fold_name) for spec in specs
        )
        backtest_rows.extend(metrics for metrics, _ in fold_results)
        prediction_frames.extend(scored for _, scored in fold_results)

    backtest_results = pd.DataFrame(backtest_rows)
    winner = choose_winner(backtest_results)
    winner_spec = spec_map[winner]

    baseline_metrics, baseline_scored = evaluate_naive(test_df)
    baseline_scored = baseline_scored.assign(fold="test_2024", spec="NaiveRolling4")

    test_rows = []
    test_predictions = [baseline_scored]
    test_metrics_map: Dict[str, Dict[str, float]] = {"NaiveRolling4": baseline_metrics}
    test_results_parallel = Parallel(**parallel_kwargs(len(specs)))(
        delayed(_evaluate_spec_task)(train_pool, test_df, spec, "test_2024") for spec in specs
    )
    for metrics, scored in test_results_parallel:
        test_rows.append(metrics)
        test_predictions.append(scored)
        test_metrics_map[str(metrics["spec"])] = metrics

    winner_test_metrics = test_metrics_map[winner]
    historical_metrics = test_metrics_map["HistoricalLagElasticNet"]
    test_results = pd.DataFrame([{"spec": "NaiveRolling4", "fold": "test_2024", **baseline_metrics}] + test_rows)

    package = fit_full_package(train_pool, winner_spec)
    winner_test_scored = next(frame for frame in test_predictions if frame["spec"].iat[0] == winner)
    coef_table = coefficient_table(package)

    predictions = pd.concat(prediction_frames + test_predictions, ignore_index=True)

    backtest_results.to_csv(CONFIG.predictive_backtest_results_file, index=False)
    predictions.to_csv(CONFIG.predictive_weekly_predictions_file, index=False)
    coef_table.to_csv(CONFIG.reports_tables_dir / "predictive_coefficients.csv", index=False)
    plot_test_fit(winner_test_scored)
    write_validation_markdown(backtest_results, test_results, winner, winner_spec)
    write_model_report(winner, winner_spec, backtest_results, winner_test_metrics, baseline_metrics, historical_metrics, coef_table)

    model_results = {
        "winner": winner,
        "winner_description": winner_spec.description,
        "baseline_test_metrics": _safe_metric_payload(baseline_metrics),
        "historical_test_metrics": _safe_metric_payload(historical_metrics),
        "winner_test_metrics": _safe_metric_payload(winner_test_metrics),
    }
    CONFIG.predictive_model_results_file.write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    with CONFIG.predictive_final_model_file.open("wb") as handle:
        pickle.dump(package, handle)

    return {
        "winner": winner,
        "backtest_results": backtest_results,
        "test_results": test_results,
        "winner_test_metrics": winner_test_metrics,
    }
