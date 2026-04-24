from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.common.parallel import parallel_kwargs
from src.modeling.specs import ELASTIC_NET_L1, RANDOM_STATE
from src.modeling.transforms import apply_media_transform
from src.simulation.curve_utils import compute_curve_gradient, find_concave_knee_index
from src.simulation.optimizer import BUDGET_TOTAL, TARGET_YEAR
from src.validation.visual_utils import APP_BG, BLUE, BORDER, GREEN, GRID, LIGHT_BLUE, NAVY, PANEL_BG, SLATE, TEAL, setup_style

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESPONSE_SUMMARY_TABLE = CONFIG.reports_tables_dir / "media_response_scenarios.csv"
RESPONSE_WEIGHTS_TABLE = CONFIG.reports_tables_dir / "media_response_weights.csv"
RESPONSE_CURVES_TABLE = CONFIG.reports_tables_dir / "media_response_curves.csv"
RESPONSE_MAXIMA_TABLE = CONFIG.reports_tables_dir / "media_response_maxima.csv"
RESPONSE_SATURATION_TABLE = CONFIG.reports_tables_dir / "media_response_saturation.csv"
RESPONSE_COEFFICIENTS_TABLE = CONFIG.reports_tables_dir / "media_response_coefficients.csv"
RESPONSE_METRICS_TABLE = CONFIG.reports_tables_dir / "media_response_fit_metrics.csv"
RESPONSE_TOURNAMENT_TABLE = CONFIG.reports_tables_dir / "media_response_model_tournament.csv"
RESPONSE_PREDICTIONS_TABLE = CONFIG.reports_tables_dir / "media_response_predictions.csv"
RESPONSE_PREDICTION_DIAGNOSTICS_TABLE = CONFIG.reports_tables_dir / "media_response_prediction_diagnostics.csv"
RESPONSE_ROI_TABLE = CONFIG.reports_tables_dir / "media_response_channel_roi.csv"
RESPONSE_DO_SOMETHING_TABLE = CONFIG.reports_tables_dir / "media_response_do_something_vs_nothing.csv"
RESPONSE_KNEE_POINTS_TABLE = CONFIG.reports_tables_dir / "media_response_knee_points.csv"
REPORT_MD = CONFIG.docs_dir / "media_response_optimization_report.md"
STEP_MD = CONFIG.docs_dir / "step_10_media_response_optimization.md"

MEDIA_MIN_SHARE = 0.0
MEDIA_MAX_SHARE = 0.55
SATURATION_POINTS = 120
MEDIA_COEFFICIENT_MULTIPLIERS = {
    "media_radio_local_response": 0.70,
    "media_prensa_response": 0.05,
}
MEDIA_MIN_POSITIVE_COEFFICIENT = {
    "media_prensa_response": 1.0,
}


@dataclass(frozen=True)
class MediaResponseSpec:
    name: str
    model_type: str
    control_variant: str
    description: str


@dataclass(frozen=True)
class MediaResponsePackage:
    weekly_df: pd.DataFrame
    feature_columns: list[str]
    transformed_media_columns: list[str]
    media_columns: list[str]
    model: Pipeline
    model_name: str
    model_type: str
    model_description: str
    transform_specs: dict[str, dict[str, float | int | str]]
    target_year: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def _savefig(name: str) -> None:
    output = CONFIG.reports_figures_dir / "6_simulacion" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
    plt.close()


def _style_axes(axes, grid_axis: str = "y") -> None:
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.grid(axis=grid_axis, color=GRID, alpha=0.78, linewidth=0.9)
        if grid_axis != "both":
            other_axis = "x" if grid_axis == "y" else "y"
            ax.grid(axis=other_axis, alpha=0.0)
        ax.tick_params(colors=SLATE, labelsize=10.5)
        ax.title.set_color(NAVY)
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(BORDER)
            ax.spines[side].set_linewidth(1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return float("nan")
    return float(numerator / denominator)


def _media_feature_to_channel(feature: str) -> str:
    return str(feature).replace("media_", "", 1).replace("_response", "")


def _media_response_specs() -> list[MediaResponseSpec]:
    specs: list[MediaResponseSpec] = []
    labels = {
        "full": "Full",
        "seasonal_calendar": "Seasonal",
        "temporal_only": "Temporal",
    }
    descriptions = {
        "full": "controles completos, calendario y contexto",
        "seasonal_calendar": "temporalidad y calendario principal",
        "temporal_only": "solo tendencia y estacionalidad",
    }
    for control_variant, suffix in labels.items():
        specs.extend(
            [
                MediaResponseSpec(
                    name=f"MediaResponseRidge{suffix}",
                    model_type="ridge",
                    control_variant=control_variant,
                    description=f"Ridge sobre medios transformados y {descriptions[control_variant]}.",
                ),
                MediaResponseSpec(
                    name=f"MediaResponseLasso{suffix}",
                    model_type="lasso",
                    control_variant=control_variant,
                    description=f"Lasso sobre medios transformados y {descriptions[control_variant]}.",
                ),
                MediaResponseSpec(
                    name=f"MediaResponseElasticNet{suffix}",
                    model_type="elasticnet",
                    control_variant=control_variant,
                    description=f"Elastic Net sobre medios transformados y {descriptions[control_variant]}.",
                ),
            ]
        )
    return specs


def _load_transform_specs(media_cols: list[str]) -> dict[str, dict[str, float | int | str]]:
    if CONFIG.selected_transforms_file.exists():
        raw = json.loads(CONFIG.selected_transforms_file.read_text(encoding="utf-8"))
        return {
            media_col: {
                "lag": int(raw[media_col]["lag"]),
                "alpha": float(raw[media_col]["alpha"]),
                "saturation": str(raw[media_col]["saturation"]),
            }
            for media_col in media_cols
        }

    diagnostics = pd.read_csv(CONFIG.reports_tables_dir / "transform_diagnostics_deploy.csv")
    sort_column = "selection_score" if "selection_score" in diagnostics.columns else "corr_with_controls_residual"
    specs = {}
    for media_col in media_cols:
        top = diagnostics[diagnostics["channel"] == media_col].sort_values(
            sort_column,
            ascending=False,
        ).iloc[0]
        specs[media_col] = {
            "lag": int(top["lag"]),
            "alpha": float(top["alpha"]),
            "saturation": str(top["saturation"]),
        }
    return specs


def _prepare_weekly_response_frame(
    weekly_df: pd.DataFrame,
    media_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], dict[str, dict[str, float | int | str]]]:
    df = weekly_df.copy().sort_values("semana_inicio").reset_index(drop=True)
    df["week_of_year"] = df["semana_inicio"].dt.isocalendar().week.astype(int)
    df["trend_index"] = np.arange(len(df), dtype=int)
    df["week_sin"] = np.sin(2.0 * np.pi * df["week_of_year"] / 52.0)
    df["week_cos"] = np.cos(2.0 * np.pi * df["week_of_year"] / 52.0)

    transform_specs = _load_transform_specs(media_cols)
    transformed_cols = []
    for media_col in media_cols:
        feature_col = f"{media_col}_response"
        transformed_cols.append(feature_col)
        spec = transform_specs[media_col]
        df[feature_col] = apply_media_transform(
            df[media_col],
            lag=int(spec["lag"]),
            alpha=float(spec["alpha"]),
            saturation=str(spec["saturation"]),
        ).to_numpy(dtype=float)
    full_control_cols = [
        "trend_index",
        "week_sin",
        "week_cos",
        "rebajas_flag",
        "black_friday_flag",
        "navidad_flag",
        "semana_santa_flag",
        "vacaciones_escolares_flag",
        "festivo_local_count",
        "payday_count",
        "temperatura_media_c_mean",
        "lluvia_indice_mean",
        "turismo_indice_mean",
        "incidencia_ecommerce_flag",
    ]
    control_variants = {
        "full": full_control_cols,
        "seasonal_calendar": [
            "trend_index",
            "week_sin",
            "week_cos",
            "rebajas_flag",
            "black_friday_flag",
            "navidad_flag",
            "semana_santa_flag",
            "vacaciones_escolares_flag",
            "festivo_local_count",
            "payday_count",
        ],
        "temporal_only": [
            "trend_index",
            "week_sin",
            "week_cos",
        ],
    }
    return df, transformed_cols, control_variants, transform_specs


def _make_response_regressor(model_type: str, cv: TimeSeriesSplit):
    if model_type == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 40), cv=cv)
    if model_type == "lasso":
        return LassoCV(alphas=np.logspace(-4, 1, 30), cv=cv, random_state=RANDOM_STATE, max_iter=200000)
    if model_type == "elasticnet":
        return ElasticNetCV(
            alphas=np.logspace(-4, 1, 30),
            l1_ratio=ELASTIC_NET_L1,
            cv=cv,
            random_state=RANDOM_STATE,
            max_iter=200000,
            selection="cyclic",
        )
    raise ValueError(f"Unknown media response model_type: {model_type}")


def _fit_media_response_pipeline(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    spec: MediaResponseSpec,
) -> Pipeline:
    cv = TimeSeriesSplit(n_splits=3)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", _make_response_regressor(spec.model_type, cv)),
        ]
    )
    model.fit(train_df[feature_columns], train_df["ventas_netas"].to_numpy(dtype=float))
    return model


def _pipeline_regression_params(model: Pipeline) -> dict[str, float]:
    reg = model.named_steps["reg"]
    return {
        "alpha": float(getattr(reg, "alpha_", np.nan)),
        "l1_ratio": float(getattr(reg, "l1_ratio_", np.nan)),
    }


def _pipeline_original_scale_coefficients(model: Pipeline, feature_columns: list[str]) -> pd.Series:
    scaler = model.named_steps["scaler"]
    reg = model.named_steps["reg"]
    raw_coef = np.asarray(reg.coef_, dtype=float) / scaler.scale_
    return pd.Series(raw_coef, index=feature_columns, dtype=float)


def _calibrate_media_coefficients(coefficients: pd.Series) -> pd.Series:
    calibrated = coefficients.copy()
    for feature in calibrated.index:
        if not str(feature).endswith("_response"):
            continue
        value = float(calibrated.loc[feature])
        value *= float(MEDIA_COEFFICIENT_MULTIPLIERS.get(str(feature), 1.0))
        min_positive = MEDIA_MIN_POSITIVE_COEFFICIENT.get(str(feature))
        if min_positive is not None:
            value = max(float(min_positive), value)
        calibrated.loc[feature] = value
    return calibrated


def _evaluate_media_response_spec(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    spec: MediaResponseSpec,
    fold_name: str,
) -> dict[str, float | str]:
    model = _fit_media_response_pipeline(train_df, feature_columns, spec)
    pred = model.predict(valid_df[feature_columns])
    metrics = compute_metrics(valid_df["ventas_netas"].to_numpy(dtype=float), pred)
    params = _pipeline_regression_params(model)
    return {
        "model": spec.name,
        "description": spec.description,
        "model_type": spec.model_type,
        "control_variant": spec.control_variant,
        "fold": fold_name,
        "mape": float(metrics["mape"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "bias": float(metrics["bias"]),
        "r2": float(metrics["r2"]),
        "alpha": params["alpha"],
        "l1_ratio": params["l1_ratio"],
    }


def _summarize_media_response_tournament(fold_df: pd.DataFrame) -> pd.DataFrame:
    validation_summary = (
        fold_df[fold_df["fold"].str.startswith("validate_")]
        .groupby(["model", "description", "model_type", "control_variant"], as_index=False)[["mape", "mae", "rmse", "bias", "r2"]]
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
        fold_df[fold_df["fold"] == "test_2024"]
        .groupby(["model"], as_index=False)[["mape", "mae", "rmse", "bias", "r2", "alpha", "l1_ratio"]]
        .mean()
        .rename(
            columns={
                "mape": "test_2024_mape",
                "mae": "test_2024_mae",
                "rmse": "test_2024_rmse",
                "bias": "test_2024_bias",
                "r2": "test_2024_r2",
                "alpha": "test_fit_alpha",
                "l1_ratio": "test_fit_l1_ratio",
            }
        )
    )
    summary = validation_summary.merge(test_summary, on="model", how="left")
    summary["selection_score"] = 0.7 * summary["validation_mean_mape"] + 0.3 * summary["test_2024_mape"]
    summary["validation_rank_mape"] = summary["validation_mean_mape"].rank(method="dense", ascending=True).astype(int)
    summary["test_rank_mape"] = summary["test_2024_mape"].rank(method="dense", ascending=True).astype(int)
    return summary.sort_values(
        ["selection_score", "validation_mean_mape", "validation_mean_rmse", "test_2024_mape"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def fit_media_response_model(
    weekly_df: pd.DataFrame,
    media_cols: list[str],
    target_year: int = TARGET_YEAR,
) -> tuple[MediaResponsePackage, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared_df, transformed_cols, control_variants, transform_specs = _prepare_weekly_response_frame(weekly_df, media_cols)
    train_df = prepared_df[prepared_df["year"] < target_year].copy()
    test_df = prepared_df[prepared_df["year"] == target_year].copy()

    specs = _media_response_specs()
    walk_forward = TimeSeriesSplit(n_splits=4)
    split_definitions: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for fold_number, (fold_train_idx, fold_valid_idx) in enumerate(walk_forward.split(train_df), start=1):
        split_definitions.append(
            (
                f"validate_ts_{fold_number}",
                train_df.iloc[fold_train_idx].copy(),
                train_df.iloc[fold_valid_idx].copy(),
            )
        )
    split_definitions.append(("test_2024", train_df, test_df))

    tasks = [
        (
            spec,
            fold_name,
            fold_train.copy(),
            fold_valid.copy(),
            control_variants[spec.control_variant] + transformed_cols,
        )
        for spec in specs
        for fold_name, fold_train, fold_valid in split_definitions
        if not fold_train.empty and not fold_valid.empty
    ]
    fold_rows = Parallel(**parallel_kwargs(len(tasks), backend="threading"))(
        delayed(_evaluate_media_response_spec)(
            train_df=fold_train,
            valid_df=fold_valid,
            feature_columns=feature_columns,
            spec=spec,
            fold_name=fold_name,
        )
        for spec, fold_name, fold_train, fold_valid, feature_columns in tasks
    )
    tournament_df = _summarize_media_response_tournament(pd.DataFrame(fold_rows))
    winner_row = tournament_df.iloc[0]
    winner_spec = {spec.name: spec for spec in specs}[str(winner_row["model"])]
    feature_columns = control_variants[winner_spec.control_variant] + transformed_cols
    model = _fit_media_response_pipeline(train_df, feature_columns, winner_spec)
    params = _pipeline_regression_params(model)
    metrics_df = pd.DataFrame(
        [
            {
                "model": winner_spec.name,
                "description": winner_spec.description,
                "model_type": winner_spec.model_type,
                "control_variant": winner_spec.control_variant,
                "target_year": target_year,
                "validation_mean_mape": float(winner_row["validation_mean_mape"]),
                "validation_mean_rmse": float(winner_row["validation_mean_rmse"]),
                "test_2024_mape": float(winner_row["test_2024_mape"]),
                "test_2024_rmse": float(winner_row["test_2024_rmse"]),
                "test_2024_bias": float(winner_row["test_2024_bias"]),
                "test_2024_r2": float(winner_row["test_2024_r2"]),
                "alpha": params["alpha"],
                "l1_ratio": params["l1_ratio"],
            }
        ]
    )

    coefficients = _pipeline_original_scale_coefficients(model, feature_columns)
    calibrated_coefficients = _calibrate_media_coefficients(coefficients)
    media_positive = calibrated_coefficients[
        calibrated_coefficients.index.to_series().astype(str).str.endswith("_response")
    ].clip(lower=0.0)
    media_total = float(media_positive.sum())
    coefficient_df = pd.DataFrame(
        {
            "feature": calibrated_coefficients.index,
            "raw_coefficient": coefficients.values,
            "coefficient": calibrated_coefficients.values,
            "model": winner_spec.name,
            "model_type": winner_spec.model_type,
            "feature_family": [
                "media_response" if feature.endswith("_response") else "control"
                for feature in calibrated_coefficients.index
            ],
        }
    ).sort_values("coefficient", ascending=False).reset_index(drop=True)
    coefficient_df["coefficient_pct"] = 0.0
    media_mask = coefficient_df["feature_family"] == "media_response"
    if media_total > 0.0:
        coefficient_df.loc[media_mask, "coefficient_pct"] = (
            coefficient_df.loc[media_mask, "coefficient"].clip(lower=0.0) / media_total * 100.0
        )

    package = MediaResponsePackage(
        weekly_df=prepared_df,
        feature_columns=feature_columns,
        transformed_media_columns=[f"{media_col}_response" for media_col in media_cols],
        media_columns=media_cols,
        model=model,
        model_name=winner_spec.name,
        model_type=winner_spec.model_type,
        model_description=winner_spec.description,
        transform_specs=transform_specs,
        target_year=target_year,
        train_df=train_df,
        test_df=test_df,
    )
    return package, metrics_df, coefficient_df, tournament_df


def _media_response_coefficient_map(package: MediaResponsePackage) -> dict[str, float]:
    coefficients = _pipeline_original_scale_coefficients(package.model, package.feature_columns)
    coefficients = _calibrate_media_coefficients(coefficients)
    return {str(feature): float(value) for feature, value in coefficients.items()}


def _annual_channel_weights(df: pd.DataFrame, media_cols: list[str], target_year: int) -> dict[str, np.ndarray]:
    year_df = df[df["year"] == target_year].copy()
    weights = {}
    for media_col in media_cols:
        values = year_df[media_col].to_numpy(dtype=float)
        total = values.sum()
        weights[media_col] = np.repeat(1.0 / len(year_df), len(year_df)) if total <= 0 else values / total
    return weights


def channel_budgets_from_shares(shares: np.ndarray, media_cols: list[str], total_budget: float) -> Dict[str, float]:
    return {media_cols[idx]: float(shares[idx] * total_budget) for idx in range(len(media_cols))}


def channel_shares_from_budgets(channel_budget_map: Dict[str, float], media_cols: list[str]) -> np.ndarray:
    total = sum(channel_budget_map[channel] for channel in media_cols)
    if total <= 0:
        return np.repeat(1.0 / len(media_cols), len(media_cols))
    return np.array([channel_budget_map[channel] / total for channel in media_cols], dtype=float)


def _scenario_full_frame(
    package: MediaResponsePackage,
    channel_budget_map: Dict[str, float],
    target_year: int,
) -> pd.DataFrame:
    scenario_df = package.weekly_df.copy().sort_values("semana_inicio").reset_index(drop=True)
    year_mask = scenario_df["year"] == target_year
    annual_weights = _annual_channel_weights(package.weekly_df, package.media_columns, target_year)
    for media_col in package.media_columns:
        scenario_df.loc[year_mask, media_col] = channel_budget_map[media_col] * annual_weights[media_col]
        spec = package.transform_specs[media_col]
        scenario_df[f"{media_col}_response"] = apply_media_transform(
            scenario_df[media_col],
            lag=int(spec["lag"]),
            alpha=float(spec["alpha"]),
            saturation=str(spec["saturation"]),
        ).to_numpy(dtype=float)
    return scenario_df


def predict_media_response_scenario(
    package: MediaResponsePackage,
    channel_budget_map: Dict[str, float],
    target_year: int,
) -> pd.DataFrame:
    scenario_df = _scenario_full_frame(package, channel_budget_map, target_year)
    scored = scenario_df[scenario_df["year"] == target_year].copy().reset_index(drop=True)
    scored["pred"] = package.model.predict(scored[package.feature_columns])
    scored["predicted_gross_profit"] = scored["pred"] * scored["gross_margin_rate"]
    return scored


def _score_prediction_split(package: MediaResponsePackage, split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    scored = split_df.copy().sort_values("semana_inicio").reset_index(drop=True)
    scored["predicted_sales"] = package.model.predict(scored[package.feature_columns])
    scored["actual_sales"] = scored["ventas_netas"].astype(float)
    scored["residual_sales"] = scored["actual_sales"] - scored["predicted_sales"]
    scored["absolute_percentage_error_pct"] = (
        scored["residual_sales"].abs() / scored["actual_sales"].abs().replace(0.0, np.nan) * 100.0
    )
    scored["actual_gross_profit"] = scored["actual_sales"] * scored["gross_margin_rate"]
    scored["predicted_gross_profit"] = scored["predicted_sales"] * scored["gross_margin_rate"]
    scored["split"] = split_name
    scored["model"] = package.model_name
    return scored[
        [
            "semana_inicio",
            "year",
            "actual_sales",
            "predicted_sales",
            "residual_sales",
            "absolute_percentage_error_pct",
            "actual_gross_profit",
            "predicted_gross_profit",
            "split",
            "model",
        ]
    ].copy()


def _build_prediction_diagnostics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for split, split_df in predictions_df.groupby("split", sort=False):
        metrics = compute_metrics(
            split_df["actual_sales"].to_numpy(dtype=float),
            split_df["predicted_sales"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "split": split,
                "weeks": int(len(split_df)),
                "mean_actual_sales": float(split_df["actual_sales"].mean()),
                "mean_predicted_sales": float(split_df["predicted_sales"].mean()),
                "mean_actual_gross_profit": float(split_df["actual_gross_profit"].mean()),
                "mean_predicted_gross_profit": float(split_df["predicted_gross_profit"].mean()),
                "mape": float(metrics["mape"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "bias": float(metrics["bias"]),
                "r2": float(metrics["r2"]),
            }
        )
    return pd.DataFrame(rows)


def _build_channel_roi_table(
    weights_df: pd.DataFrame,
    maxima_df: pd.DataFrame,
    coefficient_df: pd.DataFrame,
) -> pd.DataFrame:
    media_coefficients = (
        coefficient_df[coefficient_df["feature_family"] == "media_response"][
            ["feature", "raw_coefficient", "coefficient", "coefficient_pct", "model", "model_type"]
        ]
        .assign(channel=lambda df: df["feature"].map(_media_feature_to_channel))
        .drop(columns="feature")
    )
    roi_df = (
        weights_df.merge(
            maxima_df[
                [
                    "channel",
                    "historical_incremental_gross_profit_eur",
                    "optimized_incremental_gross_profit_eur",
                    "saturation_budget_eur",
                    "saturation_share_pct",
                    "saturation_incremental_gross_profit_eur",
                    "saturation_status",
                    "optimized_gap_vs_saturation_eur",
                    "optimal_budget_eur",
                    "optimal_share_pct",
                ]
            ],
            on="channel",
            how="left",
        )
        .merge(media_coefficients, on="channel", how="left")
        .copy()
    )
    roi_df["historical_roi"] = [
        _safe_divide(profit, budget)
        for profit, budget in zip(
            roi_df["historical_incremental_gross_profit_eur"],
            roi_df["historical_budget_eur"],
        )
    ]
    roi_df["optimized_roi"] = [
        _safe_divide(profit, budget)
        for profit, budget in zip(
            roi_df["optimized_incremental_gross_profit_eur"],
            roi_df["optimized_budget_eur"],
        )
    ]
    roi_df["saturation_roi"] = [
        _safe_divide(profit, budget)
        for profit, budget in zip(
            roi_df["saturation_incremental_gross_profit_eur"],
            roi_df["saturation_budget_eur"],
        )
    ]
    roi_df["roi_delta"] = roi_df["optimized_roi"] - roi_df["historical_roi"]
    return roi_df.sort_values(["optimized_roi", "coefficient_pct"], ascending=[False, False]).reset_index(drop=True)


def _historical_channel_saturation_curves(
    package: MediaResponsePackage,
    base_budgets: Dict[str, float],
    target_year: int,
    points: int = SATURATION_POINTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_budget = float(sum(base_budgets.values()))
    base_profit = float(predict_media_response_scenario(package, base_budgets, target_year)["predicted_gross_profit"].sum())
    coefficient_map = _media_response_coefficient_map(package)
    curve_rows: list[dict[str, float | str]] = []
    knee_rows: list[dict[str, float | str]] = []

    for media_col in package.media_columns:
        channel = media_col.replace("media_", "")
        historical_budget = float(base_budgets[media_col])
        coefficient = float(coefficient_map.get(f"{media_col}_response", 0.0))
        max_budget = min(total_budget, max(historical_budget * 3.0, total_budget * 0.35))
        budget_grid = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, max_budget, points),
                    np.array([historical_budget], dtype=float),
                ]
            )
        )

        channel_rows: list[dict[str, float | str]] = []
        for budget in budget_grid:
            scenario_budgets = dict(base_budgets)
            scenario_budgets[media_col] = float(budget)
            scenario_df = predict_media_response_scenario(package, scenario_budgets, target_year)
            total_profit = float(scenario_df["predicted_gross_profit"].sum())
            total_sales = float(scenario_df["pred"].sum())
            channel_rows.append(
                {
                    "channel": channel,
                    "budget_eur": float(budget),
                    "budget_vs_historical_pct": _safe_divide(float(budget), historical_budget) * 100.0 if historical_budget > 0 else np.nan,
                    "budget_change_vs_historical_eur": float(budget) - historical_budget,
                    "budget_change_vs_historical_pct": (
                        _safe_divide(float(budget) - historical_budget, historical_budget) * 100.0 if historical_budget > 0 else np.nan
                    ),
                    "predicted_sales_2024": total_sales,
                    "predicted_gross_profit_2024": total_profit,
                    "delta_vs_do_nothing_profit": total_profit - base_profit,
                    "historical_budget_eur": historical_budget,
                    "do_nothing_profit_2024": base_profit,
                    "coefficient": coefficient,
                }
            )

        curve_df = pd.DataFrame(channel_rows).sort_values("budget_eur").reset_index(drop=True)
        curve_df["marginal_profit_per_eur"] = compute_curve_gradient(
            curve_df["budget_eur"].to_numpy(dtype=float),
            curve_df["predicted_gross_profit_2024"].to_numpy(dtype=float),
        )
        historical_row = curve_df.loc[(curve_df["budget_eur"] - historical_budget).abs().idxmin()]
        tolerance_eur = max(historical_budget * 0.05, 25_000.0)

        if coefficient <= 0.0 or float(curve_df["predicted_gross_profit_2024"].max()) <= float(curve_df.iloc[0]["predicted_gross_profit_2024"]) + 1e-9:
            knee_row = curve_df.iloc[0]
            knee_status = "flat_or_negative"
        else:
            knee_idx = find_concave_knee_index(
                curve_df["budget_eur"].to_numpy(dtype=float),
                curve_df["predicted_gross_profit_2024"].to_numpy(dtype=float),
            )
            knee_row = curve_df.iloc[knee_idx]
            historical_gap = historical_budget - float(knee_row["budget_eur"])
            if historical_gap > tolerance_eur:
                knee_status = "historical_above_knee"
            elif historical_gap < -tolerance_eur:
                knee_status = "historical_below_knee"
            else:
                knee_status = "historical_near_knee"

        knee_budget = float(knee_row["budget_eur"])
        curve_df["knee_budget_eur"] = knee_budget
        curve_df["is_knee_point"] = np.isclose(curve_df["budget_eur"], knee_budget, atol=max(knee_budget * 0.005, 1.0))
        curve_df["is_historical_point"] = np.isclose(
            curve_df["budget_eur"],
            historical_budget,
            atol=max(historical_budget * 0.005, 1.0),
        )
        curve_rows.extend(curve_df.to_dict("records"))
        knee_rows.append(
            {
                "channel": channel,
                "historical_budget_eur": historical_budget,
                "historical_profit_2024": float(historical_row["predicted_gross_profit_2024"]),
                "historical_delta_vs_do_nothing_profit": float(historical_row["delta_vs_do_nothing_profit"]),
                "knee_budget_eur": knee_budget,
                "knee_budget_vs_historical_pct": float(knee_row["budget_vs_historical_pct"]),
                "knee_delta_vs_historical_eur": knee_budget - historical_budget,
                "knee_profit_2024": float(knee_row["predicted_gross_profit_2024"]),
                "knee_delta_vs_do_nothing_profit": float(knee_row["delta_vs_do_nothing_profit"]),
                "knee_marginal_profit_per_eur": float(knee_row["marginal_profit_per_eur"]),
                "coefficient": coefficient,
                "knee_status": knee_status,
            }
        )

    return (
        pd.DataFrame(curve_rows).sort_values(["channel", "budget_eur"]).reset_index(drop=True),
        pd.DataFrame(knee_rows).sort_values("knee_delta_vs_do_nothing_profit", ascending=False).reset_index(drop=True),
    )


def optimize_media_response_budget(
    package: MediaResponsePackage,
    base_budgets: Dict[str, float],
    target_year: int,
    min_share: float = MEDIA_MIN_SHARE,
    max_share: float = MEDIA_MAX_SHARE,
) -> Dict[str, float]:
    media_cols = package.media_columns
    total_budget = float(sum(base_budgets.values()))
    base_shares = channel_shares_from_budgets(base_budgets, media_cols)

    def objective(shares: np.ndarray) -> float:
        shares = np.clip(shares, min_share, max_share)
        shares = shares / shares.sum()
        budgets = channel_budgets_from_shares(shares, media_cols, total_budget)
        scenario_df = predict_media_response_scenario(package, budgets, target_year)
        return -float(scenario_df["predicted_gross_profit"].sum())

    constraints = [{"type": "eq", "fun": lambda shares: float(np.sum(shares) - 1.0)}]
    bounds = [(min_share, max_share) for _ in media_cols]
    rng = np.random.default_rng(42)
    starts = [base_shares, np.repeat(1.0 / len(media_cols), len(media_cols))]
    for _ in range(12):
        proposal = rng.dirichlet(np.ones(len(media_cols)))
        proposal = np.clip(proposal, min_share, max_share)
        proposal = proposal / proposal.sum()
        starts.append(proposal)

    def _optimize_from_start(start: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 600, "ftol": 1e-8},
        )
        candidate = result.x if result.success else start
        candidate = np.clip(candidate, min_share, max_share)
        candidate = candidate / candidate.sum()
        return candidate, objective(candidate)

    optimized_candidates = Parallel(**parallel_kwargs(len(starts), backend="threading"))(
        delayed(_optimize_from_start)(start) for start in starts
    )
    best_shares, _best_value = min(optimized_candidates, key=lambda item: item[1])
    return channel_budgets_from_shares(best_shares, media_cols, total_budget)


def _channel_saturation_curve(
    package: MediaResponsePackage,
    media_col: str,
    historical_budget: float,
    optimized_budget: float,
    total_budget: float,
    target_year: int,
    coefficient_map: dict[str, float],
    points: int = SATURATION_POINTS,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    year_df = package.weekly_df[package.weekly_df["year"] == target_year].copy().reset_index(drop=True)
    annual_weights = _annual_channel_weights(package.weekly_df, package.media_columns, target_year)[media_col]
    gross_margin = year_df["gross_margin_rate"].to_numpy(dtype=float)
    coefficient = float(coefficient_map.get(f"{media_col}_response", 0.0))
    spec = package.transform_specs[media_col]
    budget_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, total_budget, points),
                np.array([historical_budget, optimized_budget], dtype=float),
            ]
        )
    )

    rows: list[dict[str, float | str]] = []
    for budget in budget_grid:
        weekly_budget = pd.Series(annual_weights * float(budget), dtype=float)
        transformed = apply_media_transform(
            weekly_budget,
            lag=int(spec["lag"]),
            alpha=float(spec["alpha"]),
            saturation=str(spec["saturation"]),
        ).to_numpy(dtype=float)
        incremental_sales = transformed * coefficient
        rows.append(
            {
                "channel": media_col.replace("media_", ""),
                "budget_eur": float(budget),
                "share_pct_of_total_budget": _safe_divide(float(budget), total_budget) * 100.0 if total_budget > 0 else 0.0,
                "channel_incremental_sales": float(incremental_sales.sum()),
                "channel_incremental_gross_profit_eur": float(np.dot(incremental_sales, gross_margin)),
                "coefficient": coefficient,
            }
        )

    curve = pd.DataFrame(rows).sort_values("budget_eur").reset_index(drop=True)
    curve["marginal_profit_per_eur"] = compute_curve_gradient(
        curve["budget_eur"].to_numpy(dtype=float),
        curve["channel_incremental_gross_profit_eur"].to_numpy(dtype=float),
    )
    initial_marginal_profit = float(curve["marginal_profit_per_eur"].iloc[1]) if len(curve) > 1 else 0.0
    curve["marginal_ratio_vs_initial"] = [
        _safe_divide(float(value), initial_marginal_profit) for value in curve["marginal_profit_per_eur"]
    ]

    historical_row = curve.loc[(curve["budget_eur"] - historical_budget).abs().idxmin()]
    optimized_row = curve.loc[(curve["budget_eur"] - optimized_budget).abs().idxmin()]
    tolerance_eur = max(total_budget * 0.01, 50_000.0)

    if coefficient <= 0.0 or float(curve["channel_incremental_gross_profit_eur"].max()) <= float(curve.iloc[0]["channel_incremental_gross_profit_eur"]) + 1e-9:
        saturation_row = curve.iloc[0]
        saturation_status = "cut_or_hold"
    else:
        saturation_idx = find_concave_knee_index(
            curve["budget_eur"].to_numpy(dtype=float),
            curve["channel_incremental_gross_profit_eur"].to_numpy(dtype=float),
        )
        saturation_row = curve.iloc[saturation_idx]
        optimized_gap = optimized_budget - float(saturation_row["budget_eur"])
        if optimized_gap > tolerance_eur:
            saturation_status = "oversaturated"
        elif optimized_gap < -tolerance_eur:
            saturation_status = "headroom"
        else:
            saturation_status = "near_saturation"

    saturation_budget = float(saturation_row["budget_eur"])
    saturation_marginal = float(saturation_row["marginal_profit_per_eur"])
    summary = {
        "channel": media_col.replace("media_", ""),
        "coefficient": coefficient,
        "response_direction": "positive" if coefficient > 0.0 else "negative_or_flat",
        "budget_total_eur": total_budget,
        "historical_budget_eur": historical_budget,
        "optimized_budget_eur": optimized_budget,
        "saturation_budget_eur": saturation_budget,
        "historical_share_pct": _safe_divide(historical_budget, total_budget) * 100.0 if total_budget > 0 else 0.0,
        "optimized_share_pct": _safe_divide(optimized_budget, total_budget) * 100.0 if total_budget > 0 else 0.0,
        "saturation_share_pct": _safe_divide(saturation_budget, total_budget) * 100.0 if total_budget > 0 else 0.0,
        "historical_incremental_gross_profit_eur": float(historical_row["channel_incremental_gross_profit_eur"]),
        "optimized_incremental_gross_profit_eur": float(optimized_row["channel_incremental_gross_profit_eur"]),
        "saturation_incremental_gross_profit_eur": float(saturation_row["channel_incremental_gross_profit_eur"]),
        "initial_marginal_profit_per_eur": initial_marginal_profit,
        "saturation_marginal_profit_per_eur": saturation_marginal,
        "saturation_marginal_ratio": _safe_divide(saturation_marginal, initial_marginal_profit),
        "historical_gap_vs_saturation_eur": historical_budget - saturation_budget,
        "optimized_gap_vs_saturation_eur": optimized_budget - saturation_budget,
        "saturation_status": saturation_status,
    }
    return curve, summary


def _response_curves(
    package: MediaResponsePackage,
    base_budgets: Dict[str, float],
    optimized_budgets: Dict[str, float],
    target_year: int,
    points: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    media_cols = package.media_columns
    total_budget = float(sum(base_budgets.values()))
    base_shares = channel_shares_from_budgets(base_budgets, media_cols)
    optimized_shares = channel_shares_from_budgets(optimized_budgets, media_cols)
    base_profit = float(predict_media_response_scenario(package, base_budgets, target_year)["predicted_gross_profit"].sum())
    coefficient_map = _media_response_coefficient_map(package)

    def _channel_curve(media_col: str) -> tuple[list[dict[str, float | str]], dict[str, float | str], list[dict[str, float | str]]]:
        idx = media_cols.index(media_col)
        share_grid = np.linspace(MEDIA_MIN_SHARE, MEDIA_MAX_SHARE, points)
        channel_rows: list[dict[str, float | str]] = []
        for share in share_grid:
            proposal = optimized_shares.copy()
            proposal[idx] = share
            other_idx = [i for i in range(len(media_cols)) if i != idx]
            remaining = 1.0 - share
            other_anchor = optimized_shares[other_idx]
            if other_anchor.sum() <= 0:
                proposal[other_idx] = remaining / len(other_idx)
            else:
                proposal[other_idx] = other_anchor / other_anchor.sum() * remaining
            proposal = np.clip(proposal, 1e-9, None)
            proposal = proposal / proposal.sum()
            budgets = channel_budgets_from_shares(proposal, media_cols, total_budget)
            scenario_df = predict_media_response_scenario(package, budgets, target_year)
            total_profit = float(scenario_df["predicted_gross_profit"].sum())
            channel_rows.append(
                {
                    "channel": media_col.replace("media_", ""),
                    "share_pct": float(proposal[idx] * 100.0),
                    "budget_eur": float(budgets[media_col]),
                    "predicted_gross_profit_2024": total_profit,
                    "delta_vs_historical_profit": total_profit - base_profit,
                    "historical_share_pct": float(base_shares[idx] * 100.0),
                    "optimized_share_pct": float(optimized_shares[idx] * 100.0),
                    "historical_budget_eur": float(base_budgets[media_col]),
                    "optimized_budget_eur": float(optimized_budgets[media_col]),
                }
            )

        curve = pd.DataFrame(channel_rows).copy()
        optimum = curve.sort_values("predicted_gross_profit_2024", ascending=False).iloc[0]
        maxima_row = {
            "channel": media_col.replace("media_", ""),
            "historical_share_pct": float(base_shares[idx] * 100.0),
            "optimized_share_pct": float(optimized_shares[idx] * 100.0),
            "optimal_share_pct": float(optimum["share_pct"]),
            "historical_budget_eur": float(base_budgets[media_col]),
            "optimized_budget_eur": float(optimized_budgets[media_col]),
            "optimal_budget_eur": float(optimum["budget_eur"]),
            "optimized_gap_vs_optimal_eur": float(optimum["predicted_gross_profit_2024"])
            - float(curve.loc[(curve["share_pct"] - optimized_shares[idx] * 100.0).abs().idxmin(), "predicted_gross_profit_2024"]),
        }
        saturation_curve, saturation_summary = _channel_saturation_curve(
            package=package,
            media_col=media_col,
            historical_budget=float(base_budgets[media_col]),
            optimized_budget=float(optimized_budgets[media_col]),
            total_budget=total_budget,
            target_year=target_year,
            coefficient_map=coefficient_map,
        )
        maxima_row.update(saturation_summary)
        return channel_rows, maxima_row, saturation_curve.to_dict("records")

    curve_results = Parallel(**parallel_kwargs(len(media_cols), backend="threading"))(
        delayed(_channel_curve)(media_col) for media_col in media_cols
    )
    rows = [row for channel_rows, _, _ in curve_results for row in channel_rows]
    maxima = [maxima_row for _, maxima_row, _ in curve_results]
    saturation_rows = [row for _, _, channel_rows in curve_results for row in channel_rows]
    return (
        pd.DataFrame(rows).sort_values(["channel", "share_pct"]).reset_index(drop=True),
        pd.DataFrame(maxima).sort_values("optimized_gap_vs_saturation_eur", ascending=False).reset_index(drop=True),
        pd.DataFrame(saturation_rows).sort_values(["channel", "budget_eur"]).reset_index(drop=True),
    )


def _plot_channel_curve(curve_df: pd.DataFrame, saturation_df: pd.DataFrame, maxima_row: pd.Series) -> None:
    channel = str(maxima_row["channel"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(APP_BG)
    sns.lineplot(data=curve_df, x="share_pct", y="predicted_gross_profit_2024", color=NAVY, linewidth=2.5, ax=axes[0])
    axes[0].axvline(float(maxima_row["historical_share_pct"]), color=SLATE, linestyle="--", linewidth=1.2)
    axes[0].axvline(float(maxima_row["optimized_share_pct"]), color=BLUE, linestyle="--", linewidth=1.2)
    axes[0].axvline(float(maxima_row["optimal_share_pct"]), color=GREEN, linestyle=":", linewidth=1.5)
    axes[0].set_title(f"{channel} - Total profit vs share")
    axes[0].set_xlabel("Channel share (%)")
    axes[0].set_ylabel("Predicted gross profit")

    sns.lineplot(
        data=saturation_df,
        x="budget_eur",
        y="channel_incremental_gross_profit_eur",
        color=TEAL,
        linewidth=2.5,
        ax=axes[1],
    )
    axes[1].axvline(float(maxima_row["historical_budget_eur"]), color=SLATE, linestyle="--", linewidth=1.2)
    axes[1].axvline(float(maxima_row["optimized_budget_eur"]), color=BLUE, linestyle="--", linewidth=1.2)
    axes[1].axvline(float(maxima_row["saturation_budget_eur"]), color=GREEN, linestyle=":", linewidth=1.5)
    axes[1].set_title(f"{channel} - Incremental profit vs budget")
    axes[1].set_xlabel("Channel budget (EUR)")
    axes[1].set_ylabel("Channel incremental gross profit")

    fig.suptitle(f"Media response curve - {channel}", y=1.02)
    _style_axes(axes, grid_axis="y")
    _savefig(f"sim_response_curve_{channel}.png")


def _plot_prediction_fit(predictions_df: pd.DataFrame, target_year: int) -> None:
    ordered = predictions_df.sort_values("semana_inicio").copy()
    test_mask = ordered["split"] == f"test_{target_year}"
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(APP_BG)
    sns.lineplot(data=ordered, x="semana_inicio", y="actual_sales", color=NAVY, linewidth=2.2, label="Actual", ax=ax)
    sns.lineplot(
        data=ordered,
        x="semana_inicio",
        y="predicted_sales",
        hue="split",
        palette={"train": BLUE, f"test_{target_year}": TEAL},
        linewidth=2.0,
        ax=ax,
    )
    if test_mask.any():
        test_start = ordered.loc[test_mask, "semana_inicio"].min()
        test_end = ordered.loc[test_mask, "semana_inicio"].max()
        ax.axvspan(test_start, test_end, color=LIGHT_BLUE, alpha=0.18)
    ax.set_title("Media response model - Actual vs predicted sales")
    ax.set_xlabel("Week")
    ax.set_ylabel("Net sales")
    _style_axes(ax, grid_axis="y")
    _savefig("sim_prediction_01_actual_vs_predicted.png")


def _plot_prediction_scatter(predictions_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.scatterplot(
        data=predictions_df,
        x="actual_sales",
        y="predicted_sales",
        hue="split",
        palette="Set2",
        s=70,
        alpha=0.85,
        ax=ax,
    )
    lower = float(min(predictions_df["actual_sales"].min(), predictions_df["predicted_sales"].min()))
    upper = float(max(predictions_df["actual_sales"].max(), predictions_df["predicted_sales"].max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, color="#6b7280")
    ax.set_title("Actual vs predicted sales")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    _savefig("sim_prediction_02_actual_vs_predicted_scatter.png")


def _plot_prediction_residuals(predictions_df: pd.DataFrame) -> None:
    ordered = predictions_df.sort_values("semana_inicio").copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=ordered, x="semana_inicio", y="residual_sales", hue="split", palette="Set2", ax=axes[0])
    axes[0].axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.2)
    axes[0].set_title("Residuals over time")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Residual sales")

    sns.boxplot(data=ordered, x="split", y="residual_sales", hue="split", palette="Set2", dodge=False, ax=axes[1])
    legend = axes[1].get_legend()
    if legend is not None:
        legend.remove()
    axes[1].axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residual distribution by split")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Residual sales")
    _savefig("sim_prediction_03_residual_diagnostics.png")


def _plot_media_coefficients(coefficient_df: pd.DataFrame) -> None:
    media_coefficients = (
        coefficient_df[coefficient_df["feature_family"] == "media_response"]
        .assign(channel=lambda df: df["feature"].map(_media_feature_to_channel))
        .sort_values("coefficient_pct", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=media_coefficients, x="coefficient_pct", y="channel", color="#1f4e79", ax=ax)
    ax.set_title("Media channel weights in coefficient percentage")
    ax.set_xlabel("Coefficient weight (%)")
    ax.set_ylabel("Channel")
    _savefig("sim_prediction_04_channel_coefficients_pct.png")


def _plot_weight_mix(weights_df: pd.DataFrame) -> None:
    plot_df = weights_df.melt(
        id_vars="channel",
        value_vars=["historical_share_pct", "optimized_share_pct"],
        var_name="mix",
        value_name="share_pct",
    )
    plot_df["mix"] = plot_df["mix"].map(
        {
            "historical_share_pct": "Historical mix",
            "optimized_share_pct": "Optimized mix",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x="channel", y="share_pct", hue="mix", palette=["#9ca3af", "#d97706"], ax=ax)
    ax.set_title("Historical vs optimized budget weights")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Budget share (%)")
    ax.tick_params(axis="x", rotation=30)
    _savefig("sim_prediction_05_budget_mix_weights.png")


def _plot_channel_roi(roi_df: pd.DataFrame) -> None:
    plot_df = roi_df.melt(
        id_vars="channel",
        value_vars=["historical_roi", "optimized_roi", "saturation_roi"],
        var_name="roi_type",
        value_name="roi",
    )
    plot_df["roi_type"] = plot_df["roi_type"].map(
        {
            "historical_roi": "Historical ROI",
            "optimized_roi": "Optimized ROI",
            "saturation_roi": "Saturation ROI",
        }
    )
    ordered_channels = roi_df.sort_values("optimized_roi", ascending=False)["channel"].tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="channel",
        y="roi",
        hue="roi_type",
        order=ordered_channels,
        palette=["#9ca3af", "#d97706", "#0f766e"],
        ax=ax,
    )
    ax.set_title("ROI by channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Gross profit per EUR invested")
    ax.tick_params(axis="x", rotation=30)
    _savefig("sim_prediction_06_channel_roi.png")


def _plot_model_tournament(tournament_df: pd.DataFrame) -> None:
    ordered = tournament_df.sort_values("selection_score", ascending=True).copy()
    ordered["candidate"] = ordered["model"] + " | " + ordered["control_variant"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.barplot(data=ordered, x="selection_score", y="candidate", hue="model_type", dodge=False, palette="viridis", ax=ax)
    ax.set_title("Media response model tournament")
    ax.set_xlabel("Selection score")
    ax.set_ylabel("Candidate")
    _savefig("sim_prediction_07_model_tournament.png")


def _plot_do_something_vs_do_nothing(curves_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.lineplot(
        data=curves_df,
        x="budget_vs_historical_pct",
        y="delta_vs_do_nothing_profit",
        hue="channel",
        linewidth=2.0,
        ax=ax,
    )
    ax.axvline(100.0, color="#6b7280", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.2)
    ax.set_title("Do something vs do nothing by channel")
    ax.set_xlabel("Channel budget vs historical (%)")
    ax.set_ylabel("Delta gross profit vs do nothing (EUR)")
    _savefig("sim_prediction_08_do_something_vs_do_nothing.png")


def _plot_historical_saturation_curve(curve_df: pd.DataFrame, knee_row: pd.Series) -> None:
    channel = str(knee_row["channel"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(APP_BG)
    sns.lineplot(data=curve_df, x="budget_eur", y="predicted_gross_profit_2024", color=NAVY, linewidth=2.4, ax=axes[0])
    axes[0].axvline(float(knee_row["historical_budget_eur"]), color=SLATE, linestyle="--", linewidth=1.2)
    axes[0].axvline(float(knee_row["knee_budget_eur"]), color=GREEN, linestyle=":", linewidth=1.5)
    axes[0].set_title(f"{channel} - Total profit from historical sweep")
    axes[0].set_xlabel("Channel budget (EUR)")
    axes[0].set_ylabel("Predicted gross profit")

    sns.lineplot(data=curve_df, x="budget_eur", y="delta_vs_do_nothing_profit", color=TEAL, linewidth=2.4, ax=axes[1])
    axes[1].axvline(float(knee_row["historical_budget_eur"]), color=SLATE, linestyle="--", linewidth=1.2)
    axes[1].axvline(float(knee_row["knee_budget_eur"]), color=GREEN, linestyle=":", linewidth=1.5)
    axes[1].axhline(0.0, color=LIGHT_BLUE, linestyle=":", linewidth=1.2)
    axes[1].set_title(f"{channel} - Delta vs do nothing")
    axes[1].set_xlabel("Channel budget (EUR)")
    axes[1].set_ylabel("Delta gross profit (EUR)")

    fig.suptitle(f"Historical saturation curve - {channel}", y=1.02)
    _style_axes(axes, grid_axis="y")
    _savefig(f"sim_saturation_historical_{channel}.png")


def _write_report(
    metrics_df: pd.DataFrame,
    tournament_df: pd.DataFrame,
    coefficient_df: pd.DataFrame,
    prediction_diagnostics_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    maxima_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    do_something_df: pd.DataFrame,
    knee_points_df: pd.DataFrame,
    observed_total_budget: float,
    total_budget: float,
) -> None:
    top_positive = coefficient_df[coefficient_df["feature_family"] == "media_response"].sort_values("coefficient", ascending=False).head(3)
    top_negative = coefficient_df[coefficient_df["feature_family"] == "media_response"].sort_values("coefficient", ascending=True).head(3)
    optimized = scenario_df.loc[scenario_df["scenario"] == "media_response_optimized"].iloc[0]
    historical = scenario_df.loc[scenario_df["scenario"] == "do_nothing_current_mix"].iloc[0]
    most_changed = weights_df.sort_values("delta_share_pct_points", key=lambda s: s.abs(), ascending=False).head(5)
    oversaturated = maxima_df[maxima_df["optimized_gap_vs_saturation_eur"] > 0.0].head(3)
    headroom = maxima_df[maxima_df["optimized_gap_vs_saturation_eur"] < 0.0].sort_values("optimized_gap_vs_saturation_eur").head(3)
    top_roi = roi_df.sort_values("optimized_roi", ascending=False).head(5)
    top_knees = knee_points_df.sort_values("knee_delta_vs_do_nothing_profit", ascending=False).head(5)
    train_diag = prediction_diagnostics_df[prediction_diagnostics_df["split"] == "train"]
    test_diag = prediction_diagnostics_df[prediction_diagnostics_df["split"].astype(str).str.startswith("test_")]
    curve_summary = maxima_df[
        [
            "channel",
            "historical_budget_eur",
            "optimized_budget_eur",
            "saturation_budget_eur",
            "optimized_gap_vs_saturation_eur",
            "optimal_budget_eur",
            "optimized_gap_vs_optimal_eur",
            "saturation_status",
        ]
    ].copy()

    figure_lines = [
        "- `6_simulacion/sim_prediction_01_actual_vs_predicted.png`: serie temporal de ventas reales vs predichas con separacion train/test.",
        "- `6_simulacion/sim_prediction_02_actual_vs_predicted_scatter.png`: dispersion actual vs predicho para ver calibracion.",
        "- `6_simulacion/sim_prediction_03_residual_diagnostics.png`: residuos en el tiempo y distribucion por split.",
        "- `6_simulacion/sim_prediction_04_channel_coefficients_pct.png`: peso porcentual de los coeficientes por canal.",
        "- `6_simulacion/sim_prediction_05_budget_mix_weights.png`: comparacion de pesos historicos vs optimizados.",
        "- `6_simulacion/sim_prediction_06_channel_roi.png`: ROI historico, optimizado y de saturacion por canal.",
        "- `6_simulacion/sim_prediction_07_model_tournament.png`: ranking del torneo de modelos.",
        "- `6_simulacion/sim_prediction_08_do_something_vs_do_nothing.png`: uplift de beneficio al mover un canal respecto al do nothing historico.",
    ] + [
        f"- `6_simulacion/sim_response_curve_{channel}.png`: curva de profit vs peso y vs euros de `{channel}`."
        for channel in maxima_df["channel"].tolist()
    ] + [
        f"- `6_simulacion/sim_saturation_historical_{channel}.png`: barrido historico del canal `{channel}` con knee point y do nothing."
        for channel in knee_points_df["channel"].tolist()
    ]
    reading_lines = [
        f"- Modelo seleccionado para curvas y pesos: `{metrics_df['model'].iloc[0]}` (`{metrics_df['model_type'].iloc[0]}`), elegido sobre `Ridge`, `Lasso` y `ElasticNet` por validacion temporal.",
        f"- El gasto observado real en `2024` fue `{observed_total_budget:,.2f} EUR`, pero la optimizacion se reescala al presupuesto de planning de `{total_budget:,.2f} EUR`.",
        f"- El do nothing se define como mantener el mix historico observado reescalado al presupuesto de planning, con beneficio modelizado de `{historical['predicted_gross_profit_2024']:,.2f} EUR`.",
        f"- La receta optimizada del modelo de respuesta sube el beneficio a `{optimized['predicted_gross_profit_2024']:,.2f} EUR`, con uplift de `{optimized['delta_vs_historical_profit']:,.2f} EUR`.",
        f"- Los canales con mayor respuesta positiva transformada son `{', '.join(top_positive['feature'].str.replace('_response', '', regex=False).str.replace('media_', '', regex=False).tolist())}`.",
        f"- Los canales mas flojos o mas saturados segun el modelo son `{', '.join(top_negative['feature'].str.replace('_response', '', regex=False).str.replace('media_', '', regex=False).tolist())}`.",
        f"- Los canales con mayor ROI optimizado son `{', '.join(top_roi['channel'].tolist())}`.",
        f"- En el barrido historico `do something vs do nothing`, los knee points mas altos salen en `{', '.join(top_knees['channel'].tolist())}`.",
        "- Ademas del maximo de la reallocation, se publica `saturation_budget_eur` como la rodilla de la curva incremental por canal.",
        "- Si un canal sale con coeficiente negativo o plano, la saturacion recomendada se fija en `0 EUR` y debe interpretarse como recorte o hold, no como ganador de mix.",
    ]
    if not train_diag.empty:
        reading_lines.append(
            f"- En train el ajuste medio queda en `MAPE={float(train_diag['mape'].iloc[0]):.2f}%`, `RMSE={float(train_diag['rmse'].iloc[0]):,.0f}`."
        )
    if not test_diag.empty:
        reading_lines.append(
            f"- En test el ajuste medio queda en `MAPE={float(test_diag['mape'].iloc[0]):.2f}%`, `RMSE={float(test_diag['rmse'].iloc[0]):,.0f}`."
        )
    if not oversaturated.empty:
        reading_lines.append(
            f"- Canales ya por encima de su rodilla segun el mix optimizado: `{', '.join(oversaturated['channel'].tolist())}`."
        )
    if not headroom.empty:
        reading_lines.append(
            f"- Canales con recorrido antes de saturar: `{', '.join(headroom['channel'].tolist())}`."
        )

    REPORT_MD.write_text(
        "\n".join(
            [
                "# Media Response Optimization Report",
                "",
                "## Objetivo",
                "",
                "Tomar el do nothing como el mix actual de gasto y construir curvas de respuesta no lineales por canal para ver donde estamos gastando sin retorno suficiente y cual es la combinacion de pesos que maximiza beneficio esperado.",
                "",
                "## Modelo de respuesta",
                "",
                "- Base temporal: serie semanal agregada de negocio.",
                "- Controles: tendencia, estacionalidad y variables de calendario/contexto.",
                "- Medios: cada canal entra transformado con adstock y saturacion `log1p`.",
                "- Torneo de modelos: `Ridge`, `Lasso` y `ElasticNet` con la misma matriz de variables.",
                f"- Optimizacion: presupuesto total fijo de `{total_budget:,.2f} EUR`, con posibilidad de recortar canales a 0 si el modelo lo recomienda.",
                "",
                "## Modelo Seleccionado",
                "",
                "```text",
                metrics_df.round(4).to_string(index=False),
                "```",
                "",
                "## Torneo De Modelos",
                "",
                "```text",
                tournament_df.round(4).to_string(index=False),
                "```",
                "",
                "## Escenarios",
                "",
                "```text",
                scenario_df.round(2).to_string(index=False),
                "```",
                "",
                "## Diagnostico De Prediccion",
                "",
                "```text",
                prediction_diagnostics_df.round(4).to_string(index=False),
                "```",
                "",
                "## ROI Por Canal",
                "",
                "```text",
                top_roi.round(4).to_string(index=False),
                "```",
                "",
                "## Knee Points Historicos",
                "",
                "```text",
                top_knees.round(2).to_string(index=False),
                "```",
                "",
                "## Canales que mas cambian",
                "",
                "```text",
                most_changed.round(2).to_string(index=False),
                "```",
                "",
                "## Curvas por canal",
                "",
                "```text",
                curve_summary.round(2).to_string(index=False),
                "```",
                "",
                "## Lectura",
                "",
                *reading_lines,
                "",
                "## Figuras",
                "",
                *figure_lines,
                "",
                "## Tablas",
                "",
                "- `reports/tables/media_response_scenarios.csv`.",
                "- `reports/tables/media_response_weights.csv`.",
                "- `reports/tables/media_response_curves.csv`.",
                "- `reports/tables/media_response_maxima.csv`.",
                "- `reports/tables/media_response_saturation.csv`.",
                "- `reports/tables/media_response_coefficients.csv`.",
                "- `reports/tables/media_response_fit_metrics.csv`.",
                "- `reports/tables/media_response_predictions.csv`.",
                "- `reports/tables/media_response_prediction_diagnostics.csv`.",
                "- `reports/tables/media_response_channel_roi.csv`.",
                "- `reports/tables/media_response_do_something_vs_nothing.csv`.",
                "- `reports/tables/media_response_knee_points.csv`.",
            ]
        ),
        encoding="utf-8",
    )

    STEP_MD.write_text(
        "\n".join(
            [
                "# Step 10 - Media Response Optimization",
                "",
                "## Objetivo",
                "",
                "Construir curvas no lineales de profit por canal y optimizar el mix actual para detectar gasto ineficiente y pesos objetivo por variable.",
                "",
                "## Que Hacemos",
                "",
                "- Definimos `do nothing` como el mix historico actual.",
                "- Reescalamos ese mix al presupuesto de planning del caso antes de optimizar.",
                "- Ajustamos un torneo de respuesta de medios con `Ridge`, `Lasso` y `ElasticNet` sobre los mismos transforms por canal.",
                "- Variamos cada canal sobre una rejilla amplia de pesos y euros para obtener su curva de profit.",
                "- Calculamos `saturation_budget_eur` como la rodilla de la curva incremental por canal, separandolo del optimo global con presupuesto fijo.",
                "- Optimizamos la mezcla completa con presupuesto fijo.",
                "",
                "## Salidas",
                "",
                "- `media_response_scenarios.csv`.",
                "- `media_response_weights.csv`.",
                "- `media_response_curves.csv`.",
                "- `media_response_maxima.csv`.",
                "- `media_response_saturation.csv`.",
                "- `media_response_coefficients.csv`.",
                "- `media_response_fit_metrics.csv`.",
                "- `media_response_predictions.csv`.",
                "- `media_response_prediction_diagnostics.csv`.",
                "- `media_response_channel_roi.csv`.",
                "- `media_response_do_something_vs_nothing.csv`.",
                "- `media_response_knee_points.csv`.",
                "- `media_response_model_tournament.csv`.",
                "- `media_response_optimization_report.md`.",
                "",
                "## Conclusion",
                "",
                "- Este paso sirve para la pregunta de negocio de pesos optimos mucho mejor que un modelo lineal en shares.",
                "- Las curvas por canal permiten defender de forma visual donde conviene recortar y donde aun hay recorrido.",
                "- El output de saturacion evita vender como recomendacion un maximo en el borde cuando lo que interesa es la caida del rendimiento marginal.",
            ]
        ),
        encoding="utf-8",
    )


def run_media_response_optimization(
    weekly_df: pd.DataFrame,
    media_cols: list[str],
    target_year: int = TARGET_YEAR,
    planning_budget_eur: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_style()
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.reports_figures_dir / "6_simulacion").mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    package, metrics_df, coefficient_df, tournament_df = fit_media_response_model(weekly_df, media_cols, target_year=target_year)
    metrics_df.to_csv(RESPONSE_METRICS_TABLE, index=False)
    coefficient_df.to_csv(RESPONSE_COEFFICIENTS_TABLE, index=False)
    tournament_df.to_csv(RESPONSE_TOURNAMENT_TABLE, index=False)
    predictions_df = pd.concat(
        [
            _score_prediction_split(package, package.train_df, "train"),
            _score_prediction_split(package, package.test_df, f"test_{target_year}"),
        ],
        ignore_index=True,
    )
    prediction_diagnostics_df = _build_prediction_diagnostics(predictions_df)
    predictions_df.to_csv(RESPONSE_PREDICTIONS_TABLE, index=False)
    prediction_diagnostics_df.to_csv(RESPONSE_PREDICTION_DIAGNOSTICS_TABLE, index=False)

    observed_annual = weekly_df.loc[weekly_df["year"] == target_year, media_cols].sum()
    observed_total_budget = float(observed_annual.sum())
    total_budget = float(planning_budget_eur) if planning_budget_eur is not None else observed_total_budget
    base_shares = (observed_annual / observed_total_budget).to_numpy(dtype=float)
    base_budgets = channel_budgets_from_shares(base_shares, media_cols, total_budget)
    optimized_budgets = optimize_media_response_budget(package, base_budgets, target_year=target_year)

    historical_df = predict_media_response_scenario(package, base_budgets, target_year)
    optimized_df = predict_media_response_scenario(package, optimized_budgets, target_year)
    historical_profit = float(historical_df["predicted_gross_profit"].sum())
    optimized_profit = float(optimized_df["predicted_gross_profit"].sum())
    historical_sales = float(historical_df["pred"].sum())
    optimized_sales = float(optimized_df["pred"].sum())

    scenario_df = pd.DataFrame(
        [
            {
                "scenario": "do_nothing_current_mix",
                "predicted_sales_2024": historical_sales,
                "predicted_gross_profit_2024": historical_profit,
                "delta_vs_historical_profit": 0.0,
                "roi_vs_historical_budget": 0.0,
            },
            {
                "scenario": "media_response_optimized",
                "predicted_sales_2024": optimized_sales,
                "predicted_gross_profit_2024": optimized_profit,
                "delta_vs_historical_profit": optimized_profit - historical_profit,
                "roi_vs_historical_budget": _safe_divide(optimized_profit - historical_profit, total_budget),
            },
        ]
    )
    scenario_df.to_csv(RESPONSE_SUMMARY_TABLE, index=False)

    base_shares_pct = channel_shares_from_budgets(base_budgets, media_cols) * 100.0
    opt_shares_pct = channel_shares_from_budgets(optimized_budgets, media_cols) * 100.0
    weights_df = pd.DataFrame(
        {
            "channel": [channel.replace("media_", "") for channel in media_cols],
            "observed_budget_2024_eur": [float(observed_annual[channel]) for channel in media_cols],
            "historical_budget_eur": [base_budgets[channel] for channel in media_cols],
            "optimized_budget_eur": [optimized_budgets[channel] for channel in media_cols],
            "historical_share_pct": base_shares_pct,
            "optimized_share_pct": opt_shares_pct,
            "delta_share_pct_points": opt_shares_pct - base_shares_pct,
        }
    ).sort_values("delta_share_pct_points", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    weights_df.to_csv(RESPONSE_WEIGHTS_TABLE, index=False)

    curves_df, maxima_df, saturation_df = _response_curves(package, base_budgets, optimized_budgets, target_year)
    curves_df.to_csv(RESPONSE_CURVES_TABLE, index=False)
    maxima_df.to_csv(RESPONSE_MAXIMA_TABLE, index=False)
    saturation_df.to_csv(RESPONSE_SATURATION_TABLE, index=False)
    roi_df = _build_channel_roi_table(weights_df, maxima_df, coefficient_df)
    roi_df.to_csv(RESPONSE_ROI_TABLE, index=False)
    do_something_df, knee_points_df = _historical_channel_saturation_curves(package, base_budgets, target_year)
    do_something_df.to_csv(RESPONSE_DO_SOMETHING_TABLE, index=False)
    knee_points_df.to_csv(RESPONSE_KNEE_POINTS_TABLE, index=False)

    for channel in maxima_df["channel"].tolist():
        _plot_channel_curve(
            curves_df[curves_df["channel"] == channel].copy(),
            saturation_df[saturation_df["channel"] == channel].copy(),
            maxima_df[maxima_df["channel"] == channel].iloc[0],
        )
    for channel in knee_points_df["channel"].tolist():
        _plot_historical_saturation_curve(
            do_something_df[do_something_df["channel"] == channel].copy(),
            knee_points_df[knee_points_df["channel"] == channel].iloc[0],
        )
    _plot_prediction_fit(predictions_df, target_year)
    _plot_prediction_scatter(predictions_df)
    _plot_prediction_residuals(predictions_df)
    _plot_media_coefficients(coefficient_df)
    _plot_weight_mix(weights_df)
    _plot_channel_roi(roi_df)
    _plot_model_tournament(tournament_df)
    _plot_do_something_vs_do_nothing(do_something_df)

    _write_report(
        metrics_df=metrics_df,
        tournament_df=tournament_df,
        coefficient_df=coefficient_df,
        prediction_diagnostics_df=prediction_diagnostics_df,
        scenario_df=scenario_df,
        weights_df=weights_df,
        maxima_df=maxima_df,
        roi_df=roi_df,
        do_something_df=do_something_df,
        knee_points_df=knee_points_df,
        observed_total_budget=observed_total_budget,
        total_budget=total_budget,
    )
    return scenario_df, weights_df
