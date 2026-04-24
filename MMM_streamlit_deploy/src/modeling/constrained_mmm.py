from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.features.geo_dataset_builder import build_geo_model_dataset
from src.modeling.specs import CONTROL_COLUMNS, TRANSFORM_GRID
from src.modeling.trainer import media_columns
from src.modeling.transforms import add_transformed_media_features
from src.validation.backtesting import panel_time_cv_indices


RESULTS_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_model_comparison.csv"
GRID_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_selection_grid.csv"
COEFFICIENT_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_coefficients.csv"
CHANNEL_STABILITY_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_channel_stability.csv"
ELIGIBILITY_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_channel_eligibility.csv"
CITY_METRICS_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_city_metrics.csv"
BENCHMARK_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_benchmarks.csv"
BOOTSTRAP_TABLE = CONFIG.reports_tables_dir / "constrained_mmm_bootstrap_stability.csv"
REPORT_MD = CONFIG.docs_dir / "constrained_mmm_report.md"

TARGET_YEAR = 2024
DEFAULT_TRAIN_START_YEAR = 2021
RIDGE_ALPHA_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]
BASELINE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0]
BOOTSTRAP_ITERATIONS = 100
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class FittedConstrainedModel:
    intercept: float
    coefficients: pd.Series
    feature_columns: list[str]
    media_feature_columns: list[str]
    transform_spec: dict[str, float | int | str]
    ridge_alpha: float


def _load_geo_panel() -> pd.DataFrame:
    if not CONFIG.geo_model_dataset_file.exists():
        build_geo_model_dataset()
    df = pd.read_parquet(CONFIG.geo_model_dataset_file).copy()
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"])
    city_dummies = pd.get_dummies(df["ciudad"], prefix="city", drop_first=True, dtype=float)
    out = pd.concat([df.reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)
    return out.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)


def _weekly_metrics(scored: pd.DataFrame) -> dict[str, float]:
    weekly = (
        scored.groupby("semana_inicio", as_index=False)
        .agg(actual=("ventas_netas", "sum"), pred=("pred", "sum"))
        .sort_values("semana_inicio")
        .reset_index(drop=True)
    )
    error = weekly["pred"] - weekly["actual"]
    denom = np.where(np.abs(weekly["actual"]) > 1e-9, np.abs(weekly["actual"]), 1.0)
    return {
        "mape": float(np.mean(np.abs(error) / denom) * 100.0),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.square(error).sum() / np.square(weekly["actual"] - weekly["actual"].mean()).sum()),
    }


def _fit_constrained_ridge(
    x: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    media_feature_columns: list[str],
    ridge_alpha: float,
) -> tuple[float, np.ndarray]:
    feature_index = {column: idx for idx, column in enumerate(feature_columns)}
    positive_indices = {feature_index[column] for column in media_feature_columns}
    intercept_column = np.ones((x.shape[0], 1), dtype=float)
    design = np.column_stack([intercept_column, x])
    penalty = np.sqrt(ridge_alpha) * np.eye(x.shape[1], dtype=float)
    penalty_with_intercept = np.column_stack([np.zeros((x.shape[1], 1), dtype=float), penalty])
    augmented_x = np.vstack([design, penalty_with_intercept])
    augmented_y = np.concatenate([y, np.zeros(x.shape[1], dtype=float)])

    lower_bounds = np.full(x.shape[1] + 1, -np.inf, dtype=float)
    upper_bounds = np.full(x.shape[1] + 1, np.inf, dtype=float)
    for idx in positive_indices:
        lower_bounds[idx + 1] = 0.0

    result = lsq_linear(
        augmented_x,
        augmented_y,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        lsmr_tol="auto",
        max_iter=5000,
    )
    if not result.success:
        raise RuntimeError(f"Constrained ridge optimization failed: {result.message}")
    return float(result.x[0]), np.asarray(result.x[1:], dtype=float)


def _original_scale(
    intercept_scaled: float,
    coef_scaled: np.ndarray,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> tuple[float, pd.Series]:
    coef_original = coef_scaled / scaler.scale_
    coef_series = pd.Series(coef_original, index=feature_columns, dtype=float)
    intercept_original = float(intercept_scaled - np.sum(scaler.mean_ * coef_original))
    return intercept_original, coef_series


def _score_candidate(
    train_df: pd.DataFrame,
    city_cols: list[str],
    media_cols: list[str],
    transform_spec: dict[str, float | int | str],
    ridge_alpha: float,
) -> dict[str, float | int | str]:
    params = {media_col: transform_spec.copy() for media_col in media_cols}
    transformed = add_transformed_media_features(train_df, media_cols, params)
    media_feature_columns = sorted([column for column in transformed.columns if column.startswith("media_") and "__lag" in column])
    feature_columns = CONTROL_COLUMNS + city_cols + media_feature_columns
    cv_indices = panel_time_cv_indices(transformed[["semana_inicio"]].copy(), n_splits=3)

    fold_metrics = []
    fold_negative_media = []
    for train_idx, valid_idx in cv_indices:
        fold_train = transformed.iloc[train_idx].copy()
        fold_valid = transformed.iloc[valid_idx].copy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(fold_train[feature_columns])
        x_valid = scaler.transform(fold_valid[feature_columns])
        intercept_scaled, coef_scaled = _fit_constrained_ridge(
            x=x_train,
            y=fold_train["ventas_netas"].to_numpy(dtype=float),
            feature_columns=feature_columns,
            media_feature_columns=media_feature_columns,
            ridge_alpha=ridge_alpha,
        )
        intercept, coefficients = _original_scale(intercept_scaled, coef_scaled, scaler, feature_columns)
        fold_valid = fold_valid[["semana_inicio", "ventas_netas"]].copy()
        fold_valid["pred"] = intercept + transformed.iloc[valid_idx][feature_columns].mul(coefficients, axis=1).sum(axis=1).to_numpy()
        fold_metrics.append(_weekly_metrics(fold_valid))
        media_coef = coefficients[media_feature_columns]
        fold_negative_media.append(int((media_coef < -1e-8).sum()))

    return {
        "lag": int(transform_spec["lag"]),
        "alpha": float(transform_spec["alpha"]),
        "saturation": str(transform_spec["saturation"]),
        "ridge_alpha": float(ridge_alpha),
        "cv_mean_mape": float(np.mean([metric["mape"] for metric in fold_metrics])),
        "cv_mean_rmse": float(np.mean([metric["rmse"] for metric in fold_metrics])),
        "cv_mean_bias": float(np.mean([metric["bias"] for metric in fold_metrics])),
        "cv_max_negative_media_coefficients": int(max(fold_negative_media)),
    }


def _select_best_spec(
    train_df: pd.DataFrame,
    city_cols: list[str],
    media_cols: list[str],
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    rows = []
    for transform_spec in TRANSFORM_GRID:
        for ridge_alpha in RIDGE_ALPHA_GRID:
            rows.append(
                _score_candidate(
                    train_df=train_df,
                    city_cols=city_cols,
                    media_cols=media_cols,
                    transform_spec=transform_spec,
                    ridge_alpha=float(ridge_alpha),
                )
            )
    diagnostics = pd.DataFrame(rows).sort_values(
        ["cv_max_negative_media_coefficients", "cv_mean_mape", "cv_mean_rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = diagnostics.iloc[0].to_dict()
    return {
        "lag": int(best["lag"]),
        "alpha": float(best["alpha"]),
        "saturation": str(best["saturation"]),
        "ridge_alpha": float(best["ridge_alpha"]),
    }, diagnostics


def _fit_final_model(
    dataset: pd.DataFrame,
    train_start_year: int,
    spec: dict[str, float | int | str],
) -> tuple[FittedConstrainedModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    media_cols = media_columns(dataset)
    city_cols = sorted([column for column in dataset.columns if column.startswith("city_")])
    params = {
        media_col: {
            "lag": int(spec["lag"]),
            "alpha": float(spec["alpha"]),
            "saturation": str(spec["saturation"]),
        }
        for media_col in media_cols
    }
    transformed = add_transformed_media_features(dataset, media_cols, params)
    media_feature_columns = sorted([column for column in transformed.columns if column.startswith("media_") and "__lag" in column])
    feature_columns = CONTROL_COLUMNS + city_cols + media_feature_columns
    train_df = transformed.loc[(transformed["year"] >= train_start_year) & (transformed["year"] < TARGET_YEAR)].copy()
    test_df = transformed.loc[transformed["year"] == TARGET_YEAR].copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    intercept_scaled, coef_scaled = _fit_constrained_ridge(
        x=x_train,
        y=train_df["ventas_netas"].to_numpy(dtype=float),
        feature_columns=feature_columns,
        media_feature_columns=media_feature_columns,
        ridge_alpha=float(spec["ridge_alpha"]),
    )
    intercept, coefficients = _original_scale(intercept_scaled, coef_scaled, scaler, feature_columns)

    scored = test_df[["semana_inicio", "ciudad", "ventas_netas"]].copy().reset_index(drop=True)
    scored["pred"] = intercept + test_df[feature_columns].mul(coefficients, axis=1).sum(axis=1).to_numpy()
    contributions = test_df[feature_columns].mul(coefficients, axis=1)
    baseline_columns = [column for column in feature_columns if column not in media_feature_columns]
    baseline = test_df[["semana_inicio", "ciudad", "ventas_netas"]].copy().reset_index(drop=True)
    baseline["baseline_pred"] = intercept + contributions[baseline_columns].sum(axis=1).to_numpy()

    model = FittedConstrainedModel(
        intercept=intercept,
        coefficients=coefficients,
        feature_columns=feature_columns,
        media_feature_columns=media_feature_columns,
        transform_spec=spec,
        ridge_alpha=float(spec["ridge_alpha"]),
    )
    return model, test_df, scored, baseline


def _channel_stability(
    train_df: pd.DataFrame,
    city_cols: list[str],
    media_cols: list[str],
    best_spec: dict[str, float | int | str],
) -> pd.DataFrame:
    params = {
        media_col: {
            "lag": int(best_spec["lag"]),
            "alpha": float(best_spec["alpha"]),
            "saturation": str(best_spec["saturation"]),
        }
        for media_col in media_cols
    }
    transformed = add_transformed_media_features(train_df, media_cols, params)
    media_feature_columns = sorted([column for column in transformed.columns if column.startswith("media_") and "__lag" in column])
    feature_columns = CONTROL_COLUMNS + city_cols + media_feature_columns
    cv_indices = panel_time_cv_indices(transformed[["semana_inicio"]].copy(), n_splits=3)

    rows = []
    for fold_number, (fold_train_idx, _fold_valid_idx) in enumerate(cv_indices, start=1):
        fold_train = transformed.iloc[fold_train_idx].copy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(fold_train[feature_columns])
        intercept_scaled, coef_scaled = _fit_constrained_ridge(
            x=x_train,
            y=fold_train["ventas_netas"].to_numpy(dtype=float),
            feature_columns=feature_columns,
            media_feature_columns=media_feature_columns,
            ridge_alpha=float(best_spec["ridge_alpha"]),
        )
        _intercept, coefficients = _original_scale(intercept_scaled, coef_scaled, scaler, feature_columns)
        for feature in media_feature_columns:
            rows.append(
                {
                    "fold": fold_number,
                    "channel": feature.split("__")[0].replace("media_", "", 1),
                    "feature": feature,
                    "coefficient": float(coefficients[feature]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["channel", "feature"], as_index=False).agg(
        coefficient_mean=("coefficient", "mean"),
        coefficient_min=("coefficient", "min"),
        coefficient_max=("coefficient", "max"),
        positive_folds=("coefficient", lambda values: int((pd.Series(values) > 1e-9).sum())),
    )
    summary["sign_stable_positive"] = summary["positive_folds"] == detail["fold"].nunique()
    return summary.sort_values("coefficient_mean", ascending=False).reset_index(drop=True)


def _channel_tables(
    model: FittedConstrainedModel,
    test_df: pd.DataFrame,
    baseline: pd.DataFrame,
    stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    media_rows = []
    aggregate_rows = []
    feature_map = {feature.split("__")[0]: feature for feature in model.media_feature_columns}
    scored = baseline.copy()
    scored["pred"] = model.intercept + test_df[model.feature_columns].mul(model.coefficients, axis=1).sum(axis=1).to_numpy()

    for media_col, feature in feature_map.items():
        channel = media_col.replace("media_", "", 1)
        contribution = test_df[feature] * float(model.coefficients[feature])
        channel_df = pd.DataFrame(
            {
                "semana_inicio": test_df["semana_inicio"].to_numpy(),
                "channel": channel,
                "contribution": contribution.to_numpy(dtype=float),
                "spend_eur": test_df[media_col].to_numpy(dtype=float),
            }
        )
        weekly_channel = channel_df.groupby(["semana_inicio", "channel"], as_index=False).sum()
        media_rows.append(weekly_channel)
        spend_total = float(test_df[media_col].sum())
        contribution_total = float(contribution.sum())
        stability_row = stability.loc[stability["channel"] == channel].iloc[0].to_dict()
        aggregate_rows.append(
            {
                "channel": channel,
                "feature": feature,
                "contribution_mean": contribution_total,
                "roi_mean": contribution_total / spend_total if spend_total > 1e-9 else 0.0,
                "historical_spend_2024_eur": spend_total,
                "coefficient_mean": float(stability_row["coefficient_mean"]),
                "coefficient_min": float(stability_row["coefficient_min"]),
                "coefficient_max": float(stability_row["coefficient_max"]),
                "positive_folds": int(stability_row["positive_folds"]),
                "sign_stable_positive": bool(stability_row["sign_stable_positive"]),
                "optimization_status": (
                    "eligible"
                    if bool(stability_row["sign_stable_positive"]) and contribution_total > 0.0
                    else "hold"
                ),
            }
        )

    weekly_contributions = pd.concat(media_rows, ignore_index=True).sort_values(["semana_inicio", "channel"]).reset_index(drop=True)
    aggregate = pd.DataFrame(aggregate_rows).sort_values("contribution_mean", ascending=False).reset_index(drop=True)
    weekly_predictions = (
        scored.groupby("semana_inicio", as_index=False)
        .agg(
            actual_sales=("ventas_netas", "sum"),
            predicted_sales=("pred", "sum"),
            baseline_sales=("baseline_pred", "sum"),
        )
        .sort_values("semana_inicio")
        .reset_index(drop=True)
    )
    weekly_predictions["media_incremental"] = weekly_predictions["predicted_sales"] - weekly_predictions["baseline_sales"]
    return weekly_contributions, aggregate, weekly_predictions


def _city_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, city_df in scored.groupby("ciudad"):
        metrics = _weekly_metrics(city_df[["semana_inicio", "ventas_netas", "pred"]])
        rows.append({"ciudad": city, **metrics})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def _bootstrap_stability(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    city_cols: list[str],
    media_cols: list[str],
    best_spec: dict[str, float | int | str],
) -> pd.DataFrame:
    params = {
        media_col: {
            "lag": int(best_spec["lag"]),
            "alpha": float(best_spec["alpha"]),
            "saturation": str(best_spec["saturation"]),
        }
        for media_col in media_cols
    }
    transformed_train = add_transformed_media_features(train_df, media_cols, params)
    transformed_test = add_transformed_media_features(test_df, media_cols, params)
    media_feature_columns = sorted([column for column in transformed_train.columns if column.startswith("media_") and "__lag" in column])
    feature_columns = CONTROL_COLUMNS + city_cols + media_feature_columns
    weeks = transformed_train["semana_inicio"].drop_duplicates().sort_values().tolist()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    detail_rows = []
    for bootstrap_idx in range(BOOTSTRAP_ITERATIONS):
        sampled_weeks = rng.choice(weeks, size=len(weeks), replace=True)
        sample_df = pd.concat(
            [transformed_train.loc[transformed_train["semana_inicio"] == week] for week in sampled_weeks],
            ignore_index=True,
        )
        scaler = StandardScaler()
        x_sample = scaler.fit_transform(sample_df[feature_columns])
        intercept_scaled, coef_scaled = _fit_constrained_ridge(
            x=x_sample,
            y=sample_df["ventas_netas"].to_numpy(dtype=float),
            feature_columns=feature_columns,
            media_feature_columns=media_feature_columns,
            ridge_alpha=float(best_spec["ridge_alpha"]),
        )
        _intercept, coefficients = _original_scale(intercept_scaled, coef_scaled, scaler, feature_columns)
        contribution_matrix = transformed_test[media_feature_columns].mul(coefficients[media_feature_columns], axis=1)
        total_incremental = float(contribution_matrix.sum().sum())
        for feature in media_feature_columns:
            detail_rows.append(
                {
                    "bootstrap": bootstrap_idx,
                    "channel": feature.split("__")[0].replace("media_", "", 1),
                    "selected": float(coefficients[feature] > 1e-8),
                    "coefficient": float(coefficients[feature]),
                    "contribution_share": (
                        float(contribution_matrix[feature].sum() / total_incremental)
                        if total_incremental > 1e-9
                        else 0.0
                    ),
                }
            )

    detail = pd.DataFrame(detail_rows)
    return detail.groupby("channel", as_index=False).agg(
        bootstrap_selection_rate=("selected", "mean"),
        bootstrap_coef_median=("coefficient", "median"),
        bootstrap_coef_p05=("coefficient", lambda values: float(np.quantile(values, 0.05))),
        bootstrap_coef_p95=("coefficient", lambda values: float(np.quantile(values, 0.95))),
        bootstrap_share_median=("contribution_share", "median"),
        bootstrap_share_p05=("contribution_share", lambda values: float(np.quantile(values, 0.05))),
        bootstrap_share_p95=("contribution_share", lambda values: float(np.quantile(values, 0.95))),
    ).sort_values(["bootstrap_selection_rate", "bootstrap_share_median"], ascending=[False, False]).reset_index(drop=True)


def _baseline_only_benchmark(
    dataset: pd.DataFrame,
    train_start_year: int,
) -> dict[str, float]:
    city_cols = sorted([column for column in dataset.columns if column.startswith("city_")])
    feature_columns = CONTROL_COLUMNS + city_cols
    train_df = dataset.loc[(dataset["year"] >= train_start_year) & (dataset["year"] < TARGET_YEAR)].copy()
    test_df = dataset.loc[dataset["year"] == TARGET_YEAR].copy()
    cv_indices = panel_time_cv_indices(train_df[["semana_inicio"]].copy(), n_splits=3)

    cv_rows = []
    for alpha in BASELINE_ALPHA_GRID:
        fold_metrics = []
        for train_idx, valid_idx in cv_indices:
            fold_train = train_df.iloc[train_idx].copy()
            fold_valid = train_df.iloc[valid_idx].copy()
            scaler = StandardScaler()
            x_train = scaler.fit_transform(fold_train[feature_columns])
            x_valid = scaler.transform(fold_valid[feature_columns])
            model = Ridge(alpha=float(alpha))
            model.fit(x_train, fold_train["ventas_netas"].to_numpy(dtype=float))
            fold_scored = fold_valid[["semana_inicio", "ventas_netas"]].copy()
            fold_scored["pred"] = model.predict(x_valid)
            fold_metrics.append(_weekly_metrics(fold_scored))
        cv_rows.append(
            {
                "model": "BaselineOnlyRidge",
                "alpha": float(alpha),
                "cv_mean_mape": float(np.mean([metric["mape"] for metric in fold_metrics])),
                "cv_mean_rmse": float(np.mean([metric["rmse"] for metric in fold_metrics])),
            }
        )
    cv_table = pd.DataFrame(cv_rows).sort_values(["cv_mean_mape", "cv_mean_rmse"]).reset_index(drop=True)
    best_alpha = float(cv_table.iloc[0]["alpha"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    x_test = scaler.transform(test_df[feature_columns])
    model = Ridge(alpha=best_alpha)
    model.fit(x_train, train_df["ventas_netas"].to_numpy(dtype=float))
    scored = test_df[["semana_inicio", "ventas_netas"]].copy()
    scored["pred"] = model.predict(x_test)
    metrics = _weekly_metrics(scored)
    return {
        "model": "BaselineOnlyRidge",
        "alpha": best_alpha,
        **metrics,
    }


def _write_report(
    results: dict,
    spec: dict[str, float | int | str],
    eligibility: pd.DataFrame,
    city_metrics: pd.DataFrame,
    benchmarks: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    top_channels = eligibility.loc[eligibility["contribution_mean"] > 1.0, "channel"].head(4).tolist()
    stable_channels = eligibility.loc[eligibility["sign_stable_positive"], "channel"].tolist()
    eligible_channels = eligibility.loc[eligibility["optimization_status"] == "eligible", "channel"].tolist()
    lines = [
        "# Constrained MMM Report",
        "",
        "## Objetivo",
        "",
        "Construir un MMM operativo mas defendible que el ARIMAX global y mas estable que el bayesiano actual cuando la senal no da para identificar bien transforms por canal.",
        "",
        "## Configuracion",
        "",
        f"- Ventana de entrenamiento principal: `{results['train_start_year']}`-`{TARGET_YEAR - 1}`.",
        f"- Test holdout: `{TARGET_YEAR}`.",
        f"- Transformacion comun de medios: `lag={spec['lag']}, alpha={spec['alpha']}, saturation={spec['saturation']}`.",
        f"- Ridge alpha seleccionado por CV: `{spec['ridge_alpha']}`.",
        "",
        "## Resultados Test 2024",
        "",
        "```text",
        pd.DataFrame([results["test_metrics"]]).round(4).to_string(index=False),
        "```",
        "",
        "## Benchmarks",
        "",
        "```text",
        benchmarks.round(4).to_string(index=False),
        "```",
        "",
        "## Bootstrap Stability",
        "",
        "```text",
        bootstrap.round(4).to_string(index=False),
        "```",
        "",
        "## Lectura",
        "",
        "- El modelo impone signo no negativo solo a medios; los controles siguen siendo libres.",
        "- No intenta vender delays distintos por canal cuando el dataset no los identifica con claridad; usa una transformacion comun validada por CV.",
        f"- Canales con mayor contribucion 2024: `{', '.join(top_channels) if top_channels else 'ninguno'}`.",
        f"- Canales elegibles para optimizacion directa: `{', '.join(eligible_channels) if eligible_channels else 'ninguno'}`.",
        f"- Canales con signo estable en todos los folds: `{', '.join(stable_channels) if stable_channels else 'ninguno'}`.",
        "- La tabla bootstrap muestra si un canal reaparece de forma consistente al re-muestrear semanas de entrenamiento; esto es el mejor proxy actual de fiabilidad del peso.",
        "- El benchmark sin medios se incluye para verificar si los canales realmente anaden senal incremental o si el baseline ya explica casi todo el holdout.",
        "- Este modelo no da intervalos bayesianos; su lectura de robustez se apoya en estabilidad temporal por folds y guardrails de signo.",
        "",
        "## City Metrics",
        "",
        "```text",
        city_metrics.round(4).to_string(index=False),
        "```",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_constrained_mmm(train_start_year: int = DEFAULT_TRAIN_START_YEAR) -> dict:
    dataset = _load_geo_panel()
    media_cols = media_columns(dataset)
    city_cols = sorted([column for column in dataset.columns if column.startswith("city_")])
    train_df = dataset.loc[(dataset["year"] >= train_start_year) & (dataset["year"] < TARGET_YEAR)].copy()
    best_spec, grid = _select_best_spec(train_df=train_df, city_cols=city_cols, media_cols=media_cols)
    model, test_df, scored, baseline = _fit_final_model(dataset=dataset, train_start_year=train_start_year, spec=best_spec)
    test_metrics = _weekly_metrics(scored[["semana_inicio", "ventas_netas", "pred"]])
    stability = _channel_stability(train_df=train_df, city_cols=city_cols, media_cols=media_cols, best_spec=best_spec)
    weekly_contributions, eligibility, weekly_predictions = _channel_tables(model=model, test_df=test_df, baseline=baseline, stability=stability)
    city_metrics = _city_metrics(scored)
    baseline_benchmark = _baseline_only_benchmark(dataset=dataset, train_start_year=train_start_year)
    bootstrap = _bootstrap_stability(
        train_df=train_df,
        test_df=test_df,
        city_cols=city_cols,
        media_cols=media_cols,
        best_spec=best_spec,
    )
    eligibility = eligibility.merge(bootstrap, on="channel", how="left")
    eligibility["optimization_status"] = np.where(
        (eligibility["sign_stable_positive"]) & (eligibility["bootstrap_selection_rate"] >= 0.60) & (eligibility["contribution_mean"] > 0.0),
        "eligible",
        "hold",
    )

    results = {
        "model_name": "ConstrainedMMM",
        "train_start_year": train_start_year,
        "target_year": TARGET_YEAR,
        "test_metrics": test_metrics,
        "transform_spec": {
            "lag": int(best_spec["lag"]),
            "alpha": float(best_spec["alpha"]),
            "saturation": str(best_spec["saturation"]),
            "ridge_alpha": float(best_spec["ridge_alpha"]),
        },
    }
    benchmark_table = pd.DataFrame(
        [
            {"model": "ConstrainedMMM", "alpha": float(best_spec["ridge_alpha"]), **results["test_metrics"]},
            baseline_benchmark,
        ]
    )

    CONFIG.constrained_model_results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    baseline.to_csv(CONFIG.constrained_baseline_file, index=False)
    scored.to_csv(CONFIG.constrained_weekly_predictions_file, index=False)
    weekly_predictions.to_csv(CONFIG.constrained_weekly_summary_file, index=False)
    weekly_contributions.to_csv(CONFIG.constrained_contributions_file, index=False)
    grid.to_csv(CONFIG.constrained_cv_diagnostics_file, index=False)
    pd.DataFrame([results["test_metrics"]]).to_csv(RESULTS_TABLE, index=False)
    grid.to_csv(GRID_TABLE, index=False)
    pd.DataFrame(
        {
            "feature": model.coefficients.index,
            "coefficient": model.coefficients.values,
            "feature_family": [
                "media" if feature in model.media_feature_columns else "control"
                for feature in model.coefficients.index
            ],
        }
    ).sort_values(["feature_family", "coefficient"], ascending=[True, False]).to_csv(COEFFICIENT_TABLE, index=False)
    stability.to_csv(CHANNEL_STABILITY_TABLE, index=False)
    eligibility.to_csv(ELIGIBILITY_TABLE, index=False)
    city_metrics.to_csv(CITY_METRICS_TABLE, index=False)
    benchmark_table.to_csv(BENCHMARK_TABLE, index=False)
    bootstrap.to_csv(BOOTSTRAP_TABLE, index=False)
    _write_report(
        results=results,
        spec=best_spec,
        eligibility=eligibility,
        city_metrics=city_metrics,
        benchmarks=benchmark_table,
        bootstrap=bootstrap,
    )
    return {
        "results": results,
        "selection_grid": grid,
        "channel_eligibility": eligibility,
    }
