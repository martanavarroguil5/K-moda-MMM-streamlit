from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.modeling.beta_utils import feature_block, summarize_beta_stability, summarize_channel_stability, usability_flag_row
from src.modeling.multicollinearity import filter_beta_collinear_features
from src.modeling.selection import prepare_transformed_dataset, select_media_transforms
from src.modeling.specs import (
    BETA_ELASTIC_NET_ALPHAS,
    BETA_ELASTIC_NET_L1,
    BETA_TRANSFORM_GRID,
    CONTROL_COLUMNS,
    RANDOM_STATE,
)
from src.modeling.trainer import (
    build_feature_frame,
    city_dummy_columns,
    ensure_prerequisites,
    fit_elastic_net,
    fit_elastic_net_with_params,
    load_dataset,
    media_columns,
    original_scale_coefficients,
    original_scale_intercept,
    predict_with_model,
    standardized_coefficients,
)
from src.modeling.transforms import transformed_feature_name
from src.validation.backtesting import panel_time_cv_indices


SPEC_NAME = "ElasticNetTransformedMedia"
TRAIN_START_YEAR = 2021
BOOTSTRAP_ITERATIONS = 80
BETA_REPORT_MD = CONFIG.docs_dir / "elasticnet_beta_report.md"
BETA_RESULTS_JSON = CONFIG.processed_dir / "elasticnet_beta_results.json"
BETA_MODEL_FILE = CONFIG.processed_dir / "elasticnet_beta_model.pkl"
BETA_TRANSFORMS_JSON = CONFIG.processed_dir / "elasticnet_beta_selected_transforms.json"
BETA_BACKTEST_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_backtest.csv"
BETA_TEST_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_test_metrics.csv"
BETA_PREDICTIONS_TABLE = CONFIG.processed_dir / "elasticnet_beta_predictions.csv"
BETA_CONTRIBUTIONS_TABLE = CONFIG.processed_dir / "elasticnet_beta_channel_contributions.csv"
BETA_FOLD_COEFFICIENTS_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_fold_coefficients.csv"
BETA_SUMMARY_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_summary.csv"
BETA_DEPLOYMENT_COEFFICIENTS_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_deployment_coefficients.csv"
BETA_TRANSFORM_DIAGNOSTICS_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_transform_diagnostics.csv"
BETA_HYPERPARAM_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_hyperparameter_search.csv"
BETA_BOOTSTRAP_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_bootstrap_coefficients.csv"
BETA_COLLINEARITY_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_collinearity_filter.csv"
BETA_CHANNEL_TABLE = CONFIG.reports_tables_dir / "elasticnet_beta_channel_summary.csv"


@dataclass
class TwoStageBetaPackage:
    baseline_scaler: object
    baseline_model: object
    media_scaler: object | None
    media_model: object | None
    feature_columns: list[str]
    baseline_feature_columns: list[str]
    media_feature_columns: list[str]
    control_columns: list[str]
    media_params: dict[str, dict[str, float | int | str]]
    city_dummy_columns: list[str]
    spec_name: str
    original_coefficients: pd.Series
    standardized_coefficients: pd.Series
    intercept: float
    alpha: float
    l1_ratio: float


def _beta_validation_years(df: pd.DataFrame, train_start_year: int, test_year: int) -> list[int]:
    years = sorted(df["year"].unique().tolist())
    return [year for year in years if train_start_year < year < test_year]


def _prepare_beta_design(
    df: pd.DataFrame,
    control_cols: list[str],
    city_cols: list[str],
    media_cols: list[str],
    media_params: dict[str, dict[str, float | int | str]],
) -> tuple[pd.DataFrame, list[str]]:
    working_df, media_feature_columns = prepare_transformed_dataset(df, media_cols, media_params)
    feature_columns = list(
        build_feature_frame(
            working_df,
            control_cols,
            city_cols,
            media_feature_columns=media_feature_columns,
        ).columns
    )
    return working_df, feature_columns


def _fit_two_stage_linear(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    media_feature_columns: list[str],
    alpha: float,
    l1_ratio: float,
) -> tuple[object, object, object | None, object | None, pd.Series, pd.Series, float]:
    baseline_feature_columns = [feature for feature in feature_columns if feature not in set(media_feature_columns)]
    combined_original = pd.Series(0.0, index=feature_columns, dtype=float)
    combined_standardized = pd.Series(0.0, index=feature_columns, dtype=float)

    baseline_scaler, baseline_model = fit_elastic_net(train_df, baseline_feature_columns)
    baseline_original = original_scale_coefficients(baseline_scaler, baseline_model, baseline_feature_columns)
    baseline_standardized = standardized_coefficients(baseline_model, baseline_feature_columns)
    combined_original.loc[baseline_feature_columns] = baseline_original.reindex(baseline_feature_columns).to_numpy(dtype=float)
    combined_standardized.loc[baseline_feature_columns] = baseline_standardized.reindex(baseline_feature_columns).to_numpy(
        dtype=float
    )
    intercept = original_scale_intercept(baseline_scaler, baseline_model, baseline_feature_columns)

    media_scaler = None
    media_model = None
    if media_feature_columns:
        media_df = train_df.copy()
        media_df["_beta_media_target"] = (
            train_df["ventas_netas"].to_numpy(dtype=float)
            - predict_with_model(train_df, baseline_feature_columns, baseline_scaler, baseline_model)
        )
        media_scaler, media_model = fit_elastic_net_with_params(
            media_df,
            media_feature_columns,
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            target_col="_beta_media_target",
            positive=True,
        )
        media_original = original_scale_coefficients(media_scaler, media_model, media_feature_columns)
        media_standardized = standardized_coefficients(media_model, media_feature_columns)
        combined_original.loc[media_feature_columns] = media_original.reindex(media_feature_columns).to_numpy(dtype=float)
        combined_standardized.loc[media_feature_columns] = media_standardized.reindex(media_feature_columns).to_numpy(
            dtype=float
        )
        intercept += original_scale_intercept(media_scaler, media_model, media_feature_columns)

    return (
        baseline_scaler,
        baseline_model,
        media_scaler,
        media_model,
        combined_original,
        combined_standardized,
        float(intercept),
    )


def _package_predict(df: pd.DataFrame, package: TwoStageBetaPackage) -> np.ndarray:
    working = df.copy()
    if package.media_params:
        working, _ = prepare_transformed_dataset(working, media_columns(working), package.media_params)
    return (
        package.intercept
        + working[package.feature_columns]
        .mul(package.original_coefficients.reindex(package.feature_columns), axis=1)
        .sum(axis=1)
        .to_numpy(dtype=float)
    )


def _package_contributions(df: pd.DataFrame, package: TwoStageBetaPackage) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if package.media_params:
        working, _ = prepare_transformed_dataset(working, media_columns(working), package.media_params)
    contributions = working[package.feature_columns].mul(
        package.original_coefficients.reindex(package.feature_columns),
        axis=1,
    )
    return working, contributions


def _coefficient_frame(
    fold_name: str,
    package: TwoStageBetaPackage,
    city_cols: list[str],
) -> pd.DataFrame:
    coefficient_df = pd.DataFrame(
        {
            "fold": fold_name,
            "feature": package.feature_columns,
            "standardized_coefficient": package.standardized_coefficients.reindex(package.feature_columns).to_numpy(dtype=float),
            "coefficient": package.original_coefficients.reindex(package.feature_columns).to_numpy(dtype=float),
        }
    )
    coefficient_df["feature_block"] = coefficient_df["feature"].map(lambda feature: feature_block(feature, city_cols))
    coefficient_df["channel"] = coefficient_df["feature"].where(
        coefficient_df["feature"].str.startswith("media_"),
        None,
    )
    coefficient_df["channel"] = coefficient_df["channel"].str.split("__").str[0]
    coefficient_df["alpha"] = float(package.alpha)
    coefficient_df["l1_ratio"] = float(package.l1_ratio)
    coefficient_df["coefficient_sign"] = coefficient_df["coefficient"].map(
        lambda value: "positive" if value > 1e-8 else "negative" if value < -1e-8 else "zero"
    )
    return coefficient_df


def _summarize_coefficient_stability(
    coefficients_df: pd.DataFrame,
    media_feature_columns: list[str],
) -> dict[str, float]:
    rows = []
    for feature in media_feature_columns:
        feature_rows = coefficients_df[coefficients_df["feature"] == feature].copy()
        coefficients = feature_rows["coefficient"].to_numpy(dtype=float)
        if len(coefficients) == 0:
            rows.append(
                {
                    "feature": feature,
                    "selection_rate": 0.0,
                    "sign_consistency": 0.0,
                    "dispersion": float("inf"),
                }
            )
            continue

        positive_count = int((coefficients > 1e-8).sum())
        negative_count = int((coefficients < -1e-8).sum())
        non_zero_count = positive_count + negative_count
        active = coefficients[np.abs(coefficients) > 1e-8]
        if len(active) >= 2 and abs(active.mean()) > 1e-8:
            dispersion = float(np.std(active, ddof=0) / abs(active.mean()))
        elif len(active) == 1:
            dispersion = 1.0
        else:
            dispersion = float("inf")
        rows.append(
            {
                "feature": feature,
                "selection_rate": non_zero_count / len(coefficients),
                "sign_consistency": max(positive_count, negative_count) / max(non_zero_count, 1),
                "dispersion": dispersion,
            }
        )

    stats_df = pd.DataFrame(rows)
    dispersion_clipped = stats_df["dispersion"].replace([np.inf, -np.inf], np.nan).fillna(5.0).clip(upper=5.0)
    stable_share = (
        (
            (stats_df["selection_rate"] >= (2.0 / 3.0))
            & (stats_df["sign_consistency"] >= 0.75)
            & (dispersion_clipped <= 1.0)
        ).mean()
        if not stats_df.empty
        else 0.0
    )
    return {
        "mean_selection_rate": float(stats_df["selection_rate"].mean()) if not stats_df.empty else 0.0,
        "mean_sign_consistency": float(stats_df["sign_consistency"].mean()) if not stats_df.empty else 0.0,
        "mean_dispersion": float(dispersion_clipped.mean()) if not stats_df.empty else 5.0,
        "stable_feature_share": float(stable_share),
    }


def _hyperparameter_search(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    media_feature_columns: list[str],
    fold_label: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    cv_indices = panel_time_cv_indices(train_df[["semana_inicio"]].copy(), n_splits=3)
    rows: list[dict[str, float | str]] = []

    for alpha in BETA_ELASTIC_NET_ALPHAS:
        for l1_ratio in BETA_ELASTIC_NET_L1:
            fold_metrics: list[dict[str, float]] = []
            fold_coefficients: list[pd.DataFrame] = []
            for inner_fold_idx, (inner_train_idx, inner_valid_idx) in enumerate(cv_indices, start=1):
                inner_train = train_df.iloc[inner_train_idx].copy()
                inner_valid = train_df.iloc[inner_valid_idx].copy()
                (
                    _baseline_scaler,
                    _baseline_model,
                    _media_scaler,
                    _media_model,
                    original_coef,
                    _standardized_coef,
                    intercept,
                ) = _fit_two_stage_linear(
                    inner_train,
                    feature_columns,
                    media_feature_columns,
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                )
                pred = intercept + inner_valid[feature_columns].mul(original_coef, axis=1).sum(axis=1).to_numpy(dtype=float)
                metrics = compute_metrics(inner_valid["ventas_netas"].to_numpy(dtype=float), pred)
                metrics["inner_fold"] = inner_fold_idx
                fold_metrics.append(metrics)
                fold_coefficients.append(
                    pd.DataFrame(
                        {
                            "inner_fold": inner_fold_idx,
                            "feature": feature_columns,
                            "coefficient": original_coef.reindex(feature_columns).to_numpy(dtype=float),
                        }
                    )
                )

            coefficient_df = pd.concat(fold_coefficients, ignore_index=True)
            stability = _summarize_coefficient_stability(coefficient_df, media_feature_columns)
            mean_mape = float(np.mean([metric["mape"] for metric in fold_metrics]))
            mean_rmse = float(np.mean([metric["rmse"] for metric in fold_metrics]))
            error_score = float(1.0 / (1.0 + mean_mape + mean_rmse / 100000.0))
            objective_score = (
                0.45 * stability["stable_feature_share"]
                + 0.25 * stability["mean_sign_consistency"]
                + 0.20 * stability["mean_selection_rate"]
                + 0.07 * float(1.0 / (1.0 + stability["mean_dispersion"]))
                + 0.03 * error_score
            )
            rows.append(
                {
                    "fold": fold_label,
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "objective_score": objective_score,
                    "stable_feature_share": stability["stable_feature_share"],
                    "mean_sign_consistency": stability["mean_sign_consistency"],
                    "mean_selection_rate": stability["mean_selection_rate"],
                    "mean_dispersion": stability["mean_dispersion"],
                    "mean_mape": mean_mape,
                    "mean_rmse": mean_rmse,
                    "mean_bias": float(np.mean([metric["bias"] for metric in fold_metrics])),
                }
            )

    diagnostics_df = pd.DataFrame(rows).sort_values(
        [
            "objective_score",
            "stable_feature_share",
            "mean_sign_consistency",
            "mean_selection_rate",
            "mean_dispersion",
            "mean_mape",
            "mean_rmse",
        ],
        ascending=[False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    best = diagnostics_df.iloc[0]
    return {
        "alpha": float(best["alpha"]),
        "l1_ratio": float(best["l1_ratio"]),
        "objective_score": float(best["objective_score"]),
    }, diagnostics_df


def _fit_beta_package(
    train_df: pd.DataFrame,
    control_cols: list[str],
    city_cols: list[str],
    media_cols: list[str],
    fold_label: str,
    fixed_media_params: dict[str, dict[str, float | int | str]] | None = None,
    fixed_hyperparams: dict[str, float] | None = None,
) -> tuple[TwoStageBetaPackage, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if fixed_media_params is None:
        media_params, transform_diagnostics = select_media_transforms(
            train_df,
            media_cols,
            control_cols,
            city_cols,
            allow_negative=False,
            transform_grid=BETA_TRANSFORM_GRID,
        )
        transform_diag = transform_diagnostics.copy()
        transform_diag["selection_strategy"] = "fold_search"
    else:
        media_params = fixed_media_params
        transform_diag = pd.DataFrame(
            [
                {
                    "channel": media_col,
                    "lag": int(params["lag"]),
                    "alpha": float(params["alpha"]),
                    "saturation": str(params["saturation"]),
                    "selection_score": np.nan,
                    "avg_incremental_mae_gain": np.nan,
                    "avg_corr_signal": np.nan,
                    "positive_coef_folds": np.nan,
                    "negative_coef_folds": np.nan,
                    "non_zero_coef_folds": np.nan,
                    "sign_consistency": np.nan,
                    "dominant_sign": np.nan,
                    "selection_strategy": "validation_consensus",
                }
                for media_col, params in media_params.items()
            ]
        )
    working_train, feature_columns = _prepare_beta_design(train_df, control_cols, city_cols, media_cols, media_params)
    media_feature_columns = [column for column in feature_columns if column.startswith("media_")]
    collinearity = filter_beta_collinear_features(
        working_train,
        feature_columns,
        protected_features=control_cols + city_cols,
    )
    if fixed_hyperparams is None:
        best_params, hyperparam_diagnostics = _hyperparameter_search(
            working_train,
            feature_columns,
            media_feature_columns,
            fold_label=fold_label,
        )
        hyperparam_diag = hyperparam_diagnostics.copy()
        hyperparam_diag["selected"] = (
            (hyperparam_diag["alpha"] == best_params["alpha"])
            & (hyperparam_diag["l1_ratio"] == best_params["l1_ratio"])
        )
        hyperparam_diag["selection_strategy"] = "fold_search"
    else:
        best_params = fixed_hyperparams
        hyperparam_diag = pd.DataFrame(
            [
                {
                    "fold": fold_label,
                    "alpha": float(best_params["alpha"]),
                    "l1_ratio": float(best_params["l1_ratio"]),
                    "objective_score": np.nan,
                    "stable_feature_share": np.nan,
                    "mean_sign_consistency": np.nan,
                    "mean_selection_rate": np.nan,
                    "mean_dispersion": np.nan,
                    "mean_mape": np.nan,
                    "mean_rmse": np.nan,
                    "mean_bias": np.nan,
                    "selected": True,
                    "selection_strategy": "validation_consensus_1se",
                }
            ]
        )
    (
        baseline_scaler,
        baseline_model,
        media_scaler,
        media_model,
        original_coef,
        standardized_coef,
        intercept,
    ) = _fit_two_stage_linear(
        working_train,
        feature_columns,
        media_feature_columns,
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
    )
    package = TwoStageBetaPackage(
        baseline_scaler=baseline_scaler,
        baseline_model=baseline_model,
        media_scaler=media_scaler,
        media_model=media_model,
        feature_columns=feature_columns,
        baseline_feature_columns=[feature for feature in feature_columns if feature not in set(media_feature_columns)],
        media_feature_columns=media_feature_columns,
        control_columns=control_cols + city_cols,
        media_params=media_params,
        city_dummy_columns=city_cols,
        spec_name=SPEC_NAME,
        original_coefficients=original_coef,
        standardized_coefficients=standardized_coef,
        intercept=float(intercept),
        alpha=float(best_params["alpha"]),
        l1_ratio=float(best_params["l1_ratio"]),
    )
    transform_diag["fold"] = fold_label
    collinearity_diag = pd.DataFrame(
        {
            "feature": feature,
            "reason": "kept",
            "detail": "regularized_positive_media_stage",
            "fold": fold_label,
        }
        for feature in feature_columns
    )
    return package, transform_diag, hyperparam_diag, collinearity_diag


def _selected_transform_frame(package: TwoStageBetaPackage, fold_label: str) -> pd.DataFrame:
    rows = []
    for channel, params in package.media_params.items():
        rows.append(
            {
                "fold": fold_label,
                "channel": channel,
                "lag": int(params["lag"]),
                "alpha": float(params["alpha"]),
                "saturation": str(params["saturation"]),
                "complexity": float(params["lag"]) + float(params["alpha"]) + (0.25 if str(params["saturation"]) == "log1p" else 0.0),
            }
        )
    return pd.DataFrame(rows)


def _select_consensus_media_params(
    media_cols: list[str],
    selected_transforms_df: pd.DataFrame,
    transform_diagnostics_df: pd.DataFrame,
) -> dict[str, dict[str, float | int | str]]:
    params: dict[str, dict[str, float | int | str]] = {}

    for channel in media_cols:
        selected_rows = selected_transforms_df[selected_transforms_df["channel"] == channel].copy()
        diagnostic_rows = transform_diagnostics_df[transform_diagnostics_df["channel"] == channel].copy()

        if selected_rows.empty and diagnostic_rows.empty:
            params[channel] = {"lag": 0, "alpha": 0.0, "saturation": "none"}
            continue

        selected_summary = (
            selected_rows.groupby(["lag", "alpha", "saturation"], as_index=False)
            .agg(
                selected_count=("fold", "count"),
                mean_complexity=("complexity", "mean"),
            )
        )
        diagnostic_summary = (
            diagnostic_rows.groupby(["lag", "alpha", "saturation"], as_index=False)
            .agg(
                mean_selection_score=("selection_score", "mean"),
                mean_sign_consistency=("sign_consistency", "mean"),
                mean_non_zero_coef_folds=("non_zero_coef_folds", "mean"),
                mean_avg_corr_signal=("avg_corr_signal", "mean"),
            )
        )
        summary = diagnostic_summary.merge(
            selected_summary,
            on=["lag", "alpha", "saturation"],
            how="outer",
        ).fillna(
            {
                "selected_count": 0,
                "mean_complexity": np.nan,
                "mean_selection_score": -np.inf,
                "mean_sign_consistency": 0.0,
                "mean_non_zero_coef_folds": 0.0,
                "mean_avg_corr_signal": 0.0,
            }
        )
        if summary["mean_complexity"].isna().any():
            summary["mean_complexity"] = (
                summary["lag"].astype(float)
                + summary["alpha"].astype(float)
                + np.where(summary["saturation"].eq("log1p"), 0.25, 0.0)
            )
        summary = summary.sort_values(
            [
                "selected_count",
                "mean_sign_consistency",
                "mean_selection_score",
                "mean_non_zero_coef_folds",
                "mean_avg_corr_signal",
                "mean_complexity",
            ],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)
        chosen = summary.iloc[0]
        params[channel] = {
            "lag": int(chosen["lag"]),
            "alpha": float(chosen["alpha"]),
            "saturation": str(chosen["saturation"]),
        }

    return params


def _select_consensus_hyperparameters(hyperparam_results_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    aggregation = (
        hyperparam_results_df.groupby(["alpha", "l1_ratio"], as_index=False)
        .agg(
            fold_count=("fold", "nunique"),
            mean_objective_score=("objective_score", "mean"),
            std_objective_score=("objective_score", "std"),
            mean_stable_feature_share=("stable_feature_share", "mean"),
            mean_sign_consistency=("mean_sign_consistency", "mean"),
            mean_selection_rate=("mean_selection_rate", "mean"),
            mean_dispersion=("mean_dispersion", "mean"),
            mean_mape=("mean_mape", "mean"),
            mean_rmse=("mean_rmse", "mean"),
        )
        .sort_values(["mean_objective_score", "mean_sign_consistency", "mean_selection_rate"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    aggregation["std_objective_score"] = aggregation["std_objective_score"].fillna(0.0)
    aggregation["objective_se"] = aggregation["std_objective_score"] / np.sqrt(aggregation["fold_count"].clip(lower=1))
    best_row = aggregation.iloc[0]
    objective_threshold = float(best_row["mean_objective_score"] - best_row["objective_se"])
    candidate_pool = aggregation[aggregation["mean_objective_score"] >= objective_threshold].copy()
    candidate_pool = candidate_pool.sort_values(
        [
            "mean_stable_feature_share",
            "mean_sign_consistency",
            "mean_selection_rate",
            "mean_dispersion",
            "mean_mape",
            "mean_rmse",
            "alpha",
            "l1_ratio",
        ],
        ascending=[False, False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    selected = candidate_pool.iloc[0]
    aggregation["objective_threshold"] = objective_threshold
    aggregation["selected"] = (
        (aggregation["alpha"] == float(selected["alpha"]))
        & (aggregation["l1_ratio"] == float(selected["l1_ratio"]))
    )
    aggregation["fold"] = "validation_consensus"
    aggregation["selection_strategy"] = "validation_consensus_1se"
    return {
        "alpha": float(selected["alpha"]),
        "l1_ratio": float(selected["l1_ratio"]),
    }, aggregation


def _score_package(
    package: TwoStageBetaPackage,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    control_cols: list[str],
    city_cols: list[str],
    media_cols: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    combined = pd.concat([train_df, valid_df], ignore_index=False).sort_values(["semana_inicio", "ciudad"]).copy()
    working_df, _ = _prepare_beta_design(combined, control_cols, city_cols, media_cols, package.media_params)
    valid_work = working_df.loc[valid_df.index].copy()
    pred = package.intercept + valid_work[package.feature_columns].mul(package.original_coefficients, axis=1).sum(axis=1).to_numpy(
        dtype=float
    )
    scored = valid_df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    scored["pred"] = pred
    metrics = compute_metrics(scored["ventas_netas"], scored["pred"])
    metrics["alpha"] = float(package.alpha)
    metrics["l1_ratio"] = float(package.l1_ratio)
    media_only = package.original_coefficients.reindex(package.media_feature_columns)
    metrics["negative_media_coefficients"] = int((media_only < -1e-8).sum())
    return metrics, scored


def _bootstrap_coefficients(
    working_df: pd.DataFrame,
    feature_columns: list[str],
    media_feature_columns: list[str],
    alpha: float,
    l1_ratio: float,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    unique_weeks = pd.to_datetime(working_df["semana_inicio"]).drop_duplicates().sort_values().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, float | str | int]] = []

    for iteration in range(iterations):
        sampled_weeks = rng.choice(unique_weeks, size=len(unique_weeks), replace=True)
        bootstrap_df = (
            pd.concat(
                [working_df[working_df["semana_inicio"] == week].copy() for week in sampled_weeks],
                ignore_index=True,
            )
            .sort_values(["semana_inicio", "ciudad"])
            .reset_index(drop=True)
        )
        (
            _baseline_scaler,
            _baseline_model,
            _media_scaler,
            _media_model,
            original_coef,
            standardized_coef,
            _intercept,
        ) = _fit_two_stage_linear(
            bootstrap_df,
            feature_columns,
            media_feature_columns,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        for feature in feature_columns:
            rows.append(
                {
                    "bootstrap_iteration": iteration,
                    "feature": feature,
                    "standardized_coefficient": float(standardized_coef[feature]),
                    "coefficient": float(original_coef[feature]),
                }
            )
    return pd.DataFrame(rows)


def _summarize_bootstrap_intervals(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    if bootstrap_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "bootstrap_original_p02_5",
                "bootstrap_original_p50",
                "bootstrap_original_p97_5",
                "bootstrap_standardized_p02_5",
                "bootstrap_standardized_p50",
                "bootstrap_standardized_p97_5",
                "bootstrap_positive_probability",
                "bootstrap_negative_probability",
            ]
        )

    rows = []
    for feature, group in bootstrap_df.groupby("feature", sort=True):
        original = group["coefficient"].to_numpy(dtype=float)
        standardized = group["standardized_coefficient"].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "bootstrap_original_p02_5": float(np.quantile(original, 0.025)),
                "bootstrap_original_p50": float(np.quantile(original, 0.5)),
                "bootstrap_original_p97_5": float(np.quantile(original, 0.975)),
                "bootstrap_standardized_p02_5": float(np.quantile(standardized, 0.025)),
                "bootstrap_standardized_p50": float(np.quantile(standardized, 0.5)),
                "bootstrap_standardized_p97_5": float(np.quantile(standardized, 0.975)),
                "bootstrap_positive_probability": float(np.mean(original > 1e-8)),
                "bootstrap_negative_probability": float(np.mean(original < -1e-8)),
            }
        )
    return pd.DataFrame(rows)


def _deployment_coefficient_table(
    package: TwoStageBetaPackage,
    city_cols: list[str],
    beta_summary_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    collinearity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    standardized_coef = package.standardized_coefficients
    original_coef = package.original_coefficients
    df = pd.DataFrame(
        {
            "feature": package.feature_columns,
            "standardized_coefficient": standardized_coef.reindex(package.feature_columns).to_numpy(dtype=float),
            "coefficient": original_coef.reindex(package.feature_columns).to_numpy(dtype=float),
        }
    )
    df["feature_block"] = df["feature"].map(lambda feature: feature_block(feature, city_cols))
    df["channel"] = df["feature"].where(df["feature"].str.startswith("media_"), None)
    df["channel"] = df["channel"].str.split("__").str[0]
    df["coefficient_sign"] = df["coefficient"].map(
        lambda value: "positive" if value > 1e-8 else "negative" if value < -1e-8 else "zero"
    )
    df = df.merge(
        beta_summary_df[
            [
                "feature",
                "selection_rate",
                "sign_consistency",
                "dominant_sign",
                "stability_status",
                "mean_coefficient",
                "std_coefficient",
                "coef_cv",
            ]
        ],
        on="feature",
        how="left",
    )
    df = df.merge(bootstrap_summary_df, on="feature", how="left")
    collinearity_flags = (
        collinearity_df[collinearity_df["reason"] != "kept"][["feature"]]
        .drop_duplicates()
        .assign(collinearity_flag=True)
    )
    df = df.merge(collinearity_flags, on="feature", how="left")
    df["collinearity_flag"] = df["collinearity_flag"].eq(True)
    usability = df.apply(
        lambda row: usability_flag_row(
            feature_block=str(row["feature_block"]),
            stability_status=str(row["stability_status"]),
            selection_rate=float(row["selection_rate"]) if pd.notna(row["selection_rate"]) else 0.0,
            sign_consistency=float(row["sign_consistency"]) if pd.notna(row["sign_consistency"]) else 0.0,
            coefficient_sign=str(row["coefficient_sign"]),
            dominant_sign=str(row["dominant_sign"]) if pd.notna(row["dominant_sign"]) else "zero",
            bootstrap_positive_probability=float(row["bootstrap_positive_probability"]) if pd.notna(row["bootstrap_positive_probability"]) else 0.0,
            bootstrap_negative_probability=float(row["bootstrap_negative_probability"]) if pd.notna(row["bootstrap_negative_probability"]) else 0.0,
            collinearity_flag=bool(row["collinearity_flag"]),
        ),
        axis=1,
        result_type="expand",
    )
    usability.columns = ["usable_beta", "usable_reason"]
    df = pd.concat([df, usability], axis=1)
    return df.sort_values(["feature_block", "feature"]).reset_index(drop=True), original_coef


def _deployment_channel_table(
    deploy_package: TwoStageBetaPackage,
    deployment_coefficients_df: pd.DataFrame,
    total_fold_count: int,
    fold_coefficients_df: pd.DataFrame,
    deployment_collinearity_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = []
    dropped_features = set(
        deployment_collinearity_df[deployment_collinearity_df["reason"] != "kept"]["feature"].astype(str).tolist()
    )
    for media_col, params in deploy_package.media_params.items():
        feature = transformed_feature_name(
            media_col,
            lag=int(params["lag"]),
            alpha=float(params["alpha"]),
            saturation=str(params["saturation"]),
        )
        selected_rows.append(
            {
                "channel": media_col,
                "feature": feature,
                "selected_transform": json.dumps(params, sort_keys=True),
                "collinearity_flag": feature in dropped_features,
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    deployment_channel_df = selected_df.merge(
        deployment_coefficients_df[
            [
                "feature",
                "standardized_coefficient",
                "coefficient",
                "coefficient_sign",
                "stability_status",
                "bootstrap_original_p02_5",
                "bootstrap_original_p50",
                "bootstrap_original_p97_5",
                "bootstrap_positive_probability",
                "bootstrap_negative_probability",
            ]
        ],
        on="feature",
        how="left",
    )
    deployment_channel_df["standardized_coefficient"] = deployment_channel_df["standardized_coefficient"].fillna(np.nan)
    deployment_channel_df["coefficient"] = deployment_channel_df["coefficient"].fillna(np.nan)
    deployment_channel_df["coefficient_sign"] = deployment_channel_df["coefficient_sign"].fillna("zero")
    deployment_channel_df["stability_status"] = deployment_channel_df["stability_status"].fillna("unstable")
    channel_summary = summarize_channel_stability(
        fold_coefficients_df[fold_coefficients_df["feature_block"] == "media"].copy(),
        deployment_channel_df,
        total_fold_count=total_fold_count,
    )
    channel_summary["channel"] = channel_summary["channel"].str.replace("media_", "", regex=False)
    return channel_summary


def _write_beta_report(
    backtest_results: pd.DataFrame,
    test_results: pd.DataFrame,
    beta_summary: pd.DataFrame,
    deployment_coefficients: pd.DataFrame,
    channel_summary: pd.DataFrame,
    model_results: dict,
    hyperparam_results: pd.DataFrame,
) -> None:
    deployment_features = set(deployment_coefficients["feature"].tolist())
    deployment_beta_summary = beta_summary[beta_summary["feature"].isin(deployment_features)].copy()
    usable_media = deployment_coefficients[
        (deployment_coefficients["feature_block"] == "media") & (deployment_coefficients["usable_beta"])
    ].copy()
    stable_media = deployment_beta_summary[
        (deployment_beta_summary["feature_block"] == "media") & (deployment_beta_summary["stability_status"] == "stable")
    ].copy()
    review_media = deployment_beta_summary[
        (deployment_beta_summary["feature_block"] == "media") & (deployment_beta_summary["stability_status"] == "review")
    ].copy()
    unstable_media = deployment_beta_summary[
        (deployment_beta_summary["feature_block"] == "media") & (deployment_beta_summary["stability_status"] == "unstable")
    ].copy()
    negative_media = deployment_coefficients[
        (deployment_coefficients["feature_block"] == "media") & (deployment_coefficients["coefficient"] < -1e-8)
    ].copy()
    usable_channels = channel_summary[channel_summary["usable_beta"]].copy()
    hyperparam_view = hyperparam_results[
        [
            "fold",
            "alpha",
            "l1_ratio",
            "objective_score",
            "stable_feature_share",
            "mean_sign_consistency",
            "mean_selection_rate",
            "mean_dispersion",
            "mean_mape",
        ]
    ].copy()

    lines = [
        "# Elastic Net Beta Report",
        "",
        "## Objetivo",
        "",
        "- Ruta principal de betas basada en `ElasticNetTransformedMedia`.",
        "- Separacion explicita entre `baseline` y bloque de `media` para que las betas de medios lean incremental y no compitan con toda la estructura temporal.",
        "- Hiperparametros elegidos por estabilidad de beta en expanding-window CV; el error se usa solo como criterio secundario.",
        "",
        "## Especificacion",
        "",
        f"- Ventana de entrenamiento principal: `{model_results['train_start_year']}`-`{model_results['test_year']}`.",
        f"- Años de validacion para estabilidad: `{', '.join(map(str, model_results['validation_years'])) or 'ninguno'}`.",
        f"- Test holdout final: `{model_results['test_year']}`.",
        "- Controles baseline: tendencia, estacionalidad, calendario, macro-clima y dummies de ciudad.",
        "- Variables excluidas de la ruta principal: trafico, pedidos, ticket medio y proxies downstream contemporaneos.",
        "- Medios: `media_*` transformados con lag/adstock/saturacion y bloque media estimado con restriccion positiva.",
        f"- La especificacion final de despliegue se fija por `{model_results['deployment_selection_strategy']}` sobre validacion temporal, no por reoptimizacion oportunista en todos los datos.",
        "",
        "## Busqueda de hiperparametros",
        "",
        "```text",
        hyperparam_view.round(4).to_string(index=False),
        "```",
        "",
        "## Backtest de estabilidad",
        "",
        "```text",
        backtest_results.round(4).to_string(index=False),
        "```",
        "",
        "## Test final",
        "",
        "```text",
        test_results.round(4).to_string(index=False),
        "```",
        "",
        "## Betas de medios estables",
        "",
        "```text",
        stable_media.round(4).to_string(index=False) if not stable_media.empty else "Sin betas media estables con los guardrails actuales.",
        "```",
        "",
        "## Resumen por canal",
        "",
        "```text",
        channel_summary.round(4).to_string(index=False) if not channel_summary.empty else "Sin resumen por canal disponible.",
        "```",
        "",
        "## Betas utilizables",
        "",
        "```text",
        usable_channels.round(4).to_string(index=False) if not usable_channels.empty else "Ninguna beta media pasa aun el filtro final de usabilidad.",
        "```",
        "",
        "## Betas en revision",
        "",
        "```text",
        review_media.round(4).to_string(index=False) if not review_media.empty else "Sin betas media en revision.",
        "```",
        "",
        "## Betas inestables",
        "",
        "```text",
        unstable_media.round(4).to_string(index=False) if not unstable_media.empty else "Sin betas media inestables.",
        "```",
        "",
        "## Betas negativas en despliegue",
        "",
        "```text",
        negative_media.round(4).to_string(index=False) if not negative_media.empty else "No hay betas negativas de medios en el modelo desplegado.",
        "```",
        "",
        "## Lectura",
        "",
        f"- El modelo final usa `alpha={model_results['model_alpha']:.6f}` y `l1_ratio={model_results['model_l1_ratio']:.2f}`.",
        f"- Coeficientes media negativos en despliegue: `{model_results['negative_media_coefficients']}`.",
        f"- Betas media utilizables tras filtros finales: `{int(usable_channels.shape[0])}`.",
        "- Cada beta publicada incluye escala estandarizada, escala original, signo, estabilidad y rango bootstrap empirico.",
        "- La capa media se publica con restriccion positiva; si un canal no puede sostener una lectura positiva estable, queda penalizado o fuera por estabilidad/usabilidad.",
        "",
        "## Artefactos",
        "",
        f"- `{BETA_BACKTEST_TABLE.name}`.",
        f"- `{BETA_TEST_TABLE.name}`.",
        f"- `{BETA_FOLD_COEFFICIENTS_TABLE.name}`.",
        f"- `{BETA_SUMMARY_TABLE.name}`.",
        f"- `{BETA_DEPLOYMENT_COEFFICIENTS_TABLE.name}`.",
        f"- `{BETA_CHANNEL_TABLE.name}`.",
        f"- `{BETA_TRANSFORM_DIAGNOSTICS_TABLE.name}`.",
        f"- `{BETA_COLLINEARITY_TABLE.name}`.",
        f"- `{BETA_HYPERPARAM_TABLE.name}`.",
        f"- `{BETA_BOOTSTRAP_TABLE.name}`.",
        f"- `{BETA_RESULTS_JSON.name}`.",
        f"- `{BETA_TRANSFORMS_JSON.name}`.",
    ]
    BETA_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_beta_training_pipeline() -> dict:
    ensure_prerequisites()
    df = load_dataset()
    df = df[df["year"] >= TRAIN_START_YEAR].copy().reset_index(drop=True)

    media_cols = media_columns(df)
    city_cols = city_dummy_columns(df)
    test_year = int(df["year"].max())
    validation_years = _beta_validation_years(df, TRAIN_START_YEAR, test_year)

    backtest_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_coefficient_frames: list[pd.DataFrame] = []
    transform_diagnostics_frames: list[pd.DataFrame] = []
    hyperparam_frames: list[pd.DataFrame] = []
    collinearity_frames: list[pd.DataFrame] = []
    validation_transform_choice_frames: list[pd.DataFrame] = []
    validation_hyperparam_frames: list[pd.DataFrame] = []

    for year in validation_years:
        train_df = df[df["year"] < year].copy()
        valid_df = df[df["year"] == year].copy()
        fold_name = f"validate_{year}"
        package, transform_diag, hyperparam_diag, collinearity_diag = _fit_beta_package(
            train_df,
            CONTROL_COLUMNS,
            city_cols,
            media_cols,
            fold_label=fold_name,
        )
        metrics, scored = _score_package(
            package,
            train_df,
            valid_df,
            CONTROL_COLUMNS,
            city_cols,
            media_cols,
        )
        metrics["fold"] = fold_name
        metrics["spec"] = SPEC_NAME
        backtest_rows.append(metrics)
        prediction_frames.append(scored.assign(fold=fold_name, spec=SPEC_NAME))
        fold_coefficient_frames.append(_coefficient_frame(fold_name, package, city_cols))
        transform_diagnostics_frames.append(transform_diag)
        hyperparam_frames.append(hyperparam_diag)
        collinearity_frames.append(collinearity_diag)
        validation_transform_choice_frames.append(_selected_transform_frame(package, fold_name))
        validation_hyperparam_frames.append(hyperparam_diag.copy())

    train_eval = df[df["year"] < test_year].copy()
    test_eval = df[df["year"] == test_year].copy()
    test_package, test_transform_diag, test_hyperparam_diag, test_collinearity_diag = _fit_beta_package(
        train_eval,
        CONTROL_COLUMNS,
        city_cols,
        media_cols,
        fold_label=f"test_{test_year}",
    )
    test_metrics, test_scored = _score_package(
        test_package,
        train_eval,
        test_eval,
        CONTROL_COLUMNS,
        city_cols,
        media_cols,
    )
    test_results = pd.DataFrame([{**test_metrics, "fold": f"test_{test_year}", "spec": SPEC_NAME}])
    fold_coefficient_frames.append(_coefficient_frame(f"test_{test_year}", test_package, city_cols))
    transform_diagnostics_frames.append(test_transform_diag)
    hyperparam_frames.append(test_hyperparam_diag)
    collinearity_frames.append(test_collinearity_diag)

    validation_transform_choices_df = pd.concat(validation_transform_choice_frames, ignore_index=True)
    validation_transform_diagnostics_df = pd.concat(
        [frame for frame in transform_diagnostics_frames if frame["fold"].iloc[0].startswith("validate_")],
        ignore_index=True,
    )
    consensus_media_params = _select_consensus_media_params(
        media_cols,
        validation_transform_choices_df,
        validation_transform_diagnostics_df,
    )
    validation_hyperparam_df = pd.concat(validation_hyperparam_frames, ignore_index=True)
    consensus_hyperparams, consensus_hyperparam_df = _select_consensus_hyperparameters(
        validation_hyperparam_df[validation_hyperparam_df["selection_strategy"] == "fold_search"].copy()
    )
    hyperparam_frames.append(consensus_hyperparam_df)

    deploy_package, deploy_transform_diag, deploy_hyperparam_diag, deploy_collinearity_diag = _fit_beta_package(
        df,
        CONTROL_COLUMNS,
        city_cols,
        media_cols,
        fold_label="deployment_fit",
        fixed_media_params=consensus_media_params,
        fixed_hyperparams=consensus_hyperparams,
    )
    transform_diagnostics_frames.append(deploy_transform_diag)
    hyperparam_frames.append(deploy_hyperparam_diag)
    collinearity_frames.append(deploy_collinearity_diag)

    deployment_working_df, _ = _prepare_beta_design(
        df,
        CONTROL_COLUMNS,
        city_cols,
        media_cols,
        deploy_package.media_params,
    )
    _deployment_design_df, contributions_matrix = _package_contributions(df, deploy_package)
    deployment_pred = _package_predict(df, deploy_package)
    intercept = deploy_package.intercept
    deployment_scored = df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    deployment_scored["pred"] = deployment_pred
    deployment_scored["fold"] = "deployment_fit"
    deployment_scored["spec"] = SPEC_NAME

    contribution_table = (
        contributions_matrix[deploy_package.media_feature_columns]
        .assign(semana_inicio=df["semana_inicio"].to_numpy(), ciudad=df["ciudad"].to_numpy())
        .melt(id_vars=["semana_inicio", "ciudad"], var_name="feature", value_name="contribution")
    )
    contribution_table["channel"] = contribution_table["feature"].str.split("__").str[0]

    backtest_results = pd.DataFrame(backtest_rows)
    fold_coefficients_df = pd.concat(fold_coefficient_frames, ignore_index=True)
    beta_summary_df = summarize_beta_stability(
        fold_coefficients_df[["feature", "coefficient"]],
        deployment_coefficients=deploy_package.original_coefficients,
        city_columns=city_cols,
    )
    bootstrap_df = _bootstrap_coefficients(
        deployment_working_df,
        deploy_package.feature_columns,
        deploy_package.media_feature_columns,
        alpha=float(deploy_package.alpha),
        l1_ratio=float(deploy_package.l1_ratio),
    )
    bootstrap_summary_df = _summarize_bootstrap_intervals(bootstrap_df)
    predictions = pd.concat(
        prediction_frames + [test_scored.assign(fold=f"test_{test_year}", spec=SPEC_NAME), deployment_scored],
        ignore_index=True,
    )
    transform_diagnostics_df = pd.concat(transform_diagnostics_frames, ignore_index=True)
    hyperparam_results_df = pd.concat(hyperparam_frames, ignore_index=True)
    collinearity_df = pd.concat(collinearity_frames, ignore_index=True)

    deployment_coefficients_df, deployment_coefficients = _deployment_coefficient_table(
        deploy_package,
        city_cols,
        beta_summary_df,
        bootstrap_summary_df,
        collinearity_df[collinearity_df["fold"] == "deployment_fit"].copy(),
    )
    channel_summary_df = _deployment_channel_table(
        deploy_package,
        deployment_coefficients_df,
        total_fold_count=len(validation_years) + 1,
        fold_coefficients_df=fold_coefficients_df,
        deployment_collinearity_df=collinearity_df[collinearity_df["fold"] == "deployment_fit"].copy(),
    )
    deployment_feature_set = set(deployment_coefficients_df["feature"].tolist())
    deployment_beta_summary_df = beta_summary_df[beta_summary_df["feature"].isin(deployment_feature_set)].copy()

    stable_media_features = channel_summary_df[channel_summary_df["usable_beta"]]["channel"].tolist()
    review_media_features = deployment_beta_summary_df[
        (deployment_beta_summary_df["feature_block"] == "media") & (deployment_beta_summary_df["stability_status"] == "review")
    ]["feature"].tolist()

    model_results = {
        "winner": SPEC_NAME,
        "selection_objective": "beta_stability",
        "train_start_year": TRAIN_START_YEAR,
        "validation_years": validation_years,
        "test_year": test_year,
        "baseline_columns": CONTROL_COLUMNS + city_cols,
        "media_columns": media_cols,
        "selected_transforms": deploy_package.media_params,
        "deployment_selection_strategy": "validation_consensus_1se",
        "model_alpha": float(deploy_package.alpha),
        "model_l1_ratio": float(deploy_package.l1_ratio),
        "intercept": float(intercept),
        "test_metrics": test_metrics,
        "negative_media_coefficients": int((deployment_coefficients_df.loc[deployment_coefficients_df["feature_block"] == "media", "coefficient"] < -1e-8).sum()),
        "stable_media_features": stable_media_features,
        "review_media_features": review_media_features,
        "usable_media_features": stable_media_features,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }

    backtest_results.to_csv(BETA_BACKTEST_TABLE, index=False)
    test_results.to_csv(BETA_TEST_TABLE, index=False)
    predictions.to_csv(BETA_PREDICTIONS_TABLE, index=False)
    contribution_table.to_csv(BETA_CONTRIBUTIONS_TABLE, index=False)
    fold_coefficients_df.to_csv(BETA_FOLD_COEFFICIENTS_TABLE, index=False)
    beta_summary_df.to_csv(BETA_SUMMARY_TABLE, index=False)
    deployment_coefficients_df.to_csv(BETA_DEPLOYMENT_COEFFICIENTS_TABLE, index=False)
    channel_summary_df.to_csv(BETA_CHANNEL_TABLE, index=False)
    transform_diagnostics_df.to_csv(BETA_TRANSFORM_DIAGNOSTICS_TABLE, index=False)
    collinearity_df.to_csv(BETA_COLLINEARITY_TABLE, index=False)
    hyperparam_results_df.to_csv(BETA_HYPERPARAM_TABLE, index=False)
    bootstrap_df.to_csv(BETA_BOOTSTRAP_TABLE, index=False)
    BETA_RESULTS_JSON.write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    BETA_TRANSFORMS_JSON.write_text(json.dumps(deploy_package.media_params, indent=2), encoding="utf-8")
    with BETA_MODEL_FILE.open("wb") as handle:
        pickle.dump(deploy_package, handle)

    _write_beta_report(
        backtest_results=backtest_results,
        test_results=test_results,
        beta_summary=beta_summary_df,
        deployment_coefficients=deployment_coefficients_df,
        channel_summary=channel_summary_df,
        model_results=model_results,
        hyperparam_results=hyperparam_results_df[hyperparam_results_df["selected"]].copy(),
    )

    return {
        "backtest_results": backtest_results,
        "test_results": test_results,
        "beta_summary": beta_summary_df,
        "deployment_coefficients": deployment_coefficients_df,
        "channel_summary": channel_summary_df,
        "model_results": model_results,
    }
