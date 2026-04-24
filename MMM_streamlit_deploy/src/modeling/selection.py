from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.common.metrics import compute_metrics
from src.modeling.trainer import (
    build_feature_frame,
    fit_elastic_net,
    media_columns,
    original_scale_coefficients,
    original_scale_intercept,
    predict_with_model,
)
from src.modeling.model_package import ModelPackage
from src.modeling.specs import TRANSFORM_GRID
from src.modeling.transforms import add_transformed_media_features, apply_media_transform
from src.validation.backtesting import panel_time_cv_indices

TRANSFORM_NEAR_BEST_TOL = 0.004


def naive_predictions(df: pd.DataFrame) -> pd.Series:
    out = df.sort_values(["ciudad", "semana_inicio"]).copy()
    rolling_mean = (
        out.groupby("ciudad")["ventas_netas"]
        .transform(lambda series: series.shift(1).rolling(4, min_periods=1).mean())
    )
    seasonal = out.groupby("ciudad")["ventas_netas"].shift(52)
    out["naive_pred"] = rolling_mean.fillna(seasonal).fillna(out.groupby("ciudad")["ventas_netas"].transform("mean"))
    return out.sort_index()["naive_pred"]


def select_media_transforms(
    train_df: pd.DataFrame,
    media_cols: List[str],
    control_cols: List[str],
    city_cols: List[str],
    allow_negative: bool = False,
    transform_grid: list[dict[str, float | int | str]] | None = None,
) -> Tuple[Dict[str, Dict[str, float | int | str]], pd.DataFrame]:
    candidate_grid = transform_grid or TRANSFORM_GRID
    base_features = build_feature_frame(train_df, control_cols, city_cols)
    scaler, model = fit_elastic_net(train_df, list(base_features.columns))
    ordered = train_df.sort_values(["ciudad", "semana_inicio"])
    residual = train_df["ventas_netas"].to_numpy() - predict_with_model(train_df, list(base_features.columns), scaler, model)
    ordered = ordered.copy()
    ordered["controls_residual"] = pd.Series(residual, index=train_df.index).loc[ordered.index].to_numpy()
    cv_indices = panel_time_cv_indices(ordered[["semana_inicio"]].copy(), n_splits=3)

    diagnostics = []
    params: Dict[str, Dict[str, float | int | str]] = {}

    for media_col in media_cols:
        candidates: list[dict[str, float | int | str]] = []
        for candidate in candidate_grid:
            transformed = (
                ordered.groupby("ciudad", group_keys=False)[media_col]
                .apply(
                    lambda series: apply_media_transform(
                        series,
                        lag=int(candidate["lag"]),
                        alpha=float(candidate["alpha"]),
                        saturation=str(candidate["saturation"]),
                    )
                )
                .astype(float)
            )
            if float(transformed.std()) < 1e-9:
                diagnostics.append(
                    {
                        "channel": media_col,
                        "lag": candidate["lag"],
                        "alpha": candidate["alpha"],
                        "saturation": candidate["saturation"],
                        "selection_score": -np.inf,
                        "avg_incremental_mae_gain": -np.inf,
                        "avg_corr_signal": 0.0,
                        "positive_coef_folds": 0,
                        "negative_coef_folds": 0,
                        "non_zero_coef_folds": 0,
                        "sign_consistency": 0.0,
                        "dominant_sign": "zero",
                    }
                )
                continue

            working = ordered.copy()
            working["candidate_media"] = transformed.to_numpy(dtype=float)

            fold_scores = []
            fold_mae_gain = []
            fold_corr_signal = []
            positive_coef_folds = 0
            negative_coef_folds = 0
            non_zero_coef_folds = 0
            for train_idx, valid_idx in cv_indices:
                fold_train = working.iloc[train_idx]
                fold_valid = working.iloc[valid_idx]
                fold_model = LinearRegression(positive=not allow_negative)
                fold_model.fit(fold_train[["candidate_media"]], fold_train["controls_residual"])
                coef = float(fold_model.coef_[0])
                valid_pred = fold_model.predict(fold_valid[["candidate_media"]])
                base_mae = float(np.mean(np.abs(fold_valid["controls_residual"])))
                modeled_mae = float(np.mean(np.abs(fold_valid["controls_residual"] - valid_pred)))
                mae_gain = (base_mae - modeled_mae) / max(base_mae, 1e-9)
                corr = float(np.corrcoef(fold_valid["candidate_media"].to_numpy(), fold_valid["controls_residual"].to_numpy())[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                corr_signal = abs(corr) if allow_negative else max(corr, 0.0)
                fold_scores.append(0.7 * mae_gain + 0.3 * corr_signal)
                fold_mae_gain.append(mae_gain)
                fold_corr_signal.append(corr_signal)
                if coef > 1e-9:
                    positive_coef_folds += 1
                    non_zero_coef_folds += 1
                elif coef < -1e-9:
                    negative_coef_folds += 1
                    non_zero_coef_folds += 1

            dominant_sign_folds = max(positive_coef_folds, negative_coef_folds)
            sign_consistency = dominant_sign_folds / max(non_zero_coef_folds, 1)
            dominant_sign = "positive" if positive_coef_folds >= negative_coef_folds else "negative"

            selection_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            diagnostics.append(
                {
                    "channel": media_col,
                    "lag": candidate["lag"],
                    "alpha": candidate["alpha"],
                    "saturation": candidate["saturation"],
                    "selection_score": selection_score,
                    "avg_incremental_mae_gain": float(np.mean(fold_mae_gain)) if fold_mae_gain else -np.inf,
                    "avg_corr_signal": float(np.mean(fold_corr_signal)) if fold_corr_signal else 0.0,
                    "positive_coef_folds": int(positive_coef_folds),
                    "negative_coef_folds": int(negative_coef_folds),
                    "non_zero_coef_folds": int(non_zero_coef_folds),
                    "sign_consistency": float(sign_consistency),
                    "dominant_sign": dominant_sign if non_zero_coef_folds else "zero",
                }
            )
            complexity = float(candidate["lag"]) + float(candidate["alpha"]) + (0.25 if str(candidate["saturation"]) == "log1p" else 0.0)
            candidates.append(
                {
                    **candidate,
                    "selection_score": selection_score,
                    "positive_coef_folds": int(positive_coef_folds),
                    "negative_coef_folds": int(negative_coef_folds),
                    "non_zero_coef_folds": int(non_zero_coef_folds),
                    "sign_consistency": float(sign_consistency),
                    "complexity": complexity,
                }
            )

        if not candidates:
            params[media_col] = {"lag": 0, "alpha": 0.0, "saturation": "none"}
            continue

        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row["selection_score"]),
                float(row["sign_consistency"]),
                int(row["non_zero_coef_folds"]),
                -float(row["complexity"]),
            ),
            reverse=True,
        )
        best_score = float(ranked[0]["selection_score"])
        near_best = [
            row for row in ranked
            if float(row["selection_score"]) >= best_score - TRANSFORM_NEAR_BEST_TOL
        ]
        if allow_negative:
            stable_near_best = [
                row for row in near_best
                if int(row["non_zero_coef_folds"]) >= 2 and float(row["sign_consistency"]) >= (2.0 / 3.0)
            ]
        else:
            stable_near_best = [row for row in near_best if int(row["positive_coef_folds"]) >= 2]
        shortlist = stable_near_best if stable_near_best else near_best
        chosen = min(
            shortlist,
            key=lambda row: (
                float(row["complexity"]),
                -float(row["sign_consistency"]),
                -int(row["non_zero_coef_folds"]),
                -float(row["selection_score"]),
            ),
        )
        params[media_col] = {
            "lag": int(chosen["lag"]),
            "alpha": float(chosen["alpha"]),
            "saturation": str(chosen["saturation"]),
        }
    diagnostics_df = pd.DataFrame(diagnostics).sort_values(
        ["channel", "selection_score", "sign_consistency", "non_zero_coef_folds"],
        ascending=[True, False, False, False],
    )
    return params, diagnostics_df


def select_pooled_media_transform(
    train_df: pd.DataFrame,
    media_cols: List[str],
    control_cols: List[str],
    city_cols: List[str],
) -> Tuple[Dict[str, Dict[str, float | int | str]], pd.DataFrame]:
    ordered = train_df.sort_values(["ciudad", "semana_inicio"]).copy()
    cv_indices = panel_time_cv_indices(ordered[["semana_inicio"]].copy(), n_splits=3)
    diagnostics: list[dict[str, float | int | str]] = []

    for candidate in TRANSFORM_GRID:
        params = {
            media_col: {
                "lag": int(candidate["lag"]),
                "alpha": float(candidate["alpha"]),
                "saturation": str(candidate["saturation"]),
            }
            for media_col in media_cols
        }
        transformed_df, media_feature_cols = prepare_transformed_dataset(ordered, media_cols, params)
        fold_metrics: list[dict[str, float]] = []
        fold_negative_media = []
        for train_idx, valid_idx in cv_indices:
            fold_train = transformed_df.iloc[train_idx].copy()
            fold_valid = transformed_df.iloc[valid_idx].copy()
            feature_columns = list(
                build_feature_frame(
                    fold_train,
                    control_cols,
                    city_cols,
                    media_feature_columns=media_feature_cols,
                ).columns
            )
            scaler, model = fit_elastic_net(fold_train, feature_columns)
            pred = predict_with_model(fold_valid, feature_columns, scaler, model)
            fold_metrics.append(compute_metrics(fold_valid["ventas_netas"], pred))
            coef = original_scale_coefficients(scaler, model, feature_columns)
            media_coef = coef[[column for column in feature_columns if column.startswith("media_")]]
            fold_negative_media.append(int((media_coef < -1e-8).sum()))

        diagnostics.append(
            {
                "lag": int(candidate["lag"]),
                "alpha": float(candidate["alpha"]),
                "saturation": str(candidate["saturation"]),
                "cv_mean_mape": float(np.mean([metric["mape"] for metric in fold_metrics])),
                "cv_mean_rmse": float(np.mean([metric["rmse"] for metric in fold_metrics])),
                "cv_mean_bias": float(np.mean([metric["bias"] for metric in fold_metrics])),
                "cv_max_negative_media_coefficients": int(max(fold_negative_media)),
                "cv_mean_negative_media_coefficients": float(np.mean(fold_negative_media)),
            }
        )

    diagnostics_df = pd.DataFrame(diagnostics).sort_values(
        ["cv_max_negative_media_coefficients", "cv_mean_mape", "cv_mean_rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = diagnostics_df.iloc[0].to_dict()
    specs = {
        media_col: {
            "lag": int(best["lag"]),
            "alpha": float(best["alpha"]),
            "saturation": str(best["saturation"]),
        }
        for media_col in media_cols
    }
    return specs, diagnostics_df


def prepare_transformed_dataset(
    df: pd.DataFrame,
    media_cols: List[str],
    params: Dict[str, Dict[str, float | int | str]],
) -> Tuple[pd.DataFrame, List[str]]:
    transformed_df = add_transformed_media_features(df, media_cols, params)
    feature_columns = sorted(
        [column for column in transformed_df.columns if "__lag" in column and column.startswith("media_")]
    )
    return transformed_df, feature_columns


def evaluate_spec(
    spec_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    full_df: pd.DataFrame,
    control_cols: List[str],
    city_cols: List[str],
    media_cols: List[str],
    use_transforms: bool = False,
    include_media: bool = True,
    transform_allow_negative: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, ModelPackage | None, pd.DataFrame | None]:
    scored = valid_df[["semana_inicio", "ciudad", "ventas_netas"]].copy()

    if spec_name == "NaiveRolling4":
        scored["pred"] = full_df.loc[valid_df.index, "naive_pred"].to_numpy()
        metrics = compute_metrics(scored["ventas_netas"], scored["pred"])
        metrics["negative_media_coefficients"] = 0
        return metrics, scored, None, None

    media_params: Dict[str, Dict[str, float | int | str]] = {}
    working_df = full_df.copy()
    diagnostics = None

    if include_media and use_transforms:
        media_params, diagnostics = select_media_transforms(
            train_df,
            media_cols,
            control_cols,
            city_cols,
            allow_negative=transform_allow_negative,
        )
        working_df, media_feature_columns = prepare_transformed_dataset(working_df, media_cols, media_params)
    elif include_media:
        media_feature_columns = media_cols
    else:
        media_feature_columns = []

    train_work = working_df.loc[train_df.index].copy()
    valid_work = working_df.loc[valid_df.index].copy()
    feature_columns = list(
        build_feature_frame(
            train_work,
            control_cols,
            city_cols,
            media_feature_columns=media_feature_columns,
        ).columns
    )
    scaler, model = fit_elastic_net(train_work, feature_columns)
    scored["pred"] = predict_with_model(valid_work, feature_columns, scaler, model)
    metrics = compute_metrics(scored["ventas_netas"], scored["pred"])

    coef = original_scale_coefficients(scaler, model, feature_columns)
    media_coef = coef[[column for column in feature_columns if column.startswith("media_")]]
    metrics["negative_media_coefficients"] = int((media_coef < -1e-8).sum())
    metrics["alpha"] = float(model.alpha_)
    metrics["l1_ratio"] = float(model.l1_ratio_)

    package = ModelPackage(
        scaler=scaler,
        model=model,
        feature_columns=feature_columns,
        media_feature_columns=media_feature_columns,
        control_columns=control_cols + city_cols,
        media_params=media_params,
        city_dummy_columns=city_cols,
        spec_name=spec_name,
    )
    return metrics, scored, package, diagnostics


def choose_winner(summary: pd.DataFrame) -> str:
    pivot = summary.groupby("spec").agg(
        mean_mape=("mape", "mean"),
        mean_rmse=("rmse", "mean"),
        max_negative_media_coefficients=("negative_media_coefficients", "max"),
    ).reset_index()
    eligible = pivot[pivot["max_negative_media_coefficients"] == 0].copy()
    if eligible.empty:
        eligible = pivot.copy()
    eligible = eligible.sort_values(["mean_mape", "mean_rmse"])
    return str(eligible.iloc[0]["spec"])


def fit_final_model(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    spec_name: str,
    control_cols: List[str],
    city_cols: List[str],
    media_cols: List[str],
    transform_allow_negative: bool = False,
) -> Tuple[ModelPackage, pd.DataFrame | None]:
    if spec_name == "NaiveRolling4":
        raise ValueError("Naive model cannot be the final MMM deployment model.")

    if spec_name == "ElasticNetControls":
        return evaluate_spec(
            spec_name,
            train_df,
            train_df,
            full_df,
            control_cols,
            city_cols,
            media_cols,
            use_transforms=False,
            include_media=False,
            transform_allow_negative=transform_allow_negative,
        )[2:]

    use_transforms = spec_name == "ElasticNetTransformedMedia"
    return evaluate_spec(
        spec_name,
        train_df,
        train_df,
        full_df,
        control_cols,
        city_cols,
        media_cols,
        use_transforms=use_transforms,
        include_media=True,
        transform_allow_negative=transform_allow_negative,
    )[2:]


def predict_package(df: pd.DataFrame, package: ModelPackage) -> Tuple[pd.DataFrame, pd.Series, float]:
    working = df.copy()
    if package.media_params:
        working, _ = prepare_transformed_dataset(working, media_columns(working), package.media_params)
    coef = original_scale_coefficients(package.scaler, package.model, package.feature_columns)
    intercept = original_scale_intercept(package.scaler, package.model, package.feature_columns)
    contributions = working[package.feature_columns].mul(coef, axis=1)
    pred = contributions.sum(axis=1) + intercept
    return contributions, pred, intercept
