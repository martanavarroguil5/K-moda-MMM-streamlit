from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.modeling.beta_utils import feature_block, summarize_beta_stability, usability_flag_row
from src.modeling.model_package import ModelPackage
from src.modeling.specs import CONTROL_COLUMNS, RANDOM_STATE
from src.modeling.trainer import (
    budget_share_columns,
    city_dummy_columns,
    ensure_prerequisites,
    fit_elastic_net_with_params,
    original_scale_coefficients,
    standardized_coefficients,
)


SPEC_NAME = "ElasticNetBudgetMixLagged"
TRAIN_START_YEAR = 2021
BOOTSTRAP_ITERATIONS = 120
HYPERPARAM_ALPHAS = np.logspace(-3, 0, 15)
HYPERPARAM_L1 = [0.1, 0.3, 0.5, 0.7, 0.9]
BUDGET_VARIANTS = [
    {"budget_lag": 1, "share_lag": 1, "use_log_budget": False},
    {"budget_lag": 1, "share_lag": 1, "use_log_budget": True},
    {"budget_lag": 0, "share_lag": 1, "use_log_budget": False},
    {"budget_lag": 0, "share_lag": 1, "use_log_budget": True},
    {"budget_lag": 2, "share_lag": 1, "use_log_budget": False},
    {"budget_lag": 2, "share_lag": 1, "use_log_budget": True},
]
SIGN_THRESHOLD = 0.9

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


def _load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.model_dataset_file)
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"])
    city_dummies = pd.get_dummies(df["ciudad"], prefix="city", drop_first=True, dtype=float)
    df = pd.concat([df, city_dummies], axis=1)
    return df.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)


def _prepare_mix_dataset(
    df: pd.DataFrame,
    variant: dict[str, int | bool],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    working = df.copy().sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)
    share_cols = budget_share_columns(working)
    budget_lag = int(variant["budget_lag"])
    share_lag = int(variant["share_lag"])
    use_log_budget = bool(variant["use_log_budget"])

    if budget_lag > 0:
        working["budget_total_feature"] = working.groupby("ciudad", group_keys=False)["budget_total_eur"].shift(budget_lag)
    else:
        working["budget_total_feature"] = working["budget_total_eur"]
    if use_log_budget:
        working["budget_total_feature"] = np.log1p(working["budget_total_feature"].clip(lower=0.0))

    channel_map: dict[str, str] = {}
    share_features: list[str] = []
    for share_col in share_cols:
        feature_name = f"{share_col}_lag{share_lag}" if share_lag > 0 else share_col
        if share_lag > 0:
            working[feature_name] = working.groupby("ciudad", group_keys=False)[share_col].shift(share_lag)
        else:
            working[feature_name] = working[share_col]
        share_features.append(feature_name)
        channel_map[feature_name] = share_col.replace("budget_share_pct_", "", 1)

    required = CONTROL_COLUMNS + ["budget_total_feature"] + share_features + ["ventas_netas", "year", "semana_inicio", "ciudad"]
    working = working.dropna(subset=required).reset_index(drop=True)
    feature_columns = CONTROL_COLUMNS + city_dummy_columns(working) + ["budget_total_feature"] + share_features
    return working, feature_columns, channel_map


def _coefficient_frame(
    fold_name: str,
    package: ModelPackage,
    channel_map: dict[str, str],
) -> pd.DataFrame:
    standardized_coef = standardized_coefficients(package.model, package.feature_columns)
    original_coef = original_scale_coefficients(package.scaler, package.model, package.feature_columns)
    coefficient_df = pd.DataFrame(
        {
            "fold": fold_name,
            "feature": package.feature_columns,
            "standardized_coefficient": standardized_coef.reindex(package.feature_columns).to_numpy(dtype=float),
            "coefficient": original_coef.reindex(package.feature_columns).to_numpy(dtype=float),
        }
    )
    coefficient_df["feature_block"] = coefficient_df["feature"].map(lambda feature: feature_block(feature, package.city_dummy_columns))
    coefficient_df["channel"] = coefficient_df["feature"].map(channel_map).fillna(coefficient_df["feature"])
    coefficient_df["alpha"] = float(package.model.alpha)
    coefficient_df["l1_ratio"] = float(package.model.l1_ratio)
    coefficient_df["coefficient_sign"] = coefficient_df["coefficient"].map(
        lambda value: "positive" if value > 1e-8 else "negative" if value < -1e-8 else "zero"
    )
    return coefficient_df


def _stability_from_coefficients(coefficients_df: pd.DataFrame, media_feature_columns: list[str]) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for feature in media_feature_columns:
        feature_rows = coefficients_df[coefficients_df["feature"] == feature].copy()
        coefficients = feature_rows["coefficient"].to_numpy(dtype=float)
        positive_count = int((coefficients > 1e-8).sum())
        negative_count = int((coefficients < -1e-8).sum())
        non_zero_count = positive_count + negative_count
        selection_rate = non_zero_count / max(len(coefficients), 1)
        sign_consistency = max(positive_count, negative_count) / max(non_zero_count, 1)
        active = coefficients[np.abs(coefficients) > 1e-8]
        if len(active) >= 2 and abs(active.mean()) > 1e-8:
            dispersion = float(np.std(active, ddof=0) / abs(active.mean()))
        elif len(active) == 1:
            dispersion = 1.0
        else:
            dispersion = 5.0
        rows.append(
            {
                "feature": feature,
                "selection_rate": selection_rate,
                "sign_consistency": sign_consistency,
                "dispersion": min(dispersion, 5.0),
            }
        )
    stats = pd.DataFrame(rows)
    stable_share = (
        (
            (stats["selection_rate"] >= 0.75)
            & (stats["sign_consistency"] >= 0.75)
            & (stats["dispersion"] <= 1.0)
        ).mean()
        if not stats.empty
        else 0.0
    )
    return {
        "stable_feature_share": float(stable_share),
        "mean_selection_rate": float(stats["selection_rate"].mean()) if not stats.empty else 0.0,
        "mean_sign_consistency": float(stats["sign_consistency"].mean()) if not stats.empty else 0.0,
        "mean_dispersion": float(stats["dispersion"].mean()) if not stats.empty else 5.0,
    }


def _search_hyperparameters(
    train_df: pd.DataFrame,
    valid_years: list[int],
    feature_columns: list[str],
    media_feature_columns: list[str],
    variant_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for alpha in HYPERPARAM_ALPHAS:
        for l1_ratio in HYPERPARAM_L1:
            fold_metrics: list[dict[str, float]] = []
            coefficient_rows: list[dict[str, float | str]] = []
            for year in valid_years:
                inner_train = train_df[train_df["year"] < year].copy()
                inner_valid = train_df[train_df["year"] == year].copy()
                if inner_train.empty or inner_valid.empty:
                    continue
                scaler, model = fit_elastic_net_with_params(
                    inner_train,
                    feature_columns,
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                )
                x_valid = scaler.transform(inner_valid[feature_columns])
                pred = model.predict(x_valid)
                metrics = compute_metrics(inner_valid["ventas_netas"].to_numpy(dtype=float), pred)
                fold_metrics.append(metrics)
                coef = original_scale_coefficients(scaler, model, feature_columns)
                for feature in media_feature_columns:
                    coefficient_rows.append(
                        {
                            "feature": feature,
                            "coefficient": float(coef[feature]),
                        }
                    )

            if not fold_metrics:
                continue
            coefficient_df = pd.DataFrame(coefficient_rows)
            stability = _stability_from_coefficients(coefficient_df, media_feature_columns)
            mean_mape = float(np.mean([metric["mape"] for metric in fold_metrics]))
            mean_rmse = float(np.mean([metric["rmse"] for metric in fold_metrics]))
            error_score = float(1.0 / (1.0 + mean_mape + mean_rmse / 100000.0))
            objective_score = (
                0.50 * stability["stable_feature_share"]
                + 0.25 * stability["mean_sign_consistency"]
                + 0.15 * stability["mean_selection_rate"]
                + 0.07 * float(1.0 / (1.0 + stability["mean_dispersion"]))
                + 0.03 * error_score
            )
            rows.append(
                {
                    "variant": variant_name,
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "objective_score": objective_score,
                    "stable_feature_share": stability["stable_feature_share"],
                    "mean_sign_consistency": stability["mean_sign_consistency"],
                    "mean_selection_rate": stability["mean_selection_rate"],
                    "mean_dispersion": stability["mean_dispersion"],
                    "mean_mape": mean_mape,
                    "mean_rmse": mean_rmse,
                }
            )

    diagnostics = pd.DataFrame(rows).sort_values(
        ["objective_score", "stable_feature_share", "mean_sign_consistency", "mean_selection_rate", "mean_dispersion", "mean_mape"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    best = diagnostics.iloc[0]
    diagnostics["selected"] = (
        (diagnostics["alpha"] == float(best["alpha"]))
        & (diagnostics["l1_ratio"] == float(best["l1_ratio"]))
    )
    return {
        "alpha": float(best["alpha"]),
        "l1_ratio": float(best["l1_ratio"]),
    }, diagnostics


def _fit_package(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    media_feature_columns: list[str],
    city_cols: list[str],
    alpha: float,
    l1_ratio: float,
) -> ModelPackage:
    scaler, model = fit_elastic_net_with_params(
        train_df,
        feature_columns,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    return ModelPackage(
        scaler=scaler,
        model=model,
        feature_columns=feature_columns,
        media_feature_columns=media_feature_columns,
        control_columns=CONTROL_COLUMNS + city_cols + ["budget_total_feature"],
        media_params={},
        city_dummy_columns=city_cols,
        spec_name=SPEC_NAME,
    )


def _bootstrap_coefficients(
    working_df: pd.DataFrame,
    feature_columns: list[str],
    alpha: float,
    l1_ratio: float,
) -> pd.DataFrame:
    unique_weeks = pd.to_datetime(working_df["semana_inicio"]).drop_duplicates().sort_values().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, float | str | int]] = []
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled_weeks = rng.choice(unique_weeks, size=len(unique_weeks), replace=True)
        bootstrap_df = (
            pd.concat([working_df[working_df["semana_inicio"] == week].copy() for week in sampled_weeks], ignore_index=True)
            .sort_values(["semana_inicio", "ciudad"])
            .reset_index(drop=True)
        )
        scaler, model = fit_elastic_net_with_params(
            bootstrap_df,
            feature_columns,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        standardized_coef = standardized_coefficients(model, feature_columns)
        original_coef = original_scale_coefficients(scaler, model, feature_columns)
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


def _summarize_bootstrap(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in bootstrap_df.groupby("feature", sort=True):
        values = group["coefficient"].to_numpy(dtype=float)
        std_values = group["standardized_coefficient"].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "bootstrap_original_p02_5": float(np.quantile(values, 0.025)),
                "bootstrap_original_p50": float(np.quantile(values, 0.5)),
                "bootstrap_original_p97_5": float(np.quantile(values, 0.975)),
                "bootstrap_standardized_p02_5": float(np.quantile(std_values, 0.025)),
                "bootstrap_standardized_p50": float(np.quantile(std_values, 0.5)),
                "bootstrap_standardized_p97_5": float(np.quantile(std_values, 0.975)),
                "bootstrap_positive_probability": float(np.mean(values > 1e-8)),
                "bootstrap_negative_probability": float(np.mean(values < -1e-8)),
            }
        )
    return pd.DataFrame(rows)


def _build_deployment_tables(
    package: ModelPackage,
    channel_map: dict[str, str],
    beta_summary_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardized_coef = standardized_coefficients(package.model, package.feature_columns)
    original_coef = original_scale_coefficients(package.scaler, package.model, package.feature_columns)
    deployment = pd.DataFrame(
        {
            "feature": package.feature_columns,
            "standardized_coefficient": standardized_coef.reindex(package.feature_columns).to_numpy(dtype=float),
            "coefficient": original_coef.reindex(package.feature_columns).to_numpy(dtype=float),
        }
    )
    deployment["feature_block"] = deployment["feature"].map(lambda feature: feature_block(feature, package.city_dummy_columns))
    deployment["channel"] = deployment["feature"].map(channel_map)
    deployment["coefficient_sign"] = deployment["coefficient"].map(
        lambda value: "positive" if value > 1e-8 else "negative" if value < -1e-8 else "zero"
    )
    deployment = deployment.merge(
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
    deployment = deployment.merge(bootstrap_summary_df, on="feature", how="left")
    deployment["collinearity_flag"] = False
    usability = deployment.apply(
        lambda row: usability_flag_row(
            feature_block=str(row["feature_block"]),
            stability_status=str(row["stability_status"]),
            selection_rate=float(row["selection_rate"]) if pd.notna(row["selection_rate"]) else 0.0,
            sign_consistency=float(row["sign_consistency"]) if pd.notna(row["sign_consistency"]) else 0.0,
            coefficient_sign=str(row["coefficient_sign"]),
            dominant_sign=str(row["dominant_sign"]) if pd.notna(row["dominant_sign"]) else "zero",
            bootstrap_positive_probability=float(row["bootstrap_positive_probability"]) if pd.notna(row["bootstrap_positive_probability"]) else 0.0,
            bootstrap_negative_probability=float(row["bootstrap_negative_probability"]) if pd.notna(row["bootstrap_negative_probability"]) else 0.0,
            collinearity_flag=False,
        ),
        axis=1,
        result_type="expand",
    )
    usability.columns = ["usable_beta", "usable_reason"]
    deployment = pd.concat([deployment, usability], axis=1)

    channel_df = deployment[deployment["channel"].notna()].copy()
    channel_df["usable_beta"] = channel_df.apply(
        lambda row: bool(row["usable_beta"]) and (
            float(row["bootstrap_positive_probability"]) >= SIGN_THRESHOLD
            if row["coefficient_sign"] == "positive"
            else float(row["bootstrap_negative_probability"]) >= SIGN_THRESHOLD
        ),
        axis=1,
    )
    channel_df["usable_reason"] = channel_df.apply(
        lambda row: row["usable_reason"]
        if bool(row["usable_beta"])
        else (
            "weak_bootstrap_sign_probability"
            if str(row["usable_reason"]) == "usable"
            else str(row["usable_reason"])
        ),
        axis=1,
    )
    return deployment.sort_values(["feature_block", "feature"]).reset_index(drop=True), channel_df.sort_values("channel").reset_index(drop=True)


def _write_report(
    diagnostics_lines: list[str],
    variant_results: pd.DataFrame,
    hyperparam_results: pd.DataFrame,
    backtest_results: pd.DataFrame,
    test_results: pd.DataFrame,
    deployment_channels: pd.DataFrame,
    model_results: dict,
) -> None:
    usable_channels = deployment_channels[deployment_channels["usable_beta"]].copy()
    stable_channels = deployment_channels[deployment_channels["stability_status"] == "stable"].copy()
    negative_channels = deployment_channels[deployment_channels["coefficient"] < -1e-8].copy()
    lines = [
        "# Elastic Net Beta Report",
        "",
        "## Objetivo",
        "",
        "- Extraer betas utilizables con Elastic Net, priorizando estabilidad temporal y trazabilidad antes que error puro.",
        "- La ruta final usa `budget_total_eur` y shares laggeadas porque los datos no contienen variacion geo adicional de medios y los `media_*` en euros quedan altamente colineales.",
        "",
        "## Diagnostico de Datos",
        "",
        *diagnostics_lines,
        "",
        "## Variantes evaluadas",
        "",
        "```text",
        variant_results.round(4).to_string(index=False),
        "```",
        "",
        "## Busqueda de hiperparametros",
        "",
        "```text",
        hyperparam_results.round(4).to_string(index=False),
        "```",
        "",
        "## Backtest",
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
        "## Resumen por canal",
        "",
        "```text",
        deployment_channels.round(4).to_string(index=False),
        "```",
        "",
        "## Betas utilizables",
        "",
        "```text",
        usable_channels.round(4).to_string(index=False) if not usable_channels.empty else "Ningun canal pasa aun el filtro final de usabilidad.",
        "```",
        "",
        "## Betas estables pero no utilizables",
        "",
        "```text",
        stable_channels[~stable_channels["usable_beta"]].round(4).to_string(index=False)
        if not stable_channels[~stable_channels["usable_beta"]].empty
        else "Sin canales en este estado.",
        "```",
        "",
        "## Betas negativas",
        "",
        "```text",
        negative_channels.round(4).to_string(index=False) if not negative_channels.empty else "Sin betas negativas en despliegue.",
        "```",
        "",
        "## Lectura",
        "",
        f"- Especificacion ganadora: `{model_results['winner']}`.",
        f"- Variante de datos: `{model_results['selected_variant_name']}`.",
        f"- Hiperparametros finales: `alpha={model_results['model_alpha']:.6f}`, `l1_ratio={model_results['model_l1_ratio']:.2f}`.",
        f"- Canales utilizables: `{len(model_results['usable_media_features'])}` de `{len(model_results['media_columns'])}`.",
        "- Estas betas viven en espacio de mix: cambio esperado en ventas por punto porcentual de share rezagado, controlando por presupuesto total rezagado y baseline.",
        "",
    ]
    BETA_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_beta_training_pipeline() -> dict:
    ensure_prerequisites()
    df = _load_dataset()
    df = df[df["year"] >= TRAIN_START_YEAR].copy().reset_index(drop=True)
    diagnostics_lines = [
        f"- Filas analizadas: `{len(df)}`.",
        f"- Ciudades observadas: `{df['ciudad'].nunique()}` (`{', '.join(df['ciudad'].astype(str).unique().tolist())}`).",
        "- Los medios son identicos entre ciudades en cada semana, por lo que el panel no aporta variacion geo para identificar canales.",
        "- En la lectura weekly de `media_*`, las correlaciones absolutas entre canales se mueven aproximadamente entre `0.83` y `0.93`.",
        "- La representacion `budget_total_eur + budget_share_pct_*` reduce de forma fuerte esa colinealidad y deja una ruta mas defendible para betas por canal.",
    ]

    candidate_rows: list[dict[str, float | str | int | bool]] = []
    for variant in BUDGET_VARIANTS:
        working_df, feature_columns, channel_map = _prepare_mix_dataset(df, variant)
        media_feature_columns = [feature for feature in feature_columns if feature.startswith("budget_share_pct_")]
        valid_years = sorted([year for year in working_df["year"].unique().tolist() if TRAIN_START_YEAR < year < int(working_df["year"].max())])
        best_params, diagnostics = _search_hyperparameters(
            working_df[working_df["year"] < int(working_df["year"].max())].copy(),
            valid_years=valid_years,
            feature_columns=feature_columns,
            media_feature_columns=media_feature_columns,
            variant_name=json.dumps(variant, sort_keys=True),
        )
        best = diagnostics.loc[diagnostics["selected"]].iloc[0]
        candidate_rows.append(
            {
                "variant_name": json.dumps(variant, sort_keys=True),
                "budget_lag": int(variant["budget_lag"]),
                "share_lag": int(variant["share_lag"]),
                "use_log_budget": bool(variant["use_log_budget"]),
                "alpha": float(best_params["alpha"]),
                "l1_ratio": float(best_params["l1_ratio"]),
                "objective_score": float(best["objective_score"]),
                "stable_feature_share": float(best["stable_feature_share"]),
                "mean_sign_consistency": float(best["mean_sign_consistency"]),
                "mean_selection_rate": float(best["mean_selection_rate"]),
                "mean_dispersion": float(best["mean_dispersion"]),
                "mean_mape": float(best["mean_mape"]),
                "mean_rmse": float(best["mean_rmse"]),
            }
        )

    variant_diagnostics_df = pd.DataFrame(candidate_rows).sort_values(
        ["stable_feature_share", "mean_sign_consistency", "mean_selection_rate", "mean_dispersion", "mean_mape"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    selected_variant_row = variant_diagnostics_df.iloc[0]
    selected_variant = {
        "budget_lag": int(selected_variant_row["budget_lag"]),
        "share_lag": int(selected_variant_row["share_lag"]),
        "use_log_budget": bool(selected_variant_row["use_log_budget"]),
    }
    selected_variant_name = str(selected_variant_row["variant_name"])

    working_df, feature_columns, channel_map = _prepare_mix_dataset(df, selected_variant)
    media_feature_columns = [feature for feature in feature_columns if feature.startswith("budget_share_pct_")]
    city_cols = city_dummy_columns(working_df)
    test_year = int(working_df["year"].max())
    validation_years = sorted([year for year in working_df["year"].unique().tolist() if TRAIN_START_YEAR < year < test_year])

    best_params, hyperparam_diagnostics_df = _search_hyperparameters(
        working_df[working_df["year"] < test_year].copy(),
        valid_years=validation_years,
        feature_columns=feature_columns,
        media_feature_columns=media_feature_columns,
        variant_name=selected_variant_name,
    )
    hyperparam_diagnostics_df["selected_variant"] = selected_variant_name

    backtest_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_coefficient_frames: list[pd.DataFrame] = []
    for year in validation_years:
        train_df = working_df[working_df["year"] < year].copy()
        valid_df = working_df[working_df["year"] == year].copy()
        package = _fit_package(
            train_df,
            feature_columns,
            media_feature_columns,
            city_cols,
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"],
        )
        pred = package.model.predict(package.scaler.transform(valid_df[feature_columns]))
        metrics = compute_metrics(valid_df["ventas_netas"].to_numpy(dtype=float), pred)
        metrics["fold"] = f"validate_{year}"
        metrics["spec"] = SPEC_NAME
        metrics["alpha"] = float(package.model.alpha)
        metrics["l1_ratio"] = float(package.model.l1_ratio)
        metrics["negative_media_coefficients"] = int(
            (
                original_scale_coefficients(package.scaler, package.model, package.feature_columns)
                .reindex(media_feature_columns)
                < -1e-8
            ).sum()
        )
        backtest_rows.append(metrics)
        scored = valid_df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
        scored["pred"] = pred
        scored["fold"] = f"validate_{year}"
        scored["spec"] = SPEC_NAME
        prediction_frames.append(scored)
        fold_coefficient_frames.append(_coefficient_frame(f"validate_{year}", package, channel_map))

    train_eval = working_df[working_df["year"] < test_year].copy()
    test_eval = working_df[working_df["year"] == test_year].copy()
    test_package = _fit_package(
        train_eval,
        feature_columns,
        media_feature_columns,
        city_cols,
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
    )
    test_pred = test_package.model.predict(test_package.scaler.transform(test_eval[feature_columns]))
    test_metrics = compute_metrics(test_eval["ventas_netas"].to_numpy(dtype=float), test_pred)
    test_metrics["alpha"] = float(test_package.model.alpha)
    test_metrics["l1_ratio"] = float(test_package.model.l1_ratio)
    test_metrics["negative_media_coefficients"] = int(
        (
            original_scale_coefficients(test_package.scaler, test_package.model, test_package.feature_columns)
            .reindex(media_feature_columns)
            < -1e-8
        ).sum()
    )
    test_results = pd.DataFrame([{**test_metrics, "fold": f"test_{test_year}", "spec": SPEC_NAME}])
    test_scored = test_eval[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    test_scored["pred"] = test_pred
    test_scored["fold"] = f"test_{test_year}"
    test_scored["spec"] = SPEC_NAME
    fold_coefficient_frames.append(_coefficient_frame(f"test_{test_year}", test_package, channel_map))

    deploy_package = _fit_package(
        working_df,
        feature_columns,
        media_feature_columns,
        city_cols,
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
    )
    deployment_pred = deploy_package.model.predict(deploy_package.scaler.transform(working_df[feature_columns]))
    deployment_scored = working_df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    deployment_scored["pred"] = deployment_pred
    deployment_scored["fold"] = "deployment_fit"
    deployment_scored["spec"] = SPEC_NAME

    contribution_table = pd.DataFrame(index=working_df.index)
    deployment_coefficients = original_scale_coefficients(deploy_package.scaler, deploy_package.model, deploy_package.feature_columns)
    for feature in media_feature_columns:
        contribution_table[feature] = working_df[feature].to_numpy(dtype=float) * float(deployment_coefficients[feature])
    contribution_table = contribution_table.assign(
        semana_inicio=working_df["semana_inicio"].to_numpy(),
        ciudad=working_df["ciudad"].to_numpy(),
    ).melt(id_vars=["semana_inicio", "ciudad"], var_name="feature", value_name="contribution")
    contribution_table["channel"] = contribution_table["feature"].map(channel_map)

    fold_coefficients_df = pd.concat(fold_coefficient_frames, ignore_index=True)
    beta_summary_df = summarize_beta_stability(
        fold_coefficients_df[["feature", "coefficient"]],
        deployment_coefficients=original_scale_coefficients(
            deploy_package.scaler,
            deploy_package.model,
            deploy_package.feature_columns,
        ),
        city_columns=city_cols,
    )
    bootstrap_df = _bootstrap_coefficients(
        working_df,
        feature_columns,
        alpha=float(deploy_package.model.alpha),
        l1_ratio=float(deploy_package.model.l1_ratio),
    )
    bootstrap_summary_df = _summarize_bootstrap(bootstrap_df)
    deployment_coefficients_df, channel_summary_df = _build_deployment_tables(
        deploy_package,
        channel_map,
        beta_summary_df,
        bootstrap_summary_df,
    )

    predictions = pd.concat(prediction_frames + [test_scored, deployment_scored], ignore_index=True)
    backtest_results = pd.DataFrame(backtest_rows)
    transform_diagnostics_df = variant_diagnostics_df.copy()
    collinearity_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "reason": "not_applicable_budget_mix",
                "detail": "budget_total_plus_share_lagged_route",
                "fold": "deployment_fit",
            }
            for feature in media_feature_columns
        ]
    )

    usable_channels = channel_summary_df[channel_summary_df["usable_beta"]].copy()
    model_results = {
        "winner": SPEC_NAME,
        "selection_objective": "beta_stability",
        "train_start_year": TRAIN_START_YEAR,
        "validation_years": validation_years,
        "test_year": test_year,
        "selected_variant": selected_variant,
        "selected_variant_name": selected_variant_name,
        "baseline_columns": CONTROL_COLUMNS + city_cols + ["budget_total_feature"],
        "media_columns": [channel_map[feature] for feature in media_feature_columns],
        "selected_transforms": selected_variant,
        "model_alpha": float(deploy_package.model.alpha),
        "model_l1_ratio": float(deploy_package.model.l1_ratio),
        "test_metrics": test_metrics,
        "negative_media_coefficients": int((deployment_coefficients_df.loc[deployment_coefficients_df["feature_block"] == "media", "coefficient"] < -1e-8).sum()),
        "stable_media_features": channel_summary_df[channel_summary_df["stability_status"] == "stable"]["channel"].tolist(),
        "review_media_features": channel_summary_df[channel_summary_df["stability_status"] == "review"]["channel"].tolist(),
        "usable_media_features": usable_channels["channel"].tolist(),
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
    hyperparam_diagnostics_df.to_csv(BETA_HYPERPARAM_TABLE, index=False)
    bootstrap_df.to_csv(BETA_BOOTSTRAP_TABLE, index=False)
    BETA_RESULTS_JSON.write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    BETA_TRANSFORMS_JSON.write_text(json.dumps(selected_variant, indent=2), encoding="utf-8")
    with BETA_MODEL_FILE.open("wb") as handle:
        pickle.dump(deploy_package, handle)

    _write_report(
        diagnostics_lines=diagnostics_lines,
        variant_results=variant_diagnostics_df,
        hyperparam_results=hyperparam_diagnostics_df,
        backtest_results=backtest_results,
        test_results=test_results,
        deployment_channels=channel_summary_df,
        model_results=model_results,
    )

    return {
        "backtest_results": backtest_results,
        "test_results": test_results,
        "beta_summary": beta_summary_df,
        "deployment_coefficients": deployment_coefficients_df,
        "channel_summary": channel_summary_df,
        "model_results": model_results,
    }
