from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.features.geo_dataset_builder import build_geo_model_dataset
from src.modeling.identifiability_audit import run_identifiability_audit
from src.modeling.selection import prepare_transformed_dataset, select_media_transforms, select_pooled_media_transform
from src.modeling.specs import CONTROL_COLUMNS
from src.modeling.trainer import media_columns


RESULTS_TABLE = CONFIG.reports_tables_dir / "hierarchical_mmm_model_comparison.csv"
COEFFICIENT_TABLE = CONFIG.reports_tables_dir / "hierarchical_mmm_coefficients.csv"
QUALITY_TABLE = CONFIG.reports_tables_dir / "hierarchical_mmm_quality_checks.csv"
ELIGIBILITY_TABLE = CONFIG.reports_tables_dir / "hierarchical_mmm_channel_eligibility.csv"
TRANSFORM_TABLE = CONFIG.reports_tables_dir / "hierarchical_mmm_selected_transforms.csv"
REPORT_MD = CONFIG.docs_dir / "hierarchical_mmm_report.md"

TARGET_YEAR = 2024
DEFAULT_TRAIN_START_YEAR = 2021
DRAW_COUNT = 600
TUNE_COUNT = 1000
CHAIN_COUNT = 4
TARGET_ACCEPT = 0.99
RANDOM_SEED = 42


@dataclass(frozen=True)
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    control_columns: list[str]
    media_feature_columns: list[str]
    geo_labels: list[str]
    geo_index_train: np.ndarray
    geo_index_test: np.ndarray
    trend_train: np.ndarray
    trend_test: np.ndarray
    trend_sq_train: np.ndarray
    trend_sq_test: np.ndarray
    x_controls_train: np.ndarray
    x_controls_test: np.ndarray
    x_media_train: np.ndarray
    x_media_test: np.ndarray
    y_train_scaled: np.ndarray
    y_test_actual: np.ndarray
    y_mean: float
    y_std: float
    media_scales: dict[str, float]
    control_scaler: StandardScaler


def _load_geo_dataset() -> pd.DataFrame:
    if not CONFIG.geo_model_dataset_file.exists():
        build_geo_model_dataset()
    df = pd.read_parquet(CONFIG.geo_model_dataset_file).copy()
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"])
    return df.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)


def _add_city_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    city_dummies = pd.get_dummies(df["ciudad"], prefix="city", drop_first=True, dtype=float)
    out = pd.concat([df.reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)
    city_cols = sorted([column for column in out.columns if column.startswith("city_")])
    return out, city_cols


def _ensure_prior_config(df: pd.DataFrame, media_cols: list[str], train_start_year: int) -> dict:
    spend_train = df.loc[(df["year"] >= train_start_year) & (df["year"] < TARGET_YEAR), media_cols].sum()
    spend_total = float(spend_train.sum())
    spend_share = spend_train / spend_total if spend_total > 0 else pd.Series(1.0 / len(media_cols), index=media_cols)

    if CONFIG.hierarchical_prior_config_file.exists():
        payload = json.loads(CONFIG.hierarchical_prior_config_file.read_text(encoding="utf-8"))
        payload["sampling"] = {
            "draws": DRAW_COUNT,
            "tune": TUNE_COUNT,
            "chains": CHAIN_COUNT,
            "target_accept": TARGET_ACCEPT,
        }
        CONFIG.hierarchical_prior_config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    channels = {}
    for media_col in media_cols:
        share = float(spend_share.get(media_col, 1.0 / len(media_cols)))
        prior_mean = float(np.clip(share * 0.80, 0.04, 0.20))
        prior_sd = float(np.clip(prior_mean * 1.50 + 0.05, 0.10, 0.40))
        channels[media_col] = {
            "prior_mean": round(prior_mean, 4),
            "prior_sd": round(prior_sd, 4),
            "calibration_source": "default_weak_business_prior",
            "requires_query_volume_proxy": media_col == "media_paid_search",
            "allow_optimization_without_proxy": False if media_col == "media_paid_search" else True,
            "notes": (
                "Paid search needs query volume or experiment calibration for strong causal interpretation."
                if media_col == "media_paid_search"
                else "Weakly informative positive prior in standardized effect units."
            ),
        }

    payload = {
        "modeling_window": {
            "train_start_year": train_start_year,
            "target_year": TARGET_YEAR,
            "exclude_structural_break_years_by_default": [2020],
        },
        "sampling": {
            "draws": DRAW_COUNT,
            "tune": TUNE_COUNT,
            "chains": CHAIN_COUNT,
            "target_accept": TARGET_ACCEPT,
        },
        "channels": channels,
    }
    CONFIG.hierarchical_prior_config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    variance_ratio = (sd * sd) / max(mean * mean, 1e-9)
    sigma2 = float(np.log1p(variance_ratio))
    sigma = float(np.sqrt(max(sigma2, 1e-9)))
    mu = float(np.log(max(mean, 1e-9)) - 0.5 * sigma2)
    return mu, sigma


def _trend_to_unit_interval(series: pd.Series, lower: float, upper: float) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    if len(values) == 0:
        return values
    if upper - lower <= 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - lower) / (upper - lower) - 0.5


def _centered_trend_square(trend_values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    squared = np.square(np.asarray(trend_values, dtype=float))
    return squared - float(np.square(np.asarray(reference, dtype=float)).mean())


def _prepare_model_data(
    transformed_df: pd.DataFrame,
    city_cols: list[str],
    media_feature_cols: list[str],
    train_start_year: int,
) -> PreparedData:
    train_df = transformed_df.loc[
        (transformed_df["year"] >= train_start_year) & (transformed_df["year"] < TARGET_YEAR)
    ].copy()
    test_df = transformed_df.loc[transformed_df["year"] == TARGET_YEAR].copy()
    geo_labels = sorted(transformed_df["ciudad"].astype(str).unique().tolist())
    geo_index = {city: idx for idx, city in enumerate(geo_labels)}

    controls_no_trend = [column for column in CONTROL_COLUMNS if column != "trend_index"]
    control_columns = controls_no_trend

    control_scaler = StandardScaler()
    x_controls_train = control_scaler.fit_transform(train_df[control_columns]).astype(float)
    x_controls_test = control_scaler.transform(test_df[control_columns]).astype(float)

    media_scales: dict[str, float] = {}
    x_media_train_list = []
    x_media_test_list = []
    for feature in media_feature_cols:
        scale = float(np.quantile(train_df[feature], 0.95))
        scale = scale if scale > 1e-9 else float(max(train_df[feature].max(), 1.0))
        media_scales[feature] = scale
        x_media_train_list.append(train_df[feature].to_numpy(dtype=float) / scale)
        x_media_test_list.append(test_df[feature].to_numpy(dtype=float) / scale)

    x_media_train = np.column_stack(x_media_train_list).astype(float)
    x_media_test = np.column_stack(x_media_test_list).astype(float)

    y_train = train_df["ventas_netas"].to_numpy(dtype=float)
    y_test = test_df["ventas_netas"].to_numpy(dtype=float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    y_std = y_std if y_std > 1e-9 else 1.0
    y_train_scaled = (y_train - y_mean) / y_std

    trend_lower = float(train_df["trend_index"].min())
    trend_upper = float(train_df["trend_index"].max())
    trend_train = _trend_to_unit_interval(train_df["trend_index"], lower=trend_lower, upper=trend_upper)
    trend_test = _trend_to_unit_interval(test_df["trend_index"], lower=trend_lower, upper=trend_upper)

    return PreparedData(
        train_df=train_df,
        test_df=test_df,
        control_columns=control_columns,
        media_feature_columns=media_feature_cols,
        geo_labels=geo_labels,
        geo_index_train=train_df["ciudad"].map(geo_index).to_numpy(dtype=int),
        geo_index_test=test_df["ciudad"].map(geo_index).to_numpy(dtype=int),
        trend_train=trend_train,
        trend_test=trend_test,
        trend_sq_train=_centered_trend_square(trend_train, reference=trend_train),
        trend_sq_test=_centered_trend_square(trend_test, reference=trend_train),
        x_controls_train=x_controls_train,
        x_controls_test=x_controls_test,
        x_media_train=x_media_train,
        x_media_test=x_media_test,
        y_train_scaled=y_train_scaled,
        y_test_actual=y_test,
        y_mean=y_mean,
        y_std=y_std,
        media_scales=media_scales,
        control_scaler=control_scaler,
    )


def _build_model(
    prepared: PreparedData,
    prior_config: dict,
) -> pm.Model:
    media_base_cols = [feature.split("__")[0] for feature in prepared.media_feature_columns]
    prior_mean = []
    prior_sigma = []
    for media_col in media_base_cols:
        channel_cfg = prior_config["channels"][media_col]
        mu, sigma = _lognormal_params(
            mean=float(channel_cfg["prior_mean"]),
            sd=float(channel_cfg["prior_sd"]),
        )
        prior_mean.append(mu)
        prior_sigma.append(sigma)

    coords = {
        "obs": np.arange(len(prepared.train_df)),
        "geo": prepared.geo_labels,
        "control": prepared.control_columns,
        "media": media_base_cols,
    }

    with pm.Model(coords=coords) as model:
        geo_idx = pm.Data("geo_idx", prepared.geo_index_train, dims="obs")
        trend = pm.Data("trend", prepared.trend_train, dims="obs")
        trend_sq = pm.Data("trend_sq", prepared.trend_sq_train, dims="obs")
        controls_input = pm.Data("controls_input", prepared.x_controls_train, dims=("obs", "control"))
        media_input = pm.Data("media_input", prepared.x_media_train, dims=("obs", "media"))

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=2.0)
        beta_trend_sq = pm.Normal("beta_trend_sq", mu=0.0, sigma=1.0)
        sigma_geo = pm.HalfNormal("sigma_geo", sigma=1.0)
        geo_intercept_raw = pm.Normal("geo_intercept_raw", mu=0.0, sigma=1.0, dims="geo")
        geo_intercept = pm.Deterministic(
            "geo_intercept",
            (geo_intercept_raw - pt.mean(geo_intercept_raw)) * sigma_geo,
            dims="geo",
        )

        sigma_geo_trend = pm.HalfNormal("sigma_geo_trend", sigma=0.50)
        geo_trend_raw = pm.Normal("geo_trend_raw", mu=0.0, sigma=1.0, dims="geo")
        geo_trend = pm.Deterministic(
            "geo_trend",
            (geo_trend_raw - pt.mean(geo_trend_raw)) * sigma_geo_trend,
            dims="geo",
        )

        beta_controls = pm.Normal("beta_controls", mu=0.0, sigma=1.0, dims="control")
        beta_media = pm.LogNormal(
            "beta_media",
            mu=np.array(prior_mean, dtype=float),
            sigma=np.array(prior_sigma, dtype=float),
            dims="media",
        )

        mu = (
            alpha
            + beta_trend * trend
            + beta_trend_sq * trend_sq
            + geo_intercept[geo_idx]
            + geo_trend[geo_idx] * trend
            + pt.sum(controls_input * beta_controls, axis=1)
            + pt.sum(media_input * beta_media, axis=1)
        )

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Exponential("nu_minus_two", lam=1 / 15.0) + 2.0
        pm.StudentT("ventas_netas_scaled", nu=nu, mu=mu, sigma=sigma, observed=prepared.y_train_scaled, dims="obs")
    return model


def _posterior_arrays(idata: az.InferenceData) -> dict[str, np.ndarray]:
    posterior = idata.posterior
    arrays = {}
    for name in [
        "alpha",
        "beta_trend",
        "beta_trend_sq",
        "geo_intercept",
        "geo_trend",
        "beta_controls",
        "beta_media",
        "sigma",
        "nu_minus_two",
    ]:
        values = posterior[name].stack(sample=("chain", "draw")).transpose(..., "sample").values
        arrays[name] = np.asarray(values, dtype=float)
    return arrays


def _predict_draws(prepared: PreparedData, arrays: dict[str, np.ndarray], on_test: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geo_idx = prepared.geo_index_test if on_test else prepared.geo_index_train
    trend = prepared.trend_test if on_test else prepared.trend_train
    trend_sq = prepared.trend_sq_test if on_test else prepared.trend_sq_train
    x_controls = prepared.x_controls_test if on_test else prepared.x_controls_train
    x_media = prepared.x_media_test if on_test else prepared.x_media_train

    alpha = arrays["alpha"].reshape(-1, 1)
    beta_trend = arrays["beta_trend"].reshape(-1, 1)
    beta_trend_sq = arrays["beta_trend_sq"].reshape(-1, 1)
    geo_intercept = arrays["geo_intercept"]
    geo_trend = arrays["geo_trend"]
    beta_controls = arrays["beta_controls"]
    beta_media = arrays["beta_media"]

    control_part = beta_controls.T @ x_controls.T
    media_part = beta_media.T @ x_media.T
    mu_scaled = (
        alpha
        + beta_trend * trend.reshape(1, -1)
        + beta_trend_sq * trend_sq.reshape(1, -1)
        + geo_intercept[geo_idx, :].T
        + geo_trend[geo_idx, :].T * trend.reshape(1, -1)
        + control_part
        + media_part
    )
    baseline_scaled = (
        alpha
        + beta_trend * trend.reshape(1, -1)
        + beta_trend_sq * trend_sq.reshape(1, -1)
        + geo_intercept[geo_idx, :].T
        + geo_trend[geo_idx, :].T * trend.reshape(1, -1)
        + control_part
    )
    contributions_scaled = beta_media.T[:, :, None] * x_media.T[None, :, :]
    return mu_scaled, baseline_scaled, contributions_scaled


def _summarize_prediction_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    lo = np.quantile(values, 0.05, axis=0)
    hi = np.quantile(values, 0.95, axis=0)
    return mean, lo, hi


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), 1.0)
    return {
        "mape": float(np.mean(np.abs(actual - pred) / denom) * 100.0),
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - pred)))),
        "bias": float(np.mean(pred - actual)),
        "r2": float(1.0 - np.square(actual - pred).sum() / np.square(actual - actual.mean()).sum()),
    }


def _weekly_prediction_table(
    prepared: PreparedData,
    pred_draws: np.ndarray,
    baseline_draws: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred_draws * prepared.y_std + prepared.y_mean
    baseline = baseline_draws * prepared.y_std + prepared.y_mean
    pred_mean, pred_lo, pred_hi = _summarize_prediction_interval(pred)
    base_mean, base_lo, base_hi = _summarize_prediction_interval(baseline)

    scored = prepared.test_df[["semana_inicio", "ciudad", "ventas_netas"]].copy().reset_index(drop=True)
    scored["pred_mean"] = pred_mean
    scored["pred_p05"] = pred_lo
    scored["pred_p95"] = pred_hi
    scored["baseline_mean"] = base_mean
    scored["baseline_p05"] = base_lo
    scored["baseline_p95"] = base_hi

    weekly = (
        scored.groupby("semana_inicio", as_index=False)
        .agg(
            actual_sales=("ventas_netas", "sum"),
            predicted_sales_mean=("pred_mean", "sum"),
            predicted_sales_p05=("pred_p05", "sum"),
            predicted_sales_p95=("pred_p95", "sum"),
            baseline_sales_mean=("baseline_mean", "sum"),
            baseline_sales_p05=("baseline_p05", "sum"),
            baseline_sales_p95=("baseline_p95", "sum"),
        )
        .sort_values("semana_inicio")
        .reset_index(drop=True)
    )
    weekly["media_incremental_mean"] = weekly["predicted_sales_mean"] - weekly["baseline_sales_mean"]
    weekly["media_incremental_p05"] = weekly["predicted_sales_p05"] - weekly["baseline_sales_p95"]
    weekly["media_incremental_p95"] = weekly["predicted_sales_p95"] - weekly["baseline_sales_p05"]
    return scored, weekly


def _channel_contribution_tables(
    prepared: PreparedData,
    contribution_draws_scaled: np.ndarray,
    prior_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contribution_draws = contribution_draws_scaled * prepared.y_std
    weeks = prepared.test_df["semana_inicio"].reset_index(drop=True)
    channel_rows = []
    aggregate_rows = []
    stability_path = CONFIG.reports_tables_dir / "identifiability_channel_stability.csv"
    stability_df = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()

    for channel_idx, feature in enumerate(prepared.media_feature_columns):
        channel = feature.split("__")[0]
        channel_short = channel.replace("media_", "", 1)
        per_obs = contribution_draws[:, channel_idx, :]
        mean, lo, hi = _summarize_prediction_interval(per_obs)
        weekly_df = pd.DataFrame(
            {
                "semana_inicio": weeks.to_numpy(),
                "channel": channel_short,
                "contribution_mean": mean,
                "contribution_p05": lo,
                "contribution_p95": hi,
                "spend_eur": prepared.test_df[channel].to_numpy(dtype=float),
            }
        )
        weekly_df = weekly_df.groupby(["semana_inicio", "channel"], as_index=False).sum()
        channel_rows.append(weekly_df)

        total_draws = per_obs.sum(axis=1)
        spend_total = float(prepared.test_df[channel].sum())
        roi_draws = total_draws / spend_total if spend_total > 1e-9 else np.zeros_like(total_draws)
        channel_cfg = prior_config["channels"][channel]
        sign_row = (
            stability_df.loc[stability_df["channel"] == channel_short].iloc[0].to_dict()
            if not stability_df.empty and (stability_df["channel"] == channel_short).any()
            else {}
        )
        aggregate_rows.append(
            {
                "channel": channel_short,
                "feature": feature,
                "contribution_mean": float(total_draws.mean()),
                "contribution_p05": float(np.quantile(total_draws, 0.05)),
                "contribution_p95": float(np.quantile(total_draws, 0.95)),
                "roi_mean": float(roi_draws.mean()),
                "roi_p05": float(np.quantile(roi_draws, 0.05)),
                "roi_p95": float(np.quantile(roi_draws, 0.95)),
                "historical_spend_2024_eur": spend_total,
                "requires_query_volume_proxy": bool(channel_cfg["requires_query_volume_proxy"]),
                "allow_optimization_without_proxy": bool(channel_cfg["allow_optimization_without_proxy"]),
                "deterministic_positive_runs": int(sign_row.get("positive_runs", 0)),
                "deterministic_negative_runs": int(sign_row.get("negative_runs", 0)),
                "deterministic_sign_stable_positive": bool(sign_row.get("sign_stable_positive", False)),
                "deterministic_sign_stable_negative": bool(sign_row.get("sign_stable_negative", False)),
            }
        )

    weekly_contributions = pd.concat(channel_rows, ignore_index=True).sort_values(["semana_inicio", "channel"]).reset_index(drop=True)
    aggregate_contributions = pd.DataFrame(aggregate_rows).sort_values("contribution_mean", ascending=False).reset_index(drop=True)
    return weekly_contributions, aggregate_contributions


def _coefficient_table(idata: az.InferenceData, prepared: PreparedData, prior_config: dict) -> pd.DataFrame:
    posterior = idata.posterior
    beta_media = posterior["beta_media"]
    rows = []
    for idx, feature in enumerate(prepared.media_feature_columns):
        channel = feature.split("__")[0]
        channel_short = channel.replace("media_", "", 1)
        draws = beta_media.isel(media=idx).values.reshape(-1)
        rows.append(
            {
                "channel": channel_short,
                "feature": feature,
                "posterior_mean": float(draws.mean()),
                "posterior_p05": float(np.quantile(draws, 0.05)),
                "posterior_p95": float(np.quantile(draws, 0.95)),
                "prior_mean": float(prior_config["channels"][channel]["prior_mean"]),
                "prior_sd": float(prior_config["channels"][channel]["prior_sd"]),
                "calibration_source": str(prior_config["channels"][channel]["calibration_source"]),
            }
        )
    return pd.DataFrame(rows).sort_values("posterior_mean", ascending=False).reset_index(drop=True)


def _quality_checks(
    idata: az.InferenceData,
    baseline_weekly: pd.DataFrame,
    test_metrics: dict[str, float],
) -> pd.DataFrame:
    rhat = az.rhat(idata, method="rank")
    max_rhat = float(np.nanmax(rhat.to_array().values))
    baseline_negative_prob = float((baseline_weekly["baseline_sales_p05"] < 0.0).mean())
    train_divergences = int(idata.sample_stats["diverging"].sum().to_numpy())
    return pd.DataFrame(
        [
            {
                "max_rhat": max_rhat,
                "train_divergences": train_divergences,
                "baseline_negative_week_probability": baseline_negative_prob,
                "test_2024_mape": float(test_metrics["mape"]),
                "test_2024_rmse": float(test_metrics["rmse"]),
                "test_2024_r2": float(test_metrics["r2"]),
            }
        ]
    )


def _eligibility_table(
    aggregate_contrib: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    quality_table: pd.DataFrame,
) -> pd.DataFrame:
    merged = aggregate_contrib.merge(coefficient_table[["channel", "posterior_mean", "posterior_p05", "posterior_p95"]], on="channel", how="left")
    merged["roi_width_ratio"] = np.where(
        np.abs(merged["roi_mean"]) > 1e-9,
        (merged["roi_p95"] - merged["roi_p05"]) / np.abs(merged["roi_mean"]),
        np.inf,
    )
    merged["contribution_width_ratio"] = np.where(
        np.abs(merged["contribution_mean"]) > 1e-9,
        (merged["contribution_p95"] - merged["contribution_p05"]) / np.abs(merged["contribution_mean"]),
        np.inf,
    )

    def _status(row: pd.Series) -> str:
        if bool(row["deterministic_sign_stable_negative"]):
            return "hold"
        if bool(row["requires_query_volume_proxy"]) and not bool(row["allow_optimization_without_proxy"]):
            return "review"
        if bool(row["deterministic_sign_stable_positive"]) and row["roi_width_ratio"] < 2.0 and row["contribution_mean"] > 0.0:
            return "eligible"
        if row["deterministic_positive_runs"] >= 2 and row["contribution_mean"] > 0.0:
            return "review"
        return "hold"

    merged["optimization_status"] = merged.apply(_status, axis=1)
    merged["baseline_guardrail_pass"] = bool(quality_table.loc[0, "baseline_negative_week_probability"] < 0.2)
    return merged.sort_values(["optimization_status", "contribution_mean"], ascending=[True, False]).reset_index(drop=True)


def _write_report(
    results: dict,
    coefficient_table: pd.DataFrame,
    quality_table: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> None:
    eligible = eligibility.loc[eligibility["optimization_status"] == "eligible", "channel"].tolist()
    review = eligibility.loc[eligibility["optimization_status"] == "review", "channel"].tolist()
    hold = eligibility.loc[eligibility["optimization_status"] == "hold", "channel"].tolist()
    top_channels = coefficient_table.head(4)["channel"].tolist()
    max_rhat = float(quality_table.loc[0, "max_rhat"])
    divergences = int(quality_table.loc[0, "train_divergences"])
    sampling_ok = max_rhat <= 1.01 and divergences == 0
    operational_ready = sampling_ok and bool(eligible)
    status_label = "operativo_con_guardrails" if operational_ready else "experimental_en_revision"

    lines = [
        "# Hierarchical Bayesian MMM Report",
        "",
        "## Objetivo",
        "",
        "Implementar una ruta MMM mas defendible que la serie global actual: panel geo, priors explicitos, coeficientes media positivos por construccion y separacion clara entre baseline y efecto incremental.",
        "",
        "## Configuracion",
        "",
        f"- Estado actual: `{status_label}`.",
        f"- Ventana de entrenamiento principal: `{results['train_start_year']}`-`{TARGET_YEAR - 1}`.",
        f"- Test holdout: `{TARGET_YEAR}`.",
        f"- Geos activos en el modelo: `{results['geo_count']}`.",
        f"- Variables de medios: `{results['media_count']}`.",
        f"- Draws / tune / chains: `{results['draws']}` / `{results['tune']}` / `{results['chains']}`.",
        "",
        "## Resultados Test 2024",
        "",
        "```text",
        pd.DataFrame([results["test_metrics"]]).round(4).to_string(index=False),
        "```",
        "",
        "## Quality Checks",
        "",
        "```text",
        quality_table.round(4).to_string(index=False),
        "```",
        "",
        "## Coeficientes Media",
        "",
        "```text",
        coefficient_table.round(4).to_string(index=False),
        "```",
        "",
        "## Elegibilidad De Canales",
        "",
        "```text",
        eligibility.round(4).to_string(index=False),
        "```",
        "",
        "## Lectura",
        "",
        f"- El modelo ya no permite contribuciones negativas de medios por construccion; eso corrige una de las principales patologias del modelo anterior.",
        f"- Los canales con mayor efecto posterior medio son `{', '.join(top_channels)}`.",
        f"- Canales elegibles para optimizacion directa: `{', '.join(eligible) if eligible else 'ninguno'}`.",
        f"- Canales en revision: `{', '.join(review) if review else 'ninguno'}`.",
        f"- Canales en hold: `{', '.join(hold) if hold else 'ninguno'}`.",
        "- `paid_search` queda en revision si no existe un proxy razonable de query volume o una calibracion experimental, siguiendo la advertencia metodologica de Meridian.",
        (
            f"- El muestreo sigue en revision: `max_rhat={max_rhat:.3f}` y `divergences={divergences}`."
            if not sampling_ok
            else "- El muestreo no presenta alertas principales de convergencia."
        ),
        (
            "- No debe usarse todavia para fijar pesos finales por canal u optimizacion automatica."
            if not operational_ready
            else "- Puede usarse con guardrails en los canales elegibles."
        ),
        "- El modelo debe leerse con rangos posteriores y guardrails, no como una verdad exacta por canal.",
        "",
        "## Fuentes Metodologicas Consultadas",
        "",
        "- Google Meridian: geo-level modeling, amount of data needed, ROI priors and calibration, paid search modeling, baseline and quality checks.",
        "- Meta Robyn analyst guide to MMM.",
        "- Google research sobre calibration y Bayesian MMM con carryover/shape effects.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_hierarchical_bayesian_mmm(train_start_year: int = DEFAULT_TRAIN_START_YEAR) -> dict:
    df = _load_geo_dataset()
    working, city_cols = _add_city_dummies(df)
    media_cols = media_columns(working)
    prior_config = _ensure_prior_config(working, media_cols, train_start_year=train_start_year)

    train_for_transforms = working.loc[
        (working["year"] >= train_start_year) & (working["year"] < TARGET_YEAR)
    ].copy()
    media_params, transform_diag = select_media_transforms(
        train_df=train_for_transforms,
        media_cols=media_cols,
        control_cols=CONTROL_COLUMNS,
        city_cols=city_cols,
    )
    transform_strategy = "channel_specific"
    channel_signal = (
        transform_diag.groupby("channel", as_index=False)["avg_incremental_mae_gain"].max()
        if "avg_incremental_mae_gain" in transform_diag.columns
        else pd.DataFrame()
    )
    if not channel_signal.empty and bool((channel_signal["avg_incremental_mae_gain"] <= 0.0).all()):
        media_params, transform_diag = select_pooled_media_transform(
            train_df=train_for_transforms,
            media_cols=media_cols,
            control_cols=CONTROL_COLUMNS,
            city_cols=city_cols,
        )
        transform_strategy = "pooled_common_transform"

    transformed_df, media_feature_cols = prepare_transformed_dataset(working, media_cols, media_params)
    prepared = _prepare_model_data(
        transformed_df=transformed_df,
        city_cols=city_cols,
        media_feature_cols=media_feature_cols,
        train_start_year=train_start_year,
    )

    model = _build_model(prepared, prior_config)
    with model:
        idata = pm.sample(
            draws=int(prior_config["sampling"]["draws"]),
            tune=int(prior_config["sampling"]["tune"]),
            chains=int(prior_config["sampling"]["chains"]),
            target_accept=float(prior_config["sampling"]["target_accept"]),
            random_seed=RANDOM_SEED,
            progressbar=False,
            return_inferencedata=True,
        )

    arrays = _posterior_arrays(idata)
    pred_draws_scaled, baseline_draws_scaled, contribution_draws_scaled = _predict_draws(prepared, arrays, on_test=True)
    scored_city, weekly_pred = _weekly_prediction_table(prepared, pred_draws_scaled, baseline_draws_scaled)
    test_metrics = _metrics(weekly_pred["actual_sales"].to_numpy(dtype=float), weekly_pred["predicted_sales_mean"].to_numpy(dtype=float))
    weekly_contrib, aggregate_contrib = _channel_contribution_tables(prepared, contribution_draws_scaled, prior_config)
    coefficient_table = _coefficient_table(idata, prepared, prior_config)
    quality_table = _quality_checks(idata, weekly_pred, test_metrics)

    if not (CONFIG.reports_tables_dir / "identifiability_channel_stability.csv").exists():
        run_identifiability_audit()
        weekly_contrib, aggregate_contrib = _channel_contribution_tables(prepared, contribution_draws_scaled, prior_config)

    eligibility = _eligibility_table(aggregate_contrib, coefficient_table, quality_table)

    results = {
        "model_name": "HierarchicalBayesianMMM",
        "train_start_year": train_start_year,
        "target_year": TARGET_YEAR,
        "geo_count": int(len(prepared.geo_labels)),
        "media_count": int(len(prepared.media_feature_columns)),
        "draws": int(prior_config["sampling"]["draws"]),
        "tune": int(prior_config["sampling"]["tune"]),
        "chains": int(prior_config["sampling"]["chains"]),
        "test_metrics": test_metrics,
        "baseline_negative_week_probability": float(quality_table.loc[0, "baseline_negative_week_probability"]),
        "eligible_channels": eligibility.loc[eligibility["optimization_status"] == "eligible", "channel"].tolist(),
        "review_channels": eligibility.loc[eligibility["optimization_status"] == "review", "channel"].tolist(),
        "hold_channels": eligibility.loc[eligibility["optimization_status"] == "hold", "channel"].tolist(),
    }

    transform_rows = []
    for media_col, params in media_params.items():
        transform_rows.append(
            {
                "channel": media_col.replace("media_", "", 1),
                "selection_strategy": transform_strategy,
                **params,
            }
        )
    transform_table = pd.DataFrame(transform_rows).sort_values("channel").reset_index(drop=True)

    az.to_netcdf(idata, CONFIG.hierarchical_inference_file)
    CONFIG.hierarchical_model_results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    scored_city.to_csv(CONFIG.hierarchical_weekly_predictions_file, index=False)
    weekly_contrib.to_csv(CONFIG.hierarchical_contributions_file, index=False)
    weekly_pred.to_csv(CONFIG.hierarchical_baseline_file, index=False)
    pd.DataFrame([results["test_metrics"]]).to_csv(RESULTS_TABLE, index=False)
    coefficient_table.to_csv(COEFFICIENT_TABLE, index=False)
    quality_table.to_csv(QUALITY_TABLE, index=False)
    eligibility.to_csv(ELIGIBILITY_TABLE, index=False)
    transform_table.to_csv(TRANSFORM_TABLE, index=False)
    _write_report(results, coefficient_table, quality_table, eligibility)
    return {
        "results": results,
        "weekly_predictions": weekly_pred,
        "channel_eligibility": eligibility,
    }
