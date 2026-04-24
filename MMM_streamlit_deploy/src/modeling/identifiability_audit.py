from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.data.weekly_aggregations import (
    build_weekly_calendar,
    build_weekly_media,
    build_weekly_orders,
    build_weekly_sales,
    build_weekly_traffic,
)
from src.modeling.selection import select_media_transforms
from src.modeling.specs import CONTROL_COLUMNS, ELASTIC_NET_ALPHAS, ELASTIC_NET_L1, RANDOM_STATE
from src.modeling.transforms import apply_media_transform
from src.validation.backtesting import panel_time_cv_indices


YEARLY_HEALTH_TABLE = CONFIG.reports_tables_dir / "identifiability_yearly_health.csv"
CURRENT_MODEL_SUMMARY_TABLE = CONFIG.reports_tables_dir / "identifiability_current_model_summary.csv"
MODEL_COMPARISON_TABLE = CONFIG.reports_tables_dir / "identifiability_model_comparison.csv"
CHANNEL_STABILITY_TABLE = CONFIG.reports_tables_dir / "identifiability_channel_stability.csv"
TRANSFORM_SELECTION_TABLE = CONFIG.reports_tables_dir / "identifiability_transform_selection.csv"
REPORT_MD = CONFIG.docs_dir / "model_identifiability_audit.md"

TARGET_YEAR = 2024
KEY_COLUMNS = ["semana_inicio", "ciudad"]


@dataclass
class CandidateFit:
    model_name: str
    train_start_year: int
    metrics: dict[str, float]
    media_coefficients: pd.Series
    transform_specs: dict[str, dict[str, float | int | str]]


def _metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    actual = pd.Series(y_true, dtype=float).to_numpy()
    pred = pd.Series(y_pred, dtype=float).to_numpy()
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), 1.0)
    return {
        "mape": float(np.mean(np.abs(actual - pred) / denom) * 100.0),
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - pred)))),
        "bias": float(np.mean(pred - actual)),
        "r2": float(1.0 - np.square(actual - pred).sum() / np.square(actual - actual.mean()).sum()),
    }


def _fill_numeric_nulls(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _add_temporal_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
    weeks = out["semana_inicio"].drop_duplicates().sort_values().tolist()
    week_index = {week: idx for idx, week in enumerate(weeks)}
    out["trend_index"] = out["semana_inicio"].map(week_index).astype(int)
    out["week_of_year"] = out["semana_inicio"].dt.isocalendar().week.astype(int)
    out["year"] = out["semana_inicio"].dt.year.astype(int)
    out["week_sin"] = np.sin(2.0 * np.pi * out["week_of_year"] / 52.0)
    out["week_cos"] = np.cos(2.0 * np.pi * out["week_of_year"] / 52.0)
    return out


def _add_media_mix_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    media_cols = sorted([column for column in out.columns if column.startswith("media_")])
    out["budget_total_eur"] = out[media_cols].sum(axis=1) if media_cols else 0.0
    for media_col in media_cols:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        out[share_col] = np.where(
            out["budget_total_eur"] > 0.0,
            out[media_col] / out["budget_total_eur"] * 100.0,
            0.0,
        )
    return out


def _active_sales_cities(weekly_sales: pd.DataFrame) -> list[str]:
    totals = weekly_sales.groupby("ciudad", as_index=False)["ventas_netas"].sum()
    return sorted(totals.loc[totals["ventas_netas"] > 0.0, "ciudad"].astype(str).tolist())


def _build_city_panel_dataset() -> pd.DataFrame:
    weekly_sales, _checks = build_weekly_sales(keep_city=True)
    weekly_orders = build_weekly_orders(keep_city=True)
    weekly_calendar = build_weekly_calendar(keep_city=True)
    weekly_traffic = build_weekly_traffic(keep_city=True)
    weekly_media = build_weekly_media(keep_city=True)

    panel = weekly_calendar[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)
    dataset = (
        panel.merge(weekly_sales, on=KEY_COLUMNS, how="left")
        .merge(weekly_orders, on=KEY_COLUMNS, how="left")
        .merge(weekly_calendar, on=KEY_COLUMNS, how="left")
        .merge(weekly_traffic, on=KEY_COLUMNS, how="left")
        .merge(weekly_media, on=KEY_COLUMNS, how="left")
    )
    dataset = _fill_numeric_nulls(dataset)
    dataset = _add_temporal_features(dataset)
    dataset = _add_media_mix_features(dataset)
    active_cities = _active_sales_cities(weekly_sales)
    return (
        dataset.loc[
            (dataset["week_complete_flag"] == 1) & dataset["ciudad"].astype(str).isin(active_cities)
        ]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )


def _candidate_feature_blocks(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    city_dummies = pd.get_dummies(dataset["ciudad"], prefix="city", drop_first=True, dtype=float)
    working = pd.concat([dataset.reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)
    city_cols = sorted([column for column in working.columns if column.startswith("city_")])
    media_cols = sorted(
        [column for column in working.columns if column.startswith("media_") and "__lag" not in column]
    )
    return working, city_cols, media_cols


def _select_transforms(
    train_df: pd.DataFrame,
    media_cols: list[str],
    feature_cols: list[str],
    group_col: str = "ciudad",
) -> tuple[dict[str, dict[str, float | int | str]], pd.DataFrame]:
    return select_media_transforms(
        train_df=train_df,
        media_cols=media_cols,
        control_cols=CONTROL_COLUMNS,
        city_cols=[column for column in feature_cols if column.startswith("city_")],
    )


def _apply_transforms(
    dataset: pd.DataFrame,
    media_specs: dict[str, dict[str, float | int | str]],
    group_col: str = "ciudad",
) -> tuple[pd.DataFrame, list[str]]:
    out = dataset.sort_values([group_col, "semana_inicio"]).copy()
    feature_cols: list[str] = []
    for media_col, spec in media_specs.items():
        feature_name = (
            f"{media_col}__lag{spec['lag']}_a"
            f"{str(spec['alpha']).replace('.', '_')}_{spec['saturation']}"
        )
        out[feature_name] = (
            out.groupby(group_col, group_keys=False)[media_col]
            .apply(
                lambda series: apply_media_transform(
                    series,
                    lag=int(spec["lag"]),
                    alpha=float(spec["alpha"]),
                    saturation=str(spec["saturation"]),
                )
            )
            .astype(float)
        )
        feature_cols.append(feature_name)
    return out.reset_index(drop=True), sorted(feature_cols)


def _fit_candidate_model(
    dataset: pd.DataFrame,
    model_name: str,
    train_start_year: int,
) -> tuple[CandidateFit, pd.DataFrame]:
    working, city_cols, media_cols = _candidate_feature_blocks(dataset)
    train_df = working.loc[(working["year"] >= train_start_year) & (working["year"] < TARGET_YEAR)].copy()
    test_df = working.loc[working["year"] == TARGET_YEAR].copy()
    base_features = [*CONTROL_COLUMNS, *city_cols]
    transform_specs, transform_diag = _select_transforms(train_df, media_cols, base_features)
    transformed, media_feature_cols = _apply_transforms(working, transform_specs)

    train_work = transformed.loc[(transformed["year"] >= train_start_year) & (transformed["year"] < TARGET_YEAR)].copy()
    test_work = transformed.loc[transformed["year"] == TARGET_YEAR].copy()
    feature_cols = [*base_features, *media_feature_cols]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_work[feature_cols])
    x_test = scaler.transform(test_work[feature_cols])
    cv = panel_time_cv_indices(train_work[["semana_inicio"]].copy(), n_splits=3)
    model = ElasticNetCV(
        alphas=ELASTIC_NET_ALPHAS,
        l1_ratio=ELASTIC_NET_L1,
        cv=cv,
        random_state=RANDOM_STATE,
        max_iter=50000,
        selection="cyclic",
    )
    model.fit(x_train, train_work["ventas_netas"].to_numpy(dtype=float))
    test_pred = model.predict(x_test)

    scored = test_work[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    scored["pred"] = test_pred
    weekly_scored = scored.groupby("semana_inicio", as_index=False)[["ventas_netas", "pred"]].sum()
    metrics = _metrics(weekly_scored["ventas_netas"], weekly_scored["pred"])

    raw_coef = pd.Series(model.coef_ / scaler.scale_, index=feature_cols, dtype=float)
    media_coef = raw_coef[media_feature_cols].copy()
    metrics["negative_media_coefficients"] = float((media_coef < -1e-8).sum())
    metrics["train_start_year"] = float(train_start_year)
    metrics["distinct_selected_transforms"] = float(
        len({(spec["lag"], spec["alpha"], spec["saturation"]) for spec in transform_specs.values()})
    )

    fit = CandidateFit(
        model_name=model_name,
        train_start_year=train_start_year,
        metrics=metrics,
        media_coefficients=media_coef,
        transform_specs=transform_specs,
    )
    transform_diag["model_name"] = model_name
    transform_diag["train_start_year"] = train_start_year
    return fit, transform_diag


def _yearly_health() -> pd.DataFrame:
    causal = pd.read_parquet(CONFIG.model_dataset_file).copy()
    causal["semana_inicio"] = pd.to_datetime(causal["semana_inicio"])
    diagnostic = pd.read_parquet(CONFIG.diagnostic_dataset_file).copy()
    diagnostic["semana_inicio"] = pd.to_datetime(diagnostic["semana_inicio"])

    causal_weekly = causal.groupby("semana_inicio", as_index=False).agg(
        ventas_netas=("ventas_netas", "sum"),
        budget_total_eur=("budget_total_eur", "sum"),
        year=("year", "max"),
    )
    diagnostic_weekly = diagnostic.groupby("semana_inicio", as_index=False).agg(
        ventas_netas=("ventas_netas", "sum"),
        budget_total_eur=("budget_total_eur", "sum"),
        year=("year", "max"),
        week_complete_flag=("week_complete_flag", "min"),
    )
    summary = (
        causal_weekly.groupby("year", as_index=False)
        .agg(
            weeks=("semana_inicio", "nunique"),
            sales_total=("ventas_netas", "sum"),
            sales_mean=("ventas_netas", "mean"),
            sales_std=("ventas_netas", "std"),
            budget_total=("budget_total_eur", "sum"),
            budget_mean=("budget_total_eur", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    incomplete = (
        diagnostic_weekly.groupby("year", as_index=False)
        .agg(
            incomplete_weeks=("week_complete_flag", lambda s: int((s == 0).sum())),
            diagnostic_weeks=("semana_inicio", "nunique"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    out = summary.merge(incomplete, on="year", how="left")
    out["sales_cv"] = out["sales_std"] / out["sales_mean"]
    out["budget_share_of_sales"] = out["budget_total"] / out["sales_total"]
    return out


def _current_model_summary() -> pd.DataFrame:
    model_results = json.loads(CONFIG.model_results_file.read_text(encoding="utf-8"))
    baseline = pd.read_csv(CONFIG.reports_tables_dir / "arimax_media_baseline_decomposition.csv", parse_dates=["semana_inicio"])
    coefficients = pd.read_csv(CONFIG.reports_tables_dir / "arimax_coefficients.csv")
    business = coefficients.loc[coefficients["is_business_variable"]].copy()
    business["ci_crosses_zero"] = (business["ci_low"] <= 0.0) & (business["ci_high"] >= 0.0)

    selected_transforms = json.loads(CONFIG.selected_transforms_file.read_text(encoding="utf-8"))
    unique_transforms = sorted(
        {(int(spec["lag"]), float(spec["alpha"]), str(spec["saturation"])) for spec in selected_transforms.values()}
    )

    summary = {
        "deployment_spec": model_results.get("deployment_spec", ""),
        "test_2024_mape": float(model_results.get("test_metrics", {}).get("mape", np.nan)),
        "test_2024_rmse": float(model_results.get("test_metrics", {}).get("rmse", np.nan)),
        "test_2024_r2": float(model_results.get("test_metrics", {}).get("r2", np.nan)),
        "baseline_sales_share_pct": float(
            baseline["base_sales"].sum() / baseline["predicted_sales"].sum() * 100.0
        ),
        "media_sales_share_pct": float(
            baseline["total_media_contribution"].sum() / baseline["predicted_sales"].sum() * 100.0
        ),
        "weeks_with_positive_media_contribution": int((baseline["total_media_contribution"] > 1e-9).sum()),
        "max_weekly_media_contribution_eur": float(baseline["total_media_contribution"].max()),
        "business_variables_count": int(len(business)),
        "business_variables_ci_cross_zero_count": int(business["ci_crosses_zero"].sum()),
        "selected_transform_variants": int(len(unique_transforms)),
        "selected_transform_signature": " | ".join(
            f"lag={lag}, alpha={alpha:.2f}, sat={saturation}" for lag, alpha, saturation in unique_transforms
        ),
    }
    return pd.DataFrame([summary])


def _channel_stability_table(fits: list[CandidateFit]) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    coefficient_rows = []
    for fit in fits:
        for feature, coefficient in fit.media_coefficients.items():
            channel = feature.split("__")[0].replace("media_", "", 1)
            coefficient_rows.append(
                {
                    "model_name": fit.model_name,
                    "train_start_year": fit.train_start_year,
                    "channel": channel,
                    "coefficient": float(coefficient),
                }
            )

    coefficient_df = pd.DataFrame(coefficient_rows)
    pivot = coefficient_df.pivot_table(
        index="channel",
        columns=["model_name", "train_start_year"],
        values="coefficient",
        aggfunc="first",
    )
    pivot.columns = [f"{model_name}_{train_year}" for model_name, train_year in pivot.columns]
    pivot = pivot.reset_index()

    for row in pivot.itertuples(index=False):
        channel = str(row.channel)
        values = [float(value) for key, value in row._asdict().items() if key != "channel" and pd.notna(value)]
        positive_runs = sum(value > 0.0 for value in values)
        negative_runs = sum(value < 0.0 for value in values)
        rows.append(
            {
                "channel": channel,
                "positive_runs": positive_runs,
                "negative_runs": negative_runs,
                "sign_stable_positive": bool(positive_runs == len(values) and len(values) > 0),
                "sign_stable_negative": bool(negative_runs == len(values) and len(values) > 0),
            }
        )

    stability = pd.DataFrame(rows).merge(pivot, on="channel", how="left")
    return stability.sort_values(["sign_stable_positive", "positive_runs", "channel"], ascending=[False, False, True])


def _write_report(
    yearly_health: pd.DataFrame,
    current_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    channel_stability: pd.DataFrame,
) -> None:
    current = current_summary.iloc[0]
    best_panel = model_comparison.sort_values(["mape", "negative_media_coefficients"]).iloc[0]
    stable_positive = channel_stability.loc[channel_stability["sign_stable_positive"], "channel"].tolist()
    unstable = channel_stability.loc[
        ~(channel_stability["sign_stable_positive"] | channel_stability["sign_stable_negative"]),
        "channel",
    ].tolist()

    lines = [
        "# Model Identifiability Audit",
        "",
        "## Objetivo",
        "",
        "Revisar si la base, los coeficientes de medios y el delay/adstock se pueden defender con el dataset actual, y cuantificar cuanto empeora la identificabilidad al colapsar todo a una sola serie global.",
        "",
        "## Hallazgos Principales",
        "",
        f"- La descomposicion actual del ARIMAX deja `{current['baseline_sales_share_pct']:.4f}%` de las ventas predichas de 2024 como base y solo `{current['media_sales_share_pct']:.4f}%` como aporte de medios.",
        f"- Solo `{int(current['weeks_with_positive_media_contribution'])}` de 52 semanas tienen contribucion media positiva en la descomposicion actual; eso invalida su uso para repartir pesos por canal.",
        f"- En el ARIMAX actual, `{int(current['business_variables_ci_cross_zero_count'])}` de `{int(current['business_variables_count'])}` variables de negocio cruzan cero en su intervalo de confianza.",
        f"- El selector de delay/adstock no esta diferenciando canales: solo encuentra `{int(current['selected_transform_variants'])}` configuracion(es) ganadora(s) para todos los medios (`{current['selected_transform_signature']}`).",
        "",
        "## Salud Temporal Del Dataset",
        "",
        "```text",
        yearly_health.round(4).to_string(index=False),
        "```",
        "",
        "## Comparativa De Modelos Explicativos",
        "",
        "```text",
        model_comparison.round(4).to_string(index=False),
        "```",
        "",
        "## Estabilidad De Canales",
        "",
        "```text",
        channel_stability.round(4).to_string(index=False),
        "```",
        "",
        "## Lectura Ejecutiva",
        "",
        f"- `2024` no es un ano con mucho ruido relativo en la serie causal global: la desviacion semanal es baja y por eso un sesgo moderado hunde el `R2`, aunque el `MAPE` no parezca desastroso.",
        "- `2020` si es claramente otro regimen y conviene tratarlo como ano de shock o, como minimo, revisar pesos tambien sin ese bloque.",
        "- El problema metodologico mas serio no es 2024 sino haber colapsado una fuente originalmente geolocalizada a una unica serie `Global`; eso reduce mucho la variacion util para identificar canales.",
        f"- El mejor candidato entre los modelos revisados en este audit es `{best_panel['model_name']}` entrenado desde `{int(best_panel['train_start_year'])}`, con `MAPE {best_panel['mape']:.2f}` y `{int(best_panel['negative_media_coefficients'])}` coeficientes media negativos.",
        f"- Canales con signo positivo estable entre corridas auditadas: `{', '.join(stable_positive) if stable_positive else 'ninguno'}`.",
        f"- Canales inestables en signo: `{', '.join(unstable) if unstable else 'ninguno'}`.",
        "",
        "## Conclusiones Operativas",
        "",
        "- No se deben usar los coeficientes actuales del ARIMAX como pesos de inversion por canal.",
        "- La base actual esta sobreexplicada por tendencia y calendario: sirve para forecast, no para cuantificar incremental de medios con rigor.",
        "- El delay/adstock actual no esta bien identificado a nivel canal porque el selector converge al mismo extremo para todos.",
        "- Si hay que defender pesos ahora, deben presentarse como rangos y estabilidad de signo, no como un unico valor puntual exacto.",
        "",
        "## Artefactos",
        "",
        "- `reports/tables/identifiability_yearly_health.csv`.",
        "- `reports/tables/identifiability_current_model_summary.csv`.",
        "- `reports/tables/identifiability_model_comparison.csv`.",
        "- `reports/tables/identifiability_channel_stability.csv`.",
        "- `reports/tables/identifiability_transform_selection.csv`.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_identifiability_audit() -> dict[str, pd.DataFrame]:
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    yearly_health = _yearly_health()
    current_summary = _current_model_summary()

    global_dataset = pd.read_parquet(CONFIG.model_dataset_file).copy()
    global_dataset["semana_inicio"] = pd.to_datetime(global_dataset["semana_inicio"])
    panel_dataset = _build_city_panel_dataset()

    fits: list[CandidateFit] = []
    transform_logs: list[pd.DataFrame] = []
    for model_name, dataset, train_start in [
        ("global_transformed_media", global_dataset, 2020),
        ("panel_transformed_media", panel_dataset, 2020),
        ("panel_transformed_media", panel_dataset, 2021),
    ]:
        fit, transform_log = _fit_candidate_model(dataset, model_name, train_start_year=train_start)
        fits.append(fit)
        transform_logs.append(transform_log)

    model_comparison = pd.DataFrame(
        [
            {
                "model_name": fit.model_name,
                "train_start_year": fit.train_start_year,
                **fit.metrics,
            }
            for fit in fits
        ]
    ).sort_values(["mape", "negative_media_coefficients"]).reset_index(drop=True)
    channel_stability = _channel_stability_table(fits)
    transform_selection = pd.concat(transform_logs, ignore_index=True)

    yearly_health.to_csv(YEARLY_HEALTH_TABLE, index=False)
    current_summary.to_csv(CURRENT_MODEL_SUMMARY_TABLE, index=False)
    model_comparison.to_csv(MODEL_COMPARISON_TABLE, index=False)
    channel_stability.to_csv(CHANNEL_STABILITY_TABLE, index=False)
    transform_selection.to_csv(TRANSFORM_SELECTION_TABLE, index=False)
    _write_report(yearly_health, current_summary, model_comparison, channel_stability)

    return {
        "yearly_health": yearly_health,
        "current_summary": current_summary,
        "model_comparison": model_comparison,
        "channel_stability": channel_stability,
        "transform_selection": transform_selection,
    }
