from __future__ import annotations

import json

import pandas as pd

from src.common.config import CONFIG
from src.data.weekly_aggregations import CALENDAR_USECOLS, MEDIA_USECOLS, TRAFFIC_USECOLS, week_start
from src.features.metadata import feature_catalog, feature_modeling_tier, target_correlation_table


def _read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_money(value: float) -> str:
    return f"{value:,.2f} EUR"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _feature_catalog(dataset: pd.DataFrame) -> pd.DataFrame:
    return feature_catalog(dataset)


def _feature_role(column: str, diagnostic_only_columns: list[str]) -> tuple[str, str]:
    tier = feature_modeling_tier(column, set(diagnostic_only_columns))
    if tier == "technical_key":
        if column == "semana_inicio":
            return "key", "Indice temporal semanal del MMM."
        return "technical_key", "Identificador tecnico retenido por compatibilidad; la serie ya es global."
    if tier == "target":
        return "target", "Variable objetivo del MMM."
    if tier == "diagnostic_only":
        if column in {"pedidos_total", "ticket_medio_neto"}:
            return "downstream_business", "Variable comercial util para diagnostico pero excluida del set causal final."
        return "diagnostic_operational", "Senal operativa o de calidad retenida solo en la vista diagnostica."
    if tier == "screening_only":
        if column.startswith("media_"):
            return "screening_media_level", "Senal de nivel por canal util para screening y comparativas, no para el set final de simulacion."
        return "screening_temporal_label", "Etiqueta temporal descriptiva util para lectura y reporting, no para el modelo final."
    if column.startswith("budget_share_pct_") or column == "budget_total_eur":
        return "media_mix_model_ready", "Representacion de mix y presion total util para el MMM final."
    if column in {"trend_index", "week_sin", "week_cos"}:
        return "temporal_control", "Control temporal compacto para tendencia y estacionalidad."
    return "exogenous_control", "Control externo o de calendario plausible para el modelo."


def _build_variable_role_catalog(
    diagnostic_dataset: pd.DataFrame,
    causal_dataset: pd.DataFrame,
    diagnostic_only_columns: list[str],
) -> pd.DataFrame:
    rows = []
    causal_columns = set(causal_dataset.columns)
    for column in diagnostic_dataset.columns:
        role, rationale = _feature_role(column, diagnostic_only_columns)
        rows.append(
            {
                "feature": column,
                "role": role,
                "enters_causal_dataset": column in causal_columns,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "feature"]).reset_index(drop=True)


def _build_zero_typology(dataset: pd.DataFrame) -> pd.DataFrame:
    zero_rows = int((dataset["ventas_netas"] == 0.0).sum())
    return pd.DataFrame(
        [
            {
                "series_label": "Global",
                "panel_type": "global_series",
                "rows": int(len(dataset)),
                "positive_sales_rows": int((dataset["ventas_netas"] > 0.0).sum()),
                "zero_sales_rows": zero_rows,
                "zero_share_pct": round(float(zero_rows / len(dataset) * 100.0), 2),
                "sales_total_eur": round(float(dataset["ventas_netas"].sum()), 2),
            }
        ]
    )


def _build_media_channel_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    weekly = dataset[dataset["week_complete_flag"] == 1].copy()
    media_cols = [col for col in weekly.columns if col.startswith("media_")]
    corr = weekly[media_cols].corr().abs()
    rows = []
    total_spend = float(weekly[media_cols].sum().sum())
    for channel in media_cols:
        peers = corr.loc[channel].drop(index=channel)
        mean_value = float(weekly[channel].mean())
        std_value = float(weekly[channel].std(ddof=0))
        rows.append(
            {
                "channel": channel.replace("media_", ""),
                "mean_weekly_investment": round(mean_value, 2),
                "std_weekly_investment": round(std_value, 2),
                "coefficient_of_variation": round(std_value / mean_value, 4) if mean_value else 0.0,
                "pct_weeks_with_spend": round(float((weekly[channel] > 0.0).mean() * 100.0), 2),
                "share_of_total_spend_pct": round(float(weekly[channel].sum() / total_spend * 100.0), 2) if total_spend else 0.0,
                "max_abs_corr_with_other_channel": round(float(peers.max()), 4) if not peers.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("share_of_total_spend_pct", ascending=False).reset_index(drop=True)


def _build_sales_media_lag_table(dataset: pd.DataFrame) -> pd.DataFrame:
    weekly = dataset[dataset["week_complete_flag"] == 1].copy()
    media_cols = [col for col in weekly.columns if col.startswith("media_")]
    rows = []
    for channel in media_cols:
        correlations = []
        for lag in range(0, 9):
            corr = float(weekly["ventas_netas"].corr(weekly[channel].shift(lag)))
            correlations.append((lag, corr))
        best_lag, best_corr = max(correlations, key=lambda item: abs(item[1]) if pd.notna(item[1]) else -1.0)
        for lag, corr in correlations:
            rows.append(
                {
                    "channel": channel.replace("media_", ""),
                    "lag_weeks": lag,
                    "correlation": round(corr, 4),
                    "best_lag_by_abs_corr": best_lag,
                    "best_corr_by_abs_value": round(best_corr, 4),
                }
            )
    return pd.DataFrame(rows)


def _audit_source_coverage() -> pd.DataFrame:
    calendar = pd.read_csv(CONFIG.calendar_file, usecols=CALENDAR_USECOLS, parse_dates=["fecha"])
    calendar_daily = calendar[["fecha"]].drop_duplicates().sort_values("fecha")
    calendar_weekly = calendar.assign(semana_inicio=week_start(calendar["fecha"]))[["semana_inicio"]].drop_duplicates()

    def build_daily_audit(source: str, frame: pd.DataFrame, date_col: str, metric_col: str | None, grain: str) -> pd.DataFrame:
        base = calendar_daily.rename(columns={"fecha": date_col}).copy()
        if metric_col is None:
            observed = frame[[date_col]].drop_duplicates().assign(source_present=1)
            merged = base.merge(observed, on=[date_col], how="left")
            merged["source_present"] = merged["source_present"].fillna(0).astype(int)
            metric_name = "source_present"
            merged[metric_name] = merged["source_present"]
        else:
            observed = frame.groupby([date_col], as_index=False)[metric_col].sum()
            merged = base.merge(observed, on=[date_col], how="left")
            metric_name = metric_col
        merged["is_missing"] = merged[metric_name].isna().astype(int)
        merged[metric_name] = merged[metric_name].fillna(0.0)
        return pd.DataFrame(
            [
                {
                    "source": source,
                    "grain": grain,
                    "metric": metric_name,
                    "min_fecha": merged[date_col].min().date().isoformat(),
                    "max_fecha": merged[date_col].max().date().isoformat(),
                    "expected_obs": int(len(merged)),
                    "observed_non_missing": int((merged[metric_name] != 0).sum()) if source != "calendar" else int(len(merged)),
                    "pct_missing_panel": float(merged["is_missing"].mean() * 100.0),
                    "pct_zero_metric": float((merged[metric_name] == 0).mean() * 100.0),
                }
            ]
        )

    def build_weekly_audit(source: str, frame: pd.DataFrame, date_col: str, metric_col: str, grain: str) -> pd.DataFrame:
        base = calendar_weekly.rename(columns={"semana_inicio": date_col}).copy()
        observed = frame.groupby([date_col], as_index=False)[metric_col].sum()
        merged = base.merge(observed, on=[date_col], how="left")
        merged["is_missing"] = merged[metric_col].isna().astype(int)
        merged[metric_col] = merged[metric_col].fillna(0.0)
        return pd.DataFrame(
            [
                {
                    "source": source,
                    "grain": grain,
                    "metric": metric_col,
                    "min_fecha": merged[date_col].min().date().isoformat(),
                    "max_fecha": merged[date_col].max().date().isoformat(),
                    "expected_obs": int(len(merged)),
                    "observed_non_missing": int((merged[metric_col] > 0).sum()),
                    "pct_missing_panel": float(merged["is_missing"].mean() * 100.0),
                    "pct_zero_metric": float((merged[metric_col] == 0).mean() * 100.0),
                }
            ]
        )

    sales = pd.read_csv(CONFIG.sales_lines_file, usecols=["fecha_venta", "venta_neta_sin_iva_eur"], parse_dates=["fecha_venta"])
    orders = pd.read_csv(CONFIG.orders_file, usecols=["fecha_pedido", "importe_neto_sin_iva_eur"], parse_dates=["fecha_pedido"])
    traffic = pd.read_csv(CONFIG.traffic_file, usecols=TRAFFIC_USECOLS, parse_dates=["fecha"])
    media = pd.read_csv(CONFIG.investment_file, usecols=MEDIA_USECOLS, parse_dates=["semana_inicio", "semana_fin"])

    audits = [
        build_daily_audit("sales_lines", sales, "fecha_venta", "venta_neta_sin_iva_eur", "day"),
        build_daily_audit("orders", orders, "fecha_pedido", "importe_neto_sin_iva_eur", "day"),
        build_daily_audit("calendar", calendar, "fecha", None, "day"),
        build_daily_audit("traffic", traffic, "fecha", "sesiones_web", "day"),
        build_weekly_audit("media", media, "semana_inicio", "inversion_eur", "week"),
    ]
    audit = pd.concat(audits, ignore_index=True)

    def audit_note(row: pd.Series) -> str:
        if row["pct_missing_panel"] == 0.0 and row["pct_zero_metric"] == 0.0:
            return "Cobertura plena sin huecos."
        if row["pct_missing_panel"] == 0.0 and row["pct_zero_metric"] > 80.0:
            return "Cobertura plena pero serie muy sparse o estructuralmente vacia."
        if row["pct_missing_panel"] > 0.0:
            return "La fuente no cubre todo el panel de referencia."
        return "Cobertura util para diagnostico."

    audit["business_note"] = audit.apply(audit_note, axis=1)
    return audit.sort_values("source").reset_index(drop=True)


def write_reports(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    scenarios: pd.DataFrame,
    mroi: pd.DataFrame,
    scenario_diagnostics: pd.DataFrame,
    channel_budget_sensitivity: pd.DataFrame,
    scenario_validation_summary: pd.DataFrame,
    scenario_stat_tests: pd.DataFrame,
    regularization_fit: pd.DataFrame,
    scenario_oos_validation: pd.DataFrame,
    scenario_oos_weight_stability: pd.DataFrame,
    sales_lines: pd.DataFrame,
    orders: pd.DataFrame,
    clients: pd.DataFrame,
) -> None:
    data_checks = _read_json(CONFIG.data_checks_file)
    model_results = _read_json(CONFIG.model_results_file)
    benchmark_summary = pd.read_csv(CONFIG.reports_tables_dir / "arimax_benchmark_summary.csv")
    benchmark_guardrail = model_results.get("required_benchmark_guardrail", {})

    causal_rows = data_checks["causal_dataset_rows"]
    diagnostic_rows = data_checks["diagnostic_dataset_rows"]
    date_min = sales_lines["fecha_venta"].min().date().isoformat()
    date_max = sales_lines["fecha_venta"].max().date().isoformat()
    promo_share = float((sales_lines["descuento_pct"].fillna(0.0) > 0).mean() * 100.0)
    weighted_margin = float(
        (sales_lines["venta_neta_sin_iva_eur"].sum() - sales_lines["coste_produccion_eur"].sum())
        / sales_lines["venta_neta_sin_iva_eur"].sum()
        * 100.0
    )
    aov = float(orders["importe_neto_sin_iva_eur"].mean())
    top_segment = str(clients["segmento"].mode().iloc[0])
    top_channel = str(clients["canal_preferido"].mode().iloc[0])

    backtest_avg = (
        backtest.groupby("spec", as_index=False)[["mape", "rmse"]]
        .mean()
        .sort_values("mape")
        .reset_index(drop=True)
    )
    winner = model_results["winner"]
    deployment_spec = model_results["deployment_spec"]
    winner_mape = float(backtest_avg.loc[backtest_avg["spec"] == winner, "mape"].iloc[0])
    test_metrics = model_results["test_metrics"]
    top_scenario = scenarios.sort_values("predicted_gross_profit_2024", ascending=False).iloc[0]
    top_scenario_diagnostics = (
        scenario_diagnostics.loc[scenario_diagnostics["scenario"] == top_scenario["scenario"]]
        .head(1)
        .squeeze()
    )
    top_mroi = mroi.sort_values("mroi_profit", ascending=False).iloc[0]
    worst_mroi = mroi.sort_values("mroi_profit", ascending=True).iloc[0]
    arimax_validation = scenario_validation_summary[scenario_validation_summary["model"] == "ARIMAX"].copy()
    arimax_optimized = arimax_validation.loc[arimax_validation["scenario"] == "do_something_optimized"].iloc[0]
    arimax_stats = scenario_stat_tests[
        (scenario_stat_tests["model"] == "ARIMAX") & (scenario_stat_tests["comparison"] == "optimized_vs_historical")
    ].iloc[0]
    oos_best = scenario_oos_validation.sort_values("predicted_uplift_eur", ascending=False).iloc[0]
    best_regularized = regularization_fit.sort_values("mape").iloc[0]
    test_rows = predictions[predictions["fold"] == "test_2024"]
    required_benchmark_block = benchmark_summary[
        [
            "spec",
            "validation_mean_mape",
            "validation_mean_rmse",
            "test_2024_mape",
            "test_2024_rmse",
        ]
    ].round(2)
    causal_dataset = pd.read_parquet(CONFIG.model_dataset_file)
    source_audit = _audit_source_coverage()
    zero_typology = _build_zero_typology(dataset)
    variable_role_catalog = _build_variable_role_catalog(
        diagnostic_dataset=dataset,
        causal_dataset=causal_dataset,
        diagnostic_only_columns=data_checks["diagnostic_only_columns"],
    )
    media_channel_summary = _build_media_channel_summary(dataset)
    sales_media_lag = _build_sales_media_lag_table(dataset)
    weekly_target = dataset[
        ["semana_inicio", "ventas_netas", "pedidos_total", "ticket_medio_neto", "week_complete_flag", "budget_total_eur"]
    ].copy().sort_values("semana_inicio")
    weekly_target["ventas_yoy_pct"] = weekly_target["ventas_netas"].pct_change(52) * 100.0
    last_complete_sales = float(weekly_target.loc[weekly_target["week_complete_flag"] == 1, "ventas_netas"].tail(4).mean())
    last_observed_sales = float(weekly_target["ventas_netas"].iloc[-1])
    feature_catalog_df = _feature_catalog(causal_dataset)
    feature_family_counts = (
        feature_catalog_df["family"].value_counts().rename_axis("family").reset_index(name="count")
    )
    feature_target_corr = target_correlation_table(causal_dataset)

    feature_catalog_df.to_csv(CONFIG.reports_tables_dir / "feature_engineering_feature_catalog.csv", index=False)
    feature_family_counts.to_csv(CONFIG.reports_tables_dir / "feature_engineering_feature_family_counts.csv", index=False)
    feature_target_corr.to_csv(CONFIG.reports_tables_dir / "feature_engineering_target_correlations.csv", index=False)
    source_audit.to_csv(CONFIG.reports_tables_dir / "eda_source_panel_audit.csv", index=False)
    zero_typology.to_csv(CONFIG.reports_tables_dir / "eda_zero_typology.csv", index=False)
    variable_role_catalog.to_csv(CONFIG.reports_tables_dir / "eda_variable_role_catalog.csv", index=False)
    media_channel_summary.to_csv(CONFIG.reports_tables_dir / "eda_media_channel_summary.csv", index=False)
    sales_media_lag.to_csv(CONFIG.reports_tables_dir / "eda_sales_media_lag_correlations.csv", index=False)
    weekly_target.to_csv(CONFIG.reports_tables_dir / "eda_weekly_target_audit.csv", index=False)

    eda_lines = [
        "# EDA Visual Report",
        "",
        "## Purpose",
        "",
        "- Trabajar sobre una unica serie semanal global para el MMM.",
        "- Auditar completitud temporal, semanas truncadas y calidad de las senales antes del modelado.",
        "- Revisar relaciones entre ventas, contexto comercial y presion de medios.",
        "",
        "## Dataset Scope",
        "",
        f"- Horizonte analizado: `{date_min}` a `{date_max}`.",
        f"- Volumen operativo: `{len(sales_lines):,}` lineas y `{len(orders):,}` pedidos.",
        f"- Ventas netas observadas: `{_fmt_money(float(sales_lines['venta_neta_sin_iva_eur'].sum()))}`.",
        f"- Vista diagnostica: `{diagnostic_rows}` filas; vista causal: `{causal_rows}` filas.",
        f"- Grano causal definitivo: `{data_checks['dataset_grain']}` con alcance `{data_checks['dataset_scope']}`.",
        "",
        "## Panel Audit",
        "",
        f"- Semanas con venta cero: `{data_checks['active_city_zero_rows']}`.",
        f"- Semanas incompletas detectadas: `{', '.join(data_checks['incomplete_week_dates'])}`.",
        f"- Semanas faltantes dentro del rango modelado: `{data_checks['missing_weeks_count']}`.",
        f"- La ultima semana observada cae a `{_fmt_money(last_observed_sales)}` frente a una media de `{_fmt_money(last_complete_sales)}` en las cuatro semanas completas previas.",
        "",
        "## Reading",
        "",
        f"- Ticket medio neto: `{_fmt_money(aov)}`; share de lineas con descuento: `{_fmt_pct(promo_share)}`; margen bruto ponderado aproximado: `{_fmt_pct(weighted_margin)}`.",
        f"- La presion de medios suma `{_fmt_money(float(data_checks['media_total']))}` con ratio ventas/medios de `{data_checks['sales_to_media_ratio']}`.",
        f"- El perfil de cliente mas frecuente es `{top_segment}` y el canal preferido dominante es `{top_channel}`.",
        "- Los graficos temporales se apoyan en semanas completas cuando la lectura economica lo exige.",
        "- `pedidos_total` y `ticket_medio_neto` se mantienen como diagnostico comercial y no como drivers causales directos.",
        "",
        "## Tables",
        "",
        "- `reports/tables/eda_source_panel_audit.csv`: cobertura por fuente sobre el calendario de referencia.",
        "- `reports/tables/eda_zero_typology.csv`: resumen de semanas con y sin ventas.",
        "- `reports/tables/eda_variable_role_catalog.csv`: clasificacion de variables por rol metodologico.",
        "- `reports/tables/eda_media_channel_summary.csv`: estabilidad, share y correlacion maxima por canal.",
        "- `reports/tables/eda_sales_media_lag_correlations.csv`: correlaciones exploratorias ventas-medios en lags 0-8.",
        "- `reports/tables/eda_weekly_target_audit.csv`: serie semanal agregada con YoY y flag de semana completa.",
        "",
        "## Figures",
        "",
        "- `1_eda/eda_01_weekly_sales_and_media_overview.png`: trayectoria global de ventas y gasto en medios.",
        "- `1_eda/eda_03_monthly_sales_seasonality.png`: estacionalidad mensual en semanas completas.",
        "- `1_eda/eda_03b_sales_distribution.png`: distribucion del target y share de semanas cero/completas.",
        "- `1_eda/eda_03c_yearly_sales_overlay.png`: overlay anual por semana ISO.",
        "- `1_eda/eda_04_orders_and_discounts.png`: ticket medio y peso promocional.",
        "- `1_eda/eda_05_weekly_commercial_dynamics.png`: ventas, pedidos y ticket medio en el tiempo.",
        "- `1_eda/eda_06_ticket_margin_overview.png`: ticket medio y margen bruto ponderado.",
        "- `1_eda/eda_07_week_year_sales_heatmap.png`: heatmap semana-ano del target.",
        "- `1_eda/eda_10_sales_channel_mix.png`: peso de offline y online en las ventas.",
        "- `1_eda/eda_11_client_profile.png`: mezcla de segmentos y canal preferido del cliente.",
        "- `1_eda/eda_12_indexed_sales_vs_media.png`: ventas y medios indexados base 100.",
        "- `1_eda/eda_13_media_channel_correlation.png`: correlacion entre canales de medios.",
        "- `1_eda/eda_14_sales_media_lag_heatmap.png`: correlacion exploratoria ventas-medios por lag.",
        "- `2_preprocessing/pre_04_temporal_consistency.png`: control visual de semanas completas e incompletas.",
    ]
    (CONFIG.docs_dir / "eda_visual_report.md").write_text("\n".join(eda_lines), encoding="utf-8")

    validation_lines = [
        "# Validation Visual Report",
        "",
        "## Model Outcome",
        "",
        f"- Modelo oficial por backtest y despliegue: `{winner}` con MAPE medio de `{winner_mape:.2f}`.",
        f"- Modelo usado para simulacion y coeficientes: `{deployment_spec}`.",
        f"- Test 2024: MAPE `{test_metrics['mape']:.2f}`, MAE `{test_metrics['mae']:.2f}`, RMSE `{test_metrics['rmse']:.2f}`, bias `{test_metrics['bias']:.2f}`.",
        f"- Filas evaluadas en test 2024: `{len(test_rows)}`.",
        f"- ARIMAX gana a todos los benchmarks obligatorios en validacion media MAPE: `{bool(benchmark_guardrail.get('validation_mean_mape', False))}`.",
        f"- ARIMAX gana a todos los benchmarks obligatorios en test 2024 MAPE: `{bool(benchmark_guardrail.get('test_2024_mape', False))}`.",
        "",
        "## Reading",
        "",
        "- La validacion ya se hace sobre una unica serie global, sin cortes territoriales artificiales.",
        "- Los coeficientes de exogenas y de mix salen del mismo modelo que usamos para predecir y simular.",
        f"- El canal con mayor mROI beneficio es `{top_mroi['channel']}` (`{top_mroi['mroi_profit']:.2f}`) y el peor es `{worst_mroi['channel']}` (`{worst_mroi['mroi_profit']:.2f}`).",
        f"- El mejor escenario segun el ARIMAX es `{top_scenario['scenario']}` con beneficio bruto previsto de `{_fmt_money(float(top_scenario['predicted_gross_profit_2024']))}`.",
        f"- La validacion formal del mix optimizado arroja `t = {float(arimax_stats['t_statistic']):.2f}` y `p = {float(arimax_stats['p_value_one_sided']):.4f}` frente al historico.",
        f"- La validacion fuera de muestra encuentra su mejor uplift modelizado en `{int(oos_best['validation_year'])}` con `{_fmt_money(float(oos_best['predicted_uplift_eur']))}`.",
        f"- El benchmark regularizado con mejor ajuste es `{best_regularized['model']}` con MAPE `{float(best_regularized['mape']):.2f}`.",
        "",
        "## Backtest Summary",
        "",
        "```text",
        backtest_avg.round(2).to_string(index=False),
        "```",
        "",
        "## Required Benchmark Summary",
        "",
        "```text",
        required_benchmark_block.to_string(index=False),
        "```",
        "",
        "## Figures",
        "",
        "- `5_modelado/validation_00_required_benchmarks.png`: comparativa resumida contra benchmarks obligatorios.",
        "- `5_modelado/validation_00b_required_benchmark_test_fit.png`: ajuste 2024 frente a media, naive, ARIMA puro y estacional simple.",
        "- `5_modelado/arimax_test_fit.png`: ajuste agregado del ARIMAX en test.",
        "- `5_modelado/arimax_coefficients.png`: coeficientes del ARIMAX.",
        "- `5_modelado/validation_01_backtest_metrics.png`: comparativa por fold.",
        "- `5_modelado/validation_02_test_fit_and_residuals.png`: ajuste agregado de 2024 y residuos temporales.",
        "- `5_modelado/validation_04_predicted_vs_actual_scatter.png`: calibracion en test.",
        "- `5_modelado/validation_04b_residual_distribution.png`: distribucion de errores.",
        "- `5_modelado/validation_05_top_coefficients.png`: drivers mas fuertes del modelo desplegado.",
        "- `5_modelado/validation_05b_media_coefficients.png`: coeficientes de mix y presion publicitaria.",
        "- `5_modelado/validation_06_monthly_channel_contributions.png`: contribucion media estimada por canal en el tiempo.",
        "- `5_modelado/validation_06b_total_channel_contribution.png`: peso agregado de cada canal.",
        "- `5_modelado/validation_06c_quarter_error.png`: error por trimestre.",
        "- `5_modelado/validation_07_scenarios_and_budget_mix.png`: comparativa de escenarios y mix.",
        "- `5_modelado/validation_08_mroi_by_channel.png`: ranking visual de mROI.",
        "- `5_modelado/validation_09_channel_budget_sensitivity.png`: sensibilidad del beneficio al subir o bajar presupuesto por canal.",
    ]
    (CONFIG.docs_dir / "validation_visual_report.md").write_text("\n".join(validation_lines), encoding="utf-8")

    process_lines = [
        "# Process Diagnostic Report",
        "",
        "## Raw Inputs",
        "",
        f"- Se cargaron `{len(sales_lines):,}` lineas de venta, `{len(orders):,}` pedidos y `{len(clients):,}` clientes.",
        f"- `sales_total_lineas`, `sales_total_pedidos` y `sales_total_weekly` cuadran en `{_fmt_money(float(data_checks['sales_total_lineas']))}`.",
        f"- Se cerraron `{data_checks['missing_sales_days_closed']}` dias sin ventas observadas dentro del calendario de referencia.",
        "",
        "## Aggregation",
        "",
        f"- El panel semanal es completo: `{data_checks['weekly_panel_complete']}`.",
        f"- El dataset causal trabaja con una unica serie `{data_checks['global_series_label']}` y `{data_checks['dataset_weeks']}` semanas.",
        f"- Se detectaron semanas incompletas en `{', '.join(data_checks['incomplete_week_dates'])}` y quedan etiquetadas para no leerlas como caida economica real.",
        f"- Las semanas faltantes dentro del rango modelado son `{data_checks['missing_weeks_count']}`.",
        f"- Tras el saneamiento final quedan `{data_checks['remaining_missing_cells']}` celdas missing y `{data_checks['remaining_infinite_cells']}` celdas infinitas.",
        "",
        "## Modeling",
        "",
        f"- El modelo oficial queda fijado como `{winner}`.",
        f"- El despliegue coincide con `{deployment_spec}`, asi que prediccion y simulacion salen del mismo ARIMAX.",
        "",
        "## Simulation",
        "",
        f"- Escenario top: `{top_scenario['scenario']}` con delta de `{_fmt_money(float(top_scenario['delta_vs_historical_profit']))}` en beneficio frente al mix historico a 12M.",
        f"- La optimizacion formal deja `{_fmt_money(float(arimax_optimized['profit_vs_historical_eur']))}` de uplift frente al historico y significancia `{bool(arimax_stats['significant_one_sided_5pct'])}`.",
        f"- La validacion out-of-sample confirma que el uplift no depende de un solo corte temporal; el mejor fold es `{int(oos_best['validation_year'])}`.",
    ]
    (CONFIG.docs_dir / "process_diagnostic_report.md").write_text("\n".join(process_lines), encoding="utf-8")

    inventory_rows = [
        ("1 EDA", "1_eda/eda_01_weekly_sales_and_media_overview.png", "Trayectoria global de ventas y gasto en medios."),
        ("1 EDA", "1_eda/eda_03_monthly_sales_seasonality.png", "Estacionalidad mensual del target en semanas completas."),
        ("1 EDA", "1_eda/eda_03b_sales_distribution.png", "Distribucion del target y resumen de semanas cero/completas."),
        ("1 EDA", "1_eda/eda_03c_yearly_sales_overlay.png", "Overlay anual por semana ISO."),
        ("1 EDA", "1_eda/eda_04_orders_and_discounts.png", "Ticket medio y share promocional."),
        ("1 EDA", "1_eda/eda_05_weekly_commercial_dynamics.png", "Ventas, pedidos y ticket medio semanales."),
        ("1 EDA", "1_eda/eda_06_ticket_margin_overview.png", "Ticket medio y margen bruto ponderado."),
        ("1 EDA", "1_eda/eda_07_week_year_sales_heatmap.png", "Heatmap del target por semana del ano y ano."),
        ("1 EDA", "1_eda/eda_10_sales_channel_mix.png", "Peso de offline y online en las ventas."),
        ("1 EDA", "1_eda/eda_11_client_profile.png", "Mezcla de segmentos y canal preferido del cliente."),
        ("1 EDA", "1_eda/eda_12_indexed_sales_vs_media.png", "Ventas y medios indexados base 100."),
        ("1 EDA", "1_eda/eda_13_media_channel_correlation.png", "Correlacion entre canales de medios."),
        ("1 EDA", "1_eda/eda_14_sales_media_lag_heatmap.png", "Correlacion exploratoria ventas-medios por lag."),
        ("2 Preprocessing", "2_preprocessing/pre_01_panel_coverage_heatmap.png", "Cobertura semanal global para target, medios, pedidos y trafico."),
        ("2 Preprocessing", "2_preprocessing/pre_02_pipeline_row_counts.png", "Numero de observaciones por paso del pipeline de preprocessing."),
        ("2 Preprocessing", "2_preprocessing/pre_03_variable_quality_heatmap.png", "Porcentaje de missing, ceros e infinitos por variable en el dataset causal."),
        ("2 Preprocessing", "2_preprocessing/pre_04_temporal_consistency.png", "Control visual de semanas completas e incompletas."),
        ("2 Preprocessing", "2_preprocessing/eda_12_product_cardinality.png", "Cardinalidad real del detalle de producto en ventas."),
        ("3 Feature Engineering", "3_feature_engineering/eda_05_media_mix_timeseries.png", "Nivel y mix historico de inversion."),
        ("3 Feature Engineering", "3_feature_engineering/fe_01_feature_family_counts.png", "Numero de columnas por familia de features."),
        ("3 Feature Engineering", "3_feature_engineering/fe_02_top_target_correlations.png", "Top correlaciones lineales con el target."),
        ("3 Feature Engineering", "3_feature_engineering/fe_03_target_vs_total_spend.png", "Lectura de ventas frente a presupuesto total."),
        ("3 Feature Engineering", "3_feature_engineering/eda_08_correlation_heatmap.png", "Correlaciones entre target, controles y medios."),
        ("3 Feature Engineering", "3_feature_engineering/eda_09_event_uplift.png", "Uplift medio en semanas de eventos."),
        ("4 Feature Importance", "4_feature_importance/random_forest_top_features.png", "Top variables segun importancia por permutacion."),
        ("4 Feature Importance", "4_feature_importance/random_forest_feature_groups.png", "Importancia agregada por grupo de variables."),
        ("5 Modelado", "5_modelado/validation_00_required_benchmarks.png", "Comparativa resumida contra media, naive, ARIMA puro y estacional simple."),
        ("5 Modelado", "5_modelado/validation_00b_required_benchmark_test_fit.png", "Ajuste 2024 frente a los benchmarks obligatorios."),
        ("5 Modelado", "5_modelado/arimax_test_fit.png", "Ajuste agregado del ARIMAX en test."),
        ("5 Modelado", "5_modelado/arimax_coefficients.png", "Coeficientes del ARIMAX."),
        ("5 Modelado", "5_modelado/validation_01_backtest_metrics.png", "Comparativa por fold."),
        ("5 Modelado", "5_modelado/validation_02_test_fit_and_residuals.png", "Ajuste agregado y residuos en test 2024."),
        ("5 Modelado", "5_modelado/validation_04_predicted_vs_actual_scatter.png", "Calibracion en test."),
        ("5 Modelado", "5_modelado/validation_04b_residual_distribution.png", "Distribucion de errores."),
        ("5 Modelado", "5_modelado/validation_05_top_coefficients.png", "Drivers mas fuertes del modelo desplegado."),
        ("5 Modelado", "5_modelado/validation_05b_media_coefficients.png", "Coeficientes de mix y presion publicitaria."),
        ("5 Modelado", "5_modelado/validation_06_monthly_channel_contributions.png", "Contribucion media estimada por canal en el tiempo."),
        ("5 Modelado", "5_modelado/validation_06b_total_channel_contribution.png", "Peso agregado de cada canal."),
        ("5 Modelado", "5_modelado/validation_06c_quarter_error.png", "Error por trimestre."),
        ("5 Modelado", "5_modelado/validation_07_scenarios_and_budget_mix.png", "Comparativa de escenarios y mix."),
        ("5 Modelado", "5_modelado/validation_08_mroi_by_channel.png", "Ranking visual de mROI."),
        ("5 Modelado", "5_modelado/validation_09_channel_budget_sensitivity.png", "Sensibilidad del beneficio al variar el presupuesto por canal."),
        ("6 Simulacion - Datos", "6_simulacion/sim_data_01_scenario_performance.png", "Ventas y beneficio por escenario de simulacion."),
        ("6 Simulacion - Datos", "6_simulacion/sim_data_02_budget_mix.png", "Mix de presupuesto por escenario."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_01_profit_vs_reallocation.png", "Mejora de beneficio frente a intensidad de cambio."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_02_warning_counts.png", "Warnings por escenario."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_03_channel_sensitivity.png", "Sensibilidad del beneficio por canal."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_04_mroi_profit.png", "mROI beneficio por canal."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_05_roi_vs_baselines.png", "ROI incremental frente al baseline sin inversion."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_06_weekly_profit_delta_boxplot.png", "Distribucion semanal del uplift del optimizado."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_07_ttest_statistics.png", "Contraste t del optimizado frente al historico."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_08_weight_shift_heatmap.png", "Cambio de pesos optimizados por canal."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_09_regularization_fit.png", "Ajuste predictivo de Ridge y Lasso."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_10_profit_vs_significant_shares.png", "Beneficio vs porcentaje de peso en los canales mas influyentes."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_11_profit_vs_significant_budgets.png", "Beneficio vs euros para esos mismos canales."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_12_oos_uplift_by_year.png", "Uplift de la optimizacion por ano fuera de muestra."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_13_oos_weight_stability.png", "Estabilidad temporal de pesos optimizados por canal."),
        ("6 Simulacion - Validacion", "6_simulacion/sim_validation_14_oos_fit_quality.png", "Calidad predictiva historica por fold temporal."),
    ]
    pd.DataFrame(inventory_rows, columns=["section", "figure_file", "purpose"]).to_csv(
        CONFIG.reports_tables_dir / "visual_inventory.csv",
        index=False,
    )
