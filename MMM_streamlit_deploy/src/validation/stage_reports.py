from __future__ import annotations

import json

import pandas as pd

from src.common.config import CONFIG
from src.features.metadata import feature_catalog, target_correlation_table


def _read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_money(value: float) -> str:
    return f"{value:,.2f} EUR"


def write_stage_reports() -> None:
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)
    checks = _read_json(CONFIG.data_checks_file)
    model_results = _read_json(CONFIG.model_results_file)

    causal = pd.read_parquet(CONFIG.model_dataset_file)
    diagnostic = pd.read_parquet(CONFIG.diagnostic_dataset_file)
    sales_lines = pd.read_csv(CONFIG.sales_lines_file, parse_dates=["fecha_venta"])
    orders = pd.read_csv(CONFIG.orders_file, parse_dates=["fecha_pedido"])
    clients = pd.read_csv(CONFIG.clients_file, parse_dates=["fecha_alta"])
    rf_features = pd.read_csv(CONFIG.reports_tables_dir / "random_forest_feature_importance.csv")
    rf_groups = pd.read_csv(CONFIG.reports_tables_dir / "random_forest_feature_group_importance.csv")
    benchmark_summary = pd.read_csv(CONFIG.reports_tables_dir / "arimax_benchmark_summary.csv")
    arimax_backtest = pd.read_csv(CONFIG.reports_tables_dir / "arimax_backtest.csv")
    arimax_coefficients = pd.read_csv(CONFIG.reports_tables_dir / "arimax_coefficients.csv")
    scenarios = pd.read_csv(CONFIG.scenario_results_file)
    mroi = pd.read_csv(CONFIG.reports_tables_dir / "mroi_by_channel.csv")
    scenario_diagnostics = pd.read_csv(CONFIG.reports_tables_dir / "scenario_diagnostics.csv")

    sales_total = float(sales_lines["venta_neta_sin_iva_eur"].sum())
    date_min = sales_lines["fecha_venta"].min().date().isoformat()
    date_max = sales_lines["fecha_venta"].max().date().isoformat()
    aov = float(orders["importe_neto_sin_iva_eur"].mean())
    promo_share = float((sales_lines["descuento_pct"].fillna(0.0) > 0).mean() * 100.0)
    top_segment = str(clients["segmento"].mode().iloc[0])
    top_channel = str(clients["canal_preferido"].mode().iloc[0])

    feature_catalog_df = feature_catalog(causal)
    family_counts = feature_catalog_df["family"].value_counts().rename_axis("family").reset_index(name="count")
    tier_counts = feature_catalog_df["modeling_tier"].value_counts().rename_axis("modeling_tier").reset_index(name="count")
    target_corr = target_correlation_table(causal)

    best_scenario = scenarios.sort_values("predicted_gross_profit_2024", ascending=False).iloc[0]
    best_scenario_diagnostics = (
        scenario_diagnostics.loc[scenario_diagnostics["scenario"] == best_scenario["scenario"]]
        .head(1)
        .squeeze()
    )
    safest_scenario = scenario_diagnostics.sort_values(
        ["warning_count", "max_share_shift_pct_points", "delta_vs_historical_profit"],
        ascending=[True, True, False],
    ).iloc[0]
    best_mroi = mroi.sort_values("mroi_profit", ascending=False).iloc[0]
    worst_mroi = mroi.sort_values("mroi_profit", ascending=True).iloc[0]
    business_coefficients = arimax_coefficients[arimax_coefficients["is_business_variable"]].copy()
    benchmark_guardrail = model_results.get("required_benchmark_guardrail", {})
    benchmark_block = benchmark_summary[
        ["spec", "validation_mean_mape", "test_2024_mape", "validation_rank_mape", "test_rank_mape"]
    ].round(2)

    eda_lines = [
        "# 01. EDA Report",
        "",
        "## Objetivo",
        "",
        "- Fijar el grano del MMM como una serie `semana_inicio` global.",
        "- Auditar calidad temporal, semanas incompletas y consistencia entre ventas y medios.",
        "- Revisar patrones de negocio y contexto antes del modelado causal.",
        "",
        "## Foto del dataset",
        "",
        f"- Horizonte temporal: `{date_min}` a `{date_max}`.",
        f"- Lineas de venta: `{len(sales_lines):,}`.",
        f"- Pedidos: `{len(orders):,}`.",
        f"- Ventas netas observadas: `{_fmt_money(sales_total)}`.",
        f"- Ticket medio neto observado: `{_fmt_money(aov)}`.",
        f"- Share promocional: `{promo_share:.2f}%`.",
        "",
        "## Lecturas clave",
        "",
        f"- La vista diagnostica contiene `{checks['diagnostic_dataset_rows']}` semanas y la causal `{checks['causal_dataset_rows']}`.",
        f"- Hay `{checks['active_city_zero_rows']}` semanas con venta cero y `{checks['incomplete_weeks_in_diagnostic']}` semanas incompletas etiquetadas.",
        f"- Las semanas incompletas quedan identificadas en `{', '.join(checks['incomplete_week_dates'])}`.",
        f"- El segmento de cliente mas frecuente es `{top_segment}` y el canal preferido dominante es `{top_channel}`.",
        "- El EDA confirma que el dataset ya debe leerse como una serie semanal agregada, no como panel territorial.",
        "",
        "## Entregables",
        "",
        "- `docs/eda_visual_report.md`.",
        "- `reports/figures/1_eda`.",
        "- `reports/tables/visual_inventory.csv`.",
        "- `reports/tables/eda_source_panel_audit.csv`.",
        "- `reports/tables/eda_zero_typology.csv`.",
        "- `reports/tables/eda_variable_role_catalog.csv`.",
    ]
    (CONFIG.docs_dir / "01_eda_report.md").write_text("\n".join(eda_lines), encoding="utf-8")

    preprocessing_lines = [
        "# 02. Preprocessing Report",
        "",
        "## Objetivo",
        "",
        "- Construir una serie semanal global coherente y defendible para MMM.",
        "- Alinear ventas, pedidos, calendario, trafico y medios sobre el mismo calendario semanal.",
        "- Formalizar reglas de joins, ceros, missing, infinitos y semanas incompletas antes del modelado.",
        "",
        "## Grano Final",
        "",
        f"- Definicion de semana: `{checks['week_definition']}`.",
        f"- Grano final del panel: `{checks['dataset_grain']}` con alcance `{checks['dataset_scope']}`.",
        f"- Serie tecnica retenida por compatibilidad: `{checks['global_series_label']}`.",
        f"- El target sale de `{checks['target_source_column']}` agregada desde lineas de venta, no desde cabecera de pedido.",
        "",
        "## Resultado",
        "",
        f"- Dias sin ventas observadas cerrados contra calendario: `{checks['missing_sales_days_closed']}`.",
        f"- Panel semanal completo: `{checks['weekly_panel_complete']}`.",
        f"- Integridad de joins: filas preservadas en todos los merges `{checks['join_row_count_preserved']}` con maximo de duplicados post-join `{checks['max_duplicate_keys_after_join']}`.",
        f"- El panel diagnostico preserva la linea temporal completa; la vista causal arranca en `{causal['semana_inicio'].min().date().isoformat()}` porque excluye semanas incompletas de borde.",
        f"- Semanas incompletas detectadas: `{', '.join(checks['incomplete_week_dates'])}`.",
        f"- Semanas faltantes dentro del rango modelado: `{checks['missing_weeks_count']}`.",
        f"- Filas negativas en ventas raw: `{checks['raw_sales_negative_line_rows']}`; filas negativas tras agregacion semanal: `{checks['weekly_sales_negative_rows']}`.",
        f"- Missing restantes tras el saneamiento: `{checks['remaining_missing_cells']}`; infinitos restantes: `{checks['remaining_infinite_cells']}`.",
        "",
        "## Decisiones importantes",
        "",
        "- Las tablas transaccionales se agregan primero a nivel global y solo despues se unen a la grilla semanal.",
        "- `weekly_media` se construye sumando ciudad-canal a semana-canal antes de pivotar el mix.",
        "- Los joins se hacen con `LEFT JOIN` sobre la serie base semanal de calendario para no truncar semanas con ventas pero sin medios.",
        "- Las razones y porcentajes derivados se calculan con guardas de denominador y sustitucion explicita de `inf` por `NA`; el fill a `0.0` se limita a familias donde ese cero tiene significado operativo.",
        "- IDs y textos de alta cardinalidad se dejan fuera del dataset modelable.",
        "- La vista diagnostica conserva operativa y control de calidad; la causal se queda con semanas completas y sin proxies downstream.",
        "",
        "## Auditorias",
        "",
        "- `reports/tables/preprocessing_source_grain_audit.csv`.",
        "- `reports/tables/preprocessing_join_audit.csv`.",
        "- `reports/tables/preprocessing_panel_coverage.csv`.",
        "- `reports/tables/preprocessing_missing_zero_rules.csv`.",
        "- `reports/tables/preprocessing_dataset_summary.csv`.",
        "- `reports/tables/preprocessing_variable_quality.csv`.",
        "- `reports/tables/preprocessing_target_rollup_audit.csv`.",
        "- `reports/tables/preprocessing_temporal_consistency.csv`.",
        "- `reports/tables/preprocessing_zero_sales_audit.csv`.",
        "",
        "## Figuras recomendadas de control",
        "",
        "- `reports/figures/2_preprocessing/pre_01_panel_coverage_heatmap.png`.",
        "- `reports/figures/2_preprocessing/pre_02_pipeline_row_counts.png`.",
        "- `reports/figures/2_preprocessing/pre_03_variable_quality_heatmap.png`.",
        "- `reports/figures/2_preprocessing/pre_04_temporal_consistency.png`.",
        "",
        "## Entregables",
        "",
        "- `docs/preprocessing_decisions.md`.",
        "- `docs/step_1_preprocessing_scope.md`.",
        "- `docs/process_diagnostic_report.md`.",
        "- `reports/figures/2_preprocessing`.",
    ]
    (CONFIG.docs_dir / "02_preprocessing_report.md").write_text("\n".join(preprocessing_lines), encoding="utf-8")

    feature_importance_lines = [
        "# 04. Feature Importance Report",
        "",
        "## Objetivo",
        "",
        "- Hacer un screening previo de variables antes del ARIMAX final.",
        "- Comparar rutas de features antes de entrar en modelado.",
        "- Revisar medios, mix y exogenas sin mezclar todas las representaciones a la vez.",
        "",
        "## Lectura de specs",
        "",
        "- `RFControlsOnly`: baseline con controles exogenos y estacionalidad compacta.",
        "- `RFMediaLevels`: controles + `media_*`.",
        "- `RFBudgetMix`: controles + `budget_total_eur` + pesos `n-1`.",
        "",
        "## Top variables por permutacion",
        "",
    ]
    for row in rf_features.head(10).itertuples(index=False):
        feature_importance_lines.append(
            f"- `{row.feature}` ({row.feature_group}): `{row.permutation_importance_mean:.6f}`."
        )
    feature_importance_lines += [
        "",
        "## Top grupos de variables",
        "",
    ]
    for row in rf_groups.head(6).itertuples(index=False):
        feature_importance_lines.append(
            f"- `{row.feature_group}`: `{row.permutation_importance_mean:.6f}`."
        )
    feature_importance_lines += [
        "",
        "## Lectura",
        "",
        "- El feature importance se usa como filtro orientativo, no como regla automatica de inclusion.",
        "- Separar `media_*` de `budget_total_eur + budget_share_pct_*` vuelve el screening bastante mas interpretable.",
        "- Si la ruta con medios no mejora frente al baseline de controles, eso es una senal de prudencia antes de modelar.",
        "- La senal no lineal sigue siendo debil, asi que este paso sirve mas para descartar sobrelecturas que para imponer variables.",
        "",
        "## Entregables",
        "",
        "- `reports/tables/random_forest_spec_summary.csv`.",
        "- `docs/random_forest_feature_importance.md`.",
        "- `docs/step_4_feature_importance.md`.",
        "- `reports/figures/4_feature_importance`.",
    ]
    (CONFIG.docs_dir / "04_feature_importance_report.md").write_text(
        "\n".join(feature_importance_lines),
        encoding="utf-8",
    )

    modelado_lines = [
        "# 05. Modelado Report",
        "",
        "## Modelo oficial",
        "",
        "- Tipo: `ARIMAX / SARIMAX`.",
        f"- MAPE medio de backtest: `{arimax_backtest['mape'].mean():.2f}`.",
        f"- MAPE test 2024: `{model_results['test_metrics']['mape']:.2f}`.",
        f"- RMSE test 2024: `{model_results['test_metrics']['rmse']:.2f}`.",
        f"- Gana a todos los benchmarks obligatorios en validacion MAPE: `{bool(benchmark_guardrail.get('validation_mean_mape', False))}`.",
        f"- Gana a todos los benchmarks obligatorios en test MAPE: `{bool(benchmark_guardrail.get('test_2024_mape', False))}`.",
        "",
        "## Benchmarks obligatorios",
        "",
        "```text",
        benchmark_block.to_string(index=False),
        "```",
        "",
        "## Lectura",
        "",
        "- El modelo ya trabaja sobre una unica serie semanal global, por lo que la validacion se interpreta de forma agregada.",
        "- Los coeficientes combinan estacionalidad, contexto y mix de medios en la misma ecuacion de despliegue.",
        "- La lectura metodologica es mas limpia porque no depende de efectos fijos territoriales.",
        "",
        "## Entregables",
        "",
        "- `docs/arimax_model_report.md`.",
        "- `docs/validation_visual_report.md`.",
        "- `reports/figures/5_modelado`.",
    ]
    (CONFIG.docs_dir / "05_modelado_report.md").write_text("\n".join(modelado_lines), encoding="utf-8")

    feature_engineering_lines = [
        "# 03. Feature Engineering Report",
        "",
        "## Objetivo",
        "",
        "- Construir el panel semanal causal listo para modelado.",
        "- Hacer visibles las familias de variables que entran al ARIMAX.",
        "- Separar claramente claves, negocio, mix de medios, temporalidad y contexto.",
        "",
        "## Resumen",
        "",
        f"- Filas del dataset causal: `{checks['causal_dataset_rows']}`.",
        f"- Columnas del dataset causal: `{checks['causal_dataset_columns']}`.",
        f"- Ticket medio neto medio en diagnostico: `{checks['ticket_medio_neto_mean']}`.",
        f"- Suma media de `budget_share_pct_*` en semanas con presupuesto: `{checks['budget_share_pct_sum_mean']}`.",
        f"- Features `model_ready`: `{int(tier_counts.loc[tier_counts['modeling_tier'] == 'model_ready', 'count'].sum())}`.",
        f"- Features de `screening_only`: `{int(tier_counts.loc[tier_counts['modeling_tier'] == 'screening_only', 'count'].sum())}`.",
        "",
        "## Familias de variables",
        "",
    ]
    for row in family_counts.itertuples(index=False):
        feature_engineering_lines.append(f"- `{row.family}`: `{row.count}` columnas.")
    feature_engineering_lines += [
        "",
        "## Variables con mayor relacion lineal con el target",
        "",
    ]
    for row in target_corr.head(12).itertuples(index=False):
        feature_engineering_lines.append(f"- `{row.feature}`: `{row.correlation_with_target:.4f}`.")
    feature_engineering_lines += [
        "",
        "## Lectura metodologica",
        "",
        "- La vista causal ya no incluye proxies downstream; `pedidos_total` y `ticket_medio_neto` quedan solo en diagnostico.",
        "- `media_*` se conservan para screening y comparativas de especificacion, mientras que `budget_total_eur` y `budget_share_pct_*` forman la ruta mas cercana al MMM final.",
        "- Etiquetas temporales como `year`, `quarter` o `mes_modal` se retienen para lectura y QA, pero no lideran el ranking de features elegibles.",
        "",
        "## Salidas",
        "",
        "- `reports/tables/feature_engineering_feature_catalog.csv`.",
        "- `reports/tables/feature_engineering_feature_family_counts.csv`.",
        "- `reports/tables/feature_engineering_target_correlations.csv`.",
        "- `reports/figures/3_feature_engineering/fe_01_feature_family_counts.png`.",
        "- `reports/figures/3_feature_engineering/fe_02_top_target_correlations.png`.",
        "- `reports/figures/3_feature_engineering/fe_03_target_vs_total_spend.png`.",
        "",
        "## Conclusion",
        "",
        "- La serie causal queda lista para modelado sin dependencia de segmentacion territorial.",
        "- El objetivo es que el salto hacia `feature importance` y `modelado` sea mas legible y defendible.",
    ]
    (CONFIG.docs_dir / "03_feature_engineering_report.md").write_text(
        "\n".join(feature_engineering_lines),
        encoding="utf-8",
    )

    simulation_lines = [
        "# 06. Simulacion Report",
        "",
        "## Objetivo",
        "",
        "- Tensionar el ARIMAX con presupuesto fijo de `12M EUR`.",
        "- Medir si el juego de coeficientes genera decisiones de beneficio coherentes.",
        "- Separar lectura de datos obtenidos y validacion de la simulacion.",
        "",
        "## Datos Obtenidos",
        "",
        f"- Mejor escenario por beneficio: `{best_scenario['scenario']}`.",
        f"- Beneficio bruto previsto: `{_fmt_money(float(best_scenario['predicted_gross_profit_2024']))}`.",
        f"- Delta vs historico: `{_fmt_money(float(best_scenario['delta_vs_historical_profit']))}`.",
        f"- Mejor canal por mROI beneficio: `{best_mroi['channel']}`.",
        f"- Peor canal por mROI beneficio: `{worst_mroi['channel']}`.",
        "",
        "## Validacion De La Simulacion",
        "",
        f"- Reasignacion total del escenario ganador: `{_fmt_money(float(best_scenario_diagnostics['total_budget_reallocated_eur']))}`.",
        f"- Maximo cambio de share observado: `{float(best_scenario_diagnostics['max_share_shift_pct_points']):.2f}` puntos.",
        f"- Escenario mas prudente: `{safest_scenario['scenario']}` con `{int(safest_scenario['warning_count'])}` warnings.",
        "- La sensibilidad por canal permite ver si la mejora se mantiene al mover presupuesto alrededor del historico.",
        "",
        "## Figuras De Referencia",
        "",
        "- `6_simulacion/sim_data_01_scenario_performance.png`.",
        "- `6_simulacion/sim_data_02_budget_mix.png`.",
        "- `6_simulacion/sim_validation_03_channel_sensitivity.png`.",
        "- `6_simulacion/sim_validation_04_mroi_profit.png`.",
        "- `6_simulacion/sim_validation_05_roi_vs_baselines.png`.",
        "- `6_simulacion/sim_validation_14_oos_fit_quality.png`.",
    ]
    (CONFIG.docs_dir / "06_simulacion_report.md").write_text("\n".join(simulation_lines), encoding="utf-8")
