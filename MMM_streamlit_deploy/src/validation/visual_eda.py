from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.config import CONFIG
from src.features.metadata import feature_catalog, target_correlation_table
from src.validation.visual_utils import (
    BLUE,
    DIVERGING,
    GREEN,
    LIGHT_BLUE,
    NAVY,
    PANEL_BG,
    SAND,
    SEQUENTIAL,
    SEQUENTIAL_CMAP,
    SLATE,
    TEAL,
    savefig,
    style_axes,
)


def _weekly_dataset(dataset: pd.DataFrame, complete_only: bool = False) -> pd.DataFrame:
    weekly = dataset.sort_values("semana_inicio").copy()
    if complete_only and "week_complete_flag" in weekly.columns:
        weekly = weekly[weekly["week_complete_flag"] == 1].copy()
    return weekly.reset_index(drop=True)


def eda_weekly_sales_overview(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset)
    weekly["rolling_13w"] = weekly["ventas_netas"].rolling(13, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(weekly["semana_inicio"], weekly["ventas_netas"], color=NAVY, alpha=0.88, linewidth=2.0, label="Ventas")
    axes[0].plot(weekly["semana_inicio"], weekly["rolling_13w"], color=TEAL, linewidth=2.6, label="Media movil 13 semanas")
    axes[0].set_title("Ventas Semanales Globales")
    axes[0].legend()

    axes[1].plot(weekly["semana_inicio"], weekly["budget_total_eur"], color=GREEN, linewidth=2.2)
    axes[1].set_title("Inversion Semanal en Medios")
    axes[1].set_xlabel("Semana")
    style_axes(list(axes), grid_axis="both")
    savefig("eda", "eda_01_weekly_sales_and_media_overview.png")


def preprocessing_panel_coverage_heatmap() -> None:
    coverage = pd.read_csv(CONFIG.reports_tables_dir / "preprocessing_panel_coverage.csv", parse_dates=["semana_inicio"])
    plot_df = coverage[
        [
            "semana_inicio",
            "target_available",
            "media_available",
            "orders_available",
            "traffic_available",
            "week_complete_flag",
        ]
    ].copy()
    plot_df = plot_df.rename(
        columns={
            "target_available": "Target",
            "media_available": "Media",
            "orders_available": "Pedidos",
            "traffic_available": "Trafico",
            "week_complete_flag": "Semana completa",
        }
    )
    matrix = plot_df.set_index("semana_inicio").T.astype(float)
    plt.figure(figsize=(16, 4.8))
    sns.heatmap(matrix, cmap="YlGnBu", cbar=False)
    plt.title("Cobertura Semanal Global por Senal")
    plt.xlabel("Semana")
    plt.ylabel("")
    savefig("preprocessing", "pre_01_panel_coverage_heatmap.png")


def preprocessing_pipeline_row_counts() -> None:
    target_rollup = pd.read_csv(CONFIG.reports_tables_dir / "preprocessing_target_rollup_audit.csv")
    checks = pd.read_json(CONFIG.data_checks_file, typ="series")
    plot_df = pd.DataFrame(
        [
            {"stage": "Lineas de venta", "rows": int(target_rollup.loc[target_rollup["stage"] == "raw_sales_lines", "rows"].iloc[0])},
            {"stage": "Ventas semanales", "rows": int(target_rollup.loc[target_rollup["stage"] == "weekly_sales_global", "rows"].iloc[0])},
            {"stage": "Panel base calendario", "rows": int(checks["calendar_panel_anchor_rows"])},
            {"stage": "Dataset diagnostico", "rows": int(checks["diagnostic_dataset_rows"])},
            {"stage": "Dataset causal", "rows": int(checks["causal_dataset_rows"])},
        ]
    )
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="stage", y="rows", hue="stage", palette=sns.blend_palette(SEQUENTIAL, n_colors=len(plot_df)), legend=False)
    ax.set_yscale("log")
    ax.set_title("Observaciones por Paso del Pipeline")
    ax.set_xlabel("")
    ax.set_ylabel("Filas (escala log)")
    plt.xticks(rotation=20, ha="right")
    style_axes(ax, grid_axis="y")
    for patch, row in zip(ax.patches, plot_df.itertuples(index=False)):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{row.rows:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    savefig("preprocessing", "pre_02_pipeline_row_counts.png")


def preprocessing_variable_quality_heatmap() -> None:
    quality = pd.read_csv(CONFIG.reports_tables_dir / "preprocessing_variable_quality.csv")
    quality = quality[quality["dataset"] == "causal"].copy()
    quality["score"] = quality[["missing_pct", "zero_pct", "inf_pct"]].fillna(0.0).max(axis=1)
    top = quality.sort_values(["score", "zero_pct"], ascending=False).head(18)
    matrix = top.set_index("variable")[["missing_pct", "zero_pct", "inf_pct"]].fillna(0.0)
    plt.figure(figsize=(10, max(6, len(matrix) * 0.34)))
    sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".2f")
    plt.title("Calidad de Variables en el Dataset Causal")
    plt.xlabel("")
    plt.ylabel("")
    savefig("preprocessing", "pre_03_variable_quality_heatmap.png")


def preprocessing_temporal_consistency(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset)
    colors = weekly["week_complete_flag"].map({1: BLUE, 0: LIGHT_BLUE})
    plt.figure(figsize=(14, 4.8))
    plt.bar(weekly["semana_inicio"], weekly["days_in_week"], color=colors, width=5)
    plt.axhline(7, color=SLATE, linestyle="--", linewidth=1.2)
    plt.title("Cobertura Temporal por Semana")
    plt.xlabel("Semana")
    plt.ylabel("Dias observados")
    style_axes(plt.gca(), grid_axis="y")
    savefig("preprocessing", "pre_04_temporal_consistency.png")


def eda_monthly_seasonality(dataset: pd.DataFrame) -> None:
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    monthly = _weekly_dataset(dataset, complete_only=True)
    monthly["month_label"] = monthly["mes_modal"].map(month_map)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=monthly, x="month_label", y="ventas_netas", color=LIGHT_BLUE, saturation=0.95)
    plt.title("Distribucion de Ventas Semanales por Mes")
    plt.xlabel("Mes")
    plt.ylabel("Ventas semanales")
    style_axes(plt.gca(), grid_axis="y")
    savefig("eda", "eda_03_monthly_sales_seasonality.png")


def eda_sales_distribution(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset)
    zero_share = float((weekly["ventas_netas"] == 0.0).mean() * 100.0)
    summary = pd.DataFrame(
        [
            {"metric": "Share semanas con venta cero", "value": zero_share},
            {"metric": "Share semanas completas", "value": float((weekly["week_complete_flag"] == 1).mean() * 100.0)},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.histplot(weekly["ventas_netas"], bins=35, kde=True, color="#1f4e79", ax=axes[0])
    axes[0].set_title("Distribucion de Ventas Semanales Globales")
    axes[0].set_xlabel("Ventas semanales")

    sns.barplot(data=summary, x="metric", y="value", hue="metric", palette="crest", legend=False, ax=axes[1])
    axes[1].set_title("Resumen de Ceros y Cobertura")
    axes[1].set_ylabel("Porcentaje")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    savefig("eda", "eda_03b_sales_distribution.png")


def eda_yearly_sales_overlay(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    weekly["year"] = weekly["semana_inicio"].dt.year
    weekly["week_of_year"] = weekly["semana_inicio"].dt.isocalendar().week.astype(int)
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=weekly, x="week_of_year", y="ventas_netas", hue="year", palette="mako")
    plt.title("Overlay de Ventas por Semana ISO y Ano")
    plt.xlabel("Semana ISO")
    plt.ylabel("Ventas semanales")
    savefig("eda", "eda_03c_yearly_sales_overlay.png")


def eda_orders_and_discounts(sales_lines: pd.DataFrame, orders: pd.DataFrame) -> None:
    sales_lines = sales_lines.copy()
    orders = orders.copy()
    sales_lines["month"] = sales_lines["fecha_venta"].dt.to_period("M").dt.to_timestamp()
    orders["month"] = orders["fecha_pedido"].dt.to_period("M").dt.to_timestamp()

    discount_monthly = sales_lines.groupby("month", as_index=False).agg(
        promo_share=("descuento_pct", lambda s: float((s.fillna(0.0) > 0).mean() * 100.0)),
    )
    aov_monthly = orders.groupby("month", as_index=False)["importe_neto_sin_iva_eur"].mean()
    clipped_aov = orders["importe_neto_sin_iva_eur"].clip(upper=orders["importe_neto_sin_iva_eur"].quantile(0.99))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.histplot(clipped_aov, bins=50, kde=True, color="#1f4e79", ax=axes[0])
    axes[0].set_title("Distribucion de Ticket Medio (P99)")
    axes[0].set_xlabel("Importe neto pedido")

    axes[1].plot(aov_monthly["month"], aov_monthly["importe_neto_sin_iva_eur"], color="#1f4e79", linewidth=2.0, label="Ticket medio")
    axis_right = axes[1].twinx()
    axis_right.plot(discount_monthly["month"], discount_monthly["promo_share"], color="#d97706", linewidth=2.0, label="Share promo")
    axes[1].set_title("Ticket Medio y Share Promocional")
    axes[1].set_ylabel("Ticket medio")
    axis_right.set_ylabel("Share promo (%)")
    lines_left, labels_left = axes[1].get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axes[1].legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")
    savefig("eda", "eda_04_orders_and_discounts.png")


def eda_weekly_commercial_dynamics(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(weekly["semana_inicio"], weekly["ventas_netas"], color="#1f4e79", linewidth=2.0)
    axes[0].set_title("Ventas Netas Semanales")
    axes[0].set_ylabel("Ventas")

    axes[1].plot(weekly["semana_inicio"], weekly["pedidos_total"], color="#2f7d32", linewidth=2.0)
    axes[1].set_title("Pedidos Semanales")
    axes[1].set_ylabel("Pedidos")

    axes[2].plot(weekly["semana_inicio"], weekly["ticket_medio_neto"], color="#b45309", linewidth=2.0)
    axes[2].set_title("Ticket Medio Semanal")
    axes[2].set_ylabel("Ticket")
    axes[2].set_xlabel("Semana")
    savefig("eda", "eda_05_weekly_commercial_dynamics.png")


def eda_ticket_margin_overview(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(weekly["semana_inicio"], weekly["ticket_medio_neto"], color="#b45309", linewidth=2.0)
    axes[0].set_title("Ticket Medio Neto")
    axes[0].set_ylabel("EUR")

    axes[1].plot(weekly["semana_inicio"], weekly["margen_bruto_ponderado"], color="#7a1f5c", linewidth=2.0)
    axes[1].set_title("Margen Bruto Ponderado")
    axes[1].set_ylabel("Margen")
    axes[1].set_xlabel("Semana")
    savefig("eda", "eda_06_ticket_margin_overview.png")


def eda_week_year_heatmap(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    weekly["year"] = weekly["semana_inicio"].dt.year
    weekly["week_of_year"] = weekly["semana_inicio"].dt.isocalendar().week.astype(int)
    pivot = weekly.pivot(index="week_of_year", columns="year", values="ventas_netas")
    plt.figure(figsize=(10, 10))
    sns.heatmap(pivot, cmap=SEQUENTIAL_CMAP, linewidths=0.35, linecolor=PANEL_BG)
    plt.title("Heatmap de Ventas por Ano y Semana ISO")
    plt.xlabel("Ano")
    plt.ylabel("Semana ISO")
    savefig("eda", "eda_07_week_year_sales_heatmap.png")


def eda_media_mix(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset)
    media_cols = [col for col in weekly.columns if col.startswith("media_")]
    monthly = weekly.assign(month=weekly["semana_inicio"].dt.to_period("M").dt.to_timestamp())
    monthly = monthly.groupby("month", as_index=False)[media_cols].sum()
    melted = monthly.melt(id_vars="month", var_name="channel", value_name="investment")
    shares = melted.copy()
    shares["channel"] = shares["channel"].str.replace("media_", "", regex=False)
    shares["share"] = shares.groupby("month")["investment"].transform(
        lambda s: s / s.sum() if float(s.sum()) > 0.0 else 0.0
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    sns.lineplot(
        data=melted.assign(channel=melted["channel"].str.replace("media_", "", regex=False)),
        x="month",
        y="investment",
        hue="channel",
        ax=axes[0],
    )
    axes[0].set_title("Inversion Mensual por Canal")
    axes[0].legend(ncol=4, fontsize=9)

    share_pivot = shares.pivot(index="month", columns="channel", values="share").fillna(0.0)
    axes[1].stackplot(share_pivot.index, share_pivot.T.values, labels=share_pivot.columns)
    axes[1].set_title("Mix Mensual de Medios")
    axes[1].legend(ncol=4, fontsize=9, loc="upper left")
    savefig("feature_engineering", "eda_05_media_mix_timeseries.png")


def fe_feature_family_counts(dataset: pd.DataFrame) -> None:
    plot_df = feature_catalog(dataset)["family"].value_counts().rename_axis("family").reset_index(name="count")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="family", y="count", hue="family", palette="crest", legend=False)
    plt.title("Familias de Variables en el Dataset Causal")
    plt.xlabel("")
    plt.ylabel("Numero de columnas")
    plt.xticks(rotation=20)
    savefig("feature_engineering", "fe_01_feature_family_counts.png")


def fe_top_target_correlations(dataset: pd.DataFrame) -> None:
    plot_df = target_correlation_table(dataset).head(15).rename(columns={"correlation_with_target": "correlation"})
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, y="feature", x="correlation", hue="feature", palette=sns.blend_palette(SEQUENTIAL, n_colors=len(plot_df)), legend=False)
    plt.title("Top Correlaciones con Ventas Netas (Features Elegibles)")
    plt.xlabel("Correlacion")
    plt.ylabel("")
    style_axes(plt.gca(), grid_axis="x")
    savefig("feature_engineering", "fe_02_top_target_correlations.png")


def fe_target_vs_total_spend(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].scatter(weekly["budget_total_eur"], weekly["ventas_netas"], color=TEAL, alpha=0.7, s=36)
    axes[0].set_title("Ventas vs Presupuesto Total")
    axes[0].set_xlabel("Presupuesto total medios")
    axes[0].set_ylabel("Ventas netas")

    axes[1].plot(weekly["semana_inicio"], weekly["ventas_netas"], color=NAVY, linewidth=2.0, label="Ventas netas")
    axis_right = axes[1].twinx()
    axis_right.plot(weekly["semana_inicio"], weekly["budget_total_eur"], color=GREEN, linewidth=2.0, label="Presupuesto total")
    axes[1].set_title("Ventas y Presupuesto Total en el Tiempo")
    axes[1].set_ylabel("Ventas netas")
    axis_right.set_ylabel("Presupuesto")
    lines_left, labels_left = axes[1].get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axes[1].legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")
    style_axes(axes[0], grid_axis="both")
    style_axes(axes[1], grid_axis="y")
    axis_right.set_facecolor(PANEL_BG)
    axis_right.grid(False)
    axis_right.tick_params(colors=SLATE, labelsize=10.5)
    savefig("feature_engineering", "fe_03_target_vs_total_spend.png")


def eda_correlation_heatmap(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    candidate_columns = [
        "ventas_netas",
        "rebajas_flag",
        "black_friday_flag",
        "navidad_flag",
        "festivo_local_count",
        "payday_count",
        "temperatura_media_c_mean",
        "lluvia_indice_mean",
        "turismo_indice_mean",
        "media_paid_search",
        "media_social_paid",
        "media_radio_local",
        "media_exterior",
    ]
    selected = weekly[[column for column in candidate_columns if column in weekly.columns]]
    plt.figure(figsize=(12, 9))
    sns.heatmap(selected.corr(numeric_only=True), cmap="RdBu_r", center=0.0, annot=True, fmt=".2f")
    plt.title("Estructura de Correlacion entre Ventas, Controles y Medios")
    savefig("feature_engineering", "eda_08_correlation_heatmap.png")


def eda_event_uplift(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    event_specs = [
        ("rebajas_flag", "Rebajas"),
        ("black_friday_flag", "Black Friday"),
        ("navidad_flag", "Navidad"),
        ("semana_santa_flag", "Semana Santa"),
    ]
    rows = []
    for column, label in event_specs:
        grouped = weekly.groupby(column)["ventas_netas"].mean().reset_index()
        for _, row in grouped.iterrows():
            rows.append({"event": label, "flag": "Evento" if int(row[column]) == 1 else "Sin evento", "avg_sales": row["ventas_netas"]})
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="event", y="avg_sales", hue="flag", palette=["#9db4c0", "#d97706"])
    plt.title("Ventas Semanales Medias con y sin Eventos")
    plt.ylabel("Ventas semanales medias")
    savefig("feature_engineering", "eda_09_event_uplift.png")


def eda_sales_channel_mix(sales_lines: pd.DataFrame) -> None:
    sales_lines = sales_lines.copy()
    sales_lines["year"] = sales_lines["fecha_venta"].dt.year
    summary = sales_lines.groupby(["year", "canal_venta"], as_index=False)["venta_neta_sin_iva_eur"].sum()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary, x="year", y="venta_neta_sin_iva_eur", hue="canal_venta")
    plt.title("Mix de Ventas por Canal Comercial")
    plt.ylabel("Ventas netas")
    savefig("eda", "eda_10_sales_channel_mix.png")


def eda_client_profile(clients: pd.DataFrame) -> None:
    segment_mix = clients["segmento"].value_counts(normalize=True).mul(100).rename_axis("segmento").reset_index(name="share_pct")
    channel_mix = clients["canal_preferido"].value_counts(normalize=True).mul(100).rename_axis("canal_preferido").reset_index(name="share_pct")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(data=segment_mix, y="segmento", x="share_pct", hue="segmento", palette="crest", legend=False, ax=axes[0])
    axes[0].set_title("Mix de Segmentos de Cliente")
    axes[0].set_xlabel("Share (%)")
    axes[0].set_ylabel("")

    sns.barplot(data=channel_mix, y="canal_preferido", x="share_pct", hue="canal_preferido", palette="flare", legend=False, ax=axes[1])
    axes[1].set_title("Canal Preferido")
    axes[1].set_xlabel("Share (%)")
    axes[1].set_ylabel("")
    savefig("eda", "eda_11_client_profile.png")


def eda_indexed_sales_vs_media(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    base_sales = float(weekly.loc[weekly["ventas_netas"] > 0, "ventas_netas"].iloc[0])
    base_media = float(weekly.loc[weekly["budget_total_eur"] > 0, "budget_total_eur"].iloc[0])
    weekly["sales_index_100"] = weekly["ventas_netas"] / base_sales * 100.0
    weekly["media_index_100"] = weekly["budget_total_eur"] / base_media * 100.0

    plt.figure(figsize=(14, 6))
    plt.plot(weekly["semana_inicio"], weekly["sales_index_100"], color="#1f4e79", linewidth=2.2, label="Ventas")
    plt.plot(weekly["semana_inicio"], weekly["media_index_100"], color="#3a7d44", linewidth=2.2, label="Medios")
    plt.axhline(100.0, color="#6b7280", linestyle="--", linewidth=1.0)
    plt.title("Ventas y Medios Indexados (Base 100)")
    plt.xlabel("Semana")
    plt.ylabel("Indice")
    plt.legend()
    savefig("eda", "eda_12_indexed_sales_vs_media.png")


def eda_media_channel_correlation(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    media_cols = [col for col in weekly.columns if col.startswith("media_")]
    corr = weekly[media_cols].corr().rename(
        index=lambda s: s.replace("media_", ""),
        columns=lambda s: s.replace("media_", ""),
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0.0, annot=True, fmt=".2f")
    plt.title("Correlacion entre Canales de Medios")
    savefig("eda", "eda_13_media_channel_correlation.png")


def eda_sales_media_lag_heatmap(dataset: pd.DataFrame) -> None:
    weekly = _weekly_dataset(dataset, complete_only=True)
    media_cols = [col for col in weekly.columns if col.startswith("media_")]

    rows = []
    for channel in media_cols:
        for lag in range(0, 9):
            rows.append(
                {
                    "channel": channel.replace("media_", ""),
                    "lag_weeks": lag,
                    "correlation": float(weekly["ventas_netas"].corr(weekly[channel].shift(lag))),
                }
            )
    lag_df = pd.DataFrame(rows)
    pivot = lag_df.pivot(index="channel", columns="lag_weeks", values="correlation")
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(pivot, cmap=DIVERGING, center=0.0, annot=True, fmt=".2f", linewidths=0.35, linecolor=PANEL_BG)
    plt.title("Correlacion Ventas vs Medios por Lag")
    plt.xlabel("Lag de medios (semanas)")
    plt.ylabel("Canal")
    savefig("eda", "eda_14_sales_media_lag_heatmap.png")


def eda_product_cardinality(sales_lines: pd.DataFrame) -> None:
    plot_df = pd.DataFrame(
        {
            "field": ["sku", "categoria", "subcategoria", "articulo"],
            "unique_values": [
                int(sales_lines["sku"].nunique()),
                int(sales_lines["categoria"].nunique()),
                int(sales_lines["subcategoria"].nunique()),
                int(sales_lines["articulo"].nunique()),
            ],
        }
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="field", y="unique_values", hue="field", palette="mako", legend=False)
    plt.title("Cardinalidad del Detalle de Producto")
    plt.ylabel("Valores unicos")
    savefig("preprocessing", "eda_12_product_cardinality.png")


def generate_eda_figures(
    dataset: pd.DataFrame,
    sales_lines: pd.DataFrame,
    orders: pd.DataFrame,
    clients: pd.DataFrame,
) -> None:
    eda_weekly_sales_overview(dataset)
    preprocessing_panel_coverage_heatmap()
    preprocessing_pipeline_row_counts()
    preprocessing_variable_quality_heatmap()
    preprocessing_temporal_consistency(dataset)
    eda_monthly_seasonality(dataset)
    eda_sales_distribution(dataset)
    eda_yearly_sales_overlay(dataset)
    eda_orders_and_discounts(sales_lines, orders)
    eda_weekly_commercial_dynamics(dataset)
    eda_ticket_margin_overview(dataset)
    eda_week_year_heatmap(dataset)
    eda_media_mix(dataset)
    fe_feature_family_counts(dataset)
    fe_top_target_correlations(dataset)
    fe_target_vs_total_spend(dataset)
    eda_correlation_heatmap(dataset)
    eda_event_uplift(dataset)
    eda_sales_channel_mix(sales_lines)
    eda_client_profile(clients)
    eda_indexed_sales_vs_media(dataset)
    eda_media_channel_correlation(dataset)
    eda_sales_media_lag_heatmap(dataset)
    eda_product_cardinality(sales_lines)
