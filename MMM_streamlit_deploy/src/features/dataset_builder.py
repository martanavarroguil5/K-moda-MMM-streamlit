from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.data.weekly_aggregations import (
    GLOBAL_PANEL_LABEL,
    build_weekly_calendar,
    build_weekly_media,
    build_weekly_orders,
    build_weekly_sales,
    build_weekly_traffic,
    ensure_dirs,
)


DATASET_SCOPE = "global_weekly_series"
DIAGNOSTIC_ONLY_COLUMNS = [
    "days_in_week",
    "week_complete_flag",
    "margen_bruto_ponderado",
    "promo_share_lineas",
    "visitas_tienda_sum",
    "pedidos_tienda_sum",
    "pedidos_click_collect_sum",
    "sesiones_web_sum",
    "pedidos_online_sum",
    "tasa_conversion_tienda_mean",
    "tasa_conversion_web_mean",
]
DOWNSTREAM_ONLY_COLUMNS = [
    "pedidos_total",
    "ticket_medio_neto",
]
KEY_COLUMNS = ["semana_inicio", "ciudad"]
ZERO_FILL_PREFIXES = ("media_",)
ZERO_FILL_COLUMNS = {
    "ventas_netas",
    "promo_share_lineas",
    "margen_bruto_ponderado",
    "pedidos_total",
    "ticket_medio_neto",
    "days_in_week",
    "anio",
    "mes_modal",
    "rebajas_flag",
    "black_friday_flag",
    "navidad_flag",
    "semana_santa_flag",
    "vacaciones_escolares_flag",
    "festivo_local_count",
    "payday_count",
    "incidencia_ecommerce_flag",
    "week_complete_flag",
    "visitas_tienda_sum",
    "pedidos_tienda_sum",
    "pedidos_click_collect_sum",
    "sesiones_web_sum",
    "pedidos_online_sum",
    "tasa_conversion_tienda_mean",
    "tasa_conversion_web_mean",
}


def _duplicate_key_count(dataset: pd.DataFrame, keys: list[str] | None = None) -> int:
    join_keys = keys or KEY_COLUMNS
    return int(dataset.duplicated(join_keys).sum())


def _count_infinite_cells(dataset: pd.DataFrame) -> int:
    numeric = dataset.select_dtypes(include="number")
    if numeric.empty:
        return 0
    return int(np.isinf(numeric.to_numpy()).sum())


def _panel_from_calendar(weekly_calendar: pd.DataFrame) -> pd.DataFrame:
    return weekly_calendar[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)


def _expected_week_index(dataset: pd.DataFrame) -> pd.DatetimeIndex:
    observed_weeks = pd.to_datetime(dataset["semana_inicio"]).drop_duplicates().sort_values()
    if observed_weeks.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(observed_weeks.min(), observed_weeks.max(), freq="W-MON")


def _missing_week_dates(dataset: pd.DataFrame) -> list[str]:
    observed_weeks = pd.DatetimeIndex(pd.to_datetime(dataset["semana_inicio"]).drop_duplicates().sort_values())
    expected_weeks = _expected_week_index(dataset)
    return expected_weeks.difference(observed_weeks).strftime("%Y-%m-%d").tolist()


def _fill_numeric_nulls(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    zero_fill_cols = [
        column
        for column in out.columns
        if column in ZERO_FILL_COLUMNS or any(column.startswith(prefix) for prefix in ZERO_FILL_PREFIXES)
    ]
    if zero_fill_cols:
        out[zero_fill_cols] = out[zero_fill_cols].fillna(0.0)
    return out


def _add_temporal_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True).copy()
    weeks = dataset["semana_inicio"].drop_duplicates().sort_values()
    week_index = {week: idx for idx, week in enumerate(weeks)}
    dataset["trend_index"] = dataset["semana_inicio"].map(week_index).astype(int)
    dataset["week_of_year"] = dataset["semana_inicio"].dt.isocalendar().week.astype(int)
    dataset["year"] = dataset["semana_inicio"].dt.year.astype(int)
    dataset["quarter"] = (((dataset["semana_inicio"].dt.month - 1) // 3) + 1).astype(int)
    dataset["week_sin"] = np.sin(2 * np.pi * dataset["week_of_year"] / 52.0)
    dataset["week_cos"] = np.cos(2 * np.pi * dataset["week_of_year"] / 52.0)
    return dataset


def _add_media_mix_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    media_cols = sorted([column for column in dataset.columns if column.startswith("media_")])
    dataset["budget_total_eur"] = dataset[media_cols].sum(axis=1) if media_cols else 0.0
    for media_col in media_cols:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        dataset[share_col] = np.where(
            dataset["budget_total_eur"] > 0.0,
            dataset[media_col] / dataset["budget_total_eur"] * 100.0,
            0.0,
        )
    return dataset


def _build_diagnostic_dataset(
    weekly_sales: pd.DataFrame,
    weekly_orders: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    weekly_traffic: pd.DataFrame,
    weekly_media: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    panel = _panel_from_calendar(weekly_calendar)
    dataset = (
        panel.merge(weekly_sales, on=KEY_COLUMNS, how="left")
        .merge(weekly_orders, on=KEY_COLUMNS, how="left")
        .merge(weekly_calendar, on=KEY_COLUMNS, how="left")
        .merge(weekly_traffic, on=KEY_COLUMNS, how="left")
        .merge(weekly_media, on=KEY_COLUMNS, how="left")
    )
    merge_audit = {
        "merge_missing_cells_before_fill": int(dataset.isna().sum().sum()),
        "merge_infinite_cells_before_fill": _count_infinite_cells(dataset),
    }
    dataset = _fill_numeric_nulls(dataset)
    dataset = _add_temporal_features(dataset)
    dataset = _add_media_mix_features(dataset)
    return dataset, merge_audit


def _build_source_grain_audit(
    weekly_sales: pd.DataFrame,
    weekly_orders: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    weekly_traffic: pd.DataFrame,
    weekly_media: pd.DataFrame,
) -> pd.DataFrame:
    source_specs = [
        ("weekly_sales", weekly_sales, "day -> week", "sum of venta_neta_sin_iva_eur aggregated to the global weekly target"),
        ("weekly_orders", weekly_orders, "day -> week", "weekly count of orders and global average order value"),
        ("weekly_calendar", weekly_calendar, "day -> week", "weekly exogenous controls and completeness tags on the global timeline"),
        ("weekly_traffic", weekly_traffic, "day -> week", "weekly operational diagnostics aggregated globally from daily traffic"),
        ("weekly_media", weekly_media, "week-channel -> week", "channel pivot over global weekly media investment"),
    ]
    rows = []
    for source_name, dataset, raw_grain, construction_rule in source_specs:
        rows.append(
            {
                "source": source_name,
                "raw_or_pre_join_grain": raw_grain,
                "join_keys": "semana_inicio + ciudad(Global)",
                "construction_rule": construction_rule,
                "rows": int(len(dataset)),
                "unique_panel_keys": int(dataset[KEY_COLUMNS].drop_duplicates().shape[0]),
                "duplicate_panel_keys": _duplicate_key_count(dataset),
                "series_count": int(dataset["ciudad"].nunique()),
                "scope": DATASET_SCOPE,
                "min_week": pd.to_datetime(dataset["semana_inicio"]).min().date().isoformat(),
                "max_week": pd.to_datetime(dataset["semana_inicio"]).max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _build_join_audit(
    weekly_sales: pd.DataFrame,
    weekly_orders: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    weekly_traffic: pd.DataFrame,
    weekly_media: pd.DataFrame,
) -> pd.DataFrame:
    panel = _panel_from_calendar(weekly_calendar)
    rows = [
        {
            "step": 0,
            "source": "panel_base_from_calendar",
            "join_type": "base panel",
            "join_keys": "semana_inicio + ciudad(Global)",
            "rows_before": int(len(panel)),
            "rows_after": int(len(panel)),
            "row_delta": 0,
            "expected_panel_rows": int(len(panel)),
            "source_rows": int(len(weekly_calendar)),
            "source_duplicate_keys": _duplicate_key_count(weekly_calendar),
            "duplicate_keys_after_merge": _duplicate_key_count(panel),
            "panel_keys_without_source_match": 0,
            "row_count_preserved": True,
            "comment": "Panel ancla definido por el calendario semanal global para no truncar semanas con ventas pero sin medios.",
        }
    ]
    current = panel.copy()
    for step, (source_name, source_df) in enumerate(
        [
            ("weekly_sales", weekly_sales),
            ("weekly_orders", weekly_orders),
            ("weekly_calendar", weekly_calendar),
            ("weekly_traffic", weekly_traffic),
            ("weekly_media", weekly_media),
        ],
        start=1,
    ):
        merged = current.merge(source_df, on=KEY_COLUMNS, how="left")
        added_columns = [column for column in source_df.columns if column not in KEY_COLUMNS]
        null_source_rows = int(merged[added_columns].isna().all(axis=1).sum()) if added_columns else 0
        rows.append(
            {
                "step": step,
                "source": source_name,
                "join_type": "left",
                "join_keys": "semana_inicio + ciudad(Global)",
                "rows_before": int(len(current)),
                "rows_after": int(len(merged)),
                "row_delta": int(len(merged) - len(current)),
                "expected_panel_rows": int(len(panel)),
                "source_rows": int(len(source_df)),
                "source_duplicate_keys": _duplicate_key_count(source_df),
                "duplicate_keys_after_merge": _duplicate_key_count(merged),
                "panel_keys_without_source_match": null_source_rows,
                "row_count_preserved": bool(len(merged) == len(panel)),
                "comment": (
                    "Merge preserva la serie semanal global; los huecos quedan visibles como NA antes del fill numerico."
                    if bool(len(merged) == len(panel))
                    else "Revision necesaria: el merge altero el numero esperado de filas."
                ),
            }
        )
        current = merged
    return pd.DataFrame(rows)


def _build_missing_zero_rules() -> pd.DataFrame:
    rows = [
        {
            "variable_family": "ventas_netas",
            "source": "sales_lines",
            "missing_meaning": "No raw sales row for that day before calendar closure; after weekly rollup no NA should remain.",
            "zero_meaning": "Observed week with zero global sales after closing the daily calendar.",
            "preprocessing_rule": "Close the daily calendar first, then aggregate to weekly global sales and fill absent numeric values with 0.0 only after the joins.",
            "modeling_note": "Zero-sales weeks remain valid observations in a single-series MMM.",
        },
        {
            "variable_family": "media_*",
            "source": "investment_media",
            "missing_meaning": "No week-channel record before pivot; after pivot, missing becomes 0.0 because the source is spend-based.",
            "zero_meaning": "Observed week with no investment in that channel.",
            "preprocessing_rule": "Aggregate spend globally by week and channel, then pivot to wide format preserving explicit zero spend.",
            "modeling_note": "Media columns are valid causal candidates once aligned on the weekly time grid.",
        },
        {
            "variable_family": "calendar_*",
            "source": "calendar",
            "missing_meaning": "Should not happen because calendar is the authoritative weekly time grid.",
            "zero_meaning": "Absence of the event or count in that week.",
            "preprocessing_rule": "Aggregate daily calendar controls to the global week and retain completeness flags.",
            "modeling_note": "Calendar variables remain exogenous controls, not downstream outcomes.",
        },
        {
            "variable_family": "traffic_*",
            "source": "traffic",
            "missing_meaning": "Measurement gap before merge; after alignment, numeric gaps are filled with 0.0 to preserve the weekly index.",
            "zero_meaning": "Observed week with no operational traffic signal.",
            "preprocessing_rule": "Aggregate operational diagnostics globally and derive conversion rates from counts to avoid averaging ratio noise.",
            "modeling_note": "Traffic stays diagnostic by default and is excluded from the causal dataset.",
        },
        {
            "variable_family": "pedidos_total / ticket_medio_neto",
            "source": "orders",
            "missing_meaning": "No order row for that week before merge.",
            "zero_meaning": "No orders observed after merge fill; ticket mean is set to 0.0 only for diagnostic continuity.",
            "preprocessing_rule": "Use orders for reconciliation and descriptive diagnostics, not to define the sales target.",
            "modeling_note": "Treat as downstream business variables, not causal MMM drivers.",
        },
        {
            "variable_family": "derived ratios and shares",
            "source": "derived",
            "missing_meaning": "Appears only when a denominator is zero or a merge produced NA before numeric sanitation.",
            "zero_meaning": "No meaningful signal given the observed denominator or an explicit no-spend week.",
            "preprocessing_rule": "Guard every denominator, replace inf with NA, and only then fill numeric gaps with 0.0.",
            "modeling_note": "This avoids leaking undefined arithmetic into the MMM feature matrix.",
        },
        {
            "variable_family": "days_in_week / week_complete_flag",
            "source": "calendar",
            "missing_meaning": "Should never be missing because completeness is derived from the calendar itself.",
            "zero_meaning": "Observed week is incomplete relative to a Monday-Sunday definition.",
            "preprocessing_rule": "Tag incomplete weeks explicitly and keep them visible in every audit output.",
            "modeling_note": "Use the flag to distinguish truncation from genuine demand shocks.",
        },
    ]
    return pd.DataFrame(rows)


def _build_dataset_summary(dataset: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    numeric = dataset.select_dtypes(include="number")
    total_cells = int(dataset.shape[0] * dataset.shape[1])
    total_missing = int(dataset.isna().sum().sum())
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "rows": int(len(dataset)),
                "columns": int(len(dataset.columns)),
                "series_count": int(dataset["ciudad"].nunique()),
                "weeks": int(dataset["semana_inicio"].nunique()),
                "expected_panel_rows": int(dataset["ciudad"].nunique() * dataset["semana_inicio"].nunique()),
                "duplicate_panel_keys": _duplicate_key_count(dataset),
                "total_missing_cells": total_missing,
                "total_infinite_cells": _count_infinite_cells(dataset),
                "total_missing_pct": round(float(total_missing / total_cells * 100.0), 4) if total_cells else 0.0,
                "numeric_zero_pct": round(float((numeric == 0).mean().mean() * 100.0), 4) if not numeric.empty else 0.0,
                "scope": DATASET_SCOPE,
                "min_week": pd.to_datetime(dataset["semana_inicio"]).min().date().isoformat(),
                "max_week": pd.to_datetime(dataset["semana_inicio"]).max().date().isoformat(),
                "week_definition": "Monday-start week (Monday-Sunday), labelled by semana_inicio",
            }
        ]
    )


def _build_variable_quality_audit(dataset: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for column in dataset.columns:
        series = dataset[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        rows.append(
            {
                "dataset": dataset_name,
                "variable": column,
                "dtype": str(series.dtype),
                "missing_pct": round(float(series.isna().mean() * 100.0), 4),
                "zero_pct": round(float((series == 0).mean() * 100.0), 4) if is_numeric else None,
                "inf_pct": (
                    round(float(np.isinf(pd.to_numeric(series, errors="coerce")).mean() * 100.0), 4)
                    if is_numeric
                    else None
                ),
                "unique_values": int(series.nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def _build_weekly_coverage_audit(diagnostic_dataset: pd.DataFrame) -> pd.DataFrame:
    numeric = diagnostic_dataset.select_dtypes(include="number").copy()
    audit = diagnostic_dataset[["semana_inicio", "ciudad", "ventas_netas", "budget_total_eur", "pedidos_total", "week_complete_flag"]].copy()
    audit["traffic_available"] = (
        diagnostic_dataset["visitas_tienda_sum"] + diagnostic_dataset["sesiones_web_sum"]
    ) > 0.0
    audit["target_available"] = audit["ventas_netas"] > 0.0
    audit["media_available"] = audit["budget_total_eur"] > 0.0
    audit["orders_available"] = audit["pedidos_total"] > 0.0
    audit["numeric_zero_pct"] = (numeric == 0.0).mean(axis=1).mul(100.0).round(2)
    audit["missing_numeric_cells"] = diagnostic_dataset.select_dtypes(include="number").isna().sum(axis=1)
    return audit.sort_values("semana_inicio").reset_index(drop=True)


def _build_target_rollup_audit(
    weekly_sales: pd.DataFrame,
    diagnostic_dataset: pd.DataFrame,
    causal_dataset: pd.DataFrame,
) -> pd.DataFrame:
    raw_sales = pd.read_csv(CONFIG.sales_lines_file, usecols=["venta_neta_sin_iva_eur"])
    rows = [
        {
            "stage": "raw_sales_lines",
            "grain": "sale line",
            "rows": int(len(raw_sales)),
            "target_total_eur": round(float(raw_sales["venta_neta_sin_iva_eur"].sum()), 2),
            "note": "Original target source before any aggregation.",
        },
        {
            "stage": "weekly_sales_global",
            "grain": "semana_inicio",
            "rows": int(len(weekly_sales)),
            "target_total_eur": round(float(weekly_sales["ventas_netas"].sum()), 2),
            "note": "Weekly target after daily calendar closure and global weekly aggregation.",
        },
        {
            "stage": "diagnostic_dataset",
            "grain": "semana_inicio",
            "rows": int(len(diagnostic_dataset)),
            "target_total_eur": round(float(diagnostic_dataset["ventas_netas"].sum()), 2),
            "note": "Diagnostic weekly panel after joins and derived features.",
        },
        {
            "stage": "causal_dataset_complete_weeks",
            "grain": "semana_inicio",
            "rows": int(len(causal_dataset)),
            "target_total_eur": round(float(causal_dataset["ventas_netas"].sum()), 2),
            "note": "Model-ready causal panel after excluding incomplete edge weeks and downstream business proxies.",
        },
    ]
    audit = pd.DataFrame(rows)
    baseline = float(audit.loc[audit["stage"] == "raw_sales_lines", "target_total_eur"].iloc[0])
    audit["delta_vs_raw_eur"] = audit["target_total_eur"] - baseline
    audit["total_preserved_vs_raw"] = audit["delta_vs_raw_eur"].round(8) == 0.0
    return audit


def _build_temporal_consistency_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    expected_weeks = _expected_week_index(dataset)
    missing_weeks = _missing_week_dates(dataset)
    observed_weeks = pd.to_datetime(dataset["semana_inicio"]).drop_duplicates().sort_values()
    max_gap_days = 0
    if len(observed_weeks) > 1:
        max_gap_days = int(observed_weeks.diff().dt.days.dropna().max())
    return pd.DataFrame(
        [
            {
                "scope": DATASET_SCOPE,
                "series_label": GLOBAL_PANEL_LABEL,
                "min_week": observed_weeks.min().date().isoformat(),
                "max_week": observed_weeks.max().date().isoformat(),
                "observed_weeks": int(len(observed_weeks)),
                "expected_weeks": int(len(expected_weeks)),
                "missing_weeks_count": int(len(missing_weeks)),
                "missing_weeks": "; ".join(missing_weeks),
                "incomplete_weeks_count": int((dataset["week_complete_flag"] == 0).sum()),
                "regular_weekly_frequency": bool(len(missing_weeks) == 0),
                "max_gap_days": max_gap_days,
            }
        ]
    )


def _build_zero_sales_audit(diagnostic_dataset: pd.DataFrame) -> pd.DataFrame:
    zero_sales_rows = int((diagnostic_dataset["ventas_netas"] == 0.0).sum())
    positive_sales_rows = int((diagnostic_dataset["ventas_netas"] > 0.0).sum())
    return pd.DataFrame(
        [
            {
                "scope": DATASET_SCOPE,
                "series_label": GLOBAL_PANEL_LABEL,
                "rows": int(len(diagnostic_dataset)),
                "positive_sales_rows": positive_sales_rows,
                "zero_sales_rows": zero_sales_rows,
                "zero_share_pct": round(float(zero_sales_rows / len(diagnostic_dataset) * 100.0), 2),
                "sales_total_eur": round(float(diagnostic_dataset["ventas_netas"].sum()), 2),
            }
        ]
    )


def _build_causal_dataset(diagnostic_dataset: pd.DataFrame) -> pd.DataFrame:
    causal = diagnostic_dataset.loc[diagnostic_dataset["week_complete_flag"] == 1].copy()
    causal = causal.drop(columns=DIAGNOSTIC_ONLY_COLUMNS + DOWNSTREAM_ONLY_COLUMNS)
    return causal.sort_values(["semana_inicio", "ciudad"]).reset_index(drop=True)


def build_model_dataset() -> tuple[pd.DataFrame, dict]:
    ensure_dirs()
    weekly_sales, checks = build_weekly_sales()
    weekly_orders = build_weekly_orders()
    weekly_calendar = build_weekly_calendar()
    weekly_traffic = build_weekly_traffic()
    weekly_media = build_weekly_media()

    diagnostic_dataset, merge_audit = _build_diagnostic_dataset(
        weekly_sales=weekly_sales,
        weekly_orders=weekly_orders,
        weekly_calendar=weekly_calendar,
        weekly_traffic=weekly_traffic,
        weekly_media=weekly_media,
    )
    causal_dataset = _build_causal_dataset(diagnostic_dataset)
    source_grain_audit = _build_source_grain_audit(
        weekly_sales=weekly_sales,
        weekly_orders=weekly_orders,
        weekly_calendar=weekly_calendar,
        weekly_traffic=weekly_traffic,
        weekly_media=weekly_media,
    )
    join_audit = _build_join_audit(
        weekly_sales=weekly_sales,
        weekly_orders=weekly_orders,
        weekly_calendar=weekly_calendar,
        weekly_traffic=weekly_traffic,
        weekly_media=weekly_media,
    )
    missing_zero_rules = _build_missing_zero_rules()
    dataset_summary = pd.concat(
        [
            _build_dataset_summary(diagnostic_dataset, "diagnostic"),
            _build_dataset_summary(causal_dataset, "causal"),
        ],
        ignore_index=True,
    )
    variable_quality = pd.concat(
        [
            _build_variable_quality_audit(diagnostic_dataset, "diagnostic"),
            _build_variable_quality_audit(causal_dataset, "causal"),
        ],
        ignore_index=True,
    )
    panel_coverage = _build_weekly_coverage_audit(diagnostic_dataset)
    target_rollup_audit = _build_target_rollup_audit(
        weekly_sales=weekly_sales,
        diagnostic_dataset=diagnostic_dataset,
        causal_dataset=causal_dataset,
    )
    temporal_consistency = _build_temporal_consistency_audit(diagnostic_dataset)
    zero_sales_audit = _build_zero_sales_audit(diagnostic_dataset)

    media_cols = [column for column in causal_dataset.columns if column.startswith("media_")]
    share_cols = [column for column in causal_dataset.columns if column.startswith("budget_share_pct_")]
    positive_budget_mask = causal_dataset["budget_total_eur"] > 0.0
    missing_weeks = _missing_week_dates(diagnostic_dataset)
    checks.update(merge_audit)
    checks.update(
        {
            "dataset_scope": DATASET_SCOPE,
            "dataset_grain": "semana_inicio",
            "global_series_label": GLOBAL_PANEL_LABEL,
            "diagnostic_dataset_rows": int(len(diagnostic_dataset)),
            "diagnostic_dataset_cities": int(diagnostic_dataset["ciudad"].nunique()),
            "causal_dataset_rows": int(len(causal_dataset)),
            "causal_dataset_cities": int(causal_dataset["ciudad"].nunique()),
            "dataset_rows": int(len(causal_dataset)),
            "dataset_cities": int(causal_dataset["ciudad"].nunique()),
            "dataset_weeks": int(causal_dataset["semana_inicio"].nunique()),
            "weekly_panel_complete": bool(
                len(causal_dataset) == causal_dataset["ciudad"].nunique() * causal_dataset["semana_inicio"].nunique()
            ),
            "complete_weeks_in_diagnostic": int(diagnostic_dataset["week_complete_flag"].sum()),
            "incomplete_weeks_in_diagnostic": int((diagnostic_dataset["week_complete_flag"] == 0).sum()),
            "incomplete_week_dates": sorted(
                diagnostic_dataset.loc[diagnostic_dataset["week_complete_flag"] == 0, "semana_inicio"]
                .drop_duplicates()
                .dt.date.astype(str)
                .tolist()
            ),
            "causal_complete_weeks_only": bool((causal_dataset["semana_inicio"].isin(
                diagnostic_dataset.loc[diagnostic_dataset["week_complete_flag"] == 1, "semana_inicio"]
            )).all()),
            "causal_target_total": round(float(causal_dataset["ventas_netas"].sum()), 2),
            "diagnostic_target_total": round(float(diagnostic_dataset["ventas_netas"].sum()), 2),
            "diagnostic_target_total_preserved_vs_raw": bool(
                round(float(diagnostic_dataset["ventas_netas"].sum() - checks["sales_total_lineas"]), 8) == 0.0
            ),
            "missing_weeks_count": int(len(missing_weeks)),
            "missing_week_dates": missing_weeks,
            "structural_zero_rows": 0,
            "active_city_zero_rows": int((diagnostic_dataset["ventas_netas"] == 0.0).sum()),
            "excluded_cities_from_causal": [],
            "ticket_medio_neto_mean": round(
                float(diagnostic_dataset["ticket_medio_neto"].replace(0.0, np.nan).mean()),
                2,
            ),
            "media_total": round(float(causal_dataset[media_cols].sum().sum()), 2),
            "sales_to_media_ratio": round(
                float(causal_dataset["ventas_netas"].sum() / causal_dataset[media_cols].sum().sum()),
                4,
            ),
            "budget_zero_rows": int((causal_dataset["budget_total_eur"] == 0.0).sum()),
            "budget_share_pct_sum_mean": round(
                float(causal_dataset.loc[positive_budget_mask, share_cols].sum(axis=1).mean()),
                4,
            )
            if share_cols and positive_budget_mask.any()
            else 0.0,
            "diagnostic_dataset_columns": int(len(diagnostic_dataset.columns)),
            "causal_dataset_columns": int(len(causal_dataset.columns)),
            "diagnostic_only_columns": DIAGNOSTIC_ONLY_COLUMNS,
            "downstream_only_columns": DOWNSTREAM_ONLY_COLUMNS,
            "join_row_count_preserved": bool(join_audit["row_count_preserved"].all()),
            "max_duplicate_keys_after_join": int(join_audit["duplicate_keys_after_merge"].max()),
            "calendar_panel_anchor_rows": int(_panel_from_calendar(weekly_calendar).shape[0]),
            "media_panel_anchor_rows": int(_panel_from_calendar(weekly_calendar).shape[0]),
            "remaining_missing_cells": int(diagnostic_dataset.isna().sum().sum()),
            "remaining_infinite_cells": _count_infinite_cells(diagnostic_dataset),
        }
    )

    weekly_sales.to_parquet(CONFIG.weekly_sales_file, index=False)
    weekly_orders.to_parquet(CONFIG.weekly_orders_file, index=False)
    weekly_calendar.to_parquet(CONFIG.weekly_calendar_file, index=False)
    weekly_traffic.to_parquet(CONFIG.weekly_traffic_file, index=False)
    weekly_media.to_parquet(CONFIG.weekly_media_file, index=False)
    diagnostic_dataset.to_parquet(CONFIG.diagnostic_dataset_file, index=False)
    causal_dataset.to_parquet(CONFIG.model_dataset_file, index=False)
    source_grain_audit.to_csv(CONFIG.reports_tables_dir / "preprocessing_source_grain_audit.csv", index=False)
    join_audit.to_csv(CONFIG.reports_tables_dir / "preprocessing_join_audit.csv", index=False)
    missing_zero_rules.to_csv(CONFIG.reports_tables_dir / "preprocessing_missing_zero_rules.csv", index=False)
    dataset_summary.to_csv(CONFIG.reports_tables_dir / "preprocessing_dataset_summary.csv", index=False)
    variable_quality.to_csv(CONFIG.reports_tables_dir / "preprocessing_variable_quality.csv", index=False)
    panel_coverage.to_csv(CONFIG.reports_tables_dir / "preprocessing_panel_coverage.csv", index=False)
    target_rollup_audit.to_csv(CONFIG.reports_tables_dir / "preprocessing_target_rollup_audit.csv", index=False)
    temporal_consistency.to_csv(CONFIG.reports_tables_dir / "preprocessing_temporal_consistency.csv", index=False)
    zero_sales_audit.to_csv(CONFIG.reports_tables_dir / "preprocessing_zero_sales_audit.csv", index=False)
    CONFIG.data_checks_file.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return causal_dataset, checks
