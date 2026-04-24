from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.data.weekly_aggregations import (
    ensure_dirs,
    build_weekly_calendar,
    build_weekly_media,
    build_weekly_orders,
    build_weekly_sales,
    build_weekly_traffic,
)
from src.features.dataset_builder import DIAGNOSTIC_ONLY_COLUMNS, DOWNSTREAM_ONLY_COLUMNS


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


def _duplicate_key_count(dataset: pd.DataFrame) -> int:
    return int(dataset.duplicated(KEY_COLUMNS).sum())


def _count_infinite_cells(dataset: pd.DataFrame) -> int:
    numeric = dataset.select_dtypes(include="number")
    if numeric.empty:
        return 0
    return int(np.isinf(numeric.to_numpy()).sum())


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
    dataset = dataset.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
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
    panel = weekly_calendar[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)
    dataset = (
        panel.merge(weekly_sales, on=KEY_COLUMNS, how="left")
        .merge(weekly_orders, on=KEY_COLUMNS, how="left")
        .merge(weekly_calendar, on=KEY_COLUMNS, how="left")
        .merge(weekly_traffic, on=KEY_COLUMNS, how="left")
        .merge(weekly_media, on=KEY_COLUMNS, how="left")
    )
    merge_audit = {
        "geo_merge_missing_cells_before_fill": int(dataset.isna().sum().sum()),
        "geo_merge_infinite_cells_before_fill": _count_infinite_cells(dataset),
    }
    dataset = _fill_numeric_nulls(dataset)
    dataset = _add_temporal_features(dataset)
    dataset = _add_media_mix_features(dataset)
    return dataset, merge_audit


def _active_cities(weekly_sales: pd.DataFrame) -> list[str]:
    totals = weekly_sales.groupby("ciudad", as_index=False)["ventas_netas"].sum()
    return sorted(totals.loc[totals["ventas_netas"] > 0.0, "ciudad"].astype(str).tolist())


def _build_causal_dataset(diagnostic_dataset: pd.DataFrame, active_cities: list[str]) -> pd.DataFrame:
    causal = diagnostic_dataset.loc[
        (diagnostic_dataset["week_complete_flag"] == 1) & diagnostic_dataset["ciudad"].astype(str).isin(active_cities)
    ].copy()
    causal = causal.drop(columns=DIAGNOSTIC_ONLY_COLUMNS + DOWNSTREAM_ONLY_COLUMNS)
    return causal.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _dataset_summary(dataset: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    numeric = dataset.select_dtypes(include="number")
    total_cells = int(dataset.shape[0] * dataset.shape[1])
    total_missing = int(dataset.isna().sum().sum())
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "rows": int(len(dataset)),
                "columns": int(len(dataset.columns)),
                "cities": int(dataset["ciudad"].nunique()),
                "weeks": int(dataset["semana_inicio"].nunique()),
                "expected_panel_rows": int(dataset["ciudad"].nunique() * dataset["semana_inicio"].nunique()),
                "duplicate_panel_keys": _duplicate_key_count(dataset),
                "total_missing_cells": total_missing,
                "total_infinite_cells": _count_infinite_cells(dataset),
                "total_missing_pct": round(float(total_missing / total_cells * 100.0), 4) if total_cells else 0.0,
                "numeric_zero_pct": round(float((numeric == 0).mean().mean() * 100.0), 4) if not numeric.empty else 0.0,
                "min_week": pd.to_datetime(dataset["semana_inicio"]).min().date().isoformat(),
                "max_week": pd.to_datetime(dataset["semana_inicio"]).max().date().isoformat(),
            }
        ]
    )


def build_geo_model_dataset() -> tuple[pd.DataFrame, dict]:
    ensure_dirs()
    weekly_sales, sales_checks = build_weekly_sales(keep_city=True)
    weekly_orders = build_weekly_orders(keep_city=True)
    weekly_calendar = build_weekly_calendar(keep_city=True)
    weekly_traffic = build_weekly_traffic(keep_city=True)
    weekly_media = build_weekly_media(keep_city=True)

    active_cities = _active_cities(weekly_sales)
    diagnostic_dataset, merge_audit = _build_diagnostic_dataset(
        weekly_sales=weekly_sales,
        weekly_orders=weekly_orders,
        weekly_calendar=weekly_calendar,
        weekly_traffic=weekly_traffic,
        weekly_media=weekly_media,
    )
    causal_dataset = _build_causal_dataset(diagnostic_dataset, active_cities=active_cities)

    share_cols = [column for column in causal_dataset.columns if column.startswith("budget_share_pct_")]
    positive_budget_mask = causal_dataset["budget_total_eur"] > 0.0
    checks = {
        **sales_checks,
        **merge_audit,
        "dataset_scope": "geo_weekly_panel",
        "dataset_grain": "semana_inicio + ciudad",
        "geo_diagnostic_dataset_rows": int(len(diagnostic_dataset)),
        "geo_causal_dataset_rows": int(len(causal_dataset)),
        "geo_diagnostic_dataset_cities": int(diagnostic_dataset["ciudad"].nunique()),
        "geo_causal_dataset_cities": int(causal_dataset["ciudad"].nunique()),
        "geo_active_cities": active_cities,
        "geo_complete_weeks_in_diagnostic": int(diagnostic_dataset["week_complete_flag"].sum()),
        "geo_incomplete_weeks_in_diagnostic": int((diagnostic_dataset["week_complete_flag"] == 0).sum()),
        "geo_budget_share_pct_sum_mean": round(
            float(causal_dataset.loc[positive_budget_mask, share_cols].sum(axis=1).mean()),
            4,
        )
        if share_cols and positive_budget_mask.any()
        else 0.0,
        "geo_remaining_missing_cells": int(diagnostic_dataset.isna().sum().sum()),
        "geo_remaining_infinite_cells": _count_infinite_cells(diagnostic_dataset),
    }

    weekly_sales.to_parquet(CONFIG.geo_weekly_sales_file, index=False)
    weekly_orders.to_parquet(CONFIG.geo_weekly_orders_file, index=False)
    weekly_calendar.to_parquet(CONFIG.geo_weekly_calendar_file, index=False)
    weekly_traffic.to_parquet(CONFIG.geo_weekly_traffic_file, index=False)
    weekly_media.to_parquet(CONFIG.geo_weekly_media_file, index=False)
    diagnostic_dataset.to_parquet(CONFIG.geo_diagnostic_dataset_file, index=False)
    causal_dataset.to_parquet(CONFIG.geo_model_dataset_file, index=False)
    pd.concat(
        [
            _dataset_summary(diagnostic_dataset, "geo_diagnostic"),
            _dataset_summary(causal_dataset, "geo_causal"),
        ],
        ignore_index=True,
    ).to_csv(CONFIG.reports_tables_dir / "preprocessing_geo_dataset_summary.csv", index=False)
    CONFIG.geo_data_checks_file.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return causal_dataset, checks
