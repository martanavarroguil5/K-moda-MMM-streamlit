from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.modeling.transforms import slugify_channel


GLOBAL_PANEL_LABEL = "Global"
KEY_COLUMNS = ["semana_inicio", "ciudad"]

SALES_USECOLS = [
    "fecha_venta",
    "ciudad",
    "canal_venta",
    "categoria",
    "descuento_pct",
    "venta_neta_sin_iva_eur",
    "coste_produccion_eur",
]

ORDERS_USECOLS = [
    "fecha_pedido",
    "ciudad",
    "importe_neto_sin_iva_eur",
]

CALENDAR_USECOLS = [
    "fecha",
    "ciudad",
    "anio",
    "mes",
    "payday_flag",
    "rebajas_flag",
    "black_friday_flag",
    "navidad_flag",
    "semana_santa_flag",
    "vacaciones_escolares_flag",
    "festivo_local_flag",
    "temperatura_media_c",
    "lluvia_indice",
    "turismo_indice",
    "incidencia_ecommerce_flag",
]

TRAFFIC_USECOLS = [
    "fecha",
    "ciudad",
    "visitas_tienda",
    "pedidos_tienda",
    "pedidos_click_collect",
    "tasa_conversion_tienda_pct",
    "sesiones_web",
    "pedidos_online",
    "tasa_conversion_web_pct",
]

MEDIA_USECOLS = [
    "semana_inicio",
    "semana_fin",
    "ciudad",
    "canal_medio",
    "inversion_eur",
]


def ensure_dirs() -> None:
    for path in [
        CONFIG.intermediate_dir,
        CONFIG.processed_dir,
        CONFIG.reports_tables_dir,
        CONFIG.reports_figures_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def week_start(series: pd.Series) -> pd.Series:
    """Return the Monday that labels a Monday-Sunday business week."""
    dates = pd.to_datetime(series)
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def _replace_inf(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _mode_or_default(series: pd.Series, default: int = 0) -> int:
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return int(mode.iloc[0])


def _ordered_scope_columns(frame: pd.DataFrame) -> list[str]:
    return KEY_COLUMNS + [column for column in frame.columns if column not in set(KEY_COLUMNS)]


def _assign_scope(frame: pd.DataFrame, keep_city: bool) -> pd.DataFrame:
    out = frame.copy()
    if not keep_city:
        out["ciudad"] = GLOBAL_PANEL_LABEL
    return out[_ordered_scope_columns(out)].sort_values(KEY_COLUMNS).reset_index(drop=True)


def _daily_city_grid(calendar: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [
            sorted(pd.to_datetime(calendar["fecha"]).unique()),
            sorted(calendar["ciudad"].astype(str).unique()),
        ],
        names=["fecha", "ciudad"],
    )


def build_weekly_sales(keep_city: bool = False) -> tuple[pd.DataFrame, dict]:
    sales = _replace_inf(pd.read_csv(CONFIG.sales_lines_file, usecols=SALES_USECOLS, parse_dates=["fecha_venta"]))
    orders = _replace_inf(pd.read_csv(CONFIG.orders_file, usecols=ORDERS_USECOLS, parse_dates=["fecha_pedido"]))
    calendar = pd.read_csv(CONFIG.calendar_file, usecols=["fecha", "ciudad"], parse_dates=["fecha"])

    sales = sales.copy()
    sales["promo_flag"] = (sales["descuento_pct"].fillna(0.0) > 0.0).astype(int)
    sales["gross_margin_eur"] = (
        sales["venta_neta_sin_iva_eur"].fillna(0.0) - sales["coste_produccion_eur"].fillna(0.0)
    )

    daily_group_keys = ["fecha_venta", "ciudad"] if keep_city else ["fecha_venta"]
    sales_daily = (
        sales.groupby(daily_group_keys, as_index=False)
        .agg(
            ventas_netas=("venta_neta_sin_iva_eur", "sum"),
            gross_margin_eur=("gross_margin_eur", "sum"),
            promo_share_lineas=("promo_flag", "mean"),
        )
        .rename(columns={"fecha_venta": "fecha"})
    )

    if keep_city:
        sales_daily = sales_daily.set_index(["fecha", "ciudad"]).reindex(_daily_city_grid(calendar)).reset_index()
    else:
        all_days = pd.Index(sorted(pd.to_datetime(calendar["fecha"]).unique()), name="fecha")
        sales_daily = (
            sales_daily.set_index("fecha")
            .reindex(all_days)
            .reset_index()
            .rename(columns={"index": "fecha"})
        )
    sales_daily[["ventas_netas", "gross_margin_eur", "promo_share_lineas"]] = sales_daily[
        ["ventas_netas", "gross_margin_eur", "promo_share_lineas"]
    ].fillna(0.0)
    sales_daily["semana_inicio"] = week_start(sales_daily["fecha"])

    weekly_group_keys = KEY_COLUMNS if keep_city else ["semana_inicio"]
    weekly_sales = (
        sales_daily.groupby(weekly_group_keys, as_index=False)
        .agg(
            ventas_netas=("ventas_netas", "sum"),
            gross_margin_eur=("gross_margin_eur", "sum"),
            promo_share_lineas=("promo_share_lineas", "mean"),
        )
        .sort_values("semana_inicio")
    )
    weekly_sales["margen_bruto_ponderado"] = np.where(
        weekly_sales["ventas_netas"] > 0.0,
        weekly_sales["gross_margin_eur"] / weekly_sales["ventas_netas"],
        0.0,
    )
    weekly_sales = weekly_sales.drop(columns=["gross_margin_eur"])
    weekly_sales = _assign_scope(weekly_sales, keep_city=keep_city)

    sales_numeric = sales.select_dtypes(include="number")
    checks = {
        "week_definition": "Monday-start week (Monday-Sunday), labelled by semana_inicio",
        "target_source_column": "venta_neta_sin_iva_eur",
        "aggregation_scope": "city_weekly_panel" if keep_city else "global_weekly_series",
        "global_series_label": GLOBAL_PANEL_LABEL if not keep_city else None,
        "sales_total_lineas": round(float(sales["venta_neta_sin_iva_eur"].sum()), 2),
        "sales_total_pedidos": round(float(orders["importe_neto_sin_iva_eur"].sum()), 2),
        "sales_total_weekly": round(float(weekly_sales["ventas_netas"].sum()), 2),
        "missing_sales_days_closed": int((sales_daily["ventas_netas"] == 0.0).sum()),
        "missing_sales_day_city": int((sales_daily["ventas_netas"] == 0.0).sum()),
        "raw_sales_null_cells": int(sales.isna().sum().sum()),
        "raw_sales_infinite_cells": int(np.isinf(sales_numeric.to_numpy()).sum()) if not sales_numeric.empty else 0,
        "raw_sales_negative_line_rows": int((sales["venta_neta_sin_iva_eur"] < 0.0).sum()),
        "raw_sales_zero_line_rows": int((sales["venta_neta_sin_iva_eur"] == 0.0).sum()),
        "raw_sales_min_line_value": round(float(sales["venta_neta_sin_iva_eur"].min()), 2),
        "raw_sales_max_line_value": round(float(sales["venta_neta_sin_iva_eur"].max()), 2),
        "raw_sales_p999_line_value": round(float(sales["venta_neta_sin_iva_eur"].quantile(0.999)), 2),
        "weekly_sales_negative_rows": int((weekly_sales["ventas_netas"] < 0.0).sum()),
    }
    return weekly_sales, checks


def build_weekly_orders(keep_city: bool = False) -> pd.DataFrame:
    orders = _replace_inf(pd.read_csv(CONFIG.orders_file, usecols=ORDERS_USECOLS, parse_dates=["fecha_pedido"]))
    orders["semana_inicio"] = week_start(orders["fecha_pedido"])
    group_keys = KEY_COLUMNS if keep_city else ["semana_inicio"]
    weekly_orders = (
        orders.groupby(group_keys, as_index=False)
        .agg(
            pedidos_total=("importe_neto_sin_iva_eur", "size"),
            importe_neto_total=("importe_neto_sin_iva_eur", "sum"),
        )
        .sort_values("semana_inicio")
    )
    weekly_orders["ticket_medio_neto"] = weekly_orders["importe_neto_total"] / weekly_orders["pedidos_total"].clip(lower=1.0)
    weekly_orders = weekly_orders.drop(columns=["importe_neto_total"])
    return _assign_scope(weekly_orders, keep_city=keep_city)


def build_weekly_calendar(keep_city: bool = False) -> pd.DataFrame:
    calendar = _replace_inf(pd.read_csv(CONFIG.calendar_file, usecols=CALENDAR_USECOLS, parse_dates=["fecha"]))
    daily_group_keys = ["fecha", "ciudad"] if keep_city else ["fecha"]
    daily_scope = (
        calendar.groupby(daily_group_keys, as_index=False)
        .agg(
            anio=("anio", "max"),
            mes=("mes", lambda s: _mode_or_default(s)),
            payday_flag=("payday_flag", "max"),
            rebajas_flag=("rebajas_flag", "max"),
            black_friday_flag=("black_friday_flag", "max"),
            navidad_flag=("navidad_flag", "max"),
            semana_santa_flag=("semana_santa_flag", "max"),
            vacaciones_escolares_flag=("vacaciones_escolares_flag", "max"),
            festivo_local_count=("festivo_local_flag", "sum"),
            temperatura_media_c_mean=("temperatura_media_c", "mean"),
            lluvia_indice_mean=("lluvia_indice", "mean"),
            turismo_indice_mean=("turismo_indice", "mean"),
            incidencia_ecommerce_flag=("incidencia_ecommerce_flag", "max"),
        )
        .sort_values(daily_group_keys)
    )
    daily_scope["semana_inicio"] = week_start(daily_scope["fecha"])

    weekly_group_keys = KEY_COLUMNS if keep_city else ["semana_inicio"]
    weekly_calendar = (
        daily_scope.groupby(weekly_group_keys, as_index=False)
        .agg(
            days_in_week=("fecha", "size"),
            anio=("anio", "max"),
            mes_modal=("mes", lambda s: _mode_or_default(s)),
            rebajas_flag=("rebajas_flag", "max"),
            black_friday_flag=("black_friday_flag", "max"),
            navidad_flag=("navidad_flag", "max"),
            semana_santa_flag=("semana_santa_flag", "max"),
            vacaciones_escolares_flag=("vacaciones_escolares_flag", "max"),
            festivo_local_count=("festivo_local_count", "sum"),
            payday_count=("payday_flag", "sum"),
            temperatura_media_c_mean=("temperatura_media_c_mean", "mean"),
            lluvia_indice_mean=("lluvia_indice_mean", "mean"),
            turismo_indice_mean=("turismo_indice_mean", "mean"),
            incidencia_ecommerce_flag=("incidencia_ecommerce_flag", "max"),
        )
        .assign(week_complete_flag=lambda df: (df["days_in_week"] == 7).astype(int))
        .sort_values("semana_inicio")
    )
    return _assign_scope(weekly_calendar, keep_city=keep_city)


def build_weekly_traffic(keep_city: bool = False) -> pd.DataFrame:
    traffic = _replace_inf(pd.read_csv(CONFIG.traffic_file, usecols=TRAFFIC_USECOLS, parse_dates=["fecha"]))
    daily_group_keys = ["fecha", "ciudad"] if keep_city else ["fecha"]
    daily_scope = (
        traffic.groupby(daily_group_keys, as_index=False)
        .agg(
            visitas_tienda_sum=("visitas_tienda", "sum"),
            pedidos_tienda_sum=("pedidos_tienda", "sum"),
            pedidos_click_collect_sum=("pedidos_click_collect", "sum"),
            sesiones_web_sum=("sesiones_web", "sum"),
            pedidos_online_sum=("pedidos_online", "sum"),
        )
        .sort_values(daily_group_keys)
    )
    daily_scope["semana_inicio"] = week_start(daily_scope["fecha"])

    weekly_group_keys = KEY_COLUMNS if keep_city else ["semana_inicio"]
    weekly_traffic = (
        daily_scope.groupby(weekly_group_keys, as_index=False)
        .agg(
            visitas_tienda_sum=("visitas_tienda_sum", "sum"),
            pedidos_tienda_sum=("pedidos_tienda_sum", "sum"),
            pedidos_click_collect_sum=("pedidos_click_collect_sum", "sum"),
            sesiones_web_sum=("sesiones_web_sum", "sum"),
            pedidos_online_sum=("pedidos_online_sum", "sum"),
        )
        .sort_values("semana_inicio")
    )
    weekly_traffic["tasa_conversion_tienda_mean"] = np.where(
        weekly_traffic["visitas_tienda_sum"] > 0.0,
        weekly_traffic["pedidos_tienda_sum"] / weekly_traffic["visitas_tienda_sum"] * 100.0,
        0.0,
    )
    weekly_traffic["tasa_conversion_web_mean"] = np.where(
        weekly_traffic["sesiones_web_sum"] > 0.0,
        weekly_traffic["pedidos_online_sum"] / weekly_traffic["sesiones_web_sum"] * 100.0,
        0.0,
    )
    return _assign_scope(weekly_traffic, keep_city=keep_city)


def build_weekly_media(keep_city: bool = False) -> pd.DataFrame:
    media = _replace_inf(
        pd.read_csv(
            CONFIG.investment_file,
            usecols=MEDIA_USECOLS,
            parse_dates=["semana_inicio", "semana_fin"],
        )
    )
    media["canal_slug"] = media["canal_medio"].map(slugify_channel)
    group_keys = ["semana_inicio", "ciudad", "canal_slug"] if keep_city else ["semana_inicio", "canal_slug"]
    weekly = (
        media.groupby(group_keys, as_index=False)["inversion_eur"]
        .sum()
        .pivot(index=KEY_COLUMNS if keep_city else "semana_inicio", columns="canal_slug", values="inversion_eur")
        .fillna(0.0)
        .reset_index()
    )
    weekly.columns = [
        f"media_{column}" if column not in KEY_COLUMNS else column
        for column in weekly.columns
    ]
    return _assign_scope(weekly, keep_city=keep_city)
