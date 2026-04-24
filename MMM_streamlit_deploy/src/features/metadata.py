from __future__ import annotations

import pandas as pd


KEY_COLUMNS = {"semana_inicio", "ciudad"}
TARGET_COLUMN = "ventas_netas"
DOWNSTREAM_BUSINESS_COLUMNS = {"pedidos_total", "ticket_medio_neto"}
SCREENING_TEMPORAL_LABEL_COLUMNS = {"anio", "mes_modal", "week_of_year", "year", "quarter"}
MODEL_READY_TEMPORAL_COLUMNS = {"trend_index", "week_sin", "week_cos"}


def feature_family(column: str) -> str:
    if column in KEY_COLUMNS:
        return "keys"
    if column == TARGET_COLUMN:
        return "target"
    if column in DOWNSTREAM_BUSINESS_COLUMNS:
        return "business"
    if column.startswith("media_"):
        return "media_raw"
    if column.startswith("budget_share_pct_") or column == "budget_total_eur":
        return "media_mix"
    if column in SCREENING_TEMPORAL_LABEL_COLUMNS or column in MODEL_READY_TEMPORAL_COLUMNS:
        return "temporal"
    return "calendar_context"


def feature_modeling_tier(column: str, diagnostic_only_columns: set[str] | None = None) -> str:
    diagnostic_only = diagnostic_only_columns or set()
    if column in KEY_COLUMNS:
        return "technical_key"
    if column == TARGET_COLUMN:
        return "target"
    if column in diagnostic_only or column in DOWNSTREAM_BUSINESS_COLUMNS:
        return "diagnostic_only"
    if column.startswith("media_") or column in SCREENING_TEMPORAL_LABEL_COLUMNS:
        return "screening_only"
    return "model_ready"


def feature_note(column: str, diagnostic_only_columns: set[str] | None = None) -> str:
    tier = feature_modeling_tier(column, diagnostic_only_columns)
    if tier == "technical_key":
        return "Clave tecnica para ordenar y alinear la serie semanal."
    if tier == "target":
        return "Variable objetivo semanal del MMM."
    if tier == "diagnostic_only":
        return "Senal reservada para diagnostico/QA y no para el set causal final."
    if tier == "screening_only" and column.startswith("media_"):
        return "Senal de nivel por canal util para screening y comparativas de especificacion."
    if tier == "screening_only":
        return "Etiqueta temporal descriptiva util para lectura y reporting, no para el modelo final."
    if column.startswith("budget_share_pct_") or column == "budget_total_eur":
        return "Representacion de presion total y composicion del mix compatible con el MMM final."
    if column in MODEL_READY_TEMPORAL_COLUMNS:
        return "Control temporal compacto para tendencia y estacionalidad en el MMM."
    return "Control exogeno plausible para el MMM final."


def feature_catalog(dataset: pd.DataFrame, diagnostic_only_columns: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for column in dataset.columns:
        rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "modeling_tier": feature_modeling_tier(column, diagnostic_only_columns),
                "note": feature_note(column, diagnostic_only_columns),
            }
        )
    return pd.DataFrame(rows)


def target_correlation_table(dataset: pd.DataFrame, diagnostic_only_columns: set[str] | None = None) -> pd.DataFrame:
    catalog = feature_catalog(dataset, diagnostic_only_columns)
    eligible = catalog.loc[
        catalog["modeling_tier"].isin(["model_ready", "screening_only"])
        & ~catalog["feature"].isin(SCREENING_TEMPORAL_LABEL_COLUMNS),
        "feature",
    ].tolist()
    numeric = dataset[[TARGET_COLUMN, *[column for column in eligible if column in dataset.columns]]].select_dtypes(include="number")
    correlations = (
        numeric.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .dropna()
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return correlations.merge(catalog, on="feature", how="left").assign(
        abs_correlation=lambda df: df["correlation_with_target"].abs()
    ).sort_values("abs_correlation", ascending=False, ignore_index=True)
