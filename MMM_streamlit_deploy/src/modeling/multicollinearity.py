from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from src.common.config import CONFIG
from src.modeling.specs import CONTROL_COLUMNS, EXPLORATORY_SCREENING_COLUMNS
from src.modeling.trainer import budget_share_columns, load_dataset, media_columns


RAW_VIF_TABLE = CONFIG.reports_tables_dir / "multicollinearity_vif_raw_media.csv"
WEIGHT_VIF_TABLE = CONFIG.reports_tables_dir / "multicollinearity_vif_media_weights.csv"
CORR_TABLE = CONFIG.reports_tables_dir / "multicollinearity_high_correlation_pairs.csv"
REPORT_MD = CONFIG.docs_dir / "multicollinearity_review.md"
STEP_MD = CONFIG.docs_dir / "step_5_multicollinearity_review.md"
RAW_VIF_FIG = CONFIG.reports_figures_dir / "4_feature_importance" / "multicollinearity_vif_raw_media.png"
WEIGHT_VIF_FIG = CONFIG.reports_figures_dir / "4_feature_importance" / "multicollinearity_vif_media_weights.png"


@dataclass(frozen=True)
class MulticollinearityArtifacts:
    raw_vif: pd.DataFrame
    weight_vif: pd.DataFrame
    high_corr_pairs: pd.DataFrame
    dropped_weight_reference: str | None


@dataclass(frozen=True)
class CollinearityFilterArtifacts:
    kept_features: list[str]
    dropped: pd.DataFrame
    vif_table: pd.DataFrame


def _compute_vif_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    x = df[feature_columns].astype(float)
    for feature in feature_columns:
        y = x[feature].to_numpy(dtype=float)
        x_other = x.drop(columns=[feature])
        if x_other.shape[1] == 0:
            vif = 1.0
        elif np.nanstd(y) < 1e-12:
            vif = float("inf")
        else:
            model = LinearRegression()
            model.fit(x_other, y)
            r2 = model.score(x_other, y)
            if r2 >= 0.999999:
                vif = float("inf")
            else:
                vif = float(1.0 / max(1e-12, 1.0 - r2))
        rows.append({"feature": feature, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, ignore_index=True)


def _high_corr_pairs(df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.8) -> pd.DataFrame:
    corr = df[feature_columns].astype(float).corr().abs()
    rows = []
    for i, left in enumerate(feature_columns):
        for right in feature_columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_left": left, "feature_right": right, "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False, ignore_index=True)


def filter_beta_collinear_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    protected_features: list[str],
    *,
    corr_threshold: float = 0.95,
    vif_threshold: float = 20.0,
) -> CollinearityFilterArtifacts:
    working_features = list(feature_columns)
    protected = set(protected_features)
    dropped_rows: list[dict[str, float | str]] = []

    if len(working_features) <= 1:
        vif_table = _compute_vif_table(df, working_features) if working_features else pd.DataFrame(columns=["feature", "vif"])
        return CollinearityFilterArtifacts(kept_features=working_features, dropped=pd.DataFrame(dropped_rows), vif_table=vif_table)

    def _drop_feature(feature: str, reason: str, detail: str) -> None:
        if feature in working_features:
            working_features.remove(feature)
            dropped_rows.append({"feature": feature, "reason": reason, "detail": detail})

    corr_pairs = _high_corr_pairs(df, working_features, threshold=corr_threshold)
    for _, row in corr_pairs.iterrows():
        left = str(row["feature_left"])
        right = str(row["feature_right"])
        if left not in working_features or right not in working_features:
            continue
        if left in protected and right in protected:
            continue
        if left in protected:
            _drop_feature(right, "high_corr", f"abs_corr={float(row['abs_corr']):.4f} with protected `{left}`")
            continue
        if right in protected:
            _drop_feature(left, "high_corr", f"abs_corr={float(row['abs_corr']):.4f} with protected `{right}`")
            continue

        subset = df[[left, right]].astype(float).corr().abs()
        left_corr_sum = float(subset[left].sum())
        right_corr_sum = float(subset[right].sum())
        drop = left if left_corr_sum >= right_corr_sum else right
        keep = right if drop == left else left
        _drop_feature(drop, "high_corr", f"abs_corr={float(row['abs_corr']):.4f} with `{keep}`")

    while True:
        candidates = [feature for feature in working_features if feature not in protected]
        if len(working_features) <= 1 or not candidates:
            break
        vif_table = _compute_vif_table(df, working_features)
        candidate_vif = vif_table[vif_table["feature"].isin(candidates)].copy()
        if candidate_vif.empty:
            break
        worst = candidate_vif.sort_values("vif", ascending=False).iloc[0]
        if float(worst["vif"]) <= vif_threshold:
            break
        _drop_feature(str(worst["feature"]), "high_vif", f"vif={float(worst['vif']):.4f}")

    final_vif = _compute_vif_table(df, working_features) if working_features else pd.DataFrame(columns=["feature", "vif"])
    dropped_df = pd.DataFrame(dropped_rows)
    return CollinearityFilterArtifacts(
        kept_features=working_features,
        dropped=dropped_df.sort_values(["reason", "feature"]).reset_index(drop=True) if not dropped_df.empty else dropped_df,
        vif_table=final_vif,
    )


def _plot_vif(vif_df: pd.DataFrame, title: str, path) -> None:
    plot_df = vif_df.replace([np.inf, -np.inf], np.nan).fillna(999.0).head(15).copy()
    plot_df = plot_df.sort_values("vif", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, x="vif", y="feature", hue="feature", palette="crest", legend=False)
    plt.title(title)
    plt.xlabel("VIF")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def run_multicollinearity_review() -> MulticollinearityArtifacts:
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.reports_figures_dir / "4_feature_importance").mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    media_cols = media_columns(df)
    weight_cols = budget_share_columns(df)
    exploratory_cols = [column for column in EXPLORATORY_SCREENING_COLUMNS if column in df.columns]
    if "budget_total_eur" in df.columns:
        exploratory_cols = exploratory_cols + ["budget_total_eur"]

    base_candidate_cols = CONTROL_COLUMNS + exploratory_cols + media_cols
    raw_vif = _compute_vif_table(df, base_candidate_cols)
    raw_vif.to_csv(RAW_VIF_TABLE, index=False)

    dropped_weight_reference = None
    weight_candidate_cols = CONTROL_COLUMNS + exploratory_cols
    if weight_cols:
        dropped_weight_reference = sorted(weight_cols)[-1]
        weight_candidate_cols = weight_candidate_cols + [col for col in sorted(weight_cols) if col != dropped_weight_reference]
    weight_vif = _compute_vif_table(df, weight_candidate_cols)
    weight_vif.to_csv(WEIGHT_VIF_TABLE, index=False)

    corr_feature_cols = CONTROL_COLUMNS + exploratory_cols + media_cols + weight_cols
    high_corr = _high_corr_pairs(df, corr_feature_cols, threshold=0.8)
    high_corr.to_csv(CORR_TABLE, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    _plot_vif(raw_vif, "VIF Review - Raw Media Specification", RAW_VIF_FIG)
    _plot_vif(weight_vif, "VIF Review - Media Weights Specification", WEIGHT_VIF_FIG)

    raw_top = raw_vif.head(10)
    weight_top = weight_vif.head(10)

    report_lines = [
        "# Multicollinearity Review",
        "",
        "## Scope",
        "",
        "- Dataset usado: vista causal (`model_dataset.parquet`).",
        "- Objetivo: revisar estabilidad de variables antes del modelo lineal/hibrido.",
        "- Se comparan dos lecturas de medios:",
        "  - especificacion con `media_*` en euros.",
        "  - especificacion con `budget_total_eur` y pesos `budget_share_pct_*`.",
        f"- Para la especificacion de pesos se deja fuera `{dropped_weight_reference}` como categoria de referencia para evitar la suma exacta al 100%.",
        "",
        "## Raw Media VIF",
        "",
        "```text",
        raw_top.round(2).to_string(index=False),
        "```",
        "",
        "## Weighted Media VIF",
        "",
        "```text",
        weight_top.round(2).to_string(index=False),
        "```",
        "",
        "## High Correlation Pairs",
        "",
        "```text",
        (high_corr.head(20).round(3).to_string(index=False) if not high_corr.empty else "No pairs above threshold."),
        "```",
        "",
        "## Reading",
        "",
        "- La multicolinealidad es especialmente importante en MMM porque afecta la estabilidad de coeficientes y la lectura de contribuciones.",
        "- Las variables en pesos ayudan a interpretar mix, pero si se meten todas juntas generan dependencia exacta por la restriccion de suma.",
        "- Por eso, para una especificacion lineal con pesos, la regla sana es usar `budget_total_eur` y `n-1` pesos.",
        "- La revision se centra ya en variables candidatas del panel causal limpio, sin proxies downstream del negocio.",
        "",
        "## Artifacts",
        "",
        f"- VIF medios en euros: `{RAW_VIF_TABLE.name}`.",
        f"- VIF pesos publicitarios: `{WEIGHT_VIF_TABLE.name}`.",
        f"- Correlaciones altas: `{CORR_TABLE.name}`.",
        f"- Figura VIF medios: `4_feature_importance/{RAW_VIF_FIG.name}`.",
        f"- Figura VIF pesos: `4_feature_importance/{WEIGHT_VIF_FIG.name}`.",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    step_lines = [
        "# Step 5 - Multicollinearity Review",
        "",
        "## Objetivo",
        "",
        "Revisar si el set de variables candidato para el modelo presenta colinealidad suficiente como para volver inestables los coeficientes o sesgar la interpretacion del mix.",
        "",
        "## Que Hacemos",
        "",
        "- Calculamos VIF sobre una especificacion con medios en euros.",
        "- Calculamos VIF sobre una especificacion con presupuesto total y pesos publicitarios.",
        "- Revisamos correlaciones altas entre variables candidatas.",
        "- El chequeo se hace sobre controles, medios en euros y representaciones de mix compatibles con el panel causal actual.",
        "",
        "## Que Creamos",
        "",
        f"- `{RAW_VIF_TABLE.name}`",
        f"- `{WEIGHT_VIF_TABLE.name}`",
        f"- `{CORR_TABLE.name}`",
        f"- `4_feature_importance/{RAW_VIF_FIG.name}`",
        f"- `4_feature_importance/{WEIGHT_VIF_FIG.name}`",
        f"- `{REPORT_MD.name}`",
        "",
        "## Conclusiones",
        "",
        "- La multicolinealidad no se resuelve en preprocessing basico; es una revision de preparacion para modelado.",
        "- Si usamos pesos publicitarios, debemos imponer una variable de referencia para no forzar suma exacta al 100% dentro del modelo lineal.",
        "- Esta revision nos dejara una base mas estable para decidir si el modelo final trabaja con euros, con pesos o con una formulacion mixta.",
    ]
    STEP_MD.write_text("\n".join(step_lines), encoding="utf-8")

    return MulticollinearityArtifacts(
        raw_vif=raw_vif,
        weight_vif=weight_vif,
        high_corr_pairs=high_corr,
        dropped_weight_reference=dropped_weight_reference,
    )
