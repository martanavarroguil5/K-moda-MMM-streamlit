from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.modeling.specs import CONTROL_COLUMNS, RANDOM_STATE
from src.modeling.trainer import budget_share_columns, city_dummy_columns, load_dataset, media_columns
from src.validation.visual_utils import APP_BG, BLUE, BORDER, GREEN, GRID, LIGHT_BLUE, NAVY, PANEL_BG, SLATE, TEAL


RF_IMPORTANCE_TABLE = CONFIG.reports_tables_dir / "random_forest_feature_importance.csv"
RF_GROUP_TABLE = CONFIG.reports_tables_dir / "random_forest_feature_group_importance.csv"
RF_SPEC_SUMMARY_TABLE = CONFIG.reports_tables_dir / "random_forest_spec_summary.csv"
RF_REPORT_MD = CONFIG.docs_dir / "random_forest_feature_importance.md"
RF_SECTION_MD = CONFIG.docs_dir / "step_4_feature_importance.md"
RF_TOP_FIG = CONFIG.reports_figures_dir / "4_feature_importance" / "random_forest_top_features.png"
RF_GROUP_FIG = CONFIG.reports_figures_dir / "4_feature_importance" / "random_forest_feature_groups.png"
RF_SPEC_FIG = CONFIG.reports_figures_dir / "4_feature_importance" / "random_forest_spec_comparison.png"

FEATURE_GROUP_COLORS = {
    "media": BLUE,
    "media_mix": TEAL,
    "city": GREEN,
    "seasonality_trend": NAVY,
    "macro_weather": LIGHT_BLUE,
    "calendar_exogenous": SLATE,
}
SPEC_COLORS = {
    "RFControlsOnly": NAVY,
    "RFMediaLevels": BLUE,
    "RFBudgetMix": GREEN,
}
SPEC_LABELS = {
    "RFControlsOnly": "Controls only",
    "RFMediaLevels": "Media levels",
    "RFBudgetMix": "Budget mix",
}
METRIC_DIRECTIONS = {
    "mape": "Lower is better",
    "rmse": "Lower is better",
    "r2": "Higher is better",
}
METRIC_TITLES = {
    "mape": "MAPE on 2024 holdout",
    "rmse": "RMSE on 2024 holdout",
    "r2": "R2 on 2024 holdout",
}


@dataclass(frozen=True)
class RandomForestImportanceArtifacts:
    metrics: dict
    feature_importance: pd.DataFrame
    group_importance: pd.DataFrame
    spec_summary: pd.DataFrame
    recommended_spec: str


def _feature_group(feature: str) -> str:
    if feature.startswith("media_"):
        return "media"
    if feature == "budget_total_eur" or feature.startswith("budget_share_pct_"):
        return "media_mix"
    if feature.startswith("city_"):
        return "city"
    if feature in {"trend_index", "week_sin", "week_cos"}:
        return "seasonality_trend"
    if feature in {"temperatura_media_c_mean", "lluvia_indice_mean", "turismo_indice_mean"}:
        return "macro_weather"
    return "calendar_exogenous"


def _top_feature_lines(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    lines = []
    top = df.sort_values("permutation_importance_mean", ascending=False).head(top_n)
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['feature']}` ({row['feature_group']}): "
            f"perm `{row['permutation_importance_mean']:.5f}`, "
            f"impurity `{row['rf_importance']:.5f}`."
        )
    return lines


def _feature_group_color(group: str) -> str:
    return FEATURE_GROUP_COLORS.get(group, "#6E7681")


def _spec_color(spec: str) -> str:
    return SPEC_COLORS.get(spec, "#4C566A")


def _spec_label(spec: str) -> str:
    return SPEC_LABELS.get(spec, spec)


def _annotate_horizontal_bars(
    ax: plt.Axes,
    values: list[float],
    fmt: str = "{:.4f}",
    pad_fraction: float = 0.012,
) -> list:
    x_min, x_max = ax.get_xlim()
    span = max(x_max - x_min, 1e-9)
    pad = span * pad_fraction
    texts = []
    for patch, value in zip(ax.patches, values):
        y = patch.get_y() + patch.get_height() / 2
        if value >= 0:
            x = patch.get_width() + pad
            ha = "left"
        else:
            x = patch.get_width() - pad
            ha = "right"
        texts.append(ax.text(x, y, fmt.format(value), va="center", ha=ha, fontsize=10, color=NAVY))
    return texts


def _annotate_vertical_bars(
    ax: plt.Axes,
    values: list[float],
    fmt: str = "{:.3f}",
    pad_fraction: float = 0.02,
) -> list:
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-9)
    pad = span * pad_fraction
    texts = []
    for patch, value in zip(ax.patches, values):
        x = patch.get_x() + patch.get_width() / 2
        if value >= 0:
            y = value + pad
            va = "bottom"
        else:
            y = value - pad
            va = "top"
        texts.append(ax.text(x, y, fmt.format(value), ha="center", va=va, fontsize=10, color=NAVY))
    return texts


def _style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(axis=grid_axis, color=GRID, alpha=0.9, linewidth=0.8)
    other_axis = "y" if grid_axis == "x" else "x"
    ax.grid(axis=other_axis, visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SLATE, labelsize=10.5)
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(BORDER)
        ax.spines[side].set_linewidth(1.0)
    sns.despine(ax=ax, left=False, bottom=False)


def _weight_screening_features(df: pd.DataFrame) -> list[str]:
    weight_cols = budget_share_columns(df)
    reference_col = sorted(weight_cols)[-1] if weight_cols else None
    selected_weights = [col for col in sorted(weight_cols) if col != reference_col]
    feature_columns = CONTROL_COLUMNS + city_dummy_columns(df)
    if "budget_total_eur" in df.columns:
        feature_columns += ["budget_total_eur"]
    feature_columns += selected_weights
    return feature_columns


def _feature_set_specs(df: pd.DataFrame) -> list[tuple[str, str, list[str]]]:
    city_cols = city_dummy_columns(df)
    return [
        (
            "RFControlsOnly",
            "Controles exogenos y estacionalidad compacta, sin medios.",
            CONTROL_COLUMNS + city_cols,
        ),
        (
            "RFMediaLevels",
            "Controles exogenos mas niveles brutos por canal `media_*`.",
            CONTROL_COLUMNS + city_cols + media_columns(df),
        ),
        (
            "RFBudgetMix",
            "Controles exogenos mas `budget_total_eur` y pesos publicitarios `n-1`.",
            _weight_screening_features(df),
        ),
    ]


def _fit_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x_train = train_df[feature_columns].astype(float)
    y_train = train_df["ventas_netas"].astype(float)
    x_test = test_df[feature_columns].astype(float)
    y_test = test_df["ventas_netas"].astype(float)

    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)

    metrics = compute_metrics(y_test.to_numpy(), pred_test)
    metrics["r2"] = float(r2_score(y_test, pred_test))

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_group": [_feature_group(feature) for feature in feature_columns],
            "rf_importance": model.feature_importances_,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False, ignore_index=True)

    group_df = (
        importance_df.groupby("feature_group", as_index=False)[["rf_importance", "permutation_importance_mean"]]
        .sum()
        .sort_values("permutation_importance_mean", ascending=False, ignore_index=True)
    )
    return metrics, importance_df, group_df


def run_random_forest_feature_importance() -> RandomForestImportanceArtifacts:
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.reports_figures_dir / "4_feature_importance").mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df = df[df["year"] < 2024].copy()
    test_df = df[df["year"] == 2024].copy()

    spec_rows: list[dict] = []
    spec_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    spec_descriptions: dict[str, str] = {}

    for spec_name, description, feature_columns in _feature_set_specs(df):
        metrics, importance_df, group_df = _fit_random_forest(train_df, test_df, feature_columns)
        spec_rows.append(
            {
                "spec": spec_name,
                "description": description,
                "feature_count": int(len(feature_columns)),
                **metrics,
            }
        )
        spec_outputs[spec_name] = (importance_df, group_df)
        spec_descriptions[spec_name] = description

    spec_summary = (
        pd.DataFrame(spec_rows)
        .sort_values(["mape", "rmse"], ascending=[True, True])
        .reset_index(drop=True)
    )
    recommended_spec = str(spec_summary.iloc[0]["spec"])
    selected_metrics = spec_summary.iloc[0].to_dict()
    importance_df, group_df = spec_outputs[recommended_spec]
    importance_df = importance_df.copy()
    group_df = group_df.copy()
    importance_df["spec"] = recommended_spec
    group_df["spec"] = recommended_spec

    RF_IMPORTANCE_TABLE.write_text("", encoding="utf-8")
    RF_GROUP_TABLE.write_text("", encoding="utf-8")
    importance_df.to_csv(RF_IMPORTANCE_TABLE, index=False)
    group_df.to_csv(RF_GROUP_TABLE, index=False)
    spec_summary.to_csv(RF_SPEC_SUMMARY_TABLE, index=False)

    sns.set_theme(style="whitegrid", context="talk")

    top_plot = importance_df.head(12).copy().sort_values("permutation_importance_mean", ascending=True)
    top_colors = [_feature_group_color(group) for group in top_plot["feature_group"]]
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    fig.patch.set_facecolor(APP_BG)
    ax.barh(
        top_plot["feature"],
        top_plot["permutation_importance_mean"],
        xerr=top_plot["permutation_importance_std"],
        color=top_colors,
        edgecolor=APP_BG,
        linewidth=1.0,
        error_kw={"elinewidth": 1.1, "ecolor": SLATE, "capsize": 3},
    )
    ax.axvline(0.0, color=SLATE, linewidth=1.0, alpha=0.7)
    _style_axis(ax, grid_axis="x")
    x_min = min(float(top_plot["permutation_importance_mean"].min()), 0.0)
    x_max = max(float(top_plot["permutation_importance_mean"].max()), 0.0)
    x_pad = max((x_max - x_min) * 0.16, 2.5e-05)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_xlabel("MAPE increase after permutation")
    ax.set_ylabel("")
    _annotate_horizontal_bars(ax, top_plot["permutation_importance_mean"].tolist())
    legend_handles = [
        Patch(facecolor=_feature_group_color(group), edgecolor="none", label=group.replace("_", " ").title())
        for group in top_plot["feature_group"].drop_duplicates()
    ]
    ax.legend(handles=legend_handles, title="Feature group", loc="lower right", frameon=True, fontsize=10, title_fontsize=11)
    fig.suptitle(
        f"Top Random Forest features | {_spec_label(recommended_spec)}",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.07,
        0.94,
        "Permutation importance on 2024 holdout. Bars show mean +/- 1 std across shuffles.",
        fontsize=11,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RF_TOP_FIG, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
    plt.close(fig)

    group_plot = group_df.copy().sort_values("permutation_importance_mean", ascending=True)
    total_group_importance = group_plot["permutation_importance_mean"].abs().sum()
    group_colors = [_feature_group_color(group) for group in group_plot["feature_group"]]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor(APP_BG)
    ax.barh(
        group_plot["feature_group"],
        group_plot["permutation_importance_mean"],
        color=group_colors,
        edgecolor=APP_BG,
        linewidth=1.0,
    )
    ax.axvline(0.0, color=SLATE, linewidth=1.0, alpha=0.7)
    _style_axis(ax, grid_axis="x")
    x_min = min(float(group_plot["permutation_importance_mean"].min()), 0.0)
    x_max = max(float(group_plot["permutation_importance_mean"].max()), 0.0)
    x_pad = max((x_max - x_min) * 0.16, 2.0e-05)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_xlabel("Total permutation importance")
    ax.set_ylabel("")
    group_labels = []
    for row in group_plot.itertuples(index=False):
        share = abs(row.permutation_importance_mean) / total_group_importance if total_group_importance else 0.0
        group_labels.append(f"{row.permutation_importance_mean:.4f} | {share:.0%} abs")
    annotation_texts = _annotate_horizontal_bars(ax, group_plot["permutation_importance_mean"].tolist(), fmt="{}")
    for text_obj, label in zip(annotation_texts, group_labels):
        text_obj.set_text(label)
    fig.suptitle(
        f"Importance by feature family | {_spec_label(recommended_spec)}",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.07,
        0.94,
        "Aggregated permutation importance for the winning screening route.",
        fontsize=11,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RF_GROUP_FIG, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
    plt.close(fig)

    spec_plot = spec_summary.melt(
        id_vars=["spec", "description", "feature_count"],
        value_vars=["mape", "rmse", "r2"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    fig.patch.set_facecolor(APP_BG)
    for ax, metric in zip(axes, ["mape", "rmse", "r2"]):
        metric_df = spec_plot[spec_plot["metric"] == metric].copy()
        metric_df["spec_label"] = metric_df["spec"].map(_spec_label)
        colors = [_spec_color(spec) for spec in metric_df["spec"]]
        ax.bar(
            metric_df["spec_label"],
            metric_df["value"],
            color=colors,
            edgecolor=APP_BG,
            linewidth=1.0,
        )
        if metric == "r2":
            ax.axhline(0.0, color=SLATE, linewidth=1.0, alpha=0.7)
        _style_axis(ax, grid_axis="y")
        ax.set_title(METRIC_TITLES[metric], fontsize=14, weight="bold")
        ax.text(
            0.0,
            1.01,
            METRIC_DIRECTIONS[metric],
            transform=ax.transAxes,
            fontsize=10.5,
            color=SLATE,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=16)
        _annotate_vertical_bars(ax, metric_df["value"].tolist())
    fig.suptitle("Random Forest screening route comparison", x=0.06, y=1.03, ha="left", fontsize=18, weight="bold")
    fig.text(0.06, 0.97, "Train: 2020-2023 | Test: 2024 | Winner highlighted by metric ranking order", fontsize=11, color=SLATE)
    fig.tight_layout()
    fig.savefig(RF_SPEC_FIG, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
    plt.close(fig)

    report_lines = [
        "# Random Forest Feature Importance",
        "",
        "## Scope",
        "",
        "- Dataset usado: vista causal (`model_dataset.parquet`).",
        "- El paso 04 ya no mezcla todas las representaciones de medios en una sola corrida.",
        "- Comparamos tres specs de screening:",
        "  - `RFControlsOnly`: baseline de controles.",
        "  - `RFMediaLevels`: controles + `media_*`.",
        "  - `RFBudgetMix`: controles + `budget_total_eur` + pesos publicitarios `n-1`.",
        "- Variables excluidas: trafico, proxies downstream y combinaciones redundantes de mix dentro de una misma corrida.",
        "- Esquema temporal: entrenamiento con `2020-2023`, evaluacion con `2024`.",
        "",
        "## Spec Comparison",
        "",
        "```text",
        spec_summary[["spec", "feature_count", "mape", "rmse", "bias", "r2"]].round(4).to_string(index=False),
        "```",
        "",
        "## Recommended Screening Spec",
        "",
        f"- Spec recomendada: `{recommended_spec}`.",
        f"- Lectura: {spec_descriptions[recommended_spec]}",
        "",
        "## Test Metrics",
        "",
        f"- MAPE 2024: `{selected_metrics['mape']:.2f}`.",
        f"- MAE 2024: `{selected_metrics['mae']:.2f}`.",
        f"- RMSE 2024: `{selected_metrics['rmse']:.2f}`.",
        f"- Bias 2024: `{selected_metrics['bias']:.2f}`.",
        f"- R2 2024: `{selected_metrics['r2']:.4f}`.",
        "",
        "## Top Variables",
        "",
        *_top_feature_lines(importance_df, top_n=12),
        "",
        "## Importance By Group",
        "",
        "```text",
        group_df.round(5).to_string(index=False),
        "```",
        "",
        "## Reading",
        "",
        "- La lectura principal debe apoyarse mas en la importancia por permutacion que en la impurity importance, porque es menos sesgada.",
        "- Este Random Forest no sustituye al MMM; sirve para comparar rutas de screening antes del modelo final.",
        "- Separar `media_*` de `budget_total_eur + budget_share_pct_*` hace el paso 04 bastante mas interpretable.",
        f"- En este caso la spec recomendada es `{recommended_spec}`, pero la senal sigue siendo prudente: no se usa para imponer variables, sino para priorizar exploracion y descartar sobrelecturas.",
        "",
        "## Artifacts",
        "",
        f"- Tabla spec summary: `{RF_SPEC_SUMMARY_TABLE.name}`.",
        f"- Tabla completa de la spec recomendada: `{RF_IMPORTANCE_TABLE.name}`.",
        f"- Resumen por grupo de la spec recomendada: `{RF_GROUP_TABLE.name}`.",
        f"- Figura top features: `4_feature_importance/{RF_TOP_FIG.name}`.",
        f"- Figura por grupos: `4_feature_importance/{RF_GROUP_FIG.name}`.",
        f"- Figura comparativa de specs: `4_feature_importance/{RF_SPEC_FIG.name}`.",
    ]
    RF_REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    section_lines = [
        "# Step 4 - Feature Importance",
        "",
        "## Objetivo",
        "",
        "Evaluar, antes del modelado final, que ruta de features merece llegar viva al siguiente filtro.",
        "",
        "## Por Que Lo Hacemos",
        "",
        "- Un MMM lineal puede infravalorar relaciones no lineales o interacciones.",
        "- Un Random Forest sirve como chequeo rapido de senal sin imponer una forma funcional lineal.",
        "- La importancia por permutacion nos ayuda a distinguir variables aparentemente fuertes de variables realmente utiles fuera de muestra.",
        "- Comparar varias specs evita la sopa de variables y hace mas interpretable el screening.",
        "",
        "## Que Hacemos",
        "",
        "- Entrenamos varias corridas de `RandomForestRegressor` con la vista causal.",
        "- Separamos la ruta de `media_*` de la ruta `budget_total_eur + budget_share_pct_*`.",
        "- Excluimos trafico, proxies downstream y mezclas redundantes dentro de una misma corrida.",
        "- Evaluamos siempre en `2024` para conservar criterio temporal.",
        "",
        "## Outputs",
        "",
        f"- Tabla de specs: `{RF_SPEC_SUMMARY_TABLE.name}`.",
        f"- Tabla de variables de la spec recomendada: `{RF_IMPORTANCE_TABLE.name}`.",
        f"- Tabla por grupos de la spec recomendada: `{RF_GROUP_TABLE.name}`.",
        f"- Figura top variables: `4_feature_importance/{RF_TOP_FIG.name}`.",
        f"- Figura por grupos: `4_feature_importance/{RF_GROUP_FIG.name}`.",
        f"- Figura comparativa de specs: `4_feature_importance/{RF_SPEC_FIG.name}`.",
        f"- Informe detallado: `{RF_REPORT_MD.name}`.",
        "",
        "## Conclusiones",
        "",
        f"- La spec recomendada para screening es `{recommended_spec}`.",
        f"- Su rendimiento test es `MAPE {selected_metrics['mape']:.2f}` y `R2 {selected_metrics['r2']:.4f}`.",
        "- El paso 04 queda mejor como comparativa de rutas que como ranking absoluto de una sola tabla gigante.",
    ]
    RF_SECTION_MD.write_text("\n".join(section_lines), encoding="utf-8")

    return RandomForestImportanceArtifacts(
        metrics=selected_metrics,
        feature_importance=importance_df,
        group_importance=group_df,
        spec_summary=spec_summary,
        recommended_spec=recommended_spec,
    )
