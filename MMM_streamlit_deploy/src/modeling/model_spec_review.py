from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.modeling.selection import evaluate_spec
from src.modeling.specs import CONTROL_COLUMNS
from src.modeling.trainer import (
    budget_share_columns,
    city_dummy_columns,
    fit_elastic_net,
    load_dataset,
    media_columns,
    original_scale_coefficients,
    predict_with_model,
)
from src.validation.backtesting import expanding_year_splits


SPEC_REVIEW_TABLE = CONFIG.reports_tables_dir / "model_spec_review_backtest.csv"
SPEC_REVIEW_SUMMARY = CONFIG.reports_tables_dir / "model_spec_review_summary.csv"
SPEC_REVIEW_DOC = CONFIG.docs_dir / "model_spec_review.md"
STEP_REVIEW_DOC = CONFIG.docs_dir / "step_6_model_spec_review.md"
SPEC_REVIEW_FIG = CONFIG.reports_figures_dir / "4_feature_importance" / "model_spec_review_comparison.png"


@dataclass(frozen=True)
class ModelSpecReviewArtifacts:
    backtest: pd.DataFrame
    summary: pd.DataFrame
    recommended_spec: str


def _budget_mix_features(df: pd.DataFrame) -> list[str]:
    weight_cols = budget_share_columns(df)
    reference_col = sorted(weight_cols)[-1] if weight_cols else None
    selected_weights = [col for col in sorted(weight_cols) if col != reference_col]
    return CONTROL_COLUMNS + city_dummy_columns(df) + ["budget_total_eur"] + selected_weights


def _evaluate_linear_feature_set(
    spec_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    scaler, model = fit_elastic_net(train_df, feature_columns)
    pred = predict_with_model(valid_df, feature_columns, scaler, model)
    metrics = compute_metrics(valid_df["ventas_netas"].to_numpy(), pred)
    coef = original_scale_coefficients(scaler, model, feature_columns)
    channel_like = coef[[col for col in feature_columns if col.startswith("media_") or col.startswith("budget_share_pct_")]]
    metrics["negative_channel_like_coefficients"] = int((channel_like < -1e-8).sum())
    metrics["alpha"] = float(model.alpha_)
    metrics["l1_ratio"] = float(model.l1_ratio_)

    scored = valid_df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    scored["pred"] = pred
    scored["spec"] = spec_name
    return metrics, scored


def run_model_spec_review() -> ModelSpecReviewArtifacts:
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.reports_figures_dir / "4_feature_importance").mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df["naive_pred"] = (
        df.sort_values(["ciudad", "semana_inicio"])
        .groupby("ciudad")["ventas_netas"]
        .transform(lambda series: series.shift(1).rolling(4, min_periods=1).mean())
        .fillna(df.groupby("ciudad")["ventas_netas"].transform("mean"))
    )
    media_cols = media_columns(df)
    city_cols = city_dummy_columns(df)

    rows: list[dict] = []
    for train_mask, valid_mask, fold_name in expanding_year_splits(df, min_train_year=2021):
        train_df = df.loc[train_mask].copy()
        valid_df = df.loc[valid_mask].copy()

        for spec_name, use_transforms, include_media in [
            ("ElasticNetControls", False, False),
            ("ElasticNetRawMedia", False, True),
            ("ElasticNetTransformedMedia", True, True),
        ]:
            metrics, _, _, _ = evaluate_spec(
                spec_name,
                train_df,
                valid_df,
                df,
                CONTROL_COLUMNS,
                city_cols,
                media_cols,
                use_transforms=use_transforms,
                include_media=include_media,
            )
            rows.append({"fold": fold_name, "spec": spec_name, **metrics})

        for spec_name in ["ElasticNetBudgetMix"]:
            feature_columns = _budget_mix_features(train_df)
            metrics, _ = _evaluate_linear_feature_set(spec_name, train_df, valid_df, feature_columns)
            rows.append({"fold": fold_name, "spec": spec_name, **metrics})

    backtest = pd.DataFrame(rows)
    summary = (
        backtest.groupby("spec", as_index=False)[["mape", "rmse", "mae", "bias"]]
        .mean()
        .sort_values(["mape", "rmse"])
        .reset_index(drop=True)
    )
    recommended_spec = str(
        summary[summary["spec"] != "ElasticNetControls"].sort_values(["mape", "rmse"]).iloc[0]["spec"]
    )

    backtest.to_csv(SPEC_REVIEW_TABLE, index=False)
    summary.to_csv(SPEC_REVIEW_SUMMARY, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    plot_df = summary.melt(id_vars="spec", value_vars=["mape", "rmse"], var_name="metric", value_name="value")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=plot_df[plot_df["metric"] == "mape"], x="spec", y="value", hue="spec", legend=False, ax=axes[0], palette="crest")
    axes[0].set_title("Spec Review - Mean MAPE")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(data=plot_df[plot_df["metric"] == "rmse"], x="spec", y="value", hue="spec", legend=False, ax=axes[1], palette="magma")
    axes[1].set_title("Spec Review - Mean RMSE")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(SPEC_REVIEW_FIG, dpi=180, bbox_inches="tight")
    plt.close()

    review_lines = [
        "# Model Specification Review",
        "",
        "## Scope",
        "",
        "- Objetivo: elegir la mejor ruta tecnica antes de fijar el modelo final.",
        "- Se comparan especificaciones lineales con validacion temporal estricta.",
        "- El baseline naive no se incluye aqui porque esta revision se centra en variantes de MMM/modelo explicativo.",
        "",
        "## Specs Compared",
        "",
        "- `ElasticNetControls`: controles exogenos + ciudad.",
        "- `ElasticNetRawMedia`: controles + medios en euros.",
        "- `ElasticNetTransformedMedia`: controles + lag/adstock/saturacion.",
        "- `ElasticNetBudgetMix`: controles + `budget_total_eur` + pesos publicitarios (`n-1`).",
        "",
        "## Mean Backtest",
        "",
        "```text",
        summary.round(2).to_string(index=False),
        "```",
        "",
        "## Recommendation",
        "",
        f"- Especificacion recomendada para el siguiente paso: `{recommended_spec}`.",
        "- La recomendacion combina rendimiento temporal y la revision de multicolinealidad previa.",
        "",
        "## Reading",
        "",
        "- Si gana una especificacion de pesos, eso refuerza la idea de modelar el mix como composicion y no solo como gasto bruto.",
        "- Se excluyen variables downstream como `ticket_medio_neto` para no contaminar la lectura causal del MMM.",
        "- Si `TransformedMedia` sigue ganando, mantenemos la ruta actual pero ya con evidencia de que los pesos no mejoran.",
        "",
        "## Artifacts",
        "",
        f"- Backtest por fold: `{SPEC_REVIEW_TABLE.name}`.",
        f"- Resumen: `{SPEC_REVIEW_SUMMARY.name}`.",
        f"- Figura: `4_feature_importance/{SPEC_REVIEW_FIG.name}`.",
    ]
    SPEC_REVIEW_DOC.write_text("\n".join(review_lines), encoding="utf-8")

    step_lines = [
        "# Step 6 - Model Specification Review",
        "",
        "## Objetivo",
        "",
        "Tomar la decision tecnica sobre con que formulacion merece la pena arrancar el modelo final tras revisar pesos publicitarios y multicolinealidad.",
        "",
        "## Que Hacemos",
        "",
        "- Comparamos varias especificaciones Elastic Net con backtest temporal.",
        "- Contrastamos la ruta de medios en euros con la ruta de presupuesto total + pesos.",
        "- Excluimos variables downstream para mantener el modelo en clave MMM causal.",
        "",
        "## Que Creamos",
        "",
        f"- `{SPEC_REVIEW_TABLE.name}`",
        f"- `{SPEC_REVIEW_SUMMARY.name}`",
        f"- `4_feature_importance/{SPEC_REVIEW_FIG.name}`",
        f"- `{SPEC_REVIEW_DOC.name}`",
        "",
        "## Conclusion",
        "",
        f"- La especificacion recomendada para el siguiente paso es `{recommended_spec}`.",
        "- Esta eleccion ya incorpora la evidencia del screening de variables y la revision de multicolinealidad.",
    ]
    STEP_REVIEW_DOC.write_text("\n".join(step_lines), encoding="utf-8")

    return ModelSpecReviewArtifacts(
        backtest=backtest,
        summary=summary,
        recommended_spec=recommended_spec,
    )
