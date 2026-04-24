from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import CONFIG


def plot_test_fit(test_scored: pd.DataFrame) -> None:
    weekly = test_scored.groupby("semana_inicio", as_index=False)[["ventas_netas", "pred"]].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(weekly["semana_inicio"], weekly["ventas_netas"], label="Actual")
    plt.plot(weekly["semana_inicio"], weekly["pred"], label="Predicted")
    plt.title("Test 2024: Actual vs Predicted Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG.reports_figures_dir / "test_actual_vs_predicted.png", dpi=160)
    plt.close()


def plot_channel_contributions(contributions: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    contrib = contributions.groupby("channel", as_index=False)["contribution"].sum().sort_values("contribution", ascending=False)
    plt.bar(contrib["channel"], contrib["contribution"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Channel Contribution (Deployment Model)")
    plt.tight_layout()
    plt.savefig(CONFIG.reports_figures_dir / "channel_contributions.png", dpi=160)
    plt.close()


def dataframe_code_block(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_validation_markdown(summary: pd.DataFrame, test_results: pd.DataFrame, winner: str) -> None:
    lines = [
        "# Validation Results",
        "",
        f"- Winning specification: `{winner}`",
        "",
        "## Backtesting Summary",
        "",
        dataframe_code_block(summary),
        "",
        "## Test 2024",
        "",
        dataframe_code_block(test_results),
        "",
        "## Reading",
        "",
        "- El criterio principal fue mejora frente a baselines sin perder plausibilidad de negocio.",
        "- Un spec con coeficientes de medios negativos queda penalizado o descartado.",
        "- La validación es temporal; no se usaron splits aleatorios.",
    ]
    CONFIG.validation_results_md.write_text("\n".join(lines), encoding="utf-8")


def write_model_review_markdown(winner: str, test_metrics: Dict[str, float], negative_media_coefficients: int) -> None:
    risk_lines = [
        "# Model Review Findings",
        "",
        f"- Especificación revisada: `{winner}`",
        f"- Coeficientes de medios negativos en el modelo ganador: `{negative_media_coefficients}`",
        f"- MAPE test 2024: `{test_metrics['mape']:.2f}`",
        "",
        "## Findings",
        "",
        "- No se detecta leakage temporal en la construcción de train/validation/test.",
        "- El riesgo metodológico principal sigue siendo la inconsistencia aparente entre escala de ventas e inversión.",
        "- Las variables de tráfico se excluyeron del spec final principal para no sobrecontrolar el embudo.",
        "- Las contribuciones deben leerse como señal relativa e incremental, no como verdad financiera absoluta.",
    ]
    CONFIG.model_review_md.write_text("\n".join(risk_lines), encoding="utf-8")


def write_code_review_markdown() -> None:
    lines = [
        "# Code Review Findings",
        "",
        "- El pipeline principal corre desde scripts reproducibles y no depende de notebooks.",
        "- Las transformaciones de medios están aisladas en módulos de `src/modeling`.",
        "- La validación temporal está separada en `src/validation/backtesting.py`.",
        "- El reporting y la escritura de artefactos están desacoplados del núcleo de entrenamiento.",
    ]
    CONFIG.code_review_md.write_text("\n".join(lines), encoding="utf-8")


def save_executive_summary(test_metrics: Dict[str, float], winner: str, contribution_table: pd.DataFrame) -> None:
    top_channels = contribution_table.groupby("channel")["contribution"].sum().sort_values(ascending=False).head(3)
    lines = [
        "# Executive Summary",
        "",
        f"El modelo ganador para K-Moda es `{winner}` con validación temporal estricta y un MAPE test 2024 de `{test_metrics['mape']:.2f}`.",
        "",
        "La recomendación de negocio se apoya en un MMM semanal global que separa ventas base de contribución incremental de medios.",
        "",
        "## Canales con mayor contribución estimada",
        "",
    ]
    for channel, contribution in top_channels.items():
        lines.append(f"- `{channel}`: {contribution:,.0f} EUR de contribución modelada en el periodo de despliegue.")
    lines += [
        "",
        "## Advertencias",
        "",
        "- La escala absoluta ventas/inversión del dataset no es económicamente literal.",
        "- La recomendación debe interpretarse en clave relativa: mejor mix y mejor uso marginal del presupuesto.",
    ]
    CONFIG.executive_summary_md.write_text("\n".join(lines), encoding="utf-8")
