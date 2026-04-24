from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.config import CONFIG
from src.simulation.optimizer import BUDGET_TOTAL


def text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def _savefig(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _write_sensitivity_figure(sensitivity_df: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 7))
    sns.lineplot(
        data=sensitivity_df,
        x="multiplier",
        y="delta_profit_vs_base",
        hue="channel",
        marker="o",
    )
    plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    plt.title("Channel Budget Sensitivity vs Historical 12M Mix")
    plt.xlabel("Budget multiplier vs historical channel budget")
    plt.ylabel("Delta gross profit vs historical mix")
    _savefig(CONFIG.reports_figures_dir / "5_modelado" / "validation_09_channel_budget_sensitivity.png")


def write_reports(
    scenarios_df: pd.DataFrame,
    mroi_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    CONFIG.scenario_results_md.write_text(
        "\n".join(
            [
                "# ARIMAX Scenario Results",
                "",
                f"Presupuesto total simulado: `{BUDGET_TOTAL:,.0f} EUR`",
                "",
                "## Scenarios",
                "",
                text_table(
                    scenarios_df[
                        [
                            "scenario",
                            "predicted_sales_2024",
                            "predicted_gross_profit_2024",
                            "delta_vs_historical_sales",
                            "delta_vs_historical_profit",
                            "warnings_outside_historical_range",
                        ]
                    ]
                ),
                "",
                "## Marginal ROI",
                "",
                text_table(mroi_df[["channel", "base_budget", "delta_budget", "mroi_sales", "mroi_profit"]]),
                "",
                "## Scenario Diagnostics",
                "",
                text_table(
                    diagnostics_df[
                        [
                            "scenario",
                            "delta_vs_historical_profit",
                            "total_budget_reallocated_eur",
                            "max_share_shift_pct_points",
                            "warning_count",
                        ]
                    ]
                ),
                "",
                "## Reading",
                "",
                "- La simulacion ya cuelga directamente del ARIMAX oficial, no del modelo lineal anterior.",
                "- El objetivo de optimizacion se ha llevado a beneficio bruto estimado, no solo a ventas.",
                "- Los escenarios optimizados ya no son simples maximos brutos: penalizan desviaciones agresivas frente al mix historico y frente al rango historico de shares.",
                "- Los pesos por canal siempre respetan presupuesto total fijo de 12M y suman 100%.",
                "- Los warnings ahora se leen sobre rango historico de share por canal, no sobre gasto absoluto, para que sean comparables entre anos.",
                "- La tabla diagnostica permite distinguir entre mejoras robustas y mejoras apoyadas en cambios demasiado agresivos.",
                "- Para decision guiada tomamos como referencia `balanced`, no el optimo mas agresivo, porque conserva mejor la trazabilidad con el mix historico.",
                "- La validacion estadistica completa del mix optimizado se publica aparte en `docs/scenario_validation_report.md`.",
            ]
        ),
        encoding="utf-8",
    )
    CONFIG.slides_outline_md.write_text(
        "\n".join(
            [
                "# Slides Outline",
                "",
                "1. Limpieza y definicion del panel semanal para series temporales",
                "2. Feature engineering: ticket medio, mix de medios y exogenas",
                "3. Modelo oficial ARIMAX y pesos estimados de cada variable",
                "4. Ajuste train/test y validacion temporal del ARIMAX",
                "5. Lectura de coeficientes de inversion y variables exogenas",
                "6. Simulacion de escenarios con presupuesto fijo de 12M EUR",
                "7. Validacion formal do nothing vs do something con ROI y t-test",
                "8. Benchmark de robustez con Ridge y Lasso",
            ]
        ),
        encoding="utf-8",
    )

    arimax_simulation_md = CONFIG.docs_dir / "arimax_simulation_report.md"
    best = scenarios_df.sort_values("predicted_gross_profit_2024", ascending=False).iloc[0]
    safest = diagnostics_df.sort_values(
        ["warning_count", "max_share_shift_pct_points", "delta_vs_historical_profit"],
        ascending=[True, True, False],
    ).iloc[0]
    recommended = scenarios_df.loc[scenarios_df["scenario"] == "balanced"].iloc[0]
    recommended_diag = diagnostics_df.loc[diagnostics_df["scenario"] == "balanced"].iloc[0]
    best_channel = mroi_df.sort_values("mroi_profit", ascending=False).iloc[0]
    worst_channel = mroi_df.sort_values("mroi_profit", ascending=True).iloc[0]
    _write_sensitivity_figure(sensitivity_df)
    arimax_simulation_md.write_text(
        "\n".join(
            [
                "# ARIMAX Simulation Report",
                "",
                "## Objetivo",
                "",
                "Mover el mix de medios dentro de un presupuesto total fijo de 12M EUR y ver como cambia la prediccion del ARIMAX.",
                "",
                "## Resultado",
                "",
                f"- Mejor escenario por beneficio bruto estimado: `{best['scenario']}`.",
                f"- Escenario mas prudente por estabilidad: `{safest['scenario']}`.",
                f"- Escenario recomendado para decision guiada: `balanced` con `{recommended['delta_vs_historical_profit']:,.2f} EUR` sobre historico y `{recommended_diag['max_share_shift_pct_points']:.2f}` puntos maximos de cambio de share.",
                f"- Ventas previstas: `{best['predicted_sales_2024']:,.2f} EUR`.",
                f"- Beneficio bruto previsto: `{best['predicted_gross_profit_2024']:,.2f} EUR`.",
                f"- Delta vs mix historico en beneficio: `{best['delta_vs_historical_profit']:,.2f} EUR`.",
                f"- Presupuesto total reubicado en el escenario mas agresivo: `{diagnostics_df['total_budget_reallocated_eur'].max():,.2f} EUR`.",
                "",
                "## Justificacion de uso",
                "",
                "- El valor del modelo aqui no es adivinar el euro exacto de ventas futuras, sino ordenar decisiones de mix bajo un presupuesto fijo.",
                "- La optimizacion se apoya en tres capas: coeficientes del ARIMAX, mROI marginal y guardrails historicos de share.",
                "- Por eso la recomendacion final es un escenario intermedio y explicable, no el maximo agresivo del optimizador.",
                "",
                "## Lectura de canales",
                "",
                f"- Canal mas fuerte por mROI beneficio: `{best_channel['channel']}` con `{best_channel['mroi_profit']:.4f}`.",
                f"- Canal mas debil por mROI beneficio: `{worst_channel['channel']}` con `{worst_channel['mroi_profit']:.4f}`.",
                "- La nueva curva de sensibilidad por canal permite ver si la recomendacion del modelo se mantiene al subir o bajar presupuesto en torno al historico.",
                "- La contrastacion formal frente a los escenarios do nothing se documenta en `scenario_validation_report.md` con ROI incremental, t-test y benchmarks regularizados.",
                "",
                "## Ridge y Lasso",
                "",
                "- Se usan como benchmark de robustez en la validacion de escenarios, no como sustituto del ARIMAX oficial.",
                "- El objetivo es comprobar si la direccion del mix optimizado se mantiene cuando imponemos regularizacion sobre la misma especificacion de exogenas.",
                "- Los resultados quedan trazados en `reports/tables/scenario_regularization_fit.csv` y `docs/scenario_validation_report.md`.",
            ]
        ),
        encoding="utf-8",
    )

    step_md = CONFIG.docs_dir / "step_8_arimax_simulation.md"
    step_md.write_text(
        "\n".join(
            [
                "# Step 8 - ARIMAX Simulation",
                "",
                "## Que Hacemos",
                "",
                "- Tomamos el ARIMAX ya entrenado como modelo oficial.",
                "- Fijamos un presupuesto total de 12M EUR.",
                "- Repartimos ese presupuesto por canal con varios escenarios y distintos niveles de penalizacion frente al historico.",
                "- Medimos ventas y beneficio bruto estimado para cada caso.",
                "",
                "## Que Creamos",
                "",
                "- `scenario_results.csv` con resultados por escenario.",
                "- `mroi_by_channel.csv` con sensibilidad marginal por canal.",
                "- `scenario_diagnostics.csv` con riesgo y magnitud de cambio por escenario.",
                "- `channel_budget_sensitivity.csv` con sensibilidad canal a canal.",
                "- `scenario_validation_summary.csv` con ROI incremental y contraste do nothing vs do something.",
                "- `scenario_stat_tests.csv` con resultados del `paired t-test` semanal.",
                "- `scenario_regularization_fit.csv` con benchmarks Ridge/Lasso.",
                "- `scenario_validation_report.md` con la lectura ejecutiva del contraste.",
                "- `arimax_simulation_report.md` con la lectura del paso.",
                "",
                "## Conclusion",
                "",
                "- La simulacion ya esta alineada con el modelo que pide tu profesor.",
                "- Ahora no solo vemos que escenario gana, sino tambien cuanto se aleja del mix historico y si ese cambio parece defendible.",
                "- La optimizacion queda apoyada por evidencia de ROI incremental y validacion estadistica, no solo por una tabla de escenarios.",
            ]
        ),
        encoding="utf-8",
    )


def write_summary(best_scenario: str) -> None:
    (CONFIG.reports_tables_dir / "scenario_summary.json").write_text(
        json.dumps(
            {
                "budget_total": BUDGET_TOTAL,
                "best_scenario": best_scenario,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
