from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.validation.visual_utils import BLUE, DANGER, DIVERGING, GREEN, LIGHT_BLUE, NAVY, PANEL_BG, SEQUENTIAL, SLATE, TEAL, savefig, style_axes


def simulation_scenario_performance(scenarios: pd.DataFrame) -> None:
    plot_df = scenarios.sort_values("predicted_gross_profit_2024", ascending=False).copy()
    label_map = {
        "aggressive_optimized": "Aggressive",
        "guardrailed_optimized": "Guardrailed",
        "balanced": "Balanced",
        "conservative": "Conservative",
        "historical_mix_12m": "Historical",
        "equal_mix_12m": "Equal mix",
    }
    plot_df["scenario_label"] = plot_df["scenario"].map(label_map).fillna(plot_df["scenario"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.9), gridspec_kw={"hspace": 0.58})
    sns.barplot(
        data=plot_df,
        x="scenario_label",
        y="predicted_gross_profit_2024",
        hue="scenario_label",
        palette=SEQUENTIAL + [LIGHT_BLUE, SLATE],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Simulation Data: Gross Profit by Scenario")
    axes[0].tick_params(axis="x", rotation=12, labelsize=11, pad=2)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Predicted gross profit")

    sns.barplot(
        data=plot_df,
        x="scenario_label",
        y="delta_vs_historical_profit",
        hue="scenario_label",
        palette=[GREEN, BLUE, TEAL, LIGHT_BLUE, SLATE, DANGER],
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(0.0, color=SLATE, linestyle="--", linewidth=1.0)
    axes[1].set_title("Simulation Data: Profit Delta vs Historical Mix", pad=8)
    axes[1].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Profit delta")
    style_axes(list(axes), grid_axis="y")
    savefig("simulation", "sim_data_01_scenario_performance.png")


def simulation_budget_mix(scenarios: pd.DataFrame) -> None:
    budget_cols = [col for col in scenarios.columns if col.startswith("budget_")]
    long_df = scenarios.melt(id_vars=["scenario"], value_vars=budget_cols, var_name="channel", value_name="budget")
    long_df["channel"] = long_df["channel"].str.replace("budget_", "", regex=False)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=long_df, x="channel", y="budget", hue="scenario", palette=SEQUENTIAL + [LIGHT_BLUE, SLATE])
    plt.title("Simulation Data: Budget Allocation by Scenario")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Budget")
    style_axes(plt.gca(), grid_axis="y")
    savefig("simulation", "sim_data_02_budget_mix.png")


def simulation_diagnostics(diagnostics: pd.DataFrame) -> None:
    plot_df = diagnostics.sort_values("delta_vs_historical_profit", ascending=False).copy()
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=plot_df,
        x="total_budget_reallocated_eur",
        y="delta_vs_historical_profit",
        size="warning_count",
        hue="scenario",
        palette=SEQUENTIAL + [LIGHT_BLUE, SLATE],
        sizes=(120, 380),
    )
    for _, row in plot_df.iterrows():
        plt.text(
            row["total_budget_reallocated_eur"],
            row["delta_vs_historical_profit"],
            row["scenario"],
            fontsize=9,
            ha="left",
            va="bottom",
        )
    plt.axhline(0.0, color=SLATE, linestyle="--", linewidth=1.0)
    plt.title("Simulation Validation: Profit Improvement vs Reallocation Intensity")
    plt.xlabel("Total budget reallocated")
    plt.ylabel("Profit delta vs historical")
    style_axes(plt.gca(), grid_axis="both")
    savefig("simulation", "sim_validation_01_profit_vs_reallocation.png")


def simulation_warning_counts(diagnostics: pd.DataFrame) -> None:
    plot_df = diagnostics.sort_values("warning_count", ascending=True).copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="scenario", y="warning_count", hue="scenario", palette=[LIGHT_BLUE, BLUE, TEAL, GREEN, SLATE, NAVY], legend=False)
    plt.title("Simulation Validation: Historical-Range Warnings by Scenario")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Warning count")
    style_axes(plt.gca(), grid_axis="y")
    savefig("simulation", "sim_validation_02_warning_counts.png")


def simulation_channel_sensitivity(sensitivity: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=sensitivity,
        x="multiplier",
        y="delta_profit_vs_base",
        hue="channel",
        palette=SEQUENTIAL + [LIGHT_BLUE, SLATE, NAVY, TEAL],
        marker="o",
    )
    plt.axhline(0.0, color=SLATE, linestyle="--", linewidth=1.0)
    plt.title("Simulation Validation: Profit Sensitivity by Channel")
    plt.xlabel("Channel budget multiplier vs historical mix")
    plt.ylabel("Profit delta vs historical mix")
    style_axes(plt.gca(), grid_axis="both")
    savefig("simulation", "sim_validation_03_channel_sensitivity.png")


def simulation_mroi(mroi: pd.DataFrame) -> None:
    plot_df = mroi.sort_values("mroi_profit", ascending=False).copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="channel", y="mroi_profit", hue="channel", palette=SEQUENTIAL + [LIGHT_BLUE, SLATE, NAVY, TEAL], legend=False)
    plt.axhline(0.0, color=SLATE, linestyle="--", linewidth=1.0)
    plt.title("Simulation Validation: mROI by Channel")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mROI profit")
    style_axes(plt.gca(), grid_axis="y")
    savefig("simulation", "sim_validation_04_mroi_profit.png")


def generate_simulation_figures(
    scenarios: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    mroi: pd.DataFrame,
) -> None:
    simulation_scenario_performance(scenarios)
    simulation_budget_mix(scenarios)
    simulation_diagnostics(diagnostics)
    simulation_warning_counts(diagnostics)
    simulation_channel_sensitivity(sensitivity)
    simulation_mroi(mroi)
