from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.metrics import compute_metrics
from src.validation.visual_utils import savefig


def validation_required_benchmark_summary(benchmark_summary: pd.DataFrame) -> None:
    if benchmark_summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mape_df = benchmark_summary.melt(
        id_vars=["spec"],
        value_vars=["validation_mean_mape", "test_2024_mape"],
        var_name="sample",
        value_name="value",
    )
    rmse_df = benchmark_summary.melt(
        id_vars=["spec"],
        value_vars=["validation_mean_rmse", "test_2024_rmse"],
        var_name="sample",
        value_name="value",
    )
    mape_df["sample"] = mape_df["sample"].map(
        {
            "validation_mean_mape": "Validation mean",
            "test_2024_mape": "Test 2024",
        }
    )
    rmse_df["sample"] = rmse_df["sample"].map(
        {
            "validation_mean_rmse": "Validation mean",
            "test_2024_rmse": "Test 2024",
        }
    )

    sns.barplot(data=mape_df, x="spec", y="value", hue="sample", ax=axes[0], palette="crest")
    axes[0].set_title("Required Benchmarks: MAPE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MAPE")

    sns.barplot(data=rmse_df, x="spec", y="value", hue="sample", ax=axes[1], palette="mako")
    axes[1].set_title("Required Benchmarks: RMSE")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE")
    axes[1].legend_.remove()
    savefig("modeling", "validation_00_required_benchmarks.png")


def validation_required_benchmark_test_fit(benchmark_test_predictions: pd.DataFrame) -> None:
    if benchmark_test_predictions.empty:
        return

    ordered_specs = ["ARIMAX", "SeasonalNaive52", "ARIMABaselineNoExog", "NaiveLast", "MeanBaseline"]
    actual = (
        benchmark_test_predictions[benchmark_test_predictions["spec"] == "ARIMAX"][["semana_inicio", "ventas_netas"]]
        .drop_duplicates()
        .sort_values("semana_inicio")
    )
    plt.figure(figsize=(16, 7))
    plt.plot(actual["semana_inicio"], actual["ventas_netas"], label="Actual", color="#0f172a", linewidth=2.3)
    palette = sns.color_palette("Set2", n_colors=len(ordered_specs))
    for color, spec in zip(palette, ordered_specs):
        subset = benchmark_test_predictions[benchmark_test_predictions["spec"] == spec].sort_values("semana_inicio")
        if subset.empty:
            continue
        plt.plot(subset["semana_inicio"], subset["pred"], label=spec, linewidth=1.8, alpha=0.9, color=color)
    plt.title("Test 2024 Fit: ARIMAX vs Required Benchmarks")
    plt.legend(ncol=3, fontsize=9)
    savefig("modeling", "validation_00b_required_benchmark_test_fit.png")


def validation_backtest_metrics(backtest: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=backtest, x="fold", y="mape", hue="spec", ax=axes[0])
    axes[0].set_title("Backtest MAPE by Fold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)

    sns.barplot(data=backtest, x="fold", y="rmse", hue="spec", ax=axes[1])
    axes[1].set_title("Backtest RMSE by Fold")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend_.remove()
    savefig("modeling", "validation_01_backtest_metrics.png")


def validation_test_fit(predictions: pd.DataFrame) -> None:
    test = predictions[(predictions["fold"] == "test_2024")].copy()
    weekly = test.groupby("semana_inicio", as_index=False)[["ventas_netas", "pred"]].sum()
    weekly["residual"] = weekly["ventas_netas"] - weekly["pred"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(weekly["semana_inicio"], weekly["ventas_netas"], label="Actual", color="#1f4e79")
    axes[0].plot(weekly["semana_inicio"], weekly["pred"], label="Predicted", color="#d97904")
    axes[0].set_title("Test 2024: Actual vs Predicted")
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].bar(weekly["semana_inicio"], weekly["residual"], color="#7a1f5c")
    axes[1].set_title("Test 2024 Residuals Over Time")
    savefig("modeling", "validation_02_test_fit_and_residuals.png")


def validation_pred_vs_actual(predictions: pd.DataFrame) -> None:
    test = predictions[predictions["fold"] == "test_2024"].copy()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=test, x="ventas_netas", y="pred", alpha=0.7, s=45, color="#1f4e79")
    lims = [0, max(test["ventas_netas"].max(), test["pred"].max()) * 1.05]
    plt.plot(lims, lims, linestyle="--", color="black")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Predicted vs Actual Sales on Test 2024")
    savefig("modeling", "validation_04_predicted_vs_actual_scatter.png")


def validation_residual_distribution(predictions: pd.DataFrame) -> None:
    test = predictions[predictions["fold"] == "test_2024"].copy()
    residual = test["ventas_netas"] - test["pred"]
    plt.figure(figsize=(12, 6))
    sns.histplot(residual, bins=40, kde=True, color="#7a1f5c")
    plt.title("Residual Distribution on Test 2024")
    plt.xlabel("Residual")
    savefig("modeling", "validation_04b_residual_distribution.png")


def validation_coefficients(coefficients: pd.DataFrame) -> None:
    plot_df = coefficients.copy()
    plot_df["abs_coef"] = plot_df["coefficient"].abs()
    plot_df = plot_df.sort_values("abs_coef", ascending=False).head(15)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, y="feature", x="coefficient", hue="feature", palette="viridis", legend=False)
    plt.title("Top Deployment Coefficients by Magnitude")
    savefig("modeling", "validation_05_top_coefficients.png")


def validation_media_coefficients(coefficients: pd.DataFrame) -> None:
    plot_df = coefficients[
        coefficients["feature"].str.startswith(("media_", "budget_total_eur", "budget_share_pct_"))
    ].copy().sort_values("coefficient", ascending=False)
    if plot_df.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="feature", y="coefficient", hue="feature", palette="crest", legend=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("Media Pressure and Mix Coefficients")
    savefig("modeling", "validation_05b_media_coefficients.png")


def validation_contribution_timeline(contributions: pd.DataFrame) -> None:
    monthly = contributions.copy()
    monthly["month"] = monthly["semana_inicio"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly.groupby(["month", "channel"], as_index=False)["contribution"].sum()
    pivot = monthly.pivot(index="month", columns="channel", values="contribution").fillna(0.0)
    plt.figure(figsize=(14, 7))
    plt.stackplot(pivot.index, pivot.T.values, labels=[c.replace("media_", "") for c in pivot.columns], alpha=0.9)
    plt.title("Monthly Marketing Contribution by Channel")
    plt.legend(ncol=4, fontsize=9, loc="upper left")
    savefig("modeling", "validation_06_monthly_channel_contributions.png")


def validation_contribution_share(contributions: pd.DataFrame) -> None:
    plot_df = contributions.groupby("channel", as_index=False)["contribution"].sum().sort_values("contribution", ascending=False)
    plot_df["channel"] = plot_df["channel"].str.replace("media_", "", regex=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="channel", y="contribution", hue="channel", palette="viridis", legend=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("Total Contribution by Channel")
    savefig("modeling", "validation_06b_total_channel_contribution.png")


def validation_quarter_error(predictions: pd.DataFrame) -> None:
    test = predictions[predictions["fold"] == "test_2024"].copy()
    rows = []
    for quarter, group in test.groupby("quarter"):
        metrics = compute_metrics(group["ventas_netas"].to_numpy(), group["pred"].to_numpy())
        rows.append({"quarter": quarter, "mae": metrics["mae"], "mape": metrics["mape"]})
    plot_df = pd.DataFrame(rows).sort_values("quarter")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=plot_df, x="quarter", y="mae", hue="quarter", ax=axes[0], palette="Blues", legend=False)
    axes[0].set_title("MAE by Quarter")
    sns.barplot(data=plot_df, x="quarter", y="mape", hue="quarter", ax=axes[1], palette="Oranges", legend=False)
    axes[1].set_title("MAPE by Quarter")
    savefig("modeling", "validation_06c_quarter_error.png")


def validation_scenarios(scenarios: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    sns.barplot(data=scenarios, x="scenario", y="predicted_sales_2024", hue="scenario", ax=axes[0], palette="mako", legend=False)
    axes[0].set_title("Scenario Comparison: Predicted 2024 Sales")
    axes[0].tick_params(axis="x", rotation=20)

    budget_cols = [col for col in scenarios.columns if col.startswith("budget_")]
    long = scenarios.melt(id_vars=["scenario"], value_vars=budget_cols, var_name="channel", value_name="budget")
    long["channel"] = long["channel"].str.replace("budget_", "", regex=False)
    sns.barplot(data=long, x="channel", y="budget", hue="scenario", ax=axes[1])
    axes[1].set_title("Budget Allocation by Scenario")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(ncol=2, fontsize=8)
    savefig("modeling", "validation_07_scenarios_and_budget_mix.png")


def validation_mroi(mroi: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mroi, x="channel", y="mroi", hue="channel", palette="magma", legend=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("Marginal ROI by Channel")
    savefig("modeling", "validation_08_mroi_by_channel.png")


def generate_model_validation_figures(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    benchmark_test_predictions: pd.DataFrame,
    contributions: pd.DataFrame,
    scenarios: pd.DataFrame,
    coefficients: pd.DataFrame,
    mroi: pd.DataFrame,
) -> None:
    validation_required_benchmark_summary(benchmark_summary)
    validation_required_benchmark_test_fit(benchmark_test_predictions)
    validation_backtest_metrics(backtest)
    validation_test_fit(predictions)
    validation_pred_vs_actual(predictions)
    validation_residual_distribution(predictions)
    validation_coefficients(coefficients)
    validation_media_coefficients(coefficients)
    validation_contribution_timeline(contributions)
    validation_contribution_share(contributions)
    validation_quarter_error(predictions)
    validation_scenarios(scenarios)
    validation_mroi(mroi)
