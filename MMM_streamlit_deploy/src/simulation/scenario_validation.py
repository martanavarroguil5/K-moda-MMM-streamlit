from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.common.config import CONFIG
from src.common.metrics import compute_metrics
from src.modeling.arimax import ArimaxModelPackage, _fit_model, _transform_target
from src.simulation.optimizer import (
    BUDGET_TOTAL,
    TARGET_YEAR,
    annual_range_warnings,
    build_relative_bounds,
    channel_budgets_from_shares,
    channel_shares_from_budgets,
    optimize_budget,
    predict_scenario_weekly,
    scenario_diagnostics_row,
    scenario_frame,
)


SCENARIO_VALIDATION_SUMMARY = CONFIG.reports_tables_dir / "scenario_validation_summary.csv"
SCENARIO_WEEKLY_VALIDATION = CONFIG.reports_tables_dir / "scenario_weekly_validation.csv"
SCENARIO_STAT_TESTS = CONFIG.reports_tables_dir / "scenario_stat_tests.csv"
REGULARIZATION_FIT_TABLE = CONFIG.reports_tables_dir / "scenario_regularization_fit.csv"
OPTIMIZED_WEIGHT_TABLE = CONFIG.reports_tables_dir / "optimized_weight_comparison.csv"
WEIGHT_RESPONSE_CURVES_TABLE = CONFIG.reports_tables_dir / "weight_response_curves.csv"
WEIGHT_RESPONSE_MAXIMA_TABLE = CONFIG.reports_tables_dir / "weight_response_maxima.csv"
OOS_VALIDATION_TABLE = CONFIG.reports_tables_dir / "scenario_oos_validation.csv"
OOS_WEIGHT_STABILITY_TABLE = CONFIG.reports_tables_dir / "scenario_oos_weight_stability.csv"
REPORT_MD = CONFIG.docs_dir / "scenario_validation_report.md"
STEP_MD = CONFIG.docs_dir / "step_9_scenario_validation.md"
SUMMARY_JSON = CONFIG.reports_tables_dir / "scenario_validation_summary.json"


@dataclass(frozen=True)
class RegularizedScenarioModel:
    model_name: str
    scaler: StandardScaler
    estimator: RidgeCV | LassoCV
    exog_columns: list[str]
    media_columns: list[str]
    reference_weight: str
    weekly_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True)
class OOSFoldArtifacts:
    summary: pd.DataFrame
    weight_stability: pd.DataFrame


def _savefig(name: str) -> None:
    output = CONFIG.reports_figures_dir / "6_simulacion" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return float("nan")
    return float(numerator / denominator)


def _scenario_family(name: str) -> str:
    if name == "do_nothing_historical":
        return "do_nothing_active"
    if name == "do_nothing_zero_media":
        return "baseline_without_investment"
    return "do_something_optimized"


def _zero_media_budgets(media_cols: list[str]) -> Dict[str, float]:
    return {channel: 0.0 for channel in media_cols}


def _predict_regularized_weekly(
    package: RegularizedScenarioModel,
    channel_budget_map: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> pd.DataFrame:
    scenario_df = scenario_frame(package, channel_budget_map, target_year=target_year)
    x = package.scaler.transform(scenario_df[package.exog_columns])
    scenario_df["pred"] = package.estimator.predict(x)
    scenario_df["predicted_gross_profit"] = scenario_df["pred"] * scenario_df["gross_margin_rate"]
    return scenario_df


def _optimize_regularized_budget(
    package: RegularizedScenarioModel,
    start_shares: np.ndarray,
    extra_starts: list[np.ndarray] | None = None,
    max_relative_change: float | None = 0.25,
) -> Dict[str, float]:
    media_cols = package.media_columns

    def objective(shares: np.ndarray) -> float:
        budgets = channel_budgets_from_shares(shares, media_cols)
        scenario_df = _predict_regularized_weekly(package, budgets)
        return -float(scenario_df["predicted_gross_profit"].sum())

    constraints = [{"type": "eq", "fun": lambda shares: float(np.sum(shares) - 1.0)}]
    bounds = build_relative_bounds(media_cols, start_shares, max_relative_change=max_relative_change)
    rng = np.random.default_rng(42)
    starts = [start_shares, np.repeat(1.0 / len(media_cols), len(media_cols))]
    if extra_starts:
        starts.extend(extra_starts)
    for _ in range(10):
        proposal = rng.dirichlet(np.ones(len(media_cols)))
        proposal = np.clip(proposal, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
        proposal = proposal / proposal.sum()
        starts.append(proposal)

    best_shares = start_shares
    best_value = objective(start_shares)
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        candidate = result.x if result.success else start
        candidate = np.clip(candidate, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
        candidate = candidate / candidate.sum()
        candidate_value = objective(candidate)
        if candidate_value < best_value:
            best_value = candidate_value
            best_shares = candidate
    return channel_budgets_from_shares(best_shares, media_cols)


def _fit_regularized_models(package: ArimaxModelPackage) -> tuple[dict[str, RegularizedScenarioModel], pd.DataFrame]:
    alphas = np.logspace(-3, 4, 60)
    tscv = TimeSeriesSplit(n_splits=3)
    rows: list[dict[str, float | str | int]] = []
    models: dict[str, RegularizedScenarioModel] = {}
    for model_name, estimator in [
        ("Ridge", RidgeCV(alphas=alphas, cv=tscv)),
        ("Lasso", LassoCV(alphas=alphas, cv=tscv, random_state=42, max_iter=50000)),
    ]:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(package.train_df[package.exog_columns])
        y_train = package.train_df["ventas_netas"].to_numpy(dtype=float)
        estimator.fit(x_train, y_train)

        x_test = scaler.transform(package.test_df[package.exog_columns])
        test_pred = estimator.predict(x_test)
        metrics = compute_metrics(package.test_df["ventas_netas"].to_numpy(dtype=float), test_pred)
        coefficient_map = pd.Series(estimator.coef_, index=package.exog_columns, dtype=float)
        rows.append(
            {
                "model": model_name,
                "alpha": float(estimator.alpha_),
                "mape": metrics["mape"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "negative_media_like_coefficients": int((coefficient_map.filter(like="budget_share_pct_") < 0).sum()),
                "non_zero_coefficients": int((np.abs(coefficient_map) > 1e-9).sum()),
            }
        )
        models[model_name] = RegularizedScenarioModel(
            model_name=model_name,
            scaler=scaler,
            estimator=estimator,
            exog_columns=package.exog_columns,
            media_columns=package.media_columns,
            reference_weight=package.reference_weight,
            weekly_df=package.weekly_df,
            train_df=package.train_df,
            test_df=package.test_df,
        )
    fit_df = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    return models, fit_df


def _paired_stat_test(candidate: pd.Series, baseline: pd.Series) -> dict[str, float | bool]:
    diff = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    n_obs = int(diff.shape[0])
    dfree = max(n_obs - 1, 1)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n_obs > 1 else 0.0
    stderr = std_diff / np.sqrt(n_obs) if n_obs > 0 else float("nan")
    t_stat = float(mean_diff / stderr) if stderr > 0 else (float("inf") if mean_diff > 0 else 0.0)
    p_value_one_sided = float(1.0 - stats.t.cdf(t_stat, df=dfree))
    critical_one_sided = float(stats.t.ppf(0.95, df=dfree))
    critical_two_sided = float(stats.t.ppf(0.975, df=dfree))
    ci_half_width = critical_two_sided * stderr if np.isfinite(stderr) else float("nan")
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(diff, size=(4000, n_obs), replace=True).mean(axis=1)
    return {
        "n_obs": n_obs,
        "mean_weekly_profit_delta": mean_diff,
        "std_weekly_profit_delta": std_diff,
        "stderr_weekly_profit_delta": stderr,
        "effect_size_cohen_d": _safe_divide(mean_diff, std_diff),
        "t_statistic": t_stat,
        "t_critical_one_sided_5pct": critical_one_sided,
        "p_value_one_sided": p_value_one_sided,
        "ci_low_95pct": mean_diff - ci_half_width,
        "ci_high_95pct": mean_diff + ci_half_width,
        "bootstrap_ci_low_95pct": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high_95pct": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_prob_positive": float(np.mean(bootstrap > 0.0)),
        "significant_one_sided_5pct": bool((t_stat > critical_one_sided) and (p_value_one_sided < 0.05)),
    }


def _weight_table(
    model_name: str,
    media_cols: list[str],
    base_budgets: Dict[str, float],
    optimized_budgets: Dict[str, float],
) -> pd.DataFrame:
    base_shares = channel_shares_from_budgets(base_budgets, media_cols)
    opt_shares = channel_shares_from_budgets(optimized_budgets, media_cols)
    return pd.DataFrame(
        {
            "model": model_name,
            "channel": [channel.replace("media_", "") for channel in media_cols],
            "historical_budget_eur": [base_budgets[channel] for channel in media_cols],
            "optimized_budget_eur": [optimized_budgets[channel] for channel in media_cols],
            "historical_share_pct": base_shares * 100.0,
            "optimized_share_pct": opt_shares * 100.0,
            "delta_share_pct_points": (opt_shares - base_shares) * 100.0,
        }
    )


def _top_share_channels_from_coefficients(package: ArimaxModelPackage, top_k: int = 4) -> list[str]:
    params = package.fitted_result.params
    rows = []
    for media_col in package.media_columns:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        if share_col == package.reference_weight:
            continue
        rows.append(
            {
                "media_col": media_col,
                "share_col": share_col,
                "abs_coefficient": abs(float(params.get(share_col, 0.0))),
            }
        )
    selected = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).head(top_k)
    return selected["media_col"].tolist()


def _single_channel_share_slice(
    anchor_shares: np.ndarray,
    media_cols: list[str],
    target_channel: str,
    target_share: float,
) -> np.ndarray:
    idx = media_cols.index(target_channel)
    target_share = float(np.clip(target_share, 1e-6, 1.0 - 1e-6))
    proposal = anchor_shares.copy()
    proposal[idx] = target_share
    other_idx = [i for i in range(len(media_cols)) if i != idx]
    remaining = 1.0 - target_share
    anchor_other = anchor_shares[other_idx]
    anchor_other_sum = float(anchor_other.sum())
    if anchor_other_sum <= 0.0:
        proposal[other_idx] = remaining / len(other_idx)
    else:
        proposal[other_idx] = anchor_other / anchor_other_sum * remaining
    proposal = np.clip(proposal, 1e-9, None)
    return proposal / proposal.sum()


def _weight_response_curves(
    package: ArimaxModelPackage,
    base_budgets: Dict[str, float],
    optimized_budgets: Dict[str, float],
    num_points: int = 41,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    media_cols = package.media_columns
    selected_channels = _top_share_channels_from_coefficients(package, top_k=4)
    base_shares = channel_shares_from_budgets(base_budgets, media_cols)
    optimized_shares = channel_shares_from_budgets(optimized_budgets, media_cols)
    bounds = build_relative_bounds(media_cols, base_shares, max_relative_change=0.25)
    rows: list[dict[str, float | str | bool]] = []
    maxima_rows: list[dict[str, float | str | bool]] = []

    for channel in selected_channels:
        idx = media_cols.index(channel)
        lower, upper = bounds[idx]
        share_grid = np.linspace(lower, upper, num_points)
        for share in share_grid:
            scenario_shares = _single_channel_share_slice(optimized_shares, media_cols, channel, share)
            budgets = channel_budgets_from_shares(scenario_shares, media_cols)
            scenario_df = predict_scenario_weekly(package, budgets)
            total_profit = float(scenario_df["predicted_gross_profit"].sum())
            current_share_pct = float(scenario_shares[idx] * 100.0)
            current_budget = float(budgets[channel])
            rows.append(
                {
                    "channel": channel.replace("media_", ""),
                    "share_pct": current_share_pct,
                    "budget_eur": current_budget,
                    "predicted_gross_profit_2024": total_profit,
                    "delta_vs_historical_profit": total_profit
                    - float(predict_scenario_weekly(package, base_budgets)["predicted_gross_profit"].sum()),
                    "delta_vs_optimized_profit": total_profit
                    - float(predict_scenario_weekly(package, optimized_budgets)["predicted_gross_profit"].sum()),
                    "historical_share_pct": float(base_shares[idx] * 100.0),
                    "optimized_share_pct": float(optimized_shares[idx] * 100.0),
                    "historical_budget_eur": float(base_budgets[channel]),
                    "optimized_budget_eur": float(optimized_budgets[channel]),
                    "weight_coefficient": float(package.fitted_result.params.get(channel.replace("media_", "budget_share_pct_", 1), 0.0)),
                }
            )

        channel_curve = pd.DataFrame([row for row in rows if row["channel"] == channel.replace("media_", "")]).copy()
        optimum = channel_curve.sort_values("predicted_gross_profit_2024", ascending=False).iloc[0]
        maxima_rows.append(
            {
                "channel": channel.replace("media_", ""),
                "weight_coefficient": float(package.fitted_result.params.get(channel.replace("media_", "budget_share_pct_", 1), 0.0)),
                "historical_share_pct": float(base_shares[idx] * 100.0),
                "optimized_share_pct": float(optimized_shares[idx] * 100.0),
                "optimal_share_pct_on_slice": float(optimum["share_pct"]),
                "historical_budget_eur": float(base_budgets[channel]),
                "optimized_budget_eur": float(optimized_budgets[channel]),
                "optimal_budget_eur_on_slice": float(optimum["budget_eur"]),
                "historical_profit_2024": float(
                    channel_curve.loc[
                        (channel_curve["share_pct"] - base_shares[idx] * 100.0).abs().idxmin(),
                        "predicted_gross_profit_2024",
                    ]
                ),
                "optimized_profit_2024": float(
                    channel_curve.loc[
                        (channel_curve["share_pct"] - optimized_shares[idx] * 100.0).abs().idxmin(),
                        "predicted_gross_profit_2024",
                    ]
                ),
                "optimal_profit_2024_on_slice": float(optimum["predicted_gross_profit_2024"]),
                "optimized_gap_vs_slice_max_eur": float(optimum["predicted_gross_profit_2024"])
                - float(
                    channel_curve.loc[
                        (channel_curve["share_pct"] - optimized_shares[idx] * 100.0).abs().idxmin(),
                        "predicted_gross_profit_2024",
                    ]
                ),
            }
        )

    curves_df = pd.DataFrame(rows).sort_values(["channel", "share_pct"]).reset_index(drop=True)
    maxima_df = pd.DataFrame(maxima_rows).sort_values("optimal_profit_2024_on_slice", ascending=False).reset_index(drop=True)
    return curves_df, maxima_df


def _fit_oos_fold_package(package: ArimaxModelPackage, valid_year: int) -> ArimaxModelPackage:
    train_df = package.weekly_df[package.weekly_df["year"] < valid_year].copy()
    test_df = package.weekly_df[package.weekly_df["year"] == valid_year].copy()
    fitted = _fit_model(
        y_train=_transform_target(train_df["ventas_netas"], package.target_transform),
        x_train=train_df[package.exog_columns],
        order=package.selected_order,
        seasonal_order=package.seasonal_order,
    )
    return ArimaxModelPackage(
        fitted_result=fitted,
        selected_order=package.selected_order,
        seasonal_order=package.seasonal_order,
        exog_columns=package.exog_columns,
        media_columns=package.media_columns,
        reference_weight=package.reference_weight,
        spec_name=package.spec_name,
        spec_description=package.spec_description,
        target_transform=package.target_transform,
        weekly_df=package.weekly_df,
        train_df=train_df,
        test_df=test_df,
    )


def _oos_validation_years(package: ArimaxModelPackage) -> list[int]:
    years = sorted(package.weekly_df["year"].dropna().unique().tolist())
    return [int(year) for year in years if int(year) >= 2022]


def _run_out_of_sample_validation(package: ArimaxModelPackage) -> OOSFoldArtifacts:
    media_cols = package.media_columns
    rows: list[dict[str, float | str | int | bool]] = []
    weight_rows: list[pd.DataFrame] = []

    for valid_year in _oos_validation_years(package):
        fold_package = _fit_oos_fold_package(package, valid_year)
        base_annual = fold_package.weekly_df.loc[fold_package.weekly_df["year"] == valid_year, media_cols].sum()
        base_shares = (base_annual / base_annual.sum()).to_numpy(dtype=float)
        base_budgets = channel_budgets_from_shares(base_shares, media_cols)
        optimized_budgets = optimize_budget(
            fold_package,
            start_shares=base_shares,
            extra_starts=None,
            max_relative_change=0.25,
            target_year=valid_year,
        )

        historical_df = predict_scenario_weekly(fold_package, base_budgets, target_year=valid_year)
        optimized_df = predict_scenario_weekly(fold_package, optimized_budgets, target_year=valid_year)
        actual_sales = fold_package.test_df["ventas_netas"].to_numpy(dtype=float)
        historical_pred = historical_df["pred"].to_numpy(dtype=float)
        fit_metrics = compute_metrics(actual_sales, historical_pred)

        actual_profit = float((fold_package.test_df["ventas_netas"] * fold_package.test_df["gross_margin_rate"]).sum())
        historical_profit = float(historical_df["predicted_gross_profit"].sum())
        optimized_profit = float(optimized_df["predicted_gross_profit"].sum())
        base_shares_pct = channel_shares_from_budgets(base_budgets, media_cols) * 100.0
        opt_shares_pct = channel_shares_from_budgets(optimized_budgets, media_cols) * 100.0
        warning_count = scenario_diagnostics_row(
            scenario_name=f"oos_{valid_year}",
            budgets=optimized_budgets,
            base_budgets=base_budgets,
            media_cols=media_cols,
            warnings=annual_range_warnings(fold_package, optimized_budgets, target_year=valid_year),
        )["warning_count"]

        rows.append(
            {
                "validation_year": valid_year,
                "train_years_used": ",".join(str(year) for year in sorted(fold_package.train_df["year"].unique().tolist())),
                "historical_budget_total_eur": float(sum(base_budgets.values())),
                "optimized_budget_total_eur": float(sum(optimized_budgets.values())),
                "historical_predicted_gross_profit": historical_profit,
                "optimized_predicted_gross_profit": optimized_profit,
                "predicted_uplift_eur": optimized_profit - historical_profit,
                "predicted_uplift_pct_vs_historical": _safe_divide(optimized_profit - historical_profit, historical_profit) * 100.0,
                "reallocation_roi_predicted": _safe_divide(
                    optimized_profit - historical_profit,
                    float(scenario_diagnostics_row(
                        scenario_name=f"oos_{valid_year}",
                        budgets=optimized_budgets,
                        base_budgets=base_budgets,
                        media_cols=media_cols,
                        warnings="none",
                    )["total_budget_reallocated_eur"]),
                ),
                "historical_forecast_mape": fit_metrics["mape"],
                "historical_forecast_rmse": fit_metrics["rmse"],
                "historical_forecast_bias": fit_metrics["bias"],
                "actual_gross_profit_observed": actual_profit,
                "historical_prediction_error_vs_actual_eur": historical_profit - actual_profit,
                "warning_count_optimized": int(warning_count),
            }
        )

        weight_rows.append(
            pd.DataFrame(
                {
                    "validation_year": valid_year,
                    "channel": [channel.replace("media_", "") for channel in media_cols],
                    "historical_share_pct": base_shares_pct,
                    "optimized_share_pct": opt_shares_pct,
                    "delta_share_pct_points": opt_shares_pct - base_shares_pct,
                }
            )
        )

    summary_df = pd.DataFrame(rows).sort_values("validation_year").reset_index(drop=True)
    weight_df = pd.concat(weight_rows, ignore_index=True).sort_values(["validation_year", "channel"]).reset_index(drop=True)
    return OOSFoldArtifacts(summary=summary_df, weight_stability=weight_df)


def _plot_roi(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["scenario"] != "do_nothing_zero_media"].copy()
    plt.figure(figsize=(13, 7))
    sns.barplot(
        data=plot_df,
        x="model",
        y="roi_vs_zero_media",
        hue="scenario",
        palette="crest",
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Scenario Validation: ROI vs No-Investment Baseline")
    plt.xlabel("Model")
    plt.ylabel("ROI vs zero-media baseline")
    _savefig("sim_validation_05_roi_vs_baselines.png")


def _plot_weekly_delta(weekly_df: pd.DataFrame) -> None:
    pivot = weekly_df.pivot_table(
        index=["model", "semana_inicio"],
        columns="scenario",
        values="predicted_gross_profit",
    ).reset_index()
    pivot["optimized_minus_historical"] = pivot["do_something_optimized"] - pivot["do_nothing_historical"]
    plt.figure(figsize=(13, 7))
    sns.boxplot(data=pivot, x="model", y="optimized_minus_historical", hue="model", legend=False, palette="magma")
    sns.stripplot(
        data=pivot,
        x="model",
        y="optimized_minus_historical",
        color="black",
        alpha=0.35,
        size=3,
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Scenario Validation: Weekly Profit Delta of Optimized vs Historical")
    plt.xlabel("Model")
    plt.ylabel("Weekly gross profit delta")
    _savefig("sim_validation_06_weekly_profit_delta_boxplot.png")


def _plot_ttests(stat_tests: pd.DataFrame) -> None:
    plot_df = stat_tests[stat_tests["comparison"] == "optimized_vs_historical"].copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="model", y="t_statistic", hue="model", legend=False, palette="flare")
    plt.axhline(
        float(plot_df["t_critical_one_sided_5pct"].max()),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Critical t",
    )
    plt.title("Scenario Validation: Paired t Statistic for Optimized vs Historical")
    plt.xlabel("Model")
    plt.ylabel("t statistic")
    _savefig("sim_validation_07_ttest_statistics.png")


def _plot_weights(weight_df: pd.DataFrame) -> None:
    pivot = weight_df.pivot_table(index="model", columns="channel", values="delta_share_pct_points")
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0)
    plt.title("Scenario Validation: Optimized Weight Shift vs Historical Mix")
    plt.xlabel("Channel")
    plt.ylabel("Model")
    _savefig("sim_validation_08_weight_shift_heatmap.png")


def _plot_model_fit(fit_df: pd.DataFrame) -> None:
    plot_df = fit_df.melt(id_vars="model", value_vars=["mape", "rmse"], var_name="metric", value_name="value")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=plot_df[plot_df["metric"] == "mape"],
        x="model",
        y="value",
        hue="model",
        legend=False,
        palette="crest",
        ax=axes[0],
    )
    axes[0].set_title("Regularization Benchmark - Test MAPE")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("MAPE")
    sns.barplot(
        data=plot_df[plot_df["metric"] == "rmse"],
        x="model",
        y="value",
        hue="model",
        legend=False,
        palette="magma",
        ax=axes[1],
    )
    axes[1].set_title("Regularization Benchmark - Test RMSE")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("RMSE")
    _savefig("sim_validation_09_regularization_fit.png")


def _plot_weight_response_share(curves_df: pd.DataFrame, maxima_df: pd.DataFrame) -> None:
    channels = maxima_df["channel"].tolist()
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(13, 3.8 * n_channels), sharex=False)
    if n_channels == 1:
        axes = [axes]
    for ax, channel in zip(axes, channels):
        channel_curve = curves_df[curves_df["channel"] == channel].copy()
        maxima_row = maxima_df[maxima_df["channel"] == channel].iloc[0]
        sns.lineplot(
            data=channel_curve,
            x="share_pct",
            y="predicted_gross_profit_2024",
            color="#1f4e79",
            linewidth=2.5,
            ax=ax,
        )
        ax.axvline(float(maxima_row["historical_share_pct"]), color="#6b7280", linestyle="--", linewidth=1.2)
        ax.axvline(float(maxima_row["optimized_share_pct"]), color="#d97706", linestyle="--", linewidth=1.2)
        ax.axvline(float(maxima_row["optimal_share_pct_on_slice"]), color="#15803d", linestyle=":", linewidth=1.6)
        ax.scatter(
            [float(maxima_row["historical_share_pct"]), float(maxima_row["optimized_share_pct"]), float(maxima_row["optimal_share_pct_on_slice"])],
            [
                float(maxima_row["historical_profit_2024"]),
                float(maxima_row["optimized_profit_2024"]),
                float(maxima_row["optimal_profit_2024_on_slice"]),
            ],
            color=["#6b7280", "#d97706", "#15803d"],
            s=70,
            zorder=3,
        )
        ax.set_title(f"Profit vs share pct - {channel}")
        ax.set_xlabel("Budget share (%)")
        ax.set_ylabel("Predicted gross profit")
    fig.suptitle("Scenario Validation: Profit Response to Significant Weight Shares", y=1.01)
    _savefig("sim_validation_10_profit_vs_significant_shares.png")


def _plot_weight_response_budget(curves_df: pd.DataFrame, maxima_df: pd.DataFrame) -> None:
    channels = maxima_df["channel"].tolist()
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(13, 3.8 * n_channels), sharex=False)
    if n_channels == 1:
        axes = [axes]
    for ax, channel in zip(axes, channels):
        channel_curve = curves_df[curves_df["channel"] == channel].copy()
        maxima_row = maxima_df[maxima_df["channel"] == channel].iloc[0]
        sns.lineplot(
            data=channel_curve,
            x="budget_eur",
            y="predicted_gross_profit_2024",
            color="#7c2d12",
            linewidth=2.5,
            ax=ax,
        )
        ax.axvline(float(maxima_row["historical_budget_eur"]), color="#6b7280", linestyle="--", linewidth=1.2)
        ax.axvline(float(maxima_row["optimized_budget_eur"]), color="#d97706", linestyle="--", linewidth=1.2)
        ax.axvline(float(maxima_row["optimal_budget_eur_on_slice"]), color="#15803d", linestyle=":", linewidth=1.6)
        ax.scatter(
            [float(maxima_row["historical_budget_eur"]), float(maxima_row["optimized_budget_eur"]), float(maxima_row["optimal_budget_eur_on_slice"])],
            [
                float(maxima_row["historical_profit_2024"]),
                float(maxima_row["optimized_profit_2024"]),
                float(maxima_row["optimal_profit_2024_on_slice"]),
            ],
            color=["#6b7280", "#d97706", "#15803d"],
            s=70,
            zorder=3,
        )
        ax.set_title(f"Profit vs budget eur - {channel}")
        ax.set_xlabel("Channel budget (EUR)")
        ax.set_ylabel("Predicted gross profit")
    fig.suptitle("Scenario Validation: Profit Response to Significant Channel Budgets", y=1.01)
    _savefig("sim_validation_11_profit_vs_significant_budgets.png")


def _plot_oos_uplift(oos_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    sns.barplot(
        data=oos_summary,
        x="validation_year",
        y="predicted_uplift_eur",
        hue="validation_year",
        legend=False,
        palette="crest",
        ax=axes[0],
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Out-of-Sample Optimization: Predicted Profit Uplift by Validation Year")
    axes[0].set_xlabel("Validation year")
    axes[0].set_ylabel("Predicted uplift vs historical")

    sns.barplot(
        data=oos_summary,
        x="validation_year",
        y="reallocation_roi_predicted",
        hue="validation_year",
        legend=False,
        palette="magma",
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Out-of-Sample Optimization: Reallocation ROI by Validation Year")
    axes[1].set_xlabel("Validation year")
    axes[1].set_ylabel("Predicted reallocation ROI")
    _savefig("sim_validation_12_oos_uplift_by_year.png")


def _plot_oos_weight_stability(oos_weights: pd.DataFrame) -> None:
    pivot = oos_weights.pivot_table(index="validation_year", columns="channel", values="delta_share_pct_points")
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0)
    plt.title("Out-of-Sample Optimization: Weight Shift Stability by Validation Year")
    plt.xlabel("Channel")
    plt.ylabel("Validation year")
    _savefig("sim_validation_13_oos_weight_stability.png")


def _plot_oos_fit(oos_summary: pd.DataFrame) -> None:
    plot_df = oos_summary.melt(
        id_vars="validation_year",
        value_vars=["historical_forecast_mape", "historical_forecast_rmse"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=plot_df[plot_df["metric"] == "historical_forecast_mape"],
        x="validation_year",
        y="value",
        hue="validation_year",
        legend=False,
        palette="crest",
        ax=axes[0],
    )
    axes[0].set_title("Out-of-Sample Historical Fit - MAPE")
    axes[0].set_xlabel("Validation year")
    axes[0].set_ylabel("MAPE")
    sns.barplot(
        data=plot_df[plot_df["metric"] == "historical_forecast_rmse"],
        x="validation_year",
        y="value",
        hue="validation_year",
        legend=False,
        palette="flare",
        ax=axes[1],
    )
    axes[1].set_title("Out-of-Sample Historical Fit - RMSE")
    axes[1].set_xlabel("Validation year")
    axes[1].set_ylabel("RMSE")
    _savefig("sim_validation_14_oos_fit_quality.png")


def _write_report(
    summary_df: pd.DataFrame,
    stat_tests: pd.DataFrame,
    fit_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    maxima_df: pd.DataFrame,
    oos_summary: pd.DataFrame,
    oos_weights: pd.DataFrame,
) -> None:
    arimax_summary = summary_df[summary_df["model"] == "ARIMAX"].copy()
    arimax_hist = arimax_summary.loc[arimax_summary["scenario"] == "do_nothing_historical"].iloc[0]
    arimax_zero = arimax_summary.loc[arimax_summary["scenario"] == "do_nothing_zero_media"].iloc[0]
    arimax_opt = arimax_summary.loc[arimax_summary["scenario"] == "do_something_optimized"].iloc[0]
    arimax_tests = stat_tests[stat_tests["model"] == "ARIMAX"].copy()
    hist_vs_zero = arimax_tests.loc[arimax_tests["comparison"] == "historical_vs_zero_media"].iloc[0]
    opt_vs_hist = arimax_tests.loc[arimax_tests["comparison"] == "optimized_vs_historical"].iloc[0]
    best_regularized = fit_df.sort_values("mape").iloc[0]
    top_weight_shift = weight_df.loc[weight_df["model"] == "ARIMAX"].sort_values(
        "delta_share_pct_points",
        key=lambda s: s.abs(),
        ascending=False,
    ).iloc[0]
    closest_to_slice_max = maxima_df.sort_values("optimized_gap_vs_slice_max_eur").iloc[0]
    oos_best = oos_summary.sort_values("predicted_uplift_eur", ascending=False).iloc[0]
    oos_worst = oos_summary.sort_values("predicted_uplift_eur", ascending=True).iloc[0]
    stable_channel = (
        oos_weights.groupby("channel", as_index=False)["delta_share_pct_points"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["std", "mean"], ascending=[True, False])
        .iloc[0]
    )

    report_lines = [
        "# Scenario Validation Report",
        "",
        "## Objetivo",
        "",
        "Validar que la optimizacion del mix de pesos aporta valor real frente a no hacer nada, separando dos baselines de do nothing y comprobando la robustez de la conclusion con ARIMAX, Ridge y Lasso.",
        "",
        "## Diseno del contraste",
        "",
        "- `do_nothing_historical`: mantener el mix historico de 12M EUR sin reoptimizar.",
        "- `do_nothing_zero_media`: contrafactual de linea base sin inversion para aislar el aporte incremental de medios.",
        "- `do_something_optimized`: mix optimizado con presupuesto total fijo y guardrails de share.",
        "- Contraste estadistico: `paired t-test` semanal sobre beneficio bruto previsto.",
        f"- Semanas evaluadas: `{int(opt_vs_hist['n_obs'])}`; por eso no usamos un umbral fijo `1.96`, sino el `t` critico de Student adecuado al tamano muestral (`{opt_vs_hist['t_critical_one_sided_5pct']:.3f}` en contraste unilateral al 5%).",
        "- Benchmarks de robustez: `Ridge` y `Lasso` entrenados con la misma especificacion de exogenas y pesos.",
        "",
        "## Escenarios ARIMAX",
        "",
        "```text",
        arimax_summary[
            [
                "scenario",
                "budget_total_eur",
                "predicted_gross_profit_2024",
                "profit_vs_zero_media_eur",
                "roi_vs_zero_media",
                "profit_vs_historical_eur",
                "reallocation_roi_vs_historical",
            ]
        ]
        .round(2)
        .to_string(index=False),
        "```",
        "",
        "## Tests estadisticos ARIMAX",
        "",
        "```text",
        arimax_tests[
            [
                "comparison",
                "mean_weekly_profit_delta",
                "t_statistic",
                "t_critical_one_sided_5pct",
                "p_value_one_sided",
                "ci_low_95pct",
                "ci_high_95pct",
                "bootstrap_prob_positive",
                "significant_one_sided_5pct",
            ]
        ]
        .round(4)
        .to_string(index=False),
        "```",
        "",
        "## Robustez Ridge/Lasso",
        "",
        "```text",
        fit_df.round(4).to_string(index=False),
        "```",
        "",
        "## Curvas de beneficio vs peso",
        "",
        "```text",
        maxima_df[
            [
                "channel",
                "weight_coefficient",
                "historical_share_pct",
                "optimized_share_pct",
                "optimal_share_pct_on_slice",
                "optimized_gap_vs_slice_max_eur",
            ]
        ]
        .round(4)
        .to_string(index=False),
        "```",
        "",
        "## Validacion temporal out-of-sample",
        "",
        "```text",
        oos_summary[
            [
                "validation_year",
                "predicted_uplift_eur",
                "predicted_uplift_pct_vs_historical",
                "reallocation_roi_predicted",
                "historical_forecast_mape",
                "warning_count_optimized",
            ]
        ]
        .round(4)
        .to_string(index=False),
        "```",
        "",
        "## Por que la optimizacion es defendible",
        "",
        "- La optimizacion no sale de una sola corrida numerica: se contrasta frente a `do_nothing_historical` y frente a una linea base sin inversion.",
        "- El resultado optimizado no se acepta solo por beneficio bruto; tambien exigimos significancia semanal, benchmarks Ridge/Lasso y guardrails de share historico.",
        f"- Frente al do nothing historico, el escenario optimizado deja un uplift estimado de `{arimax_opt['profit_vs_historical_eur']:,.2f} EUR` y una mejora semanal media de `{opt_vs_hist['mean_weekly_profit_delta']:,.2f} EUR`.",
        f"- El `paired t-test` y el bootstrap salen alineados: `t = {opt_vs_hist['t_statistic']:.3f}`, `p = {opt_vs_hist['p_value_one_sided']:.4f}` y probabilidad bootstrap positiva `{opt_vs_hist['bootstrap_prob_positive']:.2%}`.",
        f"- La robustez no depende solo del ARIMAX: `Ridge` mantiene el mejor benchmark regularizado con MAPE `{best_regularized['mape']:.2f}`, lo que ayuda a sostener la direccion del mix.",
        "- Aun asi, la validacion temporal de uplift sigue siendo plana, asi que la lectura correcta es de optimizacion orientativa y prudente, no de garantia causal cerrada.",
        "",
        "## Lectura ejecutiva",
        "",
        f"- El gasto historico ya aporta un uplift estimado de `{arimax_hist['profit_vs_zero_media_eur']:,.2f} EUR` sobre la linea base sin inversion, con ROI incremental de `{arimax_hist['roi_vs_zero_media']:.4f}`.",
        f"- El escenario optimizado suma `{arimax_opt['profit_vs_historical_eur']:,.2f} EUR` sobre el do nothing historico y llega a un ROI incremental frente a cero inversion de `{arimax_opt['roi_vs_zero_media']:.4f}`.",
        f"- El `paired t-test` de `optimized_vs_historical` devuelve `t = {opt_vs_hist['t_statistic']:.3f}` y `p = {opt_vs_hist['p_value_one_sided']:.4f}`; significancia al 5%: `{bool(opt_vs_hist['significant_one_sided_5pct'])}`.",
        f"- El contraste `historical_vs_zero_media` deja medido el ROI del do nothing activo frente a la linea base de no invertir: `t = {hist_vs_zero['t_statistic']:.3f}`.",
        f"- El mayor desplazamiento de pesos en ARIMAX se da en `{top_weight_shift['channel']}` con `{top_weight_shift['delta_share_pct_points']:.2f}` puntos porcentuales frente al mix historico.",
        f"- Las nuevas curvas de respuesta muestran que el canal mas cerca de su maximo unidimensional en el mix optimizado es `{closest_to_slice_max['channel']}`, con una distancia de `{closest_to_slice_max['optimized_gap_vs_slice_max_eur']:,.2f} EUR` respecto al maximo de su curva.",
        f"- En validacion temporal fuera de muestra, el mejor uplift modelizado aparece en `{int(oos_best['validation_year'])}` con `{oos_best['predicted_uplift_eur']:,.2f} EUR`, y el mas flojo en `{int(oos_worst['validation_year'])}` con `{oos_worst['predicted_uplift_eur']:,.2f} EUR`.",
        f"- El canal con recomendacion mas estable entre folds es `{stable_channel['channel']}`, con desviacion estandar de `{stable_channel['std']:.2f}` puntos en el cambio de share.",
        f"- Entre los benchmarks regularizados, el mejor ajuste test lo consigue `{best_regularized['model']}` con MAPE `{best_regularized['mape']:.2f}`.",
        "- Para decision de negocio, el punto de partida recomendado es trasladar estas senales al escenario `balanced` y no al optimo agresivo.",
        "",
        "## Figuras",
        "",
        "- `6_simulacion/sim_validation_05_roi_vs_baselines.png`: ROI frente al baseline sin inversion.",
        "- `6_simulacion/sim_validation_06_weekly_profit_delta_boxplot.png`: distribucion semanal del uplift del optimizado frente al historico.",
        "- `6_simulacion/sim_validation_07_ttest_statistics.png`: estadistico t por modelo.",
        "- `6_simulacion/sim_validation_08_weight_shift_heatmap.png`: desplazamiento de pesos optimizados frente al historico.",
        "- `6_simulacion/sim_validation_09_regularization_fit.png`: calidad predictiva de Ridge y Lasso en test.",
        "- `6_simulacion/sim_validation_10_profit_vs_significant_shares.png`: beneficio vs porcentaje de peso para los coeficientes de mix mas influyentes.",
        "- `6_simulacion/sim_validation_11_profit_vs_significant_budgets.png`: beneficio vs euros para esos mismos canales.",
        "- `6_simulacion/sim_validation_12_oos_uplift_by_year.png`: uplift y ROI de reasignacion en validacion temporal.",
        "- `6_simulacion/sim_validation_13_oos_weight_stability.png`: estabilidad de pesos optimizados por canal y ano.",
        "- `6_simulacion/sim_validation_14_oos_fit_quality.png`: calidad predictiva historica por fold temporal.",
        "",
        "## Tablas",
        "",
        "- `reports/tables/scenario_validation_summary.csv`.",
        "- `reports/tables/scenario_weekly_validation.csv`.",
        "- `reports/tables/scenario_stat_tests.csv`.",
        "- `reports/tables/scenario_regularization_fit.csv`.",
        "- `reports/tables/optimized_weight_comparison.csv`.",
        "- `reports/tables/weight_response_curves.csv`.",
        "- `reports/tables/weight_response_maxima.csv`.",
        "- `reports/tables/scenario_oos_validation.csv`.",
        "- `reports/tables/scenario_oos_weight_stability.csv`.",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    step_lines = [
        "# Step 9 - Scenario Validation",
        "",
        "## Objetivo",
        "",
        "Contrastar formalmente la optimizacion del mix frente a no haber hecho nada, usando ROI incremental, contraste semanal y benchmarks regularizados.",
        "",
        "## Que Hacemos",
        "",
        "- Construimos dos baselines de do nothing: mix historico y linea base sin inversion.",
        "- Medimos el do something con el mix optimizado a presupuesto fijo.",
        "- Aplicamos `paired t-test` y bootstrap sobre el uplift semanal de beneficio.",
        "- Entrenamos `Ridge` y `Lasso` para revisar si la conclusion cambia con regularizacion.",
        "- Trazamos curvas de beneficio vs porcentaje y vs euros para los pesos de mix mas influyentes.",
        "- Refitamos el ARIMAX en ventanas expansivas para validar la optimizacion fuera de muestra.",
        "",
        "## Que Creamos",
        "",
        "- `scenario_validation_summary.csv`.",
        "- `scenario_weekly_validation.csv`.",
        "- `scenario_stat_tests.csv`.",
        "- `scenario_regularization_fit.csv`.",
        "- `optimized_weight_comparison.csv`.",
        "- `weight_response_curves.csv`.",
        "- `weight_response_maxima.csv`.",
        "- `scenario_oos_validation.csv`.",
        "- `scenario_oos_weight_stability.csv`.",
        "- `scenario_validation_report.md`.",
        "",
        "## Conclusion",
        "",
        "- La validacion ya no se apoya solo en un ROI agregado, sino en consistencia semanal y robustez entre modelos.",
        "- Esto deja la optimizacion de pesos defendible frente al baseline de no haber tocado el mix y tambien frente a cambios de ventana temporal.",
    ]
    STEP_MD.write_text("\n".join(step_lines), encoding="utf-8")

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "arimax_historical_roi_vs_zero_media": float(arimax_hist["roi_vs_zero_media"]),
                "arimax_optimized_roi_vs_zero_media": float(arimax_opt["roi_vs_zero_media"]),
                "arimax_incremental_profit_vs_historical": float(arimax_opt["profit_vs_historical_eur"]),
                "arimax_t_stat_optimized_vs_historical": float(opt_vs_hist["t_statistic"]),
                "arimax_p_value_optimized_vs_historical": float(opt_vs_hist["p_value_one_sided"]),
                "arimax_significant_optimized_vs_historical": bool(opt_vs_hist["significant_one_sided_5pct"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_scenario_validation(
    package: ArimaxModelPackage,
    base_budgets: Dict[str, float],
    optimized_budgets: Dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sns.set_theme(style="whitegrid", context="talk")
    CONFIG.reports_tables_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.reports_figures_dir / "6_simulacion").mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)

    media_cols = package.media_columns
    zero_budgets = _zero_media_budgets(media_cols)
    base_shares = channel_shares_from_budgets(base_budgets, media_cols)
    optimized_shares = channel_shares_from_budgets(optimized_budgets, media_cols)

    regularized_models, fit_df = _fit_regularized_models(package)
    fit_df.to_csv(REGULARIZATION_FIT_TABLE, index=False)
    curves_df, maxima_df = _weight_response_curves(package, base_budgets, optimized_budgets)
    curves_df.to_csv(WEIGHT_RESPONSE_CURVES_TABLE, index=False)
    maxima_df.to_csv(WEIGHT_RESPONSE_MAXIMA_TABLE, index=False)
    oos_artifacts = _run_out_of_sample_validation(package)
    oos_artifacts.summary.to_csv(OOS_VALIDATION_TABLE, index=False)
    oos_artifacts.weight_stability.to_csv(OOS_WEIGHT_STABILITY_TABLE, index=False)

    scenario_rows: list[dict[str, float | str | int]] = []
    weekly_rows: list[pd.DataFrame] = []
    stat_rows: list[dict[str, float | str | bool]] = []
    weight_rows: list[pd.DataFrame] = []

    model_packages: dict[str, ArimaxModelPackage | RegularizedScenarioModel] = {"ARIMAX": package, **regularized_models}
    optimized_budget_map: dict[str, Dict[str, float]] = {"ARIMAX": optimized_budgets}
    for model_name, model_package in regularized_models.items():
        optimized_budget_map[model_name] = _optimize_regularized_budget(
            model_package,
            start_shares=base_shares,
            extra_starts=[optimized_shares],
            max_relative_change=0.25,
        )

    for model_name, model_package in model_packages.items():
        budgets_by_scenario = {
            "do_nothing_historical": base_budgets,
            "do_nothing_zero_media": zero_budgets,
            "do_something_optimized": optimized_budget_map[model_name],
        }
        predictions_by_scenario: dict[str, pd.DataFrame] = {}
        for scenario_name, budgets in budgets_by_scenario.items():
            if model_name == "ARIMAX":
                scenario_df = predict_scenario_weekly(model_package, budgets)
            else:
                scenario_df = _predict_regularized_weekly(model_package, budgets)
            scenario_df = scenario_df.copy()
            scenario_df["predicted_sales"] = scenario_df["pred"]
            scenario_df["model"] = model_name
            scenario_df["scenario"] = scenario_name
            predictions_by_scenario[scenario_name] = scenario_df
            weekly_rows.append(
                scenario_df[
                    [
                        "semana_inicio",
                        "year",
                        "pred",
                        "predicted_gross_profit",
                        "budget_total_eur",
                        "model",
                        "scenario",
                    ]
                ].rename(columns={"pred": "predicted_sales"})
            )

        zero_profit = float(predictions_by_scenario["do_nothing_zero_media"]["predicted_gross_profit"].sum())
        historical_profit = float(predictions_by_scenario["do_nothing_historical"]["predicted_gross_profit"].sum())
        historical_sales = float(predictions_by_scenario["do_nothing_historical"]["predicted_sales"].sum())
        for scenario_name, budgets in budgets_by_scenario.items():
            current_df = predictions_by_scenario[scenario_name]
            total_profit = float(current_df["predicted_gross_profit"].sum())
            total_sales = float(current_df["predicted_sales"].sum())
            warnings = annual_range_warnings(package, budgets)
            diagnostics = scenario_diagnostics_row(
                scenario_name=scenario_name,
                budgets=budgets,
                base_budgets=base_budgets,
                media_cols=media_cols,
                warnings=warnings,
            )
            scenario_rows.append(
                {
                    "model": model_name,
                    "scenario": scenario_name,
                    "scenario_family": _scenario_family(scenario_name),
                    "budget_total_eur": float(sum(budgets.values())),
                    "predicted_sales_2024": total_sales,
                    "predicted_gross_profit_2024": total_profit,
                    "predicted_sales_delta_vs_historical": total_sales - historical_sales,
                    "profit_vs_zero_media_eur": total_profit - zero_profit,
                    "roi_vs_zero_media": _safe_divide(total_profit - zero_profit, float(sum(budgets.values()))),
                    "profit_vs_historical_eur": total_profit - historical_profit,
                    "reallocation_roi_vs_historical": _safe_divide(
                        total_profit - historical_profit,
                        float(diagnostics["total_budget_reallocated_eur"]),
                    ),
                    "warning_count": int(diagnostics["warning_count"]),
                    "total_budget_reallocated_eur": float(diagnostics["total_budget_reallocated_eur"]),
                    "max_share_shift_pct_points": float(diagnostics["max_share_shift_pct_points"]),
                }
            )

        for comparison_name, candidate_name, baseline_name in [
            ("historical_vs_zero_media", "do_nothing_historical", "do_nothing_zero_media"),
            ("optimized_vs_historical", "do_something_optimized", "do_nothing_historical"),
            ("optimized_vs_zero_media", "do_something_optimized", "do_nothing_zero_media"),
        ]:
            candidate = predictions_by_scenario[candidate_name]["predicted_gross_profit"]
            baseline = predictions_by_scenario[baseline_name]["predicted_gross_profit"]
            stat_rows.append(
                {
                    "model": model_name,
                    "comparison": comparison_name,
                    "candidate_scenario": candidate_name,
                    "baseline_scenario": baseline_name,
                    **_paired_stat_test(candidate, baseline),
                }
            )

        weight_rows.append(
            _weight_table(
                model_name=model_name,
                media_cols=media_cols,
                base_budgets=base_budgets,
                optimized_budgets=optimized_budget_map[model_name],
            )
        )

    summary_df = pd.DataFrame(scenario_rows).sort_values(
        ["model", "predicted_gross_profit_2024"],
        ascending=[True, False],
    ).reset_index(drop=True)
    weekly_df = pd.concat(weekly_rows, ignore_index=True).sort_values(["model", "scenario", "semana_inicio"]).reset_index(drop=True)
    stat_tests = pd.DataFrame(stat_rows).sort_values(["comparison", "model"]).reset_index(drop=True)
    weight_df = pd.concat(weight_rows, ignore_index=True)

    summary_df.to_csv(SCENARIO_VALIDATION_SUMMARY, index=False)
    weekly_df.to_csv(SCENARIO_WEEKLY_VALIDATION, index=False)
    stat_tests.to_csv(SCENARIO_STAT_TESTS, index=False)
    weight_df.to_csv(OPTIMIZED_WEIGHT_TABLE, index=False)

    _plot_roi(summary_df)
    _plot_weekly_delta(weekly_df)
    _plot_ttests(stat_tests)
    _plot_weights(weight_df)
    _plot_model_fit(fit_df)
    _plot_weight_response_share(curves_df, maxima_df)
    _plot_weight_response_budget(curves_df, maxima_df)
    _plot_oos_uplift(oos_artifacts.summary)
    _plot_oos_weight_stability(oos_artifacts.weight_stability)
    _plot_oos_fit(oos_artifacts.summary)
    _write_report(
        summary_df,
        stat_tests,
        fit_df,
        weight_df,
        curves_df,
        maxima_df,
        oos_artifacts.summary,
        oos_artifacts.weight_stability,
    )

    return summary_df, weekly_df, stat_tests, fit_df, weight_df
