from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from src.common.config import CONFIG
from src.common.parallel import parallel_kwargs
from src.simulation.optimizer import (
    BUDGET_TOTAL,
    annual_range_warnings,
    channel_budget_sensitivity,
    channel_budgets_from_shares,
    equal_share_budgets,
    load_model,
    marginal_roi,
    optimize_budget,
    predict_scenario,
    scenario_diagnostics_row,
    shift_budget,
)
from src.simulation.media_response import run_media_response_optimization
from src.simulation.reporting import write_reports, write_summary
from src.simulation.scenario_validation import run_scenario_validation


def _blend_budgets(
    base_budgets: dict[str, float],
    target_budgets: dict[str, float],
    blend_ratio: float,
) -> dict[str, float]:
    return {
        channel: float(base_budgets[channel] + blend_ratio * (target_budgets[channel] - base_budgets[channel]))
        for channel in base_budgets
    }


def run_budget_simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    package = load_model()
    media_cols = package.media_columns
    base_2024 = package.weekly_df[package.weekly_df["year"] == 2024][media_cols].sum()
    base_shares = (base_2024 / base_2024.sum()).to_numpy(dtype=float)
    base_budgets = channel_budgets_from_shares(base_shares, media_cols)
    equal_mix = equal_share_budgets(media_cols)

    mroi_df = marginal_roi(package, base_budgets)
    balanced_seed = shift_budget(base_budgets, mroi_df, total_shift_share=0.07, top_k=3, bottom_k=3)
    aggressive_seed = shift_budget(base_budgets, mroi_df, total_shift_share=0.12, top_k=3, bottom_k=3)

    balanced_shares = pd.Series(balanced_seed)[media_cols].to_numpy(dtype=float) / BUDGET_TOTAL
    aggressive_shares = pd.Series(aggressive_seed)[media_cols].to_numpy(dtype=float) / BUDGET_TOTAL

    optimized = optimize_budget(
        package,
        base_shares,
        extra_starts=[balanced_shares, aggressive_shares],
        max_relative_change=0.25,
        deviation_penalty=0.05,
        history_penalty=0.25,
        concentration_penalty=0.02,
    )
    conservative = _blend_budgets(base_budgets, optimized, blend_ratio=0.20)
    balanced = _blend_budgets(base_budgets, optimized, blend_ratio=0.40)
    guardrailed = _blend_budgets(base_budgets, optimized, blend_ratio=0.65)

    scenario_inputs = [
        ("historical_mix_12m", base_budgets),
        ("equal_mix_12m", equal_mix),
        ("conservative", conservative),
        ("balanced", balanced),
        ("guardrailed_optimized", guardrailed),
        ("aggressive_optimized", optimized),
    ]

    def _scenario_task(name: str, budgets: dict[str, float]) -> tuple[dict[str, float | str], dict[str, float | str]]:
        total_sales, total_profit, media_component = predict_scenario(package, budgets)
        warnings = annual_range_warnings(package, budgets, target_year=2024)
        row = {
            "scenario": name,
            "predicted_sales_2024": total_sales,
            "predicted_gross_profit_2024": total_profit,
            "predicted_media_component_2024": media_component,
            "warnings_outside_historical_range": warnings,
        }
        for channel, budget in budgets.items():
            row[f"budget_{channel.replace('media_', '')}"] = budget
        diagnostics_row = scenario_diagnostics_row(
            scenario_name=name,
            budgets=budgets,
            base_budgets=base_budgets,
            media_cols=media_cols,
            warnings=warnings,
        )
        return row, diagnostics_row

    scenario_results = Parallel(**parallel_kwargs(len(scenario_inputs), backend="threading"))(
        delayed(_scenario_task)(name, budgets) for name, budgets in scenario_inputs
    )
    scenarios = [scenario for scenario, _ in scenario_results]
    diagnostics = [diagnostic for _, diagnostic in scenario_results]

    scenarios_df = pd.DataFrame(scenarios).sort_values("predicted_gross_profit_2024", ascending=False).reset_index(drop=True)
    baseline = scenarios_df.loc[scenarios_df["scenario"] == "historical_mix_12m"].iloc[0]
    scenarios_df["delta_vs_historical_sales"] = scenarios_df["predicted_sales_2024"] - float(baseline["predicted_sales_2024"])
    scenarios_df["delta_vs_historical_profit"] = (
        scenarios_df["predicted_gross_profit_2024"] - float(baseline["predicted_gross_profit_2024"])
    )
    diagnostics_df = pd.DataFrame(diagnostics).merge(
        scenarios_df[["scenario", "predicted_gross_profit_2024", "delta_vs_historical_profit"]],
        on="scenario",
        how="left",
    ).sort_values("predicted_gross_profit_2024", ascending=False).reset_index(drop=True)
    sensitivity_df = channel_budget_sensitivity(package, base_budgets)

    scenarios_df.to_csv(CONFIG.scenario_results_file, index=False)
    mroi_df.to_csv(CONFIG.reports_tables_dir / "mroi_by_channel.csv", index=False)
    diagnostics_df.to_csv(CONFIG.reports_tables_dir / "scenario_diagnostics.csv", index=False)
    sensitivity_df.to_csv(CONFIG.reports_tables_dir / "channel_budget_sensitivity.csv", index=False)
    run_scenario_validation(
        package=package,
        base_budgets=base_budgets,
        optimized_budgets=guardrailed,
    )
    run_media_response_optimization(
        weekly_df=package.weekly_df,
        media_cols=media_cols,
        target_year=2024,
        planning_budget_eur=BUDGET_TOTAL,
    )
    write_reports(scenarios_df.round(2), mroi_df.round(4), diagnostics_df.round(2), sensitivity_df.round(2))
    write_summary(str(scenarios_df.iloc[0]["scenario"]))
    return scenarios_df, mroi_df
