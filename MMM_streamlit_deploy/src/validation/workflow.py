from __future__ import annotations

from src.validation.stage_reports import write_stage_reports
from src.validation.visual_eda import generate_eda_figures
from src.validation.visual_model import generate_model_validation_figures
from src.validation.visual_reporting import write_reports
from src.validation.visual_simulation import generate_simulation_figures
from src.validation.visual_utils import ensure_dirs, load_visual_artifacts, setup_style


def generate_visual_reports() -> None:
    ensure_dirs()
    setup_style()
    (
        dataset,
        predictions,
        backtest,
        benchmark_summary,
        benchmark_test_predictions,
        contributions,
        scenarios,
        scenario_diagnostics,
        channel_budget_sensitivity,
        coefficients,
        mroi,
        scenario_validation_summary,
        scenario_stat_tests,
        regularization_fit,
        scenario_oos_validation,
        scenario_oos_weight_stability,
        sales_lines,
        orders,
        clients,
    ) = load_visual_artifacts()
    generate_eda_figures(dataset, sales_lines, orders, clients)
    generate_model_validation_figures(
        predictions=predictions,
        backtest=backtest,
        benchmark_summary=benchmark_summary,
        benchmark_test_predictions=benchmark_test_predictions,
        contributions=contributions,
        scenarios=scenarios,
        coefficients=coefficients,
        mroi=mroi,
    )
    generate_simulation_figures(
        scenarios=scenarios,
        diagnostics=scenario_diagnostics,
        sensitivity=channel_budget_sensitivity,
        mroi=mroi,
    )
    write_reports(
        dataset=dataset,
        predictions=predictions,
        backtest=backtest,
        scenarios=scenarios,
        mroi=mroi,
        scenario_diagnostics=scenario_diagnostics,
        channel_budget_sensitivity=channel_budget_sensitivity,
        scenario_validation_summary=scenario_validation_summary,
        scenario_stat_tests=scenario_stat_tests,
        regularization_fit=regularization_fit,
        scenario_oos_validation=scenario_oos_validation,
        scenario_oos_weight_stability=scenario_oos_weight_stability,
        sales_lines=sales_lines,
        orders=orders,
        clients=clients,
    )
    write_stage_reports()
