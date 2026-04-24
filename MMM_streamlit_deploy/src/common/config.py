from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root: Path = Path(__file__).resolve().parents[2]
    raw_dir: Path = root / "data" / "raw" / "casomat_source"
    intermediate_dir: Path = root / "data" / "intermediate"
    processed_dir: Path = root / "data" / "processed"
    reports_tables_dir: Path = root / "reports" / "tables"
    reports_figures_dir: Path = root / "reports" / "figures"
    docs_dir: Path = root / "docs"

    clients_file: Path = raw_dir / "CASOMAT_MM_01_CLIENTES.csv"
    products_file: Path = raw_dir / "CASOMAT_MM_02_PRODUCTOS.csv"
    calendar_file: Path = raw_dir / "CASOMAT_MM_03_CALENDARIO.csv"
    traffic_file: Path = raw_dir / "CASOMAT_MM_04_TRAFICO_DIARIO.csv"
    investment_file: Path = raw_dir / "CASOMAT_MM_05_INVERSION_MEDIOS.csv"
    orders_file: Path = raw_dir / "CASOMAT_MM_06_PEDIDOS.csv"
    sales_lines_file: Path = raw_dir / "CASOMAT_MM_07_VENTAS_LINEAS.csv"

    weekly_sales_file: Path = intermediate_dir / "weekly_sales.parquet"
    weekly_orders_file: Path = intermediate_dir / "weekly_orders.parquet"
    weekly_calendar_file: Path = intermediate_dir / "weekly_calendar.parquet"
    weekly_traffic_file: Path = intermediate_dir / "weekly_traffic.parquet"
    weekly_media_file: Path = intermediate_dir / "weekly_media_wide.parquet"
    geo_weekly_sales_file: Path = intermediate_dir / "geo_weekly_sales.parquet"
    geo_weekly_orders_file: Path = intermediate_dir / "geo_weekly_orders.parquet"
    geo_weekly_calendar_file: Path = intermediate_dir / "geo_weekly_calendar.parquet"
    geo_weekly_traffic_file: Path = intermediate_dir / "geo_weekly_traffic.parquet"
    geo_weekly_media_file: Path = intermediate_dir / "geo_weekly_media_wide.parquet"
    model_dataset_file: Path = processed_dir / "model_dataset.parquet"
    diagnostic_dataset_file: Path = processed_dir / "model_dataset_diagnostic.parquet"
    geo_model_dataset_file: Path = processed_dir / "model_dataset_geo.parquet"
    geo_diagnostic_dataset_file: Path = processed_dir / "model_dataset_geo_diagnostic.parquet"
    data_checks_file: Path = processed_dir / "data_checks.json"
    geo_data_checks_file: Path = processed_dir / "geo_data_checks.json"
    model_results_file: Path = processed_dir / "model_results.json"
    predictive_model_results_file: Path = processed_dir / "predictive_model_results.json"
    hierarchical_model_results_file: Path = processed_dir / "hierarchical_mmm_results.json"
    constrained_model_results_file: Path = processed_dir / "constrained_mmm_results.json"
    contributions_file: Path = processed_dir / "channel_contributions.csv"
    weekly_predictions_file: Path = processed_dir / "weekly_predictions.csv"
    predictive_weekly_predictions_file: Path = processed_dir / "predictive_weekly_predictions.csv"
    hierarchical_weekly_predictions_file: Path = processed_dir / "hierarchical_mmm_weekly_predictions.csv"
    hierarchical_contributions_file: Path = processed_dir / "hierarchical_mmm_channel_contributions.csv"
    hierarchical_baseline_file: Path = processed_dir / "hierarchical_mmm_baseline.csv"
    hierarchical_inference_file: Path = processed_dir / "hierarchical_mmm_inference.nc"
    hierarchical_prior_config_file: Path = processed_dir / "hierarchical_mmm_prior_config.json"
    constrained_weekly_predictions_file: Path = processed_dir / "constrained_mmm_weekly_predictions.csv"
    constrained_contributions_file: Path = processed_dir / "constrained_mmm_channel_contributions.csv"
    constrained_baseline_file: Path = processed_dir / "constrained_mmm_baseline.csv"
    constrained_weekly_summary_file: Path = processed_dir / "constrained_mmm_weekly_summary.csv"
    constrained_cv_diagnostics_file: Path = processed_dir / "constrained_mmm_cv_diagnostics.csv"
    scenario_results_file: Path = processed_dir / "scenario_results.csv"
    backtest_results_file: Path = processed_dir / "backtest_results.csv"
    predictive_backtest_results_file: Path = processed_dir / "predictive_backtest_results.csv"
    final_model_file: Path = processed_dir / "final_model.pkl"
    predictive_final_model_file: Path = processed_dir / "predictive_model.pkl"
    selected_transforms_file: Path = processed_dir / "selected_transforms.json"
    validation_results_md: Path = docs_dir / "validation_results.md"
    predictive_validation_results_md: Path = docs_dir / "predictive_validation_results.md"
    model_review_md: Path = docs_dir / "model_review_findings.md"
    predictive_model_report_md: Path = docs_dir / "predictive_model_report.md"
    code_review_md: Path = docs_dir / "code_review_findings.md"
    executive_summary_md: Path = root / "executive_summary.md"
    scenario_results_md: Path = root / "scenario_results.md"
    slides_outline_md: Path = root / "slides_outline.md"


CONFIG = ProjectConfig()
