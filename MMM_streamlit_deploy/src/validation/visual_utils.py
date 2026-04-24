from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.config import CONFIG


FIG_DIR = CONFIG.reports_figures_dir
FIG_SUBDIRS = {
    "eda": FIG_DIR / "1_eda",
    "preprocessing": FIG_DIR / "2_preprocessing",
    "feature_engineering": FIG_DIR / "3_feature_engineering",
    "feature_importance": FIG_DIR / "4_feature_importance",
    "modeling": FIG_DIR / "5_modelado",
    "validation": FIG_DIR / "5_modelado",
    "business": FIG_DIR / "5_modelado",
    "simulation": FIG_DIR / "6_simulacion",
}

APP_BG = "#fdfbf8"
PANEL_BG = "#f8f4ee"
TEXT_PRIMARY = "#143d59"
TEXT_MUTED = "#5f7384"
GRID = "#dbe6ec"
BORDER = "#d4e0e7"
SLATE = "#7a8c99"
NAVY = "#143d59"
BLUE = "#2a6f97"
TEAL = "#4f8f8a"
GREEN = "#7aa95c"
LIGHT_BLUE = "#9cbcd0"
SAND = "#d8d2c8"
DANGER = "#b35c44"
SEQUENTIAL = [NAVY, BLUE, TEAL, GREEN]
DIVERGING = sns.diverging_palette(220, 140, s=70, l=45, center="light", as_cmap=True)
SEQUENTIAL_CMAP = sns.blend_palette([LIGHT_BLUE, BLUE, TEAL, GREEN], as_cmap=True)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIG_SUBDIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    CONFIG.docs_dir.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.facecolor": APP_BG,
            "axes.facecolor": PANEL_BG,
            "savefig.facecolor": APP_BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_PRIMARY,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
        },
    )
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True


def style_axes(axes, grid_axis: str = "y") -> None:
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        if ax is None:
            continue
        ax.set_facecolor(PANEL_BG)
        ax.grid(axis=grid_axis, color=GRID, alpha=0.75, linewidth=0.9)
        if grid_axis != "both":
            other_axis = "x" if grid_axis == "y" else "y"
            ax.grid(axis=other_axis, alpha=0.0)
        ax.tick_params(colors=TEXT_MUTED, labelsize=10.5)
        ax.title.set_color(TEXT_PRIMARY)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_color(TEXT_PRIMARY)
        ax.yaxis.label.set_color(TEXT_PRIMARY)
        for side in ["left", "bottom"]:
            spine = ax.spines.get(side)
            if spine is not None:
                spine.set_color(BORDER)
                spine.set_linewidth(1.0)
        for side in ["top", "right"]:
            spine = ax.spines.get(side)
            if spine is not None:
                spine.set_visible(False)


def style_figure(fig: mpl.figure.Figure) -> None:
    fig.patch.set_facecolor(APP_BG)
    fig.patch.set_edgecolor(APP_BG)


def savefig(section: str, name: str) -> None:
    style_figure(plt.gcf())
    plt.tight_layout()
    plt.savefig(FIG_SUBDIRS[section] / name, dpi=180, bbox_inches="tight", facecolor=APP_BG, edgecolor=APP_BG)
    plt.close()


def load_visual_artifacts():
    dataset = pd.read_parquet(CONFIG.diagnostic_dataset_file)
    dataset["semana_inicio"] = pd.to_datetime(dataset["semana_inicio"])
    predictions = pd.read_csv(CONFIG.weekly_predictions_file, parse_dates=["semana_inicio"])
    backtest = pd.read_csv(CONFIG.backtest_results_file)
    benchmark_summary = pd.read_csv(CONFIG.reports_tables_dir / "arimax_benchmark_summary.csv")
    benchmark_test_predictions = pd.read_csv(
        CONFIG.reports_tables_dir / "arimax_benchmark_test_predictions.csv",
        parse_dates=["semana_inicio"],
    )
    contributions = pd.read_csv(CONFIG.contributions_file, parse_dates=["semana_inicio"])
    scenarios = pd.read_csv(CONFIG.scenario_results_file)
    scenario_diagnostics = pd.read_csv(CONFIG.reports_tables_dir / "scenario_diagnostics.csv")
    channel_budget_sensitivity = pd.read_csv(CONFIG.reports_tables_dir / "channel_budget_sensitivity.csv")
    coefficients = pd.read_csv(CONFIG.reports_tables_dir / "deployment_coefficients.csv")
    mroi = pd.read_csv(CONFIG.reports_tables_dir / "mroi_by_channel.csv")
    scenario_validation_summary = pd.read_csv(CONFIG.reports_tables_dir / "scenario_validation_summary.csv")
    scenario_stat_tests = pd.read_csv(CONFIG.reports_tables_dir / "scenario_stat_tests.csv")
    regularization_fit = pd.read_csv(CONFIG.reports_tables_dir / "scenario_regularization_fit.csv")
    scenario_oos_validation = pd.read_csv(CONFIG.reports_tables_dir / "scenario_oos_validation.csv")
    scenario_oos_weight_stability = pd.read_csv(CONFIG.reports_tables_dir / "scenario_oos_weight_stability.csv")
    sales_lines = pd.read_csv(CONFIG.sales_lines_file, parse_dates=["fecha_venta"])
    orders = pd.read_csv(CONFIG.orders_file, parse_dates=["fecha_pedido"])
    clients = pd.read_csv(CONFIG.clients_file, parse_dates=["fecha_alta"])
    return (
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
    )
