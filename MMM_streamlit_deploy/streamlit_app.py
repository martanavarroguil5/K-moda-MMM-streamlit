from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.common.config import CONFIG


st.set_page_config(page_title="K-Moda MMM", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid")

CHANNELS = ["display", "email_crm", "exterior", "paid_search", "prensa", "radio_local", "social_paid", "video_online"]
LABELS = {
    "display": "Display",
    "email_crm": "Email CRM",
    "exterior": "Exterior",
    "paid_search": "Paid Search",
    "prensa": "Prensa",
    "radio_local": "Radio Local",
    "social_paid": "Social Paid",
    "video_online": "Video Online",
}
REFERENCE = pd.DataFrame(
    [
        ["Sin inversion", 0.0, 130_822_763.74, 79_722_689.37, 0.0, np.nan],
        ["Historico 12M", 12_000_000.0, 194_962_324.02, 118_810_438.65, 39_087_749.28, 3.2573],
        ["Mix eficiente 8.19M", 8_187_309.76, 188_367_004.72, 114_791_223.02, 35_068_533.65, 4.2833],
    ],
    columns=["scenario", "budget_total_eur", "predicted_sales_2024", "predicted_gross_profit_2024", "incremental_profit_vs_zero_eur", "roi_vs_zero_media"],
)
ZERO_MEDIA_SALES = 130_822_763.74
ZERO_MEDIA_PROFIT = 79_722_689.37
EFFICIENT_819M = {
    "exterior": 938_419.94,
    "video_online": 1_225_537.45,
    "paid_search": 1_779_763.34,
    "radio_local": 917_647.06,
    "social_paid": 1_367_118.44,
    "email_crm": 811_764.71,
    "display": 847_058.82,
    "prensa": 300_000.00,
}


def defendable_12m_mix() -> dict[str, float]:
    return {
        "paid_search": 2_900_000.0,
        "social_paid": 2_050_000.0,
        "video_online": 2_050_000.0,
        "exterior": 1_350_000.0,
        "radio_local": 1_050_000.0,
        "display": 1_250_000.0,
        "email_crm": 1_000_000.0,
        "prensa": 350_000.0,
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #2a6f97;
            --primary-color-rgb: 42, 111, 151;
            --secondary-background-color: #eef5f8;
            --text-color: #143d59;
            accent-color: #2a6f97;
        }
        .stApp {
            --primary-color: #2a6f97;
            --primary-color-rgb: 42, 111, 151;
            accent-color: #2a6f97;
            background:
                radial-gradient(circle at top right, rgba(36,123,160,0.14), transparent 26%),
                radial-gradient(circle at top left, rgba(170,216,111,0.18), transparent 28%),
                linear-gradient(180deg, #f7f2ea 0%, #fdfbf8 38%, #eef5f8 100%);
        }
        .hero, .note {
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(20,61,89,0.08);
            box-shadow: 0 12px 28px rgba(20,61,89,0.06);
        }
        .summary-shell {
            position: sticky;
            top: 0.5rem;
            z-index: 20;
            padding-top: 0.1rem;
            background: linear-gradient(180deg, rgba(247,242,234,0.96) 0%, rgba(247,242,234,0.90) 72%, rgba(247,242,234,0.0) 100%);
        }
        .hero {
            background: linear-gradient(135deg, #143d59 0%, #2a6f97 52%, #7aa95c 100%);
            color: white;
        }
        .hero h1 { margin: 0; font-size: 2.25rem; }
        .hero p { margin: 0.6rem 0 0; line-height: 1.55; }
        .note { background: rgba(255,255,255,0.84); }
        .formula-card code {
            white-space: pre-wrap;
            line-height: 1.7;
            font-size: 0.98rem;
            color: #173f5f;
        }
        .formula-title {
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #5f7c91;
            margin-bottom: 0.45rem;
        }
        .formula-equation {
            font-size: 1.18rem;
            line-height: 1.6;
            color: #143d59;
            font-weight: 600;
            margin: 0.1rem 0 0.8rem;
        }
        .formula-foot {
            color: #516271;
            line-height: 1.55;
        }
        .formula-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            align-items: start;
            margin-bottom: 0.65rem;
        }
        .formula-panel {
            background: rgba(247, 250, 252, 0.9);
            border: 1px solid rgba(20,61,89,0.08);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        .formula-panel .formula-title {
            margin-bottom: 0.35rem;
        }
        .formula-panel code {
            font-size: 0.94rem;
        }
        .formula-math {
            font-family: "Cambria Math", "Times New Roman", serif;
            font-size: 1.04rem;
            line-height: 1.85;
            color: #143d59;
        }
        .formula-math .line {
            display: block;
        }
        .formula-math .indent {
            display: block;
            padding-left: 1.35rem;
        }
        .note-left {
            max-width: 86%;
            margin-right: auto;
        }
        .compact-note {
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
        }
        .compact-kpi .kpi {
            min-height: 120px;
            padding: 0.58rem 0.72rem;
            margin-bottom: 0.7rem;
        }
        .compact-kpi .kpi .v {
            font-size: clamp(1.14rem, 0.96vw, 1.34rem);
            min-height: 1.55em;
            margin: 0.1rem 0 0.06rem;
        }
        .compact-kpi .kpi .m {
            font-size: 0.86rem;
            line-height: 1.18;
        }
        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #143d59;
            margin: 0.85rem 0 0.65rem;
        }
        .journey-shell {
            margin: 2.25rem 0 1.1rem;
            padding: 1.3rem 1.15rem 0.15rem;
            border-top: 1px solid rgba(20,61,89,0.12);
            background:
                radial-gradient(circle at top left, rgba(42,111,151,0.10), transparent 34%),
                linear-gradient(180deg, rgba(255,255,255,0.44) 0%, rgba(255,255,255,0.0) 100%);
        }
        .journey-kicker {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #5f7c91;
            margin-bottom: 0.2rem;
        }
        .journey-title {
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.2;
            color: #143d59;
            margin-bottom: 0.95rem;
        }
        .kpi {
            background: rgba(255,255,255,0.92);
            border-radius: 18px;
            padding: 0.64rem 0.82rem;
            min-height: 142px;
            margin-bottom: 0.95rem;
            border: 1px solid rgba(20,61,89,0.08);
            box-shadow: 0 10px 22px rgba(20,61,89,0.05);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.02rem;
        }
        .kpi .l { color: #5b6c79; font-size: 0.82rem; text-transform: uppercase; }
        .kpi .v {
            color: #143d59;
            font-size: clamp(1.22rem, 1.02vw, 1.48rem);
            font-weight: 700;
            line-height: 1.16;
            margin: 0.14rem 0 0.1rem;
            min-height: 1.85em;
        }
        .kpi .m {
            color: #516271;
            font-size: 0.9rem;
            line-height: 1.26;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        div[data-baseweb="tab-list"] {
            display: flex;
            width: 100%;
            gap: 0.42rem;
            padding: 0.2rem 0 0.85rem 0;
            align-items: stretch;
        }
        button[data-baseweb="tab"] {
            background: rgba(20,61,89,0.07) !important;
            color: #143d59 !important;
            border-radius: 999px !important;
            border: 1px solid rgba(20,61,89,0.10) !important;
            padding: 0.5rem 0.8rem !important;
            min-height: 3.15rem !important;
            flex: 1 1 0 !important;
            justify-content: center !important;
            text-align: center !important;
        }
        button[data-baseweb="tab"] > div,
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {
            width: 100%;
            text-align: center !important;
            justify-content: center !important;
        }
        button[data-baseweb="tab"]:hover {
            background: rgba(42,111,151,0.14) !important;
            color: #143d59 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #143d59 0%, #2a6f97 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(20,61,89,0.18) !important;
            box-shadow: 0 10px 24px rgba(42,111,151,0.16) !important;
        }
        div[data-baseweb="tab-border"] {
            height: 3px !important;
            background: linear-gradient(90deg, rgba(20,61,89,0.10) 0%, rgba(42,111,151,0.16) 50%, rgba(122,169,92,0.12) 100%) !important;
        }
        div[data-baseweb="tab-highlight"] {
            height: 3px !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, #2a6f97 0%, #4f8f8a 58%, #7aa95c 100%) !important;
        }
        div[data-testid="stButton"] > button {
            min-height: 3.85rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.55) !important;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.00) 36%),
                linear-gradient(135deg, #143d59 0%, #2a6f97 56%, #7aa95c 100%) !important;
            color: #ffffff !important;
            font-weight: 900;
            letter-spacing: 0.02em;
            box-shadow: 0 18px 34px rgba(20,61,89,0.24) !important;
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease, border-color 140ms ease;
        }
        div[data-testid="stButton"] > button p {
            color: #ffffff !important;
            font-size: 0.98rem;
            font-weight: 900;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(255,255,255,0.86) !important;
            color: #ffffff !important;
            filter: brightness(1.08) saturate(1.05);
            transform: translateY(-2px);
            box-shadow: 0 22px 38px rgba(20,61,89,0.30) !important;
        }
        div[data-testid="stButton"] > button:active {
            transform: translateY(0);
            box-shadow: 0 10px 18px rgba(20,61,89,0.16) !important;
        }
        .preset-option {
            min-height: 174px;
            padding: 1rem 1.05rem;
            margin: 0 0 0.62rem;
            border-radius: 18px;
            border: 1px solid rgba(20,61,89,0.12);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(247,250,252,0.92) 100%);
            box-shadow: 0 14px 28px rgba(20,61,89,0.08);
        }
        .preset-option.historico {
            border-top: 5px solid #6a7886;
        }
        .preset-option.defendible {
            border-top: 5px solid #2a6f97;
        }
        .preset-option.eficiente {
            border-top: 5px solid #7aa95c;
        }
        .preset-kicker {
            color: #5f7c91;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }
        .preset-title {
            color: #143d59;
            font-size: 1.08rem;
            font-weight: 900;
            line-height: 1.22;
            margin-bottom: 0.38rem;
        }
        .preset-text {
            color: #516271;
            font-size: 0.92rem;
            line-height: 1.42;
            margin-bottom: 0.55rem;
        }
        .preset-total {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.64rem;
            background: rgba(42,111,151,0.10);
            color: #143d59;
            font-size: 0.84rem;
            font-weight: 800;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] div,
        [data-baseweb="slider"] div {
            border-color: rgba(42,111,151,0.28) !important;
        }
        div[data-testid="stSlider"],
        [data-baseweb="slider"] {
            --primary-color: #2a6f97 !important;
            --primary-color-rgb: 42, 111, 151 !important;
            accent-color: #2a6f97 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background: rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="#ff4b4b"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="255, 75, 75"],
        [data-baseweb="slider"] div[style*="background: rgb(255, 75, 75)"],
        [data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
        [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
        [data-baseweb="slider"] div[style*="#ff4b4b"],
        [data-baseweb="slider"] div[style*="255, 75, 75"] {
            background: linear-gradient(90deg, #2a6f97 0%, #3e7f93 100%) !important;
            border-color: rgba(42,111,151,0.52) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] [aria-valuenow],
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"],
        [data-baseweb="slider"] [aria-valuenow],
        [data-baseweb="slider"] [role="slider"] {
            background-color: #2a6f97 !important;
        }
        div[data-testid="stSlider"] [role="slider"],
        [role="slider"] {
            background: linear-gradient(180deg, #2a6f97 0%, #235e82 100%) !important;
            border: 3px solid #ffffff !important;
            box-shadow: 0 0 0 2px rgba(42,111,151,0.22), 0 8px 18px rgba(20,61,89,0.16) !important;
        }
        div[data-testid="stSlider"] [data-testid="stThumbValue"],
        [data-testid="stThumbValue"] {
            color: #1f5878 !important;
            font-weight: 800 !important;
            background: rgba(255,255,255,0.96) !important;
            border-radius: 999px !important;
            padding: 0.08rem 0.42rem !important;
            box-shadow: 0 8px 16px rgba(20,61,89,0.10) !important;
        }
        div[data-testid="stSlider"] [style*="rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [style*="255, 75, 75"],
        div[data-testid="stSlider"] [style*="#ff4b4b"],
        [data-baseweb="slider"] [style*="rgb(255, 75, 75)"],
        [data-baseweb="slider"] [style*="255, 75, 75"],
        [data-baseweb="slider"] [style*="#ff4b4b"],
        [data-testid="stThumbValue"][style*="rgb(255, 75, 75)"],
        [data-testid="stThumbValue"][style*="255, 75, 75"],
        [data-testid="stThumbValue"][style*="#ff4b4b"] {
            color: #1f5878 !important;
            border-color: rgba(42,111,151,0.52) !important;
        }
        div[data-testid="stSlider"] [style*="background: rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [style*="background-color: rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [style*="background: #ff4b4b"],
        div[data-testid="stSlider"] [style*="background-color: #ff4b4b"],
        [data-baseweb="slider"] [style*="background: rgb(255, 75, 75)"],
        [data-baseweb="slider"] [style*="background-color: rgb(255, 75, 75)"],
        [data-baseweb="slider"] [style*="background: #ff4b4b"],
        [data-baseweb="slider"] [style*="background-color: #ff4b4b"] {
            background: linear-gradient(90deg, #2a6f97 0%, #3e7f93 100%) !important;
        }
        [data-baseweb="slider"] > div > div:first-child {
            background: linear-gradient(90deg, rgba(125,145,161,0.34) 0%, rgba(125,145,161,0.24) 100%) !important;
        }
        [data-baseweb="slider"] > div > div:nth-child(2) {
            background: linear-gradient(90deg, #2a6f97 0%, #3e7f93 100%) !important;
        }
        [data-baseweb="slider"] * {
            accent-color: #2a6f97 !important;
        }
        div[data-testid="stSlider"] [style*="color: rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [style*="color: #ff4b4b"],
        [data-baseweb="slider"] [style*="color: rgb(255, 75, 75)"],
        [data-baseweb="slider"] [style*="color: #ff4b4b"] {
            color: #1f5878 !important;
        }
        div[data-testid="stSlider"] [style*="border-color: rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [style*="border-color: #ff4b4b"],
        [data-baseweb="slider"] [style*="border-color: rgb(255, 75, 75)"],
        [data-baseweb="slider"] [style*="border-color: #ff4b4b"] {
            border-color: rgba(42,111,151,0.52) !important;
        }
        * {
            scrollbar-color: #2a6f97 rgba(20,61,89,0.10);
        }
        *::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        *::-webkit-scrollbar-track {
            background: rgba(20,61,89,0.08);
            border-radius: 999px;
        }
        *::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #2a6f97 0%, #4f8f8a 58%, #7aa95c 100%);
            border: 2px solid rgba(247,242,234,0.88);
            border-radius: 999px;
        }
        *::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #143d59 0%, #2a6f97 58%, #7aa95c 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    return {
        "checks": read_json(CONFIG.data_checks_file),
        "model": read_json(CONFIG.model_results_file),
        "predictive": read_json(CONFIG.predictive_model_results_file),
        "pre": read_csv(CONFIG.reports_tables_dir / "preprocessing_dataset_summary.csv"),
        "fe": read_csv(CONFIG.reports_tables_dir / "feature_engineering_target_correlations.csv"),
        "fe_counts": read_csv(CONFIG.reports_tables_dir / "feature_engineering_feature_family_counts.csv"),
        "rf": read_csv(CONFIG.reports_tables_dir / "random_forest_feature_importance.csv"),
        "rf_specs": read_csv(CONFIG.reports_tables_dir / "random_forest_spec_summary.csv"),
        "bench": read_csv(CONFIG.reports_tables_dir / "arimax_benchmark_summary.csv"),
        "coef": read_csv(CONFIG.reports_tables_dir / "arimax_coefficients.csv"),
        "weights": read_csv(CONFIG.reports_tables_dir / "media_response_weights.csv"),
        "curves": read_csv(CONFIG.reports_tables_dir / "media_response_do_something_vs_nothing.csv"),
        "knees": read_csv(CONFIG.reports_tables_dir / "media_response_knee_points.csv"),
        "roi": read_csv(CONFIG.reports_tables_dir / "media_response_channel_roi.csv"),
    }


def compact_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M EUR"
    return f"{sign}{value:,.0f} EUR"


PRESET_STORIES = {
    "Historico 12M": (
        "Base",
        "Replica el reparto historico con 12M para comparar cualquier movimiento contra el punto de partida real.",
        "historico",
    ),
    "Mix defendible 12M": (
        "Recomendacion",
        "Mantiene el presupuesto completo y reordena canales con una lectura de negocio mas prudente.",
        "defendible",
    ),
    "Mix eficiente 8.19M": (
        "Eficiencia",
        "Reduce inversion y prioriza ROI, pensado para sostener rentabilidad con menos presupuesto.",
        "eficiente",
    ),
}


def preset_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "")


def render_preset_selector(
    presets: dict[str, dict[str, float]],
    session_prefix: str,
    key_prefix: str,
    button_suffix: str = "",
) -> None:
    preset_cols = st.columns(3)
    for col, preset_name in zip(preset_cols, presets):
        kicker, copy, css_class = PRESET_STORIES[preset_name]
        total = compact_money(sum(float(value) for value in presets[preset_name].values()))
        with col:
            st.markdown(
                f"""
                <div class='preset-option {css_class}'>
                    <div class='preset-kicker'>{kicker}</div>
                    <div class='preset-title'>{preset_name}</div>
                    <div class='preset-text'>{copy}</div>
                    <div class='preset-total'>{total}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Cargar {preset_name}{button_suffix}",
                key=f"{key_prefix}_{preset_slug(preset_name)}",
                width="stretch",
                type="primary",
            ):
                for ch, value in presets[preset_name].items():
                    st.session_state[f"{session_prefix}_{ch}"] = float(value)


def kpi(label: str, value: str, meta: str) -> str:
    return f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div><div class='m'>{meta}</div></div>"


def show_kpis(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(kpi(*item), unsafe_allow_html=True)


def show_figure(
    title: str,
    relative: str,
    explanation: str,
    figure_height: int | None = None,
    note_class: str = "",
) -> None:
    path = CONFIG.reports_figures_dir / relative
    container_kwargs = {"width": "stretch"}
    if figure_height is not None:
        container_kwargs["height"] = figure_height
        container_kwargs["border"] = False
    with st.container(**container_kwargs):
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
        if path.exists():
            st.image(str(path), width="stretch")
        else:
            st.warning(f"No se encontro {path.name}.")
    note_classes = "note" if not note_class else f"note {note_class}"
    st.markdown(f"<div class='{note_classes}'>{explanation}</div>", unsafe_allow_html=True)


def show_pyplot_figure(
    title: str,
    figure: plt.Figure,
    explanation: str,
    figure_height: int | None = None,
    note_class: str = "",
) -> None:
    container_kwargs = {"width": "stretch"}
    if figure_height is not None:
        container_kwargs["height"] = figure_height
        container_kwargs["border"] = False
    with st.container(**container_kwargs):
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
        st.pyplot(figure, width="stretch")
    note_classes = "note" if not note_class else f"note {note_class}"
    st.markdown(f"<div class='{note_classes}'>{explanation}</div>", unsafe_allow_html=True)


def media_response_formula_html(roi: pd.DataFrame) -> str:
    intercept = 2_726_691.6467
    ordered_channels = [
        "display",
        "email_crm",
        "exterior",
        "paid_search",
        "prensa",
        "radio_local",
        "social_paid",
        "video_online",
    ]
    roi_indexed = roi.set_index("channel") if not roi.empty else pd.DataFrame()
    total_weight = float(roi["coefficient"].sum()) if not roi.empty else 0.0

    raw_terms: list[str] = []
    pct_terms: list[str] = []
    for channel in ordered_channels:
        if roi.empty or channel not in roi_indexed.index:
            continue
        beta = float(roi_indexed.loc[channel, "coefficient"])
        pct = float(roi_indexed.loc[channel, "coefficient_pct"])
        raw_terms.append(f"+ {beta:,.4f} x<sub>{channel},t</sub>")
        pct_terms.append(f"+ {pct:,.2f}% x<sub>{channel},t</sub>")

    raw_lines = [
        f"<span class='line'>y<sub>t</sub> = {intercept:,.4f}</span>",
        f"<span class='indent'>{' '.join(raw_terms[:4])}</span>",
        f"<span class='indent'>{' '.join(raw_terms[4:])}</span>",
        "<span class='indent'>+ &sum;<sub>j</sub> &gamma;<sub>j</sub> X<sub>j,t</sub></span>",
    ]
    pct_lines = [
        f"<span class='line'>y<sub>t</sub> = {intercept:,.4f}</span>",
        f"<span class='indent'>+ {total_weight:,.4f} &middot; (</span>",
        f"<span class='indent'>{' '.join(pct_terms[:4])}</span>",
        f"<span class='indent'>{' '.join(pct_terms[4:])}</span>",
        "<span class='indent'>)</span>",
        "<span class='indent'>+ &sum;<sub>j</sub> &gamma;<sub>j</sub> X<sub>j,t</sub></span>",
    ]

    return (
        "<div class='formula-grid'>"
        "<div class='formula-panel'>"
        "<div class='formula-title'>Coeficientes Brutos</div>"
        f"<div class='formula-math'>{''.join(raw_lines)}</div>"
        "</div>"
        "<div class='formula-panel'>"
        "<div class='formula-title'>Bloque Normalizado En Porcentaje</div>"
        f"<div class='formula-math'>{''.join(pct_lines)}</div>"
        "</div>"
        "</div>"
        "<div class='formula-foot'>"
        "Notacion: <code>x<sub>c,t</sub></code> representa la variable transformada del canal <code>c</code> en la semana <code>t</code> "
        "(por ejemplo, <code>display_response_t</code> o <code>video_online_response_t</code>). "
        "El termino <code>&sum;<sub>j</sub> &gamma;<sub>j</sub> X<sub>j,t</sub></code> recoge el bloque de exogenas no-media "
        "(interacciones temporales, estacionalidad y calendario)."
        "</div>"
    )


def interp_delta(curves: pd.DataFrame, channel: str, budget: float) -> float:
    df = curves[curves["channel"] == channel].sort_values("budget_eur")
    if df.empty:
        return 0.0
    x = df["budget_eur"].to_numpy(float)
    y = df["delta_vs_do_nothing_profit"].to_numpy(float)
    return float(np.interp(np.clip(budget, x.min(), x.max()), x, y))


@st.cache_data(show_spinner=False)
def log_curve_profile(curves: pd.DataFrame, channel: str) -> dict[str, float]:
    df = curves[curves["channel"] == channel].sort_values("budget_eur")
    if df.empty or df["budget_eur"].nunique() < 3:
        return {}
    x = df["budget_eur"].to_numpy(float)
    y = df["delta_vs_do_nothing_profit"].to_numpy(float)
    positive = x[x > 0]
    scale = float(np.nanpercentile(positive, 70)) if len(positive) else 1.0
    scale = max(scale, 1.0)
    lx = np.log1p(np.maximum(x, 0.0) / scale)
    design = np.column_stack([lx, np.ones_like(lx)])
    beta, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    log_pred = beta * lx + intercept
    lin_beta, lin_intercept = np.polyfit(x, y, 1)
    lin_pred = lin_beta * x + lin_intercept
    rmse_log = float(np.sqrt(np.mean((y - log_pred) ** 2)))
    rmse_linear = float(np.sqrt(np.mean((y - lin_pred) ** 2)))
    return {
        "scale": scale,
        "beta": float(beta),
        "intercept": float(intercept),
        "rmse_log": rmse_log,
        "rmse_linear": rmse_linear,
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


def log_delta(curves: pd.DataFrame, channel: str, budget: float) -> float:
    profile = log_curve_profile(curves, channel)
    if not profile:
        return interp_delta(curves, channel, budget)
    scaled_budget = max(float(budget), 0.0) / profile["scale"]
    return float(profile["beta"] * np.log1p(scaled_budget) + profile["intercept"])


def adjustment_delta(curves: pd.DataFrame, channel: str, budget: float, log_channels: set[str] | None = None) -> float:
    return log_delta(curves, channel, budget) if log_channels and channel in log_channels else interp_delta(curves, channel, budget)


def adjustment_fit_chart(
    curves: pd.DataFrame,
    weights: pd.DataFrame,
    knees: pd.DataFrame,
    channel: str,
    current_budget: float,
) -> plt.Figure:
    df = curves[curves["channel"] == channel].sort_values("budget_eur")
    hist_budget = float(weights.loc[weights["channel"] == channel, "historical_budget_eur"].iloc[0])
    knee_budget = float(knees.loc[knees["channel"] == channel, "knee_budget_eur"].iloc[0])
    x_max = max(float(df["budget_eur"].max()), float(current_budget), hist_budget, knee_budget)
    x_grid = np.linspace(0.0, x_max, 220)
    y_log = np.array([log_delta(curves, channel, x) for x in x_grid])

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.plot(df["budget_eur"], df["delta_vs_do_nothing_profit"], color="#143d59", linewidth=2.3, label="Curva base")
    ax.scatter(df["budget_eur"], df["delta_vs_do_nothing_profit"], color="#143d59", s=20, alpha=0.35)
    ax.plot(x_grid, y_log, color="#4f8f8a", linewidth=2.3, linestyle="--", label="Ajuste logaritmico")
    ax.axvline(hist_budget, color="#7aa95c", linestyle=":", linewidth=2, label="Historico")
    ax.axvline(knee_budget, color="#7a8c99", linestyle="-.", linewidth=1.7, label="Knee point")
    ax.axvline(current_budget, color="#2a6f97", linestyle="-", linewidth=1.7, label="Plan actual")
    ax.set_title(f"{LABELS[channel]}: curva base vs ajuste logaritmico")
    ax.set_xlabel("Inversion (EUR)")
    ax.set_ylabel("Delta vs do nothing profit (EUR)")
    ax.legend(loc="best", frameon=True)
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    return fig


def log_formula_parts(curves: pd.DataFrame, channel: str) -> dict[str, float]:
    profile = log_curve_profile(curves, channel)
    if not profile:
        return {}
    df = curves[curves["channel"] == channel].sort_values("budget_eur")
    historical_budget = float(df["historical_budget_eur"].iloc[0]) if not df.empty else profile["scale"]
    base_a = max(2.0, 1.0 + (historical_budget / profile["scale"]))
    coef_b = float(profile["beta"] * np.log(base_a))
    return {
        "base_a": float(base_a),
        "coef_b": coef_b,
        "intercept": float(profile["intercept"]),
        "scale": float(profile["scale"]),
        "rmse_log": float(profile["rmse_log"]),
        "rmse_linear": float(profile["rmse_linear"]),
    }


def knee_status(budget: float, knee: float) -> str:
    if pd.isna(knee):
        return "sin dato"
    if knee <= 0:
        return "recorte sugerido"
    ratio = budget / knee
    if ratio < 0.9:
        return "por debajo del knee"
    if ratio <= 1.1:
        return "muy cerca del knee"
    return "por encima del knee"


def scenario_chart(reference_df: pd.DataFrame) -> plt.Figure:
    df = reference_df[reference_df["scenario"] != "Sin inversion"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    sns.barplot(data=df, x="scenario", y="predicted_gross_profit_2024", hue="scenario", legend=False, palette=["#9cbcd0", "#7aa95c", "#2a6f97"], ax=axes[0])
    sns.barplot(data=df, x="scenario", y="roi_vs_zero_media", hue="scenario", legend=False, palette=["#9cbcd0", "#7aa95c", "#2a6f97"], ax=axes[1])
    axes[0].set_title("Beneficio bruto esperado")
    axes[1].set_title("ROI incremental vs cero medios")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    return fig


def do_nothing_vs_proposals_chart(reference_df: pd.DataFrame) -> plt.Figure:
    order = ["Sin inversion", "Historico 12M", "Mix eficiente 8.19M", "Mix defendible 12M"]
    labels = {
        "Sin inversion": "Do nothing\nsin inversion",
        "Historico 12M": "Historico\n12M",
        "Mix eficiente 8.19M": "Do something\n8.19M",
        "Mix defendible 12M": "Do something\n12M",
    }
    colors = {
        "Sin inversion": "#b7c2cb",
        "Historico 12M": "#9cbcd0",
        "Mix eficiente 8.19M": "#7aa95c",
        "Mix defendible 12M": "#2a6f97",
    }
    df = reference_df.set_index("scenario").loc[order].reset_index().copy()
    df["Inversion"] = df["budget_total_eur"].astype(float) / 1_000_000
    df["Beneficio bruto"] = df["predicted_gross_profit_2024"].astype(float) / 1_000_000
    df["ROI incremental"] = pd.to_numeric(df["roi_vs_zero_media"], errors="coerce").fillna(0.0)
    df["Escenario"] = df["scenario"].map(labels)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    palette = [colors[scenario] for scenario in df["scenario"]]
    columns = ["Inversion", "Beneficio bruto", "ROI incremental"]
    titles = ["Dinero invertido", "Beneficio bruto esperado", "ROI incremental vs do nothing"]
    ylabels = ["M EUR", "M EUR", "ROI"]
    for ax, column, title, ylabel in zip(axes, columns, titles, ylabels):
        ax.bar(df["Escenario"], df[column], color=palette)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        for idx, value in enumerate(df[column]):
            if column == "ROI incremental" and df.loc[idx, "scenario"] == "Sin inversion":
                label = "n/a"
            elif column == "ROI incremental":
                label = f"{value:.2f}"
            else:
                label = f"{value:.1f}M"
            ax.text(idx, value, label, ha="center", va="bottom", fontsize=8.5, color="#143d59")
    axes[0].set_ylabel("M EUR")
    fig.suptitle("Comparativa final: do nothing, historico y propuestas", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def knee_chart(knees: pd.DataFrame) -> plt.Figure:
    plot = knees.copy()
    plot["Canal"] = plot["channel"].map(LABELS)
    plot["gap_vs_knee_eur"] = plot["historical_budget_eur"] - plot["knee_budget_eur"]
    plot = plot.sort_values("gap_vs_knee_eur", ascending=False)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = ["#2a6f97" if v > 0 else "#7aa95c" for v in plot["gap_vs_knee_eur"]]
    sns.barplot(data=plot, x="Canal", y="gap_vs_knee_eur", hue="Canal", legend=False, palette=colors, ax=ax)
    ax.axhline(0, color="#7a8c99", linestyle="--", linewidth=1)
    ax.set_title("Distancia del gasto historico frente al knee point")
    ax.set_xlabel("")
    ax.set_ylabel("Historico - knee (EUR)")
    ax.tick_params(axis="x", rotation=24)
    fig.tight_layout()
    return fig


def format_delta(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{compact_money(value)}"


def interp_curve_metric(curves: pd.DataFrame, channel: str, budget: float, column: str) -> float:
    df = curves[curves["channel"] == channel].sort_values("budget_eur")
    if df.empty:
        return 0.0
    x = df["budget_eur"].to_numpy(float)
    y = df[column].to_numpy(float)
    return float(np.interp(np.clip(budget, x.min(), x.max()), x, y))


def defendable_reference(curves: pd.DataFrame) -> dict[str, float | str]:
    mix = defendable_12m_mix()
    historical_sales = float(REFERENCE.loc[REFERENCE["scenario"] == "Historico 12M", "predicted_sales_2024"].iloc[0])
    historical_profit = float(REFERENCE.loc[REFERENCE["scenario"] == "Historico 12M", "predicted_gross_profit_2024"].iloc[0])
    predicted_sales = historical_sales
    predicted_profit = historical_profit
    for channel, budget in mix.items():
        predicted_sales += interp_curve_metric(curves, channel, budget, "predicted_sales_2024") - historical_sales
        predicted_profit += interp_curve_metric(curves, channel, budget, "predicted_gross_profit_2024") - historical_profit
    incremental_profit = predicted_profit - ZERO_MEDIA_PROFIT
    return {
        "scenario": "Mix defendible 12M",
        "budget_total_eur": float(sum(mix.values())),
        "predicted_sales_2024": predicted_sales,
        "predicted_gross_profit_2024": predicted_profit,
        "incremental_profit_vs_zero_eur": incremental_profit,
        "roi_vs_zero_media": incremental_profit / 12_000_000.0,
    }


def reference_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = REFERENCE.to_dict("records")
    rows.append(defendable_reference(curves))
    return pd.DataFrame(rows)


def current_change_table(weights: pd.DataFrame) -> pd.DataFrame:
    df = weights.copy()
    defendable_12m = defendable_12m_mix()
    efficient_total = float(sum(EFFICIENT_819M.values()))
    df["Canal"] = df["channel"].map(LABELS)
    df["optimized_budget_eur"] = df["channel"].map(defendable_12m).astype(float)
    df["optimized_share_pct"] = df["optimized_budget_eur"] / 12_000_000.0 * 100.0
    df["efficient_budget_eur"] = df["channel"].map(EFFICIENT_819M).astype(float)
    df["efficient_share_pct"] = df["efficient_budget_eur"] / efficient_total * 100.0
    df["delta_budget_eur"] = df["optimized_budget_eur"] - df["historical_budget_eur"]
    df["delta_share_pct"] = df["optimized_share_pct"] - df["historical_share_pct"]
    return df.sort_values("delta_budget_eur")


def budget_results_table(weights: pd.DataFrame) -> pd.DataFrame:
    table = current_change_table(weights).sort_values("historical_budget_eur", ascending=False).copy()
    table["Delta 12M vs historico"] = table["optimized_budget_eur"] - table["historical_budget_eur"]
    table["Delta 8.19M vs historico"] = table["efficient_budget_eur"] - table["historical_budget_eur"]
    table["Historico 12M"] = table["historical_budget_eur"].map(compact_money)
    table["Mix defendible 12M"] = table["optimized_budget_eur"].map(compact_money)
    table["Mix eficiente 8.19M"] = table["efficient_budget_eur"].map(compact_money)
    table["Delta 12M vs historico"] = table["Delta 12M vs historico"].map(format_delta)
    table["Delta 8.19M vs historico"] = table["Delta 8.19M vs historico"].map(format_delta)
    return table[
        [
            "Canal",
            "Historico 12M",
            "Mix defendible 12M",
            "Mix eficiente 8.19M",
            "Delta 12M vs historico",
            "Delta 8.19M vs historico",
        ]
    ]


def historical_mix_donut_chart(weights: pd.DataFrame) -> plt.Figure:
    plot = weights[["channel", "historical_share_pct"]].copy().sort_values("historical_share_pct", ascending=False)
    plot["Canal"] = plot["channel"].map(LABELS)
    colors = ["#143d59", "#4f8f8a", "#2a6f97", "#7aa95c", "#7a8c99", "#b7c2cb", "#d2dbe2", "#ddd8d0"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    wedges, _ = ax.pie(
        plot["historical_share_pct"],
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.34, "edgecolor": "#fdfbf8"},
    )
    ax.text(0, 0.06, "12M", ha="center", va="center", fontsize=24, fontweight="bold", color="#143d59")
    ax.text(0, -0.16, "mix historico", ha="center", va="center", fontsize=10.5, color="#6a7886")
    ax.legend(
        wedges,
        [f"{label}  {share:.1f}%" for label, share in zip(plot["Canal"], plot["historical_share_pct"])],
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        handletextpad=0.55,
        borderpad=0.2,
        labelcolor="#334455",
    )
    fig.tight_layout()
    return fig


def defendable_mix_donut_chart(weights: pd.DataFrame) -> plt.Figure:
    plot = current_change_table(weights)[["channel", "optimized_share_pct"]].copy().sort_values("optimized_share_pct", ascending=False)
    plot["Canal"] = plot["channel"].map(LABELS)
    colors = ["#143d59", "#4f8f8a", "#2a6f97", "#7aa95c", "#7a8c99", "#b7c2cb", "#d2dbe2", "#ddd8d0"]

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(7.6, 5.0),
        gridspec_kw={"width_ratios": [1.35, 0.9]},
    )
    wedges, _ = ax.pie(
        plot["optimized_share_pct"],
        startangle=90,
        counterclock=False,
        colors=colors,
        radius=1.02,
        wedgeprops={"width": 0.34, "edgecolor": "#fdfbf8"},
    )
    ax.text(0, 0.05, "Mix", ha="center", va="center", fontsize=22, fontweight="bold", color="#143d59")
    ax.text(0, -0.16, "defendible", ha="center", va="center", fontsize=10.5, color="#6a7886")
    legend_ax.axis("off")
    legend_ax.legend(
        wedges,
        [f"{label}  {share:.1f}%" for label, share in zip(plot["Canal"], plot["optimized_share_pct"])],
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        fontsize=9.2,
        handlelength=1.35,
        handletextpad=0.48,
        borderpad=0.2,
        labelcolor="#334455",
    )
    ax.set(aspect="equal")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.08, wspace=0.05)
    return fig


def historical_mix_split(weights: pd.DataFrame) -> dict[str, float]:
    shares = weights.set_index("channel")["historical_share_pct"].astype(float)
    digital_channels = ["paid_search", "social_paid", "video_online", "display", "email_crm"]
    offline_channels = ["exterior", "radio_local", "prensa"]
    digital_share = float(shares.loc[digital_channels].sum())
    offline_share = float(shares.loc[offline_channels].sum())
    top3_share = float(shares.sort_values(ascending=False).head(3).sum())
    return {
        "digital_share_pct": digital_share,
        "offline_share_pct": offline_share,
        "top3_share_pct": top3_share,
    }


def defendable_mix_share_chart(weights: pd.DataFrame) -> plt.Figure:
    plot = current_change_table(weights).sort_values("historical_share_pct", ascending=False).copy()
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.bar(x - 0.25, plot["historical_share_pct"], width=0.24, color="#d9d3ca", label="Historico 12M")
    ax.bar(x, plot["optimized_share_pct"], width=0.24, color="#2a6f97", label="Mix defendible 12M")
    ax.bar(x + 0.25, plot["efficient_share_pct"], width=0.24, color="#7aa95c", label="Mix eficiente 8.19M")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["Canal"], rotation=24, ha="right")
    ax.set_ylabel("Share del presupuesto (%)")
    ax.set_title("Historico 12M vs mix defendible 12M vs mix eficiente 8.19M")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def defendable_knee_position_chart(weights: pd.DataFrame, knees: pd.DataFrame) -> plt.Figure:
    plot = (
        current_change_table(weights)
        .merge(knees[["channel", "knee_budget_eur"]], on="channel", how="left")
        .sort_values("knee_budget_eur", ascending=False)
        .copy()
    )
    y = np.arange(len(plot))
    knee_vals = plot["knee_budget_eur"].to_numpy(float) / 1_000_000
    hist_vals = plot["historical_budget_eur"].to_numpy(float) / 1_000_000
    defendable_vals = plot["optimized_budget_eur"].to_numpy(float) / 1_000_000
    efficient_vals = plot["efficient_budget_eur"].to_numpy(float) / 1_000_000

    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.barh(y, knee_vals, height=0.34, color="#ddd8d0", edgecolor="none", label="Knee point", zorder=1)
    for idx, value in enumerate(knee_vals):
        ax.vlines(value, idx - 0.22, idx + 0.22, color="#143d59", linewidth=1.2, zorder=2)
    ax.scatter(hist_vals, y, s=64, color="#9cbcd0", marker="s", label="Historico 12M", zorder=4, edgecolors="#ffffff", linewidths=0.8)
    ax.scatter(defendable_vals, y, s=76, color="#2a6f97", marker="o", label="Mix defendible 12M", zorder=5, edgecolors="#ffffff", linewidths=0.9)
    ax.scatter(efficient_vals, y, s=74, color="#7aa95c", marker="^", label="Mix eficiente 8.19M", zorder=6, edgecolors="#ffffff", linewidths=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["Canal"])
    ax.invert_yaxis()
    ax.set_xlabel("Inversion por canal (M EUR)")
    ax.set_title("Historico, 12M y 8.19M frente al knee")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def knee_point_story_html(knees: pd.DataFrame) -> str:
    if knees.empty:
        return "<div class='note'>No hay tabla de knee points disponible.</div>"
    view = knees.sort_values("knee_budget_eur", ascending=False).set_index("channel")
    rows: list[str] = []
    for channel, row in view.iterrows():
        row = view.loc[channel]
        status = str(row["knee_status"]).replace("historical_", "").replace("_", " ")
        rows.append(
            "<tr>"
            f"<td>{LABELS.get(channel, channel)}</td>"
            f"<td>{compact_money(float(row['historical_budget_eur']))}</td>"
            f"<td>{compact_money(float(row['knee_budget_eur']))}</td>"
            f"<td>{float(row['knee_marginal_profit_per_eur']):.2f}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    return (
        "<div class='note'>"
        "<strong>Lectura del knee point.</strong> El coeficiente dice cuanto puede empujar un canal, pero la curva dice hasta donde merece la pena empujarlo. "
        "El knee point marca el punto a partir del cual la pendiente marginal se aplana: seguir invirtiendo puede seguir sumando, pero cada euro adicional rinde menos.<br><br>"
        "<table style='width:100%; border-collapse:collapse;'>"
        "<thead><tr><th align='left'>Canal</th><th align='left'>Historico</th><th align='left'>Knee</th><th align='left'>Pendiente</th><th align='left'>Estado</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def saturation_summary_table(roi: pd.DataFrame, knees: pd.DataFrame) -> pd.DataFrame:
    if roi.empty:
        return pd.DataFrame()
    cols = ["channel", "coefficient", "coefficient_pct", "saturation_status"]
    roi_column = "historical_roi" if "historical_roi" in roi.columns else None
    if roi_column:
        cols.append(roi_column)
    table = roi[cols].copy()
    if not knees.empty:
        table = table.merge(
            knees[["channel", "knee_marginal_profit_per_eur", "knee_status"]],
            on="channel",
            how="left",
        )
    table["Canal"] = table["channel"].map(LABELS)
    table["Peso en bloque medios"] = table["coefficient_pct"].map(lambda value: f"{float(value):.2f}%")
    table["Coeficiente"] = table["coefficient"].map(lambda value: f"{float(value):,.2f}")
    table["ROI canal"] = table[roi_column].map(lambda value: f"{float(value):.2f}") if roi_column else "-"
    table["Pendiente knee"] = table["knee_marginal_profit_per_eur"].map(
        lambda value: f"{float(value):.2f}" if pd.notna(value) else "-"
    )
    table["Estado curva"] = table["saturation_status"].astype(str).str.replace("_", " ", regex=False)
    return table.sort_values("coefficient_pct", ascending=False)[
        ["Canal", "Peso en bloque medios", "Coeficiente", "ROI canal", "Pendiente knee", "Estado curva"]
    ]


def render_global_summary(data: dict) -> None:
    checks = data["checks"]
    model = data["model"]
    knees = data["knees"].copy()
    weights = current_change_table(data["weights"])
    defendable_vs_knee = weights.merge(knees[["channel", "knee_budget_eur"]], on="channel", how="left")
    oversat_named = defendable_vs_knee.loc[
        defendable_vs_knee["optimized_budget_eur"] > defendable_vs_knee["knee_budget_eur"],
        "channel",
    ].map(LABELS).tolist()
    oversat_text = ", ".join(oversat_named[:3]) if oversat_named else "sin alerta fuerte"
    historical_top_channel = LABELS.get(
        str(weights.sort_values("historical_budget_eur", ascending=False).iloc[0]["channel"]),
        "n/a",
    )
    optimized_top_channel = LABELS.get(
        str(weights.sort_values("optimized_budget_eur", ascending=False).iloc[0]["channel"]),
        "n/a",
    )
    anchor_budget = float(weights.sort_values("optimized_budget_eur", ascending=False).iloc[0]["optimized_budget_eur"])
    biggest_cut = weights.iloc[0]
    biggest_push = weights.iloc[-1]
    budget_cut = 12_000_000.0 - 8_187_309.76
    roi_lift = 4.2833 - 3.2573
    total_now = 8_187_309.76

    st.markdown("<div class='summary-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>K-Moda MMM Executive App</h1>
            <p>Resumen general fijo arriba y seleccion de secciones debajo. El recorrido mezcla auditoria del dato,
            modelado, comparativa de escenarios y una capa final de saturacion y knee points para defender las decisiones de mix.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_kpis(
        [
            ("Recorte total", format_delta(-budget_cut), f"De 12.0M a {total_now/1_000_000:.2f}M sin perder solidez en eficiencia del mix."),
            ("Salto de ROI", f"+{roi_lift:.2f} pts", "El nuevo mix mejora el ROI frente al historico y sostiene mejor la rentabilidad."),
            ("Mayor recorte", f"{biggest_cut['Canal']} {format_delta(float(biggest_cut['delta_budget_eur']))}", "Es el ajuste mas fuerte del nuevo mix."),
            ("Mayor apuesta", f"{biggest_push['Canal']} {format_delta(float(biggest_push['delta_budget_eur']))}", "Es el canal que mas gana peso dentro de la reasignacion."),
            (
                "Ancla del 12M",
                f"{optimized_top_channel} {compact_money(anchor_budget)}",
                f"El mix defendible mantiene a {optimized_top_channel} como principal inversion, recorta Radio Local y contiene Prensa. Canales tensionados: {oversat_text}.",
            ),
        ]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_figure(
            "Panorama comercial y de medios",
            "1_eda/eda_01_weekly_sales_and_media_overview.png",
            "La portada resume la historia completa del negocio: tendencia, estacionalidad y presion publicitaria moviendose juntas.",
            figure_height=430,
        )
    with c2:
        show_figure(
            "Escenarios de decision",
            "6_simulacion/sim_data_01_scenario_performance.png",
            "Se comparan mixes con el mismo presupuesto para decidir no solo por beneficio, sino tambien por prudencia.",
            figure_height=430,
        )

    st.markdown("<div class='section-header'>Decisiones estructurales clave: knee points y saturacion</div>", unsafe_allow_html=True)
    k1, k2 = st.columns([1.15, 0.85])
    with k1:
        st.pyplot(knee_chart(knees), width="stretch")
    with k2:
        knee_view = knees.copy()
        knee_view["Canal"] = knee_view["channel"].map(LABELS)
        knee_view["Historico"] = knee_view["historical_budget_eur"].map(compact_money)
        knee_view["Knee budget"] = knee_view["knee_budget_eur"].map(compact_money)
        knee_view["Estado"] = knee_view["knee_status"].str.replace("historical_", "", regex=False).str.replace("_", " ", regex=False)
        st.dataframe(knee_view[["Canal", "Historico", "Knee budget", "Estado"]], width="stretch", hide_index=True)
    st.markdown(
        "<div class='note'>Esta capa es clave para el relato final: no basta con decir que mix mejora el beneficio. "
        "Tambien hay que explicar que canales estan ya por encima de su zona eficiente y cuales siguen teniendo recorrido antes de saturar.</div>",
        unsafe_allow_html=True,
    )
    top_changes = weights.copy()
    top_changes["Historico"] = top_changes["historical_budget_eur"].map(compact_money)
    top_changes["Mix defendible 12M"] = top_changes["optimized_budget_eur"].map(compact_money)
    top_changes["Cambio"] = top_changes["delta_budget_eur"].map(format_delta)
    top_changes["Cambio share"] = top_changes["delta_share_pct"].map(lambda v: f"{v:+.2f} pts")
    st.markdown("<div class='section-header'>Cambios principales de la implementación</div>", unsafe_allow_html=True)
    st.dataframe(top_changes[["Canal", "Historico", "Mix defendible 12M", "Cambio", "Cambio share"]], width="stretch", hide_index=True)
    st.markdown(
        "<div class='note'>Esta tabla ya no habla de cómo invertíamos antes por costumbre, sino de qué está cambiando ahora mismo en la propuesta: "
        "dónde recortamos, dónde apostamos más y cuánto se mueve realmente el mix.</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    data = load_data()
    checks = data["checks"]
    model = data["model"]

    render_global_summary(data)
    st.markdown(
        """
        <div class='journey-shell'>
            <div class='journey-kicker'>Navegacion del recorrido</div>
            <div class='journey-title'>Selecciona una seccion del recorrido MMM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(
        [
            "01. EDA",
            "02. Pre-processing",
            "03. Feature engineering",
            "04. Feature importance",
            "05. Modelado",
            "06. Resultados 12M vs 8M",
            "07. Simulacion",
            "08. Ajuste logaritmico",
        ]
    )

    with tabs[0]:
        weights = data["weights"]
        mix_split = historical_mix_split(weights)
        show_kpis(
            [
                ("Lineas de venta", "8.0M", "Base transaccional original del proyecto."),
                ("Semanas diagnosticas", "262", "Vista completa para QA y revision temporal."),
                ("Semanas incompletas", "2", "Solo en los bordes del rango."),
                ("Ticket medio", f"{float(checks.get('ticket_medio_neto_mean', 0.0)):.2f} EUR", "Sirve como referencia de negocio."),
            ]
        )
        a, b = st.columns(2)
        with a:
            show_figure(
                "Evolucion semanal",
                "1_eda/eda_01_weekly_sales_and_media_overview.png",
                "Aqui ya se ve que las ventas no van por libre: cambian con el tiempo y conviven con una presion publicitaria que tambien se mueve.",
            )
        with b:
            show_figure(
                "Estacionalidad",
                "1_eda/eda_03_monthly_sales_seasonality.png",
                "No parece una serie caotica. Hay un patron bastante estable a lo largo del ano y eso luego conviene recogerlo en el modelo.",
            )
        c, d = st.columns(2)
        with c:
            show_figure("Heatmap semanal", "1_eda/eda_07_week_year_sales_heatmap.png", "Lo importante aqui es que no hablamos solo de picos sueltos: tambien hay cambios de nivel bastante claros entre anos.")
        with d:
            show_figure("Lags ventas-medios", "1_eda/eda_14_sales_media_lag_heatmap.png", "La relacion con medios no parece instantanea, asi que tiene sentido trabajar despues con memoria y rezagos.")
        st.markdown("<div class='section-header'>Como invertiamos historicamente</div>", unsafe_allow_html=True)
        m1, m2 = st.columns([1.15, 0.85])
        with m1:
            st.pyplot(historical_mix_donut_chart(weights), width="stretch")
        with m2:
            st.markdown("<div class='compact-kpi'>", unsafe_allow_html=True)
            show_kpis(
                [
                    ("Digital", f"{mix_split['digital_share_pct']:.1f}%", "Peso del presupuesto historico en Paid Search, Social Paid, Video Online, Display y Email CRM."),
                    ("Offline", f"{mix_split['offline_share_pct']:.1f}%", "Peso historico combinado de Exterior, Radio Local y Prensa."),
                ]
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='note compact-note'>El punto de partida ya venia bastante concentrado, asi que la optimizacion no arranca desde un mix neutro sino desde una apuesta muy marcada.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='note compact-note'>Al final, casi todo el peso estaba muy concentrado: el {mix_split['top3_share_pct']:.1f}% del presupuesto se iba a solo 3 canales, y ademas con bastante sesgo hacia digital.</div>",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        pre = data["pre"]
        show_kpis(
            [
                ("Filas causales", str(int(checks.get("causal_dataset_rows", 0))), "Vista final sin semanas parciales."),
                ("Panel completo", str(bool(checks.get("weekly_panel_complete", False))), "No faltan semanas en el rango modelado."),
                ("Missing finales", str(int(checks.get("remaining_missing_cells", 0))), "Limpieza cerrada a cero."),
                ("Duplicados post-join", str(int(checks.get("max_duplicate_keys_after_join", 0))), "Los joins preservan integridad."),
            ]
        )
        a, b = st.columns(2)
        with a:
            show_figure("Cobertura temporal", "2_preprocessing/pre_04_temporal_consistency.png", "Sirve para ver que el calendario util queda estable y que las pocas incidencias se concentran en los bordes del periodo.")
        with b:
            show_figure("Conteo del pipeline", "2_preprocessing/pre_02_pipeline_row_counts.png", "El salto de volumen no es un problema: es la conversion de dato transaccional a panel semanal listo para modelar.")
        st.markdown("<div class='section-header'>Que nos deja el pre-processing</div>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(
                "<div class='note'><strong>Calendario limpio.</strong> El panel util queda estable casi todo el periodo. Las incidencias no contaminan la serie central y se quedan en los bordes.</div>",
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                "<div class='note'><strong>Panel consistente.</strong> Llegamos al modelado sin missing finales, sin infinitos y sin duplicados de clave. Aqui la prioridad era dejar un dataset fiable, no decorarlo.</div>",
                unsafe_allow_html=True,
            )
        with p3:
            st.markdown(
                "<div class='note'><strong>Decision metodologica.</strong> La vista causal no intenta conservar todo el ruido transaccional: se queda con semanas completas para poder comparar, explicar y simular mejor.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='section-header'>Resumen del dataset preparado</div>", unsafe_allow_html=True)
        st.dataframe(pre, width="stretch", hide_index=True)
        st.markdown("<div class='note'>La diferencia entre vista diagnostica y vista causal no es una perdida rara de datos, sino una decision metodologica para trabajar con semanas completas.</div>", unsafe_allow_html=True)

    with tabs[2]:
        fe_counts = data["fe_counts"]
        fe = data["fe"].sort_values("abs_correlation", ascending=False).head(12)
        show_kpis(
            [
                ("Shape causal", "260 x 39", "Vista compacta y presentable."),
                ("Model ready", "23", "Features listas para la ruta principal."),
                ("Screening only", "13", "Variables retenidas solo para cribado."),
                ("Top correlacion", str(fe.iloc[0]["feature"]), "La tendencia sigue siendo la senal mas fuerte."),
            ]
        )
        st.bar_chart(fe_counts.set_index("family"), height=320)
        st.markdown("<div class='note'>Las familias de variables cuentan una historia mucho mas limpia que un listado suelto de columnas.</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            show_figure("Correlaciones clave", "3_feature_engineering/fe_02_top_target_correlations.png", "El presupuesto total y la tendencia explican mucho, pero no todo.")
        with b:
            show_figure("Ventas vs presupuesto", "3_feature_engineering/fe_03_target_vs_total_spend.png", "Invertir mas no equivale automaticamente a vender mas; tambien importa el reparto.")
        st.markdown("<div class='section-header'>Variables con mayor relacion con ventas</div>", unsafe_allow_html=True)
        st.dataframe(fe[["feature", "family", "modeling_tier", "correlation_with_target", "note"]], width="stretch", hide_index=True)

    with tabs[3]:
        rf_specs = data["rf_specs"].sort_values("mape")
        rf = data["rf"].sort_values("permutation_importance_mean", ascending=False).head(12)
        top_rf_feature = str(rf.iloc[0]["feature"])
        top_rf_group = str(
            rf.groupby("feature_group")["permutation_importance_mean"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        show_kpis(
            [
                ("Spec screening", str(rf_specs.iloc[0]["spec"]), "Gana la version mas prudente."),
                ("MAPE screening", f"{rf_specs.iloc[0]['mape']:.2f}", "Las diferencias entre specs son pequenas."),
                ("Top feature", str(rf.iloc[0]["feature"]), "Senal prioritaria en el cribado."),
                ("Mensaje clave", "Filtro", "Este bloque prioriza; no aprueba causalidad."),
            ]
        )
        st.markdown(
            "<div class='note'>Aqui conviene ser prudentes: las diferencias entre specs son pequenas. Esta capa se usa como screening orientativo para comparar familias de variables, no como veredicto final del modelo.</div>",
            unsafe_allow_html=True,
        )
        a, b = st.columns(2)
        with a:
            show_figure("Comparativa de specs", "4_feature_importance/random_forest_spec_comparison.png", "Esta vista ayuda a ver rapido que las tres rutas quedan bastante cerca entre si. Sirve para comparar enfoques de screening sin sacar conclusiones fuertes todavia.")
        with b:
            show_figure("Grupos de variables", "4_feature_importance/random_forest_feature_groups.png", "Agrupar la senal por familias hace el resultado mucho mas defendible.")
        c, d = st.columns(2)
        with c:
            show_figure("Top features", "4_feature_importance/random_forest_top_features.png", "Mas que fiarnos del ranking al detalle, esto nos sirve para ver por donde asoma algo de senal antes del modelado serio.")
        with d:
            st.markdown(
                f"<div class='note'>Esta capa nos sirve sobre todo como filtro inicial: <code>{top_rf_feature}</code> aparece arriba y la familia <code>{top_rf_group}</code> concentra buena parte de la senal que destaca en el screening.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='section-header'>Resumen del feature importance</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='note'><strong>En corto.</strong> En este screening lo que mas aparece arriba es <code>{top_rf_feature}</code>, y por familias destaca sobre todo <code>{top_rf_group}</code>. La utilidad real de esta capa es darnos una primera pista de donde asoma senal antes de pasar al modelado serio.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(rf[["feature", "feature_group", "spec", "permutation_importance_mean"]], width="stretch", hide_index=True)

    with tabs[4]:
        bench = data["bench"].sort_values("test_2024_mape")
        coef = data["coef"].sort_values("coefficient", ascending=False).head(12)
        formula_html = media_response_formula_html(data["roi"])
        show_kpis(
            [
                ("Modelo oficial", str(model.get("winner", "ARIMAX")), f"Especificacion oficial: {model.get('deployment_spec', 'ARIMAXLaggedBudgetMixSeasonal')}."),
                ("MAPE backtest", f"{bench.loc[bench['spec'] == 'ARIMAX', 'validation_mean_mape'].iloc[0]:.2f}", "Media de validacion temporal."),
                ("MAPE test 2024", f"{model.get('test_metrics', {}).get('mape', 0.0):.2f}", "ARIMAX gana en test."),
                ("Orden", str(tuple(model.get("selected_order", []))), "Con componente estacional semanal/anual."),
            ]
        )
        st.markdown("<div class='section-header'>Benchmarks obligatorios</div>", unsafe_allow_html=True)
        st.dataframe(bench[["spec", "validation_mean_mape", "test_2024_mape", "test_rank_mape"]], width="stretch", hide_index=True)
        a, b = st.columns([1.65, 1.0])
        with a:
            show_figure(
                "Ventas reales vs predichas",
                "6_simulacion/sim_prediction_01_actual_vs_predicted.png",
                "Esta vista resume mejor el modelo completo: entrena sobre el historico, valida el tramo 2024 y comprueba que captura los cambios de nivel sin perder estabilidad.",
                figure_height=410,
            )
        with b:
            show_pyplot_figure(
                "Mix defendible 12M",
                defendable_mix_donut_chart(data["weights"]),
                "Este donut resume el reparto que el modelo sostiene para el mix defendible 12M y ayuda a leer rapido donde se concentra el peso final por canal.",
                figure_height=380,
            )
        st.markdown("<div class='section-header'>Formula resumida del mix</div>", unsafe_allow_html=True)
        st.markdown(
            (
                "<div class='note formula-card'>"
                f"{formula_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        coef_chart_path = CONFIG.reports_figures_dir / "5_modelado/arimax_coefficients.png"
        st.markdown("<div class='section-header'>Coeficientes exogenos</div>", unsafe_allow_html=True)
        coef_left, coef_right = st.columns(2)
        with coef_left:
            if coef_chart_path.exists():
                st.image(str(coef_chart_path), width=620)
            else:
                st.warning(f"No se encontro {coef_chart_path.name}.")
        with coef_right:
            st.table(coef[["parameter", "coefficient", "ci_low", "ci_high"]].reset_index(drop=True))
        st.markdown(
            "<div class='note'>El modelo final mezcla tendencia, calendario y mix de medios en la misma ecuacion. La grafica ayuda a leer el peso relativo y la tabla deja el detalle numerico a mano.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-header'>Curvas de saturacion y knee point</div>", unsafe_allow_html=True)
        st.markdown(
            (
                "<div class='note'>"
                "Con los coeficientes ya sabemos que canales tienen mas peso en el bloque de medios, pero todavia falta una pieza del modelo: "
                "la respuesta no crece de forma infinita. Las curvas de saturacion muestran como cambia la pendiente de cada canal y el "
                "<strong>knee point</strong> marca la zona donde empieza a perderse retorno marginal. Esta parte todavia no decide cuanto invertir; "
                "solo explica la forma de la respuesta del canal."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(saturation_summary_table(data["roi"], data["knees"]), width="stretch", hide_index=True)
        saturation_channels = ["exterior", "video_online", "paid_search", "radio_local", "social_paid", "email_crm", "display", "prensa"]
        saturation_notes = {
            "exterior": "Mayor peso del bloque de medios (40.62%). La curva sube con fuerza al principio, pero el knee recuerda que potencia no significa escala infinita.",
            "video_online": "Segundo mayor peso (27.71%). Tiene respuesta clara y util, aunque la pendiente se suaviza al acercarse al knee.",
            "paid_search": "Peso alto (17.02%) y senal muy relevante. La curva ayuda a separar demanda capturable de sobrepresion del canal.",
            "radio_local": "Peso medio (5.33%). Aporta, pero el knee marca un tramo eficiente mas acotado que en los canales principales.",
            "social_paid": "Peso medio-bajo (5.02%). Funciona como apoyo incremental, con saturacion antes que los canales de mayor respuesta.",
            "email_crm": "Peso pequeno (2.48%). La lectura esta en aprovechar el tramo eficiente sin esperar que escale como motor principal.",
            "display": "Peso pequeno (1.82%). Respuesta positiva pero limitada; encaja mejor como refuerzo del mix que como palanca central.",
            "prensa": "Peso practicamente nulo (0.0002%). La curva queda plana o poco eficiente, asi que el modelo no le asigna margen real de empuje.",
        }
        for row_start in range(0, len(saturation_channels), 4):
            curve_cols = st.columns(4)
            for col, channel in zip(curve_cols, saturation_channels[row_start : row_start + 4]):
                with col:
                    show_figure(
                        LABELS[channel],
                        f"6_simulacion/sim_saturation_historical_{channel}.png",
                        saturation_notes[channel],
                    )

    with tabs[5]:
        knees = data["knees"].copy()
        reference_df = reference_table(data["curves"])
        historical_row = reference_df.loc[reference_df["scenario"] == "Historico 12M"].iloc[0]
        defendable_row = reference_df.loc[reference_df["scenario"] == "Mix defendible 12M"].iloc[0]
        efficient_row = reference_df.loc[reference_df["scenario"] == "Mix eficiente 8.19M"].iloc[0]
        historical_profit = float(historical_row["predicted_gross_profit_2024"])
        historical_roi = float(historical_row["roi_vs_zero_media"])
        defendable_profit = float(defendable_row["predicted_gross_profit_2024"])
        defendable_roi = float(defendable_row["roi_vs_zero_media"])
        efficient_profit = float(efficient_row["predicted_gross_profit_2024"])
        efficient_roi = float(efficient_row["roi_vs_zero_media"])
        efficient_budget = float(efficient_row["budget_total_eur"])
        defendable_delta = defendable_profit - historical_profit
        efficient_delta = efficient_profit - historical_profit
        roi_lift = efficient_roi - historical_roi
        efficient_saving = 12_000_000.0 - efficient_budget
        show_kpis(
            [
                (
                    "Base historica 12M",
                    f"{compact_money(historical_profit)} / ROI {historical_roi:.2f}",
                    "Punto de partida: mismo presupuesto historico, beneficio y rentabilidad incremental de referencia.",
                ),
                (
                    "Mix defendible 12M",
                    f"{compact_money(defendable_profit)} / ROI {defendable_roi:.2f}",
                    f"Mantiene 12M y reordena canales para ganar {format_delta(defendable_delta)} de beneficio frente al historico.",
                ),
                (
                    "Mix eficiente 8.19M",
                    f"{compact_money(efficient_profit)} / ROI {efficient_roi:.2f}",
                    f"Reduce inversion en {compact_money(efficient_saving)} y prioriza eficiencia aunque el beneficio quede {format_delta(efficient_delta)} vs historico.",
                ),
                (
                    "Lectura conjunta",
                    f"+{roi_lift:.2f} pts ROI / {compact_money(efficient_saving)} ahorro",
                    "12M busca defender beneficio total; 8.19M busca maximizar rentabilidad con menos dinero invertido.",
                ),
            ]
        )
        st.markdown("<div class='section-header'>Mix defendible frente al knee</div>", unsafe_allow_html=True)
        knee_left, knee_right = st.columns([1.45, 1.0])
        with knee_left:
            st.pyplot(defendable_knee_position_chart(data["weights"], knees), width="stretch")
            st.markdown("<div class='section-header'>Comparativa del mix</div>", unsafe_allow_html=True)
            st.pyplot(defendable_mix_share_chart(data["weights"]), width="stretch")
            st.markdown(
                "<div class='note'>Esta comparacion ya usa el mix defendible 12M y el mix eficiente 8.19M. Asi se ve de forma limpia como cambia el reparto frente al historico.</div>",
                unsafe_allow_html=True,
            )
        with knee_right:
            st.markdown(knee_point_story_html(knees), unsafe_allow_html=True)
        st.markdown("<div class='note'>8.19M es la mejor referencia de eficiencia; para 12M se muestra aqui el mix defendible de negocio, no el simple reescalado mecanico del 8.19M.</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Detalle de inversion por canal</div>", unsafe_allow_html=True)
        st.dataframe(budget_results_table(data["weights"]), width="stretch", hide_index=True)
        st.markdown("<div class='section-header'>Eficiencia de las propuestas</div>", unsafe_allow_html=True)
        st.pyplot(scenario_chart(reference_df), width="stretch")
        st.dataframe(reference_df, width="stretch", hide_index=True)
        st.markdown("<div class='section-header'>Do nothing vs do something</div>", unsafe_allow_html=True)
        show_pyplot_figure(
            "Comparativa final de escenarios",
            do_nothing_vs_proposals_chart(reference_df),
            "Aqui cerramos el apartado comparando el escenario sin inversion con el historico y con las dos propuestas: el do something eficiente de 8.19M y el do something defendible de 12M. Es una lectura mas directa que juntar todas las curvas de canal en una sola vista.",
        )

    with tabs[6]:
        weights = data["weights"]
        curves = data["curves"]
        knees = data["knees"][["channel", "knee_budget_eur"]]
        presets = {
            "Historico 12M": {row["channel"]: float(row["historical_budget_eur"]) for _, row in weights.iterrows()},
            "Mix defendible 12M": defendable_12m_mix(),
            "Mix eficiente 8.19M": EFFICIENT_819M,
        }
        if "sim_initialized" not in st.session_state:
            for _, row in weights.iterrows():
                st.session_state[f"budget_{row['channel']}"] = float(row["historical_budget_eur"])
            st.session_state["sim_initialized"] = True

        st.markdown(
            "<div class='note'><strong>Como leer esta simulacion.</strong> Si subes mucho dinero en un canal aislado, puedes ver saltos muy fuertes de retorno en esta vista exploratoria. "
            "Eso no significa que la realidad crezca de forma lineal e ilimitada. En la optimizacion real hemos tenido en cuenta que las curvas de respuesta se saturan, "
            "que existe un punto de rendimiento decreciente y que los <em>knee points</em> marcan precisamente esa zona donde una inversion adicional empieza a devolver menos valor marginal. "
            "Por eso esta herramienta sirve para jugar con escenarios y entender sensibilidades, pero la recomendacion final sigue siendo la del mix defendible 12M con saturacion incorporada.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='note'><strong>Como funciona ahora.</strong> En vez de tocar pesos, aqui introduces euros por canal. "
            "La app calcula automaticamente el presupuesto total, el mix implicito y el delta frente al historico con ese mismo total.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-header'>Elige un mix de partida</div>", unsafe_allow_html=True)
        render_preset_selector(presets, session_prefix="budget", key_prefix="preset_07")

        grid = st.columns(4)
        raw_budgets = {}
        slider_caps = {}
        for _, row in weights.iterrows():
            ch = row["channel"]
            knee_row = knees.loc[knees["channel"] == ch, "knee_budget_eur"]
            knee_value = float(knee_row.iloc[0]) if not knee_row.empty else 0.0
            cap_base = max(
                float(row["historical_budget_eur"]),
                float(row["optimized_budget_eur"]),
                float(EFFICIENT_819M.get(ch, 0.0)),
                knee_value,
            )
            slider_caps[ch] = max(int(np.ceil((cap_base * 1.6) / 50_000.0) * 50_000), 300_000)
        for i, ch in enumerate(CHANNELS):
            with grid[i % 4]:
                raw_budgets[ch] = float(
                    st.slider(
                        LABELS[ch],
                        min_value=0,
                        max_value=int(slider_caps[ch]),
                        step=25_000,
                        key=f"budget_{ch}",
                    )
                )
        total_budget = float(sum(raw_budgets.values()))
        if total_budget <= 0:
            st.warning("Introduce al menos inversion en un canal para activar la simulacion.")
            return

        rows = []
        plan_profit = float(curves["do_nothing_profit_2024"].iloc[0])
        hist_profit = float(curves["do_nothing_profit_2024"].iloc[0])
        for _, row in weights.iterrows():
            ch = row["channel"]
            sel_budget = float(raw_budgets[ch])
            sel_share = sel_budget / total_budget * 100.0 if total_budget > 0 else 0.0
            hist_budget = total_budget * float(row["historical_share_pct"]) / 100.0
            sel_delta = interp_delta(curves, ch, sel_budget)
            hist_delta = interp_delta(curves, ch, hist_budget)
            plan_profit += sel_delta
            hist_profit += hist_delta
            rows.append(
                {
                    "channel": ch,
                    "Canal": LABELS[ch],
                    "Share plan": sel_share,
                    "Budget plan": sel_budget,
                    "Budget hist. mismo total": hist_budget,
                    "Delta vs historico": sel_delta - hist_delta,
                    "Base delta": sel_delta,
                }
            )

        sim = pd.DataFrame(rows)
        sim = sim.merge(knees.assign(Canal=lambda d: d["channel"].map(LABELS))[["Canal", "knee_budget_eur"]], on="Canal", how="left")
        sim["Estado knee"] = sim.apply(lambda r: knee_status(float(r["Budget plan"]), float(r["knee_budget_eur"])), axis=1)
        show_kpis(
            [
                ("Beneficio estimado", compact_money(plan_profit), "Estimacion exploratoria del plan actual."),
                ("Delta vs hist. escalado", compact_money(plan_profit - hist_profit), "Compara tu mix contra el historico con el mismo total."),
                ("Incremental vs cero", compact_money(plan_profit - ZERO_MEDIA_PROFIT), "Valor incremental frente a no invertir."),
                ("Presupuesto total", compact_money(total_budget), "Suma directa de los euros que acabas de asignar."),
                ("ROI incremental", f"{(plan_profit - ZERO_MEDIA_PROFIT) / total_budget:.2f}", "Rentabilidad estimada del plan."),
            ]
        )
        sim_view = sim.copy()
        sim_view["Share plan"] = sim_view["Share plan"].map(lambda v: f"{v:.2f}%")
        sim_view["Budget plan"] = sim_view["Budget plan"].map(compact_money)
        sim_view["Budget hist. mismo total"] = sim_view["Budget hist. mismo total"].map(compact_money)
        sim_view["Delta vs historico"] = sim_view["Delta vs historico"].map(format_delta)
        sim_view["knee_budget_eur"] = sim_view["knee_budget_eur"].map(compact_money)
        st.dataframe(
            sim_view[["Canal", "Share plan", "Budget plan", "Budget hist. mismo total", "Delta vs historico", "knee_budget_eur", "Estado knee"]],
            width="stretch",
            hide_index=True,
        )

        chosen = st.selectbox("Canal para revisar saturacion y respuesta", options=CHANNELS, format_func=lambda x: LABELS[x])
        p1, p2 = st.columns(2)
        with p1:
            show_figure(f"Curva de respuesta de {LABELS[chosen]}", f"6_simulacion/sim_response_curve_{chosen}.png", "Esta curva muestra como cambia el beneficio total cuando ese canal gana o pierde peso dentro del mix.")
        with p2:
            show_figure(f"Saturacion historica de {LABELS[chosen]}", f"6_simulacion/sim_saturation_historical_{chosen}.png", "Esta grafica aterriza el knee point en euros y ayuda a ver si el canal esta todavia en zona de headroom o ya en sobreinversion.")

    shape_scores = {}
    for ch in CHANNELS:
        profile = log_curve_profile(curves, ch)
        if profile and profile["rmse_linear"] > 0:
            shape_scores[ch] = max(0.0, 1.0 - (profile["rmse_log"] / profile["rmse_linear"]))
    suggested_log_channels = [ch for ch, score in sorted(shape_scores.items(), key=lambda item: item[1], reverse=True) if score > 0.08][:3]
    suggested_text = ", ".join(LABELS[ch] for ch in suggested_log_channels) if suggested_log_channels else "sin una ventaja clara frente al ajuste actual"

    with tabs[7]:
        st.markdown("<div class='section-header'>Curvas logaritmicas</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='note'><strong>Que estamos haciendo aqui.</strong> Esta octava seccion replica la simulacion 07, "
            "pero sustituye la componente base por una componente logaritmica solo en los 3 canales cuya curva de saturacion encaja mejor con ese patron. "
            f"Los tres canales seleccionados automaticamente son: {suggested_text}. Los otros 5 canales conservan exactamente la referencia base anterior.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='note'><strong>Formula que usamos.</strong> Para esos 3 canales hemos estimado cada respuesta como "
            "<code>intercepto + beta * ln(1 + inversion / escala)</code>. Esto es equivalente a escribir "
            "<code>intercepto + B * log_A(1 + inversion / escala)</code>. La base <code>A</code> y el coeficiente <code>B</code> no son independientes: "
            "si cambias la base, el coeficiente se reescala. Por eso ajustamos en logaritmo natural y luego reportamos para cada canal su propia base A y su propio coeficiente B.</div>",
            unsafe_allow_html=True,
        )
        log_channels = suggested_log_channels
        log_channel_set = set(log_channels)
        st.markdown(
            f"<div class='note'><strong>Canales con ajuste logaritmico activo.</strong> {', '.join(LABELS[ch] for ch in log_channels)}.</div>",
            unsafe_allow_html=True,
        )

        if "logsim_initialized" not in st.session_state:
            for _, row in weights.iterrows():
                st.session_state[f"log_budget_{row['channel']}"] = float(row["historical_budget_eur"])
            st.session_state["logsim_initialized"] = True

        render_preset_selector(presets, session_prefix="log_budget", key_prefix="preset_08", button_suffix=" en 08")

        log_grid = st.columns(4)
        log_budgets = {}
        log_slider_caps = {}
        for _, row in weights.iterrows():
            ch = row["channel"]
            knee_row = knees.loc[knees["channel"] == ch, "knee_budget_eur"]
            knee_value = float(knee_row.iloc[0]) if not knee_row.empty else 0.0
            cap_base = max(
                float(row["historical_budget_eur"]),
                float(row["optimized_budget_eur"]),
                float(EFFICIENT_819M.get(ch, 0.0)),
                knee_value,
            )
            log_slider_caps[ch] = max(int(np.ceil((cap_base * 1.6) / 50_000.0) * 50_000), 300_000)
        for i, ch in enumerate(CHANNELS):
            with log_grid[i % 4]:
                log_budgets[ch] = float(
                    st.slider(
                        f"{LABELS[ch]} ",
                        min_value=0,
                        max_value=int(log_slider_caps[ch]),
                        step=25_000,
                        key=f"log_budget_{ch}",
                    )
                )
        log_total_budget = float(sum(log_budgets.values()))
        if log_total_budget <= 0:
            st.warning("Introduce al menos inversion en un canal para activar la simulacion logaritmica.")
            return

        base08_profit = float(curves["do_nothing_profit_2024"].iloc[0])
        mixed_profit = float(curves["do_nothing_profit_2024"].iloc[0])
        mixed_hist_profit = float(curves["do_nothing_profit_2024"].iloc[0])
        mixed_rows = []
        params_rows = []
        for _, row in weights.iterrows():
            ch = str(row["channel"])
            sel_budget = float(log_budgets[ch])
            sel_share = sel_budget / log_total_budget * 100.0 if log_total_budget > 0 else 0.0
            hist_budget = log_total_budget * float(row["historical_share_pct"]) / 100.0
            base_delta = interp_delta(curves, ch, sel_budget)
            base08_profit += base_delta
            mixed_delta = adjustment_delta(curves, ch, sel_budget, log_channel_set)
            mixed_hist_delta = adjustment_delta(curves, ch, hist_budget, log_channel_set)
            mixed_profit += mixed_delta
            mixed_hist_profit += mixed_hist_delta
            mixed_rows.append(
                {
                    "Canal": LABELS[ch],
                    "Modelo": "Logaritmico" if ch in log_channel_set else "Base actual",
                    "Share plan": sel_share,
                    "Budget plan": sel_budget,
                    "Budget hist. mismo total": hist_budget,
                    "Delta vs historico": mixed_delta - mixed_hist_delta,
                    "Gap vs base": mixed_delta - base_delta,
                }
            )
            if ch in log_channel_set:
                params = log_formula_parts(curves, ch)
                params_rows.append(
                    {
                        "Canal": LABELS[ch],
                        "Base A": params.get("base_a", np.nan),
                        "Coef B": params.get("coef_b", np.nan),
                        "Intercepto": params.get("intercept", np.nan),
                        "Escala": params.get("scale", np.nan),
                        "Mejora ajuste": shape_scores.get(ch, 0.0),
                    }
                )

        show_kpis(
            [
                ("Beneficio ajuste 08", compact_money(mixed_profit), "Simulacion con ajuste logaritmico solo en 3 canales."),
                ("Gap vs base 08", compact_money(mixed_profit - base08_profit), "Cuanto cambia el resultado frente a esta misma simulacion sin logaritmos."),
                ("Delta vs hist. escalado", compact_money(mixed_profit - mixed_hist_profit), "Comparacion contra el historico con el mismo presupuesto total."),
                ("Canales con log", str(len(log_channels)), "Solo los 3 canales mas compatibles con una curva logaritmica."),
                ("ROI mixto", f"{(mixed_profit - ZERO_MEDIA_PROFIT) / log_total_budget:.2f}", "ROI incremental bajo el ajuste combinado."),
            ]
        )

        sim08_view = pd.DataFrame(mixed_rows)
        sim08_view["Share plan"] = sim08_view["Share plan"].map(lambda v: f"{v:.2f}%")
        sim08_view["Budget plan"] = sim08_view["Budget plan"].map(compact_money)
        sim08_view["Budget hist. mismo total"] = sim08_view["Budget hist. mismo total"].map(compact_money)
        sim08_view["Delta vs historico"] = sim08_view["Delta vs historico"].map(format_delta)
        sim08_view["Gap vs base"] = sim08_view["Gap vs base"].map(format_delta)
        st.dataframe(sim08_view, width="stretch", hide_index=True)

        params_view = pd.DataFrame(params_rows)
        params_view["Base A"] = params_view["Base A"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        params_view["Coef B"] = params_view["Coef B"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
        params_view["Intercepto"] = params_view["Intercepto"].map(lambda v: compact_money(v) if pd.notna(v) else "-")
        params_view["Escala"] = params_view["Escala"].map(lambda v: compact_money(v) if pd.notna(v) else "-")
        params_view["Mejora ajuste"] = params_view["Mejora ajuste"].map(lambda v: f"{v:.0%}")
        st.dataframe(params_view, width="stretch", hide_index=True)
        st.markdown(
            "<div class='note'>Solo mostramos los 3 canales que pasan a logaritmico. "
            "Cada uno tiene su propia base A, su propio coeficiente B, su propio intercepto y su propia escala. Los otros 5 canales permanecen con la referencia base de la simulacion 07.</div>",
            unsafe_allow_html=True,
        )
        focus_channel = st.selectbox(
            "Canal para revisar el ajuste logaritmico",
            options=log_channels,
            format_func=lambda x: LABELS[x],
        )
        fx1, fx2 = st.columns([1.2, 0.8])
        with fx1:
            st.pyplot(adjustment_fit_chart(curves, weights, knees, focus_channel, log_budgets[focus_channel]), width="stretch")
        with fx2:
            base_focus = interp_delta(curves, focus_channel, log_budgets[focus_channel])
            log_focus = log_delta(curves, focus_channel, log_budgets[focus_channel])
            fit_gap = shape_scores.get(focus_channel, 0.0)
            params = log_formula_parts(curves, focus_channel)
            st.markdown(
                "<div class='note'>La linea continua recoge la curva usada hoy en la simulacion 07. "
                "La discontinua muestra el ajuste logaritmico para este canal. "
                f"En {LABELS[focus_channel]}, el plan actual de {compact_money(log_budgets[focus_channel])} pasaria de {compact_money(base_focus)} a {compact_money(log_focus)}. "
                f"Su formula reportada queda como <code>{compact_money(params.get('intercept', 0.0))} + {params.get('coef_b', 0.0):,.0f} * log_{params.get('base_a', 2.0):.2f}(1 + inversion / {max(params.get('scale', 1.0), 1.0):,.0f})</code>. "
                f"La mejora relativa del encaje logaritmico frente a una recta simple es del {fit_gap:.0%}.</div>",
                unsafe_allow_html=True,
            )
            st.image(str(CONFIG.reports_figures_dir / f"6_simulacion/sim_saturation_historical_{focus_channel}.png"), width="stretch")
            st.markdown(
                "<div class='note'><strong>Contexto de saturacion.</strong> Esta grafica historica sigue siendo la referencia para defender el caso real ante negocio. "
                "La seccion 08 no sustituye el mix oficial: ayuda a visualizar mejor canales donde la pendiente cae muy rapido y un ajuste logaritmico resulta mas natural.</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
