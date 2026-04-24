from __future__ import annotations

import numpy as np


CONTROL_COLUMNS = [
    "trend_index",
    "week_sin",
    "week_cos",
    "rebajas_flag",
    "black_friday_flag",
    "navidad_flag",
    "semana_santa_flag",
    "vacaciones_escolares_flag",
    "festivo_local_count",
    "payday_count",
    "temperatura_media_c_mean",
    "lluvia_indice_mean",
    "turismo_indice_mean",
    "incidencia_ecommerce_flag",
]

TRAFFIC_COLUMNS = [
    "visitas_tienda_sum",
    "sesiones_web_sum",
    "tasa_conversion_tienda_mean",
    "tasa_conversion_web_mean",
]

# No downstream sales proxies in the final MMM workflow.
EXPLORATORY_SCREENING_COLUMNS = []

TRANSFORM_GRID = [
    {"lag": lag, "alpha": alpha, "saturation": saturation}
    for lag in [0, 1, 2]
    for alpha in [0.0, 0.25, 0.5, 0.75, 0.9]
    for saturation in ["none", "log1p"]
]

BETA_TRANSFORM_GRID = [
    {"lag": lag, "alpha": alpha, "saturation": saturation}
    for lag in [0, 1, 2]
    for alpha in [0.0, 0.5, 0.9]
    for saturation in ["none", "log1p"]
]

ELASTIC_NET_ALPHAS = np.logspace(-3, 1, 20)
ELASTIC_NET_L1 = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_ELASTIC_NET_ALPHAS = np.logspace(-1, 1, 8)
BETA_ELASTIC_NET_L1 = [0.1, 0.3, 0.5, 0.7]
RANDOM_STATE = 42
