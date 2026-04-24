from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def slugify_channel(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def geometric_adstock(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    carry = 0.0
    for idx, value in enumerate(values):
        carry = float(value) + alpha * carry
        result[idx] = carry
    return result


def apply_lag(values: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return values.astype(float)
    result = np.zeros_like(values, dtype=float)
    result[lag:] = values[:-lag]
    return result


def logistic_saturation(values: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        return values.astype(float)
    return 1.0 - np.exp(-values / lam)


@dataclass
class MediaTransformSpec:
    lag: int
    alpha: float
    saturation_lambda: float | None = None


def transformed_feature_name(media_col: str, lag: int, alpha: float, saturation: str) -> str:
    return (
        f"{media_col}__lag{lag}_a"
        f"{str(alpha).replace('.', '_')}_{saturation}"
    )


def transform_media_series(values: np.ndarray, spec: MediaTransformSpec) -> np.ndarray:
    lagged = apply_lag(values.astype(float), spec.lag)
    stocked = geometric_adstock(lagged, spec.alpha)
    if spec.saturation_lambda is None:
        return stocked
    return logistic_saturation(stocked, spec.saturation_lambda)


def apply_media_transform(values: pd.Series, lag: int, alpha: float, saturation: str) -> pd.Series:
    spec = MediaTransformSpec(lag=lag, alpha=alpha, saturation_lambda=None)
    transformed = transform_media_series(values.to_numpy(dtype=float), spec)
    series = pd.Series(transformed, index=values.index, dtype=float)
    if saturation == "log1p":
        return np.log1p(series)
    if saturation != "none":
        raise ValueError(f"Unsupported saturation '{saturation}'.")
    return series


def add_transformed_media_features(
    df: pd.DataFrame,
    media_columns: list[str],
    channel_params: dict[str, dict[str, float | int | str]],
    group_col: str = "ciudad",
    date_col: str = "semana_inicio",
) -> pd.DataFrame:
    out = df.sort_values([group_col, date_col]).copy()
    for media_col in media_columns:
        params = channel_params[media_col]
        feature_name = transformed_feature_name(
            media_col,
            lag=int(params["lag"]),
            alpha=float(params["alpha"]),
            saturation=str(params["saturation"]),
        )
        out[feature_name] = (
            out.groupby(group_col, group_keys=False)[media_col]
            .apply(
                lambda series: apply_media_transform(
                    series,
                    lag=int(params["lag"]),
                    alpha=float(params["alpha"]),
                    saturation=str(params["saturation"]),
                )
            )
            .astype(float)
        )
    return out


def transform_media_panel(
    df: pd.DataFrame,
    city_col: str,
    date_col: str,
    media_columns: list[str],
    specs: dict[str, MediaTransformSpec],
) -> pd.DataFrame:
    transformed = df.sort_values([city_col, date_col]).copy()
    for column in media_columns:
        spec = specs[column]
        transformed[f"{column}_transformed"] = (
            transformed.groupby(city_col, sort=False)[column]
            .transform(lambda s: transform_media_series(s.to_numpy(dtype=float), spec))
            .astype(float)
        )
    return transformed
