from __future__ import annotations

import math
from typing import Dict

import numpy as np


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.where(denom < 1e-9, np.nan, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.nanmean(ratio) * 100.0)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom < 1e-9:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if abs(ss_tot) < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "wmape": weighted_mean_absolute_percentage_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "bias": mean_bias(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
