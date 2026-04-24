from __future__ import annotations

import numpy as np


def compute_curve_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or len(x_arr) != len(y_arr):
        raise ValueError("x and y must be one-dimensional arrays with the same length.")
    if len(x_arr) == 0:
        raise ValueError("x and y must contain at least one point.")
    if len(x_arr) == 1:
        return np.zeros(1, dtype=float)
    return np.gradient(y_arr, x_arr, edge_order=1).astype(float)


def find_concave_knee_index(x: np.ndarray, y: np.ndarray) -> int:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or len(x_arr) != len(y_arr):
        raise ValueError("x and y must be one-dimensional arrays with the same length.")
    if len(x_arr) == 0:
        raise ValueError("x and y must contain at least one point.")
    if len(x_arr) == 1:
        return 0

    x_span = float(x_arr.max() - x_arr.min())
    y_span = float(y_arr.max() - y_arr.min())
    if x_span <= 1e-12 or y_span <= 1e-12:
        return 0

    x_norm = (x_arr - x_arr.min()) / x_span
    y_norm = (y_arr - y_arr.min()) / y_span
    if y_norm[-1] <= y_norm[0] + 1e-12:
        return 0

    return int(np.argmax(y_norm - x_norm))
