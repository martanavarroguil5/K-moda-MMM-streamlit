from __future__ import annotations

import pickle
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from src.common.config import CONFIG
from src.common.parallel import parallel_kwargs
from src.modeling.arimax import ArimaxModelPackage, _forecast_package


BUDGET_TOTAL = 12_000_000.0
MIN_SHARE = 0.03
MAX_SHARE = 0.30
TARGET_YEAR = 2024
SENSITIVITY_MULTIPLIERS = [0.70, 0.85, 1.00, 1.15, 1.30]
SHARE_WARNING_TOLERANCE_PCT_POINTS = 0.25


def load_model() -> ArimaxModelPackage:
    with CONFIG.final_model_file.open("rb") as handle:
        return pickle.load(handle)


def _seed_previous_values(package: ArimaxModelPackage, target_year: int) -> dict[str, float]:
    history = package.weekly_df[package.weekly_df["year"] < target_year].copy()
    if history.empty:
        return {}
    last_row = history.sort_values("semana_inicio").iloc[-1]
    values = {
        "budget_total_eur": float(last_row.get("budget_total_eur", 0.0)),
    }
    for media_col in package.media_columns:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        values[media_col] = float(last_row.get(media_col, 0.0))
        values[share_col] = float(last_row.get(share_col, 0.0))
    return values


def _add_lagged_exogenous_columns(
    package: ArimaxModelPackage,
    scenario_df: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    lagged_columns = [column for column in package.exog_columns if column.endswith("_lag1")]
    if not lagged_columns:
        return scenario_df

    seeded = _seed_previous_values(package, target_year)
    for lagged_col in lagged_columns:
        base_col = lagged_col[: -len("_lag1")]
        if base_col not in scenario_df.columns:
            continue
        scenario_df[lagged_col] = scenario_df[base_col].shift(1)
        scenario_df.loc[scenario_df.index[0], lagged_col] = seeded.get(base_col, 0.0)
        scenario_df[lagged_col] = scenario_df[lagged_col].fillna(0.0)
    return scenario_df


def annual_channel_weights(package: ArimaxModelPackage, target_year: int = TARGET_YEAR) -> Dict[str, np.ndarray]:
    year_df = package.weekly_df[package.weekly_df["year"] == target_year].copy()
    weights = {}
    for media_col in package.media_columns:
        values = year_df[media_col].to_numpy(dtype=float)
        total = values.sum()
        weights[media_col] = np.repeat(1.0 / len(year_df), len(year_df)) if total <= 0 else values / total
    return weights


def channel_budgets_from_shares(shares: np.ndarray, media_cols: list[str]) -> Dict[str, float]:
    return {media_cols[idx]: float(shares[idx] * BUDGET_TOTAL) for idx in range(len(media_cols))}


def channel_shares_from_budgets(channel_budget_map: Dict[str, float], media_cols: list[str]) -> np.ndarray:
    total = sum(channel_budget_map[channel] for channel in media_cols)
    if total <= 0:
        return np.repeat(1.0 / len(media_cols), len(media_cols))
    return np.array([channel_budget_map[channel] / total for channel in media_cols], dtype=float)


def _historical_share_frame(package: ArimaxModelPackage, target_year: int) -> pd.DataFrame:
    annual = package.weekly_df.groupby("year")[package.media_columns].sum()
    annual = annual[annual.sum(axis=1) > 0.0].copy()
    history = annual[annual.index <= target_year].copy()
    if history.empty:
        history = annual.copy()
    return history.div(history.sum(axis=1), axis=0)


def _historical_share_envelope(
    package: ArimaxModelPackage,
    target_year: int,
    tolerance_pct_points: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    share_history = _historical_share_frame(package, target_year)
    tolerance = tolerance_pct_points / 100.0
    lower = np.clip(share_history.min(axis=0).to_numpy(dtype=float) - tolerance, MIN_SHARE, MAX_SHARE)
    upper = np.clip(share_history.max(axis=0).to_numpy(dtype=float) + tolerance, MIN_SHARE, MAX_SHARE)
    return lower, upper


def _repair_shares(shares: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    target_total = 1.0
    if float(lower.sum()) - 1e-12 > target_total or float(upper.sum()) + 1e-12 < target_total:
        raise ValueError("Share bounds are infeasible for a simplex total of 1.0.")

    candidate = np.asarray(shares, dtype=float)
    if not np.isfinite(candidate).all():
        candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
    if float(candidate.sum()) <= 0.0:
        candidate = lower.copy()
    candidate = np.clip(candidate, lower, upper)

    for _ in range(100):
        diff = target_total - float(candidate.sum())
        if abs(diff) < 1e-10:
            break
        room = (upper - candidate) if diff > 0.0 else (candidate - lower)
        room = np.clip(room, 0.0, None)
        room_total = float(room.sum())
        if room_total <= 1e-12:
            break
        candidate = candidate + diff * (room / room_total)
        candidate = np.clip(candidate, lower, upper)

    residual = target_total - float(candidate.sum())
    if abs(residual) > 1e-8:
        direction_room = np.clip((upper - candidate) if residual > 0.0 else (candidate - lower), 0.0, None)
        movable = np.where(direction_room > 1e-12)[0]
        if len(movable) == 0:
            raise ValueError("Unable to project channel shares onto feasible bounds.")
        step = residual / float(len(movable))
        for idx in movable:
            candidate[idx] = np.clip(candidate[idx] + step, lower[idx], upper[idx])
        residual = target_total - float(candidate.sum())
        if abs(residual) > 1e-8:
            room = np.clip((upper - candidate) if residual > 0.0 else (candidate - lower), 0.0, None)
            room_total = float(room.sum())
            if room_total <= 1e-12:
                raise ValueError("Unable to satisfy share bounds after projection.")
            candidate = candidate + residual * (room / room_total)
            candidate = np.clip(candidate, lower, upper)

    if abs(target_total - float(candidate.sum())) > 1e-8:
        raise ValueError("Projected shares do not sum to the required total within tolerance.")
    return candidate


def _scenario_frame(
    package: ArimaxModelPackage,
    channel_budget_map: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> pd.DataFrame:
    scenario_df = package.test_df[package.test_df["year"] == target_year].copy().reset_index(drop=True)
    weights = annual_channel_weights(package, target_year)
    for media_col in package.media_columns:
        scenario_df[media_col] = channel_budget_map[media_col] * weights[media_col]

    scenario_df["budget_total_eur"] = scenario_df[package.media_columns].sum(axis=1)
    positive_budget = scenario_df["budget_total_eur"] > 0
    for media_col in package.media_columns:
        share_col = media_col.replace("media_", "budget_share_pct_", 1)
        scenario_df[share_col] = 0.0
        scenario_df.loc[positive_budget, share_col] = (
            scenario_df.loc[positive_budget, media_col] / scenario_df.loc[positive_budget, "budget_total_eur"] * 100.0
        )
    return _add_lagged_exogenous_columns(package, scenario_df, target_year)


def scenario_frame(
    package: ArimaxModelPackage,
    channel_budget_map: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> pd.DataFrame:
    return _scenario_frame(package, channel_budget_map, target_year=target_year)


def _zero_media_budgets(package: ArimaxModelPackage) -> Dict[str, float]:
    return {media_col: 0.0 for media_col in package.media_columns}


def _attach_baseline_decomposition(
    package: ArimaxModelPackage,
    scenario_df: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    zero_media_df = _scenario_frame(package, _zero_media_budgets(package), target_year=target_year)
    zero_media_pred = _forecast_package(package, zero_media_df)
    total_media_contribution = (scenario_df["pred"] - zero_media_pred).clip(lower=0.0)
    scenario_df["base_sales"] = scenario_df["pred"] - total_media_contribution
    scenario_df["total_media_contribution"] = total_media_contribution
    scenario_df["predicted_base_gross_profit"] = scenario_df["base_sales"] * scenario_df["gross_margin_rate"]
    return scenario_df


def predict_scenario(
    package: ArimaxModelPackage,
    channel_budget_map: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> tuple[float, float, float]:
    scenario_df = predict_scenario_weekly(package, channel_budget_map, target_year)
    scenario_df["predicted_gross_profit"] = scenario_df["pred"] * scenario_df["gross_margin_rate"]
    total_sales = float(scenario_df["pred"].sum())
    total_profit = float(scenario_df["predicted_gross_profit"].sum())
    media_component = float(scenario_df["total_media_contribution"].sum())
    return total_sales, total_profit, media_component


def predict_scenario_weekly(
    package: ArimaxModelPackage,
    channel_budget_map: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> pd.DataFrame:
    scenario_df = _scenario_frame(package, channel_budget_map, target_year)
    scenario_df["pred"] = _forecast_package(package, scenario_df).to_numpy(dtype=float)
    scenario_df = _attach_baseline_decomposition(package, scenario_df, target_year=target_year)
    scenario_df["predicted_gross_profit"] = scenario_df["pred"] * scenario_df["gross_margin_rate"]
    return scenario_df


def _normalize_with_bounds(channel_budget_map: Dict[str, float]) -> Dict[str, float]:
    channels = list(channel_budget_map.keys())
    if not channels:
        return {}
    shares = np.array([channel_budget_map[channel] / BUDGET_TOTAL for channel in channels], dtype=float)
    bounds = [(MIN_SHARE, MAX_SHARE) for _ in channels]
    repaired = _repair_shares(shares, bounds)
    return {channel: float(repaired[idx] * BUDGET_TOTAL) for idx, channel in enumerate(channels)}


def build_relative_bounds(
    media_cols: list[str],
    base_shares: np.ndarray,
    max_relative_change: float | None = None,
) -> list[tuple[float, float]]:
    bounds = []
    for idx, _channel in enumerate(media_cols):
        lower = MIN_SHARE
        upper = MAX_SHARE
        if max_relative_change is not None:
            lower = max(lower, float(base_shares[idx] * (1.0 - max_relative_change)))
            upper = min(upper, float(base_shares[idx] * (1.0 + max_relative_change)))
        if lower > upper:
            anchor = min(MAX_SHARE, max(MIN_SHARE, float(base_shares[idx])))
            lower = anchor
            upper = anchor
        bounds.append((lower, upper))
    return bounds


def marginal_roi(
    package: ArimaxModelPackage,
    base_budgets: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> pd.DataFrame:
    base_total_sales, base_total_profit, _ = predict_scenario(package, base_budgets, target_year)

    def _marginal_roi_row(channel: str, budget: float) -> dict[str, float | str]:
        delta = max(10_000.0, budget * 0.01)
        scenario = dict(base_budgets)
        scenario[channel] += delta
        donor_channels = [name for name in scenario if name != channel]
        donor_total = sum(base_budgets[name] for name in donor_channels)
        for donor in donor_channels:
            scenario[donor] -= delta * (base_budgets[donor] / donor_total)
        scenario = _normalize_with_bounds(scenario)
        total_sales, total_profit, _ = predict_scenario(package, scenario, target_year)
        return {
            "channel": channel.replace("media_", ""),
            "base_budget": budget,
            "delta_budget": delta,
            "mroi_sales": (total_sales - base_total_sales) / delta,
            "mroi_profit": (total_profit - base_total_profit) / delta,
            "mroi": (total_profit - base_total_profit) / delta,
        }

    rows = Parallel(**parallel_kwargs(len(base_budgets), backend="threading"))(
        delayed(_marginal_roi_row)(channel, budget) for channel, budget in base_budgets.items()
    )
    return pd.DataFrame(rows).sort_values("mroi_profit", ascending=False).reset_index(drop=True)


def optimize_budget(
    package: ArimaxModelPackage,
    start_shares: np.ndarray,
    extra_starts: list[np.ndarray] | None = None,
    max_relative_change: float | None = None,
    deviation_penalty: float = 0.15,
    history_penalty: float = 0.60,
    concentration_penalty: float = 0.05,
    history_tolerance_pct_points: float = 0.50,
    target_year: int = TARGET_YEAR,
) -> Dict[str, float]:
    media_cols = package.media_columns
    start_shares = np.asarray(start_shares, dtype=float)
    hist_lower, hist_upper = _historical_share_envelope(
        package,
        target_year=target_year,
        tolerance_pct_points=history_tolerance_pct_points,
    )

    def objective(shares: np.ndarray) -> float:
        repaired = _repair_shares(shares, bounds)
        budgets = channel_budgets_from_shares(repaired, media_cols)
        _, total_profit, _ = predict_scenario(package, budgets, target_year=target_year)
        share_shift_l1 = float(np.abs(repaired - start_shares).sum())
        envelope_gap = float(np.clip(hist_lower - repaired, 0.0, None).sum() + np.clip(repaired - hist_upper, 0.0, None).sum())
        concentration = float(np.square(repaired - start_shares).sum())
        penalty = BUDGET_TOTAL * (
            deviation_penalty * share_shift_l1
            + history_penalty * envelope_gap
            + concentration_penalty * concentration
        )
        return -(total_profit - penalty)

    constraints = [{"type": "eq", "fun": lambda shares: float(np.sum(shares) - 1.0)}]
    bounds = build_relative_bounds(media_cols, start_shares, max_relative_change=max_relative_change)
    rng = np.random.default_rng(42)
    starts = [start_shares, np.repeat(1.0 / len(media_cols), len(media_cols))]
    if extra_starts:
        starts.extend(extra_starts)
    for _ in range(10):
        proposal = rng.dirichlet(np.ones(len(media_cols)))
        proposal = _repair_shares(proposal, bounds)
        starts.append(proposal)

    def _optimize_from_start(start: np.ndarray) -> tuple[np.ndarray, float]:
        start = _repair_shares(start, bounds)
        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        candidate = _repair_shares(result.x if result.success else start, bounds)
        return candidate, objective(candidate)

    optimized_candidates = Parallel(**parallel_kwargs(len(starts), backend="threading"))(
        delayed(_optimize_from_start)(start) for start in starts
    )
    best_shares, _best_objective = min(optimized_candidates, key=lambda item: item[1])
    return channel_budgets_from_shares(best_shares, media_cols)


def shift_budget(
    base_budgets: Dict[str, float],
    ranking: pd.DataFrame,
    total_shift_share: float,
    top_k: int,
    bottom_k: int,
) -> Dict[str, float]:
    scenario = dict(base_budgets)
    shift_amount = BUDGET_TOTAL * total_shift_share
    top_channels = [f"media_{channel}" for channel in ranking.head(top_k)["channel"].tolist()]
    bottom_channels = [f"media_{channel}" for channel in ranking.tail(bottom_k)["channel"].tolist()]

    for channel in bottom_channels:
        scenario[channel] -= shift_amount / len(bottom_channels)
    for channel in top_channels:
        scenario[channel] += shift_amount / len(top_channels)

    return _normalize_with_bounds(scenario)


def equal_share_budgets(media_cols: list[str]) -> Dict[str, float]:
    equal_share = 1.0 / len(media_cols)
    return {channel: equal_share * BUDGET_TOTAL for channel in media_cols}


def scenario_from_channel_multiplier(
    base_budgets: Dict[str, float],
    target_channel: str,
    multiplier: float,
) -> Dict[str, float]:
    scenario = dict(base_budgets)
    original_value = base_budgets[target_channel]
    new_value = original_value * multiplier
    delta = new_value - original_value
    scenario[target_channel] = new_value

    donor_channels = [channel for channel in scenario if channel != target_channel]
    donor_total = sum(base_budgets[channel] for channel in donor_channels)
    for donor in donor_channels:
        scenario[donor] -= delta * (base_budgets[donor] / donor_total)
    return _normalize_with_bounds(scenario)


def scenario_diagnostics_row(
    scenario_name: str,
    budgets: Dict[str, float],
    base_budgets: Dict[str, float],
    media_cols: list[str],
    warnings: str,
) -> dict[str, float | str]:
    base_shares = channel_shares_from_budgets(base_budgets, media_cols)
    scenario_shares = channel_shares_from_budgets(budgets, media_cols)
    share_shift = np.abs(scenario_shares - base_shares)
    budget_shift = np.abs(
        np.array([budgets[channel] - base_budgets[channel] for channel in media_cols], dtype=float)
    )
    warning_count = 0 if warnings == "none" else len([item for item in warnings.split(",") if item.strip()])
    return {
        "scenario": scenario_name,
        "max_share_shift_pct_points": float(share_shift.max() * 100.0),
        "mean_share_shift_pct_points": float(share_shift.mean() * 100.0),
        "max_budget_shift_eur": float(budget_shift.max()),
        "total_budget_reallocated_eur": float(budget_shift.sum() / 2.0),
        "warning_count": int(warning_count),
    }


def channel_budget_sensitivity(
    package: ArimaxModelPackage,
    base_budgets: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> pd.DataFrame:
    base_sales, base_profit, _ = predict_scenario(package, base_budgets, target_year)

    def _sensitivity_row(channel: str, multiplier: float) -> dict[str, float | str]:
        scenario = scenario_from_channel_multiplier(base_budgets, channel, multiplier)
        total_sales, total_profit, _ = predict_scenario(package, scenario, target_year)
        return {
            "channel": channel.replace("media_", ""),
            "multiplier": multiplier,
            "predicted_sales_2024": total_sales,
            "predicted_gross_profit_2024": total_profit,
            "delta_sales_vs_base": total_sales - base_sales,
            "delta_profit_vs_base": total_profit - base_profit,
        }

    tasks = [(channel, multiplier) for channel in package.media_columns for multiplier in SENSITIVITY_MULTIPLIERS]
    rows = Parallel(**parallel_kwargs(len(tasks), backend="threading"))(
        delayed(_sensitivity_row)(channel, multiplier) for channel, multiplier in tasks
    )
    return pd.DataFrame(rows)


def annual_range_warnings(
    package: ArimaxModelPackage,
    scenario_budgets: Dict[str, float],
    target_year: int = TARGET_YEAR,
) -> str:
    lower, upper = _historical_share_envelope(
        package,
        target_year=target_year,
        tolerance_pct_points=SHARE_WARNING_TOLERANCE_PCT_POINTS,
    )
    scenario_shares = channel_shares_from_budgets(scenario_budgets, package.media_columns)
    warnings = []
    for idx, media_col in enumerate(package.media_columns):
        if scenario_shares[idx] < lower[idx] or scenario_shares[idx] > upper[idx]:
            warnings.append(media_col.replace("media_", ""))
    return ", ".join(sorted(warnings)) if warnings else "none"
