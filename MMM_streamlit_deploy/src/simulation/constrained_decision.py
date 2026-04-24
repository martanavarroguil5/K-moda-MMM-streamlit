from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize

from src.common.config import CONFIG
from src.common.parallel import parallel_kwargs
from src.modeling.constrained_mmm import (
    DEFAULT_TRAIN_START_YEAR,
    ELIGIBILITY_TABLE,
    TARGET_YEAR,
    _fit_final_model,
    _load_geo_panel,
    run_constrained_mmm,
)
from src.modeling.trainer import media_columns
from src.modeling.transforms import add_transformed_media_features


WEIGHTS_TABLE = CONFIG.reports_tables_dir / "constrained_decision_weights.csv"
SCENARIOS_TABLE = CONFIG.reports_tables_dir / "constrained_decision_scenarios.csv"
CONSTRAINTS_TABLE = CONFIG.reports_tables_dir / "constrained_decision_constraints.csv"
STAT_TESTS_TABLE = CONFIG.reports_tables_dir / "constrained_decision_stat_tests.csv"
WEEKLY_SCENARIOS_FILE = CONFIG.processed_dir / "constrained_decision_weekly_scenarios.csv"
SUMMARY_JSON = CONFIG.processed_dir / "constrained_decision_summary.json"
REPORT_MD = CONFIG.docs_dir / "constrained_decision_report.md"

MIN_SHARE = 0.03
MAX_SHARE = 0.30
MAX_RELATIVE_CHANGE = 0.25
HISTORY_TOLERANCE_PCT_POINTS = 0.50
OPTIMIZER_RANDOM_STARTS = 16
OPTIMIZER_SEED = 42


@dataclass(frozen=True)
class ConstrainedDecisionPackage:
    dataset: pd.DataFrame
    model: object
    eligibility: pd.DataFrame
    media_cols: list[str]
    transform_spec: dict[str, float | int | str]
    target_year: int
    train_start_year: int
    observed_budget_eur: float
    planning_budget_eur: float
    observed_budgets: dict[str, float]
    base_budgets: dict[str, float]
    annual_profiles: dict[str, np.ndarray]


def channel_shares_from_budgets(
    channel_budget_map: dict[str, float],
    media_cols: list[str],
) -> np.ndarray:
    total = float(sum(channel_budget_map[channel] for channel in media_cols))
    if total <= 0.0:
        return np.repeat(1.0 / len(media_cols), len(media_cols))
    return np.array([channel_budget_map[channel] / total for channel in media_cols], dtype=float)


def channel_budgets_from_shares(
    shares: np.ndarray,
    media_cols: list[str],
    total_budget: float,
) -> dict[str, float]:
    return {media_cols[idx]: float(shares[idx] * total_budget) for idx in range(len(media_cols))}


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return float("nan")
    return float(numerator / denominator)


def _scenario_family(name: str) -> str:
    if name == "do_nothing_zero_media":
        return "baseline_without_investment"
    if name in {"do_nothing_observed", "do_nothing_historical"}:
        return "do_nothing_active"
    return "do_something_optimized"


def _fill_margin_rates(dataset: pd.DataFrame) -> pd.DataFrame:
    sales = pd.read_parquet(CONFIG.geo_weekly_sales_file)[["semana_inicio", "ciudad", "margen_bruto_ponderado"]].copy()
    sales["semana_inicio"] = pd.to_datetime(sales["semana_inicio"])
    sales = sales.rename(columns={"margen_bruto_ponderado": "gross_margin_rate"})
    out = dataset.merge(sales, on=["semana_inicio", "ciudad"], how="left")
    out["gross_margin_rate"] = out.groupby("ciudad")["gross_margin_rate"].transform(
        lambda series: series.replace(0.0, np.nan).fillna(series.replace(0.0, np.nan).mean())
    )
    global_margin = float(out["gross_margin_rate"].replace(0.0, np.nan).mean())
    out["gross_margin_rate"] = out["gross_margin_rate"].fillna(global_margin).clip(lower=0.0)
    return out


def _load_decision_package(
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    target_year: int = TARGET_YEAR,
    planning_budget_eur: float | None = None,
) -> ConstrainedDecisionPackage:
    if not CONFIG.constrained_model_results_file.exists() or not ELIGIBILITY_TABLE.exists():
        run_constrained_mmm(train_start_year=train_start_year)

    results = json.loads(CONFIG.constrained_model_results_file.read_text(encoding="utf-8"))
    spec = results["transform_spec"]
    dataset = _fill_margin_rates(_load_geo_panel())
    model, _test_df, _scored, _baseline = _fit_final_model(dataset=dataset, train_start_year=train_start_year, spec=spec)
    eligibility = pd.read_csv(ELIGIBILITY_TABLE)
    media_cols = media_columns(dataset)
    target_df = (
        dataset.loc[dataset["year"] == target_year, ["semana_inicio", "ciudad"] + media_cols]
        .sort_values(["ciudad", "semana_inicio"])
        .reset_index(drop=True)
    )

    annual_profiles: dict[str, np.ndarray] = {}
    for media_col in media_cols:
        values = target_df[media_col].to_numpy(dtype=float)
        total = float(values.sum())
        annual_profiles[media_col] = np.repeat(1.0 / len(values), len(values)) if total <= 0.0 else values / total

    observed_budgets = {media_col: float(target_df[media_col].sum()) for media_col in media_cols}
    observed_budget_eur = float(sum(observed_budgets.values()))
    planning_budget_value = float(planning_budget_eur) if planning_budget_eur is not None else observed_budget_eur
    base_shares = channel_shares_from_budgets(observed_budgets, media_cols)
    base_budgets = channel_budgets_from_shares(base_shares, media_cols, planning_budget_value)
    return ConstrainedDecisionPackage(
        dataset=dataset,
        model=model,
        eligibility=eligibility,
        media_cols=media_cols,
        transform_spec=spec,
        target_year=target_year,
        train_start_year=train_start_year,
        observed_budget_eur=observed_budget_eur,
        planning_budget_eur=planning_budget_value,
        observed_budgets=observed_budgets,
        base_budgets=base_budgets,
        annual_profiles=annual_profiles,
    )


def _share_history(package: ConstrainedDecisionPackage) -> pd.DataFrame:
    annual = (
        package.dataset.loc[
            (package.dataset["year"] >= package.train_start_year) & (package.dataset["year"] <= package.target_year),
            ["year"] + package.media_cols,
        ]
        .groupby("year")[package.media_cols]
        .sum()
    )
    annual = annual.loc[annual.sum(axis=1) > 0.0].copy()
    return annual.div(annual.sum(axis=1), axis=0)


def _build_channel_constraints(package: ConstrainedDecisionPackage) -> pd.DataFrame:
    share_history = _share_history(package)
    base_shares = channel_shares_from_budgets(package.base_budgets, package.media_cols)
    tolerance = HISTORY_TOLERANCE_PCT_POINTS / 100.0
    lower = np.clip(share_history.min(axis=0).to_numpy(dtype=float) - tolerance, MIN_SHARE, MAX_SHARE)
    upper = np.clip(share_history.max(axis=0).to_numpy(dtype=float) + tolerance, MIN_SHARE, MAX_SHARE)
    relative_lower = np.clip(base_shares * (1.0 - MAX_RELATIVE_CHANGE), MIN_SHARE, MAX_SHARE)
    relative_upper = np.clip(base_shares * (1.0 + MAX_RELATIVE_CHANGE), MIN_SHARE, MAX_SHARE)
    lower = np.maximum(lower, relative_lower)
    upper = np.minimum(upper, relative_upper)

    for idx in range(len(package.media_cols)):
        if lower[idx] > upper[idx]:
            lower[idx] = base_shares[idx]
            upper[idx] = base_shares[idx]

    status_map = package.eligibility.set_index("channel")["optimization_status"].to_dict()
    rows = []
    for idx, media_col in enumerate(package.media_cols):
        channel = media_col.replace("media_", "", 1)
        rows.append(
            {
                "channel": channel,
                "media_col": media_col,
                "historical_share_pct": float(base_shares[idx] * 100.0),
                "lower_share_pct": float(lower[idx] * 100.0),
                "upper_share_pct": float(upper[idx] * 100.0),
                "optimization_role": "eligible" if status_map.get(channel, "hold") == "eligible" else "hold",
            }
        )
    return pd.DataFrame(rows)


def _hold_pool_structure(
    package: ConstrainedDecisionPackage,
    constraints: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, float], float, float]:
    eligible_media = constraints.loc[constraints["optimization_role"] == "eligible", "media_col"].tolist()
    hold_media = [media_col for media_col in package.media_cols if media_col not in eligible_media]
    base_shares = channel_shares_from_budgets(package.base_budgets, package.media_cols)
    share_map = dict(zip(package.media_cols, base_shares))

    if not hold_media:
        return eligible_media, hold_media, {}, 0.0, 0.0

    hold_total = float(sum(share_map[media_col] for media_col in hold_media))
    hold_ratios = {
        media_col: (share_map[media_col] / hold_total if hold_total > 0.0 else 1.0 / len(hold_media))
        for media_col in hold_media
    }
    lower_residual = 0.0
    upper_residual = 1.0
    for media_col in hold_media:
        row = constraints.loc[constraints["media_col"] == media_col].iloc[0]
        ratio = hold_ratios[media_col]
        if ratio <= 0.0:
            continue
        lower_residual = max(lower_residual, float(row["lower_share_pct"]) / 100.0 / ratio)
        upper_residual = min(upper_residual, float(row["upper_share_pct"]) / 100.0 / ratio)

    base_residual = hold_total
    if lower_residual > upper_residual:
        lower_residual = base_residual
        upper_residual = base_residual
    return eligible_media, hold_media, hold_ratios, lower_residual, upper_residual


def _project_eligible_vector(
    candidate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    min_total: float,
    max_total: float,
) -> np.ndarray:
    proposal = np.clip(np.asarray(candidate, dtype=float), lower, upper)
    for _ in range(50):
        total = float(proposal.sum())
        if min_total - 1e-10 <= total <= max_total + 1e-10:
            break
        if total < min_total:
            room = np.clip(upper - proposal, 0.0, None)
            room_total = float(room.sum())
            if room_total <= 1e-12:
                break
            proposal = proposal + (min_total - total) * (room / room_total)
        else:
            room = np.clip(proposal - lower, 0.0, None)
            room_total = float(room.sum())
            if room_total <= 1e-12:
                break
            proposal = proposal - (total - max_total) * (room / room_total)
        proposal = np.clip(proposal, lower, upper)
    return proposal


def _assemble_full_shares(
    package: ConstrainedDecisionPackage,
    eligible_media: list[str],
    hold_media: list[str],
    hold_ratios: dict[str, float],
    eligible_shares: np.ndarray,
) -> np.ndarray:
    share_map = {media_col: 0.0 for media_col in package.media_cols}
    for idx, media_col in enumerate(eligible_media):
        share_map[media_col] = float(eligible_shares[idx])

    residual_share = max(0.0, 1.0 - float(np.sum(eligible_shares)))
    for media_col in hold_media:
        share_map[media_col] = residual_share * hold_ratios[media_col]

    shares = np.array([share_map[media_col] for media_col in package.media_cols], dtype=float)
    total = float(shares.sum())
    return shares if total <= 0.0 else shares / total


def _scenario_target_frame(
    package: ConstrainedDecisionPackage,
    channel_budget_map: dict[str, float],
) -> pd.DataFrame:
    scenario_df = package.dataset.sort_values(["ciudad", "semana_inicio"]).copy().reset_index(drop=True)
    target_mask = scenario_df["year"] == package.target_year
    for media_col in package.media_cols:
        scenario_df.loc[target_mask, media_col] = channel_budget_map[media_col] * package.annual_profiles[media_col]

    transform_params = {
        media_col: {
            "lag": int(package.transform_spec["lag"]),
            "alpha": float(package.transform_spec["alpha"]),
            "saturation": str(package.transform_spec["saturation"]),
        }
        for media_col in package.media_cols
    }
    transformed = add_transformed_media_features(scenario_df, package.media_cols, transform_params)
    target_df = transformed.loc[transformed["year"] == package.target_year].copy().reset_index(drop=True)

    baseline_columns = [
        column for column in package.model.feature_columns if column not in package.model.media_feature_columns
    ]
    target_df["predicted_sales"] = (
        package.model.intercept
        + target_df[package.model.feature_columns].mul(package.model.coefficients, axis=1).sum(axis=1).to_numpy()
    )
    target_df["baseline_sales"] = (
        package.model.intercept
        + target_df[baseline_columns].mul(package.model.coefficients[baseline_columns], axis=1).sum(axis=1).to_numpy()
    )
    target_df["media_incremental_sales"] = target_df["predicted_sales"] - target_df["baseline_sales"]
    target_df["predicted_gross_profit"] = target_df["predicted_sales"] * target_df["gross_margin_rate"]
    target_df["baseline_gross_profit"] = target_df["baseline_sales"] * target_df["gross_margin_rate"]
    target_df["media_incremental_gross_profit"] = (
        target_df["predicted_gross_profit"] - target_df["baseline_gross_profit"]
    )
    return target_df


def predict_constrained_scenario(
    package: ConstrainedDecisionPackage,
    channel_budget_map: dict[str, float],
    scenario_name: str,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    target_df = _scenario_target_frame(package, channel_budget_map)
    weekly = (
        target_df.groupby("semana_inicio", as_index=False)
        .agg(
            actual_sales=("ventas_netas", "sum"),
            predicted_sales=("predicted_sales", "sum"),
            baseline_sales=("baseline_sales", "sum"),
            predicted_gross_profit=("predicted_gross_profit", "sum"),
            baseline_gross_profit=("baseline_gross_profit", "sum"),
        )
        .sort_values("semana_inicio")
        .reset_index(drop=True)
    )
    weekly["media_incremental_sales"] = weekly["predicted_sales"] - weekly["baseline_sales"]
    weekly["media_incremental_gross_profit"] = weekly["predicted_gross_profit"] - weekly["baseline_gross_profit"]
    weekly["scenario"] = scenario_name

    totals = {
        "scenario": scenario_name,
        "total_budget_eur": float(sum(channel_budget_map.values())),
        "predicted_sales_2024": float(weekly["predicted_sales"].sum()),
        "baseline_sales_2024": float(weekly["baseline_sales"].sum()),
        "media_incremental_sales_2024": float(weekly["media_incremental_sales"].sum()),
        "predicted_gross_profit_2024": float(weekly["predicted_gross_profit"].sum()),
        "baseline_gross_profit_2024": float(weekly["baseline_gross_profit"].sum()),
        "media_incremental_gross_profit_2024": float(weekly["media_incremental_gross_profit"].sum()),
    }
    return weekly, totals


def _paired_stat_test(candidate: pd.Series, baseline: pd.Series) -> dict[str, float | bool]:
    diff = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    n_obs = int(diff.shape[0])
    dfree = max(n_obs - 1, 1)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n_obs > 1 else 0.0
    stderr = std_diff / np.sqrt(n_obs) if n_obs > 0 else float("nan")
    t_stat = float(mean_diff / stderr) if stderr > 0 else (float("inf") if mean_diff > 0 else 0.0)
    p_value_one_sided = float(1.0 - stats.t.cdf(t_stat, df=dfree))
    critical_one_sided = float(stats.t.ppf(0.95, df=dfree))
    critical_two_sided = float(stats.t.ppf(0.975, df=dfree))
    ci_half_width = critical_two_sided * stderr if np.isfinite(stderr) else float("nan")
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(diff, size=(4000, n_obs), replace=True).mean(axis=1)
    return {
        "n_obs": n_obs,
        "mean_weekly_profit_delta": mean_diff,
        "std_weekly_profit_delta": std_diff,
        "stderr_weekly_profit_delta": stderr,
        "effect_size_cohen_d": _safe_divide(mean_diff, std_diff),
        "t_statistic": t_stat,
        "t_critical_one_sided_5pct": critical_one_sided,
        "p_value_one_sided": p_value_one_sided,
        "ci_low_95pct": mean_diff - ci_half_width,
        "ci_high_95pct": mean_diff + ci_half_width,
        "bootstrap_ci_low_95pct": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high_95pct": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_prob_positive": float(np.mean(bootstrap > 0.0)),
        "significant_one_sided_5pct": bool((t_stat > critical_one_sided) and (p_value_one_sided < 0.05)),
    }


def optimize_constrained_budget(
    package: ConstrainedDecisionPackage,
    constraints: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float | str]]:
    eligible_media, hold_media, hold_ratios, min_hold_total, max_hold_total = _hold_pool_structure(package, constraints)
    total_budget = float(sum(package.base_budgets.values()))
    if not eligible_media:
        return dict(package.base_budgets), {"status": "no_eligible_channels", "eligible_media": []}

    constraint_map = constraints.set_index("media_col")
    lower = np.array([float(constraint_map.loc[media_col, "lower_share_pct"]) / 100.0 for media_col in eligible_media], dtype=float)
    upper = np.array([float(constraint_map.loc[media_col, "upper_share_pct"]) / 100.0 for media_col in eligible_media], dtype=float)
    min_total = max(0.0, 1.0 - max_hold_total)
    max_total = min(1.0, 1.0 - min_hold_total)
    if min_total > max_total:
        base_total = float(sum(package.base_budgets[media_col] for media_col in eligible_media) / total_budget)
        min_total = base_total
        max_total = base_total
    base_candidate = np.array([package.base_budgets[media_col] / total_budget for media_col in eligible_media], dtype=float)
    base_candidate = _project_eligible_vector(base_candidate, lower, upper, min_total, max_total)

    def objective(candidate: np.ndarray) -> float:
        projected = _project_eligible_vector(candidate, lower, upper, min_total, max_total)
        full_shares = _assemble_full_shares(
            package=package,
            eligible_media=eligible_media,
            hold_media=hold_media,
            hold_ratios=hold_ratios,
            eligible_shares=projected,
        )
        budgets = channel_budgets_from_shares(full_shares, package.media_cols, total_budget)
        _weekly, totals = predict_constrained_scenario(package, budgets, scenario_name="candidate")
        return -float(totals["predicted_gross_profit_2024"])

    constraints_list = [
        {"type": "ineq", "fun": lambda shares, lb=min_total: float(np.sum(shares) - lb)},
        {"type": "ineq", "fun": lambda shares, ub=max_total: float(ub - np.sum(shares))},
    ]
    bounds = list(zip(lower, upper))
    rng = np.random.default_rng(OPTIMIZER_SEED)
    starts = [
        base_candidate,
        lower.copy(),
        upper.copy(),
        _project_eligible_vector((lower + upper) / 2.0, lower, upper, min_total, max_total),
    ]
    for _ in range(OPTIMIZER_RANDOM_STARTS):
        proposal = rng.uniform(lower, upper)
        starts.append(_project_eligible_vector(proposal, lower, upper, min_total, max_total))

    def _optimize_from_start(start: np.ndarray) -> tuple[np.ndarray, float]:
        start = _project_eligible_vector(start, lower, upper, min_total, max_total)
        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        candidate = _project_eligible_vector(result.x if result.success else start, lower, upper, min_total, max_total)
        return candidate, objective(candidate)

    candidates = Parallel(**parallel_kwargs(len(starts), backend="threading"))(
        delayed(_optimize_from_start)(start) for start in starts
    )
    best_candidate, best_objective = min(candidates, key=lambda item: item[1])
    best_shares = _assemble_full_shares(
        package=package,
        eligible_media=eligible_media,
        hold_media=hold_media,
        hold_ratios=hold_ratios,
        eligible_shares=best_candidate,
    )
    best_budgets = channel_budgets_from_shares(best_shares, package.media_cols, total_budget)
    optimization_meta = {
        "status": "optimized",
        "eligible_media": eligible_media,
        "hold_media": hold_media,
        "optimizer_objective": float(-best_objective),
        "eligible_share_min_pct": float(min_total * 100.0),
        "eligible_share_max_pct": float(max_total * 100.0),
    }
    return best_budgets, optimization_meta


def _round_to_100(weights: pd.Series) -> pd.Series:
    if weights.empty:
        return weights
    scaled = weights / weights.sum() * 100.0
    floored = np.floor(scaled).astype(int)
    remainder = int(100 - floored.sum())
    fractional = (scaled - floored).sort_values(ascending=False)
    rounded = floored.copy()
    for idx in fractional.index[:remainder]:
        rounded.loc[idx] += 1
    return rounded.astype(int)


def _confidence_level(status: str, bootstrap_selection_rate: float) -> str:
    if status == "eligible" and bootstrap_selection_rate >= 0.80:
        return "high"
    if status == "eligible":
        return "medium"
    return "low"


def _note_for_row(row: pd.Series) -> str:
    if row["model_status"] == "eligible":
        return "Stable sign and bootstrap selection support individual optimization."
    if row["bootstrap_selection_rate"] >= 0.10:
        return "Weak model signal; keep in locked pool and review with external calibration."
    return "No stable standalone signal; keep in locked pool and avoid individual reallocation."


def _build_executive_weights(
    package: ConstrainedDecisionPackage,
    optimized_budgets: dict[str, float],
) -> pd.DataFrame:
    weights = package.eligibility.copy()
    historical_shares = channel_shares_from_budgets(package.base_budgets, package.media_cols)
    optimized_shares = channel_shares_from_budgets(optimized_budgets, package.media_cols)
    historical_share_map = dict(zip(package.media_cols, historical_shares))
    optimized_share_map = dict(zip(package.media_cols, optimized_shares))

    eligible_mask = weights["optimization_status"] == "eligible"
    published = pd.Series(0, index=weights.index, dtype=int)
    if eligible_mask.any():
        eligible_contributions = weights.loc[eligible_mask, "contribution_mean"].clip(lower=0.0)
        published.loc[eligible_mask] = _round_to_100(eligible_contributions)

    weights["historical_share_pct"] = weights["channel"].map(
        lambda channel: float(historical_share_map[f"media_{channel}"] * 100.0)
    )
    weights["recommended_share_pct"] = weights["channel"].map(
        lambda channel: float(optimized_share_map[f"media_{channel}"] * 100.0)
    )
    weights["recommended_budget_eur"] = weights["channel"].map(
        lambda channel: float(optimized_budgets[f"media_{channel}"])
    )
    weights["published_weight_pct"] = published
    weights["confidence_level"] = weights.apply(
        lambda row: _confidence_level(str(row["optimization_status"]), float(row["bootstrap_selection_rate"])),
        axis=1,
    )
    weights["model_status"] = weights["optimization_status"]
    weights["action"] = np.where(
        weights["model_status"] == "eligible",
        "optimize_individually",
        "keep_locked_pool",
    )
    weights["note"] = weights.apply(_note_for_row, axis=1)

    return (
        weights[
            [
                "channel",
                "historical_share_pct",
                "recommended_share_pct",
                "recommended_budget_eur",
                "published_weight_pct",
                "confidence_level",
                "model_status",
                "bootstrap_selection_rate",
                "contribution_mean",
                "roi_mean",
                "action",
                "note",
            ]
        ]
        .rename(columns={"contribution_mean": "contribution_mean_eur"})
        .sort_values(["published_weight_pct", "recommended_share_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _scenario_diagnostics_row(
    scenario_name: str,
    budgets: dict[str, float],
    observed_budgets: dict[str, float],
    historical_budgets: dict[str, float],
    media_cols: list[str],
    totals: dict[str, float | str],
) -> dict[str, float | str]:
    observed_shares = channel_shares_from_budgets(observed_budgets, media_cols)
    historical_shares = channel_shares_from_budgets(historical_budgets, media_cols)
    current_shares = channel_shares_from_budgets(budgets, media_cols)
    share_shift_historical = np.abs(current_shares - historical_shares)
    share_shift_observed = np.abs(current_shares - observed_shares)
    budget_shift_historical = np.abs(
        np.array([budgets[channel] - historical_budgets[channel] for channel in media_cols], dtype=float)
    )
    budget_shift_observed = np.abs(
        np.array([budgets[channel] - observed_budgets[channel] for channel in media_cols], dtype=float)
    )
    return {
        "scenario": scenario_name,
        "scenario_family": _scenario_family(scenario_name),
        "predicted_sales_2024": float(totals["predicted_sales_2024"]),
        "baseline_sales_2024": float(totals["baseline_sales_2024"]),
        "media_incremental_sales_2024": float(totals["media_incremental_sales_2024"]),
        "predicted_gross_profit_2024": float(totals["predicted_gross_profit_2024"]),
        "baseline_gross_profit_2024": float(totals["baseline_gross_profit_2024"]),
        "media_incremental_gross_profit_2024": float(totals["media_incremental_gross_profit_2024"]),
        "total_budget_eur": float(sum(budgets.values())),
        "max_share_shift_vs_historical_pct_points": float(share_shift_historical.max() * 100.0),
        "mean_share_shift_vs_historical_pct_points": float(share_shift_historical.mean() * 100.0),
        "total_budget_reallocated_vs_historical_eur": float(budget_shift_historical.sum() / 2.0),
        "max_share_shift_vs_observed_pct_points": float(share_shift_observed.max() * 100.0),
        "mean_share_shift_vs_observed_pct_points": float(share_shift_observed.mean() * 100.0),
        "total_budget_reallocated_vs_observed_eur": float(budget_shift_observed.sum() / 2.0),
    }


def _write_report(
    package: ConstrainedDecisionPackage,
    optimization_meta: dict[str, float | str],
    constraints: pd.DataFrame,
    weights: pd.DataFrame,
    scenarios: pd.DataFrame,
    stat_tests: pd.DataFrame,
) -> None:
    optimized = scenarios.loc[scenarios["scenario"] == "do_something_optimized"].iloc[0]
    historical = scenarios.loc[scenarios["scenario"] == "do_nothing_historical"].iloc[0]
    observed = scenarios.loc[scenarios["scenario"] == "do_nothing_observed"].iloc[0]
    zero_media = scenarios.loc[scenarios["scenario"] == "do_nothing_zero_media"].iloc[0]
    eligible = weights.loc[weights["model_status"] == "eligible", "channel"].tolist()
    lines = [
        "# Constrained Decision Report",
        "",
        "## Objetivo",
        "",
        "Construir una capa operativa limpia sobre el ConstrainedMMM que convierta elegibilidad tecnica en una recomendacion ejecutiva sin vender precision falsa.",
        "",
        "## Reglas Operativas",
        "",
        f"- Presupuesto observado en `{package.target_year}`: `{package.observed_budget_eur:,.2f} EUR`.",
        f"- Presupuesto de planning optimizado en esta corrida: `{historical['total_budget_eur']:,.2f} EUR`.",
        "- Solo se optimizan de forma individual los canales con `optimization_status = eligible`.",
        "- Los canales `hold` permanecen en un pool bloqueado y conservan sus proporciones historicas relativas dentro de ese pool.",
        f"- Guardrails de share por canal: interseccion entre `[{MIN_SHARE:.0%}, {MAX_SHARE:.0%}]`, envolvente historica `2021-{package.target_year}` con tolerancia de `{HISTORY_TOLERANCE_PCT_POINTS:.2f}` puntos y cap relativo de `+/-{MAX_RELATIVE_CHANGE:.0%}`.",
        "",
        "## Escenarios Clave",
        "",
        f"- Canales elegibles: `{', '.join(eligible) if eligible else 'ninguno'}`.",
        f"- Estado de la optimizacion: `{optimization_meta['status']}`.",
        f"- `do_nothing_observed`: seguir como en `2024`, con `{observed['predicted_gross_profit_2024']:,.2f} EUR` de beneficio bruto estimado.",
        f"- `do_nothing_historical`: gastar el budget de planning con el mix historico, con `{historical['predicted_gross_profit_2024']:,.2f} EUR`.",
        f"- `do_nothing_zero_media`: contrafactual sin inversion, con `{zero_media['predicted_gross_profit_2024']:,.2f} EUR`.",
        f"- `do_something_optimized`: mix optimizado solo dentro del subconjunto fiable, con `{optimized['predicted_gross_profit_2024']:,.2f} EUR`.",
        f"- Uplift estimado del optimizado vs `do_nothing_historical`: `{optimized['profit_vs_historical_eur']:,.2f} EUR`.",
        f"- Uplift estimado del optimizado vs `do_nothing_zero_media`: `{optimized['profit_vs_zero_media_eur']:,.2f} EUR`.",
        "",
        "## Tabla Ejecutiva De Pesos",
        "",
        "```text",
        weights.round(4).to_string(index=False),
        "```",
        "",
        "## Escenarios",
        "",
        "```text",
        scenarios.round(4).to_string(index=False),
        "```",
        "",
        "## Contrastes Semanales",
        "",
        "```text",
        stat_tests.round(4).to_string(index=False),
        "```",
        "",
        "## Guardrails",
        "",
        "```text",
        constraints.round(4).to_string(index=False),
        "```",
        "",
        "## Notas Metodologicas",
        "",
        "- Meridian recomienda fijar restricciones explicitas de gasto y revisar la plausibilidad del baseline antes de usar optimizacion para planning.",
        "- Robyn recomienda convertir curvas y elegibilidad en una capa de budget allocator, pero validando siempre los outputs antes de implementarlos.",
        "- En esta repo el `published_weight_pct` se interpreta solo dentro del subconjunto defendible, mientras que `recommended_share_pct` es el share total de presupuesto sugerido manteniendo el pool bloqueado.",
        "",
        "## Fuentes Consultadas",
        "",
        "- Google Meridian Optimization overview: https://developers.google.com/meridian/docs/user-guide/optimization-overview",
        "- Google Meridian Budget optimization scenarios: https://developers.google.com/meridian/docs/user-guide/budget-optimization-scenarios",
        "- Google Meridian Health checks: https://developers.google.com/meridian/docs/post-modeling/health-checks",
        "- Meta Robyn Features: https://facebookexperimental.github.io/Robyn/docs/features/",
        "- Meta Robyn Analyst's Guide to MMM: https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_constrained_decision_layer(
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    target_year: int = TARGET_YEAR,
    planning_budget_eur: float | None = None,
) -> dict[str, pd.DataFrame | dict[str, float | str]]:
    package = _load_decision_package(
        train_start_year=train_start_year,
        target_year=target_year,
        planning_budget_eur=planning_budget_eur,
    )
    constraints = _build_channel_constraints(package)
    optimized_budgets, optimization_meta = optimize_constrained_budget(package, constraints)

    scenario_rows: list[dict[str, float | str]] = []
    weekly_frames: list[pd.DataFrame] = []
    predictions_by_scenario: dict[str, pd.DataFrame] = {}
    budgets_by_scenario = {
        "do_nothing_observed": package.observed_budgets,
        "do_nothing_historical": package.base_budgets,
        "do_nothing_zero_media": {media_col: 0.0 for media_col in package.media_cols},
        "do_something_optimized": optimized_budgets,
    }

    for scenario_name, budgets in budgets_by_scenario.items():
        weekly, totals = predict_constrained_scenario(package, budgets, scenario_name=scenario_name)
        predictions_by_scenario[scenario_name] = weekly.copy()
        weekly_frames.append(weekly)
        scenario_rows.append(
            _scenario_diagnostics_row(
                scenario_name=scenario_name,
                budgets=budgets,
                observed_budgets=package.observed_budgets,
                historical_budgets=package.base_budgets,
                media_cols=package.media_cols,
                totals=totals,
            )
        )
    scenarios = pd.DataFrame(scenario_rows).reset_index(drop=True)
    zero_profit = float(scenarios.loc[scenarios["scenario"] == "do_nothing_zero_media", "predicted_gross_profit_2024"].iloc[0])
    observed_profit = float(scenarios.loc[scenarios["scenario"] == "do_nothing_observed", "predicted_gross_profit_2024"].iloc[0])
    observed_sales = float(scenarios.loc[scenarios["scenario"] == "do_nothing_observed", "predicted_sales_2024"].iloc[0])
    historical_profit = float(scenarios.loc[scenarios["scenario"] == "do_nothing_historical", "predicted_gross_profit_2024"].iloc[0])
    historical_sales = float(scenarios.loc[scenarios["scenario"] == "do_nothing_historical", "predicted_sales_2024"].iloc[0])

    scenarios["predicted_sales_delta_vs_observed"] = scenarios["predicted_sales_2024"] - observed_sales
    scenarios["predicted_sales_delta_vs_historical"] = scenarios["predicted_sales_2024"] - historical_sales
    scenarios["profit_vs_zero_media_eur"] = scenarios["predicted_gross_profit_2024"] - zero_profit
    scenarios["roi_vs_zero_media"] = scenarios.apply(
        lambda row: _safe_divide(float(row["profit_vs_zero_media_eur"]), float(row["total_budget_eur"])),
        axis=1,
    )
    scenarios["profit_vs_observed_eur"] = scenarios["predicted_gross_profit_2024"] - observed_profit
    scenarios["incremental_budget_vs_observed_eur"] = scenarios["total_budget_eur"] - package.observed_budget_eur
    scenarios["incremental_budget_roi_vs_observed"] = scenarios.apply(
        lambda row: _safe_divide(float(row["profit_vs_observed_eur"]), float(row["incremental_budget_vs_observed_eur"])),
        axis=1,
    )
    scenarios["profit_vs_historical_eur"] = scenarios["predicted_gross_profit_2024"] - historical_profit
    scenarios["reallocation_roi_vs_historical"] = scenarios.apply(
        lambda row: _safe_divide(
            float(row["profit_vs_historical_eur"]),
            float(row["total_budget_reallocated_vs_historical_eur"]),
        ),
        axis=1,
    )
    scenarios = scenarios.sort_values(["predicted_gross_profit_2024", "scenario"], ascending=[False, True]).reset_index(drop=True)

    stat_rows = []
    for comparison_name, candidate_name, baseline_name in [
        ("observed_vs_zero_media", "do_nothing_observed", "do_nothing_zero_media"),
        ("historical_vs_zero_media", "do_nothing_historical", "do_nothing_zero_media"),
        ("historical_vs_observed", "do_nothing_historical", "do_nothing_observed"),
        ("optimized_vs_historical", "do_something_optimized", "do_nothing_historical"),
        ("optimized_vs_observed", "do_something_optimized", "do_nothing_observed"),
        ("optimized_vs_zero_media", "do_something_optimized", "do_nothing_zero_media"),
    ]:
        candidate = predictions_by_scenario[candidate_name]["predicted_gross_profit"]
        baseline = predictions_by_scenario[baseline_name]["predicted_gross_profit"]
        stat_rows.append(
            {
                "comparison": comparison_name,
                "candidate_scenario": candidate_name,
                "baseline_scenario": baseline_name,
                **_paired_stat_test(candidate, baseline),
            }
        )
    stat_tests = pd.DataFrame(stat_rows).sort_values("comparison").reset_index(drop=True)

    weights = _build_executive_weights(package, optimized_budgets)
    weekly_scenarios = pd.concat(weekly_frames, ignore_index=True)

    constraints.to_csv(CONSTRAINTS_TABLE, index=False)
    weights.to_csv(WEIGHTS_TABLE, index=False)
    scenarios.to_csv(SCENARIOS_TABLE, index=False)
    stat_tests.to_csv(STAT_TESTS_TABLE, index=False)
    weekly_scenarios.to_csv(WEEKLY_SCENARIOS_FILE, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "model_name": "ConstrainedMMMDecisionLayer",
                "target_year": target_year,
                "train_start_year": train_start_year,
                "optimization_meta": optimization_meta,
                "eligible_channels": weights.loc[weights["model_status"] == "eligible", "channel"].tolist(),
                "observed_budget_eur": package.observed_budget_eur,
                "historical_budget_eur": float(sum(package.base_budgets.values())),
                "recommended_budget_eur": float(sum(optimized_budgets.values())),
                "zero_media_profit_2024": zero_profit,
                "observed_profit_2024": observed_profit,
                "historical_profit_2024": historical_profit,
                "recommended_profit_2024": float(
                    scenarios.loc[scenarios["scenario"] == "do_something_optimized", "predicted_gross_profit_2024"].iloc[0]
                ),
                "historical_roi_vs_zero_media": float(
                    scenarios.loc[scenarios["scenario"] == "do_nothing_historical", "roi_vs_zero_media"].iloc[0]
                ),
                "recommended_profit_uplift_eur": float(
                    scenarios.loc[scenarios["scenario"] == "do_something_optimized", "profit_vs_historical_eur"].iloc[0]
                ),
                "recommended_profit_vs_zero_media_eur": float(
                    scenarios.loc[scenarios["scenario"] == "do_something_optimized", "profit_vs_zero_media_eur"].iloc[0]
                ),
                "optimized_vs_historical_t_stat": float(
                    stat_tests.loc[stat_tests["comparison"] == "optimized_vs_historical", "t_statistic"].iloc[0]
                ),
                "optimized_vs_historical_p_value": float(
                    stat_tests.loc[stat_tests["comparison"] == "optimized_vs_historical", "p_value_one_sided"].iloc[0]
                ),
                "optimized_vs_historical_significant": bool(
                    stat_tests.loc[stat_tests["comparison"] == "optimized_vs_historical", "significant_one_sided_5pct"].iloc[0]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        package=package,
        optimization_meta=optimization_meta,
        constraints=constraints,
        weights=weights,
        scenarios=scenarios,
        stat_tests=stat_tests,
    )
    return {
        "weights": weights,
        "scenarios": scenarios,
        "stat_tests": stat_tests,
        "constraints": constraints,
        "summary": json.loads(SUMMARY_JSON.read_text(encoding="utf-8")),
    }
