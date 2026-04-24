from __future__ import annotations

import numpy as np
import pandas as pd


def feature_block(feature: str, city_columns: list[str] | None = None) -> str:
    city_set = set(city_columns or [])
    if feature in city_set or feature.startswith("city_"):
        return "city"
    if feature.startswith("media_") or feature.startswith("budget_share_pct_"):
        return "media"
    return "baseline"


def _dominant_sign(
    positive_count: int,
    negative_count: int,
    non_zero_count: int,
    sign_consistency: float,
) -> str:
    if non_zero_count == 0:
        return "zero"
    if sign_consistency < 0.6:
        return "mixed"
    return "positive" if positive_count >= negative_count else "negative"


def _stability_status(selection_rate: float, sign_consistency: float, dominant_sign: str) -> str:
    if dominant_sign in {"zero", "mixed"}:
        return "unstable"
    if selection_rate >= (2.0 / 3.0) and sign_consistency >= 0.75:
        return "stable"
    if selection_rate >= 0.5 and sign_consistency >= (2.0 / 3.0):
        return "review"
    return "unstable"


def _deployment_sign(value: float, coefficient_threshold: float) -> str:
    if value > coefficient_threshold:
        return "positive"
    if value < -coefficient_threshold:
        return "negative"
    return "zero"


def summarize_beta_stability(
    coefficients_df: pd.DataFrame,
    deployment_coefficients: pd.Series | None = None,
    *,
    city_columns: list[str] | None = None,
    coefficient_threshold: float = 1e-8,
) -> pd.DataFrame:
    required_columns = {"feature", "coefficient"}
    missing = required_columns.difference(coefficients_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for beta stability summary: {sorted(missing)}")

    deployment_map = deployment_coefficients if deployment_coefficients is not None else pd.Series(dtype=float)
    rows: list[dict[str, float | str | int]] = []

    grouped = {
        feature: group["coefficient"].to_numpy(dtype=float)
        for feature, group in coefficients_df.groupby("feature", sort=True)
    }
    all_features = sorted(set(grouped) | set(deployment_map.index.tolist()))

    for feature in all_features:
        coefficients = grouped.get(feature, np.array([], dtype=float))
        positive_mask = coefficients > coefficient_threshold
        negative_mask = coefficients < -coefficient_threshold
        selected_mask = positive_mask | negative_mask

        positive_count = int(positive_mask.sum())
        negative_count = int(negative_mask.sum())
        non_zero_count = int(selected_mask.sum())
        selection_rate = float(selected_mask.mean()) if len(coefficients) else 0.0
        sign_consistency = max(positive_count, negative_count) / max(non_zero_count, 1)
        dominant_sign = _dominant_sign(positive_count, negative_count, non_zero_count, sign_consistency)

        if len(coefficients):
            mean_coef = float(np.mean(coefficients))
            median_coef = float(np.median(coefficients))
            std_coef = float(np.std(coefficients, ddof=0))
        else:
            mean_coef = float("nan")
            median_coef = float("nan")
            std_coef = float("nan")
        coef_cv = float(std_coef / abs(mean_coef)) if abs(mean_coef) > coefficient_threshold else float("nan")
        deployment_coef = float(deployment_map.get(feature, np.nan))
        deployment_sign = _deployment_sign(deployment_coef, coefficient_threshold)
        stability_status = _stability_status(selection_rate, sign_consistency, dominant_sign)
        if (
            dominant_sign not in {"zero", "mixed"}
            and deployment_sign not in {"zero", dominant_sign}
        ):
            stability_status = "unstable"

        rows.append(
            {
                "feature": feature,
                "feature_block": feature_block(feature, city_columns=city_columns),
                "fold_count": int(len(coefficients)),
                "non_zero_fold_count": non_zero_count,
                "selection_rate": selection_rate,
                "positive_fold_rate": positive_count / max(len(coefficients), 1),
                "negative_fold_rate": negative_count / max(len(coefficients), 1),
                "sign_consistency": float(sign_consistency),
                "dominant_sign": dominant_sign,
                "mean_coefficient": mean_coef,
                "median_coefficient": median_coef,
                "std_coefficient": std_coef,
                "coef_cv": coef_cv,
                "deployment_coefficient": deployment_coef,
                "deployment_sign": deployment_sign,
                "stability_status": stability_status,
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        ["feature_block", "stability_status", "selection_rate", "sign_consistency", "feature"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)


def usability_flag_row(
    *,
    feature_block: str,
    stability_status: str,
    selection_rate: float,
    sign_consistency: float,
    coefficient_sign: str,
    dominant_sign: str,
    bootstrap_positive_probability: float,
    bootstrap_negative_probability: float,
    collinearity_flag: bool,
) -> tuple[bool, str]:
    if feature_block != "media":
        return False, "not_media"
    if collinearity_flag:
        return False, "collinearity_filtered"
    if stability_status != "stable":
        return False, f"stability_{stability_status}"
    if selection_rate < 0.6:
        return False, "low_selection_rate"
    if sign_consistency < 0.75:
        return False, "low_sign_consistency"
    if coefficient_sign != dominant_sign:
        return False, "deployment_sign_mismatch"
    sign_probability = bootstrap_positive_probability if coefficient_sign == "positive" else bootstrap_negative_probability
    if sign_probability < 0.9:
        return False, "weak_bootstrap_sign_probability"
    return True, "usable"


def summarize_channel_stability(
    fold_coefficients_df: pd.DataFrame,
    deployment_channel_df: pd.DataFrame,
    *,
    total_fold_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool | int]] = []
    grouped = fold_coefficients_df.groupby("channel", sort=True) if not fold_coefficients_df.empty else []

    per_channel = {}
    for channel, group in grouped:
        active = group[np.abs(group["coefficient"].to_numpy(dtype=float)) > 1e-8].copy()
        positive_count = int((active["coefficient"] > 1e-8).sum())
        negative_count = int((active["coefficient"] < -1e-8).sum())
        non_zero_count = int(active.shape[0])
        selection_rate = non_zero_count / max(total_fold_count, 1)
        sign_consistency = max(positive_count, negative_count) / max(non_zero_count, 1)
        dominant_sign = _dominant_sign(positive_count, negative_count, non_zero_count, sign_consistency)
        if non_zero_count:
            dominant_feature_share = float(active["feature"].value_counts(normalize=True).iloc[0])
            dominant_feature = str(active["feature"].value_counts().index[0])
        else:
            dominant_feature_share = 0.0
            dominant_feature = ""
        per_channel[channel] = {
            "fold_count": total_fold_count,
            "non_zero_fold_count": non_zero_count,
            "selection_rate": selection_rate,
            "sign_consistency": sign_consistency,
            "dominant_sign": dominant_sign,
            "dominant_feature": dominant_feature,
            "dominant_feature_share": dominant_feature_share,
            "channel_stability_status": _stability_status(selection_rate, sign_consistency, dominant_sign),
        }

    all_channels = sorted(set(per_channel) | set(deployment_channel_df["channel"].dropna().astype(str).tolist()))
    for channel in all_channels:
        deploy = deployment_channel_df[deployment_channel_df["channel"] == channel].copy()
        if deploy.empty:
            continue
        deploy_row = deploy.iloc[0]
        stats = per_channel.get(
            channel,
            {
                "fold_count": total_fold_count,
                "non_zero_fold_count": 0,
                "selection_rate": 0.0,
                "sign_consistency": 0.0,
                "dominant_sign": "zero",
                "dominant_feature": "",
                "dominant_feature_share": 0.0,
                "channel_stability_status": "unstable",
            },
        )
        usable_beta, usable_reason = usability_flag_row(
            feature_block="media",
            stability_status=str(stats["channel_stability_status"]),
            selection_rate=float(stats["selection_rate"]),
            sign_consistency=float(stats["sign_consistency"]),
            coefficient_sign=str(deploy_row["coefficient_sign"]),
            dominant_sign=str(stats["dominant_sign"]),
            bootstrap_positive_probability=float(deploy_row["bootstrap_positive_probability"]) if pd.notna(deploy_row["bootstrap_positive_probability"]) else 0.0,
            bootstrap_negative_probability=float(deploy_row["bootstrap_negative_probability"]) if pd.notna(deploy_row["bootstrap_negative_probability"]) else 0.0,
            collinearity_flag=bool(deploy_row["collinearity_flag"]),
        )
        if usable_beta and float(stats["dominant_feature_share"]) < 0.5:
            usable_beta = False
            usable_reason = "transform_instability"
        rows.append(
            {
                "channel": channel,
                "deployment_feature": str(deploy_row["feature"]),
                "deployment_coefficient": float(deploy_row["coefficient"]),
                "deployment_standardized_coefficient": float(deploy_row["standardized_coefficient"]),
                "deployment_sign": str(deploy_row["coefficient_sign"]),
                "fold_count": int(stats["fold_count"]),
                "non_zero_fold_count": int(stats["non_zero_fold_count"]),
                "selection_rate": float(stats["selection_rate"]),
                "sign_consistency": float(stats["sign_consistency"]),
                "dominant_sign": str(stats["dominant_sign"]),
                "dominant_feature": str(stats["dominant_feature"]),
                "dominant_feature_share": float(stats["dominant_feature_share"]),
                "bootstrap_original_p02_5": float(deploy_row["bootstrap_original_p02_5"]) if pd.notna(deploy_row["bootstrap_original_p02_5"]) else np.nan,
                "bootstrap_original_p50": float(deploy_row["bootstrap_original_p50"]) if pd.notna(deploy_row["bootstrap_original_p50"]) else np.nan,
                "bootstrap_original_p97_5": float(deploy_row["bootstrap_original_p97_5"]) if pd.notna(deploy_row["bootstrap_original_p97_5"]) else np.nan,
                "bootstrap_positive_probability": float(deploy_row["bootstrap_positive_probability"]) if pd.notna(deploy_row["bootstrap_positive_probability"]) else 0.0,
                "bootstrap_negative_probability": float(deploy_row["bootstrap_negative_probability"]) if pd.notna(deploy_row["bootstrap_negative_probability"]) else 0.0,
                "collinearity_flag": bool(deploy_row["collinearity_flag"]),
                "stability_status": str(stats["channel_stability_status"]),
                "usable_beta": bool(usable_beta),
                "usable_reason": usable_reason,
            }
        )

    return pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)
