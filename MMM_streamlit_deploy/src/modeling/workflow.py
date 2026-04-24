from __future__ import annotations

import json
import pickle

import pandas as pd

from src.common.config import CONFIG
from src.modeling.reporting import (
    plot_channel_contributions,
    plot_test_fit,
    save_executive_summary,
    write_code_review_markdown,
    write_model_review_markdown,
    write_validation_markdown,
)
from src.modeling.selection import (
    choose_winner,
    evaluate_spec,
    fit_final_model,
    naive_predictions,
    predict_package,
)
from src.modeling.specs import CONTROL_COLUMNS
from src.modeling.trainer import (
    city_dummy_columns,
    ensure_prerequisites,
    load_dataset,
    media_columns,
    original_scale_coefficients,
)
from src.validation.backtesting import expanding_year_splits, quarter_label


def run_training_pipeline() -> dict:
    ensure_prerequisites()
    df = load_dataset()
    df["naive_pred"] = naive_predictions(df)
    media_cols = media_columns(df)
    city_cols = city_dummy_columns(df)

    backtest_rows = []
    prediction_frames = []

    for train_mask, valid_mask, fold_name in expanding_year_splits(df, min_train_year=2021):
        train_df = df.loc[train_mask].copy()
        valid_df = df.loc[valid_mask].copy()
        for spec_name, use_transforms, include_media in [
            ("NaiveRolling4", False, False),
            ("ElasticNetControls", False, False),
            ("ElasticNetRawMedia", False, True),
            ("ElasticNetTransformedMedia", True, True),
        ]:
            metrics, scored, _, diagnostics = evaluate_spec(
                spec_name,
                train_df,
                valid_df,
                df,
                CONTROL_COLUMNS,
                city_cols,
                media_cols,
                use_transforms=use_transforms,
                include_media=include_media,
            )
            metrics["fold"] = fold_name
            metrics["spec"] = spec_name
            backtest_rows.append(metrics)
            scored["fold"] = fold_name
            scored["spec"] = spec_name
            prediction_frames.append(scored)
            if diagnostics is not None:
                diagnostics["fold"] = fold_name

    backtest_results = pd.DataFrame(backtest_rows)
    winner = choose_winner(backtest_results)

    train_eval = df[df["year"] < 2024].copy()
    test_eval = df[df["year"] == 2024].copy()
    spec_map = {
        "NaiveRolling4": (False, False),
        "ElasticNetControls": (False, False),
        "ElasticNetRawMedia": (False, True),
        "ElasticNetTransformedMedia": (True, True),
    }
    use_transforms, include_media = spec_map[winner]
    test_metrics, test_scored, _, diagnostics = evaluate_spec(
        winner,
        train_eval,
        test_eval,
        df,
        CONTROL_COLUMNS,
        city_cols,
        media_cols,
        use_transforms=use_transforms,
        include_media=include_media,
    )
    test_results = pd.DataFrame([{**test_metrics, "spec": winner, "fold": "test_2024"}])

    deploy_package, deploy_diagnostics = fit_final_model(
        df,
        df,
        winner if winner != "NaiveRolling4" else "ElasticNetTransformedMedia",
        CONTROL_COLUMNS,
        city_cols,
        media_cols,
    )

    contributions_matrix, deployment_pred, intercept = predict_package(df, deploy_package)
    deployment_scored = df[["semana_inicio", "ciudad", "ventas_netas"]].copy()
    deployment_scored["pred"] = deployment_pred
    deployment_scored["phase"] = "deployment_fit"

    feature_coef = original_scale_coefficients(
        deploy_package.scaler,
        deploy_package.model,
        deploy_package.feature_columns,
    )
    contribution_table = (
        contributions_matrix[[column for column in deploy_package.media_feature_columns]]
        .assign(semana_inicio=df["semana_inicio"].to_numpy(), ciudad=df["ciudad"].to_numpy())
        .melt(id_vars=["semana_inicio", "ciudad"], var_name="feature", value_name="contribution")
    )
    contribution_table["channel"] = contribution_table["feature"].str.split("__").str[0]
    contribution_table["spec"] = deploy_package.spec_name

    predictions = pd.concat(
        prediction_frames
        + [
            test_scored.assign(spec=winner, fold="test_2024"),
            deployment_scored.assign(spec=deploy_package.spec_name, fold="deployment_fit"),
        ],
        ignore_index=True,
    )
    predictions["quarter"] = quarter_label(predictions["semana_inicio"])

    model_results = {
        "winner": winner,
        "deployment_spec": deploy_package.spec_name,
        "test_metrics": test_metrics,
        "selected_transforms": deploy_package.media_params,
        "model_alpha": float(deploy_package.model.alpha_),
        "model_l1_ratio": float(deploy_package.model.l1_ratio_),
        "intercept": intercept,
        "negative_media_coefficients": int(
            (feature_coef[[col for col in deploy_package.feature_columns if col.startswith("media_")]] < -1e-8).sum()
        ),
    }

    backtest_results.to_csv(CONFIG.backtest_results_file, index=False)
    predictions.to_csv(CONFIG.weekly_predictions_file, index=False)
    contribution_table.to_csv(CONFIG.contributions_file, index=False)
    CONFIG.model_results_file.write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    CONFIG.selected_transforms_file.write_text(json.dumps(deploy_package.media_params, indent=2), encoding="utf-8")
    with CONFIG.final_model_file.open("wb") as handle:
        pickle.dump(deploy_package, handle)

    plot_test_fit(test_scored)
    plot_channel_contributions(contribution_table)
    write_validation_markdown(backtest_results.round(4), test_results.round(4), winner)
    write_model_review_markdown(winner, test_metrics, model_results["negative_media_coefficients"])
    write_code_review_markdown()
    save_executive_summary(test_metrics, winner, contribution_table)

    if diagnostics is not None:
        diagnostics.to_csv(CONFIG.reports_tables_dir / "transform_diagnostics_test.csv", index=False)
    if deploy_diagnostics is not None:
        deploy_diagnostics.to_csv(CONFIG.reports_tables_dir / "transform_diagnostics_deploy.csv", index=False)
    pd.DataFrame(
        {
            "feature": feature_coef.index,
            "coefficient": feature_coef.values,
        }
    ).to_csv(CONFIG.reports_tables_dir / "deployment_coefficients.csv", index=False)

    return {
        "backtest_results": backtest_results,
        "winner": winner,
        "test_metrics": test_metrics,
    }
