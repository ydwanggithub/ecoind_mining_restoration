"""Fit the primary XGBoost model with out of fold tree SHAP attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "model_matrix_sample.csv.gz"
DEFAULT_LEDGER = ROOT / "data" / "predictor_ledger.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "xgboost_shap"
TARGET = "regain_sen_slope_2000_2025"
SEED = 42

XGB_PARAMETERS = {
    "n_estimators": 850,
    "learning_rate": 0.03,
    "max_depth": 0,
    "max_leaves": 128,
    "grow_policy": "lossguide",
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.03,
    "reg_lambda": 0.7,
    "gamma": 0.0,
}


def retained_predictors(ledger: pd.DataFrame) -> list[str]:
    retained = ledger["retained"].astype(str).str.lower().eq("true")
    values = ledger.loc[retained, "name"].tolist()
    if len(values) != 57:
        raise ValueError(f"Expected 57 retained predictors, found {len(values)}")
    return values


def metric_record(
    fold: int, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | int]:
    return {
        "fold": fold,
        "n_test": len(y_true),
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--estimators", type=int, default=XGB_PARAMETERS["n_estimators"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    ledger = pd.read_csv(args.ledger)
    features = retained_predictors(ledger)
    required = {"sample_id", TARGET, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x_raw = frame[features]
    y = frame[TARGET].to_numpy(dtype=float)
    splits = KFold(args.folds, shuffle=True, random_state=SEED)
    predictions = np.full(len(frame), np.nan, dtype=float)
    shap_values = np.full((len(frame), len(features)), np.nan, dtype=np.float32)
    fold_ids = np.full(len(frame), -1, dtype=int)
    metrics: list[dict[str, float | int]] = []

    parameters = dict(XGB_PARAMETERS)
    parameters["n_estimators"] = args.estimators
    for fold, (train_index, test_index) in enumerate(
        splits.split(x_raw), start=1
    ):
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x_raw.iloc[train_index])
        x_test = imputer.transform(x_raw.iloc[test_index])
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
            **parameters,
        )
        model.fit(x_train, y[train_index])
        fold_prediction = model.predict(x_test)
        predictions[test_index] = fold_prediction
        fold_ids[test_index] = fold
        contributions = model.get_booster().predict(
            xgb.DMatrix(x_test, feature_names=features),
            pred_contribs=True,
        )
        shap_values[test_index] = contributions[:, :-1]
        metrics.append(metric_record(fold, y[test_index], fold_prediction))

    if np.isnan(predictions).any() or np.isnan(shap_values).any():
        raise RuntimeError("Out of fold predictions or SHAP values are incomplete.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(
        args.output_dir / "fold_metrics.csv", index=False
    )
    summary = pd.DataFrame(
        [
            {
                "n": len(frame),
                "n_predictors": len(features),
                "folds": args.folds,
                "r2_pooled_oof": r2_score(y, predictions),
                "rmse_pooled_oof": mean_squared_error(y, predictions) ** 0.5,
                "mae_pooled_oof": mean_absolute_error(y, predictions),
            }
        ]
    )
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "observed": y,
            "predicted_oof": predictions,
            "fold": fold_ids,
        }
    ).to_csv(args.output_dir / "oof_predictions.csv", index=False)

    mean_absolute = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_absolute})
        .merge(
            ledger[["name", "display_name", "abbreviation", "family"]],
            left_on="feature",
            right_on="name",
            how="left",
        )
        .drop(columns="name")
        .sort_values("mean_abs_shap", ascending=False)
    )
    importance.to_csv(
        args.output_dir / "shap_feature_importance.csv", index=False
    )
    family = (
        importance.groupby("family", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    family["share_percent"] = (
        100 * family["mean_abs_shap"] / family["mean_abs_shap"].sum()
    )
    family.to_csv(
        args.output_dir / "shap_family_importance.csv", index=False
    )
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
