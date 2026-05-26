"""Train a Random Forest regressor on synthetic predictors of REGAIN.

This demo shows the bare structure of a recovery-attribution pipeline:
load features, split, fit, score, and inspect feature importances.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_regain_demo.csv"
FEATURES = ["vegetation_slope", "moisture_recent", "elevation", "initial_gap", "human_pressure"]
TARGET = "target_regain"
SEED = 42


def main() -> None:
    df = pd.read_csv(CSV_PATH).dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    print(f"Trained on {len(X_tr)} cells, evaluated on {len(X_te)} cells")
    print(f"R^2: {r2_score(y_te, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_te, y_pred):.3f}")
    print()
    print("Feature importances (Gini):")
    order = np.argsort(rf.feature_importances_)[::-1]
    for idx in order:
        print(f"  {FEATURES[idx]:<20s} {rf.feature_importances_[idx]:.3f}")


if __name__ == "__main__":
    main()
