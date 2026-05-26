"""Compute SHAP values and draw a global summary bar plot.

The plot is saved to demo_code/shap_summary_demo.png. The example illustrates
how a per-cell attribution layer is built once a Random Forest model is fitted;
real applications would use the full predictor set from the corresponding
manuscript Methods section.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_regain_demo.csv"
OUT_PNG = Path(__file__).resolve().parent / "shap_summary_demo.png"
FEATURES = ["vegetation_slope", "moisture_recent", "elevation", "initial_gap", "human_pressure"]
TARGET = "target_regain"
SEED = 42


def main() -> None:
    df = pd.read_csv(CSV_PATH).dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=4, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    sorted_features = [FEATURES[i] for i in order]
    sorted_values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(sorted_features, sorted_values, color="#4A6FA5")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP importance (synthetic demo)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PNG}")
    print("Top-3 features:", sorted_features[-1], sorted_features[-2], sorted_features[-3])


if __name__ == "__main__":
    main()
