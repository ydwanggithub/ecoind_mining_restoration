"""Estimate empirical lower and upper recovery boundaries without plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "model_matrix_sample.csv.gz"
DEFAULT_OUTPUT = ROOT / "outputs" / "quantile_boundaries"
TARGET = "regain_sen_slope_2000_2025"
FEATURES = (
    "forest_frac_slope_2000_2025",
    "built_frac_slope_2000_2025",
    "mndwi_mean_2000_2025",
    "ssrd_wm2_slope_2000_2025",
    "slope_mean",
    "ntl_viirs_slope_2014_2025",
)
SEED = 42


def adjusted_r2(y: np.ndarray, fitted: np.ndarray, parameters: int) -> float:
    n = len(y)
    if n <= parameters + 1:
        return float("-inf")
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    if total <= 0:
        return float("-inf")
    r2 = 1.0 - residual / total
    return 1.0 - (1.0 - r2) * (n - 1) / (n - parameters - 1)


def boundary_points(
    frame: pd.DataFrame,
    feature: str,
    quantile: float,
    bins: int,
    minimum_bin_count: int,
) -> pd.DataFrame:
    values = frame[[feature, TARGET]].dropna().copy()
    low, high = values[feature].quantile([0.005, 0.995])
    values = values[values[feature].between(low, high)]
    edges = np.linspace(values[feature].min(), values[feature].max(), bins + 1)
    values["bin"] = pd.cut(
        values[feature], bins=edges, include_lowest=True, duplicates="drop"
    )
    grouped = values.groupby("bin", observed=True)
    points = grouped.agg(
        predictor=(feature, "median"),
        response=(TARGET, lambda item: item.quantile(quantile)),
        n=(TARGET, "size"),
    ).reset_index(drop=True)
    points = points[points["n"] >= minimum_bin_count].copy()
    points["feature"] = feature
    points["quantile"] = quantile
    return points


def fit_polynomial(points: pd.DataFrame) -> dict[str, object]:
    x = points["predictor"].to_numpy(dtype=float)
    y = points["response"].to_numpy(dtype=float)
    candidates: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for degree in (2, 3):
        coefficient = np.polyfit(x, y, degree)
        fitted = np.polyval(coefficient, x)
        candidates.append(
            (degree, coefficient, fitted, adjusted_r2(y, fitted, degree + 1))
        )
    degree, coefficient, fitted, score = max(
        candidates, key=lambda item: item[3]
    )
    derivative = np.polyder(coefficient)
    roots = np.roots(derivative)
    interior = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-9 and x.min() < root.real < x.max()
    )
    return {
        "degree": degree,
        "coefficient": coefficient,
        "adjusted_r2": score,
        "turning_points": interior,
        "fitted": fitted,
    }


def bootstrap_turns(
    frame: pd.DataFrame,
    feature: str,
    quantile: float,
    bins: int,
    minimum_bin_count: int,
    replicates: int,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    turns: list[float] = []
    for _ in range(replicates):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        points = boundary_points(
            sample, feature, quantile, bins, minimum_bin_count
        )
        if len(points) < 8:
            continue
        fit = fit_polynomial(points)
        if fit["turning_points"]:
            turns.append(fit["turning_points"][0])
    return np.asarray(turns, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--minimum-bin-count", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    missing = sorted({TARGET, *FEATURES}.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_points: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for feature in FEATURES:
        for quantile in (0.05, 0.95):
            points = boundary_points(
                frame, feature, quantile, args.bins, args.minimum_bin_count
            )
            if len(points) < 8:
                continue
            fit = fit_polynomial(points)
            points["fitted"] = fit["fitted"]
            all_points.append(points)
            turns = bootstrap_turns(
                frame,
                feature,
                quantile,
                args.bins,
                args.minimum_bin_count,
                args.bootstrap,
            )
            summaries.append(
                {
                    "feature": feature,
                    "quantile": quantile,
                    "degree": fit["degree"],
                    "adjusted_r2": fit["adjusted_r2"],
                    "point_turning_points": ";".join(
                        f"{value:.8g}" for value in fit["turning_points"]
                    ),
                    "bootstrap_turn_count": len(turns),
                    "bootstrap_turn_median": (
                        np.median(turns) if len(turns) else np.nan
                    ),
                    "bootstrap_turn_ci_low": (
                        np.quantile(turns, 0.025) if len(turns) else np.nan
                    ),
                    "bootstrap_turn_ci_high": (
                        np.quantile(turns, 0.975) if len(turns) else np.nan
                    ),
                }
            )
    if not all_points:
        raise RuntimeError("No boundary fits passed the minimum data requirement.")
    pd.concat(all_points, ignore_index=True).to_csv(
        args.output_dir / "boundary_points_and_fits.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "boundary_summary.csv", index=False
    )
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
