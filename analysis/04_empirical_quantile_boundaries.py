"""Estimate empirical upper and lower recovery boundaries without plotting.

The procedure follows the analysis reported in the manuscript. The observed
range of each predictor is divided into equal-width bins, and every bin with
at least 15 cells contributes its median predictor value and the 5th and 95th
percentiles of the REGAIN slope. A quadratic polynomial is fitted to each set
of bin percentiles, and a cubic replaces it only when the cubic has the higher
adjusted R2. Interior turning points are stationary points of the fitted curve
inside the central 98% of the supported range; the upper boundary requires a
maximum and the lower boundary a minimum.

Uncertainty comes from resampling whole 10 km blocks with replacement. Block
membership is read from the randomly renumbered ``block_10km_label`` column of
``data/cv_fold_assignments.csv.gz``. A turning point is classed as stable when
at least 95% of the resamples give valid fits, at least 80% give a turning
point of the same type, the 95% percentile interval lies inside the supported
range and spans no more than half of it, the interval contains the point
estimate, and the point fit has an R2 of at least 0.30.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "model_matrix_sample.csv.gz"
DEFAULT_FOLDS = ROOT / "data" / "cv_fold_assignments.csv.gz"
DEFAULT_OUTPUT = ROOT / "outputs" / "quantile_boundaries"
TARGET = "regain_sen_slope_2000_2025"
BLOCK_COLUMN = "block_10km_label"
FEATURES = (
    "forest_frac_slope_2000_2025",
    "built_frac_slope_2000_2025",
    "mndwi_mean_2000_2025",
    "ssrd_wm2_slope_2000_2025",
    "slope_mean",
    "ntl_viirs_slope_2014_2025",
)
BOUNDARIES = ("upper", "lower")
SEED = 20260923
CURVE_POINTS = 301


@dataclass
class PolynomialFit:
    degree: int
    coefficients: np.ndarray
    center: float
    scale: float
    r2: float
    adjusted_r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        return np.polyval(self.coefficients, z)


def binned_boundaries(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    minimum_bin_count: int,
) -> pd.DataFrame:
    """Median predictor value and 5th and 95th response percentiles per bin."""
    columns = ["x", "lower", "upper", "n"]
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if len(x) < minimum_bin_count * 5 or np.min(x) == np.max(x):
        return pd.DataFrame(columns=columns)
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    ids = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    rows: list[dict[str, float]] = []
    for bin_id in range(bins):
        mask = ids == bin_id
        count = int(mask.sum())
        if count < minimum_bin_count:
            continue
        rows.append(
            {
                "x": float(np.median(x[mask])),
                "lower": float(np.quantile(y[mask], 0.05)),
                "upper": float(np.quantile(y[mask], 0.95)),
                "n": count,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)


def polynomial_fit(x: np.ndarray, y: np.ndarray) -> PolynomialFit | None:
    """Quadratic fit, replaced by a cubic when its adjusted R2 is higher."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) <= 4 or np.min(x) == np.max(x):
        return None
    center = float((np.min(x) + np.max(x)) / 2)
    scale = float((np.max(x) - np.min(x)) / 2)
    z = (x - center) / scale
    candidates: list[PolynomialFit] = []
    for degree in (2, 3):
        if len(x) <= degree + 1:
            continue
        coefficients = np.polyfit(z, y, degree)
        fitted = np.polyval(coefficients, z)
        residual = float(np.sum((y - fitted) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 0.0 if total <= 0 else 1.0 - residual / total
        adjusted = 1.0 - (1.0 - r2) * (len(x) - 1) / (len(x) - degree - 1)
        candidates.append(
            PolynomialFit(degree, coefficients, center, scale, r2, adjusted)
        )
    quadratic = next(item for item in candidates if item.degree == 2)
    cubic = next((item for item in candidates if item.degree == 3), None)
    if (
        cubic is not None
        and cubic.adjusted_r2 > quadratic.adjusted_r2
        and cubic.adjusted_r2 > 0
    ):
        return cubic
    return quadratic


def turning_point(
    fit: PolynomialFit,
    x_min: float,
    x_max: float,
    upper: bool,
    typed: bool = True,
) -> tuple[float, int]:
    """Interior stationary point and the number of candidate points.

    With ``typed`` the upper boundary keeps only maxima and the lower boundary
    only minima. Without it, any interior stationary point is a candidate, as
    in the turning point inference of the original release.
    """
    roots = np.roots(np.polyder(fit.coefficients))
    z = roots[np.isclose(roots.imag, 0, atol=1e-8)].real
    x = fit.center + fit.scale * z
    tolerance = 0.01 * (x_max - x_min)
    keep = (x > x_min + tolerance) & (x < x_max - tolerance)
    if typed:
        curvature = np.polyval(np.polyder(fit.coefficients, 2), z)
        keep &= (curvature < 0) if upper else (curvature > 0)
    x = x[keep]
    if not len(x):
        return np.nan, 0
    values = fit.predict(x)
    index = int(np.argmax(values) if upper else np.argmin(values))
    return float(x[index]), len(x)


def point_estimates(
    feature: str,
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    minimum_bin_count: int,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    boundary = binned_boundaries(x, y, bins, minimum_bin_count)
    if len(boundary) <= 4:
        return None, boundary
    x_min = float(boundary["x"].min())
    x_max = float(boundary["x"].max())
    record: dict[str, object] = {
        "feature": feature,
        "supported_x_min": x_min,
        "supported_x_max": x_max,
        "n_boundary_bins": len(boundary),
    }
    for label in BOUNDARIES:
        upper = label == "upper"
        fit = polynomial_fit(boundary["x"].to_numpy(), boundary[label].to_numpy())
        record[f"{label}_degree"] = fit.degree
        record[f"{label}_r2"] = fit.r2
        record[f"{label}_adjusted_r2"] = fit.adjusted_r2
        record[f"{label}_turning_x"] = turning_point(
            fit, x_min, x_max, upper, typed=False
        )[0]
        record[f"{label}_typed_turning_x"] = turning_point(
            fit, x_min, x_max, upper
        )[0]
        boundary[f"{label}_fitted"] = fit.predict(boundary["x"].to_numpy())
    boundary.insert(0, "feature", feature)
    return record, boundary


def bootstrap_feature(
    task: tuple[str, np.ndarray, np.ndarray, list[np.ndarray], dict, int, int, int, int],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit both boundaries on whole-block resamples of the data."""
    feature, x, y, groups, record, replicates, seed, bins, minimum_bin_count = task
    rng = np.random.default_rng(seed)
    grid = np.linspace(
        record["supported_x_min"], record["supported_x_max"], CURVE_POINTS
    )
    turns = np.full((replicates, 2), np.nan)
    counts = np.zeros((replicates, 2), dtype=int)
    degrees = np.full((replicates, 2), -1, dtype=int)
    curves = np.full((replicates, 2, CURVE_POINTS), np.nan, dtype=np.float32)
    for replicate in range(replicates):
        draw = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[j] for j in draw])
        boundary = binned_boundaries(x[index], y[index], bins, minimum_bin_count)
        if len(boundary) <= 4:
            continue
        for j, label in enumerate(BOUNDARIES):
            fit = polynomial_fit(boundary["x"].to_numpy(), boundary[label].to_numpy())
            if fit is None:
                continue
            curves[replicate, j] = fit.predict(grid)
            degrees[replicate, j] = fit.degree
            turns[replicate, j], counts[replicate, j] = turning_point(
                fit, grid[0], grid[-1], label == "upper"
            )
    return feature, grid, turns, counts, degrees, curves


def summarize(
    record: dict[str, object],
    grid: np.ndarray,
    turns: np.ndarray,
    counts: np.ndarray,
    degrees: np.ndarray,
    curves: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    replicates = len(turns)
    x_min = record["supported_x_min"]
    x_max = record["supported_x_max"]
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for j, label in enumerate(BOUNDARIES):
        values = turns[:, j]
        valid = np.isfinite(values)
        successful = int((degrees[:, j] > 0).sum())
        if valid.any():
            low, high = np.quantile(values[valid], [0.025, 0.975])
        else:
            low = high = np.nan
        estimate = record[f"{label}_typed_turning_x"]
        stable = bool(
            successful >= 0.95 * replicates
            and valid.mean() >= 0.80
            and low >= x_min
            and high <= x_max
            and record[f"{label}_r2"] >= 0.30
            and high - low <= 0.5 * (x_max - x_min)
            and low <= estimate <= high
        )
        rows.append(
            {
                "feature": record["feature"],
                "boundary": label,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "successful_fits": successful,
                "turning_support": float(valid.mean()),
                "no_typed_turn": int((~valid).sum()),
                "multiple_typed_turns": int((counts[:, j] > 1).sum()),
                "stable": stable,
            }
        )
        interval = np.nanquantile(curves[:, j], [0.025, 0.975], axis=0)
        for k, value in enumerate(grid):
            curve_rows.append(
                {
                    "feature": record["feature"],
                    "boundary": label,
                    "x": value,
                    "ci_low": interval[0, k],
                    "ci_high": interval[1, k],
                }
            )
    return rows, curve_rows


def load_inputs(data: Path, folds: Path) -> pd.DataFrame:
    frame = pd.read_csv(data)
    missing = sorted({"sample_id", TARGET, *FEATURES}.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = pd.read_csv(folds, usecols=["sample_id", BLOCK_COLUMN])
    frame = frame.merge(labels, on="sample_id", how="left", validate="one_to_one")
    if frame[BLOCK_COLUMN].isna().any():
        raise ValueError("Some rows have no 10 km block label.")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--minimum-bin-count", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_inputs(args.data, args.folds)
    labels = frame[BLOCK_COLUMN].to_numpy()
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    y = frame[TARGET].to_numpy(dtype=float)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, object]] = {}
    bins_and_fits: list[pd.DataFrame] = []
    tasks = []
    for feature in FEATURES:
        x = frame[feature].to_numpy(dtype=float)
        record, boundary = point_estimates(
            feature, x, y, args.bins, args.minimum_bin_count
        )
        if record is None:
            print(f"Skipped {feature}: too few bins with {args.minimum_bin_count} cells.")
            continue
        records[feature] = record
        bins_and_fits.append(boundary)
        tasks.append(
            (feature, x, y, groups, record, args.bootstrap, SEED,
             args.bins, args.minimum_bin_count)
        )
    if not records:
        raise RuntimeError("No boundary fits passed the minimum data requirement.")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(bootstrap_feature, tasks))
    else:
        results = [bootstrap_feature(task) for task in tasks]

    summary: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for feature, grid, turns, counts, degrees, curves in results:
        rows, curve = summarize(records[feature], grid, turns, counts, degrees, curves)
        summary.extend(rows)
        curve_rows.extend(curve)

    pd.concat(bins_and_fits, ignore_index=True).to_csv(
        args.output_dir / "boundary_bins_and_fits.csv", index=False
    )
    pd.DataFrame(records.values()).to_csv(
        args.output_dir / "boundary_point_estimates.csv", index=False
    )
    pd.DataFrame(summary).to_csv(
        args.output_dir / "boundary_bootstrap_summary.csv", index=False
    )
    pd.DataFrame(curve_rows).to_csv(
        args.output_dir / "boundary_curve_intervals.csv", index=False
    )
    print(
        f"{len(groups)} blocks, {args.bootstrap} resamples. "
        f"Wrote outputs to {args.output_dir}"
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
