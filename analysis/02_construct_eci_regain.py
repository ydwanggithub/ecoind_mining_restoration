"""Construct annual ECI, REGAIN, and the long term recovery response."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "annual_spectral_sample.csv"
DEFAULT_CONTRACT = ROOT / "data" / "eci_scaling_contract.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "eci_regain"


def scale_component(
    values: pd.Series, low: float, high: float, direction: int
) -> pd.Series:
    if high <= low:
        raise ValueError("The scaling upper bound must exceed the lower bound.")
    scaled = ((values.astype(float) - low) / (high - low)).clip(0.0, 1.0)
    if direction == -1:
        scaled = 1.0 - scaled
    elif direction != 1:
        raise ValueError(f"Unsupported component direction: {direction}")
    return scaled


def construct_eci(frame: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    score_columns: list[str] = []
    for row in contract.itertuples(index=False):
        result[row.score] = scale_component(
            result[row.source],
            float(row.q_low),
            float(row.q_high),
            int(row.direction),
        )
        score_columns.append(row.score)
    result["eci_spectral"] = result[score_columns].mean(axis=1)
    return result


def construct_regain(frame: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby("sample_id")["eci_spectral"]
    result["eci_cell_min"] = grouped.transform("min")
    result["eci_cell_max"] = grouped.transform("max")
    result["eci_cell_range"] = result["eci_cell_max"] - result["eci_cell_min"]
    valid = result["eci_cell_range"] > tolerance
    result["regain_annual"] = np.nan
    result.loc[valid, "regain_annual"] = (
        result.loc[valid, "eci_spectral"] - result.loc[valid, "eci_cell_min"]
    ) / result.loc[valid, "eci_cell_range"]
    return result


def recovery_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for sample_id, group in frame.dropna(subset=["regain_annual"]).groupby(
        "sample_id"
    ):
        ordered = group.sort_values("year")
        slope = float(
            theilslopes(
                ordered["regain_annual"].to_numpy(dtype=float),
                ordered["year"].to_numpy(dtype=float),
            ).slope
        )
        latest_year = ordered["year"].max()
        records.append(
            {
                "sample_id": sample_id,
                "regain_sen_slope_2000_2025": slope,
                "regain_2025": float(
                    ordered.loc[
                        ordered["year"].eq(latest_year), "regain_annual"
                    ].iloc[0]
                ),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annual = pd.read_csv(args.input)
    contract = pd.read_csv(args.contract)
    required = {"sample_id", "year", *contract["source"].tolist()}
    missing = sorted(required.difference(annual.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    derived = construct_regain(construct_eci(annual, contract))
    slopes = recovery_slopes(derived)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.output_dir / "annual_eci_regain.csv", index=False)
    slopes.to_csv(args.output_dir / "cell_recovery_response.csv", index=False)

    if "expected_eci_spectral" in annual:
        error = np.nanmax(
            np.abs(derived["eci_spectral"] - annual["expected_eci_spectral"])
        )
        print(f"Maximum ECI reconstruction error: {error:.3e}")
    if "expected_regain_annual" in annual:
        error = np.nanmax(
            np.abs(derived["regain_annual"] - annual["expected_regain_annual"])
        )
        print(f"Maximum REGAIN reconstruction error: {error:.3e}")
    print(f"Annual records: {len(derived):,}")
    print(f"Valid recovery responses: {len(slopes):,}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
