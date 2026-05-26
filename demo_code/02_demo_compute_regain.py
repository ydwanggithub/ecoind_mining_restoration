"""Compute REGAIN from synthetic initial and current ecological deficits.

REGAIN = (D_init - D_curr) / (D_init + D_curr)

where D_init = max(0, reference_ceiling - observed_initial)
      D_curr = max(0, reference_ceiling - observed_current)

REGAIN takes values from -1 to +1 by definition, with no imposed clipping.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_regain_demo.csv"


def compute_regain(d_init: np.ndarray, d_curr: np.ndarray, gap_floor: float = 0.02) -> np.ndarray:
    valid = d_init > gap_floor
    regain = np.full_like(d_init, np.nan, dtype=float)
    s = d_init + d_curr
    safe = valid & (s > 0)
    regain[safe] = (d_init[safe] - d_curr[safe]) / s[safe]
    return regain


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    d_init = np.maximum(0.0, df["reference_ceiling"] - df["observed_initial"]).to_numpy()
    d_curr = np.maximum(0.0, df["reference_ceiling"] - df["observed_current"]).to_numpy()
    df["regain"] = compute_regain(d_init, d_curr)
    n_valid = df["regain"].notna().sum()
    print(f"REGAIN computed for {n_valid} of {len(df)} cells")
    print(f"Median REGAIN: {df['regain'].median():.3f}")
    print(f"Range: [{df['regain'].min():.3f}, {df['regain'].max():.3f}]")
    print()
    print("Sample with REGAIN appended:")
    print(df[["cell_id", "observed_initial", "observed_current", "reference_ceiling", "regain"]].head())


if __name__ == "__main__":
    main()
