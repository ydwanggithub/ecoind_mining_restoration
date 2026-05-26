"""One-shot generator for demo_data/sample_regain_demo.csv (synthetic data).

This script is included so that anyone can regenerate the bundled CSV
locally with the same fixed random seed. Run once; the resulting CSV is
also committed to the repository.

The values are fully synthetic and do not correspond to any real location
or measurement. Field names match the columns expected by the five demo
scripts.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT_CSV = Path(__file__).resolve().parent.parent / "demo_data" / "sample_regain_demo.csv"
N = 500
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 100, N)
    y = rng.uniform(0, 100, N)
    observed_initial = rng.uniform(0.30, 0.70, N)
    observed_current = observed_initial + rng.normal(0.10, 0.08, N)
    observed_current = np.clip(observed_current, 0.0, 1.0)
    reference_ceiling = np.minimum(1.0, observed_current + rng.uniform(0.05, 0.30, N))

    vegetation_slope = rng.normal(0.0, 0.01, N)
    moisture_recent = rng.uniform(0.1, 0.6, N)
    elevation = rng.uniform(50, 1200, N)
    initial_gap = np.maximum(0.0, reference_ceiling - observed_initial)
    human_pressure = rng.uniform(0.0, 1.0, N)

    d_init = initial_gap.copy()
    d_curr = np.maximum(0.0, reference_ceiling - observed_current)
    target_regain = np.where(d_init > 0.02, (d_init - d_curr) / (d_init + d_curr + 1e-9), np.nan)

    df = pd.DataFrame({
        "cell_id": np.arange(N),
        "x": x.round(2),
        "y": y.round(2),
        "observed_initial": observed_initial.round(4),
        "observed_current": observed_current.round(4),
        "reference_ceiling": reference_ceiling.round(4),
        "vegetation_slope": vegetation_slope.round(5),
        "moisture_recent": moisture_recent.round(4),
        "elevation": elevation.round(1),
        "initial_gap": initial_gap.round(4),
        "human_pressure": human_pressure.round(4),
        "target_regain": target_regain.round(4),
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df)} synthetic cells")


if __name__ == "__main__":
    main()
