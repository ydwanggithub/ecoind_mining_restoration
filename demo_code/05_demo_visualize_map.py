"""Scatter the synthetic cells over a square extent coloured by REGAIN.

Produces demo_code/regain_map_demo.png. The synthetic coordinates do not
correspond to any real location; the example only illustrates how a per-cell
REGAIN layer would be rendered.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_regain_demo.csv"
OUT_PNG = Path(__file__).resolve().parent / "regain_map_demo.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    d_init = np.maximum(0.0, df["reference_ceiling"] - df["observed_initial"])
    d_curr = np.maximum(0.0, df["reference_ceiling"] - df["observed_current"])
    valid = d_init > 0.02
    regain = np.where(valid, (d_init - d_curr) / (d_init + d_curr + 1e-9), np.nan)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sc = ax.scatter(df["x"], df["y"], c=regain, cmap="RdBu", norm=Normalize(-1, 1), s=12, edgecolor="none")
    ax.set_xlabel("x (synthetic units)")
    ax.set_ylabel("y (synthetic units)")
    ax.set_title("REGAIN demonstration map")
    ax.set_aspect("equal")
    cb = plt.colorbar(sc, ax=ax, fraction=0.045)
    cb.set_label("REGAIN")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PNG}")
    print(f"Median REGAIN: {np.nanmedian(regain):.3f}")
    print(f"Valid cells: {int(valid.sum())} of {len(df)}")


if __name__ == "__main__":
    main()
