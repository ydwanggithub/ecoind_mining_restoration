# -*- coding: utf-8 -*-
"""Recompute the trend and status columns of the nested-radius table with the published status rule
(r2_quick_checks.py: stalled = evaluable, neither significant increase nor decrease, 2025 REGAIN < 0.75), which the
first run (r2_nested_radius.py) approximated with 'not significant'; XGBoost columns are carried over unchanged.
Writes nested_radius_summary_v2.csv next to the first table.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
O1 = Path(str(P.ROUND1 / "outputs"))
OUT = Path(str(P.ROUND2 / "outputs" / "A12_nested_radius"))
dist = pd.read_csv(O1 / "hjk_landscape/roi_distance_analysis_cells.csv.gz", usecols=["grid_id", "distance_km"]).set_index("grid_id")
m = pd.read_csv(O1 / "f_annual/modified_mk_cells.csv.gz")
z = np.load(O1 / "f_annual/annual_roi5.npz", allow_pickle=True)
yrs = list(z["year"])
reg = pd.DataFrame({"grid_id": z["grid_id"], "r2025": z["regain"][:, yrs.index(2025)]})
cells = m.merge(reg, on="grid_id", how="left", validate="one_to_one").merge(dist, left_on="grid_id", right_index=True, how="left")


def bh(pv):
    n = len(pv)
    o = np.argsort(pv)
    q = np.empty(n)
    q[o] = np.minimum(np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1], 1)
    return q


def status_shares(sel, use_q):
    p, s = sel.modified_p.to_numpy(float), sel.sen_slope.to_numpy(float)
    ev = np.isfinite(p)
    val = p.copy()
    if use_q:
        val[ev] = bh(p[ev])
    sig = ev & (val < 0.05)
    dec, inc = sig & (s < 0), sig & (s > 0)
    hi = sel.r2025.to_numpy(float) >= 0.75
    stalled = ev & ~dec & ~inc & ~hi
    return 100 * np.mean(inc), 100 * np.mean(dec), 100 * np.mean(stalled), 100 * np.mean(stalled | dec)


first = pd.read_csv(OUT / "nested_radius_summary.csv").set_index("neighborhood")
rows = []
for name, R in (("1 km", 1.0), ("2 km", 2.0), ("3 km", 3.0), ("5 km", 5.0), ("ROI5, all cells", None)):
    sel = cells if R is None else cells[cells.distance_km <= R]
    inc, dec, st, sd = status_shares(sel, False)
    incq, decq, stq, sdq = status_shares(sel, True)
    rows.append({"neighborhood": name, "n_cells": len(sel), "improve_pct": inc, "decline_pct": dec, "stalled_pct": st,
                 "stalled_or_declining_pct": sd, "improve_fdr_pct": incq, "decline_fdr_pct": decq,
                 "stalled_or_declining_fdr_pct": sdq})
new = pd.DataFrame(rows).set_index("neighborhood")
keep = [c for c in first.columns if c not in new.columns]
out = new.join(first[keep])
check = out.loc["ROI5, all cells", "stalled_or_declining_pct"]
assert abs(check - 18.475165) < 1e-3, f"ROI5 stalled+declining {check} does not reproduce the published 18.475%"
out.reset_index().to_csv(OUT / "nested_radius_summary_v2.csv", index=False)
print(out[["n_cells", "improve_pct", "improve_fdr_pct", "decline_pct", "stalled_or_declining_pct",
           "stalled_or_declining_fdr_pct", "xgb_r2", "land_cover_share", "r2_drop_without_land_cover"]].round(3).to_string())
