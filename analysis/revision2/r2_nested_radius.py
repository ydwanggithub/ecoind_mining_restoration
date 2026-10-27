# -*- coding: utf-8 -*-
"""Round-2 check (ECOLIND-62293R1, Reviewer 3 on the 5 km buffer): do the main results hold in cumulative
neighborhoods of 1, 2, 3 and 5 km around the registered sites (cell centroid distance) and in the full ROI5?

Per neighborhood: ECI medians in 2000 and 2025; shares of significant improvement and decline under the per-cell
Hamed-Rao test and under Benjamini-Hochberg FDR recomputed within the neighborhood; share of stalled or declining
cells under the primary status rule (2025 REGAIN >= 0.75, per-cell alpha 0.05); XGBoost (primary configuration,
the same 10 km block folds restricted to the neighborhood) pooled R2, TreeSHAP group shares and the R2 lost when the
land cover group is removed. Writes only to round-2 analysis/outputs/A12_nested_radius/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
O1 = R1 / "outputs"
OUT = Path(str(P.ROUND2 / "outputs" / "A12_nested_radius"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import Y, read_inputs  # noqa: E402

PARAMS = json.loads((R1 / "config" / "bcd_model_protocol.json").read_text(encoding="utf-8"))["baseline"]
data, oof, _, features, _ = read_inputs()
fam = pd.read_csv(O1 / "bcd_models/shap_rank_comparison.csv")
fam = fam[fam.configuration == "baseline"].set_index("feature")["family"].to_dict()
lc = [f for f in features if fam[f] == "Land cover"]
y_all = data[Y].to_numpy(float)
folds_all = oof.fold_spatial_block_10km.to_numpy()

dist = pd.read_csv(O1 / "hjk_landscape/roi_distance_analysis_cells.csv.gz",
                   usecols=["grid_id", "distance_km", "eci_2000", "eci_2025", "eci_change"]).set_index("grid_id")
m = pd.read_csv(O1 / "f_annual/modified_mk_cells.csv.gz")
z = np.load(O1 / "f_annual/annual_roi5.npz", allow_pickle=True)
yrs = list(z["year"])
reg = pd.DataFrame({"grid_id": z["grid_id"], "r2025": z["regain"][:, yrs.index(2025)]})
cells = m.merge(reg, on="grid_id", how="left", validate="one_to_one").merge(dist, left_on="grid_id", right_index=True,
                                                                          how="left", validate="one_to_one")
d_model = dist.distance_km.reindex(oof.grid_id).to_numpy()
assert np.isfinite(d_model).all(), "model cells without a distance"


def bh(pv):
    n = len(pv)
    o = np.argsort(pv)
    q = np.empty(n)
    q[o] = np.minimum(np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1], 1)
    return q


def xgb_run(keep, cols, want_shap):
    X = data.loc[keep, cols].to_numpy(float)
    y = y_all[keep]
    folds = folds_all[keep]
    pred = np.full(len(y), np.nan)
    contrib = np.full(X.shape, np.nan)
    for k in np.unique(folds):
        te = folds == k
        mdl = xgb.XGBRegressor(**PARAMS)
        mdl.fit(X[~te], y[~te])
        pred[te] = mdl.predict(X[te])
        if want_shap:
            contrib[te] = mdl.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
    out = {"r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)), "n_folds": int(len(np.unique(folds)))}
    if want_shap:
        imp = pd.Series(np.abs(contrib).mean(0), index=[fam[c] for c in cols]).groupby(level=0).sum()
        out["shares"] = (imp / imp.sum()).to_dict()
    return out


rows = []
for name, R in (("1 km", 1.0), ("2 km", 2.0), ("3 km", 3.0), ("5 km", 5.0), ("ROI5, all cells", None)):
    sel = cells if R is None else cells[cells.distance_km <= R]
    p, s = sel.modified_p.to_numpy(float), sel.sen_slope.to_numpy(float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    q[ok] = bh(p[ok])
    sig, sigq = ok & (p < 0.05), ok & (q < 0.05)
    hi = sel.r2025.to_numpy(float) >= 0.75
    stalled = ok & ~sig & ~hi
    decline = sig & (s < 0)
    row = {"neighborhood": name, "n_cells": len(sel), "eci_2000_median": sel.eci_2000.median(),
           "eci_2025_median": sel.eci_2025.median(), "eci_change_median": sel.eci_change.median(),
           "improve_pct": 100 * np.mean(sig & (s > 0)), "decline_pct": 100 * np.mean(decline),
           "improve_fdr_pct": 100 * np.mean(sigq & (s > 0)), "decline_fdr_pct": 100 * np.mean(sigq & (s < 0)),
           "stalled_or_declining_pct": 100 * np.mean(stalled | decline)}
    keep = np.ones(len(data), bool) if R is None else d_model <= R
    full = xgb_run(keep, features, True)
    nolc = xgb_run(keep, [f for f in features if f not in lc], False)
    sh = pd.Series(full["shares"]).sort_values(ascending=False)
    row.update({"model_cells": int(keep.sum()), "folds": full["n_folds"], "xgb_r2": full["r2"],
                "land_cover_share": sh["Land cover"], "land_cover_rank": int(list(sh.index).index("Land cover") + 1),
                "second_group": sh.index[1], "second_share": sh.iloc[1], "r2_without_land_cover": nolc["r2"],
                "r2_drop_without_land_cover": full["r2"] - nolc["r2"]})
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
pd.DataFrame(rows).to_csv(OUT / "nested_radius_summary.csv", index=False)
print("done")
