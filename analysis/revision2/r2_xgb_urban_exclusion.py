# -*- coding: utf-8 -*-
"""Round-2 check (ECOLIND-62293R1, Reviewer 3 comments 6 and 7): is land cover still the leading predictor group of
recovery direction once the fastest-urbanizing cells are removed?

Primary XGBoost configuration, the same 10 km block folds, all 57 predictors versus the land cover group removed, on
(a) all cells, (b) cells without the 5% fastest built-up expansion, (c) the same plus Zhanggong removed. Reports pooled
R2 and TreeSHAP group shares. Writes only to round-2 analysis/outputs/A10_sem/.
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
OUT = Path(str(P.ROUND2 / "outputs" / "A10_sem"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import Y, read_inputs  # noqa: E402

PARAMS = json.loads((R1 / "config" / "bcd_model_protocol.json").read_text(encoding="utf-8"))["baseline"]
data, oof, _, features, _ = read_inputs()
fam = pd.read_csv(R1 / "outputs/bcd_models/shap_rank_comparison.csv")
fam = fam[fam.configuration == "baseline"].set_index("feature")["family"].to_dict()
county = pd.read_csv(R1 / "outputs/m_body/revised_cell_products.csv.gz", usecols=["grid_id", "county"]).set_index("grid_id").county
cty = county.reindex(oof.grid_id).to_numpy()
blt = data["built_frac_slope_2000_2025"].to_numpy()
y_all = data[Y].to_numpy(float)
folds_all = oof.fold_spatial_block_10km.to_numpy()
lc = [f for f in features if fam[f] == "Land cover"]


def run(keep, cols, want_shap):
    X = data.loc[keep, cols].to_numpy(float)
    y = y_all[keep]
    folds = folds_all[keep]
    pred = np.full(len(y), np.nan)
    contrib = np.full(X.shape, np.nan)
    for k in np.unique(folds):
        te = folds == k
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
        if want_shap:
            contrib[te] = m.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    out = {"pooled_r2": float(r2)}
    if want_shap:
        imp = np.abs(contrib).mean(0)
        g = pd.Series(imp, index=[fam[c] for c in cols]).groupby(level=0).sum()
        out["group_shares"] = (g / g.sum()).round(4).to_dict()
        out["top5"] = [cols[i] for i in np.argsort(-imp)[:5]]
    return out


variants = {"all cells": np.ones(len(data), bool),
            "without fastest built-up expansion (top 5%)": blt <= np.quantile(blt, 0.95),
            "without top 5% and Zhanggong": (blt <= np.quantile(blt, 0.95)) & (cty != "Zhanggong")}
res = {}
for name, keep in variants.items():
    full = run(keep, features, True)
    nolc = run(keep, [f for f in features if f not in lc], False)
    res[name] = {"n_cells": int(keep.sum()), "all_57": full, "without_land_cover_group_r2": nolc["pooled_r2"],
                 "r2_drop_without_land_cover": full["pooled_r2"] - nolc["pooled_r2"]}
    print(name, json.dumps(res[name], indent=1), flush=True)
(OUT / "xgb_urban_exclusion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
