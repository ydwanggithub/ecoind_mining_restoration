# -*- coding: utf-8 -*-
"""Unrounded TreeSHAP group shares for the XGBoost refits without rapidly urbanizing cells (r2_xgb_urban_exclusion.py
stored shares rounded to four decimals). Same configuration, folds and cell selections. Writes
A10_sem/xgb_urban_exclusion_shares_exact.csv.
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
rows = []
for name, keep in (("without fastest built-up expansion (top 5%)", blt <= np.quantile(blt, 0.95)),
                   ("without top 5% and Zhanggong", (blt <= np.quantile(blt, 0.95)) & (cty != "Zhanggong"))):
    X = data.loc[keep, features].to_numpy(float)
    y = y_all[keep]
    folds = folds_all[keep]
    pred = np.full(len(y), np.nan)
    contrib = np.full(X.shape, np.nan)
    for k in np.unique(folds):
        te = folds == k
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
        contrib[te] = m.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    imp = pd.Series(np.abs(contrib).mean(0), index=[fam[c] for c in features]).groupby(level=0).sum()
    sh = imp / imp.sum()
    rows.append({"variant": name, "n_cells": int(keep.sum()), "pooled_r2": float(r2), **{f"share_{g}": float(v) for g, v in sh.items()}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "xgb_urban_exclusion_shares_exact.csv", index=False)
