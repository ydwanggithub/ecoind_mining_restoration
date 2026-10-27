# -*- coding: utf-8 -*-
"""Response correlogram on the round-1 pair sample and binning (a_spatial_audit.py): distance bins closed on the
right (np.searchsorted side='left', so a pair exactly 1 km apart falls in the 0-1 km bin), mean product of globally
standardized values. The XGBoost 10 km residual correlogram is recomputed the same way and must reproduce round 1.
Writes A23_spatial_shap/response_correlogram_r1bins.csv.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
OUT = Path(str(P.ROUND2 / "outputs" / "A23_spatial_shap"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import Y, read_inputs  # noqa: E402

data, oof, _, _, _ = read_inputs()
y = data[Y].to_numpy(float)
pairs = np.load(R1 / "outputs/a_spatial/residual_pair_sample.npz")
a, b, dist = pairs["first"], pairs["second"], pairs["distance_km"]
binid = np.searchsorted([1, 2, 5, 10, 20, 40], dist, side="left")
zy = (y - y.mean()) / y.std()
res = y - oof["pred_xgboost_spatial_block_10km"].to_numpy()
zr = (res - res.mean()) / res.std()
r1 = pd.read_csv(R1 / "outputs/a_spatial/residual_spatial_correlogram.csv")
r1 = r1[(r1.model == "xgboost") & (r1.cv == "spatial_block_10km")].reset_index(drop=True)
rows = []
for k, (lo, hi) in enumerate(zip([0, 1, 2, 5, 10, 20], [1, 2, 5, 10, 20, 40])):
    ix = binid == k
    rows.append({"distance_low_km": lo, "distance_high_km": hi, "pair_draws": int(ix.sum()),
                 "mean_distance_km": float(dist[ix].mean()), "response_correlation": float(np.mean(zy[a[ix]] * zy[b[ix]])),
                 "residual_correlation_xgb_10km": float(np.mean(zr[a[ix]] * zr[b[ix]]))})
out = pd.DataFrame(rows)
assert np.allclose(out.residual_correlation_xgb_10km, r1.residual_distance_correlation, atol=1e-12), "round-1 residuals not reproduced"
assert (out.pair_draws.to_numpy() == r1.pair_draws.to_numpy()).all()
out.to_csv(OUT / "response_correlogram_r1bins.csv", index=False)
print(out.round(4).to_string(index=False))
