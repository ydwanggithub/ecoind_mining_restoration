# -*- coding: utf-8 -*-
"""Round-2 check (ECOLIND-62293R1, Reviewer 3 comments 6 and 7): does the negative human activity path through land
cover persist when the central urban district (Zhanggong) or the cells with the fastest built-up expansion are removed?

Reuses the round-1 covariance implementation of the plspm 0.5.7 Mode B / PATH model (e_sem_diagnostics.py, read only),
the 25 retained indicators and the structural ordering; 1,000 bootstrap refits resampling whole 10 km blocks per variant.
Writes only to round-2 analysis/outputs/A10_sem/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
OUT = Path(str(P.ROUND2 / "outputs" / "A10_sem"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import Y, read_inputs  # noqa: E402
from e_sem_diagnostics import ORDER, covariance_fit, engine, moments  # noqa: E402

data, oof, _, _, _ = read_inputs()
eng, source = engine()
selected = json.loads((R1 / "config" / "e_sem_protocol.json").read_text(encoding="utf-8"))["selected"]
columns = [c for g in ORDER[:-1] for c in selected[g]] + [Y]
groups = [np.array([columns.index(c) for c in selected[g]], dtype=int) for g in ORDER[:-1]] + [np.array([len(columns) - 1])]
path = np.zeros((6, 6))
for a, b in eng.structural_edges("main"):
    path[ORDER.index(b), ORDER.index(a)] = 1
anchors = [columns.index(source.ORIENTATION_ANCHORS[g]) for g in ORDER]
county = pd.read_csv(R1 / "outputs/m_body/revised_cell_products.csv.gz", usecols=["grid_id", "county"]).set_index("grid_id").county
cty = county.reindex(oof.grid_id).to_numpy()
blt = data["built_frac_slope_2000_2025"].to_numpy()
X_all = data[columns].to_numpy(dtype=float)
block_all = oof.spatial_block_10km.to_numpy()
H, L, R = ORDER.index("Human"), ORDER.index("LandCover"), ORDER.index("REGAIN")

variants = {
    "full": np.ones(len(data), bool),
    "without Zhanggong": cty != "Zhanggong",
    "without fastest built-up expansion (top 5%)": blt <= np.quantile(blt, 0.95),
    "without both": (cty != "Zhanggong") & (blt <= np.quantile(blt, 0.95)),
}
rows = []
with threadpool_limits(limits=1):
    ref = covariance_fit(*moments(X_all), groups, path, anchors)
    for name, keep in variants.items():
        X = X_all[keep]
        fit = covariance_fit(*moments(X), groups, path, anchors, ref["loadings"] if name != "full" else None)
        block = block_all[keep]
        uniq, inv = np.unique(block, return_inverse=True)
        idx = [np.flatnonzero(inv == j) for j in range(len(uniq))]
        counts = np.array([len(i) for i in idx])
        sums = np.array([X[i].sum(0) for i in idx])
        secs = np.array([X[i].T @ X[i] for i in idx])
        rng = np.random.default_rng(20260923)
        draws = []
        for _ in range(1000):
            mlt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            n = int(mlt @ counts)
            try:
                f = covariance_fit(mlt @ sums / n, np.tensordot(mlt, secs, axes=1) / n, n, groups, path, anchors, ref["loadings"])
                draws.append((f["paths"][R, H], f["total"][R, H] - f["paths"][R, H], f["total"][R, H], f["total"][R, L], f["r2"][R]))
            except Exception:
                pass
        dr = np.array(draws)
        est = (fit["paths"][R, H], fit["total"][R, H] - fit["paths"][R, H], fit["total"][R, H], fit["total"][R, L], fit["r2"][R])
        for k, nm in enumerate(["human_direct", "human_indirect", "human_total", "landcover_total", "regain_r2"]):
            lo, hi = np.quantile(dr[:, k], [0.025, 0.975])
            rows.append({"variant": name, "n_cells": int(keep.sum()), "n_blocks": int(len(uniq)), "quantity": nm,
                         "estimate": float(est[k]), "ci_low": float(lo), "ci_high": float(hi), "successful_draws": int(len(dr))})
        print(name, int(keep.sum()), [round(float(e), 3) for e in est], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "sem_urban_exclusion.csv", index=False)
print(res.round(3).to_string())
