# -*- coding: utf-8 -*-
"""Companion to r2_sem_urban_exclusion.py: total associations of every construct with recovery direction in each
urban-exclusion variant, so the ranking of land cover can be checked (Reviewer 3 comments 6e and 7b).

Same engine, indicators, structural ordering, frozen full-sample loadings and 1,000 whole-block bootstrap refits.
Also reports how the cells with the fastest built-up expansion are spread across counties.
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
R = ORDER.index("REGAIN")
top5 = blt > np.quantile(blt, 0.95)

variants = {
    "full": np.ones(len(data), bool),
    "without Zhanggong": cty != "Zhanggong",
    "without fastest built-up expansion (top 5%)": ~top5,
    "without both": (cty != "Zhanggong") & ~top5,
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
                draws.append([f["total"][R, j] for j in range(len(ORDER) - 1)])
            except Exception:
                pass
        dr = np.array(draws)
        for j, g in enumerate(ORDER[:-1]):
            lo, hi = np.quantile(dr[:, j], [0.025, 0.975])
            rows.append({"variant": name, "n_cells": int(keep.sum()), "construct": g, "total_association": float(fit["total"][R, j]),
                         "ci_low": float(lo), "ci_high": float(hi), "successful_draws": int(len(dr))})
        print(name, {g: round(float(fit["total"][R, j]), 3) for j, g in enumerate(ORDER[:-1])}, flush=True)
res = pd.DataFrame(rows)
res["rank_by_abs"] = res.groupby("variant")["total_association"].transform(lambda s: s.abs().rank(ascending=False)).astype(int)
res.to_csv(OUT / "sem_urban_exclusion_all_totals.csv", index=False)
print(res.round(3).to_string())

spread = pd.Series(cty[top5]).value_counts().rename_axis("county").reset_index(name="top5_cells")
spread["share_pct"] = 100 * spread.top5_cells / top5.sum()
spread.to_csv(OUT / "top5_builtup_cells_by_county.csv", index=False)
print("top 5% built-up cells:", int(top5.sum()), "in", spread.shape[0], "counties")
print(spread.round(1).to_string(index=False))
