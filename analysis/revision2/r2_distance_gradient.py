# -*- coding: utf-8 -*-
"""Round-2 analysis (ECOLIND-62293R1, Reviewer 3 on the 5 km buffer): ECI by distance from the registered sites,
0-15 km, and the share of Lingbei footprint area by distance.

Distance bands use the distance from each 500 m cell centroid to the nearest registered site (ROI5 and BG15 cells
import _paths as P  # internal input folders, set by environment variables (README)
from the round-1 landscape tables; centroids rebuilt from grid_id, checked against stored ROI5 distances).
For each band we report raw medians (IQR) and means standardized to the ROI5 composition of strata, either
terrain only (elevation x slope) or terrain and 2000 land cover as in Table S9; strata need >=30 cells in a band.
95% intervals come from 1,000 bootstrap draws of whole 10 km blocks (block = floor of centroid / 10 km).
Writes only to round-2 analysis/outputs/A11_buffer_explore/.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1 / "outputs" / "hjk_landscape"))
OUT = Path(str(P.ROUND2 / "outputs" / "A11_buffer_explore"))
OUT.mkdir(parents=True, exist_ok=True)
sites = gpd.read_file(str(P.VECTORS / "Ganzhou_IARE_mining_sites_utm.shp")).to_crs(32650)
tree = cKDTree(np.column_stack([sites.geometry.x, sites.geometry.y]))
VARS = ["eci_2000", "eci_2025", "eci_change"]
BINS = [0, 1, 2, 3, 5, 7.5, 10, 15]
LABELS = ["0-1", "1-2", "2-3", "3-5", "5-7.5", "7.5-10", "10-15"]

land = pd.read_csv(R1 / "landscape_comparison_cells.csv.gz").drop_duplicates("grid_id")
# BG15 cells that duplicate ROI5 cells are removed, as in Table S9 (6,507 cells)
shared = (land.domain == "BG15") & land.shared_BG15_footprint.astype(str).str.lower().eq("true")
print("BG15 cells shared with ROI5 removed:", int(shared.sum()))
land = land[~shared].copy()
parts = land.grid_id.str.split("_", expand=True)
x = (parts[1].astype(int).to_numpy() + 0.5) * 500.0
y = (parts[2].astype(int).to_numpy() + 0.5) * 500.0
land["distance_km"] = tree.query(np.column_stack([x, y]), k=1)[0] / 1000
roi = pd.read_csv(R1 / "roi_distance_analysis_cells.csv.gz", usecols=["grid_id", "distance_km"]).set_index("grid_id").distance_km
chk = land.set_index("grid_id").distance_km.reindex(roi.index)
assert float(np.nanmax(np.abs(chk - roi))) < 1e-6, "centroid reconstruction does not match stored ROI5 distances"
land["block"] = (np.floor(x / 10000).astype(int) * 1000 + np.floor(y / 10000).astype(int))
land["band"] = pd.cut(land.distance_km, BINS, labels=LABELS, right=False)
land = land.dropna(subset=["band"] + VARS).copy()
land["s_terrain"] = land.elevation_bin.astype(str) + "|" + land.slope_bin.astype(str)
land["s_full"] = land.s_terrain + "|" + land.dominant_cover.astype(str)

ublock, binv = np.unique(land.block.to_numpy(), return_inverse=True)
band_i = land.band.cat.codes.to_numpy()
rng = np.random.default_rng(20261022)
draws = [np.bincount(rng.integers(0, len(ublock), len(ublock)), minlength=len(ublock)) for _ in range(1000)]
rows = {lab: {"band_km": lab, "n_cells": int((land.band == lab).sum()), "n_blocks": int(land.loc[land.band == lab, "block"].nunique())}
        for lab in LABELS}
for lab in LABELS:
    g = land[land.band == lab]
    for v in VARS:
        q = g[v].quantile([0.25, 0.5, 0.75])
        rows[lab].update({f"raw_{v}_median": q[0.5], f"raw_{v}_q25": q[0.25], f"raw_{v}_q75": q[0.75]})

for scheme in ("s_terrain", "s_full"):
    s_codes, s_inv = np.unique(land[scheme].to_numpy(), return_inverse=True)
    ref = land[land.domain == "ROI5"][scheme].value_counts(normalize=True).reindex(s_codes).fillna(0).to_numpy()
    nb, ns, nk = len(ublock), len(s_codes), len(LABELS)
    cnt = np.zeros((nb, nk, ns))
    np.add.at(cnt, (binv, band_i, s_inv), 1)
    sums = np.zeros((len(VARS), nb, nk, ns))
    for j, v in enumerate(VARS):
        np.add.at(sums[j], (binv, band_i, s_inv), land[v].to_numpy())
    eligible = cnt.sum(0) >= 30            # strata with >= 30 cells in a band (full sample)

    def standardized(mult):
        c = np.tensordot(mult, cnt, axes=1)                      # band x stratum
        out = np.full((len(VARS), nk), np.nan)
        for k in range(nk):
            ok = eligible[k] & (c[k] > 0)
            w = ref * ok
            if w.sum() == 0:
                continue
            for j in range(len(VARS)):
                s = np.tensordot(mult, sums[j][:, k, :], axes=1)
                out[j, k] = np.sum(w[ok] * s[ok] / c[k][ok]) / w.sum()
        return out

    est = standardized(np.ones(nb))
    boot = np.array([standardized(m) for m in draws])
    cover = [100 * float((ref * eligible[k]).sum()) for k in range(nk)]
    tag = "terrain" if scheme == "s_terrain" else "full"
    for k, lab in enumerate(LABELS):
        rows[lab][f"{tag}_strata_cover_pct"] = cover[k]
        for j, v in enumerate(VARS):
            lo, hi = np.nanquantile(boot[:, j, k], [0.025, 0.975])
            rows[lab].update({f"{tag}_{v}": est[j, k], f"{tag}_{v}_lo": lo, f"{tag}_{v}_hi": hi})
res = pd.DataFrame(list(rows.values()))
res.to_csv(OUT / "eci_distance_gradient_bootstrap.csv", index=False)
show = ["band_km", "n_cells", "raw_eci_2000_median", "raw_eci_2025_median", "raw_eci_change_median",
        "terrain_eci_2000", "terrain_eci_2025", "terrain_eci_change", "terrain_eci_change_lo", "terrain_eci_change_hi",
        "full_eci_2000", "full_eci_2025", "full_eci_change", "full_eci_change_lo", "full_eci_change_hi",
        "terrain_strata_cover_pct", "full_strata_cover_pct"]
print(res[show].round(3).to_string(index=False))

# cumulative share of Lingbei footprint area by distance (30 m sampling, as in r2_buffer_explore.py)
fp = gpd.read_file(str(P.VECTORS / "Dingnan_mining_sites_SAM2025.shp")).to_crs(32650)
pts = []
for geom in fp.geometry[fp.geometry.notna()]:
    x0, y0, x1, y1 = geom.bounds
    gx, gy = np.meshgrid(np.arange(x0 + 15, x1, 30.0), np.arange(y0 + 15, y1, 30.0))
    cand = gpd.GeoSeries(gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=32650)
    inside = cand[cand.within(geom)] if len(cand) else cand
    if len(inside) == 0:
        c = geom.representative_point()
        pts.append((c.x, c.y))
    else:
        pts.extend(zip(inside.x, inside.y))
d_fp = tree.query(np.array(pts), k=1)[0] / 1000
grid = np.round(np.arange(0, 5.001, 0.05), 2)
pd.DataFrame({"distance_km": grid, "cum_share_pct": [100 * np.mean(d_fp <= g) for g in grid]}).to_csv(
    OUT / "lingbei_footprint_cumulative_by_distance.csv", index=False)
print("footprint area within 1/2/3 km: %.1f / %.1f / %.1f %%; max %.2f km" %
      (100 * np.mean(d_fp <= 1), 100 * np.mean(d_fp <= 2), 100 * np.mean(d_fp <= 3), d_fp.max()))
