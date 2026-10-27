# -*- coding: utf-8 -*-
"""Round-2 check (ECOLIND-62293R1): where the Lingbei mining area (441 km2 rectangle used in Fig. 14) stands within
the registered sites and ROI5, so that its role as the core mining area of the footprint check can be described.

1. Registered sites inside the rectangle, by county and by recorded deposit scale; site density against ROI5 and the
   prefecture; local site density (sites within 2 km of each site) inside and outside Lingbei.
2. Lingbei cells against all ROI5 cells: ECI in 2000 and 2025 and its change, REGAIN slope, significant trends,
   recovery status, trajectory classes, bright bare detection, elevation, slope and forest fraction.
Writes only to round-2 analysis/outputs/A13_lingbei/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
O1 = R1 / "outputs"
OUT = Path(str(P.ROUND2 / "outputs" / "A13_lingbei"))
OUT.mkdir(parents=True, exist_ok=True)
VEC = Path(str(P.VECTORS))
PINYIN = {"章贡区": "Zhanggong", "南康区": "Nankang", "赣县区": "Ganxian", "信丰县": "Xinfeng", "大余县": "Dayu",
          "上犹县": "Shangyou", "崇义县": "Chongyi", "安远县": "Anyuan", "龙南市": "Longnan", "龙南县": "Longnan",
          "定南县": "Dingnan", "全南县": "Quannan", "宁都县": "Ningdu", "于都县": "Yudu", "兴国县": "Xingguo",
          "会昌县": "Huichang", "寻乌县": "Xunwu", "石城县": "Shicheng", "瑞金市": "Ruijin"}

sites = gpd.read_file(VEC / "Ganzhou_IARE_mining_sites_utm.shp").to_crs(32650)
rect = gpd.read_file(VEC / "Dingnan_lingbei_REMA_rec_bound.shp").to_crs(32650)
cty = gpd.read_file(str(P.BOUNDARY / "Ganzhou_counties.shp")).to_crs(32650)
cty["county"] = cty["name"].map(PINYIN).fillna(cty["name"])
sites = gpd.sjoin(sites, cty[["county", "geometry"]], how="left", predicate="within").drop(columns="index_right")
inside = sites.within(rect.unary_union)
rect_km2 = rect.area.sum() / 1e6
xy = np.column_stack([sites.geometry.x, sites.geometry.y])
tree = cKDTree(xy)
n2 = np.array([len(tree.query_ball_point(p, 2000)) - 1 for p in xy])
lines = []
lines.append(f"sites inside Lingbei rectangle: {int(inside.sum())} of {len(sites)} ({100 * inside.mean():.1f}%), rectangle {rect_km2:.1f} km2 "
             f"= {100 * rect_km2 / 39431:.2f}% of the prefecture")
lines.append(f"site density per 100 km2: Lingbei {100 * inside.sum() / rect_km2:.1f}; ROI5 {100 * len(sites) / 8769:.1f}; prefecture {100 * len(sites) / 39431:.2f}")
lines.append(f"neighbours within 2 km (median): Lingbei sites {np.median(n2[inside]):.0f}; other sites {np.median(n2[~inside]):.0f}")
cnt = sites.county.value_counts()
lines.append("sites by county: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
lines.append("Lingbei sites by county: " + ", ".join(f"{k} {v}" for k, v in sites[inside].county.value_counts().items()))
scale_col = "规模"
if scale_col in sites.columns:
    lines.append("deposit scale, all sites: " + ", ".join(f"{k} {v}" for k, v in sites[scale_col].value_counts(dropna=False).items()))
    lines.append("deposit scale, Lingbei sites: " + ", ".join(f"{k} {v}" for k, v in sites[inside][scale_col].value_counts(dropna=False).items()))

# Lingbei cells against ROI5
fc = pd.read_csv(O1 / "n_footprint_check/footprint_cells.csv.gz", usecols=["grid_id", "in_rema"])
ling = set(fc.loc[fc.in_rema.astype(str).str.lower().eq("true"), "grid_id"])
dist = pd.read_csv(O1 / "hjk_landscape/roi_distance_analysis_cells.csv.gz",
                   usecols=["grid_id", "eci_2000", "eci_2025", "eci_change", "regain_sen_slope_2000_2025", "trajectory_class"])
mk = pd.read_csv(O1 / "f_annual/modified_mk_cells.csv.gz", usecols=["grid_id", "modified_p", "sen_slope", "modified_status"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import read_inputs  # noqa: E402
data, oof, _, features, _ = read_inputs()
pred = data[["grid_id", "elevation_mean", "slope_mean", "forest_frac_mean_2000_2025"] +
            [c for c in data.columns if "bright" in c.lower() or "bare_freq" in c.lower()][:1]]
cells = dist.merge(mk, on="grid_id", how="left").merge(pred, on="grid_id", how="left")
cells["Lingbei"] = cells.grid_id.isin(ling)
bb_col = [c for c in pred.columns if c not in ("grid_id", "elevation_mean", "slope_mean", "forest_frac_mean_2000_2025")]
rows = []
for name, g in (("Lingbei", cells[cells.Lingbei]), ("ROI5", cells)):
    sig = g.modified_p < 0.05
    r = {"domain": name, "n_cells": len(g), "eci_2000_median": g.eci_2000.median(), "eci_2025_median": g.eci_2025.median(),
         "eci_change_median": g.eci_change.median(), "regain_slope_median": g.regain_sen_slope_2000_2025.median(),
         "positive_slope_pct": 100 * np.mean(g.sen_slope > 0), "sig_improve_pct": 100 * np.mean(sig & (g.sen_slope > 0)),
         "sig_decline_pct": 100 * np.mean(sig & (g.sen_slope < 0)),
         "elevation_median": g.elevation_mean.median(), "slope_median": g.slope_mean.median(),
         "forest_frac_median": g.forest_frac_mean_2000_2025.median()}
    if bb_col:
        r["bright_bare_detected_pct"] = 100 * np.mean(g[bb_col[0]] > 0)
    for k, v in g.modified_status.value_counts(normalize=True).items():
        r[f"status_{k}_pct"] = 100 * v
    for k, v in g.trajectory_class.value_counts(normalize=True).items():
        r[f"traj_{k}_pct"] = 100 * v
    rows.append(r)
tab = pd.DataFrame(rows).set_index("domain").T
tab.to_csv(OUT / "lingbei_vs_roi5.csv")
# where Lingbei medians fall within the ROI5 distributions (percentiles)
for v in ("eci_2000", "eci_2025", "eci_change", "regain_sen_slope_2000_2025", "elevation_mean", "slope_mean"):
    med = cells.loc[cells.Lingbei, v].median()
    lines.append(f"Lingbei median {v} = {med:.3f} sits at the {100 * np.mean(cells[v] <= med):.0f}th percentile of ROI5")
(OUT / "lingbei_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(tab.round(3).to_string())
