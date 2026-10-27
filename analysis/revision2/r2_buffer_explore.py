# -*- coding: utf-8 -*-
"""Round-2 exploration (ECOLIND-62293R1, Reviewer 3 on the 5 km buffer): how far mapped disturbance and the
condition gradient extend from the registered sites.

1. Lingbei field-checked footprints (2025): share of footprint area within given distances of the nearest
   registered site (30 m point sampling inside the polygons).
2. Distance gradient of ECI from the sites out to 15 km, using the round-1 ROI5 and BG15 cell tables
   (cell centroids rebuilt from grid_id and checked against the stored ROI5 distances); raw medians and means
   standardized to the ROI5 composition of elevation, slope and 2000 land cover strata.
Read-only inputs; writes only to round-2 analysis/outputs/A11_buffer_explore/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1 / "outputs" / "hjk_landscape"))
OUT = Path(str(P.ROUND2 / "outputs" / "A11_buffer_explore"))
OUT.mkdir(parents=True, exist_ok=True)
sites = gpd.read_file(str(P.VECTORS / "Ganzhou_IARE_mining_sites_utm.shp")).to_crs(32650)
tree = cKDTree(np.column_stack([sites.geometry.x, sites.geometry.y]))

# 1. footprint area by distance to the nearest registered site
fp = gpd.read_file(str(P.VECTORS / "Dingnan_mining_sites_SAM2025.shp")).to_crs(32650)
fp = fp[fp.geometry.notna() & ~fp.geometry.is_empty]
pts = []
for geom in fp.geometry:
    x0, y0, x1, y1 = geom.bounds
    xs = np.arange(x0 + 15, x1, 30.0)
    ys = np.arange(y0 + 15, y1, 30.0)
    if len(xs) == 0 or len(ys) == 0:
        c = geom.representative_point()
        pts.append((c.x, c.y))
        continue
    gx, gy = np.meshgrid(xs, ys)
    cand = gpd.GeoSeries(gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=32650)
    inside = cand[cand.within(geom)]
    if len(inside) == 0:
        c = geom.representative_point()
        pts.append((c.x, c.y))
    else:
        pts.extend(zip(inside.x, inside.y))
pts = np.array(pts)
d_fp = tree.query(pts, k=1)[0] / 1000
cuts = [0.25, 0.5, 1, 1.5, 2, 3, 5]
fp_tab = pd.DataFrame({"within_km": cuts, "share_of_footprint_area_pct": [100 * np.mean(d_fp <= c) for c in cuts]})
fp_tab.to_csv(OUT / "lingbei_footprint_area_by_distance.csv", index=False)
print("footprints:", len(fp), "polygons; area km2:", round(fp.area.sum() / 1e6, 2), "; sample points:", len(pts))
print("footprint distance quantiles (km): 50%", round(np.quantile(d_fp, 0.5), 2), "90%", round(np.quantile(d_fp, 0.9), 2),
      "95%", round(np.quantile(d_fp, 0.95), 2), "max", round(d_fp.max(), 2))
print(fp_tab.round(1).to_string(index=False))

# 2. distance gradient of ECI to 15 km
roi = pd.read_csv(R1 / "roi_distance_analysis_cells.csv.gz")
land = pd.read_csv(R1 / "landscape_comparison_cells.csv.gz")


def centroids(ids):
    parts = ids.str.split("_", expand=True)
    return (parts[1].astype(int).to_numpy() + 0.5) * 500.0, (parts[2].astype(int).to_numpy() + 0.5) * 500.0


x, y = centroids(roi.grid_id)
check = tree.query(np.column_stack([x, y]), k=1)[0] / 1000
print("centroid check vs stored ROI5 distance: max abs diff km =", round(float(np.nanmax(np.abs(check - roi.distance_km))), 4))
x, y = centroids(land.grid_id)
land["distance_km"] = tree.query(np.column_stack([x, y]), k=1)[0] / 1000
land = land.drop_duplicates("grid_id")
bins = [0, 1, 2, 3, 5, 7.5, 10, 15, np.inf]
labels = ["0-1", "1-2", "2-3", "3-5", "5-7.5", "7.5-10", "10-15", ">15"]
land["band"] = pd.cut(land.distance_km, bins, labels=labels, right=False)
land["stratum"] = land.elevation_bin.astype(str) + "|" + land.slope_bin.astype(str) + "|" + land.dominant_cover.astype(str)
ref = land[land.domain == "ROI5"].stratum.value_counts(normalize=True)
rows = []
for b, g in land.groupby("band", observed=True):
    m = g.groupby("stratum")[["eci_2000", "eci_2025", "eci_change"]].mean()
    n = g.stratum.value_counts()
    common = ref.index.intersection(m.index[n.reindex(m.index).fillna(0) >= 30])
    w = ref.reindex(common)
    std = (m.loc[common].mul(w, axis=0).sum() / w.sum()) if len(common) else pd.Series(np.nan, index=m.columns)
    rows.append({"band_km": b, "n_cells": len(g), "median_eci_2000": g.eci_2000.median(), "median_eci_2025": g.eci_2025.median(),
                 "median_change": g.eci_change.median(), "std_eci_2000": std.eci_2000, "std_eci_2025": std.eci_2025,
                 "std_change": std.eci_change, "roi5_strata_covered_pct": 100 * float(w.sum())})
grad = pd.DataFrame(rows)
grad.to_csv(OUT / "eci_distance_gradient_to_15km.csv", index=False)
print(grad.round(3).to_string(index=False))
