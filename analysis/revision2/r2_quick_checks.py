# -*- coding: utf-8 -*-
"""Round-2 quick checks (ECOLIND-62293R1). Reads round-1 outputs read only; writes to round-2 analysis/outputs.

A1  False discovery rate (Benjamini-Hochberg, Benjamini-Yekutieli) on Hamed-Rao corrected trend tests; recovery
    status under five specifications; agreement of individual labels (R3 comments 1, 4, 6, 8).
A5  k = 3 versus k = 5 trajectory solutions: crosswalk summary and separation of the two declining classes (R3-5).
A6  Lingbei footprint check: bootstrap intervals for group medians; mapped footprint area as a share of the ROI5
    cells in the mining area rectangle (R3-3, R3-6).
A7  Sign of the locally dominant land cover contribution versus the direction of land cover change (R3-6).
A9  Systematic offset between the mixed Landsat record and the Landsat 7 record, 2000-2012 versus 2013-2023 (R3-3).
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from sklearn.metrics import silhouette_samples

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
O1 = R1 / "outputs"
R2 = Path(str(P.ROUND2))
OUT = R2 / "outputs" / "quick_checks"
OUT.mkdir(parents=True, exist_ok=True)
VEC = Path(str(P.VECTORS))
summary = {}

# ---------------- A1 ----------------
m = pd.read_csv(O1 / "f_annual/modified_mk_cells.csv.gz")
z = np.load(O1 / "f_annual/annual_roi5.npz", allow_pickle=True)
yrs = list(z["year"])
reg = pd.DataFrame({"grid_id": z["grid_id"], "r2025": z["regain"][:, yrs.index(2025)],
                    "r2325": z["regain"][:, [yrs.index(y) for y in (2023, 2024, 2025)]].mean(1)})
d = m.merge(reg, on="grid_id", how="left", validate="one_to_one")
p = d.modified_p.to_numpy(float)
s = d.sen_slope.to_numpy(float)
ok = np.isfinite(p)


def bh(pv):
    n = len(pv)
    o = np.argsort(pv)
    q = np.empty(n)
    q[o] = np.minimum(np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1], 1)
    return q


q = np.full(len(p), np.nan)
q[ok] = bh(p[ok])
cm = np.sum(1 / np.arange(1, ok.sum() + 1))
qby = np.full(len(p), np.nan)
qby[ok] = np.minimum(q[ok] * cm, 1)
N = len(d)
rows = []
for name, val, a in [("Hamed-Rao, per-cell p<0.05", p, 0.05), ("Benjamini-Hochberg q<0.05", q, 0.05),
                     ("Benjamini-Hochberg q<0.10", q, 0.10), ("Benjamini-Yekutieli q<0.05", qby, 0.05)]:
    sig = np.isfinite(val) & (val < a)
    rows.append({"criterion": name, "significant_improvement_pct": 100 * np.sum(sig & (s > 0)) / N,
                 "significant_decline_pct": 100 * np.sum(sig & (s < 0)) / N,
                 "no_significant_trend_pct": 100 * np.sum(np.isfinite(val) & ~sig) / N,
                 "unevaluable_pct": 100 * np.sum(~np.isfinite(val)) / N})
fdr = pd.DataFrame(rows)
fdr.to_csv(OUT / "A1_fdr_trend_shares.csv", index=False)
# largest per-cell p accepted by BH at q<0.05
summary["A1_bh_threshold_p"] = float(np.max(p[ok][q[ok] < 0.05]))


def status(pos, sigp, alpha=0.05, thr=0.75):
    lab = np.full(len(pos), "unevaluable", dtype=object)
    ev = np.isfinite(sigp)
    sig = ev & (sigp < alpha)
    dec, inc, hi = sig & (s < 0), sig & (s > 0), pos >= thr
    lab[ev & dec] = "significant_decline"
    lab[ev & inc & hi] = "consolidated_recovery"
    lab[ev & inc & ~hi] = "advancing_recovery"
    lab[ev & ~dec & ~inc & hi] = "high_position_monitoring"
    lab[ev & ~dec & ~inc & ~hi] = "stalled_recovery"
    return lab


V = {"primary (2025, 0.75, per-cell p)": status(d.r2025.values, p),
     "FDR q<0.05": status(d.r2025.values, q),
     "2023-2025 mean endpoint": status(d.r2325.values, p),
     "threshold 0.70": status(d.r2025.values, p, thr=0.70),
     "threshold 0.80": status(d.r2025.values, p, thr=0.80)}
pri = V["primary (2025, 0.75, per-cell p)"]
internal = {"consolidated_recovery": "consolidated_recovery", "advancing_recovery": "advancing_recovery",
            "high_position_monitoring": "high_state_monitoring", "stalled_recovery": "stalled_recovery",
            "significant_decline": "renewed_decline", "unevaluable": "unevaluable_variance"}
summary["A1_reproduction_fraction"] = float(np.mean(pd.Series(pri).map(internal).values == d.modified_status.values))
assert summary["A1_reproduction_fraction"] > 0.9999, "primary status does not reproduce the published classes"
classes = ["consolidated_recovery", "advancing_recovery", "high_position_monitoring", "stalled_recovery",
           "significant_decline", "unevaluable"]
sh = []
for k, v in V.items():
    vc = pd.Series(v).value_counts()
    r = {"specification": k, **{c: 100 * vc.get(c, 0) / N for c in classes}}
    r["stalled_plus_decline"] = r["stalled_recovery"] + r["significant_decline"]
    r["labels_changed_vs_primary_pct"] = 100 * np.mean(v != pri)
    sh.append(r)
pd.DataFrame(sh).to_csv(OUT / "A1_status_specifications.csv", index=False)
agree = np.sum(np.vstack([V[k] == pri for k in V]), axis=0)  # 1..5 (primary always agrees with itself)
stab = pd.DataFrame({"grid_id": d.grid_id, "primary": pri, "n_specifications_agreeing": agree})
stab.to_csv(OUT / "A1_label_agreement_by_cell.csv.gz", index=False)
by_class = stab.groupby("primary").n_specifications_agreeing.apply(lambda x: 100 * np.mean(x == 5)).reindex(classes)
summary["A1"] = {"fdr": fdr.to_dict("records"), "identical_in_all_five_pct": float(100 * np.mean(agree == 5)),
                 "identical_by_primary_class_pct": by_class.round(2).to_dict(),
                 "stalled_plus_decline_range": [float(min(r["stalled_plus_decline"] for r in sh)),
                                                float(max(r["stalled_plus_decline"] for r in sh))]}
print("A1", json.dumps(summary["A1"], indent=1))

# ---------------- A5 ----------------
ka = pd.read_csv(O1 / "f_trajectory/k3_k5_assignments.csv.gz")
print("A5 columns", list(ka.columns))
cw = pd.read_csv(O1 / "f_trajectory/k3_k5_crosswalk_counts.csv")
summary["A5_crosswalk_columns"] = list(cw.columns)
k3c = [c for c in ka.columns if "k3" in c.lower()][0]
k5c = [c for c in ka.columns if "k5" in c.lower()][0]
reg_all = z["regain"]
gid = pd.Series(np.arange(len(z["grid_id"])), index=z["grid_id"])
ka = ka[ka.grid_id.isin(gid.index)].copy()
X = reg_all[gid[ka.grid_id].to_numpy()]
Xs = np.copy(X)
Xs[:, 1:-1] = (X[:, :-2] + X[:, 1:-1] + X[:, 2:]) / 3  # centered three-year mean, ends kept
ct = pd.crosstab(ka[k3c], ka[k5c])
ct.to_csv(OUT / "A5_k3_k5_crosstab.csv")
rng = np.random.default_rng(20261021)
idx = rng.choice(len(ka), size=min(12000, len(ka)), replace=False)
sil5 = silhouette_samples(Xs[idx], ka[k5c].to_numpy()[idx])
sil_by_k5 = pd.Series(sil5).groupby(ka[k5c].to_numpy()[idx]).mean()
med_curves = pd.DataFrame(Xs).groupby(ka[k5c].to_numpy()).median()
med_curves.columns = yrs
med_curves.to_csv(OUT / "A5_k5_median_curves.csv")
summary["A5"] = {"k3_column": k3c, "k5_column": k5c, "crosstab": ct.to_dict(),
                 "k5_mean_silhouette_by_class_sample12000": sil_by_k5.round(3).to_dict()}
print("A5", json.dumps(summary["A5"], indent=1, default=str))

# ---------------- A6 ----------------
f = pd.read_csv(O1 / "n_footprint_check/footprint_cells.csv.gz")
f["group"] = f["group"].fillna("none")
order = ["none", "0-10%", "10-25%", ">=25%"]
B = 4000
rng = np.random.default_rng(20261022)
ci = []
for g in order:
    sub = f[f.group == g]
    for col in ["eci2000", "eci2025", "eci_change", "regain_2025"]:
        v = sub[col].to_numpy(float)
        bs = np.median(rng.choice(v, size=(B, len(v)), replace=True), axis=1)
        ci.append({"group": g, "n": len(v), "variable": col, "median": float(np.median(v)),
                   "ci95_low": float(np.percentile(bs, 2.5)), "ci95_high": float(np.percentile(bs, 97.5))})
hi10 = f[f.group.isin(["10-25%", ">=25%"])]
no = f[f.group == "none"]
for col in ["eci2025", "eci_change", "regain_2025"]:
    a_, b_ = hi10[col].to_numpy(float), no[col].to_numpy(float)
    bs = np.median(rng.choice(a_, (B, len(a_))), 1) - np.median(rng.choice(b_, (B, len(b_))), 1)
    ci.append({"group": ">=10% minus none", "n": len(a_), "variable": col, "median": float(np.median(a_) - np.median(b_)),
               "ci95_low": float(np.percentile(bs, 2.5)), "ci95_high": float(np.percentile(bs, 97.5))})
pd.DataFrame(ci).to_csv(OUT / "A6_footprint_group_medians_ci.csv", index=False)
cells = pd.read_csv(O1 / "m_body/revised_cell_products.csv.gz", usecols=["grid_id", "centroid_x_utm50", "centroid_y_utm50"])
cells = cells[cells.grid_id.isin(f.grid_id)]
sq = unary_union([box(x - 250, y - 250, x + 250, y + 250) for x, y in zip(cells.centroid_x_utm50, cells.centroid_y_utm50)])
fp = gpd.read_file(VEC / "Dingnan_mining_sites_SAM2025.shp").to_crs(32650)
fp_union = unary_union(fp.geometry.buffer(0))
summary["A6"] = {"cells": int(len(cells)), "cell_area_km2": sq.area / 1e6, "footprint_polygons": int(len(fp)),
                 "footprint_area_km2": fp_union.area / 1e6, "footprint_area_in_cells_km2": fp_union.intersection(sq).area / 1e6,
                 "footprint_share_of_cells_pct": 100 * fp_union.intersection(sq).area / sq.area}
print("A6", json.dumps(summary["A6"], indent=1))

# ---------------- A7 ----------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import read_inputs  # noqa: E402
data, oof, _, features, shapz = read_inputs()
fam = pd.read_csv(O1 / "bcd_models/shap_rank_comparison.csv")
fam = fam[fam.configuration == "baseline"].set_index("feature")["family"].to_dict()
S = shapz["shap_values"]
groups = sorted(set(fam.values()))
absg = np.column_stack([np.abs(S[:, [fam[f] == g for f in features]]).sum(1) for g in groups])
sgnd = np.column_stack([S[:, [fam[f] == g for f in features]].sum(1) for g in groups])
dom = np.array(groups)[absg.argmax(1)]
dom_sign = np.sign(sgnd[np.arange(len(S)), absg.argmax(1)])
lc = dom == "Land cover"
tab = pd.DataFrame({"dominant": dom, "sign": dom_sign, "for_s": data["forest_frac_slope_2000_2025"],
                    "crp_s": data["cropland_frac_slope_2000_2025"], "blt_s": data["built_frac_slope_2000_2025"],
                    "y": data["regain_sen_slope_2000_2025"]})
t = tab[lc]
res7 = {"land_cover_dominant_pct": float(100 * lc.mean()), "positive_share_among_lc_dominant_pct": float(100 * np.mean(t.sign > 0)),
        "positive_pct_when_forest_gain": float(100 * np.mean(t[t.for_s > 0].sign > 0)),
        "positive_pct_when_forest_loss": float(100 * np.mean(t[t.for_s <= 0].sign > 0)),
        "share_lc_dominant_with_forest_gain_pct": float(100 * np.mean(t.for_s > 0)),
        "negative_cells_with_cropland_or_builtup_gain_pct": float(100 * np.mean(((t.crp_s > 0) | (t.blt_s > 0))[t.sign < 0])),
        "positive_cells_with_cropland_or_builtup_gain_pct": float(100 * np.mean(((t.crp_s > 0) | (t.blt_s > 0))[t.sign > 0])),
        "median_response_positive": float(t[t.sign > 0].y.median()), "median_response_negative": float(t[t.sign < 0].y.median()),
        "dominant_group_reproduces_published": float(np.mean(dom == pd.read_csv(O1 / "m_body/revised_cell_products.csv.gz", usecols=["grid_id", "dominant_family"]).set_index("grid_id").loc[oof.grid_id, "dominant_family"].to_numpy()))}
summary["A7"] = res7
print("A7", json.dumps(res7, indent=1))

# ---------------- A9 ----------------
l7 = np.load(O1 / "g_sensor/annual_L7_grid_eci.npz", allow_pickle=True)
assert list(l7["grid_id"]) == list(z["grid_id"])
yl = list(l7["year"])
diff = []
for yv in range(2000, 2024):
    mix = z["eci"][:, yrs.index(yv)]
    sev = l7["eci"][:, yl.index(yv)]
    okc = np.isfinite(mix) & np.isfinite(sev)
    dd = mix[okc] - sev[okc]
    diff.append({"year": yv, "n": int(okc.sum()), "median_mixed": float(np.median(mix[okc])), "median_L7": float(np.median(sev[okc])),
                 "median_difference_mixed_minus_L7": float(np.median(dd)), "iqr_low": float(np.percentile(dd, 25)),
                 "iqr_high": float(np.percentile(dd, 75))})
diff = pd.DataFrame(diff)
diff.to_csv(OUT / "A9_mixed_vs_L7_annual_offset.csv", index=False)
pre = diff[(diff.year <= 2012)].median_difference_mixed_minus_L7
post = diff[(diff.year >= 2013)].median_difference_mixed_minus_L7
summary["A9"] = {"median_offset_2000_2012": float(pre.median()), "median_offset_2013_2023": float(post.median()),
                 "mean_offset_2000_2012": float(pre.mean()), "mean_offset_2013_2023": float(post.mean()),
                 "shift_after_2013": float(post.median() - pre.median()),
                 "regional_median_eci_change_2000_2025": 0.716 - 0.560}
print("A9", json.dumps(summary["A9"], indent=1))
(OUT / "quick_checks_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
