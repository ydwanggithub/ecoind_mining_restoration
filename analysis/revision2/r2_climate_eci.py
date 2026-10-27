# -*- coding: utf-8 -*-
"""Round-2 check (ECOLIND-62293R1, Reviewer 3 comments 7 and 8): statistical link between annual climate anomalies
and changes in ecological condition, in ROI5 and in the surrounding band (BG15, cells shared with ROI5 removed).

Interannual: domain medians of annual ECI and domain means of ERA5-Land precipitation, mean temperature and climatic
water deficit (cell means, as in Table S11; the first run used domain medians of the climate inputs, kept as
*_climate_median_v1 files)
(columns prcp_mm_mean, tmean_c_mean, def_mm_mean of the annual GEE exports); Pearson and Spearman correlations of the
year-to-year ECI change with concurrent and previous-year anomalies (n = 25), and of detrended ECI with anomalies.
2025: cell-level ECI change from 2024 to 2025 against the 2025 anomaly relative to each cell's 2000-2024 mean.
Writes only to round-2 analysis/outputs/A8_climate/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
IN = Path(str(P.MODEL_INPUTS))
O1 = Path(str(P.ROUND1 / "outputs"))
OUT = Path(str(P.ROUND2 / "outputs" / "A8_climate"))
OUT.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2000, 2026))
CLIM = ["prcp_mm_mean", "tmean_c_mean", "def_mm_mean"]

z = np.load(O1 / "f_annual/annual_all_domains.npz", allow_pickle=True)
assert list(z["year"]) == YEARS
eci = pd.DataFrame(z["eci"], columns=YEARS)
eci.insert(0, "grid_id", z["grid_id"])
eci.insert(1, "domain", z["domain"])

frames = []
for y in YEARS:
    d = pd.read_csv(IN / f"ecoind_gee_annual_{y}_merged_from_shards.csv", usecols=["grid_id", "row", "col", "domain"] + CLIM)
    d["year"] = y
    frames.append(d)
clim = pd.concat(frames, ignore_index=True)
cells = clim[clim.year == 2025][["grid_id", "row", "col", "domain"]]
roi_rc = set(map(tuple, cells[cells.domain == "ROI5"][["row", "col"]].to_numpy()))
bg = cells[cells.domain == "BG15"]
overlap = bg[[(r, c) in roi_rc for r, c in bg[["row", "col"]].to_numpy()]].grid_id
summary = {"bg15_cells_total": int(len(bg)), "bg15_cells_overlapping_roi5": int(len(overlap))}
keep = set(cells.grid_id) - set(overlap)
clim = clim[clim.grid_id.isin(keep)]
eci = eci[eci.grid_id.isin(keep)]
print(summary, flush=True)

# domain medians per year
med = []
for dom in ["ROI5", "BG15"]:
    e = eci[eci.domain == dom]
    c = clim[clim.domain == dom]
    for y in YEARS:
        cy = c[c.year == y]
        med.append({"domain": dom, "year": y, "eci_median": float(np.nanmedian(e[y])),
                    **{k: float(np.nanmean(cy[k])) for k in CLIM}})
med = pd.DataFrame(med)
rows = []
for dom in ["ROI5", "BG15"]:
    m = med[med.domain == dom].set_index("year")
    anom = m[CLIM] - m[CLIM].mean()
    de = m.eci_median.diff()
    t = np.array(YEARS, dtype=float)
    detr = m.eci_median - np.polyval(np.polyfit(t, m.eci_median, 1), t)
    tests = {}
    for k in CLIM:
        tests[f"dECI_vs_{k}_anom_t"] = (de.iloc[1:], anom[k].iloc[1:])
        tests[f"dECI_vs_{k}_anom_t-1"] = (de.iloc[1:], anom[k].shift(1).iloc[1:])
        tests[f"dECI_vs_d{k}_t"] = (de.iloc[1:], m[k].diff().iloc[1:])
        tests[f"detrendedECI_vs_{k}_anom_t"] = (detr, anom[k])
        tests[f"detrendedECI_vs_{k}_anom_t-1"] = (detr.iloc[1:], anom[k].shift(1).iloc[1:])
    for name, (a, b) in tests.items():
        a, b = np.asarray(a, float), np.asarray(b, float)
        okk = np.isfinite(a) & np.isfinite(b)
        pr, pp = stats.pearsonr(a[okk], b[okk])
        sr, sp = stats.spearmanr(a[okk], b[okk])
        rows.append({"domain": dom, "test": name, "n": int(okk.sum()), "pearson_r": pr, "pearson_p": pp,
                     "spearman_rho": sr, "spearman_p": sp})
corr = pd.DataFrame(rows)
med.to_csv(OUT / "domain_annual_medians.csv", index=False)
corr.to_csv(OUT / "interannual_correlations.csv", index=False)
print(corr[corr.test.str.startswith("dECI")].round(3).to_string(), flush=True)

# 2025 cell level
c = clim.pivot_table(index="grid_id", columns="year", values=CLIM)
xs = []
for dom in ["ROI5", "BG15"]:
    e = eci[eci.domain == dom].set_index("grid_id")
    dE = e[2025] - e[2024]
    cc = c.loc[e.index]
    an = {k: cc[(k, 2025)] - cc[k].loc[:, list(range(2000, 2025))].mean(axis=1) for k in CLIM}
    for k in CLIM:
        okk = np.isfinite(dE.values) & np.isfinite(an[k].values)
        sr, sp = stats.spearmanr(dE.values[okk], an[k].values[okk])
        qs = pd.qcut(an[k][okk], 5, labels=False, duplicates="drop")
        qmed = pd.Series(dE.values[okk]).groupby(qs.values).median().round(4).tolist()
        xs.append({"domain": dom, "anomaly": k, "n": int(okk.sum()), "spearman_rho": sr, "spearman_p": sp,
                   "median_dECI_by_anomaly_quintile_low_to_high": qmed,
                   "share_cells_eci_fell_pct": float(100 * np.mean(dE.values[okk] < 0)),
                   "median_anomaly_2025": float(np.median(an[k].values[okk]))})
xsd = pd.DataFrame(xs)
xsd.to_csv(OUT / "cells_2025_change_vs_anomaly.csv", index=False)
print(xsd.to_string(), flush=True)
summary["interannual"] = corr.round(4).to_dict("records")
summary["cells_2025"] = xsd.to_dict("records")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
