# -*- coding: utf-8 -*-
"""Round-2 checks (ECOLIND-62293R1): joint sensitivity of recovery status, bright bare strata, separation of the two
declining trajectory classes.

J1  Recovery status under all 12 combinations of endpoint (2025; 2023-2025 mean), high position threshold (0.70, 0.75,
    0.80) and trend criterion (per-cell Hamed-Rao p < 0.05; Benjamini-Hochberg q < 0.05) (R1-3, R3-6, R3-8).
J2  Region-wide strata of persistent bright bare detection frequency (none; quartiles of positive values): ECI 2000,
    ECI 2025, paired change, REGAIN 2025 and REGAIN slope, medians with interquartile ranges (R3-3, R1-4).
J3  Two-class silhouette of persistent decline versus recovery followed by late setback, on smoothed annual REGAIN,
    with the k = 3 crosswalk (R3-5).
Reads round-1 outputs read only; writes to round-2 analysis/outputs/quick_checks/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
O1 = R1 / "outputs"
OUT = Path(str(P.ROUND2 / "outputs" / "quick_checks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import read_inputs  # noqa: E402

summary = {}
# ---------------- J1 ----------------
m = pd.read_csv(O1 / "f_annual/modified_mk_cells.csv.gz")
z = np.load(O1 / "f_annual/annual_roi5.npz", allow_pickle=True)
yrs = list(z["year"])
reg = pd.DataFrame({"grid_id": z["grid_id"], "r2025": z["regain"][:, yrs.index(2025)],
                    "r2325": z["regain"][:, [yrs.index(y) for y in (2023, 2024, 2025)]].mean(1)})
d = m.merge(reg, on="grid_id", how="left", validate="one_to_one")
p, s = d.modified_p.to_numpy(float), d.sen_slope.to_numpy(float)
ok = np.isfinite(p)
q = np.full(len(p), np.nan)
o = np.argsort(p[ok])
n = ok.sum()
qq = np.empty(n)
qq[o] = np.minimum(np.minimum.accumulate((p[ok][o] * n / np.arange(1, n + 1))[::-1])[::-1], 1)
q[ok] = qq


def status(pos, sig_p, thr):
    lab = np.full(len(pos), "unevaluable", dtype=object)
    ev = np.isfinite(sig_p)
    sig = ev & (sig_p < 0.05)
    dec, inc, hi = sig & (s < 0), sig & (s > 0), pos >= thr
    lab[ev & dec] = "significant_decline"
    lab[ev & inc & hi] = "consolidated_recovery"
    lab[ev & inc & ~hi] = "advancing_recovery"
    lab[ev & ~dec & ~inc & hi] = "high_position_monitoring"
    lab[ev & ~dec & ~inc & ~hi] = "stalled_recovery"
    return lab


rows, labs = [], []
for end_name, pos in [("2025", d.r2025.values), ("2023-2025 mean", d.r2325.values)]:
    for thr in (0.70, 0.75, 0.80):
        for crit_name, sp in [("per-cell p<0.05", p), ("BH q<0.05", q)]:
            lab = status(pos, sp, thr)
            labs.append(lab)
            vc = pd.Series(lab).value_counts()
            r = {"endpoint": end_name, "threshold": thr, "trend_criterion": crit_name}
            for c in ["consolidated_recovery", "advancing_recovery", "high_position_monitoring", "stalled_recovery", "significant_decline", "unevaluable"]:
                r[c] = 100 * vc.get(c, 0) / len(lab)
            r["stalled_plus_decline"] = r["stalled_recovery"] + r["significant_decline"]
            rows.append(r)
J1 = pd.DataFrame(rows)
J1.to_csv(OUT / "J1_status_12_combinations.csv", index=False)
L = np.vstack(labs)
primary = status(d.r2025.values, p, 0.75)
same_all = np.all(L == primary, axis=0)
stalled_dec_any = np.isin(L, ["stalled_recovery", "significant_decline"])
summary["J1"] = {"stalled_plus_decline_min": float(J1.stalled_plus_decline.min()),
                 "stalled_plus_decline_max": float(J1.stalled_plus_decline.max()),
                 "labels_identical_all_12_pct": float(100 * same_all.mean()),
                 "cells_stalled_or_declining_in_all_12_pct": float(100 * stalled_dec_any.all(0).mean()),
                 "cells_stalled_or_declining_in_any_pct": float(100 * stalled_dec_any.any(0).mean()),
                 "primary_stalled_or_declining_kept_in_all_12_pct": float(100 * stalled_dec_any.all(0)[np.isin(primary, ["stalled_recovery", "significant_decline"])].mean())}
print("J1", json.dumps(summary["J1"], indent=1))

# ---------------- J2 ----------------
data, oof, _, _, _ = read_inputs()
c = pd.read_csv(O1 / "m_body/revised_cell_products.csv.gz", usecols=["grid_id", "eci_2000", "eci_2025", "regain_2025", "mdist_km"])
dd = data[["grid_id", "mining_disturbance_intensity_2000_2025", "regain_sen_slope_2000_2025"]].merge(c, on="grid_id", validate="one_to_one")
dd["eci_change"] = dd.eci_2025 - dd.eci_2000
mdi = dd.mining_disturbance_intensity_2000_2025
pos = dd[mdi > 0].copy()
pos["stratum"] = pd.qcut(pos.mining_disturbance_intensity_2000_2025, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
allg = pd.concat([dd[mdi == 0].assign(stratum="none"), pos])
rows = []
for g, sub in allg.groupby("stratum", sort=False):
    r = {"stratum": g, "n": len(sub), "detection_frequency_min": float(sub.mining_disturbance_intensity_2000_2025.min()),
         "detection_frequency_max": float(sub.mining_disturbance_intensity_2000_2025.max()),
         "median_distance_km": float(sub.mdist_km.median())}
    for col in ["eci_2000", "eci_2025", "eci_change", "regain_2025", "regain_sen_slope_2000_2025"]:
        v = sub[col].to_numpy(float)
        r[col + "_median"] = float(np.median(v))
        r[col + "_q25"] = float(np.percentile(v, 25))
        r[col + "_q75"] = float(np.percentile(v, 75))
    rows.append(r)
J2 = pd.DataFrame(rows).set_index("stratum").loc[["none", "Q1", "Q2", "Q3", "Q4"]].reset_index()
J2.to_csv(OUT / "J2_bright_bare_strata.csv", index=False)
summary["J2"] = {"share_cells_with_detection_pct": float(100 * np.mean(mdi > 0)),
                 "mean_annual_bright_bare_area_km2": float((mdi * 0.25).sum()),
                 "mean_annual_bright_bare_share_of_roi5_pct": float(100 * mdi.mean()),
                 "table": J2[["stratum", "n", "eci_2000_median", "eci_2025_median", "eci_change_median", "regain_2025_median",
                              "regain_sen_slope_2000_2025_median"]].round(3).to_dict("records")}
print("J2", json.dumps(summary["J2"], indent=1))

# ---------------- J3 ----------------
ka = pd.read_csv(O1 / "f_trajectory/k3_k5_assignments.csv.gz")
gid = pd.Series(np.arange(len(z["grid_id"])), index=z["grid_id"])
X = z["regain"][gid[ka.grid_id].to_numpy()]
Xs = X.copy()
Xs[:, 1:-1] = (X[:, :-2] + X[:, 1:-1] + X[:, 2:]) / 3
two = ka.submitted_class.isin(["persistent_decline", "temporary_recovery_with_late_setback"]).to_numpy()
lab2 = ka.submitted_class.to_numpy()[two]
sil2 = silhouette_score(Xs[two], lab2)
rng = np.random.default_rng(20261021)
idx = rng.choice(len(ka), 12000, replace=False)
sil5 = silhouette_score(Xs[idx], ka.k5_cluster_reproduced.to_numpy()[idx])
sil3 = silhouette_score(Xs[idx], ka.k3_cluster.to_numpy()[idx])
k3_decline = ka.k3_cluster.to_numpy() == 1
summary["J3"] = {"two_class_silhouette_persistent_vs_late_setback": float(sil2),
                 "overall_silhouette_k5_sample": float(sil5), "overall_silhouette_k3_sample": float(sil3),
                 "k3_declining_class_composition": ka[k3_decline].submitted_class.value_counts().to_dict(),
                 "share_of_late_setback_in_k3_declining_pct": float(100 * np.mean(k3_decline[ka.submitted_class == "temporary_recovery_with_late_setback"])),
                 "share_of_persistent_in_k3_declining_pct": float(100 * np.mean(k3_decline[ka.submitted_class == "persistent_decline"]))}
print("J3", json.dumps(summary["J3"], indent=1))
(OUT / "J_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
