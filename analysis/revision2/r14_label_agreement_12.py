# -*- coding: utf-8 -*-
"""Round 14 (2026-10-26), for Reviewer 3, comment 6 ("Report individual label uncertainty or mark this uncertainty in
Figure 13"): agreement of each cell's recovery status across the 12 combinations of endpoint (2025; 2023-2025 mean),
high position threshold (0.70, 0.75, 0.80) and trend criterion (per-cell Hamed-Rao p < 0.05; Benjamini-Hochberg
q < 0.05), recomputed exactly as J1 of r2_joint_status_mdi_k.py, which kept only regional summaries.

For every analysed cell (the 38,273 cells of Fig. 13) it writes the primary label, the number of the 12 combinations
that give the primary label and the number in which the cell is stalled or declining, and it checks the regional
values reported in Section 4.7 and Table S13 (63.5% of cells keep their label in all 12 combinations; 84% of the
cells stalled or declining under the primary specification remain so in all 12).
Reads round-1 outputs read only; writes analysis/outputs/quick_checks/S4_label_agreement_by_cell.csv.gz and
S4_label_agreement_summary.json.

    python -I r14_label_agreement_12.py
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R1 = Path(str(P.ROUND1))
O1 = R1 / "outputs"
OUT = Path(str(P.ROUND2 / "outputs" / "quick_checks"))

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


labs, names = [], []
for end_name, pos in [("2025", d.r2025.values), ("2023-2025 mean", d.r2325.values)]:
    for thr in (0.70, 0.75, 0.80):
        for crit_name, sp in [("per-cell p<0.05", p), ("BH q<0.05", q)]:
            labs.append(status(pos, sp, thr))
            names.append((end_name, thr, crit_name))
L = np.vstack(labs)
primary = status(d.r2025.values, p, 0.75)
assert names.index(("2025", 0.75, "per-cell p<0.05")) == 2 and np.array_equal(L[2], primary)
SD = ["stalled_recovery", "significant_decline"]
cell = pd.DataFrame({"grid_id": d.grid_id.to_numpy(), "primary_label": primary,
                     "n_combinations_same_label": (L == primary).sum(0),
                     "n_combinations_stalled_or_declining": np.isin(L, SD).sum(0)})

# the 38,273 analysed cells of Fig. 13 (cells with a zero ECI range have no REGAIN and were excluded)
CELLS = R1 / "outputs" / "m_body" / "revised_cell_products.csv.gz"
mc = pd.read_csv(CELLS, usecols=["grid_id", "management_context"])
assert len(mc) == 38273
cell = mc.merge(cell, on="grid_id", how="left", validate="one_to_one")
assert cell.primary_label.notna().all()
same = cell.primary_label.replace({"unevaluable": "unevaluable_variance"}) == cell.management_context
print("cells in J1:", len(d), " analysed cells:", len(cell), " primary label equals Fig. 13 class:", int(same.sum()))
assert same.all()


def pct(x):
    return float(100 * np.mean(x))


sd_primary = cell.primary_label.isin(SD)
summary = {
    "n_cells_J1": int(len(d)), "n_cells_analysed": int(len(cell)),
    "same_label_all_12_pct_J1_cells": pct((L == primary).all(0)),
    "same_label_all_12_pct": pct(cell.n_combinations_same_label == 12),
    "stalled_or_declining_all_12_pct": pct(cell.n_combinations_stalled_or_declining == 12),
    "stalled_or_declining_any_pct": pct(cell.n_combinations_stalled_or_declining >= 1),
    "primary_stalled_or_declining_pct": pct(sd_primary),
    "primary_stalled_or_declining_kept_all_12_pct": pct(cell.n_combinations_stalled_or_declining[sd_primary] == 12),
    "stalled_or_declining_some_not_primary_pct": pct((cell.n_combinations_stalled_or_declining >= 1) & ~sd_primary),
    "primary_stalled_or_declining_not_all_12_pct": pct(sd_primary & (cell.n_combinations_stalled_or_declining < 12)),
    "distribution_same_label": cell.n_combinations_same_label.value_counts().sort_index().to_dict(),
    "distribution_stalled_or_declining": cell.n_combinations_stalled_or_declining.value_counts().sort_index().to_dict(),
    "same_label_all_12_by_primary_class_pct": cell.groupby("primary_label").n_combinations_same_label.apply(
        lambda v: float(100 * np.mean(v == 12))).to_dict(),
}
J = json.loads((OUT / "J_summary.json").read_text(encoding="utf-8"))["J1"]
print(json.dumps(summary, indent=1, default=str))
print("J_summary:", json.dumps(J, indent=1))
assert abs(summary["same_label_all_12_pct_J1_cells"] - J["labels_identical_all_12_pct"]) < 1e-9
assert round(summary["same_label_all_12_pct"], 1) == 63.5
assert round(summary["primary_stalled_or_declining_kept_all_12_pct"]) == 84
cell.to_csv(OUT / "S4_label_agreement_by_cell.csv.gz", index=False)
(OUT / "S4_label_agreement_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print("written", OUT / "S4_label_agreement_by_cell.csv.gz")
