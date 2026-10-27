# -*- coding: utf-8 -*-
"""Round-2 checks on spatial validation and attribution stability (ECOLIND-62293R1, Reviewer 1 comment 2,
Reviewer 3 comments 4 and 5).

1. Reproduce the primary XGBoost 10 km block model (pooled R2 and TreeSHAP group shares).
2. Buffered block cross validation: drop training cells within 2 km or 5 km of any test cell.
3. Distance correlogram of the response, with the same pair sample and formula as the residual correlogram.
4. Repeated block cross validation: 10 random assignments of the 237 blocks to five balanced folds, with
   pooled R2, TreeSHAP group shares and predictor ranks for each repeat.
5. TreeSHAP group shares of the model refitted without the 17 Landsat-derived predictors.

Reads the frozen round-1 inputs through work_20260923/analysis/scripts/revision_common.py (read only) and
writes only to the round-2 folder analysis/outputs/A23_spatial_shap/.
"""
import _paths as P  # internal input folders, set by environment variables (README)
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree

R1 = Path(str(P.ROUND1))
R2 = Path(str(P.ROUND2))
OUT = R2 / "outputs" / "A23_spatial_shap"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_common import read_inputs, Y  # noqa: E402

PARAMS = json.loads((R1 / "config" / "bcd_model_protocol.json").read_text(encoding="utf-8"))["baseline"]
N_REPEATS = 10
BUFFERS_KM = [2.0, 5.0]


def load():
    data, oof, _, features, shap = read_inputs()
    fam = pd.read_csv(R1 / "outputs" / "bcd_models" / "shap_rank_comparison.csv")
    fam = fam[fam.configuration == "baseline"].set_index("feature")["family"].to_dict()
    X = data[features].to_numpy(dtype=float)
    y = data[Y].to_numpy(dtype=float)
    xy = oof[["centroid_x_utm50", "centroid_y_utm50"]].to_numpy(dtype=float)
    blocks = oof["spatial_block_10km"].to_numpy()
    folds = oof["fold_spatial_block_10km"].to_numpy()
    return data, oof, features, shap, fam, X, y, xy, blocks, folds


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_fold(X, y, tr, te, want_shap):
    m = xgb.XGBRegressor(**PARAMS)
    m.fit(X[tr], y[tr])
    pred = m.predict(X[te])
    contrib = None
    if want_shap:
        contrib = m.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
    return pred, contrib


def cross_validate(X, y, folds, want_shap=True, train_mask_fn=None):
    pred = np.full(len(y), np.nan)
    contrib = np.full(X.shape, np.nan) if want_shap else None
    removed = []
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        if train_mask_fn is not None:
            keep = train_mask_fn(te, tr)
            removed.append(int(tr.sum() - keep.sum()))
            tr = keep
        p, c = fit_fold(X, y, tr, te, want_shap)
        pred[te] = p
        if want_shap:
            contrib[te] = c
    return pred, contrib, removed


def shares(contrib, features, fam):
    imp = np.mean(np.abs(contrib), axis=0)
    df = pd.DataFrame({"feature": features, "mean_abs_shap": imp, "family": [fam[f] for f in features]})
    df["share"] = df.mean_abs_shap / df.mean_abs_shap.sum()
    df["rank"] = df.mean_abs_shap.rank(ascending=False, method="first").astype(int)
    grp = df.groupby("family").mean_abs_shap.sum()
    return df, (grp / grp.sum()).to_dict()


def balanced_folds(blocks, seed, n_folds=5):
    ids, counts = np.unique(blocks, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    load = np.zeros(n_folds)
    assign = {}
    for i in order:
        k = int(np.argmin(load))
        assign[ids[i]] = k + 1
        load[k] += counts[i]
    return np.array([assign[b] for b in blocks])


def one_repeat(seed):
    _, _, features, _, fam, X, y, _, blocks, _ = load()
    f = balanced_folds(blocks, seed)
    pred, contrib, _ = cross_validate(X, y, f, want_shap=True)
    df, grp = shares(contrib, features, fam)
    df.insert(0, "repeat_seed", seed)
    sizes = np.bincount(f)[1:].tolist()
    return seed, r2(y, pred), grp, df, sizes


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data, oof, features, shap, fam, X, y, xy, blocks, folds = load()
    summary = {"params": PARAMS, "n_cells": int(len(y)), "n_blocks": int(len(np.unique(blocks)))}

    # 1. reproduce the primary model
    pred0, contrib0, _ = cross_validate(X, y, folds, want_shap=True)
    df0, grp0 = shares(contrib0, features, fam)
    stored = shap["shap_values"]
    _, grp_stored = shares(stored, features, fam)
    summary["reproduction"] = {
        "pooled_r2_recomputed": r2(y, pred0),
        "pooled_r2_stored_oof": r2(y, oof["pred_xgboost_spatial_block_10km"].to_numpy()),
        "max_abs_pred_diff_vs_stored": float(np.max(np.abs(pred0 - shap["prediction"]))),
        "shap_corr_vs_stored": float(np.corrcoef(contrib0.ravel(), stored.ravel())[0, 1]),
        "group_shares_recomputed": grp0,
        "group_shares_stored": grp_stored,
    }
    df0.to_csv(OUT / "baseline_feature_importance_recomputed.csv", index=False)
    print("reproduction", json.dumps(summary["reproduction"], indent=1), flush=True)

    # 2. buffered block cross validation
    rows = [{"buffer_km": 0.0, "pooled_r2": r2(y, pred0), "median_training_cells_removed": 0}]
    for b in BUFFERS_KM:
        def mask(te, tr, b=b):
            d, _ = cKDTree(xy[te]).query(xy[tr])
            keep = tr.copy()
            keep[np.where(tr)[0][d < b * 1000]] = False
            return keep
        pb, _, removed = cross_validate(X, y, folds, want_shap=False, train_mask_fn=mask)
        rows.append({"buffer_km": b, "pooled_r2": r2(y, pb), "median_training_cells_removed": float(np.median(removed)),
                     "training_cells_removed_by_fold": removed})
        print("buffer", b, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "buffered_block_cv.csv", index=False)
    summary["buffered_block_cv"] = rows

    # 3. distance correlogram of the response and of the primary residuals (same pairs, same formula)
    pairs = np.load(R1 / "outputs" / "a_spatial" / "residual_pair_sample.npz")
    a, bb, dist = pairs["first"], pairs["second"], pairs["distance_km"]
    edges = [0, 1, 2, 5, 10, 20, 40]
    zy = (y - y.mean()) / y.std()
    res = y - oof["pred_xgboost_spatial_block_10km"].to_numpy()
    zr = (res - res.mean()) / res.std()
    corr = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ix = (dist >= lo) & (dist < hi)
        corr.append({"distance_low_km": lo, "distance_high_km": hi, "pair_draws": int(ix.sum()),
                     "response_correlation": float(np.mean(zy[a[ix]] * zy[bb[ix]])),
                     "residual_correlation_xgb_10km": float(np.mean(zr[a[ix]] * zr[bb[ix]]))})
    pd.DataFrame(corr).to_csv(OUT / "distance_correlogram_response_residual.csv", index=False)
    summary["correlogram"] = corr
    print("correlogram done", flush=True)

    # 5. model without the 17 Landsat-derived predictors
    shared = json.loads((R1 / "outputs" / "o_review_round3" / "shared_source_check.json").read_text(encoding="utf-8"))
    nl = [f for f in features if f not in shared["landsat_predictors"]]
    Xn = data[nl].to_numpy(dtype=float)
    pn, cn, _ = cross_validate(Xn, y, folds, want_shap=True)
    dfn, grpn = shares(cn, nl, fam)
    dfn.to_csv(OUT / "nonlandsat_feature_importance.csv", index=False)
    summary["non_landsat"] = {"n_predictors": len(nl), "pooled_r2": r2(y, pn), "group_shares": grpn,
                              "top10": dfn.sort_values("rank").head(10)[["feature", "family", "share"]].to_dict("records")}
    print("non-landsat", summary["non_landsat"]["pooled_r2"], grpn, flush=True)
    (OUT / "summary_partial.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    # 4. repeated block cross validation
    reps, feats = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for seed, rr, grp, df, sizes in ex.map(one_repeat, range(1, N_REPEATS + 1)):
            reps.append({"repeat_seed": seed, "pooled_r2": rr, "fold_sizes": sizes, **{"share_" + k: v for k, v in grp.items()}})
            feats.append(df)
            print("repeat", seed, round(rr, 4), {k: round(v, 3) for k, v in grp.items()}, flush=True)
    rep = pd.DataFrame(reps)
    rep.to_csv(OUT / "repeated_cv_summary.csv", index=False)
    fe = pd.concat(feats)
    fe.to_csv(OUT / "repeated_cv_feature_importance.csv", index=False)
    stat = {}
    for c in [c for c in rep.columns if c.startswith("share_")] + ["pooled_r2"]:
        stat[c] = {"mean": float(rep[c].mean()), "sd": float(rep[c].std(ddof=1)), "min": float(rep[c].min()), "max": float(rep[c].max())}
    top = df0.sort_values("rank").head(12).feature.tolist()
    ranks = fe[fe.feature.isin(top)].groupby("feature")["rank"].agg(["median", "min", "max"]).reindex(top)
    ranks.to_csv(OUT / "repeated_cv_top12_rank_range.csv")
    summary["repeated_cv"] = {"n_repeats": N_REPEATS, "stats": stat,
                              "forest_trend_rank1_count": int((fe[(fe.feature == "forest_frac_slope_2000_2025")]["rank"] == 1).sum()),
                              "top12_rank_range": ranks.reset_index().to_dict("records")}
    summary["elapsed_min"] = (time.time() - t0) / 60
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print("done in %.1f min" % summary["elapsed_min"], flush=True)


if __name__ == "__main__":
    main()
