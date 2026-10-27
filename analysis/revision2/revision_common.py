"""Frozen source locations and explicitly checked one-to-one analytical joins."""
import _paths as P  # internal input folders, set by environment variables (README)
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=P.ROUND1
CFG=json.loads((ROOT/'config/f_annual_protocol.json').read_text(encoding='utf-8'))
PACKAGE=Path(CFG['package_root']);RELEASE=Path(CFG['release_root'])
BACKUP=PACKAGE/'_build/analysis_backups/20260719_before_random_only_rebuild'
FIG=PACKAGE/'_build/figure_sources'
Y='regain_sen_slope_2000_2025'

def read_inputs():
    matrix=pd.read_csv(RELEASE/'data/model_matrix_full_deidentified.csv.gz')
    oof=pd.read_csv(BACKUP/'experiment_tables/model_validation_oof_predictions.csv.gz')
    ledger=pd.read_csv(RELEASE/'data/predictor_ledger.csv')
    shap=np.load(BACKUP/'experiment_models/xgboost_spatial10_oof_shap_values.npz',allow_pickle=True)
    features=shap['features'].tolist()
    assert len(matrix)==len(oof)==38273 and len(features)==57
    assert np.max(np.abs(matrix[Y].to_numpy()-oof[Y].to_numpy()))<1e-12
    assert np.max(np.abs(matrix[features].to_numpy(dtype=np.float32)-shap['feature_values']))==0
    assert np.array_equal(oof.fold_spatial_block_10km.to_numpy(),shap['fold_id'])
    data=matrix.copy();data.insert(0,'grid_id',oof.grid_id.to_numpy())
    assert not data[features+[Y]].isna().any().any()
    return data,oof,ledger,features,shap

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda:stream.read(8*1024*1024),b''):h.update(piece)
    return h.hexdigest()
