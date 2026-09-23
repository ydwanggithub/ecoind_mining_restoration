# Aggregate results

These tables were exported from the full analysis used in the manuscript.
They contain no coordinates, internal grid identifiers, registered mineral
site locations, Earth Engine asset names, or account information.

- `annual_eci_regain_summary.csv` reports annual ECI and REGAIN distribution
  summaries.
- `model_validation_summary.csv` reports random fivefold validation metrics
  for the four tree ensembles, as in the original release. Metrics for the
  10 km and 20 km block designs used in the revision are in
  `revision/spatial_validation/`.
- `xgboost_shap_feature_importance.csv` and
  `xgboost_shap_family_importance.csv` summarize out of fold XGBoost
  attribution under random folds. The attribution reported in the revised
  manuscript comes from the 10 km spatial test folds and is provided in
  `revision/spatial_validation/xgboost_spatial10_shap_feature_importance.csv`
  and `revision/spatial_validation/spatial10_family_importance.csv`.
- `pls_sem_path_inference.csv` reports the ordinary cell bootstrap inference
  for the prespecified pathway model in the original release. The whole-block
  bootstrap intervals used in the revision are in `revision/pls_sem/`.
- `dual_constraint_summary.csv` reports the empirical lower and upper boundary
  fits.
- `predictor_family_summary.csv` records the candidate and retained predictor
  counts by group.
- `oof_predictions_deidentified.csv.gz` gives the observed response and the
  out-of-fold predictions of all four tree models under the random, 10 km
  block and 20 km block designs.
- `revision/` contains the tables for the analyses added in the revision.
