# Revision result tables

These tables support the analyses added or updated in the revised manuscript.
All values are aggregate summaries, model diagnostics or sensitivity results.
Cell identifiers, coordinates, spatial block identifiers and administrative
labels are not included.

| Folder | Contents |
|---|---|
| `spatial_validation/` | Pooled and fold-level performance under random, 10 km block and 20 km block cross validation; distances to the nearest training cell; residual correlation by distance; group ablation under matched partitions; feature and group SHAP importance and locally dominant groups from the 10 km spatial test folds |
| `model_sensitivity/` | XGBoost parameter variants, the forest cover trend removal test, and the comparison of XGBoost and LightGBM SHAP rankings |
| `pls_sem/` | Reproduced PLS-SEM paths and outer weights, whole 10 km block bootstrap intervals, construct correlations and structural VIFs |
| `pls_sem_fit/` | Independent cSEM reproduction, SRMR, indicator correlation matrices and residual correlations between groups |
| `annual_records_and_trends/` | Annual component and ECI summaries, component clipping, component correlations, the ECI range distribution, and original versus Hamed-Rao corrected Mann-Kendall classes |
| `trajectory_classes/` | Three-class and five-class trajectory centers and their crosswalk |
| `trajectory_sensitivity/` | Class shares and agreement when single years are interpolated, with year-by-year crosswalks |
| `sensor_checks/` | Annual clear-observation counts, Landsat 7 compared with the mixed-sensor record, and the scene inventory |
| `index_components/` | Principal component diagnostics and the comparison of 2nd-98th and 1st-99th percentile scaling anchors |
| `adjacent_landscape_comparison/` | Stratified comparison of the study landscape with the adjacent background domain (BG15) |
| `climate_context/` | Annual regional climate anomalies from ERA5-Land |
| `endpoint_sensitivity/` | Monitoring class shares and transitions when the 2025 endpoint is replaced by the 2023-2025 mean |
| `landscape_status_and_distance/` | ECI by trajectory and status class, locally dominant groups by status class, and cumulative distance neighborhoods |
| `empirical_boundaries/` | Point estimates, block bootstrap turning points, pointwise curve intervals and bin number sensitivity |
| `figure_and_table_sources/` | Source values for PLS-SEM effects and weights, status summaries, response surface contrasts, and figure panels without administrative labels |

Out-of-fold predictions for all four tree models under the three
cross-validation designs are provided separately in
`data/results/oof_predictions_deidentified.csv.gz`, and the fold assignments
used for those designs are in `data/cv_fold_assignments.csv.gz`.
