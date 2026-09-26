# Revision result tables

These tables support the analyses added or updated in the revised manuscript.
All values are aggregate summaries, model diagnostics or sensitivity results.
Cell identifiers, coordinates and spatial block identifiers are not included.
The Fig. 5 summaries for the four focal landscapes use the landscape names
shown in the manuscript; county-level tables are not included.

| Folder | Contents |
|---|---|
| `spatial_validation/` | Pooled and fold-level performance under random, 10 km block and 20 km block cross validation; distances to the nearest training cell; residual correlation by distance; group ablation under matched partitions; feature and group SHAP importance and locally dominant groups from the 10 km spatial test folds |
| `model_sensitivity/` | XGBoost parameter variants, the forest cover trend removal test, the comparison of XGBoost and LightGBM SHAP rankings, and the 10 km block refits with all predictors, without the 17 Landsat-derived predictors and with those predictors alone |
| `pls_sem/` | Reproduced PLS-SEM paths and outer weights, whole 10 km block bootstrap intervals, construct correlations and structural VIFs |
| `pls_sem_fit/` | Independent cSEM reproduction, SRMR, indicator correlation matrices and residual correlations between groups |
| `annual_records_and_trends/` | Annual component and ECI summaries, component clipping, component correlations, the ECI range distribution, and original versus Hamed-Rao corrected Mann-Kendall classes |
| `trajectory_classes/` | Three-class and five-class trajectory centers and their crosswalk |
| `trajectory_sensitivity/` | Class shares and agreement when single years are interpolated, with year-by-year crosswalks; class shares under the main record, two interpolated records, the mixed-sensor record and the Landsat 7 record for 2000-2023, with a summary of label and share changes |
| `sensor_checks/` | Annual clear-observation counts, Landsat 7 compared with the mixed-sensor record, and the scene inventory |
| `index_components/` | Principal component diagnostics and the comparison of 2nd-98th and 1st-99th percentile scaling anchors |
| `adjacent_landscape_comparison/` | Stratified comparison of ROI5 with the adjacent background band (BG15) |
| `climate_context/` | Annual regional climate anomalies from ERA5-Land |
| `endpoint_sensitivity/` | Monitoring class shares and transitions when the 2025 endpoint is replaced by the 2023-2025 mean, and class shares for high position thresholds from 0.65 to 0.85 |
| `landscape_status_and_distance/` | ECI by trajectory and status class, locally dominant groups by status class, and cumulative distance neighborhoods |
| `empirical_boundaries/` | Point estimates, block bootstrap turning points, pointwise curve intervals and bin number sensitivity |
| `figure_and_table_sources/` | Source values for PLS-SEM effects and weights, status summaries, response surface contrasts, and panels of Figs. 5, 11 and 13 |
| `land_cover_closure/` | Slopes and correlations of the other six land cover trends on forest cover trend, and checks that the seven land cover fractions sum to one |
| `partial_dependence_checks/` | Observed support and retained range of the six two-predictor partial dependence surfaces, and the four surfaces with forest cover trend recomputed along the average joint change of the other six land cover trends |
| `footprint_check/` | Median ECI, REGAIN and status class shares by the share of each cell covered by mining footprints mapped on 2025 imagery, in three comparison areas of the Lingbei mining area, with Mann-Whitney comparisons |

Notes on the tables added for the final revision:

- In the status tables, `high_state_monitoring` and `renewed_decline` correspond
  to the high position monitoring and significant decline classes of the
  manuscript. High position means REGAIN in the final year of at least the
  stated threshold; 0.75 is the threshold used in the manuscript.
- In `trajectory_sensitivity/class_shares_by_record.csv`, the classes of each
  alternative record are matched to the main classes by maximum overlap.
- In `model_sensitivity/landsat_derived_predictor_refit.csv`, the pooled RMSE
  equals the square root of one minus the pooled R2, multiplied by the
  population standard deviation of the response. The 17 predictors are listed
  in `landsat_derived_predictors.csv`.
- In `land_cover_closure/closure_summary.json`, the cropland offset ratio is
  the negative sum of cropland trends divided by the sum of forest trends over
  cells in which forest cover increased.
- `partial_dependence_checks/closure_surfaces.csv` uses the grid of
  `figure_and_table_sources/Fig11/response_surface_values.csv`, whose values it
  reproduces in the `predicted_original` column. Observed support was checked
  for the two focal predictors of each surface.
- The Mann-Whitney tests in `footprint_check/footprint_tests.csv` treat cells as
  independent and serve as descriptive comparisons.

Out-of-fold predictions for all four tree models under the three
cross-validation designs are provided separately in
`data/results/oof_predictions_deidentified.csv.gz`, and the fold assignments
used for those designs are in `data/cv_fold_assignments.csv.gz`.
