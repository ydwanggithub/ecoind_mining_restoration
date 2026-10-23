# Second revision result tables

These tables support the analyses added in the second revision of the
manuscript. All values are aggregate summaries, model diagnostics or
sensitivity results. Cell identifiers, coordinates and the original spatial
block identifiers are not included. Table and figure numbers refer to the
manuscript and its Supplementary Material.

| Folder | Contents | Manuscript |
|---|---|---|
| `climate_context/` | Annual median ECI with domain means of ERA5-Land precipitation, mean temperature and climatic water deficit for ROI5 and BG15; correlations of interannual changes in ECI with climate; the 2025 change in ECI of each cell compared with its climate anomalies | Fig. 4g; Tables S9 and S11 |
| `distance_from_sites/` | ECI in 2000 and 2025 by distance from the nearest registered site, for ROI5 and BG15, as medians and as standardized means with block bootstrap intervals; the share of the mapped Lingbei footprint area within each distance of a registered site | Fig. 14h, i; Table S6 |
| `cumulative_neighborhoods/` | Trend shares, recovery status, XGBoost performance and land cover attribution within 1, 2, 3 and 5 km of the sites and for all ROI5 cells | Table S6 |
| `footprint_check/` | Group medians with bootstrap intervals by the share of each cell covered by mapped footprints, and the Lingbei mining area compared with ROI5 | Table S18 |
| `bright_bare_exposure/` | ECI, REGAIN and the REGAIN slope for ROI5 cells without bright bare detections and for the quartiles of detection frequency | Fig. 14j; Table S18 |
| `multiple_testing_and_status/` | Trend classes under the per-cell, Benjamini-Hochberg and Benjamini-Yekutieli criteria; recovery status shares in 12 combinations of endpoint, high position threshold and trend criterion; label changes relative to the primary specification | Tables S10 and S13 |
| `attribution_stability/` | XGBoost performance, group shares and feature ranks under ten random assignments of the 10 km blocks to folds, and SHAP importance of the 40 predictors not derived from Landsat | Table S14 |
| `urban_exclusion/` | PLS-SEM effects and total associations, and XGBoost performance and group shares, without the 5% of cells with the fastest built-up fraction trend, without Zhanggong, or without both | Tables S5 and S14 |
| `spatial_dependence/` | XGBoost under 10 km block validation with 2 km and 5 km buffers, and the correlation of the response and of XGBoost residuals by distance | Table S15 |
| `sensor_checks/` | Annual median ECI from the mixed Landsat record and from Landsat 7 alone, and their difference | Table S11 |

Notes on the tables:

- BG15 is the band 5-15 km from the registered sites, without the 6,507 cells
  it shares with ROI5.
- In `climate_context/interannual_correlations.csv`, changes are
  year-to-year differences (n = 25) and anomalies are departures from the
  2000-2025 mean. In `cell_eci_change_2025_vs_climate_anomaly.csv`, the change
  in ECI of each cell from 2024 to 2025 is compared with its 2025 anomalies
  relative to its own 2000-2024 mean; the quintile column lists the median
  change from the lowest to the highest anomaly quintile.
- In `distance_from_sites/eci_by_distance_band.csv`, `raw_` columns are
  medians and quartiles of all cells in each band. `std_` columns are band
  means standardized to the ROI5 composition of elevation, slope and 2000 land
  cover strata, using strata with at least 30 cells in the band, and
  `std_terrain_only_` columns use elevation and slope strata only. Intervals
  (`_lo`, `_hi`) come from 1,000 bootstrap draws of whole 10 km blocks.
- The Lingbei footprint tables measure the distance from points placed 30 m
  apart inside each footprint polygon, mapped on 2025 imagery, to the nearest
  registered site.
- In `footprint_check/footprint_group_medians_ci.csv`, intervals come from
  4,000 resamples of cells within each group. Because neighboring cells are
  not independent, they are narrower than for independent samples.
- In `multiple_testing_and_status/trend_classes_by_criterion.csv`, the 57
  cells with nonpositive corrected variance are unevaluable, and a cell with a
  significant test but a Sen slope of zero is counted with no significant
  directional trend. The primary status specification uses REGAIN in 2025, a
  high position threshold of 0.75 and the per-cell Hamed-Rao test at
  p < 0.05.
- In `attribution_stability/`, the 237 blocks were assigned in random order to
  the fold with the fewest cells (seeds 1-10), and XGBoost was refitted with
  the primary configuration in each repeat. The model with the 40 predictors
  not derived from Landsat had a pooled R2 of 0.295 under the primary 10 km
  block folds.
- In `urban_exclusion/`, the PLS-SEM refits align composite signs with the
  full-sample fit and reuse its 1,000 block bootstrap draws.
- In `spatial_dependence/distance_correlation_response_and_residuals.csv`,
  correlations are mean products of globally standardized values within each
  distance bin, from a common sample of 686,313 cell pairs within 40 km.
  Residuals are the pooled test-fold residuals of XGBoost under 10 km block
  validation.
