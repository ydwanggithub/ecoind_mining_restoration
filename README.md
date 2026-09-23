# Ecological recovery across ionic rare earth mining landscapes

This repository accompanies the manuscript:

> **Spatial heterogeneity and driving mechanisms of ecological recovery across the ionic rare earth mining landscapes of Ganzhou in southern China from 2000 to 2025**

The study follows ecological condition around 401 officially registered ionic
rare earth mineral sites in Ganzhou, southern China, using annual Landsat
records from 2000 to 2025. It examines spatial differences in long term
recovery direction with spatially cross-validated XGBoost, SHAP attribution
and PLS-SEM.

The repository provides a privacy screened implementation of the main
analytical sequence, the complete deidentified model matrix, the fold
assignments and out-of-fold predictions used for model evaluation, a smaller
working example, and aggregate result tables. This release corresponds to the
revised *Ecological Indicators* manuscript.

## Analytical sequence

1. Annual Landsat observations are summarized as kNDVI, NBR, NDMI and BSI.
2. The four components are scaled with one fixed 2000-2025 contract and averaged
   to form the ecological condition index (ECI).
3. Annual ECI at each location is scaled within the range observed at that
   location:

   `REGAIN_it = (ECI_it - min_t ECI_it) / (max_t ECI_it - min_t ECI_it)`

4. The Theil-Sen slope of annual REGAIN (recovery direction) is the single
   response used in the machine learning and pathway analyses. Trend
   significance uses the Hamed-Rao corrected Mann-Kendall test.
5. Fifty nine contextual candidates from climate, hydrology, terrain, land
   cover and human activity are screened without using the response. Fifty
   seven predictors are retained.
6. XGBoost, LightGBM, CatBoost and Random Forest are evaluated with fivefold
   cross validation in which whole 10 km spatial blocks are withheld. The
   20 km block design and random folds serve as comparisons. Exact tree SHAP
   contributions are calculated only for observations in each test fold.
7. PLS-SEM, with whole-block bootstrap intervals, and empirical quantile
   boundaries, with whole-block resampling, provide complementary summaries of
   relationships among predictor groups and of the upper and lower limits of
   the response.

REGAIN describes the annual position of a location within its observed
2000-2025 ECI range. Its reference is temporal and local, rather than a
reconstructed condition before mining or an ecological optimum.

## Repository contents

```text
analysis/
  01_extract_landsat_components_gee.py
  02_construct_eci_regain.py
  03_fit_xgboost_shap.py
  04_empirical_quantile_boundaries.py

data/
  annual_spectral_sample.csv
  model_matrix_full_deidentified.csv.gz
  model_matrix_sample.csv.gz
  model_matrix_preview.csv
  cv_fold_assignments.csv.gz
  eci_scaling_contract.csv
  predictor_ledger.csv
  release_metadata.json
  results/
    annual_eci_regain_summary.csv
    model_validation_summary.csv
    oof_predictions_deidentified.csv.gz
    xgboost_shap_feature_importance.csv
    xgboost_shap_family_importance.csv
    pls_sem_path_inference.csv
    dual_constraint_summary.csv
    predictor_family_summary.csv
    revision/            (tables for the analyses added in the revision)

figures/
  Fig01_study_area.png ... Fig13_recovery_management_contexts.png

tests/
  verify_release.py
```

`model_matrix_full_deidentified.csv.gz` contains all 38,273 observations used
in the model analysis. It includes the REGAIN trend response and the 59
candidate predictors recorded in `predictor_ledger.csv`. Internal cell
identifiers, coordinates, administrative labels, spatial block identifiers,
asset paths and registered site locations have been removed. Anonymous
`sample_id` values preserve one to one row identity without disclosing
location.

`cv_fold_assignments.csv.gz` gives, for each `sample_id`, the fold used in the
random fivefold, 10 km block and 20 km block designs. The block identifiers
themselves are not released. `oof_predictions_deidentified.csv.gz` contains
the observed response and the out-of-fold predictions of all four tree models
under the three designs. `results/revision/` holds the tables for the
analyses added in the revision; its README lists the contents of each folder.

`model_matrix_sample.csv.gz` contains 2,500 rows for a quicker trial run.
`model_matrix_preview.csv` contains 50 of those rows in an uncompressed form
that can be viewed directly on GitHub. The annual spectral sample supports a
small reconstruction of ECI and annual REGAIN.

Project specific Earth Engine identifiers are intentionally absent. The
Landsat extraction template reads the Earth Engine project and analysis grid
asset from environment variables supplied by the user. Figure rendering code
is outside the scope of this release. The `figures` directory contains the
figures of the originally submitted manuscript; the revised figures will be
added once they are finalized.

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate regain-reproducibility
```

Run the local workflow:

```bash
python analysis/02_construct_eci_regain.py
python analysis/03_fit_xgboost_shap.py
python analysis/04_empirical_quantile_boundaries.py --bootstrap 100
python tests/verify_release.py
```

The default model and boundary commands use the 2,500 row example. To rerun
the XGBoost and SHAP analysis with the complete deidentified matrix under the
primary 10 km spatial block design, use:

```bash
python analysis/03_fit_xgboost_shap.py \
  --data data/model_matrix_full_deidentified.csv.gz \
  --cv-design spatial_block_10km \
  --output-dir outputs/xgboost_shap_block10
```

The full matrix retains the row order used in the manuscript analysis. With
XGBoost 2.1.4 pinned in `environment.yml`, this command reproduces the pooled
10 km block validation statistics reported in the revised manuscript
(`R2 = 0.611991`, `RMSE = 0.007749` and `MAE = 0.005975`) and the SHAP
importance values, including the group shares of land cover (41.5%) and
hydrology (25.4%). Replacing `spatial_block_10km` with `spatial_block_20km` or
`random_5fold` runs the comparison designs, whose statistics are listed in
`results/revision/spatial_validation/oof_performance_recalculated.csv`.
Omitting `--cv-design` uses the seeded shuffled fivefold partition of the
original release (`R2 = 0.710556`). The smaller example is intended for code
inspection and a faster trial run.

The Earth Engine template is optional:

```bash
set EE_PROJECT=your-earth-engine-project
set ROI_GRID_ASSET=projects/your-project/assets/your-grid
python analysis/01_extract_landsat_components_gee.py
```

The script only starts an export when `--start-export` is supplied.

## Source data

The manuscript draws on provider hosted products, including Landsat Collection
2 surface reflectance, ERA5-Land, TerraClimate, CLCD, VIIRS nighttime lights,
GHS-POP, GRIP roads, SRTM and MERIT Hydro. Access and licensing remain with
the respective providers. The official mineral site inventory is not
redistributed.

## License

The code is released under the MIT License. The deidentified model matrix,
fold assignments, working samples and aggregate tables are provided for
scholarly reproduction of the reported workflow.
