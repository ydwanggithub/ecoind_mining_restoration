# Ecological recovery across ionic rare earth mining landscapes

This repository accompanies the manuscript:

> **Ecological recovery heterogeneity and its driving mechanisms across ionic rare earth mining landscapes revealed by interpretable machine learning using remote sensing time series from 2000 to 2025**

It provides a privacy screened implementation of the main analytical sequence,
deidentified example data, and aggregate result tables. The release follows the
current *Ecological Informatics* manuscript.

## Analytical sequence

1. Annual Landsat observations are summarized as kNDVI, NBR, NDMI, and BSI.
2. The four components are scaled with one fixed 2000-2025 contract and averaged
   to form the ecological condition index (ECI).
3. Annual ECI at each location is scaled within the range observed at that
   location:

   `REGAIN_it = (ECI_it - min_t ECI_it) / (max_t ECI_it - min_t ECI_it)`

4. The Theil-Sen slope of annual REGAIN is the single response used in the
   machine learning and pathway analyses.
5. Fifty nine contextual candidates from climate, hydrology, terrain, land
   cover, and human activity are screened without using the response. Fifty
   seven predictors are retained.
6. XGBoost is evaluated with shuffled random fivefold cross validation. Exact
   tree SHAP contributions are calculated only for observations omitted from
   each training fold.
7. Path modeling and empirical quantile boundaries provide complementary
   summaries of relationships among driver groups and nonlinear response
   limits.

REGAIN describes the annual position of a location within its observed
2000-2025 ECI range. It does not represent a reconstructed condition before
mining, an ecological optimum, or a pollution indicator.

## Repository contents

```text
analysis/
  01_extract_landsat_components_gee.py
  02_construct_eci_regain.py
  03_fit_xgboost_shap.py
  04_empirical_quantile_boundaries.py

data/
  annual_spectral_sample.csv
  model_matrix_sample.csv.gz
  eci_scaling_contract.csv
  predictor_ledger.csv
  release_metadata.json
  results/
    annual_eci_regain_summary.csv
    model_validation_summary.csv
    xgboost_shap_feature_importance.csv
    xgboost_shap_family_importance.csv
    pls_sem_path_inference.csv
    dual_constraint_summary.csv

figures/
  Fig01_study_area.png ... Fig13_recovery_management_contexts.png

tests/
  verify_release.py
```

The sample tables retain observed numerical values but replace internal cell
identifiers and remove coordinates, administrative labels, spatial block
identifiers, asset paths, and registered site locations. They are intended to
exercise the released code and document the data schema. Aggregate result
tables report the full analysis summarized in the manuscript.

Project specific Earth Engine identifiers are intentionally absent. The
Landsat extraction template reads the Earth Engine project and analysis grid
asset from environment variables supplied by the user. Figure rendering code
is outside the scope of this release; the `figures` directory contains only
the current static manuscript figures.

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

The model and boundary scripts use the deidentified sample, so their numerical
results are demonstration results rather than replacements for the full sample
statistics in `data/results/`.

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
GHS-POP, GRIP roads, SRTM, and MERIT Hydro. Access and licensing remain with
the respective providers. The official mineral site inventory is not
redistributed.

## License

The code is released under the MIT License. The bundled sample and aggregate
tables are provided for scholarly reproduction of the reported workflow.
