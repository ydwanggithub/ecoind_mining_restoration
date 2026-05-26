# ecoind_mining_restoration

Companion demonstration code and sample data for the REGAIN recovery indicator,
released alongside the manuscript

> Spatially heterogeneous recovery realization in rare-earth mining landscapes
> (under review)

REGAIN (REcovery GAp INdex) compares the initial and current ecological deficits
between observed and environmentally potential reference conditions through a
sum-normalised difference, with values bounded in [-1, +1] by definition and no
imposed clipping.

## What this repository contains

This repository provides a small, self-contained reference implementation so
that other groups can reproduce the indicator construction, the spatial
machine-learning attribution, and the basic mapping workflow on their own data.
It does not contain the full source datasets, the GEE-side preprocessing
pipeline, or the manuscript text or figures.

```
demo_data/
  sample_regain_demo.csv   synthetic 500-cell tabular sample (random seed fixed)

demo_code/
  01_demo_load_data.py     load the sample CSV and print a head summary
  02_demo_compute_regain.py compute REGAIN from synthetic initial / current ECI
  03_demo_train_rf.py      train a Random Forest regressor on synthetic predictors
  04_demo_shap_summary.py  produce a global SHAP bar and dependence plot
  05_demo_visualize_map.py scatter the synthetic cells over a square extent

figures/
  Fig01_study_design.png ... Fig14_restoration_priority_synthesis.png
  Rendered PNG copies of the 14 manuscript figures, provided for quick visual
  reference. See the manuscript captions for full panel descriptions.
```

## Quick start

```bash
conda env create -f environment.yml
conda activate regain-demo
python demo_code/01_demo_load_data.py
python demo_code/02_demo_compute_regain.py
python demo_code/03_demo_train_rf.py
python demo_code/04_demo_shap_summary.py
python demo_code/05_demo_visualize_map.py
```

Each script is independent and runs on the bundled synthetic CSV in about 5
seconds on a laptop. No GPU or remote service is required.

## Source datasets used in the manuscript

The manuscript itself draws on publicly available remote-sensing products:

- Landsat Collection 2 surface reflectance (USGS, via Google Earth Engine)
- CLCD annual land cover (Yang and Huang 2021)
- ERA5-Land monthly climate aggregates (Munoz Sabater 2021)
- VIIRS night-time lights (Elvidge et al. 2017)
- SRTM 30 m elevation
- Registered mining-right point inventory (Ganzhou Natural Resource Bureau,
  available under the original access conditions of the data provider)

This repository does not redistribute these sources; consult each provider for
licence terms and access.

## Licence

Released under the MIT Licence (see LICENSE).

## Citation

If you use this code or the REGAIN indicator in your own work, please cite the
manuscript once published.
