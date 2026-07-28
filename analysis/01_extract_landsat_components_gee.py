"""Prepare annual Landsat spectral components in Google Earth Engine.

The template contains no account, project, or asset identifiers. Supply those
values through environment variables:

    EE_PROJECT
    ROI_GRID_ASSET

By default the script validates the inputs and prints the first annual
collection. An export starts only when ``--start-export`` is supplied.
"""

from __future__ import annotations

import argparse
import os

import ee


START_YEAR = 2000
END_YEAR = 2025
SCALE_FACTOR = 0.0000275
OFFSET = -0.2


def require_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running.")
    return value


def mask_landsat(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    unsaturated = image.select("QA_RADSAT").eq(0)
    return image.updateMask(clear).updateMask(unsaturated)


def prepare_tm_etm(image: ee.Image) -> ee.Image:
    scaled = (
        image.select(["SR_B1", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
        .multiply(SCALE_FACTOR)
        .add(OFFSET)
        .rename(["blue", "red", "nir", "swir1", "swir2"])
    )
    return add_indices(scaled).copyProperties(image, ["system:time_start"])


def prepare_oli(image: ee.Image) -> ee.Image:
    scaled = (
        image.select(["SR_B2", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(SCALE_FACTOR)
        .add(OFFSET)
        .rename(["blue", "red", "nir", "swir1", "swir2"])
    )
    return add_indices(scaled).copyProperties(image, ["system:time_start"])


def add_indices(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(["nir", "red"]).rename("ndvi_mean")
    kndvi = ndvi.pow(2).tanh().rename("kndvi_mean")
    nbr = image.normalizedDifference(["nir", "swir2"]).rename("nbr_mean")
    ndmi = image.normalizedDifference(["nir", "swir1"]).rename("ndmi_mean")
    bsi = image.expression(
        "((s1 + r) - (n + b)) / ((s1 + r) + (n + b))",
        {
            "s1": image.select("swir1"),
            "r": image.select("red"),
            "n": image.select("nir"),
            "b": image.select("blue"),
        },
    ).rename("bsi_mean")
    return ee.Image.cat([ndvi, kndvi, nbr, ndmi, bsi])


def landsat_collection(roi: ee.Geometry) -> ee.ImageCollection:
    collection_ids = (
        ("LANDSAT/LT05/C02/T1_L2", prepare_tm_etm),
        ("LANDSAT/LE07/C02/T1_L2", prepare_tm_etm),
        ("LANDSAT/LC08/C02/T1_L2", prepare_oli),
        ("LANDSAT/LC09/C02/T1_L2", prepare_oli),
    )
    merged = ee.ImageCollection([])
    for collection_id, prepare in collection_ids:
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(roi)
            .filterDate(f"{START_YEAR}-01-01", f"{END_YEAR + 1}-01-01")
            .map(mask_landsat)
            .map(prepare)
        )
        merged = merged.merge(collection)
    return merged


def annual_composites(images: ee.ImageCollection) -> ee.ImageCollection:
    def one_year(year: ee.Number) -> ee.Image:
        year = ee.Number(year)
        annual = images.filter(ee.Filter.calendarRange(year, year, "year")).median()
        return annual.set(
            {
                "year": year,
                "system:time_start": ee.Date.fromYMD(year, 7, 1).millis(),
            }
        )

    years = ee.List.sequence(START_YEAR, END_YEAR)
    return ee.ImageCollection.fromImages(years.map(one_year))


def export_table(annual: ee.ImageCollection, grid: ee.FeatureCollection) -> ee.batch.Task:
    def summarize_year(year: ee.Number) -> ee.FeatureCollection:
        year = ee.Number(year)
        image = annual.filter(ee.Filter.eq("year", year)).first()
        return image.reduceRegions(
            collection=grid,
            reducer=ee.Reducer.mean(),
            scale=30,
            tileScale=4,
        ).map(lambda feature: feature.set("year", year))

    per_year = ee.FeatureCollection(
        ee.List.sequence(START_YEAR, END_YEAR).map(summarize_year)
    ).flatten()
    folder = os.environ.get("EE_EXPORT_FOLDER", "regain_exports")
    return ee.batch.Export.table.toDrive(
        collection=per_year,
        description="annual_landsat_components_2000_2025",
        folder=folder,
        fileFormat="CSV",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = require_environment("EE_PROJECT")
    grid_asset = require_environment("ROI_GRID_ASSET")
    ee.Initialize(project=project)
    grid = ee.FeatureCollection(grid_asset)
    annual = annual_composites(landsat_collection(grid.geometry()))
    print("Annual composites:", annual.size().getInfo())
    print("First image bands:", annual.first().bandNames().getInfo())
    if args.start_export:
        task = export_table(annual, grid)
        task.start()
        print("Started export:", task.id)
    else:
        print("Validation complete. Add --start-export to create a Drive export.")


if __name__ == "__main__":
    main()
