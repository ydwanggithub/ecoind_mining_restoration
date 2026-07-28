"""Verify the public release structure, data schema, and privacy boundary."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_COLUMNS = {
    "grid_id",
    "row",
    "col",
    "centroid_lon",
    "centroid_lat",
    "centroid_x_utm50",
    "centroid_y_utm50",
    "spatial_block_10km",
    "spatial_block_20km",
    "county",
    "site_id",
}
PRIVATE_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\", re.IGNORECASE),
    "local project path": re.compile(r"[A-Za-z]:\\\\GeoAI\\\\", re.IGNORECASE),
    "Earth Engine user asset": re.compile(
        r"\busers/[^ \t\r\n\"']+", re.IGNORECASE
    ),
    "hard coded Earth Engine project": re.compile(
        r"(?:ee\.Initialize\s*\(\s*project\s*=\s*[\"']|PROJECT\s*=\s*[\"'])",
        re.IGNORECASE,
    ),
    "credential material": re.compile(
        r"(?:private[_ -]?key|service[_ -]?account|api[_ -]?key|access[_ -]?token)\s*[:=]",
        re.IGNORECASE,
    ),
}


def fail(message: str) -> None:
    raise AssertionError(message)


def check_data() -> None:
    annual = pd.read_csv(ROOT / "data" / "annual_spectral_sample.csv")
    model = pd.read_csv(ROOT / "data" / "model_matrix_sample.csv.gz")
    ledger = pd.read_csv(ROOT / "data" / "predictor_ledger.csv")
    if FORBIDDEN_COLUMNS.intersection(annual.columns):
        fail(
            "Annual sample contains private columns: "
            f"{FORBIDDEN_COLUMNS.intersection(annual.columns)}"
        )
    if FORBIDDEN_COLUMNS.intersection(model.columns):
        fail(
            "Model sample contains private columns: "
            f"{FORBIDDEN_COLUMNS.intersection(model.columns)}"
        )
    if annual["sample_id"].nunique() != 120 or len(annual) != 120 * 26:
        fail("Annual sample must contain 120 cells and 26 years per cell.")
    if len(model) != 2500:
        fail("Model sample must contain 2500 deidentified cells.")
    retained = ledger["retained"].astype(str).str.lower().eq("true")
    if len(ledger) != 59 or retained.sum() != 57:
        fail(
            "Predictor ledger must contain 59 candidates and 57 retained predictors."
        )
    metadata = json.loads(
        (ROOT / "data" / "release_metadata.json").read_text(encoding="utf-8")
    )
    if (
        metadata["coordinates_removed"] is not True
        or metadata["internal_identifiers_removed"] is not True
    ):
        fail("Release metadata does not assert the deidentification boundary.")


def check_private_strings() -> None:
    extensions = {".py", ".yml", ".yaml", ".json", ".md", ".txt", ".csv"}
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or path.resolve() == Path(__file__).resolve()
            or path.suffix.lower() not in extensions
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                fail(f"{label} found in {path.relative_to(ROOT)}")
        if path.suffix == ".py" and re.search(
            r"\b(matplotlib|seaborn|plotly|cartopy)\b", text
        ):
            fail(f"Visualization dependency found in {path.relative_to(ROOT)}")


def check_eci_reconstruction() -> None:
    command = [
        sys.executable,
        str(ROOT / "analysis" / "02_construct_eci_regain.py"),
        "--output-dir",
        str(ROOT / "outputs" / "verification_eci"),
    ]
    subprocess.run(command, check=True)
    derived = pd.read_csv(
        ROOT / "outputs" / "verification_eci" / "annual_eci_regain.csv"
    )
    annual = pd.read_csv(ROOT / "data" / "annual_spectral_sample.csv")
    if (
        np.nanmax(
            np.abs(
                derived["eci_spectral"] - annual["expected_eci_spectral"]
            )
        )
        > 1e-10
    ):
        fail("ECI reconstruction differs from the released expected values.")
    if (
        np.nanmax(
            np.abs(
                derived["regain_annual"] - annual["expected_regain_annual"]
            )
        )
        > 1e-10
    ):
        fail("REGAIN reconstruction differs from the released expected values.")


def main() -> None:
    check_data()
    check_private_strings()
    check_eci_reconstruction()
    print("Release verification passed.")


if __name__ == "__main__":
    main()
