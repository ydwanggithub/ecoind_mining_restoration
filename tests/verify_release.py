"""Verify the public release structure, data schema, and privacy boundary."""

from __future__ import annotations

import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_data() -> None:
    annual = pd.read_csv(ROOT / "data" / "annual_spectral_sample.csv")
    full_model = pd.read_csv(
        ROOT / "data" / "model_matrix_full_deidentified.csv.gz"
    )
    model = pd.read_csv(ROOT / "data" / "model_matrix_sample.csv.gz")
    preview = pd.read_csv(ROOT / "data" / "model_matrix_preview.csv")
    ledger = pd.read_csv(ROOT / "data" / "predictor_ledger.csv")
    for label, frame in {
        "Annual sample": annual,
        "Full model matrix": full_model,
        "Model sample": model,
        "Model preview": preview,
    }.items():
        forbidden = FORBIDDEN_COLUMNS.intersection(frame.columns)
        if forbidden:
            fail(f"{label} contains private columns: {forbidden}")
    if annual["sample_id"].nunique() != 120 or len(annual) != 120 * 26:
        fail("Annual sample must contain 120 cells and 26 years per cell.")
    if len(full_model) != 38273:
        fail("Full model matrix must contain 38273 deidentified cells.")
    if len(model) != 2500:
        fail("Model sample must contain 2500 deidentified cells.")
    if len(preview) != 50:
        fail("Model preview must contain 50 deidentified cells.")
    if full_model["sample_id"].duplicated().any():
        fail("Full model matrix contains duplicate anonymous identifiers.")
    if full_model.isna().any().any():
        fail("Full model matrix contains missing values.")
    for label, frame in {
        "Annual sample": annual,
        "Full model matrix": full_model,
        "Model sample": model,
        "Model preview": preview,
    }.items():
        if not frame["sample_id"].str.fullmatch(r"cell_\d{5}").all():
            fail(f"{label} contains an invalid anonymous identifier.")
    if not set(model["sample_id"]).issubset(set(full_model["sample_id"])):
        fail("Model sample is not a subset of the full model matrix.")
    if not set(annual["sample_id"]).issubset(set(model["sample_id"])):
        fail("Annual spectral sample is not linked to the model sample.")
    if not preview.equals(model.head(50)):
        fail("Browser preview does not match the first 50 model sample rows.")
    retained = ledger["retained"].astype(str).str.lower().eq("true")
    if len(ledger) != 59 or retained.sum() != 57:
        fail(
            "Predictor ledger must contain 59 candidates and 57 retained predictors."
        )
    expected_columns = {
        "sample_id",
        "regain_sen_slope_2000_2025",
        *ledger["name"].tolist(),
    }
    for label, frame in {
        "Full model matrix": full_model,
        "Model sample": model,
        "Model preview": preview,
    }.items():
        if set(frame.columns) != expected_columns:
            fail(f"{label} does not match the released predictor ledger.")
    metadata = json.loads(
        (ROOT / "data" / "release_metadata.json").read_text(encoding="utf-8")
    )
    if (
        metadata["coordinates_removed"] is not True
        or metadata["internal_identifiers_removed"] is not True
    ):
        fail("Release metadata does not assert the deidentification boundary.")
    if metadata["full_model_matrix_rows"] != 38273:
        fail("Release metadata reports an incorrect full model row count.")
    if metadata["xgboost_version"] != "2.1.4":
        fail("Release metadata does not record the locked XGBoost version.")
    for relative_path, expected_hash in metadata["checksums_sha256"].items():
        path = ROOT / relative_path
        if not path.is_file():
            fail(f"Release checksum target is missing: {relative_path}")
        if sha256(path) != expected_hash:
            fail(f"Release checksum mismatch: {relative_path}")


def check_private_strings() -> None:
    extensions = {".py", ".yml", ".yaml", ".json", ".md", ".txt", ".csv"}
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or any(part.startswith(".venv") for part in path.parts)
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
