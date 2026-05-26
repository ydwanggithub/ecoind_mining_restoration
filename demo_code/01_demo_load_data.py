"""Load the bundled synthetic demo CSV and print a head summary.

Run from the repository root:
    python demo_code/01_demo_load_data.py
"""
from pathlib import Path
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "sample_regain_demo.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("First 5 rows:")
    print(df.head())
    print()
    print("Summary statistics:")
    print(df.describe().round(3))


if __name__ == "__main__":
    main()
