"""Download PhiUSIIL from the official UCI repository using ucimlrepo."""

from pathlib import Path
import sys

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"


def main():
    print("Fetching UCI dataset 967 (PhiUSIIL Phishing URL Dataset)...")
    dataset = fetch_ucirepo(id=967)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    target.columns = ["label"]
    df = pd.concat([features, target], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
