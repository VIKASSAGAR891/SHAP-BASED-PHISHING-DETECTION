"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import transform_urls

REQUIRED_COLUMNS = {"URL", "label"}


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place PhiUSIIL_Phishing_URL_Dataset.csv in data/ "
            "or run scripts/download_dataset.py."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["URL", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels encoded as 0/1.")

    X = transform_urls(df["URL"])
    y = df["label"]
    return X, y
