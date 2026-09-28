"""Command-line entry point for training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_PATH, MODEL_PATH
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost and CatBoost for URL phishing detection.")
    parser.add_argument("--sample", type=int, default=None, help="Optional smaller sample for quick development runs.")
    args = parser.parse_args()

    metrics = train(str(DATASET_PATH), sample_size=args.sample)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model bundle: {MODEL_PATH}")


if __name__ == "__main__":
    main()
