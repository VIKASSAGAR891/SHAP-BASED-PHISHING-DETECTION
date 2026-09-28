"""Evaluation utilities for the trained phishing detection model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from config import ARTIFACT_DIR, DATASET_PATH, MODEL_PATH
from src.data import load_dataset

def evaluate_saved_model(
    dataset_path: str | Path = DATASET_PATH,
) -> dict:
    """Evaluate the saved model on the complete dataset.

    This evaluation is intended for diagnostic evidence after the model
    has already been trained. The official model metrics remain those
    obtained from the held-out test set during training.
    """

    bundle = joblib.load(MODEL_PATH)

    X, y = load_dataset(dataset_path)

    # Apply exactly the same feature selector used during training.
    X_selected = bundle["selector"].transform(X)

    model = bundle["model"]

    predictions = (
        model.predict(X_selected)
        .astype(int)
        .ravel()
    )

    # ---------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------
    metrics = {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision_phishing": float(
            precision_score(
                y,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_phishing": float(
            recall_score(
                y,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_phishing": float(
            f1_score(
                y,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
    }

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------
    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    # Save numerical confusion matrix.
    confusion_data = {
        "labels": [
            "phishing",
            "legitimate",
        ],
        "matrix": cm.tolist(),
    }

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        ARTIFACT_DIR / "confusion_matrix.json"
    ).write_text(
        json.dumps(
            confusion_data,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save confusion matrix figure.
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Phishing",
            "Legitimate",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"Confusion Matrix — {bundle['model_name']}"
    )

    fig.tight_layout()

    fig.savefig(
        ARTIFACT_DIR / "confusion_matrix_full.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------
    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=[
            "Phishing",
            "Legitimate",
        ],
        output_dict=True,
        zero_division=0,
    )

    (
        ARTIFACT_DIR / "classification_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------
    feature_importance = {}

    selected_features = bundle[
        "selected_features"
    ]

    if hasattr(model, "feature_importances_"):

        importance_values = (
            model.feature_importances_
        )

        feature_importance = {
            feature: float(score)
            for feature, score in zip(
                selected_features,
                importance_values,
            )
        }

        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        (
            ARTIFACT_DIR / "feature_importance.json"
        ).write_text(
            json.dumps(
                feature_importance,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---------------------------------------------------------------
    # Final evaluation summary
    # ---------------------------------------------------------------
    evaluation = {
        "model": bundle["model_name"],
        "dataset_rows_evaluated": int(len(y)),
        "metrics": metrics,
        "confusion_matrix": confusion_data,
        "classification_report": report,
        "feature_importance": feature_importance,
    }

    (
        ARTIFACT_DIR / "evaluation_summary.json"
    ).write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )

    return evaluation


if __name__ == "__main__":

    result = evaluate_saved_model()

    print(
        json.dumps(
            result["metrics"],
            indent=2,
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        np.array(
            result["confusion_matrix"]["matrix"]
        )
    )

    print(
        "\nSelected model:",
        result["model"],
    )