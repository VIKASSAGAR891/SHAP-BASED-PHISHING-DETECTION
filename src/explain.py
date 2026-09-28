"""Explainable AI utilities using SHAP for the phishing detector.

This module provides:
1. Global SHAP feature importance.
2. Individual URL explanations.
3. Human-readable explanations for dashboard use.

The model is trained using URL-only features, so the explanation
pipeline also operates only on the submitted URL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import ARTIFACT_DIR, MODEL_PATH
from src.features import (
    FEATURE_COLUMNS,
    extract_url_features,
    explain_feature,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_bundle():
    """Load the trained phishing model bundle."""

    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------------------------

def create_explainer(model):
    """Create a TreeExplainer for the trained XGBoost model."""

    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# Prepare a single URL
# ---------------------------------------------------------------------------

def prepare_url_features(
    url: str,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert one URL into the exact feature representation used by the model.

    Returns
    -------
    raw_features:
        All 18 URL-derived features.

    selected_features:
        The 15 features used by the trained model.
    """

    raw_features = pd.DataFrame(
        [
            extract_url_features(url)
        ],
        columns=FEATURE_COLUMNS,
    ).fillna(0)

    selected_features = bundle["selector"].transform(
        raw_features
    )

    selected_features = pd.DataFrame(
        selected_features,
        columns=bundle["selected_features"],
    )

    return raw_features, selected_features


# ---------------------------------------------------------------------------
# Individual URL explanation
# ---------------------------------------------------------------------------

def explain_url(
    url: str,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a prediction and SHAP explanation for one URL.

    The returned structure is designed to be directly usable by
    the dashboard API.
    """

    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]

    raw_features, selected_features = prepare_url_features(
        url,
        bundle,
    )

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    prediction = int(
        np.asarray(
            model.predict(selected_features)
        ).ravel()[0]
    )

    probabilities = model.predict_proba(
        selected_features
    )[0]

    phishing_probability = float(
        probabilities[0]
    )

    legitimate_probability = float(
        probabilities[1]
    )

    label = (
        "phishing"
        if prediction == 0
        else "legitimate"
    )

    risk_score = (
        phishing_probability
        if label == "phishing"
        else legitimate_probability
    )

    # ---------------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------------

    explainer = create_explainer(model)

    shap_values = explainer.shap_values(
        selected_features
    )

    # XGBoost binary classification normally returns an array:
    #
    # shape = (samples, features)
    #
    # Some SHAP versions can return a list or Explanation object,
    # so we normalise the output.

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    shap_row = shap_values[0]

    # ---------------------------------------------------------------
    # Build feature contribution table
    # ---------------------------------------------------------------

    contributions = []

    for feature, value, shap_value in zip(
        bundle["selected_features"],
        selected_features.iloc[0].values,
        shap_row,
    ):

        # SHAP for this binary XGBoost model is oriented toward
        # the legitimate class (class 1).
        #
        # For a phishing prediction (class 0), invert the direction
        # so that positive values mean "pushes toward the predicted
        # class".

        contribution = float(shap_value)

        if prediction == 0:
            contribution = -contribution

        contributions.append(
            {
                "feature": feature,
                "value": float(value),
                "shap_value": float(shap_value),
                "contribution_to_prediction": contribution,
                "direction": (
                    "supports prediction"
                    if contribution > 0
                    else "opposes prediction"
                ),
                "description": explain_feature(
                    feature
                ),
            }
        )

    # Strongest contributors first.
    contributions.sort(
        key=lambda item: abs(
            item["contribution_to_prediction"]
        ),
        reverse=True,
    )

    top_contributors = contributions[:5]

    # ---------------------------------------------------------------
    # Human-readable indicators
    # ---------------------------------------------------------------

    indicators = []

    for item in top_contributors:

        if item["contribution_to_prediction"] > 0:

            indicators.append(
                {
                    "feature": item["feature"],
                    "description": item["description"],
                    "effect": "supports prediction",
                }
            )

    # ---------------------------------------------------------------
    # Return dashboard-ready result
    # ---------------------------------------------------------------

    return {
        "url": url,
        "prediction": label,
        "prediction_label": (
            "PHISHING DETECTED"
            if label == "phishing"
            else "LEGITIMATE URL"
        ),
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "risk_score": float(risk_score),
        "risk_score_percent": round(
            risk_score * 100,
            2,
        ),
        "top_contributors": top_contributors,
        "indicators": indicators,
        "features": {
            feature: float(
                raw_features.iloc[0][feature]
            )
            for feature in FEATURE_COLUMNS
        },
    }


# ---------------------------------------------------------------------------
# Global SHAP analysis
# ---------------------------------------------------------------------------

def generate_global_shap(
    bundle: dict[str, Any] | None = None,
    sample_size: int = 5000,
) -> dict[str, float]:
    """Generate global mean absolute SHAP importance.

    A sample is used rather than explaining the complete dataset,
    which keeps the analysis computationally manageable.
    """

    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]

    # Load the dataset through the existing data pipeline.
    from config import DATASET_PATH
    from src.data import load_dataset

    X, _ = load_dataset(DATASET_PATH)

    # Apply exactly the same feature selector.
    X_selected = bundle[
        "selector"
    ].transform(X)

    X_selected = pd.DataFrame(
        X_selected,
        columns=bundle["selected_features"],
    )

    # Sample for computational efficiency.
    if len(X_selected) > sample_size:
        X_sample = X_selected.sample(
            n=sample_size,
            random_state=42,
        )
    else:
        X_sample = X_selected

    explainer = create_explainer(model)

    shap_values = explainer.shap_values(
        X_sample
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    importance = {
        feature: float(score)
        for feature, score in zip(
            bundle["selected_features"],
            mean_absolute_shap,
        )
    }

    importance = dict(
        sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        ARTIFACT_DIR / "shap_global_importance.json"
    ).write_text(
        json.dumps(
            importance,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Save plot
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    features = list(
        importance.keys()
    )[::-1]

    values = [
        importance[feature]
        for feature in features
    ]

    ax.barh(
        features,
        values,
    )

    ax.set_title(
        "Global SHAP Feature Importance — XGBoost"
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Value"
    )

    fig.tight_layout()

    fig.savefig(
        ARTIFACT_DIR
        / "shap_global_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return importance


# ---------------------------------------------------------------------------
# Individual explanation plot
# ---------------------------------------------------------------------------

def save_url_explanation_plot(
    url: str,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and save a SHAP explanation plot for one URL."""

    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]

    _, selected_features = prepare_url_features(
        url,
        bundle,
    )

    explainer = create_explainer(model)

    shap_values = explainer.shap_values(
        selected_features
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    # Create Explanation object for waterfall plot.
    base_value = explainer.expected_value

    if isinstance(base_value, np.ndarray):
        base_value = float(
            np.asarray(base_value).ravel()[-1]
        )
    else:
        base_value = float(
            base_value
        )

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=selected_features.iloc[0].values,
        feature_names=bundle[
            "selected_features"
        ],
    )

    fig = plt.figure(
        figsize=(10, 7)
    )

    shap.plots.waterfall(
        explanation,
        max_display=10,
        show=False,
    )

    fig = plt.gcf()

    fig.tight_layout()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        ARTIFACT_DIR
        / "shap_url_explanation.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "url": url,
        "plot": str(output_path),
    }

def explain_prediction(
    model,
    selected,
    feature_names,
    row_index: int = 0,
    predicted_class: int | None = None,
):
    """Return SHAP feature contributions for one prediction.

    The returned values are oriented toward the predicted class so
    the dashboard can identify which features support or oppose the
    prediction.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        selected
    )

    if isinstance(shap_values, list):
        # Binary XGBoost models may return one array per class.
        if predicted_class is not None:
            class_index = int(predicted_class)
        else:
            class_index = -1

        shap_values = shap_values[class_index]

    if hasattr(
        shap_values,
        "values",
    ):
        shap_values = shap_values.values

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        if predicted_class is not None:
            shap_values = shap_values[
                :,
                :,
                int(predicted_class),
            ]
        else:
            shap_values = shap_values[
                :,
                :,
                -1,
            ]

    values = shap_values[
        row_index
    ]

    pairs = []

    for feature, value in zip(
        feature_names,
        values,
    ):

        contribution = float(value)

        # For class 0 (phishing), reverse the direction so that
        # positive values represent support for the predicted class.
        if predicted_class == 0:
            contribution = -contribution

        pairs.append(
            (
                feature,
                contribution,
            )
        )

    pairs.sort(
        key=lambda item: abs(
            item[1]
        ),
        reverse=True,
    )

    return pairs
# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    bundle = load_model_bundle()

    print("=" * 70)
    print("PHISHING DETECTION — SHAP ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Global explanation
    # ---------------------------------------------------------------

    print("\nGenerating global SHAP importance...")

    global_importance = generate_global_shap(
        bundle=bundle,
        sample_size=5000,
    )

    print("\nTop global SHAP features:")

    for index, (
        feature,
        value,
    ) in enumerate(
        global_importance.items(),
        start=1,
    ):

        print(
            f"{index:2}. "
            f"{feature:<30} "
            f"{value:.6f}"
        )

    # ---------------------------------------------------------------
    # Example URL
    # ---------------------------------------------------------------

    example_url = (
        "https://example.com/login/verify?id=12345"
    )

    print(
        "\nGenerating individual explanation..."
    )

    result = explain_url(
        example_url,
        bundle,
    )

    print(
        "\nURL:",
        example_url,
    )

    print(
        "Prediction:",
        result["prediction_label"],
    )

    print(
        "Phishing probability:",
        f"{result['phishing_probability'] * 100:.2f}%",
    )

    print(
        "Legitimate probability:",
        f"{result['legitimate_probability'] * 100:.2f}%",
    )

    print(
        "Risk score:",
        f"{result['risk_score_percent']:.2f}%",
    )

    print("\nTop contributors:")

    for item in result[
        "top_contributors"
    ]:

        print(
            f"- {item['feature']}: "
            f"{item['contribution_to_prediction']:.6f} "
            f"({item['direction']})"
        )

    # ---------------------------------------------------------------
    # Save individual explanation
    # ---------------------------------------------------------------

    save_url_explanation_plot(
        example_url,
        bundle,
    )

    (
        ARTIFACT_DIR
        / "shap_example_explanation.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSHAP analysis completed."
    )

    print(
        "Artifacts saved to:",
        ARTIFACT_DIR,
    )