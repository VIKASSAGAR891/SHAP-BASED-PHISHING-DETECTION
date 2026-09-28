"""Generate a Cumulative Accuracy Profile (CAP) curve for the selected model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ARTIFACT_DIR, DATASET_PATH, RANDOM_STATE, TEST_SIZE
from src.data import load_dataset


def generate_cap_curve() -> None:
    # ---------------------------------------------------------------
    # Load the same dataset and trained model bundle
    # ---------------------------------------------------------------
    X, y = load_dataset(DATASET_PATH)

    bundle = joblib.load(PROJECT_ROOT / "models" / "phishing_model.joblib")

    model = bundle["model"]
    selector = bundle["selector"]

    # ---------------------------------------------------------------
    # Recreate the same held-out test split used during training
    # ---------------------------------------------------------------
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Apply the already-fitted feature selector.
    X_test_selected = selector.transform(X_test)

    # ---------------------------------------------------------------
    # Get phishing probabilities
    # Label 0 = phishing
    # Label 1 = legitimate
    # ---------------------------------------------------------------
    probabilities = model.predict_proba(X_test_selected)

    classes = [int(c) for c in model.classes_]
    phishing_index = classes.index(0)

    phishing_scores = probabilities[:, phishing_index]

    y_test_array = np.asarray(y_test).astype(int)

    # ---------------------------------------------------------------
    # Sort URLs from highest to lowest phishing probability
    # ---------------------------------------------------------------
    order = np.argsort(-phishing_scores)

    sorted_labels = y_test_array[order]

    total_phishing = np.sum(y_test_array == 0)

    if total_phishing == 0:
        raise RuntimeError("No phishing samples were found in the test set.")

    cumulative_phishing = np.cumsum(sorted_labels == 0)

    population_percentage = (
        np.arange(1, len(sorted_labels) + 1) / len(sorted_labels)
    )

    captured_phishing_percentage = (
        cumulative_phishing / total_phishing
    )

    # ---------------------------------------------------------------
    # Random baseline
    # ---------------------------------------------------------------
    random_baseline = population_percentage

    # ---------------------------------------------------------------
    # Ideal CAP curve
    # ---------------------------------------------------------------
    prevalence = total_phishing / len(sorted_labels)

    ideal_curve = np.minimum(
        population_percentage / prevalence,
        1.0,
    )

    # ---------------------------------------------------------------
    # CAP score
    #
    # Area under the model CAP curve compared with the
    # random and ideal areas.
    # ---------------------------------------------------------------
    model_area = np.trapezoid(
        captured_phishing_percentage,
        population_percentage,
    )

    random_area = np.trapezoid(
        random_baseline,
        population_percentage,
    )

    ideal_area = np.trapezoid(
        ideal_curve,
        population_percentage,
    )

    cap_score = (
        (model_area - random_area)
        / (ideal_area - random_area)
    )

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Save CAP curve
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        population_percentage * 100,
        captured_phishing_percentage * 100,
        linewidth=2,
        label=f"{bundle['model_name']} (CAP = {cap_score:.4f})",
    )

    ax.plot(
        population_percentage * 100,
        random_baseline * 100,
        linestyle="--",
        linewidth=1.5,
        label="Random baseline",
    )

    ax.plot(
        population_percentage * 100,
        ideal_curve * 100,
        linestyle=":",
        linewidth=1.5,
        label="Ideal model",
    )

    ax.set_title(
        f"Cumulative Accuracy Profile — {bundle['model_name']}"
    )

    ax.set_xlabel(
        "Percentage of URLs examined"
    )

    ax.set_ylabel(
        "Percentage of phishing URLs identified"
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output_path = ARTIFACT_DIR / "cap_curve_xgboost.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Save numerical results
    # ---------------------------------------------------------------
    cap_results = {
        "model": bundle["model_name"],
        "test_samples": int(len(y_test_array)),
        "phishing_samples": int(total_phishing),
        "legitimate_samples": int(np.sum(y_test_array == 1)),
        "phishing_prevalence": float(prevalence),
        "cap_score": float(cap_score),
        "model_area": float(model_area),
        "random_area": float(random_area),
        "ideal_area": float(ideal_area),
    }

    results_path = ARTIFACT_DIR / "cap_results.json"

    results_path.write_text(
        json.dumps(
            cap_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            cap_results,
            indent=2,
        )
    )

    print()
    print(f"CAP curve saved to: {output_path}")
    print(f"CAP results saved to: {results_path}")


if __name__ == "__main__":
    generate_cap_curve()