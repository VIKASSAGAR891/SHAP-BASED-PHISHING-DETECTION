"""
Generate XGBoost training-vs-validation accuracy curve.

This script is used only for model-development evidence.
It does not overwrite the final trained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ARTIFACT_DIR,
    DATASET_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_K_FEATURES,
)
from src.data import load_dataset  # noqa: E402


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def generate_learning_curve() -> None:
    print("Loading dataset...")

    X, y = load_dataset(str(DATASET_PATH))

    print(f"Dataset rows: {len(X)}")
    print(f"Original features: {X.shape[1]}")

    # -------------------------------------------------------------
    # Split into training and validation data.
    #
    # This curve is independent of the final held-out test set
    # used in the main evaluation pipeline.
    # -------------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------------------------------------------------
    # Feature selection.
    #
    # Fit selector ONLY on training data to avoid data leakage.
    # -------------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=min(TOP_K_FEATURES, X_train.shape[1]),
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)

    selected_features = X.columns[selector.get_support()].tolist()

    print(f"Selected features: {len(selected_features)}")
    print("Features:")
    for feature in selected_features:
        print(f"  - {feature}")

    # -------------------------------------------------------------
    # XGBoost configuration.
    #
    # These settings match the existing project training pipeline.
    # -------------------------------------------------------------

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric=["logloss", "error"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    print("Recording training and validation performance...")

    model.fit(
        X_train_selected,
        y_train,
        eval_set=[
            (X_train_selected, y_train),
            (X_val_selected, y_val),
        ],
        verbose=False,
    )

    # -------------------------------------------------------------
    # Retrieve evaluation history.
    #
    # XGBoost's "error" metric is classification error:
    #
    # accuracy = 1 - error
    # -------------------------------------------------------------

    history = model.evals_result()

    train_error = np.asarray(history["validation_0"]["error"])
    val_error = np.asarray(history["validation_1"]["error"])

    train_accuracy = (1.0 - train_error) * 100
    val_accuracy = (1.0 - val_error) * 100

    iterations = np.arange(1, len(train_accuracy) + 1)

    # -------------------------------------------------------------
    # Find best validation iteration.
    # -------------------------------------------------------------

    best_index = int(np.argmax(val_accuracy))

    best_iteration = int(iterations[best_index])
    best_val_accuracy = float(val_accuracy[best_index])
    corresponding_train_accuracy = float(train_accuracy[best_index])

    # -------------------------------------------------------------
    # Save learning curve.
    # -------------------------------------------------------------

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ARTIFACT_DIR / "xgboost_training_validation_accuracy.png"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.plot(
        iterations,
        train_accuracy,
        linewidth=2.2,
        label="Training Accuracy",
    )

    ax.plot(
        iterations,
        val_accuracy,
        linewidth=2.2,
        label="Validation Accuracy",
    )

    ax.set_title(
        "XGBoost Training and Validation Accuracy",
        fontsize=15,
        pad=12,
    )

    ax.set_xlabel(
        "Boosting Iterations",
        fontsize=11,
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=11,
    )

    ax.set_ylim(
        max(90, min(train_accuracy.min(), val_accuracy.min()) - 1),
        100.2,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
    )

    ax.text(
        0.02,
        0.03,
        (
            f"Best validation accuracy: {best_val_accuracy:.2f}% "
            f"at iteration {best_iteration}"
        ),
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Print summary.
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("LEARNING CURVE GENERATED")
    print("=" * 60)

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Selected features:  {len(selected_features)}")
    print(f"Boosting iterations: {len(iterations)}")

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}%"
    )

    print(
        f"Training accuracy at that point: "
        f"{corresponding_train_accuracy:.4f}%"
    )

    print(
        f"\nSaved graph to:\n{output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    generate_learning_curve()