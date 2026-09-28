"""Training pipeline for XGBoost and CatBoost using URL-derived features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    ARTIFACT_DIR,
    FEATURES_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_K_FEATURES,
)
from src.data import load_dataset


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }


def train(dataset_path: str, sample_size: int | None = None) -> dict:
    X, y = load_dataset(dataset_path)

    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    selector = SelectKBest(score_func=f_classif, k=min(TOP_K_FEATURES, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = {
        str(name): float(score) if np.isfinite(score) else 0.0
        for name, score in zip(X.columns, selector.scores_)
    }

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            depth=7,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        ),
    }

    results = {}
    fitted = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_selected, y_train)
        pred = model.predict(X_test_selected).astype(int).ravel()
        results[name] = _metrics(y_test, pred)
        fitted[name] = model
        predictions[name] = pred

    # Positive class for evaluation is phishing (label 0). Select the model by F1,
    # then recall, so missing phishing cases is treated as especially important.
    best_name = sorted(
        results,
        key=lambda n: (results[n]["f1"], results[n]["recall"]),
        reverse=True,
    )[0]

    bundle = {
        "model": fitted[best_name],
        "models": fitted,
        "selector": selector,
        "feature_columns": list(X.columns),
        "selected_features": selected_features,
        "feature_scores": feature_scores,
        "model_name": best_name,
        "label_mapping": {"0": "phishing", "1": "legitimate"},
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, MODEL_PATH)

    # Save held-out confusion matrix for the selected model.
    cm = confusion_matrix(y_test, predictions[best_name], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Legitimate"]).plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    metrics = {
        "dataset_rows_used": int(len(X)),
        "feature_count_before_selection": int(X.shape[1]),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "feature_scores": feature_scores,
        "models": results,
        "selected_model": best_name,
        "label_mapping": {"0": "phishing", "1": "legitimate"},
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    FEATURES_PATH.write_text(json.dumps(selected_features, indent=2), encoding="utf-8")
    return metrics
