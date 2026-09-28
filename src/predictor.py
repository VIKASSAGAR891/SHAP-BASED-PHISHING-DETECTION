"""Inference service used by the Flask phishing detection dashboard."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_PATH
from src.features import extract_url_features, explain_feature
from src.explain import explain_prediction


class Predictor:
    """Load the trained model and perform URL-level inference."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.bundle = None
        self._active_model_name = None

        if self.model_path.exists():
            self.bundle = joblib.load(self.model_path)
            self._active_model_name = self.bundle.get("model_name")

    @property
    def ready(self) -> bool:
        """Return whether the trained model is available."""
        return self.bundle is not None

    @property
    def model_name(self) -> str | None:
        """Return the currently selected model name."""
        return self._active_model_name if self.ready else None

    @property
    def available_models(self) -> list[str]:
        """Return the model names available in the loaded bundle."""
        if not self.bundle:
            return []

        models = self.bundle.get("models")
        if isinstance(models, dict) and models:
            return list(models)

        model_name = self.bundle.get("model_name")
        return [model_name] if model_name else []

    def select_model(self, model_name: str) -> str:
        """Select a trained model for subsequent predictions."""
        if model_name not in self.available_models:
            raise ValueError(
                f"Model must be one of: {', '.join(self.available_models)}."
            )

        self._active_model_name = model_name
        return model_name

    def _active_model(self):
        models = self.bundle.get("models") if self.bundle else None
        if isinstance(models, dict) and self.model_name in models:
            return models[self.model_name]
        return self.bundle["model"]

    def reload(self):
        """Reload the trained model from disk."""
        self.bundle = (
            joblib.load(self.model_path)
            if self.model_path.exists()
            else None
        )
        available_models = self.available_models
        if self._active_model_name not in available_models:
            self._active_model_name = (
                self.bundle.get("model_name")
                if self.bundle
                else None
            )

    # ------------------------------------------------------------------
    # URL validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_url(url: str) -> tuple[bool, str]:
        """Validate the URL submitted by the dashboard."""

        value = str(url or "").strip()

        if not value:
            return False, "Enter a URL to analyse."

        if len(value) > 2048:
            return (
                False,
                "The URL is longer than the supported 2048-character limit.",
            )

        candidate = value

        if not re.match(
            r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
            candidate,
        ):
            candidate = "https://" + candidate

        try:
            parsed = urlsplit(candidate)

            if parsed.scheme not in {
                "http",
                "https",
            }:
                return (
                    False,
                    "Only HTTP and HTTPS URLs are supported.",
                )

            if not parsed.hostname:
                return (
                    False,
                    "Enter a valid URL with a hostname.",
                )

            return True, ""

        except ValueError:
            return (
                False,
                "The URL could not be parsed safely.",
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, url: str) -> dict:
        """Analyse one URL using the trained XGBoost model."""

        if not self.ready:
            raise RuntimeError(
                "No trained model found. "
                "Run scripts/run_training.py first."
            )

        # --------------------------------------------------------------
        # 1. Extract URL features
        # --------------------------------------------------------------

        features = extract_url_features(url)

        frame = pd.DataFrame(
            [features],
            columns=self.bundle["feature_columns"],
        ).fillna(0)

        # --------------------------------------------------------------
        # 2. Apply the same feature selector used during training
        # --------------------------------------------------------------

        selected = self.bundle[
            "selector"
        ].transform(frame)

        selected = pd.DataFrame(
            selected,
            columns=self.bundle[
                "selected_features"
            ],
        )

        # --------------------------------------------------------------
        # 3. Run the selected model
        # --------------------------------------------------------------

        model = self._active_model()

        prediction = int(
            np.asarray(
                model.predict(selected)
            ).ravel()[0]
        )

        probabilities = model.predict_proba(
            selected
        )[0]

        classes = [
            int(c)
            for c in model.classes_
        ]

        phishing_index = classes.index(0)
        legitimate_index = classes.index(1)

        phishing_probability = float(
            probabilities[phishing_index]
        )

        legitimate_probability = float(
            probabilities[legitimate_index]
        )

        # Label mapping established during training:
        #
        # 0 = phishing
        # 1 = legitimate

        result = (
            "PHISHING"
            if prediction == 0
            else "LEGITIMATE"
        )

        # IMPORTANT:
        # Risk score always represents the probability of phishing,
        # regardless of the predicted class.

        risk_score = round(
            phishing_probability * 100,
            2,
        )

        # --------------------------------------------------------------
        # 4. SHAP explanation
        # --------------------------------------------------------------

        shap_pairs = explain_prediction(
            model=model,
            selected=selected,
            feature_names=self.bundle[
                "selected_features"
            ],
            row_index=0,
            predicted_class=prediction,
        )

        explanation = []

        for item in shap_pairs:

            if isinstance(item, dict):

                feature = item["feature"]
                impact = float(
                    item["impact"]
                )

            else:

                feature, impact = item

                feature = str(feature)
                impact = float(impact)

            explanation.append(
                {
                    "feature": feature,
                    "description": explain_feature(
                        feature
                    ),
                    "impact": impact,
                    "direction": (
                        "supports prediction"
                        if impact > 0
                        else "opposes prediction"
                    ),
                    "raw_value": float(
                        features.get(
                            feature,
                            0,
                        )
                    ),
                }
            )

        # Strongest contributors first.
        explanation.sort(
            key=lambda item: abs(
                item["impact"]
            ),
            reverse=True,
        )

        # Keep the dashboard compact.
        top_explanation = explanation[:5]

        # --------------------------------------------------------------
        # 5. Human-readable URL indicators
        # --------------------------------------------------------------

        indicators = []

        if features["URLLength"] >= 100:
            indicators.append(
                "Unusually long URL"
            )

        if features["NoOfSubDomain"] >= 3:
            indicators.append(
                "Multiple subdomain levels"
            )

        if features["HasObfuscation"]:
            indicators.append(
                "URL obfuscation pattern"
            )

        if features["IsDomainIP"]:
            indicators.append(
                "IP address used as hostname"
            )

        if (
            features["NoOfQMarkInURL"] >= 2
            or features["NoOfAmpersandInURL"] >= 4
        ):
            indicators.append(
                "Complex query string"
            )

        if features[
            "SpacialCharRatioInURL"
        ] >= 0.12:
            indicators.append(
                "High special-character ratio"
            )

        if features["IsHTTPS"] == 0:
            indicators.append(
                "HTTPS not detected"
            )

        if not indicators:
            indicators = [
                f"{item['feature']} influenced the model prediction"
                for item in explanation[:3]
                if item["direction"] == "supports prediction"
            ]

        if not indicators:
            indicators.append(
                "The prediction was based on the combined URL feature pattern"
            )

        # --------------------------------------------------------------
        # 6. Dashboard-ready response
        # --------------------------------------------------------------

        return {
            "url": url,

            "result": result,

            "risk_score": risk_score,

            "phishing_probability": round(
                phishing_probability * 100,
                2,
            ),

            "legitimate_probability": round(
                legitimate_probability * 100,
                2,
            ),

            "model_name": self.model_name,

            "indicators": indicators,

            "explanation": top_explanation,

            "features": features,

            "selected_features": list(
                self.bundle[
                    "selected_features"
                ]
            ),
        }