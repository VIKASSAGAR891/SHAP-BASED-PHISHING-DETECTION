from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.predictor import Predictor

app = Flask(__name__)
predictor = Predictor()


@app.get("/")
def index():
    return render_template(
        "index.html",
        ready=predictor.ready,
        model_name=predictor.model_name,
        available_models=predictor.available_models,
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": predictor.ready,
        "model": predictor.model_name,
        "analysis_mode": "URL-only / pre-fetch",
    })


@app.post("/api/analyse")
def analyse():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()
    valid, message = predictor.validate_url(url)
    if not valid:
        return jsonify({"error": message}), 400

    try:
        result = predictor.predict(url)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    return jsonify(result)


@app.post("/api/model")
def select_model():
    payload = request.get_json(silent=True) or {}
    model_name = str(payload.get("model", "")).strip()

    try:
        selected = predictor.select_model(model_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "model": selected,
        "available_models": predictor.available_models,
    })


@app.post("/api/reload")
def reload_model():
    predictor.reload()
    return jsonify({"model_ready": predictor.ready, "model": predictor.model_name})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
