"""Local Flask interface for the phishing URL detector."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import shap
from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_COLUMNS, FEATURE_NAMES, features_as_dataframe

MODEL_PATH = PROJECT_ROOT / "models/final_random_forest.joblib"
PHISHING_LABEL = 0
MAX_URL_LENGTH = 2_048

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model.named_steps["model"])
phishing_class_index = list(model.classes_).index(PHISHING_LABEL)


def risk_level(phishing_probability: float) -> tuple[str, str]:
    if phishing_probability >= 0.70:
        return "High", "high"
    if phishing_probability >= 0.25:
        return "Medium", "medium"
    return "Low", "low"


def explain(feature_frame) -> list[dict]:
    transformed = model.named_steps["preprocessing"].transform(feature_frame)
    shap_values = explainer(transformed)
    contributions = shap_values.values[0, :, phishing_class_index]
    feature_names = [name.removeprefix("numeric__") for name in model.named_steps["preprocessing"].get_feature_names_out()]
    items = []
    for feature_name, contribution in zip(feature_names, contributions):
        items.append({
            "name": FEATURE_NAMES.get(feature_name, feature_name),
            "value": feature_frame.iloc[0][feature_name],
            "contribution": float(contribution),
            "effect": "Raises phishing risk" if contribution > 0 else "Reduces phishing risk",
        })
    return sorted(items, key=lambda item: abs(item["contribution"]), reverse=True)[:6]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_url = ""
    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()
        if len(submitted_url) > MAX_URL_LENGTH:
            error = f"Please enter a URL shorter than {MAX_URL_LENGTH:,} characters."
        else:
            try:
                feature_frame = features_as_dataframe(submitted_url)
                probabilities = model.predict_proba(feature_frame)[0]
                phishing_probability = float(probabilities[phishing_class_index])
                confidence = phishing_probability if phishing_probability >= 0.5 else 1 - phishing_probability
                risk, risk_style = risk_level(phishing_probability)
                result = {
                    "prediction": "Phishing" if phishing_probability >= 0.5 else "Legitimate",
                    "phishing_probability": phishing_probability * 100,
                    "confidence": confidence * 100,
                    "risk": risk,
                    "risk_style": risk_style,
                    "explanation": explain(feature_frame),
                    "features": [
                        {"name": FEATURE_NAMES.get(name, name), "value": feature_frame.iloc[0][name]}
                        for name in FEATURE_COLUMNS
                    ],
                }
            except ValueError as exception:
                error = str(exception)
    return render_template("index.html", result=result, error=error, submitted_url=submitted_url)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
