"""Phase 6: extract features from a raw URL and classify it offline."""

from pathlib import Path
import sys

# Permit direct execution with: .venv/bin/python notebooks/06_feature_extractor_demo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.feature_extractor import FEATURE_COLUMNS, extract_features, features_as_dataframe

MODEL_PATH = PROJECT_ROOT / "models/final_random_forest.joblib"
EXAMPLE_URL = "https://www.google.com"


def predict_url(url: str) -> dict:
    """Classify an input URL without opening it on the network."""
    model = joblib.load(MODEL_PATH)
    feature_frame = features_as_dataframe(url)
    probabilities = model.predict_proba(feature_frame)[0]
    phishing_index = list(model.classes_).index(0)
    phishing_probability = float(probabilities[phishing_index])
    prediction = "Phishing" if phishing_probability >= 0.5 else "Legitimate"
    return {
        "url": url,
        "prediction": prediction,
        "phishing_probability": phishing_probability,
        "features": extract_features(url),
    }


if __name__ == "__main__":
    result = predict_url(EXAMPLE_URL)
    print("URL:", result["url"])
    print("Prediction:", result["prediction"])
    print(f"Phishing probability: {result['phishing_probability']:.2%}")
    print("\nExtracted model features:")
    for name in FEATURE_COLUMNS:
        print(f"- {name}: {result['features'][name]}")
