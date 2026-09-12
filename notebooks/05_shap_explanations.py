"""Phase 5: create global and local SHAP explanations for the final model."""

import json
import os
from pathlib import Path

# Set these before importing SHAP/matplotlib so plot caches stay writable.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/phishing-url-matplotlib")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

MODEL_PATH = Path("models/final_random_forest.joblib")
TEST_PATH = Path("data/processed/test.csv")
RESULT_DIR = Path("results/shap")
TARGET = "ClassLabel"
URL_COLUMN = "URL"
PHISHING_LABEL = 0
LABEL_NAMES = {0: "Phishing", 1: "Legitimate"}

FEATURE_NAMES = {
    "url_length": "URL length",
    "has_ip_address": "IP address in URL",
    "dot_count": "Number of dots",
    "https_flag": "HTTPS used",
    "url_entropy": "URL entropy",
    "token_count": "URL token count",
    "subdomain_count": "Subdomain count",
    "query_param_count": "Query parameter count",
    "tld_length": "TLD length",
    "path_length": "Path length",
    "has_hyphen_in_domain": "Hyphen in domain",
    "number_of_digits": "Number of digits",
    "tld_popularity": "Popular TLD",
    "suspicious_file_extension": "Suspicious file extension",
    "domain_name_length": "Domain name length",
    "percentage_numeric_chars": "Numeric-character percentage",
}


def local_explanation(values, sample_index, phishing_class_index, sample, phishing_probability):
    """Return the strongest local SHAP effects in plain, report-friendly JSON."""
    contributions = values.values[sample_index, :, phishing_class_index]
    items = []
    for feature, contribution in zip(values.feature_names, contributions):
        clean_name = feature.removeprefix("numeric__")
        items.append({
            "feature": clean_name,
            "display_name": FEATURE_NAMES.get(clean_name, clean_name),
            "value": float(sample[clean_name]),
            "shap_contribution_to_phishing": float(contribution),
            "effect": "increased phishing prediction" if contribution > 0 else "decreased phishing prediction",
        })
    items.sort(key=lambda item: abs(item["shap_contribution_to_phishing"]), reverse=True)
    prediction = PHISHING_LABEL if phishing_probability >= 0.5 else 1
    return {
        "url": sample[URL_COLUMN],
        "actual_label": LABEL_NAMES[int(sample[TARGET])],
        "predicted_label": LABEL_NAMES[prediction],
        "phishing_probability": round(float(phishing_probability), 6),
        "top_contributing_features": items[:6],
    }


def save_waterfall(values, index, class_index, filename):
    plt.figure()
    shap.plots.waterfall(values[index, :, class_index], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run Phase 4 before creating SHAP explanations.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    full_test = pd.read_csv(TEST_PATH)
    # SHAP is computationally expensive. A reproducible 2,000-row sample gives
    # stable, readable explanations without needing every held-out record.
    test = full_test.sample(n=min(2_000, len(full_test)), random_state=42).reset_index(drop=True)
    feature_columns = [column for column in test.columns if column not in [URL_COLUMN, TARGET]]
    transformed_features = model.named_steps["preprocessing"].transform(test[feature_columns])
    transformed_names = [name.removeprefix("numeric__") for name in model.named_steps["preprocessing"].get_feature_names_out()]

    explainer = shap.TreeExplainer(model.named_steps["model"])
    values = explainer(transformed_features)
    values.feature_names = transformed_names
    phishing_class_index = list(model.classes_).index(PHISHING_LABEL)
    phishing_values = values[:, :, phishing_class_index]

    global_importance = pd.DataFrame({
        "feature": transformed_names,
        "display_name": [FEATURE_NAMES.get(name, name) for name in transformed_names],
        "mean_absolute_shap_value_for_phishing": abs(phishing_values.values).mean(axis=0),
    }).sort_values("mean_absolute_shap_value_for_phishing", ascending=False)
    global_importance.to_csv(RESULT_DIR / "global_shap_feature_importance.csv", index=False)

    shap.plots.bar(phishing_values, max_display=16, show=False)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "global_shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    probabilities = model.predict_proba(test[feature_columns])[:, phishing_class_index]
    predicted_phishing = test.index[probabilities >= 0.5]
    predicted_legitimate = test.index[probabilities < 0.5]
    phishing_example_index = int(predicted_phishing[probabilities[predicted_phishing].argmax()])
    legitimate_example_index = int(predicted_legitimate[probabilities[predicted_legitimate].argmin()])

    examples = {
        "phishing_example": local_explanation(values, phishing_example_index, phishing_class_index, test.iloc[phishing_example_index], probabilities[phishing_example_index]),
        "legitimate_example": local_explanation(values, legitimate_example_index, phishing_class_index, test.iloc[legitimate_example_index], probabilities[legitimate_example_index]),
        "note": "Positive SHAP values push the model toward phishing; negative values push it toward legitimate. SHAP explains this model's behaviour, not causal truth.",
    }
    (RESULT_DIR / "local_shap_explanations.json").write_text(json.dumps(examples, indent=2))
    save_waterfall(values, phishing_example_index, phishing_class_index, "phishing_example_waterfall.png")
    save_waterfall(values, legitimate_example_index, phishing_class_index, "legitimate_example_waterfall.png")

    print("Phase 5 complete")
    print("Top global SHAP features:")
    print(global_importance.head(8).to_string(index=False))
    print("\nSaved global and local SHAP plots to results/shap/")


if __name__ == "__main__":
    main()
