"""Phase 3: train and compare baseline phishing-URL classifiers."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models/baseline")
RESULT_DIR = Path("results")
TARGET = "ClassLabel"
EXCLUDED_COLUMNS = ["URL", TARGET]
PHISHING_LABEL = 0


def make_pipeline(model, feature_columns, scale_features=False):
    """Keep preprocessing inside the pipeline to prevent train/test leakage."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    preprocessor = Pipeline(steps)
    return Pipeline([
        ("preprocessing", ColumnTransformer(
            [("numeric", preprocessor, feature_columns)], remainder="drop"
        )),
        ("model", model),
    ])


def evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    phishing_probability = probabilities[:, list(pipeline.classes_).index(PHISHING_LABEL)]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "phishing_precision": precision_score(y_test, predictions, pos_label=PHISHING_LABEL),
        "phishing_recall": recall_score(y_test, predictions, pos_label=PHISHING_LABEL),
        "phishing_f1": f1_score(y_test, predictions, pos_label=PHISHING_LABEL),
        "roc_auc": roc_auc_score((y_test == PHISHING_LABEL).astype(int), phishing_probability),
        "true_phishing_detected": int(matrix[0, 0]),
        "missed_phishing": int(matrix[0, 1]),
        "legitimate_misflagged": int(matrix[1, 0]),
        "legitimate_correct": int(matrix[1, 1]),
    }
    return pipeline, metrics, matrix


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Run notebooks/02_preprocess_and_split.py before this script.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [column for column in train.columns if column not in EXCLUDED_COLUMNS]
    X_train, y_train = train[feature_columns], train[TARGET]
    X_test, y_test = test[feature_columns], test[TARGET]

    candidates = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            feature_columns,
            scale_features=True,
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE), feature_columns
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            feature_columns,
        ),
    }

    all_metrics = []
    for name, pipeline in candidates.items():
        fitted, metrics, matrix = evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        filename = name.lower().replace(" ", "_")
        joblib.dump(fitted, MODEL_DIR / f"{filename}.joblib")
        pd.DataFrame(
            matrix,
            index=["actual_phishing", "actual_legitimate"],
            columns=["predicted_phishing", "predicted_legitimate"],
        ).to_csv(RESULT_DIR / f"{filename}_confusion_matrix.csv")
        all_metrics.append(metrics)
        print(f"Finished: {name}")

    comparison = pd.DataFrame(all_metrics).sort_values(
        ["phishing_f1", "roc_auc"], ascending=False
    )
    comparison.to_csv(RESULT_DIR / "baseline_model_comparison.csv", index=False)
    rounded = comparison.round(4)
    markdown_lines = [
        "| " + " | ".join(rounded.columns) + " |",
        "| " + " | ".join(["---"] * len(rounded.columns)) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rounded.itertuples(index=False, name=None)
    )
    (RESULT_DIR / "baseline_model_comparison.md").write_text("\n".join(markdown_lines) + "\n")
    print("\nBaseline comparison (phishing is the positive class):")
    print(comparison.round(4).to_string(index=False))
    print("\nPhase 3 complete. These are baseline results; Phase 4 will select and tune a final model.")


if __name__ == "__main__":
    main()
