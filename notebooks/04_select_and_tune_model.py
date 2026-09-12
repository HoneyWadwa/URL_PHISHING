"""Phase 4: tune, select, and save the final phishing-URL classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
PHISHING_LABEL = 0
TARGET = "ClassLabel"
URL_COLUMN = "URL"
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")


def make_pipeline(feature_columns):
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), feature_columns)
        ], remainder="drop")),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def test_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    phishing_probability = probabilities[:, list(model.classes_).index(PHISHING_LABEL)]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "phishing_precision": precision_score(y_test, predictions, pos_label=PHISHING_LABEL),
        "phishing_recall": recall_score(y_test, predictions, pos_label=PHISHING_LABEL),
        "phishing_f1": f1_score(y_test, predictions, pos_label=PHISHING_LABEL),
        "roc_auc": roc_auc_score((y_test == PHISHING_LABEL).astype(int), phishing_probability),
    }


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Run Phase 2 before Phase 4.")
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    all_features = [column for column in train.columns if column not in [URL_COLUMN, TARGET]]
    X_train, y_train = train[all_features], train[TARGET]
    X_test, y_test = test[all_features], test[TARGET]

    # A compact grid is enough for a college project and avoids needless tuning.
    parameter_grid = {
        "model__n_estimators": [200],
        "model__max_depth": [None, 20],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }
    cross_validation = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(all_features),
        parameter_grid,
        scoring=make_scorer(f1_score, pos_label=PHISHING_LABEL),
        cv=cross_validation,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_
    final_metrics = test_metrics(final_model, X_test, y_test)
    joblib.dump(final_model, MODEL_DIR / "final_random_forest.joblib")

    # Robustness check: retrain without a perfectly class-aligned feature.
    robust_features = [column for column in all_features if column != "has_ip_address"]
    robust_model = make_pipeline(robust_features)
    robust_model.set_params(**search.best_params_)
    robust_model.fit(train[robust_features], y_train)
    robust_metrics = test_metrics(robust_model, test[robust_features], y_test)

    forest = final_model.named_steps["model"]
    importance = pd.DataFrame({
        "feature": all_features,
        "gini_importance": forest.feature_importances_,
    }).sort_values("gini_importance", ascending=False)
    importance.to_csv(RESULT_DIR / "final_model_feature_importance.csv", index=False)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results[["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(10).to_csv(
        RESULT_DIR / "random_forest_tuning_results.csv", index=False
    )
    report = {
        "selection_method": "2-fold stratified cross-validation on training data; phishing F1 was optimized",
        "best_cross_validation_phishing_f1": search.best_score_,
        "best_parameters": search.best_params_,
        "final_model_features": all_features,
        "held_out_test_metrics": final_metrics,
        "robustness_check_without_has_ip_address": robust_metrics,
        "interpretation": "The robustness result estimates the dependency on has_ip_address. It is not a replacement for external-dataset evaluation.",
    }
    (RESULT_DIR / "phase_4_model_selection_report.json").write_text(json.dumps(report, indent=2))

    print("Phase 4 complete")
    print("Best CV phishing F1:", round(search.best_score_, 6))
    print("Best parameters:", search.best_params_)
    print("Final held-out test metrics:", {key: round(value, 6) for key, value in final_metrics.items()})
    print("No-IP robustness metrics:", {key: round(value, 6) for key, value in robust_metrics.items()})
    print("\nTop five features:")
    print(importance.head().to_string(index=False))


if __name__ == "__main__":
    main()
