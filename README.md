# Explainable AI-Based Phishing URL Detection

This college AI project identifies a URL as **phishing** or **legitimate**, estimates confidence and risk, and later explains the result with SHAP.

## Current stage: Phase 1 — understand the dataset

The original dataset is at `data/url_features_extracted1.csv`. Phase 2 produces
cleaned data and an 80/20 stratified split in `data/processed/`; the original is
never changed.

## Project roadmap

1. Dataset analysis and cleaning
2. Preprocessing and stratified train/test split
3. Train and compare baseline models
4. Select and save the best model
5. Add SHAP explanations
6. Build a feature extractor for newly entered URLs
7. Build the Flask web app
8. Test and document the final project

## Phase 2 outputs

Run the following from this project folder:

```bash
.venv/bin/python notebooks/02_preprocess_and_split.py
```

It removes the row with no label and exact duplicate rows, then creates
`train.csv`, `test.csv`, `legitphish_clean.csv`, and `manifest.json`.

## Phase 3: baseline models

```bash
.venv/bin/python notebooks/03_train_baseline_models.py
```

This trains Logistic Regression, Decision Tree, and Random Forest. It saves
the candidate model pipelines in `models/baseline/` and evaluation outputs in
`results/`. In all metrics, phishing (`ClassLabel = 0`) is treated as the
positive class, because failing to detect phishing is the more important error.

## Phase 4: final model selection

```bash
.venv/bin/python notebooks/04_select_and_tune_model.py
```

This uses three-fold cross-validation on the training data to tune Random
Forest for phishing F1-score, evaluates the chosen model on the held-out test
set, and saves it as `models/final_random_forest.joblib`. It also performs a
robustness check excluding `has_ip_address`, a dataset-specific feature.

## Phase 5: SHAP explanations

```bash
.venv/bin/python notebooks/05_shap_explanations.py
```

This creates global feature importance and local explanations for representative
phishing and legitimate predictions in `results/shap/`. A positive SHAP value
pushes the model toward phishing; a negative one pushes it toward legitimate.

## Phase 6: raw-URL feature extraction

```bash
.venv/bin/python notebooks/06_feature_extractor_demo.py
```

`src/feature_extractor.py` safely parses a raw URL without visiting it and
returns the 16 model columns in the exact training order. Its documented lexical
rules were validated against a 1,000-URL sample from the dataset; the web app
uses this extractor for every user input.

## Phase 7: Flask web application

```bash
.venv/bin/python app/app.py
```

Then open [http://127.0.0.1:5050](http://127.0.0.1:5050) in your browser. The
page provides the prediction, confidence, risk band, live SHAP explanation, and
the features extracted from the submitted URL. It does not visit submitted URLs.

## Install the Python packages

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`ClassLabel` is the target: `0` means phishing and `1` means legitimate. The
actual CSV has 16 engineered predictors, plus `URL` and `ClassLabel`.
