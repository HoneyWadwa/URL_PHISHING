"""Phase 1: inspect the LegitPhish data before training a model."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/url_features_extracted1.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("../data/url_features_extracted1.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print(f"\nExact duplicate rows: {df.duplicated().sum():,}")

label_names = {0: "Phishing", 1: "Legitimate"}
counts = df["ClassLabel"].value_counts().sort_index()
summary = pd.DataFrame({
    "class_name": counts.index.map(label_names),
    "count": counts.values,
    "percentage": (counts.values / len(df) * 100).round(2),
})
print("\nClass distribution:\n", summary.to_string(index=False))

feature_columns = [column for column in df.columns if column not in ["URL", "ClassLabel"]]
print(f"\nPredictors ({len(feature_columns)}):\n", feature_columns)
print("\nFeature summary:\n", df[feature_columns].describe().T)

# An early leakage check: unusually near-perfect correlation deserves review.
correlations = df[feature_columns].corrwith(df["ClassLabel"]).abs().sort_values(ascending=False)
print("\nAbsolute correlation with ClassLabel:\n", correlations)

print("\nBinary-feature label checks (possible data artifacts/leakage):")
for column in ["https_flag", "has_ip_address"]:
    print(f"\n{column} by ClassLabel:\n", pd.crosstab(df[column], df["ClassLabel"]))

print("\nPhase 1 decision: remove the one row with a missing ClassLabel before Phase 2.")
print("Investigate the near-perfect binary-feature relationships before choosing final features.")
