"""Phase 2: clean the LegitPhish data and create reproducible train/test sets.

The original CSV is never changed. Output files are written to data/processed/.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
RAW_DATA_PATH = Path("data/url_features_extracted1.csv")
OUTPUT_DIR = Path("data/processed")
TARGET = "ClassLabel"
URL_COLUMN = "URL"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_DATA_PATH)
    starting_rows = len(raw)

    # A missing target cannot be used for supervised learning.
    cleaned = raw.dropna(subset=[TARGET]).copy()
    rows_without_target = starting_rows - len(cleaned)
    cleaned[TARGET] = cleaned[TARGET].astype("int64")

    # Remove only exact repeats. Rows with equal engineered features but a
    # different URL are deliberately retained: they are distinct observations.
    exact_duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    feature_columns = [
        column for column in cleaned.columns if column not in [URL_COLUMN, TARGET]
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned.drop(columns=[TARGET]),
        cleaned[TARGET],
        test_size=TEST_SIZE,
        stratify=cleaned[TARGET],
        random_state=RANDOM_STATE,
    )

    train = X_train.copy()
    train[TARGET] = y_train
    test = X_test.copy()
    test[TARGET] = y_test
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    cleaned.to_csv(OUTPUT_DIR / "legitphish_clean.csv", index=False)

    # The secondary feature set is for a robustness comparison in Phase 3.
    # `has_ip_address` maps perfectly to phishing in this dataset, so a model
    # using it may score excellently here but generalize poorly elsewhere.
    manifest = {
        "source_file": str(RAW_DATA_PATH),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target": TARGET,
        "url_column_excluded_from_model": URL_COLUMN,
        "rows_in_raw_data": starting_rows,
        "rows_removed_missing_target": rows_without_target,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "rows_after_cleaning": len(cleaned),
        "train_rows": len(train),
        "test_rows": len(test),
        "class_counts_after_cleaning": {
            str(label): int(count)
            for label, count in cleaned[TARGET].value_counts().sort_index().items()
        },
        "feature_sets": {
            "all_features": feature_columns,
            "without_ip_address_for_robustness": [
                column for column in feature_columns if column != "has_ip_address"
            ],
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Phase 2 complete")
    print(f"Raw rows: {starting_rows:,}")
    print(f"Removed missing targets: {rows_without_target:,}")
    print(f"Removed exact duplicates: {exact_duplicate_rows:,}")
    print(f"Clean rows: {len(cleaned):,}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print("\nTrain class counts:")
    print(train[TARGET].value_counts().sort_index())
    print("\nTest class counts:")
    print(test[TARGET].value_counts().sort_index())


if __name__ == "__main__":
    main()
