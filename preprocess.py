import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    PROCESSED_DATA_DIR,
    CATEGORICAL_COLS,
    RANDOM_SEED,
    ATTACK_CATEGORY_MAP,
    CLASS_NAMES,
)

# Same base columns from data_loader.py (features + label)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label"
]

def load_nsl_kdd_with_headers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # 43 columns => base + difficulty
    if df.shape[1] == len(NSL_KDD_COLUMNS) + 1:
        df.columns = NSL_KDD_COLUMNS + ["difficulty_level"]
    elif df.shape[1] == len(NSL_KDD_COLUMNS):
        df.columns = NSL_KDD_COLUMNS
    else:
        raise ValueError(f"Unexpected columns in {path}: got {df.shape[1]}")
    return df

def map_to_five_classes(raw_label: str) -> str:
    raw_label = str(raw_label).strip()
    if raw_label == "normal":
        return "Normal"
    # everything else must be in the attack map
    mapped = ATTACK_CATEGORY_MAP.get(raw_label)
    if mapped is None:
        return "__UNMAPPED__"
    return mapped

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw train/test...")
    train_df = load_nsl_kdd_with_headers(TRAIN_FILE)
    test_df = load_nsl_kdd_with_headers(TEST_FILE)

    # Drop difficulty level if present
    if "difficulty_level" in train_df.columns:
        train_df = train_df.drop(columns=["difficulty_level"])
    if "difficulty_level" in test_df.columns:
        test_df = test_df.drop(columns=["difficulty_level"])

    print("Mapping labels to 5 classes...")
    train_df["class"] = train_df["label"].apply(map_to_five_classes)
    test_df["class"] = test_df["label"].apply(map_to_five_classes)

    # Check unmapped labels
    unmapped_train = train_df[train_df["class"] == "__UNMAPPED__"]["label"].unique()
    unmapped_test = test_df[test_df["class"] == "__UNMAPPED__"]["label"].unique()

    if len(unmapped_train) > 0 or len(unmapped_test) > 0:
        print("\n❌ Found unmapped attack labels. Add these to ATTACK_CATEGORY_MAP in config.py:")
        if len(unmapped_train) > 0:
            print("Unmapped in TRAIN:", sorted(unmapped_train))
        if len(unmapped_test) > 0:
            print("Unmapped in TEST:", sorted(unmapped_test))
        raise SystemExit("Stop: fix mapping first.")

    # Print class distribution (important for Chapter 4 later)
    print("\n--- CLASS DISTRIBUTION (TRAIN) ---")
    print(train_df["class"].value_counts())

    print("\n--- CLASS DISTRIBUTION (TEST) ---")
    print(test_df["class"].value_counts())

    # Build preprocessing transformer (same one used later by model + streamlit)
    feature_cols = [c for c in train_df.columns if c not in ["label", "class"]]

    X_train = train_df[feature_cols]
    y_train = train_df["class"]

    # Identify numeric columns automatically
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    print("\nFitting preprocessor on training data...")
    preprocessor.fit(X_train)

    # Save processed datasets 
    train_out = PROCESSED_DATA_DIR / "train_mapped.csv"
    test_out = PROCESSED_DATA_DIR / "test_mapped.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Saved mapped train -> {train_out}")
    print(f"Saved mapped test  -> {test_out}")

    # Save preprocessor artifact
    artifact = {
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "class_names": CLASS_NAMES,
    }
    joblib.dump(artifact, "models/preprocessor.pkl")
    print("Saved preprocessor -> models/preprocessor.pkl")

    # Export class distribution table for report
    dist_df = pd.DataFrame({
        "train": train_df["class"].value_counts(),
        "test": test_df["class"].value_counts(),
    }).fillna(0).astype(int)
    dist_path = "reports/tables/class_distribution.csv"
    pd.DataFrame(dist_df).to_csv(dist_path)
    print(f"Saved class distribution -> {dist_path}")

    print("\n✅ Phase 2 completed successfully.")

if __name__ == "__main__":
    main()
