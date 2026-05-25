import pandas as pd

from src.config import TRAIN_FILE, TEST_FILE, HAS_DIFFICULTY_COL

# 41 feature names + label + (optional) difficulty_level
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

def load_nsl_kdd_txt(path: str) -> pd.DataFrame:
    # NSL-KDD txt is comma-separated
    df = pd.read_csv(path, header=None)

    # Some versions have an extra column (difficulty level)
    if df.shape[1] == len(NSL_KDD_COLUMNS) + 1:
        df.columns = NSL_KDD_COLUMNS + ["difficulty_level"]
    elif df.shape[1] == len(NSL_KDD_COLUMNS):
        df.columns = NSL_KDD_COLUMNS
    else:
        raise ValueError(
            f"Unexpected number of columns in {path}. "
            f"Got {df.shape[1]} columns; expected {len(NSL_KDD_COLUMNS)} or {len(NSL_KDD_COLUMNS)+1}."
        )
    return df

def main():
    print("Loading datasets...")
    train_df = load_nsl_kdd_txt(TRAIN_FILE)
    test_df = load_nsl_kdd_txt(TEST_FILE)

    print("\n--- BASIC INFO ---")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    print("\n--- COLUMNS ---")
    print(train_df.columns.tolist())

    print("\n--- LABEL CHECK ---")
    print("Train labels (top 20):")
    print(train_df["label"].value_counts().head(20))

    print("\nTest labels (top 20):")
    print(test_df["label"].value_counts().head(20))

    if "difficulty_level" in train_df.columns:
        print("\nDetected difficulty_level column ✅")
    else:
        print("\nNo difficulty_level column detected ✅")

    print("\n--- MISSING VALUES (Train) ---")
    missing = train_df.isna().sum()
    missing = missing[missing > 0]
    print(missing if not missing.empty else "No missing values found.")

if __name__ == "__main__":
    main()
