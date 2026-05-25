import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train_mapped.csv"
    test_path = PROCESSED_DATA_DIR / "test_mapped.csv"

    print("Loading processed train/test...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Load preprocessor artifact
    artifact = joblib.load(MODELS_DIR / "preprocessor.pkl")
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]

    X_train = train_df[feature_cols]
    y_train = train_df["class"]

    X_test = test_df[feature_cols]
    y_test = test_df["class"]

    print("Transforming features with saved preprocessor...")
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print("Training Random Forest (multi-class)...")
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",  # helps minority classes (U2R/R2L)
    )

    model.fit(X_train_t, y_train)

    # Quick sanity evaluation (full evaluation happens in Phase 4)
    train_acc = accuracy_score(y_train, model.predict(X_train_t))
    test_acc = accuracy_score(y_test, model.predict(X_test_t))

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    out_path = MODELS_DIR / "rf_multiclass.pkl"
    joblib.dump(model, out_path)
    print(f"Saved model -> {out_path}")

    print("\n✅ Phase 3 completed successfully.")

if __name__ == "__main__":
    main()
