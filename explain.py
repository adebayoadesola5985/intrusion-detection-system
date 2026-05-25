import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def normalize_multiclass_shap(shap_values, n_classes: int):
    """
    Return a list of length n_classes where each entry is (n_samples, n_features).
    Handles SHAP returning:
      - list of arrays
      - array of shape (n_samples, n_features, n_classes)
      - array of shape (n_classes, n_samples, n_features)
    """
    if isinstance(shap_values, list):
        return shap_values

    sv = np.array(shap_values)

    if sv.ndim == 3:
        # (n_samples, n_features, n_classes)
        if sv.shape[2] == n_classes:
            return [sv[:, :, i] for i in range(n_classes)]
        # (n_classes, n_samples, n_features)
        if sv.shape[0] == n_classes:
            return [sv[i, :, :] for i in range(n_classes)]

    raise ValueError(f"Unsupported SHAP values shape: {sv.shape}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed test data...")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_mapped.csv")

    print("Loading model and preprocessor...")
    artifact = joblib.load(MODELS_DIR / "preprocessor.pkl")
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]
    class_names = artifact["class_names"]

    model = joblib.load(MODELS_DIR / "rf_multiclass.pkl")

    X_test = test_df[feature_cols]
    print("Transforming test features...")
    X_test_t = preprocessor.transform(X_test)

    # --- Sample for SHAP ---
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = X_test_t.shape[0]

    if n_rows > SAMPLE_SIZE:
        idx_sample = rng.choice(n_rows, SAMPLE_SIZE, replace=False)
        X_shap = X_test_t[idx_sample]
        print(f"Using SHAP sample: {SAMPLE_SIZE} / {n_rows} rows")
    else:
        idx_sample = np.arange(n_rows)  # ensure idx_sample exists
        X_shap = X_test_t
        print(f"Using full set for SHAP: {n_rows} rows")

    # --- Check class coverage in SHAP sample ---
    candidate_label_cols = [c for c in test_df.columns if c not in feature_cols]
    print("\nNon-feature columns (label should be here):", candidate_label_cols)

    preferred = ["label", "mapped_label", "class", "target", "y"]
    label_col = next((c for c in preferred if c in test_df.columns), None)

    if label_col is None:
        if not candidate_label_cols:
            raise ValueError("No label column found. test_df has only feature columns.")
        label_col = candidate_label_cols[0]

    print("Using label column:", label_col)

    y_sample = test_df.iloc[idx_sample][label_col]
    print("\nSHAP sample class distribution:")
    print(y_sample.value_counts())

    print("\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print(f"Computing SHAP values on {X_shap.shape[0]} samples...")
    raw_shap_values = explainer.shap_values(X_shap)

    # Normalize to list-of-arrays format
    shap_values_by_class = normalize_multiclass_shap(raw_shap_values, n_classes=len(class_names))

    feature_names = preprocessor.get_feature_names_out()

    # --- GLOBAL SUMMARY ---
    # Save one combined "overall" by averaging absolute SHAP across classes
    print("Saving global SHAP summary (mean |SHAP| across classes)...")
    mean_abs = np.mean([np.abs(sv) for sv in shap_values_by_class], axis=0)  # (n_samples, n_features)

    plt.figure()
    shap.summary_plot(
        mean_abs,
        X_shap,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Summary — Mean |SHAP| Across Classes")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_overall.png", dpi=200)
    plt.close()

    # --- PER-CLASS SHAP SUMMARY ---
    for i, class_name in enumerate(class_names):
        print(f"Saving SHAP summary for class: {class_name}")
        plt.figure()
        shap.summary_plot(
            shap_values_by_class[i],
            X_shap,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Summary — {class_name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_summary_{class_name}.png", dpi=200)
        plt.close()

    # --- LOCAL EXPLANATION (single instance from sample) ---
    print("Generating local explanation example...")
    local_idx = rng.integers(0, X_shap.shape[0])

    predicted_class = model.predict(X_shap[local_idx].reshape(1, -1))[0]
    class_index = class_names.index(predicted_class)

    # expected_value can be a list/array for multi-class
    expected = explainer.expected_value
    base_value = expected[class_index] if isinstance(expected, (list, np.ndarray)) else expected

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_by_class[class_index][local_idx],
            base_values=base_value,
            data=X_shap[local_idx],
            feature_names=feature_names,
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_local_example.png", dpi=200)
    plt.close()

    print(f"Local explanation saved (Predicted: {predicted_class})")
    print("\n✅ Phase 5 completed successfully.")


if __name__ == "__main__":
    main()