import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR

def save_confusion_matrix(y_true, y_pred, labels, out_path, normalize=None, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=None, values_format=".2f" if normalize else "d", colorbar=True)
    ax.set_title(title if title else "Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed train/test...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_mapped.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_mapped.csv")

    print("Loading preprocessor + model...")
    artifact = joblib.load(MODELS_DIR / "preprocessor.pkl")
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]
    class_names = artifact["class_names"]

    model = joblib.load(MODELS_DIR / "rf_multiclass.pkl")

    X_test = test_df[feature_cols]
    y_test = test_df["class"]

    print("Transforming test features...")
    X_test_t = preprocessor.transform(X_test)

    print("Predicting...")
    y_pred = model.predict(X_test_t)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy:   {acc:.4f}")
    print(f"Macro P:    {macro_p:.4f}")
    print(f"Macro R:    {macro_r:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1:.4f}")

    # Classification report table
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = TABLES_DIR / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"\nSaved classification report -> {report_path}")

    # Overall metrics json
    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    metrics_path = TABLES_DIR / "overall_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved overall metrics -> {metrics_path}")

    # Confusion matrices (raw + normalized)
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    save_confusion_matrix(
        y_test, y_pred, class_names, cm_path,
        normalize=None, title="Confusion Matrix (Counts)"
    )
    print(f"Saved confusion matrix -> {cm_path}")

    cmn_path = FIGURES_DIR / "confusion_matrix_normalized.png"
    save_confusion_matrix(
        y_test, y_pred, class_names, cmn_path,
        normalize="true", title="Confusion Matrix (Normalized)"
    )
    print(f"Saved normalized confusion matrix -> {cmn_path}")

    # Top confusions table
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    conf_rows = []
    for i, true_label in enumerate(class_names):
        for j, pred_label in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                conf_rows.append((true_label, pred_label, int(cm[i, j])))
    conf_df = pd.DataFrame(conf_rows, columns=["true_class", "predicted_class", "count"])
    conf_df = conf_df.sort_values("count", ascending=False).reset_index(drop=True)
    conf_path = TABLES_DIR / "top_confusions.csv"
    conf_df.to_csv(conf_path, index=False)
    print(f"Saved top confusions -> {conf_path}")

    print("\n✅ Phase 4 completed successfully.")

if __name__ == "__main__":
    main()
