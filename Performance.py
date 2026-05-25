from pathlib import Path
import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(page_title="Performance • IDS Dashboard", layout="wide")
st.title("Model Performance")

def load_json(path: Path):
    return json.loads(path.read_text())

def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Missing file: {path.name}")

metrics_path = TABLES_DIR / "overall_metrics.json"
report_path = TABLES_DIR / "classification_report.csv"

# ---- Metrics ----
c1, c2, c3 = st.columns(3)
if metrics_path.exists():
    metrics = load_json(metrics_path)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
    c3.metric("Weighted F1", f"{metrics['weighted_f1']:.4f}")
else:
    st.warning("overall_metrics.json not found. Run: python -m src.evaluate")

st.markdown("---")

# ---- Classification Report ----
st.subheader("Classification report (per class)")
if report_path.exists():
    report_df = pd.read_csv(report_path)

    # ✅ Clean the common extra index column from CSV exports
    if "Unnamed: 0" in report_df.columns:
        report_df = report_df.rename(columns={"Unnamed: 0": "class"})

    # ✅ Make it look like a proper report table
    if "class" in report_df.columns:
        report_df = report_df.set_index("class")

    # ✅ Round numeric columns for neat display (safe, display-only)
    for col in ["precision", "recall", "f1-score"]:
        if col in report_df.columns:
            report_df[col] = pd.to_numeric(report_df[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(report_df[col]):
                report_df[col] = report_df[col].round(4)

    if "support" in report_df.columns:
        report_df["support"] = pd.to_numeric(report_df["support"], errors="ignore")

    st.dataframe(report_df, use_container_width=True, height=420)
else:
    st.warning("classification_report.csv not found. Run: python -m src.evaluate")

st.markdown("---")

# ---- Confusion Matrices ----
st.subheader("Confusion matrices")
colA, colB = st.columns(2)
with colA:
    show_image(FIGURES_DIR / "confusion_matrix.png", "Confusion Matrix (Counts)")
with colB:
    show_image(FIGURES_DIR / "confusion_matrix_normalized.png", "Confusion Matrix (Normalized)")
