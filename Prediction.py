from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "rf_multiclass.pkl"

st.set_page_config(page_title="Predict • IDS Dashboard", layout="wide")
st.title("Predict Intrusion Category")
st.caption(
    "Upload network traffic records (CSV). The system will classify each row into "
    "Normal / DoS / Probe / R2L / U2R."
)

@st.cache_resource
def load_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing: {PREPROCESSOR_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {MODEL_PATH}")

    pre_obj = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    # ✅ preprocessor.pkl might be a dict bundle or the transformer itself
    if isinstance(pre_obj, dict):
        # try common keys
        for key in ["preprocessor", "transformer", "ct", "column_transformer"]:
            if key in pre_obj:
                preprocessor = pre_obj[key]
                break
        else:
            raise ValueError(
                "preprocessor.pkl loaded as a dict, but no known preprocessor key was found. "
                f"Available keys: {list(pre_obj.keys())}"
            )
    else:
        preprocessor = pre_obj

    return preprocessor, model

def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep preprocessing consistent with training:
    - Drop known non-feature columns if present.
    - Do NOT change feature engineering.
    """
    df = df.copy()
    for col in ["label", "difficulty_level"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV containing the same feature columns used during training.")
    st.stop()

try:
    input_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(input_df.head(20), use_container_width=True)

# Load model + preprocessor
try:
    preprocessor, model = load_artifacts()
    st.caption(f"Loaded preprocessor type: `{type(preprocessor).__name__}`")  # ✅ quick confirmation
except Exception as e:
    st.error(str(e))
    st.stop()

# Prepare features
X = clean_input_df(input_df)

st.markdown("---")
run = st.button("Run prediction", type="primary")

if run:
    try:
        X_trans = preprocessor.transform(X)
        preds = model.predict(X_trans)

        out_df = input_df.copy()
        out_df["predicted_class"] = preds

        st.subheader("Predictions")
        st.dataframe(out_df.head(50), use_container_width=True)

        # Download results
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            data=csv_bytes,
            file_name="nids_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(
            "Prediction failed. Most common causes:\n"
            "- Uploaded CSV columns do not match the training feature columns\n"
            "- Categorical columns contain unseen categories\n\n"
            f"Error: {e}"
        )
