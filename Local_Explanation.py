from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(page_title="Local Explanation • IDS Dashboard", layout="wide")
st.title("Local Explanation")
st.caption("Local SHAP explanation for a single network connection (saved example from Phase 5).")

IMG_WIDTH = 610  # consistent, professional sizing

def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, width=IMG_WIDTH)
    else:
        st.warning(f"Missing file: {path.name}")

st.markdown("### What this shows")
st.markdown(
    """
This waterfall plot explains **one individual prediction** made by the Random Forest model.

- The model starts from a **baseline prediction**.
- Each feature then **pushes the prediction up or down**.
- The final value represents the model’s output for this specific instance.
"""
)

st.markdown("---")

# Center the SHAP plot for visual balance
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    show_image(
        FIGURES_DIR / "shap_local_example.png",
        "Local SHAP Waterfall"
    )

