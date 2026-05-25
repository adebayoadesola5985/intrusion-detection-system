from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(page_title="Global Explainability • IDS Dashboard", layout="wide")
st.title("Global Explainability (SHAP)")

IMG_WIDTH = 600  # keep your chosen size

def show_image(path: Path, caption: str, key: str):
    if path.exists():
        # Read bytes to avoid browser/streamlit caching issues
        img_bytes = path.read_bytes()
        st.image(img_bytes, caption=caption, width=IMG_WIDTH)
    else:
        st.warning(f"Missing file: {path.name}")

# ---- Overall plot ----
overall_path = FIGURES_DIR / "shap_summary_overall.png"
show_image(
    overall_path,
    "SHAP Summary — Mean |SHAP| Across Classes",
    key=f"overall_{overall_path.name}",
)

st.markdown("---")
st.subheader("Per-class SHAP summaries")

class_options = ["Normal", "DoS", "Probe", "R2L", "U2R"]
selected = st.selectbox("Choose class", class_options, index=1)

class_path = FIGURES_DIR / f"shap_summary_{selected}.png"

# Helpful: show which file is being loaded (for debugging)
st.caption(f"Showing: `{class_path.name}`")

show_image(
    class_path,
    f"SHAP Summary — {selected}",
    key=f"class_{selected}",
)
