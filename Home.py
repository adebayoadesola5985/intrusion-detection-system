import streamlit as st

st.set_page_config(
    page_title="Explainable IDS Dashboard",
    layout="wide",
)


# Global style polish (hide link icons, reduce padding, hide footer/menu)
st.markdown("""
<style>
a[data-testid="stHeaderActionElements"] {display: none;}
.block-container {padding-top: 2rem;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Explainable Network Intrusion Detection System (NIDS)")
st.caption("Multi-class IDS • Random Forest • SHAP Explainability")

st.markdown("")

# Quick “cards” / overview metrics (visual polish)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", "NSL-KDD")
c2.metric("Model", "Random Forest")
c3.metric("Explainability", "SHAP")
c4.metric("Dashboard", "Streamlit")

st.markdown("---")

st.subheader("How to use this dashboard")
st.markdown(
    """
- Use the **left sidebar** to switch pages.
- Start with **Overview** to inspect the dataset class imbalance.
- Use **Performance** to review metrics, the classification report, and confusion matrices.
- Use **Explainability** to explore global SHAP plots (overall + per-class).
- Use **Local Explanation** to view a saved SHAP waterfall for a single instance.
"""
)
