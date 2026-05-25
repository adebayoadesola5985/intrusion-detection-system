from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

st.set_page_config(page_title="Overview • IDS Dashboard", layout="wide")
st.title("Overview")

class_dist_path = TABLES_DIR / "class_distribution.csv"

# ✅ Cleaner header row: title left, toggle right (less “in your face”)
h1, h2 = st.columns([4, 1])
with h1:
    st.subheader("Dataset class distribution")
with h2:
    show_table = st.toggle("📋 Show table", value=False)

if class_dist_path.exists():
    dist_df = pd.read_csv(class_dist_path, index_col=0)

    # pick the count column safely
    numeric_cols = dist_df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        st.warning("class_distribution.csv has no numeric columns to display.")
    else:
        count_col = "count" if "count" in dist_df.columns else numeric_cols[0]
        counts = dist_df[count_col].copy()

        # IMPORTANT: only show ONE at a time
        if show_table:
            table_df = pd.DataFrame({
                "count": counts.astype(int),
                "percentage": (counts / counts.sum() * 100).round(2)
            }).sort_values("count", ascending=False)

            st.dataframe(table_df, use_container_width=True)
        else:
            # ✅ Replace st.bar_chart with Altair so axis labels don't protrude
            chart_df = counts.sort_values(ascending=False).reset_index()
            chart_df.columns = ["class", "count"]

            # ✅ Make the y-axis stop cleanly (add small headroom)
            y_max = int(chart_df["count"].max() * 1.10)

            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "class:N",
                        sort="-y",
                        axis=alt.Axis(title=None, labelAngle=90)
                    ),
                    y=alt.Y(
                        "count:Q",
                        scale=alt.Scale(domain=[0, y_max]),
                        axis=alt.Axis(
                            title="Number of samples",
                            format="~s",   # 10k, 20k, 30k
                            tickCount=6
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("class:N", title="Class"),
                        alt.Tooltip("count:Q", title="Count", format=",")
                    ],
                )
                .properties(height=420)
            )

            st.altair_chart(chart, use_container_width=True)

else:
    st.warning("class_distribution.csv not found. Run: python -m src.preprocess")
