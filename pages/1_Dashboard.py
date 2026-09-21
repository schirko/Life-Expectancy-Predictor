import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard | Life Expectancy Predictor", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data.csv")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_meta():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)


st.title("📊 Dashboard")

if not os.path.exists(DATA_PATH):
    st.error("No cleaned dataset found. Run `python train_model.py` from the project folder first.")
    st.stop()

df = load_data()
meta = load_meta()
# The target column name comes from the training config, so this page
# tracks whatever dataset/config train_model.py was last run with, rather
# than assuming "Life_Expectancy" specifically. Note: the Country/Year/Status
# filters below still assume that shape of identifying columns, since this
# dashboard was designed around WHO-style country-year panel data -- a
# genuinely different dataset shape would need its own dashboard filters.
target_col = meta.get("target_col", "Life_Expectancy") if meta else "Life_Expectancy"
target_label = target_col.replace("_", " ")

# ---- Sidebar filters ----
st.sidebar.header("Filters")
year_min, year_max = int(df.Year.min()), int(df.Year.max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

status_options = sorted(df.Status.unique().tolist())
selected_status = st.sidebar.multiselect("Country status", status_options, default=status_options)

# Only offer countries that actually match the selected status(es), so the
# two filters can't contradict each other.
available_countries = sorted(df[df.Status.isin(selected_status)].Country.unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries (optional, leave empty for all in the selected status)",
    available_countries,
    default=[],
)

filtered = df[
    (df.Year >= year_range[0])
    & (df.Year <= year_range[1])
    & (df.Status.isin(selected_status))
]
if selected_countries:
    filtered = filtered[filtered.Country.isin(selected_countries)]

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ---- Top metrics ----
col1, col2, col3 = st.columns(3)
col1.metric("Rows in view", f"{len(filtered):,}")
col2.metric("Countries in view", f"{filtered.Country.nunique():,}")
col3.metric(f"Avg. {target_label}", f"{filtered[target_col].mean():.1f} yrs")

st.divider()

# ---- Trend over time ----
st.subheader(f"{target_label} Over Time")
trend = filtered.groupby(["Year", "Status"], as_index=False)[target_col].mean()
fig_trend = px.line(
    trend,
    x="Year",
    y=target_col,
    color="Status",
    markers=True,
    labels={target_col: f"Avg. {target_label}"},
)
st.plotly_chart(fig_trend, use_container_width=True)

col_left, col_right = st.columns(2)

# ---- Rankings ----
with col_left:
    st.subheader("Country Rankings")
    rank_stat = st.radio("Rank by", ["Mean", "Median"], horizontal=True)
    top_n = st.slider("Number of countries", 5, 20, 10)

    agg_func = "mean" if rank_stat == "Mean" else "median"
    country_stats = (
        filtered.groupby("Country")[target_col].agg(agg_func).sort_values(ascending=False)
    )

    tab_top, tab_bottom = st.tabs(["Top", "Bottom"])
    with tab_top:
        top_df = country_stats.head(top_n).reset_index()
        fig_top = px.bar(top_df, x=target_col, y="Country", orientation="h")
        fig_top.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)
    with tab_bottom:
        bottom_df = country_stats.tail(top_n).sort_values(ascending=True).reset_index()
        fig_bottom = px.bar(bottom_df, x=target_col, y="Country", orientation="h")
        st.plotly_chart(fig_bottom, use_container_width=True)

# ---- Feature importance ----
with col_right:
    st.subheader("What Drives the Model's Predictions")
    if meta:
        imp_df = pd.DataFrame(
            list(meta["feature_importances"].items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("From the Random Forest model's built-in feature importances.")
    else:
        st.info("Train the model (`python train_model.py`) to see feature importances.")

st.divider()

# ---- Correlation heatmap ----
st.subheader("Correlation Between Features")
numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Year"]
corr = filtered[numeric_cols].corr()
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# ---- Distribution ----
st.subheader(f"Distribution of {target_label}")
fig_hist = px.histogram(filtered, x=target_col, nbins=40, color="Status", marginal="box")
st.plotly_chart(fig_hist, use_container_width=True)
