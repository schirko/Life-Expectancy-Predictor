import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard | Life Expectancy Explorer", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data.csv")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("📊 Dashboard")

if not os.path.exists(DATA_PATH):
    st.error("No cleaned dataset found. Run `python train_model.py` from the project folder first.")
    st.stop()

df = load_data()

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
col3.metric("Avg. life expectancy", f"{filtered.Life_Expectancy.mean():.1f} yrs")

st.divider()

# ---- Trend over time ----
st.subheader("Life expectancy over time")
trend = filtered.groupby(["Year", "Status"], as_index=False).Life_Expectancy.mean()
fig_trend = px.line(
    trend,
    x="Year",
    y="Life_Expectancy",
    color="Status",
    markers=True,
    labels={"Life_Expectancy": "Avg. Life Expectancy"},
)
st.plotly_chart(fig_trend, use_container_width=True)

col_left, col_right = st.columns(2)

# ---- Rankings ----
with col_left:
    st.subheader("Country rankings")
    rank_stat = st.radio("Rank by", ["Mean", "Median"], horizontal=True)
    top_n = st.slider("Number of countries", 5, 20, 10)

    agg_func = "mean" if rank_stat == "Mean" else "median"
    country_stats = (
        filtered.groupby("Country").Life_Expectancy.agg(agg_func).sort_values(ascending=False)
    )

    tab_top, tab_bottom = st.tabs(["Top", "Bottom"])
    with tab_top:
        top_df = country_stats.head(top_n).reset_index()
        fig_top = px.bar(top_df, x="Life_Expectancy", y="Country", orientation="h")
        fig_top.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)
    with tab_bottom:
        bottom_df = country_stats.tail(top_n).sort_values(ascending=True).reset_index()
        fig_bottom = px.bar(bottom_df, x="Life_Expectancy", y="Country", orientation="h")
        st.plotly_chart(fig_bottom, use_container_width=True)

# ---- Feature importance ----
with col_right:
    st.subheader("What drives the model's predictions")
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
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
st.subheader("Correlation between features")
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
st.subheader("Distribution of life expectancy")
fig_hist = px.histogram(filtered, x="Life_Expectancy", nbins=40, color="Status", marginal="box")
st.plotly_chart(fig_hist, use_container_width=True)
