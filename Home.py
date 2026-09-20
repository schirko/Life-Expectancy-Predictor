import json
import os

import streamlit as st

st.set_page_config(
    page_title="Life Expectancy Explorer",
    page_icon="🌍",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

st.title("🌍 Life Expectancy Explorer")
st.markdown(
    """
Welcome! This app explores WHO life expectancy data from **2000–2015** across
~190 countries, and lets you predict life expectancy from a set of health
and economic indicators using a **Random Forest** model trained on that data.

Use the sidebar to navigate:

- **📊 Dashboard** — explore trends, correlations, and country rankings
- **🔮 Predict** — enter indicator values and get a predicted life expectancy

---
"""
)

meta_path = os.path.join(MODEL_DIR, "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model test R²", f"{meta['test_r2']:.3f}")
    col2.metric("Test MAE (years)", f"{meta['test_mae']:.2f}")
    col3.metric("Test RMSE (years)", f"{meta['test_rmse']:.2f}")

    st.caption(
        "Metrics are from a held-out test set (20% of rows), not seen during training."
    )
else:
    st.warning(
        "No trained model found yet. Run `python train_model.py` first, "
        "then restart the app."
    )

st.markdown(
    """
### About this project
This app is built on a data-science pipeline that:
1. Cleaned messy WHO source data (column names, implicit/explicit nulls)
2. Imputed missing values using per-country time-series interpolation
3. Winsorized outliers to reduce their influence on the model
4. Compared five model types (linear, polynomial, decision tree, random forest,
   logistic classification) and selected Random Forest for its strong,
   well-generalizing performance
"""
)
