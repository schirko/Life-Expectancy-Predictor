import json
import os

import streamlit as st

st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🌍",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

st.title("🌍 Life Expectancy Predictor")
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
    col1.metric("Model Test R²", f"{meta['test_r2']:.3f}")
    col2.metric("Test MAE (Years)", f"{meta['test_mae']:.2f}")
    col3.metric("Test RMSE (Years)", f"{meta['test_rmse']:.2f}")

    st.caption(
        "Metrics are from a held-out test set (20% of rows), not seen during training."
    )
else:
    st.warning(
        "No trained model found yet. Run `python train_model.py` first, "
        "then restart the app."
    )

st.divider()

st.markdown("### About This Project")
st.markdown(
    """
This app is built on a data-science pipeline that took raw, messy WHO data
all the way through to a deployed, interactive prediction tool. It started
as a graduate data-science coursework project and was later turned into this
app to make the model explorable rather than leaving it in a notebook.
"""
)

st.markdown("### Data Source")
st.markdown(
    """
The underlying dataset is the **WHO Life Expectancy** dataset, compiled from
World Health Organization and United Nations records. It covers **2,929
country-year observations** across roughly **193 countries** from **2000 to
2015**, with indicators spanning:

- Mortality (adult mortality, infant/under-five deaths, HIV/AIDS)
- Immunization coverage (Polio, Diphtheria, Hepatitis B)
- Economic factors (health expenditure, income composition of resources)
- Social factors (schooling, alcohol consumption, thinness prevalence)
"""
)

st.markdown("### Methodology")
st.markdown(
    """
1. **Cleaned** messy source data — normalized column names and resolved
   both explicit (`NaN`) and implicit (e.g. `0` used as a placeholder) missing
   values
2. **Imputed** missing values using per-country time-series interpolation,
   falling back to country and then global medians where interpolation
   wasn't possible
3. **Winsorized** outliers on a per-feature basis to reduce their influence
   on the trained model
4. **Compared five modeling approaches** (see table below) and selected
   Random Forest for its strong, well-generalizing performance
"""
)

st.markdown("### Model Comparison")
model_comparison = [
    {"Model": "Simple Linear Regression (Schooling → Life Expectancy)", "Task": "Regression", "Performance": "R² = 0.55"},
    {"Model": "Multiple Linear Regression", "Task": "Regression", "Performance": "R² = 0.85"},
    {"Model": "Polynomial Regression", "Task": "Regression", "Performance": "R² ≈ 0.57 (overfits past degree 3)"},
    {"Model": "Decision Tree", "Task": "Regression", "Performance": "R² = 0.91"},
    {"Model": "Random Forest (used in this app)", "Task": "Regression", "Performance": "R² = 0.96"},
    {"Model": "Logistic Regression (High/Low Life Expectancy)", "Task": "Classification", "Performance": "88% accuracy"},
]
st.dataframe(model_comparison, use_container_width=True, hide_index=True)

st.markdown("### Limitations To Keep In Mind")
st.markdown(
    """
- **Correlation, not causation.** A strong predictor (like Adult Mortality)
  reflects a statistical relationship, not necessarily a direct cause-and-effect
  link.
- **Historical data.** The dataset ends in 2015, so predictions reflect
  relationships from that period and may not capture more recent shifts in
  global health.
- **Imputed values.** Some indicators were filled in through interpolation
  where the original data had gaps, which introduces some uncertainty into
  those country-years.
- **Model uncertainty.** The Predict page shows a rough uncertainty range
  based on the spread across the Random Forest's individual trees — treat
  single-point predictions as estimates, not exact figures.
"""
)

st.divider()
st.caption("Source code: [github.com/schirko/LifeExpectancyApp](https://github.com/schirko/LifeExpectancyApp)")