import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="Predict | Life Expectancy Predictor", page_icon="🔮", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data.csv")

st.title("🔮 Predict Life Expectancy")

if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    st.error("No trained model found. Run `python train_model.py` from the project folder first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    # Leading underscore tells Streamlit not to hash this argument (models
    # aren't easily hashable); caching is keyed on the function name alone,
    # which is fine here since there's only ever one model.
    return shap.TreeExplainer(_model)


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
meta = load_meta()
df = load_data()
predictor_cols = meta["predictor_cols"]
ranges = meta["feature_ranges"]

st.markdown(
    "Enter values for the indicators below, then click **Predict** to estimate "
    "life expectancy using the trained Random Forest model "
    f"(test R² = {meta['test_r2']:.3f}, avg. error ≈ {meta['test_mae']:.1f} years)."
)

# Features shown as sliders (bounded rate/percentage-style indicators)
SLIDER_FEATURES = [
    "Adult_Mortality",
    "HIV_AIDS",
    "Under-Five_Deaths",
    "Polio",
    "Diphtheria",
    "Hepatitis_B",
    "Thinness_10-19_Years",
]

# Features shown as raw number entry (less intuitive bounds / more precision-sensitive)
NUMBER_INPUT_FEATURES = [
    "Alcohol",
    "Pct_Expenditure",
    "Ttl_Expend",
    "Income_Comp_Of_Resources",
    "Schooling",
]

FEATURE_LABELS = {
    "Adult_Mortality": "Adult Mortality (per 1,000 adults)",
    "HIV_AIDS": "HIV/AIDS deaths (per 1,000, ages 0-4)",
    "Under-Five_Deaths": "Under-Five Deaths (per 1,000)",
    "Polio": "Polio immunization (%)",
    "Diphtheria": "Diphtheria (DTP3) immunization (%)",
    "Hepatitis_B": "Hepatitis B immunization (%)",
    "Thinness_10-19_Years": "Thinness, ages 10-19 (%)",
    "Alcohol": "Alcohol consumption (litres/capita)",
    "Pct_Expenditure": "Health expenditure (% of GDP per capita)",
    "Ttl_Expend": "Total govt. health expenditure (% of total govt. expenditure)",
    "Income_Comp_Of_Resources": "Income Composition of Resources (0-1 index)",
    "Schooling": "Schooling (avg. years)",
}

# Longer, plain-English explanations shown in the glossary expander and as
# per-field help text (hover the "?" icon next to each input).
FEATURE_DESCRIPTIONS = {
    "Adult_Mortality": "Probability of dying between ages 15-60, per 1,000 population. Higher = worse.",
    "HIV_AIDS": "Deaths per 1,000 live births caused by HIV/AIDS, ages 0-4.",
    "Under-Five_Deaths": "Deaths per 1,000 population among children under age 5.",
    "Polio": "% of 1-year-olds immunized against polio.",
    "Diphtheria": "% of 1-year-olds immunized against diphtheria (DTP3 vaccine).",
    "Hepatitis_B": "% of 1-year-olds immunized against Hepatitis B.",
    "Thinness_10-19_Years": "% of 10-19 year-olds classified as underweight (thin) for their age.",
    "Alcohol": "Liters of pure alcohol consumed per capita (age 15+ population).",
    "Pct_Expenditure": "Health expenditure as a percentage of GDP per capita.",
    "Ttl_Expend": "Total government health spending as a percentage of total government spending.",
    "Income_Comp_Of_Resources": (
        "A UN Human Development Index sub-component (0-1 scale) capturing how "
        "effectively a country converts income into human development outcomes. "
        "Higher = more effective."
    ),
    "Schooling": "Average number of years of schooling completed.",
}

st.subheader("Indicators")

with st.expander("📖 What Do These Indicators Mean?"):
    glossary_df = pd.DataFrame(
        [
            {"Indicator": FEATURE_LABELS.get(f, f), "Meaning": FEATURE_DESCRIPTIONS.get(f, "")}
            for f in SLIDER_FEATURES + NUMBER_INPUT_FEATURES
        ]
    )
    st.dataframe(glossary_df, use_container_width=True, hide_index=True)

col_a, col_b = st.columns(2)

values = {}

with col_a:
    st.markdown("**Sliders**")
    for feat in SLIDER_FEATURES:
        lo, hi, med = ranges[feat]
        step = (hi - lo) / 100 if hi > lo else 1.0
        values[feat] = st.slider(
            FEATURE_LABELS.get(feat, feat),
            min_value=round(lo, 2),
            max_value=round(hi, 2),
            value=round(med, 2),
            step=round(step, 2) if step > 0 else 1.0,
            help=FEATURE_DESCRIPTIONS.get(feat, ""),
            key=f"slider_{feat}",
        )

with col_b:
    st.markdown("**Direct Entry**")
    for feat in NUMBER_INPUT_FEATURES:
        lo, hi, med = ranges[feat]
        values[feat] = st.number_input(
            FEATURE_LABELS.get(feat, feat),
            min_value=round(lo, 3),
            max_value=round(hi * 1.5, 3),  # allow entering somewhat beyond observed max
            value=round(med, 3),
            help=f"{FEATURE_DESCRIPTIONS.get(feat, '')} (Observed range in data: {lo:.2f} – {hi:.2f})",
            key=f"num_{feat}",
        )

# Fill in any remaining predictor columns not shown in the UI (e.g. Measles)
# using their dataset median, so the model always gets a full feature vector.
for feat in predictor_cols:
    if feat not in values:
        _, _, med = ranges.get(feat, (0, 0, 0))
        values[feat] = med

st.divider()

predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=False)

if predict_clicked:
    x = np.array([[values[c] for c in predictor_cols]])
    prediction = model.predict(x)[0]

    # Spread across the forest's individual trees, as a rough uncertainty band
    tree_preds = np.array([tree.predict(x)[0] for tree in model.estimators_])
    lower, upper = np.percentile(tree_preds, [10, 90])

    st.subheader("Result")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted life expectancy", f"{prediction:.1f} years")
        st.caption(f"~80% of the model's trees predicted between {lower:.1f} and {upper:.1f} years.")

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df["Life_Expectancy"],
                name="Dataset distribution",
                opacity=0.6,
                nbinsx=40,
            )
        )
        fig.add_vline(
            x=prediction,
            line_width=3,
            line_color="crimson",
            annotation_text="Your prediction",
            annotation_position="top",
        )
        fig.update_layout(
            title="Where this prediction falls in the dataset",
            xaxis_title="Life Expectancy",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why This Prediction (SHAP)")
    explainer = load_explainer(model)
    shap_raw = explainer.shap_values(x)
    # TreeExplainer on a regressor returns a 2D array (n_samples, n_features);
    # guard against SHAP versions that wrap this in a list instead.
    if isinstance(shap_raw, list):
        shap_raw = shap_raw[0]
    shap_row = np.array(shap_raw).reshape(-1)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.array(base_value).reshape(-1)[0]

    shap_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(c, c) for c in predictor_cols],
            "Contribution": shap_row,
        }
    ).sort_values("Contribution", key=lambda s: s.abs(), ascending=True)

    fig_shap = go.Figure(
        go.Bar(
            x=shap_df["Contribution"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=["#d62728" if v < 0 else "#2ca02c" for v in shap_df["Contribution"]],
        )
    )
    fig_shap.update_layout(
        title=f"Baseline (avg. prediction): {base_value:.1f} yrs  →  Your prediction: {prediction:.1f} yrs",
        xaxis_title="Impact on Predicted Life Expectancy (Years)",
        height=420,
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption(
        "Green bars pushed this specific prediction higher; red bars pushed it lower — "
        "relative to the model's average prediction across the training data (the "
        "baseline). These are SHAP (SHapley Additive exPlanations) values from the "
        "Random Forest's TreeExplainer, which fairly attribute this individual "
        "prediction to each input feature, unlike the global importances below."
    )

    st.subheader("Overall Feature Importance (All Predictions)")
    imp_df = pd.DataFrame(
        list(meta["feature_importances"].items()), columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False)
    st.dataframe(imp_df, use_container_width=True, hide_index=True)
    st.caption(
        "These are overall model feature importances (not specific to this single "
        "prediction), showing which indicators the Random Forest relies on most "
        "across all its predictions. Contrast with the SHAP chart above, which "
        "explains only this one prediction."
    )
else:
    st.info("Set your indicator values above and click **Predict**.")