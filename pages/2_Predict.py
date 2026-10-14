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
target_col = meta.get("target_col", "Life_Expectancy")

st.markdown(
    f"Enter values for the indicators below, then click **Predict** to estimate "
    f"{meta.get('dataset_name', 'the target')} using the trained Random Forest model "
    f"(test R² = {meta['test_r2']:.3f}, avg. error ≈ {meta['test_mae']:.1f} years)."
)

# Predictor metadata (label, description, slider vs. number, whether to show
# a control at all) comes entirely from the training config via metadata.json
# -- nothing about which features exist or how to describe them is hardcoded
# here, so swapping in a differently-shaped dataset's config doesn't require
# touching this page.
predictors_meta = meta.get("predictors", [])
FEATURE_LABELS = {p["column"]: p.get("label", p["column"]) for p in predictors_meta}
FEATURE_DESCRIPTIONS = {p["column"]: p.get("description", "") for p in predictors_meta}

visible_predictors = [p for p in predictors_meta if p.get("show_in_ui", True)]
slider_specs = [p for p in visible_predictors if p.get("input_type") == "slider"]
number_specs = [p for p in visible_predictors if p.get("input_type") != "slider"]

st.subheader("Indicators")

with st.expander("📖 What Do These Indicators Mean?"):
    glossary_df = pd.DataFrame(
        [
            {"Indicator": p.get("label", p["column"]), "Meaning": p.get("description", "")}
            for p in visible_predictors
        ]
    )
    st.dataframe(glossary_df, use_container_width=True, hide_index=True)

col_a, col_b = st.columns(2)

values = {}
defaults = {}  # the value each control starts at (the dataset median)

with col_a:
    st.markdown("**Sliders**")
    if not slider_specs:
        st.caption("No slider-type indicators configured.")
    for spec in slider_specs:
        feat = spec["column"]
        lo, hi, med = ranges[feat]
        step = (hi - lo) / 100 if hi > lo else 1.0
        defaults[feat] = round(med, 2)
        values[feat] = st.slider(
            spec.get("label", feat),
            min_value=round(lo, 2),
            max_value=round(hi, 2),
            value=round(med, 2),
            step=round(step, 2) if step > 0 else 1.0,
            help=spec.get("description", ""),
            key=f"slider_{feat}",
        )

with col_b:
    st.markdown("**Direct Entry**")
    if not number_specs:
        st.caption("No direct-entry indicators configured.")
    for spec in number_specs:
        feat = spec["column"]
        lo, hi, med = ranges[feat]
        defaults[feat] = round(med, 3)
        values[feat] = st.number_input(
            spec.get("label", feat),
            min_value=round(lo, 3),
            max_value=round(hi * 1.5, 3),  # allow entering somewhat beyond observed max
            value=round(med, 3),
            help=f"{spec.get('description', '')} (Observed range in data: {lo:.2f} – {hi:.2f})",
            key=f"num_{feat}",
        )

# Fill in any remaining predictor columns not shown in the UI (e.g. Measles)
# using their dataset median, so the model always gets a full feature vector.
for feat in predictor_cols:
    if feat not in values:
        _, _, med = ranges.get(feat, (0, 0, 0))
        values[feat] = med

# ---- Thin-data check ----
# During testing, the model's worst miss was a country with no real data for
# three inputs: those had been filled with the all-country median, so the model
# saw an "average" country and predicted accordingly. The same thing happens
# here if most controls are left at their starting value (the median).
THIN_DATA_SHARE = 0.5  # warn when at least this share of visible inputs is untouched
untouched = [
    FEATURE_LABELS.get(p["column"], p["column"])
    for p in visible_predictors
    if p["column"] in defaults and abs(values[p["column"]] - defaults[p["column"]]) < 1e-9
]
n_visible = len([p for p in visible_predictors if p["column"] in defaults])
thin_data = n_visible > 0 and len(untouched) / n_visible >= THIN_DATA_SHARE
st.caption(
    f"{len(untouched)} of {n_visible} inputs are still at their starting value "
    "(the dataset median)."
)

st.divider()

predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=False)

if predict_clicked:
    x = np.array([[values[c] for c in predictor_cols]])
    prediction = model.predict(x)[0]

    # Spread across the forest's individual trees, as a rough uncertainty band
    tree_preds = np.array([tree.predict(x)[0] for tree in model.estimators_])
    lower, upper = np.percentile(tree_preds, [10, 90])

    if thin_data:
        st.warning(
            f"**Thin-data warning: {len(untouched)} of {n_visible} inputs are still at their "
            "starting value (the dataset median).** The model treats those as typical values, "
            "so this prediction will drift toward the all-country average. During validation, "
            "the model's biggest miss came from a country whose inputs had been filled with "
            "medians in exactly this way. Enter real values for more inputs for a more "
            "meaningful estimate."
        )
        with st.expander("Which inputs are untouched?"):
            st.write(", ".join(untouched))

    st.subheader("Result")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(f"Predicted {meta.get('dataset_name', 'Value')}", f"{prediction:.1f} years")
        st.caption(f"~80% of the model's trees predicted between {lower:.1f} and {upper:.1f} years.")

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df[target_col],
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
            xaxis_title=target_col.replace("_", " "),
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
