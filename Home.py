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
- **🔬 Diagnostics** — residuals, permutation importance, partial dependence, unseen-country validation, a four-model showdown, uncertainty calibration, hyperparameter tuning, and data quality

---
"""
)

meta_path = os.path.join(MODEL_DIR, "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test R²", f"{meta['test_r2']:.3f}")
    col2.metric("Test MAE (Years)", f"{meta['test_mae']:.2f}")
    col3.metric("Test RMSE (Years)", f"{meta['test_rmse']:.2f}")
    if "cv_r2_mean" in meta:
        col4.metric(
            "Unseen-Country CV R²",
            f"{meta['cv_r2_mean']:.3f}",
            help=f"± {meta.get('cv_r2_std', 0):.3f} across folds. Each fold holds out whole countries, so this measures accuracy on countries the model never saw during training.",
        )

    st.caption(
        f"Trained on **{meta.get('dataset_name', 'this dataset')}** "
        f"({meta.get('n_rows', '?'):,} rows) using config `{meta.get('config_path', '?')}`. "
        "Test metrics are from a held-out 20% split; the CV score holds out "
        "entire countries (GroupKFold by country), so it reflects how well the model "
        "generalizes to countries it has never seen. See Diagnostics → Unseen Countries."
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

st.markdown("### Architecture: Config-Driven, Not Hardcoded")
st.markdown(
    """
Rather than hardcoding this dataset's column names and cleaning rules
directly into the app's code, all of that dataset-specific knowledge lives
in a separate YAML config file (`config/who_life_expectancy.yaml`):
column renames, which values count as missing, per-feature outlier limits,
the model's predictor list, and how each predictor should appear in the
Predict page's UI.

The cleaning pipeline, training script, and Predict page all read from
whichever config is active rather than containing dataset-specific logic
themselves. A training run also validates the config against the CSV's
actual columns up front, failing fast with a clear message if they don't
match, rather than partway through cleaning.

In practice, this means a **similarly-shaped** WHO-style CSV (same kind of
country-year panel data, a numeric target, numeric predictors) can be
swapped in by writing a new config file and re-running training — without
touching the Python code.
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
st.caption(
    "Note: figures below are from the original notebook's single train/test "
    "split. This app's currently-deployed Random Forest additionally reports "
    "country-grouped cross-validation results above, which test generalization to unseen countries."
)
model_comparison = [
    {"Model": "Simple Linear Regression (Schooling → Life Expectancy)", "Task": "Regression", "Performance": "R² = 0.55"},
    {"Model": "Multiple Linear Regression", "Task": "Regression", "Performance": "R² ≈ 0.85"},
    {"Model": "Polynomial Regression", "Task": "Regression", "Performance": "R² ≈ 0.57 (plateaus at degree 3, no meaningful overfitting up to degree 25)"},
    {"Model": "Decision Tree", "Task": "Regression", "Performance": "R² ≈ 0.91"},
    {"Model": "Random Forest (used in this app)", "Task": "Regression", "Performance": "R² = 0.96"},
    {"Model": "Logistic Regression (High/Low Life Expectancy)", "Task": "Classification", "Performance": "88% accuracy"},
]
st.dataframe(model_comparison, use_container_width=True, hide_index=True)

with st.expander("🔬 See the technical work behind each model"):
    st.caption(
        "Actual code from the original notebook for each of the 6 models above "
        "(trimmed to the core fitting/evaluation logic — plotting code omitted). "
        "All models split data 80/20 with `random_state=42` for reproducibility."
    )

    model_details = [
        {
            "name": "Simple Linear Regression (Schooling → Life Expectancy)",
            "note": (
                "Baseline: a single predictor (Schooling) against Life Expectancy. "
                "R² = 0.55 shows schooling alone explains just over half the variance."
            ),
            "code": '''x = df.Schooling.values.reshape(-1, 1)
y = df['Life_Expectancy'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)
regr = LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
r2_score(y_test, y_pred)   # 0.55''',
        },
        {
            "name": "Multiple Linear Regression",
            "note": (
                "All 12 predictors, linear coefficients. R² jumps to 0.85 — most of "
                "the model's signal is linear, which the Random Forest later improves on "
                "by capturing non-linear interactions. Also reports 5-fold CV on the "
                "training set alone, a pattern later extended to the full dataset for "
                "the deployed Random Forest."
            ),
            "code": '''predictor_cols = [c for c in feature_vars if c != 'Life_Expectancy']
x = df[predictor_cols].values
y = df['Life_Expectancy'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)
regr = LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
r2_score(y_test, y_pred)   # 0.85
cross_val_score(regr, x_train, y_train, cv=5, scoring="r2").mean()''',
        },
        {
            "name": "Polynomial Regression",
            "note": (
                "Swept polynomial degrees 1–25 (single predictor: Schooling), comparing "
                "train vs. held-out test R² at each degree. Performance plateaus around "
                "degree 3 (~0.57); higher degrees add negligible benefit. With only one "
                "predictor there isn't enough complexity for the model to meaningfully "
                "overfit — the train/test gap stays under 0.02 all the way to degree 25, "
                "well under the notebook's own 0.05 overfitting threshold."
            ),
            "code": '''x = df.Schooling.values.reshape(-1, 1)
y = df['Life_Expectancy'].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

for degree in [1, 2, 3, 5, 8, 15]:
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)
    regr = LinearRegression().fit(x_train_poly, y_train)

    train_r2 = r2_score(y_train, regr.predict(x_train_poly))
    test_r2 = r2_score(y_test, regr.predict(x_test_poly))
    # train/test R2 stay within ~0.006 of each other at every degree
    # tested (1-15) -- no meaningful overfitting signature here''',
        },
        {
            "name": "Decision Tree",
            "note": (
                "All 12 predictors, `max_depth=6` capped deliberately to reduce "
                "overfitting on a single tree. R² = 0.91 — a single tree already "
                "captures most of what the Random Forest later achieves."
            ),
            "code": '''predictor_cols = [c for c in feature_vars if c != 'Life_Expectancy']
x = df[predictor_cols].values
y = df['Life_Expectancy'].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

regr = DecisionTreeRegressor(max_depth=6, random_state=42)
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
r2_score(y_test, y_pred)   # 0.91''',
        },
        {
            "name": "Random Forest (used in this app)",
            "note": (
                "200 trees, no depth cap (the ensemble averaging controls overfitting "
                "instead of a single tree's depth limit). R² = 0.96 on this single "
                "split; this app's Home page above also reports the more robust "
                "country-grouped CV score (≈ 0.90) for the currently-deployed model."
            ),
            "code": '''predictor_cols = [c for c in feature_vars if c != 'Life_Expectancy']
x = df[predictor_cols].values
y = df['Life_Expectancy'].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

regr = RandomForestRegressor(n_estimators=200, random_state=42)
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
r2_score(y_test, y_pred)   # 0.96''',
        },
        {
            "name": "Logistic Regression (High/Low Life Expectancy)",
            "note": (
                "Reframed as binary classification: split at the median Life "
                "Expectancy (High vs. Low), min-max normalized all 12 predictors "
                "(logistic regression is scale-sensitive, unlike tree-based models), "
                "and used the `liblinear` solver. 88.05% accuracy on the held-out set, "
                "confirmed by confusion matrix and ROC AUC in the original notebook.\n\n"
                "One methodology note: normalization stats (min/max) here are computed "
                "on the full dataset before the train/test split — technically a mild "
                "form of data leakage (best practice fits the scaler on train data only, "
                "then applies it to test). Checked independently: recomputing with "
                "train-only stats gives the identical 88.05% accuracy, so it had no "
                "practical effect here, but it's a pattern worth avoiding in general."
            ),
            "code": '''median_le = df['Life_Expectancy'].median()
df['LE_Class'] = (df['Life_Expectancy'] >= median_le).astype(int)  # 1=High, 0=Low

X = df[predictor_cols]
X = (X - X.min()) / (X.max() - X.min())   # min-max normalize
y = df['LE_Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
logi = LogisticRegression(solver="liblinear")
logi.fit(X_train, y_train)

accuracy_score(y_test, logi.predict(X_test))   # 0.8805''',
        },
    ]

    for detail in model_details:
        st.markdown(f"**{detail['name']}**")
        st.caption(detail["note"])
        st.code(detail["code"], language="python")
        st.markdown("")

st.markdown("### Model Card")
st.markdown(
    "A condensed summary in the format commonly used in industry to document "
    "a model's purpose, data, and known limitations in one place — see the "
    "🔬 Diagnostics page for the underlying analysis behind these claims."
)
if os.path.exists(meta_path):
    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        st.markdown(
            f"""
**Model Details**
- Type: Random Forest Regressor (scikit-learn)
- Trees: 200, unlimited depth
- Trained on: {meta.get('n_rows', '?'):,} rows, {len(meta.get('predictor_cols', [])):,} predictors
- Config: `{meta.get('config_path', '?')}`

**Intended Use**
- Educational / portfolio demonstration of a full ML pipeline
- Exploring which health & economic factors associate with life expectancy
- **Not** intended for real-world public-health policy or individual
  clinical decisions
"""
        )
    with mc_col2:
        st.markdown(
            f"""
**Performance**
- Test R²: {meta['test_r2']:.3f}  |  Unseen-country CV R² (GroupKFold): {meta.get('cv_r2_mean', 0):.3f} (± {meta.get('cv_r2_std', 0):.3f})
- Test MAE: {meta['test_mae']:.2f} years  |  Test RMSE: {meta['test_rmse']:.2f} years
- Uncertainty interval: see 🔬 Diagnostics → Uncertainty Calibration for an
  empirical check of how trustworthy the Predict page's ranges actually are

**Training Data**
- WHO/UN life expectancy indicators, 2000–2015, ~193 countries
- See 🔬 Diagnostics → Data Quality for missingness before/after cleaning
"""
        )
    st.markdown(
        """
**Ethical Considerations & Caveats**
- Predictions reflect historical statistical associations, not causal
  mechanisms or policy recommendations
- The training data's geographic and temporal coverage (2000–2015) may not
  represent current conditions or underrepresented regions equally
- See "Limitations To Keep In Mind" below for the full list
"""
    )
else:
    st.info("Train the model to populate the Model Card with real metrics.")

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
- **Same-country leakage in random splits.** The data is a panel: each
  country appears in up to 16 years. A random 80/20 split puts a country's
  other years in training, so the test score partly measures memorizing
  countries. Holding out whole countries (GroupKFold) gives a lower but more
  honest R² for countries the model has never seen. See Diagnostics →
  Unseen Countries for the comparison and the hardest-to-predict countries.
"""
)

st.divider()
st.caption("Source code: [github.com/schirko/LifeExpectancyApp](https://github.com/schirko/LifeExpectancyApp)")
