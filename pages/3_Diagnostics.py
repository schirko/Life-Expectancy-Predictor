import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Diagnostics | Life Expectancy Predictor", page_icon="🔬", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DIAGNOSTICS_PATH = os.path.join(BASE_DIR, "model", "diagnostics.json")
TUNING_PATH = os.path.join(BASE_DIR, "model", "tuning_results.json")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")


@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


st.title("🔬 Model Diagnostics")
st.markdown(
    "Deeper analysis behind the headline metrics: residuals, permutation "
    "importance, partial dependence, uncertainty calibration, hyperparameter "
    "tuning, and data quality — all computed from the held-out test set "
    "during training, not invented for display."
)

diagnostics = load_json(DIAGNOSTICS_PATH)
meta = load_json(META_PATH)

if diagnostics is None:
    st.error(
        "No diagnostics found. Run `python train_model.py` (the current version "
        "computes and saves `model/diagnostics.json` automatically)."
    )
    st.stop()

target_label = meta.get("target_col", "Life_Expectancy").replace("_", " ") if meta else "Life Expectancy"

tab_residuals, tab_perm, tab_pdp, tab_calibration, tab_tuning, tab_quality = st.tabs(
    [
        "Residuals",
        "Permutation Importance",
        "Partial Dependence",
        "Uncertainty Calibration",
        "Hyperparameter Tuning",
        "Data Quality",
    ]
)

# ---- Residual diagnostics ----
with tab_residuals:
    with st.expander("📚 What is residual analysis, and why does it matter?"):
        st.markdown(
            """
A single R² score tells you *how much* variance a model explains overall,
but hides *where* it's wrong. Residual analysis breaks that open by looking
at every individual prediction error (residual = actual − predicted) and
checking for patterns:

- **Actual vs. Predicted plot:** if points cluster tightly around the
  diagonal, predictions track reality well across the whole range. If they
  fan out or curve away from the diagonal at one end (e.g. the model
  consistently underpredicts very high or low values), that's a real,
  fixable weakness a single R² number would never reveal.
- **Residuals vs. Predicted plot:** the classic pattern to watch for is a
  **funnel shape** (errors growing as predictions get larger or smaller) —
  called *heteroscedasticity* — which means the model is more trustworthy
  in some ranges than others. A **curve** instead of random scatter
  suggests a relationship the model hasn't captured (it's missing some
  non-linearity or interaction).
- **Distribution of residuals:** should be roughly centered on zero and
  reasonably symmetric. A skewed distribution means the model is
  systematically over- or under-predicting in one direction more than
  the other.

**Why this matters in practice:** two models can have identical R² scores
while one is reliable everywhere and the other is only reliable for
"typical" cases and badly wrong on edge cases. Residual analysis is how you
tell those two situations apart.
"""
        )

    st.subheader("Actual vs. Predicted")
    st.caption(
        "How to read this: the x-axis is the true value, the y-axis is what the "
        "model predicted. Each point is one held-out test-set row. Points hugging "
        "the dashed diagonal line closely, with only minor scatter, is what a "
        "strong R² looks like visually — a systematic curve or fan shape away "
        "from the line would signal a problem a single R² number wouldn't reveal."
    )
    y_test = diagnostics["residuals"]["y_test"]
    y_pred = diagnostics["residuals"]["y_pred"]
    residuals = diagnostics["residuals"]["residuals"]

    lims = [min(min(y_test), min(y_pred)) - 2, max(max(y_test), max(y_pred)) + 2]
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", opacity=0.5, name="Predictions"))
    fig_avp.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="crimson", dash="dash"), name="Perfect prediction"))
    fig_avp.update_layout(xaxis_title=f"Actual {target_label}", yaxis_title=f"Predicted {target_label}", height=450)
    st.plotly_chart(fig_avp, use_container_width=True)

    st.subheader("Residuals vs. Predicted")
    st.caption(
        "How to read this: x-axis is the predicted value, y-axis is the error "
        "(actual − predicted). You want a random cloud centered on zero with no "
        "shape. A funnel that widens toward one side means the model is less "
        "reliable at certain prediction ranges; a curve instead of random scatter "
        "means the model is missing some non-linear relationship."
    )
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", opacity=0.5))
    fig_resid.add_hline(y=0, line_color="crimson", line_dash="dash")
    fig_resid.update_layout(xaxis_title=f"Predicted {target_label}", yaxis_title="Residual (Actual − Predicted)", height=450)
    st.plotly_chart(fig_resid, use_container_width=True)

    st.subheader("Distribution of Residuals")
    st.caption(
        "How to read this: just the shape of every error, all together. Should "
        "look roughly like a bell curve centered near zero. A skew to one side "
        "would mean the model systematically over- or under-predicts more often "
        "than the reverse."
    )
    fig_hist = px.histogram(x=residuals, nbins=40, labels={"x": "Residual (years)"})
    fig_hist.add_vline(x=0, line_color="crimson", line_dash="dash")
    st.plotly_chart(fig_hist, use_container_width=True)
    mean_resid = float(np.mean(residuals))
    st.caption(
        f"Mean residual: {mean_resid:+.3f} years — close to zero means predictions "
        f"aren't systematically biased high or low overall."
    )

# ---- Permutation importance ----
with tab_perm:
    with st.expander("📚 Why do two importance measures disagree, and which should I trust?"):
        st.markdown(
            """
**Built-in (Gini/impurity) importance** is computed for free while the tree
is being built: every time a feature is used to split a node, the
improvement in prediction purity at that split is credited to that
feature, summed across every tree. The problem: it's a biased measure. A
feature with many unique continuous values (like Adult Mortality's
hundreds of distinct values) gets far more *opportunities* to be chosen
for a split than a feature with fewer distinct values — inflating its
apparent importance regardless of whether it's actually more predictive.

**Permutation importance** sidesteps this entirely: it doesn't look at
tree structure at all. Instead, it takes the *already-trained* model,
shuffles one feature's values randomly in the test set (destroying that
feature's relationship with the target while leaving everything else
intact), and measures how much test performance (R²) drops. A feature
that matters a lot causes a big drop when scrambled; a feature that barely
matters causes almost no change. This is repeated 10 times per feature (to
average out randomness from the shuffling) and works with *any* model
type, not just trees.

**The practical lesson from this app's own numbers:** built-in importance
says HIV/AIDS (58%) dominates Adult Mortality (14%) by roughly 4-to-1.
Permutation importance says they're nearly tied (26% vs 24%). The
permutation version is the one to trust when the two disagree — it's
measuring actual predictive contribution, not an artifact of how trees
happen to get built.
"""
        )

    st.subheader("Permutation Importance vs. Built-In Feature Importance")
    st.caption(
        "How to read the bar chart: each bar's length is how much test R² "
        "dropped when that feature's values got randomly shuffled — longer bar = "
        "model relies on it more. The whiskers are ± one standard deviation "
        "across 10 repeated shuffles (shuffling is random, so it's repeated and "
        "averaged for stability)."
    )
    st.markdown(
        """
The Random Forest's default `feature_importances_` (shown on the Home and
Dashboard pages) is a **Gini/impurity-based** measure, known to be biased
toward continuous and high-cardinality features. **Permutation importance**
is a more trustworthy alternative: it shuffles one feature at a time in the
held-out test set and measures how much model performance (R²) drops — a
feature that matters a lot causes a big drop when shuffled.
"""
    )
    perm_df = pd.DataFrame(diagnostics["permutation_importance"])
    built_in = meta.get("feature_importances", {}) if meta else {}
    perm_df["built_in_importance"] = perm_df["feature"].map(built_in)
    perm_df = perm_df.sort_values("importance_mean", ascending=True)

    fig_perm = go.Figure()
    fig_perm.add_trace(
        go.Bar(
            y=perm_df["feature"],
            x=perm_df["importance_mean"],
            error_x=dict(type="data", array=perm_df["importance_std"]),
            orientation="h",
            name="Permutation importance (R² drop)",
        )
    )
    fig_perm.update_layout(
        xaxis_title="Mean R² drop when shuffled (± std across 10 repeats)",
        height=450,
    )
    st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("**Side-by-side comparison** (both re-scaled to sum to 1 for comparability):")
    compare_df = perm_df.copy()
    compare_df["permutation_importance"] = compare_df["importance_mean"] / compare_df["importance_mean"].sum()
    compare_df["built_in_importance"] = compare_df["built_in_importance"] / compare_df["built_in_importance"].sum()
    compare_df = compare_df.sort_values("permutation_importance", ascending=False)
    st.dataframe(
        compare_df[["feature", "permutation_importance", "built_in_importance"]].rename(
            columns={"feature": "Feature", "permutation_importance": "Permutation", "built_in_importance": "Built-In (Gini)"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Notice HIV/AIDS and Adult Mortality are much closer under permutation "
        "importance than under the built-in Gini importance, which overstates "
        "HIV/AIDS's dominance relative to Adult Mortality."
    )

# ---- Partial dependence ----
with tab_pdp:
    with st.expander("📚 Importance tells you HOW MUCH; partial dependence tells you WHAT SHAPE"):
        st.markdown(
            """
Feature importance (either kind) answers "how much does this feature
matter overall?" — a single number. It says nothing about the *direction*
or *shape* of the relationship: does higher HIV/AIDS prevalence always
lower predicted life expectancy, or only past some threshold? Is the
effect a straight line, or does it level off?

A **partial dependence plot** answers that by holding every other feature
fixed and sweeping the one feature of interest across its observed range,
recording how the model's average prediction changes. The result is a
picture of the relationship the model actually learned — which might be
linear, might plateau, might have a sharp threshold, or might even be
non-monotonic (rise then fall).

**An honest limitation worth knowing:** partial dependence assumes
features can be varied independently of each other. If two features are
strongly correlated in reality (which several of this dataset's health
indicators are), the "holding everything else fixed" scenario can
represent a combination that never actually occurs in real countries —
so treat the shape as what the *model* learned, not necessarily a claim
about causal reality.
"""
        )

    st.subheader("Partial Dependence Plots")
    st.markdown(
        """
For the top 3 features (by permutation importance), these plots show how the
model's **predicted** life expectancy changes as that one feature varies,
holding all other features at their observed values and averaging the
effect — answering "what relationship did the model actually learn," beyond
just an importance ranking.
"""
    )
    st.caption(
        "How to read each line: x-axis is that feature's value range, y-axis is "
        "the model's predicted life expectancy as that value changes. What "
        "matters is the slope and shape — a steady downward trend, a flattening "
        "past some point (diminishing effect), or a non-linear bend all tell you "
        "something a single importance number can't."
    )
    pdp_data = diagnostics["partial_dependence"]
    cols = st.columns(len(pdp_data))
    for col, (feat, data) in zip(cols, pdp_data.items()):
        with col:
            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(x=data["grid_values"], y=data["predicted"], mode="lines"))
            fig_pdp.update_layout(
                title=feat.replace("_", " "),
                xaxis_title=feat.replace("_", " "),
                yaxis_title=f"Predicted {target_label}",
                height=350,
            )
            st.plotly_chart(fig_pdp, use_container_width=True)

# ---- Uncertainty calibration ----
with tab_calibration:
    with st.expander("📚 What does \"calibration\" mean, and why isn't a tree-spread interval rigorous?"):
        st.markdown(
            """
A prediction interval is **calibrated** if it means what it claims: an
80% interval should contain the true value roughly 80% of the time when
checked against many real outcomes — no more, no less. An interval that's
too narrow is *overconfident* (dangerous — it understates real
uncertainty). One that's too wide is *conservative* (safe, but less
useful — it's not telling you much).

This app's Predict page estimates uncertainty by looking at the spread of
predictions across the Random Forest's 200 individual trees (10th to 90th
percentile). That's an intuitive, cheap approximation, but it isn't a
statistically rigorous prediction interval — it reflects how much the
trees *disagree* with each other, not a formally derived confidence level.
The check on this tab (does the stated 80% actually cover 80% of held-out
test points?) is exactly how you'd audit whether that approximation is
trustworthy in practice, rather than just assuming it.

**More rigorous alternatives**, if this were a production system: quantile
regression forests (train the forest to directly predict quantiles rather
than just the mean), or conformal prediction (a model-agnostic framework
with a formal statistical guarantee on coverage). Both are more complex to
implement than what's here, which is a reasonable tradeoff for a portfolio
project but worth naming if asked "how would you improve this."
"""
        )

    st.subheader("Is the Predict Page's Uncertainty Range Trustworthy?")
    cal = diagnostics["calibration"]
    col1, col2 = st.columns(2)
    col1.metric("Stated Interval", f"{cal['stated_interval_pct']}%")
    col2.metric("Empirical Coverage (Test Set)", f"{cal['empirical_coverage_pct']}%")
    st.caption(cal["note"])

    diff = cal["empirical_coverage_pct"] - cal["stated_interval_pct"]
    if abs(diff) <= 3:
        verdict = "well-calibrated — the stated range means roughly what it claims."
    elif diff > 3:
        verdict = f"conservative — the interval is wider than it needs to be, covering {diff:.1f} percentage points more often than stated."
    else:
        verdict = f"overconfident — the interval is narrower than it should be, by {abs(diff):.1f} percentage points."
    st.markdown(f"**Verdict:** the Predict page's uncertainty range is {verdict}")

# ---- Hyperparameter tuning ----
with tab_tuning:
    with st.expander("📚 Why might a \"better\" CV score not translate to a better test score?"):
        st.markdown(
            """
Random Forests have a useful property: because each tree trains on a
random bootstrap sample with a random subset of features, and predictions
are averaged across many trees, the ensemble is naturally resistant to
overfitting even with a lot of trees or deep individual trees — unlike a
single decision tree, where depth needs careful control. This is why
`n_estimators=200` with unlimited depth was a reasonable starting choice
rather than an obviously wrong one.

Grid search picks whichever combination scores best on **cross-validation
folds carved from the training set**. But the final test set is a *single*
held-out sample — its score has some irreducible noise/variance just from
which particular rows happened to land in it. When the true underlying
difference between two hyperparameter settings is smaller than that noise,
grid search can "win" on CV folds by an amount that doesn't reproduce on
the test set, or even reverses slightly, exactly like what happened here.

**The mature reading of a null result:** this is not tuning failing — it's
tuning correctly reporting that this dataset, at this size, doesn't have
enough signal left to meaningfully distinguish between reasonable
hyperparameter choices in this range. Knowing when further optimization
isn't worth pursuing is as valuable a skill as running the optimization.
"""
        )

    st.subheader("Hyperparameter Tuning (GridSearchCV)")
    tuning = load_json(TUNING_PATH)
    if tuning is None:
        st.info(
            "Tuning hasn't been run yet. Run `python tune_hyperparameters.py` "
            "from the project folder (takes a few minutes) to populate this section."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Default Test R²", f"{tuning['default_test_r2']:.4f}")
        col2.metric("Best-Found Test R²", f"{tuning['best_test_r2']:.4f}")
        col3.metric("Improvement", f"{tuning['improvement_over_default']:+.4f}")

        st.markdown(f"**Best params found:** `{tuning['best_params']}`")
        st.markdown(f"**Default params:** `{tuning['default_params']}`")
        st.caption(
            f"Searched {tuning['n_combinations']} combinations × 5-fold CV in "
            f"{tuning['search_time_seconds']:.0f}s."
        )

        if tuning["improvement_over_default"] <= 0.001:
            st.markdown(
                "**Honest takeaway:** tuning found no meaningful improvement over the "
                "existing defaults — the difference is within noise. This is a "
                "legitimate outcome, not a failed search: it means the original "
                "`n_estimators=200` choice was already reasonable for this dataset."
            )
        else:
            st.markdown(
                f"**Takeaway:** tuning found a real improvement of "
                f"{tuning['improvement_over_default']:.4f} R² over the defaults."
            )

        with st.expander("See all combinations tried"):
            results_df = pd.DataFrame(tuning["all_results"])
            params_df = pd.json_normalize(results_df["params"])
            results_df = pd.concat([params_df, results_df[["cv_r2_mean", "cv_r2_std"]]], axis=1)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

# ---- Data quality ----
with tab_quality:
    with st.expander("📚 Why does \"0% missing\" in the raw CSV lie about the real data quality?"):
        st.markdown(
            """
Pandas' `.isnull()` only catches **explicit** nulls — actual `NaN` values
in the data. But real-world datasets often encode missingness
**implicitly**: a sensor that failed might log `0` instead of leaving a
blank, or a reporting country might submit `0` when they mean "not
reported" rather than "verified zero." Naively trusting `.isnull()` alone
would tell you these columns are perfectly complete when they aren't.

**Why this matters for a model:** if left untreated, an implicit zero
gets used as a literal, correct data point during training. For something
like Infant Deaths, a country that simply didn't report the figure would
train the model as if that country had *zero* infant deaths — a very
different (and wrong) signal than "we don't know." This silently biases
the model without throwing any error or warning, which is exactly what
makes it dangerous: nothing about a normal `.isnull().sum()` check would
catch it.

**The general lesson:** always ask *why* a value might be missing and
*how* that missingness could be encoded, not just whether `.isnull()`
returns anything. This is domain-specific — the right implicit-null rules
depend entirely on what's realistic for the field being measured (a
plausible BMI range, a plausible mortality count), not a one-size-fits-all
check.
"""
        )

    st.subheader("Missingness: Before vs. After Cleaning")
    st.caption(
        "How to read this: one group of 3 bars per feature. The story is in the "
        "GAP between the 1st bar (raw CSV) and 2nd bar (after implicit-null "
        "detection) — a big jump there means values that looked \"complete\" "
        "were actually placeholders for missing data. The 3rd bar (after "
        "imputation) should be ~0% everywhere, confirming the pipeline actually "
        "resolved what it found."
    )
    miss_df = pd.DataFrame(diagnostics["missingness"])
    miss_df = miss_df[miss_df["after_implicit_nulls_missing_pct"] > 0].sort_values(
        "after_implicit_nulls_missing_pct", ascending=True
    )

    if miss_df.empty:
        st.info("No missing values found at any pipeline stage for this dataset.")
    else:
        fig_miss = go.Figure()
        fig_miss.add_trace(go.Bar(y=miss_df["column"], x=miss_df["raw_missing_pct"], name="Raw CSV", orientation="h"))
        fig_miss.add_trace(
            go.Bar(
                y=miss_df["column"],
                x=miss_df["after_implicit_nulls_missing_pct"],
                name="After implicit-null detection",
                orientation="h",
            )
        )
        fig_miss.add_trace(
            go.Bar(y=miss_df["column"], x=miss_df["final_missing_pct"], name="After imputation", orientation="h")
        )
        fig_miss.update_layout(barmode="group", xaxis_title="% Missing", height=max(350, 40 * len(miss_df)))
        st.plotly_chart(fig_miss, use_container_width=True)
        st.caption(
            "Notice several columns (e.g. BMI, Infant Deaths, Under-Five Deaths) show "
            "0% missing in the raw CSV but a substantial rate once implicit nulls "
            "(values like a literal 0 standing in for missing data) are correctly "
            "identified — the true missingness was hidden until that step."
        )
