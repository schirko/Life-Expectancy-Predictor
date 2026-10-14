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
    "Deeper analysis behind the headline metrics: residuals, validation on "
    "unseen countries, a head-to-head of four model families, permutation importance, partial dependence, uncertainty "
    "calibration, hyperparameter tuning, and data quality — all computed from "
    "held-out data during training, not invented for display."
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

tab_residuals, tab_group, tab_showdown, tab_perm, tab_pdp, tab_calibration, tab_tuning, tab_quality = st.tabs(
    [
        "Residuals",
        "Unseen Countries",
        "Model Showdown",
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

# ---- Unseen countries (group cross-validation) ----
with tab_group:
    gcv = diagnostics.get("group_cv")
    if not gcv:
        st.info(
            "No group cross-validation results found. Re-run `python train_model.py` "
            "(the current version computes them automatically)."
        )
    else:
        grp_name = gcv["group_column"]
        rnd, grp = gcv["random"], gcv["group"]
        gap_r2 = rnd["r2_mean"] - grp["r2_mean"]
        mae_ratio = grp["oof_mae"] / rnd["oof_mae"]

        st.subheader("Can the Model Predict for a Country It Has Never Seen?")

        st.markdown("#### What We Are Looking For")
        st.markdown(
            f"""
The data has **{gcv['n_groups']} {grp_name.lower()}s**, each with about
**{gcv['avg_rows_per_group']:.0f} rows** (one per year). In a normal random
train/test split, a country's 2005 row can be in the test set while its
2004 and 2006 rows are in the training set. The model has effectively seen
that country before, and countries change slowly from year to year, so those
neighbouring years make the test much easier than it should be.

The question here: **how well does the model do when an entire country is
held out, so it has to predict for a country it knows nothing about?** That
is closer to how the Predict page is really used, because a visitor types in
a combination of values that may not match any country in the training data.
"""
        )

        st.markdown("#### The Code")
        st.code(
            f'''from sklearn.model_selection import KFold, GroupKFold

# Random folds: rows are shuffled, so a country's other years can sit in training
random_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Group folds: every row of a country goes to the SAME fold, so each test
# fold contains only countries the model never saw while training
group_cv = GroupKFold(n_splits=5)
for train_idx, test_idx in group_cv.split(x, y, groups=df["{grp_name}"]):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x[train_idx], y[train_idx])
    out_of_fold_pred[test_idx] = model.predict(x[test_idx])''',
            language="python",
        )

        st.markdown("#### The Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² — Random Folds", f"{rnd['r2_mean']:.3f}")
        m2.metric("R² — Unseen Countries", f"{grp['r2_mean']:.3f}", f"{-gap_r2:+.3f}", delta_color="inverse")
        m3.metric("Avg. Miss — Random Folds", f"{rnd['oof_mae']:.2f} yrs")
        m4.metric(
            "Avg. Miss — Unseen Countries",
            f"{grp['oof_mae']:.2f} yrs",
            f"{grp['oof_mae'] - rnd['oof_mae']:+.2f} yrs",
            delta_color="inverse",
        )

        folds = [f"Fold {i + 1}" for i in range(len(rnd["fold_r2"]))]
        fig_folds = go.Figure()
        fig_folds.add_trace(go.Bar(x=folds, y=rnd["fold_r2"], name="Random folds"))
        fig_folds.add_trace(go.Bar(x=folds, y=grp["fold_r2"], name="Unseen countries"))
        fig_folds.update_layout(
            barmode="group",
            yaxis_title="R² on the held-out fold",
            yaxis_range=[min(min(rnd["fold_r2"]), min(grp["fold_r2"])) - 0.03, 1.0],
            height=380,
        )
        st.plotly_chart(fig_folds, use_container_width=True)

        lower_every_fold = all(g < r for g, r in zip(grp["fold_r2"], rnd["fold_r2"]))
        fold_sentence = (
            "The second bar (unseen countries) is lower than the first in every fold, "
            "so the drop is not a fluke of one lucky or unlucky split."
            if lower_every_fold
            else "The second bar (unseen countries) is not lower in every fold, so "
            "the drop is not uniform across folds."
        )

        st.markdown("#### How to Read It")
        st.markdown(
            f"""
- **The bar chart:** each pair of bars is one of 5 folds, with random folds
  first and unseen countries second. {fold_sentence}
- **The spread:** random folds range from {min(rnd['fold_r2']):.3f} to
  {max(rnd['fold_r2']):.3f}; unseen-country folds range from
  {min(grp['fold_r2']):.3f} to {max(grp['fold_r2']):.3f}. Which countries land
  in a fold matters much more once whole countries are held out.
- **"Avg. miss"** is the mean absolute error: how many years off the
  prediction is, on average, for each row.
"""
        )

        st.markdown("#### Which Countries Are Hardest to Predict?")
        per_group = pd.DataFrame(gcv["per_group_error"])
        fig_err = px.histogram(
            per_group,
            x="mean_abs_error",
            nbins=30,
            labels={"mean_abs_error": f"Average miss for that {grp_name.lower()} (years)"},
        )
        fig_err.add_vline(
            x=gcv["median_group_error"],
            line_color="crimson",
            line_dash="dash",
            annotation_text=f"Median: {gcv['median_group_error']:.2f} yrs",
            annotation_position="top right",
        )
        fig_err.update_layout(yaxis_title=f"Number of {grp_name.lower()}s", height=380)
        st.plotly_chart(fig_err, use_container_width=True)
        st.caption(
            "How to read this: each count is a country. Most countries sit near the "
            "median; the long tail on the right is the small group the model misses "
            "badly when it has never seen them."
        )

        col_worst, col_status = st.columns(2)
        with col_worst:
            st.markdown("**10 Hardest Countries**")
            worst = per_group.head(10).rename(
                columns={"group": grp_name, "mean_abs_error": "Avg. Miss (yrs)", "n_rows": "Rows"}
            )
            st.dataframe(worst, use_container_width=True, hide_index=True)
        with col_status:
            if gcv.get("error_by_status"):
                st.markdown("**Avg. Miss by Country Status**")
                status_df = pd.DataFrame(gcv["error_by_status"]).rename(
                    columns={
                        "status": "Status",
                        "mean_abs_error": "Avg. Miss (yrs)",
                        "n_rows": "Rows",
                        "n_groups": "Countries",
                    }
                )
                st.dataframe(status_df, use_container_width=True, hide_index=True)

        hc = gcv.get("hard_cases")
        if hc:
            conc = hc["concentration"]
            st.markdown("#### Why Are the Hardest Countries Hard? An Investigation")

            st.markdown("**What we are looking for**")
            st.markdown(
                "Three possible explanations, tested one at a time: (1) the country "
                "has little or no real data, so the model is guessing from filler "
                "values; (2) the model pulls its predictions toward the middle, so "
                "the extremes are always missed; (3) the misses cluster around "
                "particular kinds of countries."
            )

            st.markdown("**The code**")
            st.code(
                "# Signed error: positive = model predicted too low, negative = too high\n"
                "err = y - oof_pred                      # oof_pred came from GroupKFold\n"
                "by_country = df.groupby('Country')['err'].agg(['mean', lambda e: e.abs().mean()])\n\n"
                "# 1. Concentration: how much error do the worst 10% of countries carry?\n"
                "# 2. Shrinkage: average signed error within bands of the TRUE value\n"
                "# 3. Missing data: predictors with no data at all for a country\n"
                "#    (before imputation) vs. that country's error",
                language="python",
            )

            st.markdown("**The result**")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                f"Error carried by worst {conc['top_pct']}% of countries",
                f"{conc['share_of_total_error_pct']:.0f}%",
                help=f"{conc['n_top_groups']} countries.",
            )
            m2.metric(
                f"R² without those {conc['n_top_groups']}",
                f"{conc['r2_without_top']:.3f}",
                delta=f"{conc['r2_without_top'] - conc['r2_all']:+.3f}",
            )
            m3.metric(
                f"R² without {conc['hardest_group']} alone",
                f"{conc['r2_without_hardest']:.3f}",
                delta=f"{conc['r2_without_hardest'] - conc['r2_all']:+.3f}",
                help=f"{conc['hardest_group']} is off by {conc['hardest_mae']:.1f} years on average.",
            )

            band_df = pd.DataFrame(hc["by_target_band"])
            band_df["label"] = band_df["band"] + " (" + band_df["range"] + " yrs)"
            fig_band = go.Figure(
                go.Bar(
                    x=band_df["label"],
                    y=band_df["bias"],
                    marker_color=["crimson" if b < 0 else "steelblue" for b in band_df["bias"]],
                    text=[f"{b:+.2f}" for b in band_df["bias"]],
                    textposition="outside",
                )
            )
            fig_band.add_hline(y=0, line_color="gray")
            fig_band.update_layout(
                title="Average Signed Error by True Life Expectancy",
                xaxis_title="Actual life expectancy of the country-year",
                yaxis_title="Actual minus predicted (years)",
                height=380,
            )
            st.plotly_chart(fig_band, use_container_width=True)
            st.caption(
                "How to read this: a red bar below zero means the model predicted too "
                "HIGH for those country-years; a blue bar above zero means it predicted "
                "too LOW. Bars near zero in the middle and large bars at the ends are "
                "the signature of a model that pulls everything toward the average."
            )

            hard_df = pd.DataFrame(hc["hardest"])
            hard_df["Model predicted"] = hard_df["bias"].apply(
                lambda b: "too high" if b < 0 else "too low"
            )
            hard_df = hard_df.rename(
                columns={
                    "group": grp_name,
                    "mean_abs_error": "Avg. Miss (yrs)",
                    "bias": "Signed Error (yrs)",
                    "mean_actual": "Actual Life Exp.",
                    "status": "Status",
                    "empty_predictors": "Inputs With No Data",
                }
            )
            st.markdown(f"**The {len(hard_df)} hardest countries, in detail**")
            st.dataframe(
                hard_df[
                    [grp_name, "Avg. Miss (yrs)", "Signed Error (yrs)", "Model predicted",
                     "Actual Life Exp.", "Status", "Inputs With No Data"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            col_miss, col_corr = st.columns(2)
            with col_miss:
                st.markdown("**Does missing data explain the error?**")
                miss_df2 = pd.DataFrame(hc["missing_data_check"]).rename(
                    columns={
                        "empty_predictors": "Inputs With No Data",
                        "n_groups": "Countries",
                        "mean_abs_error": "Avg. Miss (yrs)",
                    }
                )
                st.dataframe(miss_df2, use_container_width=True, hide_index=True)
                st.caption(
                    f"Rank correlation between a country's error and its number of "
                    f"empty inputs: {hc['missing_data_spearman']:+.2f} (0 = no relationship)."
                )
            with col_corr:
                st.markdown("**Which inputs track the error?**")
                corr_df = pd.DataFrame(hc["feature_error_correlations"]).rename(
                    columns={"feature": "Input", "spearman": "Rank Correlation With Error"}
                )
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
                st.caption("Computed on country averages. Positive = higher input, larger miss.")

            st.markdown("**How to read it**")
            st.markdown(
                "- The three metrics show how much the hardest countries drag the overall "
                "score. The bar chart shows *direction*: errors are not random noise, they "
                "lean one way at each end of the range.\n"
                "- The detail table separates two kinds of miss: the model guessing too "
                "high for countries that did worse than their indicators suggest, and "
                "too low for countries that did better.\n"
                "- In the missing-data table, a real effect would show error rising "
                "steadily as the count of empty inputs rises."
            )

            st.markdown("**What it means**")
            worst_band = hc["by_target_band"][0]
            best_band = hc["by_target_band"][-1]
            hardest_row = hc["hardest"][0]
            miss_flat = abs(hc["missing_data_spearman"]) < 0.15
            top_feat = hc["feature_error_correlations"][0]
            over_ct = sum(1 for r in hc["hardest"] if r["bias"] < 0)
            st.markdown(
                f"""
- **A small group carries a disproportionate share of the error.** The worst
  {conc['top_pct']}% of countries ({conc['n_top_groups']}) produce
  {conc['share_of_total_error_pct']:.0f}% of the total miss. Removing them lifts R²
  from {conc['r2_all']:.3f} to {conc['r2_without_top']:.3f}.
- **{conc['hardest_group']} is a special case, and a data problem.** It is off by
  {conc['hardest_mae']:.1f} years, more than double the next country, and it has
  {hardest_row['empty_predictors']} inputs with no data at all. Those were filled
  with the all-country median, so the model saw an "average" country and predicted
  accordingly. Its R² effect alone is {conc['r2_without_hardest'] - conc['r2_all']:+.3f}.
- **Missing data is {"not" if miss_flat else "partly"} the general explanation.**
  Across all countries the link between empty inputs and error is
  {hc['missing_data_spearman']:+.2f}. {"So the hypothesis fails as a general rule: " + conc['hardest_group'] + " is the exception, not the pattern." if miss_flat else "A relationship exists but is modest."}
- **The model pulls toward the middle.** For the lowest-life-expectancy band
  ({worst_band['range']} years) the model predicts too high by
  {abs(worst_band['bias']):.1f} years on average, and for the highest band
  ({best_band['range']} years) too low by {abs(best_band['bias']):.1f}. Extremes get
  softened. The Model Showdown tab shows gradient boosting and even linear
  regression do the same, so this is mostly about the data: when the inputs only
  partly determine the outcome, any model's estimates regress toward the average.
- **Health-crisis countries are harder.** The input whose country average best
  tracks the miss is **{top_feat['feature']}** (rank correlation
  {top_feat['spearman']:+.2f}). Countries hit hard by epidemics or conflict have
  life expectancy that swings for reasons the indicators capture only partly.
- **Both directions appear.** {over_ct} of the {len(hc['hardest'])} hardest
  countries are over-predicted. The rest are under-predicted, meaning they live
  longer than their income, schooling and spending suggest.
"""
            )

            st.markdown("**What's next and why**")
            st.markdown(
                "- **Done: thin-data warning.** The Predict page now warns when half or "
                "more of the inputs are still at their starting median value, since "
                f"that is exactly what broke {conc['hardest_group']}.\n"
                "- **Done: gradient boosting.** See the Model Showdown tab: it misses the same "
                "countries, so the cause is mostly the data, not the Random Forest.\n"
                "- **Test the shrinkage fix.** Fit a small correction on the out-of-fold "
                "predictions and see whether the extreme-band bias shrinks without hurting "
                "the middle."
            )

        st.markdown("#### What It Means")
        unsh = gcv["unshuffled"]
        st.markdown(
            f"""
- **Both numbers are correct; they answer different questions.** R²
  {rnd['r2_mean']:.3f} answers "how well can the model fill in a country it
  has partly seen?" R² {grp['r2_mean']:.3f} answers "how well can it predict
  for a brand-new country?" The Predict page is closer to the second question.
- **The honest cost is real but not dramatic.** Holding out whole countries
  lowers R² by {gap_r2:.3f} and makes the average miss about
  {mae_ratio:.1f}× larger ({rnd['oof_mae']:.2f} → {grp['oof_mae']:.2f} years).
  The model still explains roughly {grp['r2_mean'] * 100:.0f}% of the variation
  between unseen countries.
- **A correction to the earlier "5-fold CV" explanation.** The plain
  (un-shuffled) 5-fold CV this app used to report scored
  {unsh['r2_mean']:.3f}, close to the unseen-country result. The reason is that
  the rows are sorted by country, so each fold was nearly a block of whole
  countries (only {unsh['groups_straddling_folds']} of {gcv['n_groups']}
  countries straddled a fold boundary). That CV was accidentally testing on
  unseen countries. It was not "more robust than a single split by luck."
- **Errors are uneven.** The typical country is off by about
  {gcv['median_group_error']:.1f} years, but {gcv['pct_groups_over_3yrs']:.0f}%
  of countries are off by more than 3 years, and a few are far worse (see the
  table). A single average hides that.
- **Developed vs. developing:** under this test the average miss is
  {next((s['mean_abs_error'] for s in gcv.get('error_by_status', []) if s['status'] == 'Developed'), float('nan')):.2f}
  years for developed countries and
  {next((s['mean_abs_error'] for s in gcv.get('error_by_status', []) if s['status'] == 'Developing'), float('nan')):.2f}
  for developing ones. That is a hint, not a verdict: there are far fewer
  developed countries, so the comparison is noisy.
"""
        )

        st.markdown("#### What's Next and Why")
        st.markdown(
            """
1. **Done: country-based tuning.** The hyperparameter search was re-run with
   country folds (see the Hyperparameter Tuning tab): no setting beat the
   defaults.
2. **Done: the hardest-countries investigation** (above). Next is a
   gradient-boosting comparison to separate "model" from "data" causes.
3. **Use the same unseen-country setup for a proper developed-vs-developing
   fairness check,** with enough rows per group to say something firm.
4. **A caveat on this test itself:** outlier trimming (winsorizing) and the
   fallback median fills were computed on the whole dataset before the folds
   were made, so a small amount of information still crosses the split. Its
   size has not been measured. Per-country interpolation only uses that
   country's own rows, so it does not leak across countries.
"""
        )

# ---- Model showdown ----
with tab_showdown:
    sd = diagnostics.get("model_showdown")
    st.subheader("Model Showdown: Four Algorithms, Same Unseen Countries")
    if not sd:
        st.info("Not available yet. Re-run `python train_model.py` to compute it.")
    else:
        models_df = pd.DataFrame(sd["models"])
        rf = models_df.iloc[0]
        best = models_df.sort_values("r2_mean", ascending=False).iloc[0]
        boost = models_df[models_df["model"].str.contains("Boosting")]
        lin = models_df[models_df["model"].str.contains("Linear")].iloc[0]

        st.markdown("#### What We Are Looking For")
        st.markdown(
            "The previous investigation found that a handful of countries are missed "
            "badly and that predictions are pulled toward the average. The question "
            "here is whether that is **the Random Forest's fault or the data's**. "
            "If a different algorithm, given the identical country-held-out folds, "
            "misses the same countries by the same amounts, the algorithm is not the "
            "problem."
        )

        st.markdown("#### The Code")
        st.code(
            "models = {\n"
            "    'Random Forest':        RandomForestRegressor(n_estimators=200),\n"
            "    'Gradient Boosting':    GradientBoostingRegressor(),\n"
            "    'Hist Gradient Boosting': HistGradientBoostingRegressor(),\n"
            "    'Linear Regression':    make_pipeline(StandardScaler(), LinearRegression()),\n"
            "}\n"
            "for name, model in models.items():\n"
            "    for train_idx, test_idx in GroupKFold(5).split(x, y, countries):\n"
            "        oof[test_idx] = model.fit(x[train_idx], y[train_idx]).predict(x[test_idx])",
            language="python",
        )

        st.markdown("#### The Result")
        show = models_df.rename(
            columns={
                "model": "Model",
                "r2_mean": "R² (unseen countries)",
                "r2_std": "R² Std. Dev.",
                "mae": "Avg. Miss (yrs)",
                "bias_lowest_10pct": "Error at Lowest 10% (yrs)",
                "bias_highest_10pct": "Error at Highest 10% (yrs)",
                "hardest_country": "Hardest Country",
                "hardest_country_mae": "Its Miss (yrs)",
                "top10pct_error_share": "Error From Worst 10% of Countries (%)",
                "rank_agreement_with_rf": "Agrees With RF on Hard Countries",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)

        fig_r2 = go.Figure(
            go.Bar(
                x=models_df["model"],
                y=models_df["r2_mean"],
                error_y=dict(type="data", array=models_df["r2_std"], visible=True),
                text=[f"{v:.3f}" for v in models_df["r2_mean"]],
                textposition="outside",
                marker_color=["#1f77b4" if i == 0 else "#9aa5b1" for i in range(len(models_df))],
            )
        )
        fig_r2.update_layout(
            title="R² on Countries the Model Has Never Seen (bars = fold-to-fold spread)",
            yaxis_title="R²",
            yaxis_range=[max(0, models_df["r2_mean"].min() - 0.08), 1],
            height=380,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        fig_bias = go.Figure()
        fig_bias.add_trace(go.Bar(name="Lowest 10% of life expectancy", x=models_df["model"],
                                  y=models_df["bias_lowest_10pct"], marker_color="crimson"))
        fig_bias.add_trace(go.Bar(name="Highest 10% of life expectancy", x=models_df["model"],
                                  y=models_df["bias_highest_10pct"], marker_color="steelblue"))
        fig_bias.add_hline(y=0, line_color="gray")
        fig_bias.update_layout(
            title="Average Signed Error at the Extremes (actual minus predicted)",
            yaxis_title="Years", barmode="group", height=380,
        )
        st.plotly_chart(fig_bias, use_container_width=True)
        if sd.get("hard_in_every_model"):
            st.markdown(
                f"**Countries in every model's {sd['n_top_countries']} hardest:** "
                + ", ".join(sd["hard_in_every_model"])
            )

        st.markdown("#### How to Read It")
        st.markdown(
            "- In the R² chart, taller is better, and the small whiskers show how much "
            "the score moves from fold to fold. Differences smaller than the whiskers "
            "are not real differences.\n"
            "- In the error chart, a red bar below zero means the model predicts too "
            "HIGH for the lowest-life-expectancy country-years; a blue bar above zero "
            "means too LOW for the highest. Bars near zero would mean no pull toward the "
            "average.\n"
            "- \"Agrees with RF\" is a rank correlation: 1.0 means the two models find "
            "exactly the same countries hard."
        )

        st.markdown("#### What It Means")
        gap = best["r2_mean"] - rf["r2_mean"]
        boost_close = bool((abs(boost["r2_mean"] - rf["r2_mean"]) <= rf["r2_std"]).all())
        st.markdown(
            f"""
- **No algorithm change fixes the problem.** The best model here is
  **{best['model']}** at R² {best['r2_mean']:.3f}, only {gap:+.3f} from the deployed
  Random Forest ({rf['r2_mean']:.3f}). {"The gradient-boosting scores sit inside the fold-to-fold spread (about " + f"{rf['r2_std']:.3f}" + "), so they are effectively tied." if boost_close else "At least one boosting model differs by more than the fold-to-fold spread."}
- **The same countries are hard for everyone.** {len(sd.get('hard_in_every_model', []))} countries
  appear in the worst {sd['n_top_countries']} for every model, including
  {rf['hardest_country']}, which each tree-based model misses by about
  {boost['hardest_country_mae'].min():.0f} to {rf['hardest_country_mae']:.0f} years.
  That points at the data, not the algorithm.
- **The pull toward the average is not a Random Forest quirk.** Both boosting
  models and linear regression under-predict the highest and over-predict the
  lowest life expectancies. Linear regression does it most
  ({lin['bias_lowest_10pct']:+.1f} years at the low end,
  {lin['bias_highest_10pct']:+.1f} at the high end). When inputs only partly
  determine the outcome, every model regresses toward the middle.
- **The Random Forest is a sound choice, not a lucky one.** It is within noise of
  the best model, much better than linear regression (R² {lin['r2_mean']:.3f}), and
  it supports the SHAP explanations and tree-spread interval used elsewhere in the
  app. Switching algorithms would buy nothing measurable and cost both.
"""
        )

        st.markdown("#### What's Next and Why")
        st.markdown(
            "- **Add information the inputs lack.** The hardest countries are ones where "
            "conflict, epidemics or reporting gaps move life expectancy in ways these "
            "13 indicators do not capture. Better inputs would help more than a better "
            "algorithm, but they would have to come from outside this dataset.\n"
            "- **A proper developed-vs-developing fairness check** on the unseen-country "
            "setup, with enough countries per group to say something firm.\n"
            "- **Remove the small winsorizing leak** (limits computed on all data before "
            "the folds) and measure whether it changes any of these numbers."
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
    st.markdown(f"**Verdict (random test split):** the Predict page's uncertainty range is {verdict}")

    unseen_cal = (diagnostics.get("group_cv") or {}).get("interval_unseen")
    if unseen_cal:
        st.markdown("#### The Same Check on Countries the Model Has Never Seen")
        ucol1, ucol2 = st.columns(2)
        ucol1.metric("Stated Interval", f"{unseen_cal['stated_pct']}%")
        ucol2.metric(
            "Coverage on Unseen Countries",
            f"{unseen_cal['coverage_pct']}%",
            f"{unseen_cal['coverage_pct'] - unseen_cal['stated_pct']:+.1f} pts",
            delta_color="off",
        )
        udiff = unseen_cal["coverage_pct"] - unseen_cal["stated_pct"]
        if abs(udiff) <= 3:
            uverdict = (
                f"close to what it claims, "
                f"{'slightly under' if udiff < 0 else 'slightly over'} by {abs(udiff):.1f} points."
            )
        elif udiff > 3:
            uverdict = f"conservative, covering {udiff:.1f} points more often than stated."
        else:
            uverdict = f"overconfident, covering {abs(udiff):.1f} points less often than stated."
        st.markdown(
            f"**Verdict (unseen countries):** the range is {uverdict} Its average width "
            f"is {unseen_cal['mean_width_years']:.1f} years."
        )
        st.caption(
            "Why this matters: the random-split check above lets the same countries "
            "appear in training and testing, which flatters the interval (it looked "
            "conservative). A visitor to the Predict page is usually asking about "
            "something the model has not seen, so the unseen-country figure is the "
            "fairer reading. See the Unseen Countries tab for how the test is built."
        )

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

    st.subheader("Hyperparameter Tuning (GridSearchCV, Country-Held-Out Folds)")
    tuning = load_json(TUNING_PATH)
    if tuning is None:
        st.info(
            "Tuning hasn't been run yet. Run `python tune_hyperparameters.py` "
            "from the project folder (takes a few minutes) to populate this section."
        )
    else:
        grouped = "cv_method" in tuning
        results_list = tuning["all_results"]
        spread = results_list[0]["cv_r2_mean"] - results_list[-1]["cv_r2_mean"]
        fold_sd = results_list[0]["cv_r2_std"]

        st.markdown("#### What We Are Looking For")
        st.markdown(
            "Whether any Random Forest settings (number of trees, tree depth, "
            "minimum rows per leaf) predict **countries the model has never seen** "
            "better than the defaults. The search was re-run with every fold, and "
            "the final test set, holding out whole countries."
        )

        st.markdown("#### The Code")
        st.code(
            "from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit\n\n"
            "# Hold out ~20% of countries as the final test set\n"
            "train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=42)\n"
            "                           .split(x, y, groups))\n\n"
            "# Every CV fold also holds out whole countries\n"
            "grid = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,\n"
            "                    cv=GroupKFold(n_splits=5), scoring='r2', n_jobs=-1)\n"
            "grid.fit(x[train_idx], y[train_idx], groups=groups[train_idx])",
            language="python",
        )

        st.markdown("#### The Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Default Test R² (unseen countries)", f"{tuning['default_test_r2']:.4f}")
        col2.metric("Best-Found Test R²", f"{tuning['best_test_r2']:.4f}")
        col3.metric("Improvement", f"{tuning['improvement_over_default']:+.4f}")
        st.markdown(f"**Best params found:** `{tuning['best_params']}`")
        st.markdown(f"**Default params:** `{tuning['default_params']}`")
        st.caption(
            f"Searched {tuning['n_combinations']} combinations × 5 folds in "
            f"{tuning['search_time_seconds']:.0f}s."
            + (
                f" Trained on {tuning['n_train_countries']} countries, tested on "
                f"{tuning['n_test_countries']} unseen ones."
                if grouped else ""
            )
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Best grouped-CV R²", f"{tuning['best_cv_r2']:.4f}")
        c2.metric("Spread across all 36 settings", f"{spread:.4f}")
        c3.metric("Fold-to-fold noise (std)", f"{fold_sd:.4f}")

        rc = tuning.get("rank_comparison")
        if rc:
            st.markdown("**Did switching to country folds change the picture?**")
            st.markdown(
                f"- Random-fold winner: `{rc['random_cv_best_params']}`; country-fold winner: "
                f"`{tuning['best_params']}` — "
                + ("the same setting.\n" if rc["same_winner"] else "a different setting.\n")
                + f"- Best-vs-worst gap: {rc['random_cv_best_r2'] - rc['random_cv_worst_r2']:.4f} R² "
                f"with random folds, {tuning['best_cv_r2'] - rc['grouped_cv_worst_r2']:.4f} with country folds.\n"
                f"- Rank agreement between the two methods (Spearman): {rc['spearman_rank_correlation']:.2f} "
                "(1.0 = identical ordering)."
            )

        st.markdown("#### How to Read It")
        st.markdown(
            "The three small numbers are the key. If the *spread* across all settings "
            "is much smaller than the *fold-to-fold noise*, the search cannot tell the "
            "settings apart: differences that small would likely reshuffle with a "
            "different set of countries. The expander below lists every combination."
        )

        st.markdown("#### What It Means")
        if abs(tuning["improvement_over_default"]) <= 0.005 or spread < fold_sd:
            st.markdown(
                f"**Honest takeaway:** no setting is meaningfully better. The best and "
                f"worst of {tuning['n_combinations']} combinations differ by only {spread:.4f} R², "
                f"about {spread / fold_sd:.0%} of the {fold_sd:.3f} fold-to-fold noise, and the "
                f"tuned model scores {tuning['improvement_over_default']:+.4f} against the default on "
                "unseen countries. The original `n_estimators=200` default stands. "
                "What limits accuracy on new countries is the information in the "
                "features, not the forest's settings."
            )
        else:
            st.markdown(
                f"**Takeaway:** tuning found an improvement of "
                f"{tuning['improvement_over_default']:.4f} R² on unseen countries."
            )

        st.markdown("#### What's Next and Why")
        st.markdown(
            "- Try a different model family (gradient boosting) in the comparison: if "
            "settings do not matter, the algorithm might.\n"
            "- Look at the hardest countries (Unseen Countries tab): their errors, not "
            "the hyperparameters, dominate the unseen-country gap.\n"
            "- Test with a different random country split to see how much the 0.90 "
            "figure itself moves."
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
