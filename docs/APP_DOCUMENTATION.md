# Life Expectancy Predictor — App Documentation

> Snapshot exported from the live documentation Doc on 2026-10-13. The Doc is the master copy; refresh this file after each milestone.
>
> **2026-11-11:** milestone 13 ("Start from a real country") was written directly into this file. The live Doc does not have it yet and needs the same additions: the paragraph under "Using the Predict Page" and build-log row 13.

A Streamlit app that explores WHO life expectancy data (2000–2015) and predicts life expectancy from health and economic indicators using a Random Forest model, with SHAP-based explanations for each prediction.

## Overview

The Life Expectancy Predictor has these pages, reachable from the sidebar:

- **Dashboard** — explore the WHO dataset: trends over time, country rankings, a feature correlation heatmap, and the distribution of life expectancy, all filterable by year range, country status (Developed/Developing), and specific countries.
- **Predict** — enter values for 12 health/economic indicators (via sliders and direct-entry fields) and get a predicted life expectancy from a trained Random Forest, along with a SHAP explanation showing which inputs pushed that specific prediction up or down.
- **Diagnostics** — deeper analyses of the trained model: residuals, validation on unseen countries, permutation importance, partial dependence, uncertainty calibration, hyperparameter tuning, and data quality. See "Model Diagnostics: Deeper Analysis" below.

The project started as a graduate data-science coursework notebook (cleaning, EDA, and several model types compared) and was rebuilt into this deployed, interactive app.

## Architecture

The app is config-driven: dataset-specific knowledge (column names, cleaning rules, predictor list, UI labels) lives in a YAML config file, not hardcoded in Python.

| File | Purpose |
| --- | --- |
| `Home.py` | Landing page: dataset overview, model metrics, methodology, limitations |
| `pages/1_Dashboard.py` | Interactive exploration: trends, rankings, correlations |
| `pages/2_Predict.py` | Prediction form + SHAP explanation |
| `pages/3_Diagnostics.py` | Displays the precomputed diagnostics |
| `config/who_life_expectancy.yaml` | All dataset-specific knowledge: renames, null rules, winsorize limits, predictors, UI hints |
| `config_loader.py` | Loads a config and validates it against a CSV's actual columns |
| `data_prep.py` | Generic cleaning pipeline (reads whatever config it's given) |
| `train_model.py` | Trains the Random Forest, runs country-grouped cross-validation, computes diagnostics, saves model + metadata |
| `tune_hyperparameters.py` | Optional grid search with country-held-out folds |
| `model/` | Trained model, metadata, diagnostics and tuning results |

Because the Python code contains no dataset-specific logic, swapping in a similarly shaped WHO-style CSV only requires writing a new config file and re-running training — see "Swapping in a New Dataset".

## Setup and Installation

1. Create and activate a virtual environment (Python 3.10–3.12 recommended; very new Python releases can lag on `shap`'s compiled dependencies):

   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   (In PowerShell: `.\venv\Scripts\Activate.ps1`)
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Train the model (validates the config, cleans the data, trains, cross-validates, computes diagnostics, saves outputs to `model/` and `data/clean_data.csv`):

   ```
   python train_model.py
   ```
4. Run the app:

   ```
   streamlit run Home.py
   ```
5. Open `http://localhost:8501` in a browser.

**Deployment:** since 2026-11-11 the app runs on Scott's own server (a VPS managed with Coolify) at <https://lifeexpectancy.intellidec.net>, built from the GitHub repo at [github.com/schirko/LifeExpectancyApp](https://github.com/schirko/LifeExpectancyApp). See "Deployment" below.

## Data and Methodology

Source: WHO/UN life expectancy dataset, 2,929 country-year rows across ~193 countries, 2000–2015.

The cleaning pipeline (`data_prep.py`, config-driven):

1. **Normalize headers** — strip, title-case, and apply configured renames (e.g. `Hiv/Aids` → `HIV_AIDS`)
2. **Fix implicit nulls** — values that are technically present but really mean "missing" (e.g. a literal `0` in Infant Deaths, or a BMI outside a plausible range) are converted to actual nulls
3. **Impute** — missing values are filled by interpolating each country's own time series first, falling back to that country's median, then the global median
4. **Winsorize** — extreme outliers are trimmed per feature (specific percentile limits per column, tuned in the original notebook)
5. **Drop** highly collinear or unreliable columns (e.g. GDP, Population, BMI) before training

After cleaning: 2,928 model-ready rows (one row dropped for a null target).

## Model and Performance

The deployed model is a **Random Forest Regressor** (200 trees), chosen after comparing approaches in the original notebook:

| Model | Performance |
| --- | --- |
| Simple linear regression (Schooling only) | R² = 0.55 |
| Multiple linear regression | R² = 0.85 |
| Polynomial regression | R² ≈ 0.57 (plateaus past degree 3) |
| Decision tree | R² = 0.91 |
| **Random Forest (deployed)** | **R² = 0.96** |
| Logistic regression (High/Low classification) | 88% accuracy |

**Two performance readings are reported, and they tell different stories:**

- **Single 80/20 train/test split:** R² = 0.962, MAE = 1.13 years, RMSE = 1.84 years
- **Country-grouped cross-validation** (GroupKFold, 5 folds; each fold holds out whole countries, so it measures accuracy on countries the model never saw): R² = 0.902 (± 0.020). The same model under ordinary random 5-fold CV scores 0.961 (± 0.006).

The gap between these two numbers is expected and worth understanding: the data is a panel (each country appears in up to 16 years), so a random split puts a country's other years in training and partly rewards memorizing countries. Holding out whole countries removes that leakage. An earlier version of this document attributed the gap to split luck; that was wrong.

**Top predictors** by built-in feature importance: HIV/AIDS (58.5%), Income Composition of Resources (17.5%), and Adult Mortality (14.2%) dominate — all three plausibly reflect a country's overall health-system quality.

## Auditing the Model Comparison Table

The Home page's Model Comparison section has an expandable **"See the technical work behind each model"** panel showing the actual notebook code behind each of the 6 models' performance numbers — not just the final figures, but how they were produced. Auditing that code turned up two findings worth documenting.

**What overfitting means, briefly:** a model overfits when it starts fitting noise specific to its training data rather than the underlying pattern — it keeps looking better on training data while getting *worse* on data it hasn't seen. The standard way to catch it is comparing training-set performance against held-out test-set performance: a small, stable gap is healthy; a growing gap as model complexity increases is the warning sign.

**Finding 1 — a corrected claim about polynomial regression.** An earlier draft of this documentation and the app's own summary text claimed the polynomial regression model "overfits past degree 3." Re-running the comparison (degrees 1 through 25) shows this was inaccurate: the train/test R² gap stays under 0.02 at every degree tested, never approaching the notebook's own 0.05 overfitting threshold. With only one predictor (Schooling), there simply isn't enough model complexity for meaningful overfitting to occur — performance plateaus around R² ≈ 0.57 at degree 3 and higher degrees add negligible benefit. Both the app and this doc have been corrected to reflect the plateau, not a false overfitting narrative.

**Finding 2 — a real but harmless methodology note in the logistic regression cell.** That cell normalizes its 12 predictors (min-max scaling) *before* splitting into train/test sets, meaning the scaling statistics technically include test-set data — a mild, common form of data leakage (best practice fits a scaler on training data only, then applies those same statistics to the test set). Recomputing with train-only statistics gives an identical 88.05% accuracy, so it made no practical difference here, but it's a pattern worth avoiding on principle, and worth being able to explain if asked.

## Model Diagnostics: Deeper Analysis

The app's Diagnostics page goes beyond the headline R² figure with eight kinds of analysis, each computed for real against held-out data (not asserted). This section explains what each one is, why it matters, and what it actually found.

### Residual Analysis

A single R² hides *where* a model is wrong. Residual analysis (residual = actual − predicted) checks for patterns: an actual-vs-predicted plot that fans away from the diagonal, or a residuals-vs-predicted plot with a funnel shape (heteroscedasticity) or curve, both signal a real weakness a summary statistic wouldn't reveal. **Finding:** mean residual is 0.02 years (essentially zero — no systematic over- or under-prediction) with a standard deviation of 1.84 years, consistent with the reported test RMSE.

### Permutation Importance vs. Built-In (Gini) Importance

The Random Forest's default `feature_importances_` credits a feature every time it's used for a split, which is known to be biased toward continuous/high-cardinality features. Permutation importance instead shuffles one feature at a time in the test set and measures the resulting drop in R² — model-agnostic, and immune to that bias. **Finding:** built-in importance shows HIV/AIDS (58%) dominating Adult Mortality (14%) roughly 4-to-1; permutation importance shows them nearly tied (26% vs. 24%). The permutation version is the one to trust when they disagree.

### Partial Dependence Plots

Importance (either kind) answers "how much does this feature matter," a single number. Partial dependence answers "what shape is the relationship" — holding every other feature fixed and sweeping one feature across its range to see how the average prediction responds. Computed here for the top 3 features by permutation importance. Caveat worth naming if asked: partial dependence assumes features vary independently, which can be misleading when features are correlated in reality, as several health indicators in this dataset are.

### Uncertainty Calibration

A prediction interval is calibrated if it means what it claims — an 80% interval should contain the true value ~80% of the time. The Predict page's tree-spread range (10th–90th percentile across the forest's 200 trees) is a cheap, intuitive approximation, not a statistically rigorous interval like quantile regression forests or conformal prediction would provide. **Finding:** checked empirically against the test set, on a random test split the stated 80% interval covers about 88% of true values, which looks conservative. That figure is flattered by same-country overlap: on countries the model has never seen, coverage drops to 77.7% (mean width 6.65 years), slightly overconfident but within 3 points of the target.

### Unseen Countries (Group Cross-Validation)

The question: how well does the model predict a country it has never seen? Random splits cannot answer it, because each country has up to 16 yearly rows and the other years land in training. Group cross-validation (scikit-learn `GroupKFold`, grouped by Country) keeps every row of a country in the same fold.

**Finding:** R² falls from 0.961 (random 5-fold) to 0.902 (country-grouped), and the average miss grows from 1.16 to 2.11 years. The median country is off by 1.96 years, and 13.1% of countries are off by more than 3 years. The Diagnostics page lists the hardest countries and compares developed and developing groups.

**Why the earlier 0.910 was misread:** the original 5-fold CV ran unshuffled on country-sorted data, so folds were accidentally country-like (only 4 of 183 countries straddled folds). That is why it sat near the grouped score, not because of split luck.

**Caveat:** winsorizing limits and fallback medians were computed on the full dataset before splitting, a small leak worth removing in a later pass.

**Why are the hardest countries hard?** Three explanations were tested against the out-of-fold predictions. (1) *Concentration:* the worst 10% of countries (18) carry 22% of total error; removing them lifts unseen-country R² from 0.906 to 0.925. (2) *Missing data:* Somalia, off by 11.4 years (more than double any other country), has no data for three inputs (Ttl_Expend, Income_Comp_Of_Resources, Schooling), which were filled with the all-country median, so the model saw an average country. But across all countries the link between empty inputs and error is -0.01, so missing data explains Somalia, not the pattern. (3) *Shrinkage toward the middle:* in the lowest life-expectancy band (46-55 years) the model predicts 2.6 years too high on average; in the highest band (80-89) it predicts 0.7 years too low, and the Model Showdown below shows gradient boosting and even linear regression do the same, so this is mostly about the data: when inputs only partly determine the outcome, every model regresses toward the average. The input whose country average best tracks error is HIV_AIDS (rank correlation +0.40). Countries such as Republic of Korea, Cuba and Guatemala are under-predicted: they live longer than their indicators suggest.

### Model Showdown (Four Algorithms, Same Unseen Countries)

The question: are the hardest countries hard because of the Random Forest, or because of the data? Four model families were scored on the identical country-held-out folds (`GroupKFold`, 5 folds), so only the algorithm differs: Random Forest (deployed), Gradient Boosting, Hist Gradient Boosting, and Linear Regression (scaled).

**Finding:** the algorithm does not matter much. R² on unseen countries is 0.902 (Random Forest), 0.895 (Gradient Boosting), 0.904 (Hist Gradient Boosting) and 0.831 (Linear Regression). The three tree models are within the fold-to-fold spread of about 0.02, so they are effectively tied. Seven countries (Côte d'Ivoire, Dominican Republic, Ethiopia, Jamaica, Sierra Leone, Somalia, Swaziland) are in the worst 18 for every model, and Somalia is missed by 10 to 11 years by all three tree models. All four models under-predict the highest and over-predict the lowest life expectancies; linear regression does it most (-3.6 years at the low end, +3.0 at the high end). Conclusion: the cause is mostly the data, and the Random Forest is a sound choice that also supports SHAP and the tree-spread interval.

### Hyperparameter Tuning (GridSearchCV)

Random Forests resist overfitting even with many/deep trees, because each tree trains on a random bootstrap sample and predictions are averaged — unlike a single decision tree, where depth needs careful capping. A 36-combination grid search (`n_estimators`, `max_depth`, `min_samples_leaf`, 5-fold CV) found the best CV-scoring configuration (300 trees, max_depth=15) actually scored marginally *lower* on the held-out test set (0.9616) than the original default (0.9617). **Reading this correctly:** this isn't a failed search — cross-validation fold scores and a single held-out test score both carry some noise, and when the true difference between configurations is smaller than that noise, grid search can "win" on CV without that gain reproducing on test data. The original `n_estimators=200` default was already reasonable.

**Re-run with country-held-out folds:** the search above used random folds, which leak a country's other years into training. Repeating it with `GroupKFold` (and a test set of 37 unseen countries) gave the same message more strongly. The best setting (`max_depth=15`, 200 trees) scored 0.886 in grouped CV, but all 36 combinations fell within 0.0016 R² of each other while fold-to-fold noise was 0.043. On unseen countries the tuned model scored 0.9044 against 0.9049 for the default. The winner changed from the random-fold search, and the two methods agree only moderately on ordering (Spearman 0.59). Conclusion: forest settings are not what limits accuracy on new countries.

### Data Quality: Implicit Missingness

`.isnull()` only catches explicit `NaN` values, missing real-world missingness encoded as a placeholder (e.g. a literal `0` standing in for "not reported"). Left untreated, a model trains on that placeholder as if it were a true, verified value. **Finding:** several columns show 0% missing by a naive `.isnull()` check but a substantial rate once implicit nulls are correctly identified — BMI (49.4%), Infant Deaths (28.6%), and Under-Five Deaths (26.5%) — all of which drop to 0% only after the pipeline's interpolation/imputation step. The true data-quality picture was invisible until this step ran.

## Using the Dashboard

Sidebar filters — Year range, Country status (Developed/Developing), and Countries — narrow every chart on the page at once. The Countries list only offers countries matching the currently-selected status, so the two filters can't contradict each other.

**What each chart shows:**

- **Life Expectancy Over Time** — average life expectancy by year, split by Developed vs. Developing status. Use this to see the gap between the two groups and whether it's narrowing.
- **Country Rankings** — top/bottom N countries by mean or median life expectancy, within the current filters.
- **What Drives the Model's Predictions** — the Random Forest's global feature importances, shown as a bar chart for quick scanning.
- **Correlation Between Features** — a heatmap of pairwise correlations across all numeric columns. Useful for sanity-checking the model's feature importances against raw correlation, and for spotting collinearity.
- **Distribution of Life Expectancy** — a histogram with a marginal box plot, split by status, showing the overall shape and spread of the target variable.

## Using the Predict Page and Interpreting SHAP

Enter values via 7 sliders (bounded/rate-style indicators like Adult Mortality and immunization percentages) and 5 direct-entry fields (Alcohol, expenditure, Income Composition, Schooling). A 13th feature, Measles, feeds the model at its dataset median but has no visible control. An expandable glossary defines every indicator in plain English.

**Start from a real country (added 2026-11-11):** a box at the top of the page lets a visitor pick a country and a year and click **Fill in these values**. The controls then show that country-year's values from the cleaned data, and a green line states the recorded life expectancy. From there the visitor can change any input to ask a what-if question. **Reset to typical values** puts every control back to the all-country median. Three things to know:

- With no inputs changed, the result compares the model's estimate with the recorded figure. A close match is expected, because the model saw most country-years in training; the page says so and points to Diagnostics → Unseen Countries.
- With inputs changed, the result shows the estimate with the changes, the difference from the model's estimate for the country as recorded, and a table of what was changed. A caption states that this shows how the model's estimate moves, not what would happen in that country.
- "Recorded values" come from the cleaned data, so where the original records had a gap the value is the pipeline's fill-in (interpolation or a median). The page says this in one sentence.

How it works: every Streamlit control stores its current value in `st.session_state` under its `key`. The page seeds those entries with the medians on the first visit, and the two buttons run small functions (`load_country`, `reset_inputs`) that overwrite the entries before the page redraws. The controls no longer pass `value=`, because a control cannot take a starting value from both places. The country's values are also kept in `st.session_state["loaded_from"]` so the page can tell which inputs were changed. The feature only appears when the data has `Country` and `Year` columns.

Clicking **Predict** returns three things, in order:

1. **Predicted life expectancy**, with an uncertainty range from the spread across the Random Forest's 200 individual trees (~80% of trees fell within the stated range) — a rough indicator, not a statistically rigorous confidence interval.
2. **Why This Prediction (SHAP)** — a bar chart specific to the values just entered. Green bars pushed this one prediction above the model's average (baseline); red bars pushed it below. This answers "why did THIS prediction come out this way," as opposed to the global importances, which answer "what matters most across ALL predictions." The two can legitimately disagree for an individual case — e.g. a country with very low Alcohol consumption might see that feature barely matter globally but weigh heavily for that specific prediction.
3. **Overall Feature Importance** — the same global importances shown on the Dashboard, included here for direct comparison against the SHAP chart above it.

**How to read a SHAP result:** the chart's title states the baseline (average prediction across training data) and the final prediction. Each bar's length is that feature's contribution, in years, toward moving from baseline to final prediction. Bars are sorted so the most influential feature for that specific case appears at the top.

**Thin-data warning:** the Predict page counts how many inputs are still at their starting value (the dataset median) and shows that count under the controls. If half or more are untouched, the result is preceded by a warning, because the model treats median values as typical and the prediction drifts toward the all-country average. This comes from the hardest-countries investigation: the model's biggest miss, Somalia, had three inputs with no data that were filled with medians in exactly this way. The threshold (`THIN_DATA_SHARE = 0.5`) is a constant at the top of that block in `pages/2_Predict.py`.

## Limitations to Keep in Mind

- **Correlation, not causation.** A strong predictor (e.g. Adult Mortality) reflects a statistical relationship, not a proven causal mechanism.
- **Historical data.** The dataset ends in 2015; predictions reflect relationships from that period and may not capture more recent shifts in global health.
- **Imputed values.** Some indicators were filled via interpolation where the original data had gaps, introducing some uncertainty into those specific country-years.
- **Model uncertainty.** The tree-spread range on the Predict page is a rough indicator, not a rigorous confidence interval.
- **Single-split vs. cross-validation.** The 0.962 single-split R² partly reflects same-country leakage; the 0.902 country-grouped R² is the more trustworthy figure for new countries. Typical error for unseen countries is about 2.1 years versus 1.2 for the random split.
- **Dashboard filters assume WHO-style panel data.** The Country/Year/Status filters are structural, not config-driven — a genuinely different dataset shape would need its own dashboard filters, even though the cleaning pipeline and Predict page would work unchanged.

## Swapping in a New Dataset

To point the app at a different, similarly-shaped WHO-style CSV:

1. Copy `config/who_life_expectancy.yaml` to a new file
2. Edit `dataset.source_csv`, `target_column`, `column_renames`, `winsorize_limits`, `drop_columns`, and the `predictors` list (label, description, slider vs. number, and whether to show it in the UI) to match the new CSV
3. Run `python train_model.py config/your_new_config.yaml`
4. If the config doesn't match the CSV's actual columns, training fails immediately with a specific list of what's missing, rather than a cryptic error partway through cleaning
5. Restart the app — the Predict page's controls, labels, and descriptions update automatically from `model/metadata.json`, with no page code to touch

## Deployment

Since 2026-11-11 the app runs on Scott's own server, a VPS managed with **Coolify**, at <https://lifeexpectancy.intellidec.net>. It still uses the Streamlit *library* (a Python package installed on the server); it no longer depends on Streamlit's hosting service.

| Setting | Value |
| --- | --- |
| Source | [github.com/schirko/LifeExpectancyApp](https://github.com/schirko/LifeExpectancyApp), branch `master`, through Coolify's GitHub App |
| Build | `Dockerfile` at the top of the repository (Python 3.12, installs `requirements.txt`, starts `streamlit run Home.py`) |
| Port | 8501 |
| Address | `https://lifeexpectancy.intellidec.net` (an A record for `lifeexpectancy` points at the VPS; Coolify obtains the HTTPS certificate) |
| Health check | `/_stcore/health`, defined in the `Dockerfile` |
| Secrets | None. The app reads only files in the repository. |

To update the live app: push to `master`, then redeploy in Coolify (automatic if Coolify's auto-deploy is on for this resource). `.dockerignore` keeps the virtual environment, notes and editor settings out of the build.

**Before 2026-11-11** the app was hosted on Streamlit Community Cloud, which redeployed on every push to `master`. That copy is to be deleted once the server copy is confirmed.

## Build Log: What, Why and How

Each row is one milestone, in the order built. Add a row for every new milestone so the project stays interview-ready.

| Step | What | Why | How |
| --- | --- | --- | --- |
| 1. Dashboard and Predict pages | Streamlit multipage app: charts of trends, correlations and rankings, plus a form that predicts life expectancy | Gives an interviewer something to click, not just a notebook | `Home.py` plus `pages/`; Plotly charts; a saved Random Forest (`rf_model.joblib`) loaded by the Predict page |
| 2. Deployment | Public app on Streamlit Community Cloud, fed from GitHub | A live link is easier to share than a repo | Push to `master` on `schirko/LifeExpectancyApp`; the host redeploys on each push; classic personal access token (`repo` scope) for git auth |
| 3. SHAP explanations | Per-prediction breakdown of which inputs pushed the estimate up or down | A single number is hard to trust without reasons | `shap.TreeExplainer` on the Random Forest, shown on the Predict page |
| 4. Config-driven refactor | Dataset, target and predictors defined in `config/who_life_expectancy.yaml`; generic cleaning in `data_prep.py` | Lets a second dataset be swapped in without rewriting pages | `config_loader.py` validates the YAML against the CSV's columns; cleaning covers header normalization, per-country interpolation, winsorizing and column drops |
| 5. Model comparison audit | Expandable code and notes for each of the six notebook models; overfitting explained | The student-era notebook may contain mistakes; an interviewer will probe them | Re-ran each model, corrected the claim that polynomial regression overfits past degree 3, noted the scaler fit before the split (no practical effect) |
| 6. Diagnostics page | Residuals, permutation importance, partial dependence, uncertainty calibration, tuning results, data quality | Shows the model was checked, not just fitted | `train_model.py` computes everything into `model/diagnostics.json`; `pages/3_Diagnostics.py` only displays it |
| 7. Plain-language explanations | Every chart has what we are looking for, how to read it, what it means, and what's next | Makes findings understandable to non-specialists and to the author while learning | Fixed headings on each tab, with some sentences filled from the computed numbers |
| 8. Unseen Countries test | Group cross-validation that holds out whole countries | Random splits leak a country's other years into training and flatter the score | `GroupKFold(5)` by Country; R² 0.961 random vs 0.902 grouped; interval coverage 77.7% on unseen countries; the earlier 0.910 explanation was corrected |
| 9. Tuning with country folds | Re-ran the 36-combination grid search with whole countries held out in every fold and in the test set | Random folds flatter the score and may pick settings that memorize countries; the milestone-8 finding called for a fair re-test | `tune_hyperparameters.py` now uses GroupKFold(5) and GroupShuffleSplit; result: 0.0016 R² spread vs 0.043 fold noise, tuned 0.9044 vs default 0.9049, so keep the defaults |
| 10. Hardest-countries investigation | Tested why some countries are missed badly: error concentration, missing data, and shrinkage toward the middle | Knowing where and why a model fails is more convincing in an interview than a single score, and it shows which fix is worth building | Out-of-fold predictions from GroupKFold, grouped by country; signed error by band of true value; Spearman correlations; result: worst 10% of countries carry 22% of error, Somalia is a data problem, extremes are pulled toward the average |
| 11. Thin-data warning | Predict page counts untouched inputs and warns when half or more are still at the median | Median-filled inputs made the model predict an average country (Somalia, 11 years off); users should know when they are in that situation | `pages/2_Predict.py` records each control's default, compares entered values to it, shows a live count, and adds an `st.warning` before the result; tested headlessly with all-default and mostly-changed inputs |
| 12. Model Showdown | Scored Random Forest, Gradient Boosting, Hist Gradient Boosting and Linear Regression on identical country-held-out folds | Separates "the algorithm is the problem" from "the data is the problem" before spending effort on a different model | `compute_model_showdown()` in `train_model.py` reuses `GroupKFold(5)`; result: R² 0.902 / 0.895 / 0.904 / 0.831, same hard countries across models, so keep the Random Forest |
| 13. Start from a real country | Predict page can fill its 12 controls from any country and year in the data, then compares the estimate with the recorded figure or shows the effect of the visitor's changes | Visitors do not know indicator values by heart; starting every control at the median gave them nothing recognizable to work from. First step in preparing the app as a free tool for IntelliDec | `pages/2_Predict.py`: control values seeded in `st.session_state`, `load_country` and `reset_inputs` button callbacks, a `loaded_from` record, a changed-inputs table and a metric with the difference; thin-data check no longer counts a real value that equals the median; tested headlessly with `streamlit.testing.v1.AppTest` (first visit, all-median predict, fill, predict as recorded, what-if, reset, second country) |
| 14. Move to own server | App runs on Scott's VPS at `lifeexpectancy.intellidec.net`, built by Coolify from a `Dockerfile` | Scott wanted his last app off Streamlit's hosting; an own address also suits its new role as a free tool on IntelliDec | Added `Dockerfile` and `.dockerignore`; Coolify resource with build strategy Dockerfile, port 8501, domain with HTTPS; DNS A record for the subdomain. Problems met and fixed on the way: the Dockerfile had not been pushed (build failed with "no such file"), the port defaulted to 3000, and the browser warned "not secure" until the app was running and the certificate issued |

**Found while building milestone 13 (not yet fixed):**

- **Status labels.** Canada, Finland, France, Greece, Israel and the Republic of Korea are labelled "Developing" in the source data's Status column. Every chart that splits by status inherits this.
- **Health expenditure stored as zero.** `Pct_Expenditure` is exactly 0 in 606 of 2,928 cleaned rows, across 181 countries (for example Canada, Australia and Austria in 2015). These are almost certainly missing values stored as zero, which the cleaning rules do not treat as missing for this column. The feature carries about 0.5% of the model's built-in importance, so the effect on predictions is small, but a visitor who loads Canada now sees a health expenditure of 0.
- **Deprecated option.** `use_container_width`, used throughout the app, is deprecated in current Streamlit and prints a warning on every run.

**Still open:** test a shrinkage correction, source outside data on conflict and epidemics, a developed vs. developing fairness check, automated tests, and removing the full-dataset winsorizing leak.
