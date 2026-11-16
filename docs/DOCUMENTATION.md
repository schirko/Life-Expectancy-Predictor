# Life Expectancy Predictor — Project Documentation

> For the conclusions only, see [FINDINGS.md](FINDINGS.md).

A data-science project that predicts a country's life expectancy from WHO health and economic indicators (2000–2015) with a Random Forest, and tests honestly how well it does on countries it has never seen. The results are published as a static page at [schirko.com/scott/life-expectancy](https://schirko.com/scott/life-expectancy/), generated from the saved results by `site_build/build_site.py`.

> **2026-11-15:** the project no longer uses Streamlit. The interactive app (Dashboard, Predict and Diagnostics pages, plus its Docker and dev-container setup) was removed in milestone 17; see "Decision: Streamlit Removed" and the build log. The analysis, model and diagnostics are unchanged.

## Overview

The project has three parts:

- **Training and evaluation** (`train_model.py`): cleans the data, trains the Random Forest, scores it on a random split and on countries held out entirely, and computes the diagnostics described below into `model/diagnostics.json`.
- **Optional analyses** (`tune_hyperparameters.py`, `analysis/leak_check.py`): a country-held-out grid search, and the outlier-clipping leak check from milestone 16.
- **The project page** (`site_build/`): redraws the charts from the saved results, fills in the numbers, exports the original notebook and this document, and writes the page into the portfolio folder.

The project started as a graduate data-science coursework notebook (cleaning, EDA, and several model types compared, in `notebooks/`).

## Architecture

The pipeline is config-driven: dataset-specific knowledge (column names, cleaning rules, predictor list and labels) lives in a YAML config file, not hardcoded in Python.

| File | Purpose |
| --- | --- |
| `config/who_life_expectancy.yaml` | All dataset-specific knowledge: renames, null rules, winsorize limits, predictors and their labels |
| `config_loader.py` | Loads a config and validates it against a CSV's actual columns |
| `data_prep.py` | Generic cleaning pipeline (reads whatever config it's given), including `FoldWinsorizer`, which learns outlier caps from training rows only |
| `train_model.py` | Trains the Random Forest, runs country-grouped cross-validation, computes diagnostics, saves model + metadata |
| `tune_hyperparameters.py` | Optional grid search with country-held-out folds |
| `analysis/leak_check.py` | Optional: measures how much the old all-rows outlier clipping changed the scores (milestone 16) |
| `analysis/fairness_check.py` | Developed vs. developing comparison on unseen countries (milestone 18) |
| `analysis/interval_calibration.py` | Sizes the 80% uncertainty band per group (milestone 19) |
| `analysis/shrinkage_check.py` | Tests whether a recalibration step can undo the pull toward the middle (milestone 20) |
| `site_build/` | Generates the static project page (`build_site.py`, `site_template.html`) |
| `notebooks/` | The original practicum notebook, exported to the page |
| `model/` | Trained model, metadata, diagnostics, tuning results and the leak check |

Because the Python code contains no dataset-specific logic, swapping in a similarly shaped WHO-style CSV only requires writing a new config file and re-running training — see "Swapping in a New Dataset".

## Setup and Installation

1. Create and activate a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```
2. Install dependencies (the second file is only needed to build the project page):

   ```
   pip install -r requirements.txt
   pip install -r site_build/requirements-site.txt
   ```
3. Train the model (validates the config, cleans the data, trains, cross-validates, computes diagnostics, saves outputs to `model/` and `data/clean_data.csv`):

   ```
   python train_model.py
   ```
4. Optional analyses:

   ```
   python tune_hyperparameters.py
   python -m analysis.leak_check
   python -m analysis.fairness_check
   python -m analysis.interval_calibration
   python -m analysis.shrinkage_check
   ```
5. Build the project page (written to `SITE_DIR` from `.env`, the Schirko portfolio folder; without it, to a git-ignored `site/` folder):

   ```
   python -m site_build.build_site
   ```


## Data and Methodology

Source: WHO/UN life expectancy dataset, 2,929 country-year rows across ~193 countries, 2000–2015.

The cleaning pipeline (`data_prep.py`, config-driven):

1. **Normalize headers** — strip, title-case, and apply configured renames (e.g. `Hiv/Aids` → `HIV_AIDS`)
2. **Fix implicit nulls** — values that are technically present but really mean "missing" (e.g. a literal `0` in Infant Deaths, or a BMI outside a plausible range) are converted to actual nulls
3. **Impute** — missing values are filled by interpolating each country's own time series first, falling back to that country's median, then the global median
4. **Winsorize** — extreme outliers are trimmed per feature (specific percentile limits per column, tuned in the original notebook). Since milestone 16 the limits are learned from the training rows only, inside each split, and life expectancy itself is never trimmed (see "Outlier Clipping Leak Check" below). `data/clean_data.csv` holds the cleaned data before any trimming
5. **Drop** highly collinear or unreliable columns (e.g. GDP, Population, BMI) before training

After cleaning: 2,928 model-ready rows (one row dropped for a null target).

## Model and Performance

The model is a **Random Forest Regressor** (200 trees), chosen after comparing approaches in the original notebook:

| Model | Performance |
| --- | --- |
| Simple linear regression (Schooling only) | R² = 0.55 |
| Multiple linear regression | R² = 0.85 |
| Polynomial regression | R² ≈ 0.57 (plateaus past degree 3) |
| Decision tree | R² = 0.91 |
| **Random Forest (chosen)** | **R² = 0.96** |
| Logistic regression (High/Low classification) | 88% accuracy |

**Two performance readings are reported, and they tell different stories:**

- **Single 80/20 train/test split:** R² = 0.960, MAE = 1.15 years, RMSE = 1.88 years
- **Country-grouped cross-validation** (GroupKFold, 5 folds; each fold holds out whole countries, so it measures accuracy on countries the model never saw): R² = 0.901 (± 0.020). The same model under ordinary random 5-fold CV scores 0.960 (± 0.005).

These are the figures after the outlier-clipping leak was removed (milestone 16) and the health-expenditure zeros were treated as missing (milestone 21). Before milestone 16 they were 0.962 / 1.13 / 1.84 and 0.902; the changes are explained under "Outlier Clipping Leak Check" and "Health-Expenditure Zeros Fixed".

The gap between these two numbers is expected and worth understanding: the data is a panel (each country appears in up to 16 years), so a random split puts a country's other years in training and partly rewards memorizing countries. Holding out whole countries removes that leakage. An earlier version of this document attributed the gap to split luck; that was wrong.

**Top predictors** by built-in feature importance: HIV/AIDS (58.4%), Income Composition of Resources (17.3%), and Adult Mortality (14.5%) dominate — all three plausibly reflect a country's overall health-system quality.

## Auditing the Model Comparison Table

The six models in the original notebook (table above) were re-run from the notebook's own code to check how each figure was produced, not just the final numbers. That audit turned up two findings worth documenting.

**What overfitting means, briefly:** a model overfits when it starts fitting noise specific to its training data rather than the underlying pattern — it keeps looking better on training data while getting *worse* on data it hasn't seen. The standard way to catch it is comparing training-set performance against held-out test-set performance: a small, stable gap is healthy; a growing gap as model complexity increases is the warning sign.

**Finding 1 — a corrected claim about polynomial regression.** An earlier draft of this documentation claimed the polynomial regression model "overfits past degree 3." Re-running the comparison (degrees 1 through 25) shows this was inaccurate: the train/test R² gap stays under 0.02 at every degree tested, never approaching the notebook's own 0.05 overfitting threshold. With only one predictor (Schooling), there simply isn't enough model complexity for meaningful overfitting to occur — performance plateaus around R² ≈ 0.57 at degree 3 and higher degrees add negligible benefit. Both the app and this doc have been corrected to reflect the plateau, not a false overfitting narrative.

**Finding 2 — a real but harmless methodology note in the logistic regression cell.** That cell normalizes its 12 predictors (min-max scaling) *before* splitting into train/test sets, meaning the scaling statistics technically include test-set data — a mild, common form of data leakage (best practice fits a scaler on training data only, then applies those same statistics to the test set). Recomputing with train-only statistics gives an identical 88.05% accuracy, so it made no practical difference here, but it's a pattern worth avoiding on principle, and worth being able to explain if asked.

## Model Diagnostics: Deeper Analysis

`train_model.py` goes beyond the headline R² figure with several kinds of analysis, each computed against held-out data (not asserted) and saved in `model/diagnostics.json`. This section explains what each one is, why it matters, and what it actually found.

### Residual Analysis

A single R² hides *where* a model is wrong. Residual analysis (residual = actual − predicted) checks for patterns: an actual-vs-predicted plot that fans away from the diagonal, or a residuals-vs-predicted plot with a funnel shape (heteroscedasticity) or curve, both signal a real weakness a summary statistic wouldn't reveal. **Finding:** mean residual is 0.02 years (essentially zero — no systematic over- or under-prediction) with a standard deviation of 1.88 years, consistent with the reported test RMSE.

### Permutation Importance vs. Built-In (Gini) Importance

The Random Forest's default `feature_importances_` credits a feature every time it's used for a split, which is known to be biased toward continuous/high-cardinality features. Permutation importance instead shuffles one feature at a time in the test set and measures the resulting drop in R² — model-agnostic, and immune to that bias. **Finding:** built-in importance shows HIV/AIDS (58%) dominating Adult Mortality (14%) roughly 4-to-1; permutation importance shows them nearly tied (26% vs. 25%). The permutation version is the one to trust when they disagree.

### Partial Dependence Plots

Importance (either kind) answers "how much does this feature matter," a single number. Partial dependence answers "what shape is the relationship" — holding every other feature fixed and sweeping one feature across its range to see how the average prediction responds. Computed here for the top 3 features by permutation importance. Caveat worth naming if asked: partial dependence assumes features vary independently, which can be misleading when features are correlated in reality, as several health indicators in this dataset are.

### Uncertainty Calibration

A prediction interval is calibrated if it means what it claims — an 80% interval should contain the true value ~80% of the time. The tree-spread range (10th–90th percentile across the forest's 200 trees) is a cheap, intuitive approximation, not a statistically rigorous interval like quantile regression forests or conformal prediction would provide. **Finding:** checked empirically against the test set, on a random test split the stated 80% interval covers about 87% of true values, which looks conservative. That figure is flattered by same-country overlap: on countries the model has never seen, coverage drops to 77.2% (mean width 6.68 years), slightly overconfident but within 3 points of the target.

### Unseen Countries (Group Cross-Validation)

The question: how well does the model predict a country it has never seen? Random splits cannot answer it, because each country has up to 16 yearly rows and the other years land in training. Group cross-validation (scikit-learn `GroupKFold`, grouped by Country) keeps every row of a country in the same fold.

**Finding:** R² falls from 0.960 (random 5-fold) to 0.901 (country-grouped), and the average miss grows from 1.18 to 2.14 years. The median country is off by 1.96 years, and 14.2% of countries are off by more than 3 years. `model/diagnostics.json` lists the hardest countries and compares developed and developing groups.

**Why the earlier 0.910 was misread:** the original 5-fold CV ran unshuffled on country-sorted data, so folds were accidentally country-like (only 4 of 183 countries straddled folds). That is why it sat near the grouped score, not because of split luck.

**Caveat (updated in milestone 16):** winsorizing limits used to be computed on the full dataset before splitting. That leak is now removed and measured (see "Outlier Clipping Leak Check"). The all-country fallback median, used only when a country has no data at all for an input, still includes held-out countries.

**Why are the hardest countries hard?** Three explanations were tested against the out-of-fold predictions. (1) *Concentration:* the worst 10% of countries (18) carry 22% of total error; removing them lifts unseen-country R² from 0.905 to 0.924. (2) *Missing data:* Somalia, off by 11.5 years (more than double any other country), has no data for three inputs (Ttl_Expend, Income_Comp_Of_Resources, Schooling), which were filled with the all-country median, so the model saw an average country. But across all countries the link between empty inputs and error is 0.00, so missing data explains Somalia, not the pattern. (3) *Shrinkage toward the middle:* in the lowest life-expectancy band (36-55 years) the model predicts 2.5 years too high on average; in the highest band (80-89) it predicts 0.7 years too low, and the Model Showdown below shows gradient boosting and even linear regression do the same, so this is mostly about the data: when inputs only partly determine the outcome, every model regresses toward the average (milestone 20 confirms this is regression to the mean and cannot be corrected after the fact). The input whose country average best tracks error is HIV_AIDS (rank correlation +0.44). Countries such as Republic of Korea, Cuba and Guatemala are under-predicted: they live longer than their indicators suggest.

### Model Showdown (Four Algorithms, Same Unseen Countries)

The question: are the hardest countries hard because of the Random Forest, or because of the data? Four model families were scored on the identical country-held-out folds (`GroupKFold`, 5 folds), so only the algorithm differs: Random Forest (deployed), Gradient Boosting, Hist Gradient Boosting, and Linear Regression (scaled).

**Finding:** the algorithm does not matter much. R² on unseen countries is 0.901 (Random Forest), 0.896 (Gradient Boosting), 0.901 (Hist Gradient Boosting) and 0.825 (Linear Regression). The three tree models are within the fold-to-fold spread of about 0.02, so they are effectively tied. Six countries (Côte d'Ivoire, Ethiopia, Republic of Korea, Sierra Leone, Somalia, Swaziland) are in the worst 18 for every model, and Somalia is missed by 9 to 12 years by all three tree models. All four models under-predict the highest and over-predict the lowest life expectancies; linear regression does it most (-3.8 years at the low end, +3.1 at the high end). (Before milestone 16 the scores were 0.902 / 0.895 / 0.904 / 0.831 and the shared list had seven countries, including Dominican Republic and Jamaica instead of Republic of Korea; the conclusion is unchanged.) Conclusion: the cause is mostly the data, and the Random Forest is a sound choice that also supports SHAP and the tree-spread interval.

### Hyperparameter Tuning (GridSearchCV)

Random Forests resist overfitting even with many/deep trees, because each tree trains on a random bootstrap sample and predictions are averaged — unlike a single decision tree, where depth needs careful capping. A 36-combination grid search (`n_estimators`, `max_depth`, `min_samples_leaf`, 5-fold CV) found the best CV-scoring configuration (300 trees, max_depth=15) actually scored marginally *lower* on the held-out test set (0.9616) than the original default (0.9617). **Reading this correctly:** this isn't a failed search — cross-validation fold scores and a single held-out test score both carry some noise, and when the true difference between configurations is smaller than that noise, grid search can "win" on CV without that gain reproducing on test data. The original `n_estimators=200` default was already reasonable.

**Re-run with country-held-out folds:** the search above used random folds, which leak a country's other years into training. Repeating it with `GroupKFold` (and a test set of 37 unseen countries) gave the same message more strongly. The best setting (`max_depth=15`, 200 trees) scored 0.886 in grouped CV, but all 36 combinations fell within 0.0016 R² of each other while fold-to-fold noise was 0.043. On unseen countries the tuned model scored 0.9044 against 0.9049 for the default. The winner changed from the random-fold search, and the two methods agree only moderately on ordering (Spearman 0.59). Conclusion: forest settings are not what limits accuracy on new countries.

**Re-run after milestone 16 (leak-free clipping):** the best setting is now `max_depth=20`, `min_samples_leaf=2`, 300 trees (grouped CV 0.883). All 36 combinations fall within 0.0017 R² of each other against fold noise of about 0.045, and on unseen countries the tuned model scores 0.904 against 0.903 for the default. Same conclusion: no setting is meaningfully better than the defaults.

### Outlier Clipping Leak Check (Milestone 16)

**What we are looking for.** Winsorizing caps extreme values (for example, the top 16% of HIV/AIDS figures are set to the value at the 84th percentile). The original pipeline did this once, on all 2,928 rows, before splitting. Two problems followed: (1) the cap values were partly set by the test rows, a data leak; (2) life expectancy itself, the target, was capped at the bottom 1%, so 28 rows had their recorded value raised (the lowest true value, 36.3 years, became 45.6) and the model was scored against altered answers. The question: how much did this flatter the scores?

**How it was fixed.** A new `FoldWinsorizer` in `data_prep.py` learns each cap from the training rows only and then clips any rows (training or test). It sits first in a scikit-learn `Pipeline`, so every cross-validation fold, the Model Showdown and the tuning search learn their own caps. It reproduces the old scipy `winsorize` exactly when fitted on all rows (checked: maximum difference 0.0). The target is no longer clipped.

**How it was measured.** `analysis/leak_check.py` scores four versions on the same splits:

| Version | R² random split | Avg miss, random (yrs) | R² unseen countries | Avg miss, unseen (yrs) |
| --- | --- | --- | --- | --- |
| A. Old: clipped on all rows, target clipped | 0.9615 | 1.132 | 0.9019 | 2.115 |
| B. Features clipped per training split, target clipped | 0.9616 | 1.128 | 0.9025 | 2.119 |
| C. Clipped on all rows, target not clipped | 0.9605 | 1.147 | 0.9004 | 2.137 |
| D. New: clipped per training split, target not clipped | 0.9603 | 1.147 | 0.9010 | 2.139 |

**How to read it.** A vs B isolates the feature leak; A vs C isolates the clipped target; D is what the project uses now.

**What it means.** The feature leak was negligible (unseen-country R² +0.0006). Scoring against the true target cost a little more (R² -0.0015, average miss +0.022 years), because the clipped rows are the lowest life expectancies, which the model already over-predicts. Overall the headline numbers stand: 0.962 → 0.960 on a random split, 0.902 → 0.901 on unseen countries. The earlier conclusions did not depend on the leak; now they are measured without it.

**What's next and why.** One small leak remains: a country with no data at all for an input gets the all-country median, which includes held-out countries (Somalia is the clearest case). It is a low priority. The evaluation is now clean enough for the developed-vs-developing fairness check.

### Fairness Check: Developed vs. Developing (Milestone 18)

**What we are looking for.** On countries the model has never seen, developing countries are missed by more years than developed ones. Is the model treating them worse, or is their life expectancy simply harder to predict (wider range, more extreme values, more gaps in the data)? Status is not a model input; it is only used here to group the results.

**How it was done.** `analysis/fairness_check.py` predicts every row with a model that never saw that country (the same `GroupKFold(5)` folds and leak-free clipping as `train_model.py`), then compares the two groups four ways: average miss, bias and R² within each group; the same comparison inside matching life-expectancy bands; the 80% tree-spread interval's coverage and width; and a bootstrap over countries (2,000 resamples of whole countries) for a range on the gap. Everything runs twice: with the source data's labels, and with six advanced economies the source labels "Developing" (Canada, Finland, France, Greece, Israel, Republic of Korea) moved to "Developed". Borderline cases such as Qatar and Estonia were left as the source has them.

| Labels | Group | Countries | Avg miss (yrs) | Bias (yrs) | 80% interval covers | Interval width (yrs) | Inputs filled in |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source | Developed | 32 | 1.70 | -0.21 | 82.0% | 5.50 | 10.8% |
| Source | Developing | 151 | 2.23 | -0.03 | 76.2% | 6.93 | 6.4% |
| Corrected | Developed | 38 | 1.81 | -0.05 | 80.8% | 5.55 | 10.5% |
| Corrected | Developing | 145 | 2.22 | -0.07 | 76.3% | 6.98 | 6.3% |

Inside the same life-expectancy band (corrected labels): 70-75 years, developed 1.21 vs developing 1.48 (only 9 developed countries); 75-80 years, 1.63 vs 2.26.

**How to read it.** "Avg miss" is the typical error in years. "Bias" is the average signed error (negative = the model predicts too high); near zero means neither group is pushed up or down overall. "Covers" is how often the 80% band holds the true value; 80% is the target. "Inputs filled in" is the share of input values that were missing in the source and had to be interpolated.

**What it means.**

- **The gap is real but modest.** With corrected labels, developing countries are missed by 0.41 years more (95% range 0.10 to 0.73); with the source labels the gap is 0.54 (0.24 to 0.85). Neither range includes zero.
- **It is not just because developing countries are harder cases.** Comparing countries with the same life expectancy, the gap remains (1.48 vs 1.21, and 2.26 vs 1.63). Re-weighting developing countries to the developed mix of life expectancies does not shrink it either (2.66 years).
- **It is not missing data.** Developed countries actually had more filled-in inputs (10.5% vs 6.3%).
- **It is not systematic bias.** Both groups' average signed error is under a tenth of a year, so the model does not push either group up or down.
- **The uncertainty band is too narrow for developing countries.** It holds the true value about 81% of the time for developed countries, close to the intended 80%, but 76.3% for developing ones, even though it is already wider.
- **The labels mattered.** Counting the six advanced economies as developed shrinks the gap by about a quarter (0.54 → 0.41), because several of them are harder to predict than a typical developed country (the Republic of Korea is the 6th-hardest of 183 countries).
- **R² within a group is misleading here.** Developing countries score higher (0.88 vs 0.59) only because their life expectancies spread far wider (standard deviation 8.7 vs 3.9 years); the average miss is the fair comparison.

**What's next and why.** The 13 inputs describe developed countries well and developing countries less well: within the same life expectancy, the developing countries vary in ways the inputs do not capture (conflict, epidemics, how reliable the reporting is). The fix is better inputs, not a different model. A wider interval for developing countries would make the uncertainty honest in the meantime; milestone 19 does this. The six-country relabelling is used only in this analysis; the source data is unchanged.

### Uncertainty Band Sized Per Group (Milestone 19)

**What we are looking for.** The fairness check found the 80% band holds the true value 80.8% of the time for developed countries but only 76.3% for developing ones. A band that says "80%" should mean it for both groups. Can it be re-sized so it does, without simply making it wider for everyone?

**How it was done.** `analysis/interval_calibration.py` takes the raw band (10th to 90th percentile of the forest's 200 trees, predicted for countries the model never saw) and stretches it around the prediction by a factor *k*. Three versions are compared: (A) the raw band; (B) one factor for everyone, chosen so overall coverage is 80%; (C) one factor per group, each chosen so that group's coverage is 80%. The factors are cross-fitted: for each of the 5 country folds, *k* is chosen on the other four folds and applied to the held-out one, so no country helps size its own band. Groups use the corrected Status labels from milestone 18.

| Version | Developed: covers / width | Developing: covers / width | All: covers / width |
| --- | --- | --- | --- |
| A. Raw band | 80.8% / 5.55 yrs | 76.3% / 6.98 yrs | 77.2% / 6.68 yrs |
| B. One factor for everyone | 82.7% / 6.16 yrs | 78.9% / 7.67 yrs | 79.7% / 7.36 yrs |
| C. One factor per group | 81.1% / 5.81 yrs | 79.9% / 7.83 yrs | 80.1% / 7.41 yrs |

Factors fitted on all countries: developed 1.00 (no change), developing 1.13, one factor for everyone 1.10.

**How to read it.** "Covers" is how often the band held the true value on countries it was not tuned on; 80% is the target. "Width" is the average distance from the bottom to the top of the band, in years. A narrower band at the same coverage is better.

**What it means.**

- **One factor for everyone is the wrong fix.** It brings the overall figure to 80% but over-covers developed countries (82.7%, a wider band than they need) and still under-covers developing ones (78.9%).
- **Per-group factors fix both.** Developed countries keep essentially their raw band (81.1%), and developing countries get a band about 12% wider (7.83 vs 6.98 years) that now holds the truth 79.9% of the time.
- **The cost is honest, not large.** The overall band grows from 6.68 to 7.41 years, and that extra width goes to the group that needed it.
- **It was tested fairly.** Cross-fitting means each country's coverage was measured with a factor chosen without it.

**What's next and why.** The factors are saved in `model/interval_calibration.json` for any future use of the interval. Widening the band is a patch, not a cure: it admits the model knows less about developing countries. The cure is inputs that capture what varies among them (conflict, epidemics, reporting quality).

### Shrinkage-Correction Test (Milestone 20)

**What we are looking for.** On countries it has never seen, the model over-predicts the lowest life expectancies (by about 2.5 years in the lowest tenth) and under-predicts the highest, and every model in the showdown does the same. Can a correction step applied after the model ("recalibration") undo this pull toward the middle?

**How it was done.** `analysis/shrinkage_check.py` measures the pull two ways. *By true value*: rows are grouped by their actual life expectancy. *By prediction* (calibration): rows are grouped by what the model predicted, and actual is regressed on predicted; a slope above 1 would mean the predictions are squeezed and stretching them would help. Two corrections were then tested against no correction: **linear** (actual ≈ a + b × prediction) and **isotonic** (a free-form increasing curve). Both were learned honestly with nested cross-validation: for each of the 5 country folds, the correction was fitted on predictions for the other training countries (an inner country split), then applied to held-out countries the model and the correction never saw.

| Version | Avg miss (yrs) | R² | Calibration slope | Lowest 10% of true values | Highest 10% of true values | Lowest 10% of predictions | Highest 10% of predictions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. No correction | 2.139 | 0.905 | 0.99 | -2.52 | +0.63 | -0.16 | -0.50 |
| B. Linear | 2.143 | 0.904 | 1.00 | -2.68 | +0.77 | -0.35 | -0.35 |
| C. Isotonic | 2.207 | 0.899 | 1.00 | -2.75 | +1.03 | -0.33 | -0.06 |

**How to read it.** The last four columns are average errors in years (actual minus predicted); negative means the model predicted too high. A calibration slope of 1 means that, on average, a prediction of 60 years really does correspond to 60 years.

**What it means.**

- **The predictions are not squeezed.** The calibration slope is 0.99, and grouped by prediction every tenth is within about half a year of zero. So when the model says 50, the truth averages about 50.
- **The pull is regression to the mean.** Grouped by true value, the lowest tenth looks over-predicted by 2.5 years. That happens with any imperfect model: the rows with the most extreme true values are disproportionately the ones the inputs under-explain, so their predictions sit closer to the middle. It is a property of the errors, not a bias in the model.
- **Correcting it makes things worse.** Linear recalibration barely changes anything (average miss 2.139 → 2.143); isotonic recalibration is worse (2.207), and both make the true-value pattern slightly larger, not smaller. Stretching predictions that are already calibrated adds error.
- **This revises how the pattern should be described.** Earlier sections (Hardest Countries, Model Showdown) called it "shrinkage toward the middle"; it is better described as regression to the mean, and it cannot be fixed after the fact. Only better inputs, which explain more of the extreme cases, would shrink it.

**What's next and why.** No change to the model. The remaining route to better predictions for the most extreme countries is the same one the fairness check pointed to: inputs that capture conflict, epidemics and reporting quality.

### Health-Expenditure Zeros Fixed (Milestone 21)

**What we are looking for.** `Pct_Expenditure` (health spending as a share of GDP) was exactly 0 in 606 of 2,928 rows: every country in 2015, and 26 countries in every year. No country spends nothing on health, so these are "not reported", not zero. Left as zeros, the model learned from 606 false values. Does fixing them change anything?

**How it was done.** One line in `config/who_life_expectancy.yaml`: `Pct_Expenditure` joined `Infant_Deaths` and `Under-Five_Deaths` in the `zero_as_null` list, so a 0 there is now treated as missing. The usual gap-filling then applies: each country's own other years first (so 2015 takes the country's trend), then that country's median, then the all-country median for the 26 countries with no real figure at all. Every script was re-run and every number in this document, the findings and the project page was updated.

| Measure (unseen countries unless noted) | Before | After |
| --- | --- | --- |
| R² | 0.9015 | 0.9010 |
| Average miss | 2.131 years | 2.139 years |
| R² on a random split | 0.9600 | 0.9603 |
| Fairness gap (developing minus developed) | 0.40 years | 0.41 years |
| Health expenditure's share of the model's importance | 0.5% | 0.6% |

**How to read it.** Each row is the same test before and after the fix. Differences in the third decimal are within the normal wobble between runs of this kind of test (fold-to-fold R² varies by about ±0.02).

**What it means.**

- **The fix is right but changes almost nothing.** Every measure moved by less than a hundredth. Health expenditure carries well under 1% of the model's weight, so correcting 606 of its values could not move the results much.
- **No conclusion changed.** Every finding in this document holds with the same wording; only the last digits moved.
- **It was still worth doing.** A model should not learn from values that are known to be wrong, and an interviewer who spots 606 zeros would rightly ask about them.

**What's next and why.** The 26 countries with no real health-expenditure figure now get the all-country median, the same small fallback leak noted under "Outlier Clipping Leak Check". It stays low priority.

### Data Quality: Implicit Missingness

`.isnull()` only catches explicit `NaN` values, missing real-world missingness encoded as a placeholder (e.g. a literal `0` standing in for "not reported"). Left untreated, a model trains on that placeholder as if it were a true, verified value. **Finding:** several columns show 0% missing by a naive `.isnull()` check but a substantial rate once implicit nulls are correctly identified — BMI (49.4%), Infant Deaths (28.6%), Under-Five Deaths (26.5%) and, since milestone 21, health expenditure (20.7%) — all of which drop to 0% only after the pipeline's interpolation/imputation step. The true data-quality picture was invisible until this step ran.

## Limitations to Keep in Mind

- **Correlation, not causation.** A strong predictor (e.g. Adult Mortality) reflects a statistical relationship, not a proven causal mechanism.
- **Historical data.** The dataset ends in 2015; predictions reflect relationships from that period and may not capture more recent shifts in global health.
- **Imputed values.** Some indicators were filled via interpolation where the original data had gaps, introducing some uncertainty into those specific country-years.
- **Model uncertainty.** The tree-spread range (10th–90th percentile across the trees) is a rough indicator, not a rigorous confidence interval.
- **Single-split vs. cross-validation.** The 0.960 single-split R² partly reflects same-country leakage; the 0.901 country-grouped R² is the more trustworthy figure for new countries. Typical error for unseen countries is about 2.1 years versus 1.2 for the random split.

## Swapping in a New Dataset

To point the project at a different, similarly-shaped WHO-style CSV:

1. Copy `config/who_life_expectancy.yaml` to a new file
2. Edit `dataset.source_csv`, `target_column`, `column_renames`, `winsorize_limits`, `drop_columns`, and the `predictors` list (column, label and description) to match the new CSV
3. Run `python train_model.py config/your_new_config.yaml`
4. If the config doesn't match the CSV's actual columns, training fails immediately with a specific list of what's missing, rather than a cryptic error partway through cleaning
5. Rebuild the project page with `python -m site_build.build_site`

## Publishing Rules (Ownership)

**Standing rule (2026-11-16).** This project writes only inside its own folder in the Schirko portfolio, `scott/life-expectancy/` (served at schirko.com/scott/life-expectancy/), plus its card screenshot in `scott/images/screenshots/`, which is added by hand. The portfolio project owns `scott/index.html` and `scott/styles.css`, and this project never writes them. If the Life Expectancy card on the portfolio needs changing, the new text is handed to the portfolio project, which makes the change.

- `site_build/build_site.py` writes only to `SITE_DIR` (`index.html`, `images/`, `notebooks/`, `docs/` inside `scott/life-expectancy/`). It has no step that writes `scott/index.html`. As a safeguard it now stops with an error if `SITE_DIR` points at the portfolio root.
- History: on 2026-11-15 the card in `scott/index.html` was edited directly from this project (adding the card, updating the repo link, replacing the Streamlit tag). Those were one-off manual edits, not build steps, and will not be repeated.
- When committing in the Schirko repo, stage only this project's folder: `git add scott/life-expectancy`.

## Decision: Streamlit Removed

**2026-11-15.** The project no longer uses Streamlit. Its public face is the static page at [schirko.com/scott/life-expectancy](https://schirko.com/scott/life-expectancy/), which explains the analysis in plain language with charts and links to the code and notebook. Keeping an interactive app meant running a server for it, and all of Scott's projects now live as static pages under schirko.com/scott. Removed in milestone 17: `Home.py`, `pages/` (Dashboard, Predict, Diagnostics), `Dockerfile`, `.dockerignore`, `.devcontainer/`, the `streamlit`, `plotly` and `shap` packages, and the UI-only settings (`input_type`, `show_in_ui`, slider ranges) in the config and model metadata.

**What was lost, and where it went.** The Predict page's SHAP explanations and "start from a real country" feature have no static equivalent and are gone (see build-log rows 3, 11 and 13 for what they did). The diagnostics are still computed by `train_model.py`; their explanations live in this document and the main ones are on the project page.

**Hosting history.** The app ran on Streamlit Community Cloud, then from 2026-11-11 on Scott's VPS (Coolify, built from a `Dockerfile`) at lifeexpectancy.intellidec.net. That address was retired on 2026-11-15. The code is at [github.com/schirko/Life-Expectancy-Predictor](https://github.com/schirko/Life-Expectancy-Predictor) (named LifeExpectancyApp until 2026-11-15).


## Build Log: What, Why and How

Each row is one milestone, in the order built. Add a row for every new milestone so the project stays interview-ready. Rows 1–3, 6, 11, 13 and 14 describe the Streamlit app removed in milestone 17; they are kept as history.

| Step | What | Why | How |
| --- | --- | --- | --- |
| 1. Dashboard and Predict pages | Streamlit multipage app: charts of trends, correlations and rankings, plus a form that predicts life expectancy | Gives an interviewer something to click, not just a notebook | `Home.py` plus `pages/`; Plotly charts; a saved Random Forest (`rf_model.joblib`) loaded by the Predict page |
| 2. Deployment | Public app on Streamlit Community Cloud, fed from GitHub | A live link is easier to share than a repo | Push to `master` on `schirko/Life-Expectancy-Predictor` (named `LifeExpectancyApp` until 2026-11-15); the host redeploys on each push; classic personal access token (`repo` scope) for git auth |
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
| 15. Portfolio page | Added a generated static project page at schirko.com/scott/life-expectancy/ and a card on the portfolio | An employer-facing front door that explains the results in plain language and links to the code and notebook, without a public github.io site | `site_build/build_site.py` + `site_template.html` (matplotlib charts, nbconvert notebook export, mistune docs); writes to `SITE_DIR` in `.env` (the Schirko `scott/` folder, served by Coolify); noindex on every page; card added to Schirko `scott/index.html` |
| 16. Outlier-clipping leak removed | Winsorizing limits are learned from the training rows inside every split, and life expectancy is no longer clipped; the effect was measured four ways | Clipping on all rows before splitting let test rows set the caps, and clipping the target scored the model against altered answers; both had to be ruled out before trusting the scores | `FoldWinsorizer` in `data_prep.py`, used as the first step of a `Pipeline` in `train_model.py` and `tune_hyperparameters.py`; `analysis/leak_check.py` saves `model/leak_check.json`, shown on Diagnostics → Data Quality. Result: unseen-country R² 0.902 → 0.901, random split 0.962 → 0.960; findings unchanged |
| 17. Streamlit removed | Deleted the Streamlit app (`Home.py`, `pages/`), its Docker and dev-container setup, the `streamlit`, `plotly` and `shap` packages, and the UI-only config settings | The project is presented as a static page under schirko.com/scott; an app nobody can reach is code to maintain for nothing | `git rm` of the app files; `requirements.txt` trimmed; `input_type`/`show_in_ui` removed from the config, `config_loader.py` and the model metadata; `feature_ranges` and the all-rows clipped copy removed from `train_model.py`, so `data/clean_data.csv` now holds the unclipped cleaned data; docs rewritten around the project page |
| 18. Fairness check | Compared developed and developing countries on unseen-country predictions: average miss, bias, like-for-like by life-expectancy band, interval coverage, and a country bootstrap; with source and corrected Status labels | A model that is worse for one group of countries should say so; a portfolio project should test it, not assume it | `analysis/fairness_check.py` (GroupKFold(5), leak-free clipping) saves `model/fairness_check.json`; six advanced economies labelled "Developing" relabelled for the comparison only. Result: developing countries missed by 0.40 years more (95% range 0.09-0.72), gap holds within matching life-expectancy bands, no group bias, interval covers 76.5% vs 80.1% |
| 19. Uncertainty band sized per group | Stretched the 80% tree-spread band by a separate factor for developed and developing countries, chosen by cross-fitting on held-out countries | The fairness check showed the band under-covered developing countries (76.5%); an "80%" band should mean 80% for every group | `analysis/interval_calibration.py` saves `model/interval_calibration.json`. Result: developed 80.4%, developing 79.5% (band about 10% wider), versus 82.1% / 78.9% with one shared factor |
| 20. Shrinkage-correction test | Measured the pull toward the middle by true value and by prediction, and tested linear and isotonic recalibration with nested country cross-validation | Before "fixing" a bias, check it is one; an interviewer will ask whether the over-prediction of low life expectancies can be corrected | `analysis/shrinkage_check.py` saves `model/shrinkage_check.json`. Result: calibration slope 0.99, errors by prediction within about 0.5 years, so the pattern is regression to the mean; corrections raised the average miss (2.131 → 2.134 linear, 2.208 isotonic); no model change |
| 21. Health-expenditure zeros fixed | `Pct_Expenditure` added to the zero-means-missing rule; every script re-run and every number in the docs, findings and project page updated | 606 rows (every country in 2015, 26 countries in every year) reported 0% health spending, which is "not reported", not zero; the model should not learn from values known to be wrong | One line in `config/who_life_expectancy.yaml` (`zero_as_null`). Result: unseen-country R² 0.9015 → 0.9010, average miss 2.131 → 2.139 years, fairness gap 0.40 → 0.41; no conclusion changed |

**Found while building milestone 13:**

- **Status labels.** Canada, Finland, France, Greece, Israel and the Republic of Korea are labelled "Developing" in the source data's Status column. The fairness check (milestone 18) relabels them for its comparison; the source data is unchanged.
- **Health expenditure stored as zero.** Fixed in milestone 21: `Pct_Expenditure` was exactly 0 in 606 of 2,928 rows (every country in 2015, and 26 countries in every year). A 0 in that column is now treated as missing; see "Health-Expenditure Zeros Fixed".

**Still open:** source outside data on conflict and epidemics, automated tests, and the small fallback-median leak (countries with no data at all for an input).
