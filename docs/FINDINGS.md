# Life Expectancy Predictor — Findings

Every conclusion from the project in one place, most important first. If you are new to the project or to data science, read "Start here" and the word list first. Each finding gives what we were looking for, the key numbers, what it means, and what's next. The method behind each one is in the [project documentation](DOCUMENTATION.md) (section named in brackets), and the numbers come from the results files listed with it. To see the reasoning unfold with the real results on screen, read the seven numbered notebooks in `notebooks/` (on the project page under "The analysis, step by step"): findings 10 and 8 are notebooks 1 and 2, finding 1 is notebook 3, finding 2 is notebook 4, findings 4 and 5 are notebook 5, findings 6 and 7 are notebook 6, finding 3 is notebook 7, and finding 11 is notebook 8.

## Start here: the project in plain English

**The question.** Can you estimate how long people in a country live, on average, from a handful of health and economic statistics, such as how many adults die young, how common HIV/AIDS is, and how much schooling people get? The data is the World Health Organization's figures for 183 countries from 2000 to 2015.

**What was built.** A machine-learning model (a "Random Forest", explained below) that learns the pattern from countries it has seen and makes an estimate for any country.

**The honest answer.** For a country the model has never seen, its estimate is typically within about **2 years** of the real figure. That is good, but not as good as the first test suggested (about 1 year off). That first test let the model peek at other years of the same country, which is a bit like grading a student on questions they've already seen. Most of this project is about checking the model honestly and finding out where and why it goes wrong.

**What we learned, in one paragraph.** Three statistics do most of the work: deaths from HIV/AIDS, adult deaths, and an index of income and education. Swapping in a fancier model or fine-tuning it does not help. Neither does adjusting its answers afterwards. The model is limited by what the data contains, not by the method. It struggles most with a small number of countries (Somalia, Sierra Leone, Angola) and is somewhat less accurate for developing countries than developed ones, even when comparing countries with similar life spans. Its stated uncertainty was too confident for developing countries, so that was corrected. A flaw in how the data was cleaned, and 606 health-spending figures wrongly recorded as zero, were found and fixed; neither made a real difference.

**What would make it better.** The project's own guess was outside data on conflict, epidemics and reporting quality. That was tested last (finding 11), and it did not help. The honest conclusion is that the 13 inputs carry what can be carried about a country at this level of detail; what is left is country-specific. Sub-national data is the one idea left standing.

## Words used in this document

| Term | What it means here |
| --- | --- |
| **Model** | A formula learned from data that turns a country's statistics into an estimate of its life expectancy. |
| **Random Forest** | The kind of model used: 200 simple decision trees, each making its own estimate, averaged together. |
| **Unseen country** | A country left out of the model's training entirely and used only to test it. This is the fair test. |
| **Average miss** | How far off the estimate is, on average, in years. Smaller is better. |
| **R²** | A score from 0 to 1 for how much of the difference between countries the model explains. 1 is perfect; 0.90 means it explains about 90%. |
| **Bias** | Whether the model tends to guess too high or too low for a group (as opposed to just being off in both directions). |
| **Uncertainty band** | A range the model gives around its estimate, meant to contain the real figure 80% of the time. |
| **Coverage** | How often the band really does contain the real figure. 80% is the target. |
| **Regression to the mean** | When something is extreme, an imperfect estimate of it usually lands closer to average. Not a mistake in the model; it happens with any estimate that isn't perfect. |
| **Data leak** | When information from the test data sneaks into training, making results look better than they are. |
| **95% range** | The range the true answer very likely falls in, given how much the result varies when the data is resampled. |

## The short version, in numbers

A Random Forest predicts a country's life expectancy from 13 WHO health and economic indicators to within about **2.1 years on countries it has never seen** (R² 0.90). The often-quoted 0.96 is inflated by letting the model see other years of the same country. Three inputs (HIV/AIDS deaths, adult mortality and the income-composition index) do most of the work. Changing or tuning the algorithm doesn't help, and neither does correcting predictions after the fact: the limit is the inputs. Errors are concentrated in a few countries and are somewhat larger for developing countries, even like for like. The project's uncertainty band was resized so it is honest for both groups. A data leak in the cleaning step and 606 false zeros in the data were found, fixed and measured; neither changed the conclusions.

| # | Finding | Key number |
| --- | --- | --- |
| 1 | Honest accuracy is lower than the headline | R² 0.90 on unseen countries vs 0.96 on a random split |
| 2 | Three inputs drive the predictions | HIV/AIDS, adult mortality, income composition |
| 3 | The algorithm and its settings are not the bottleneck | Tree models tied at about 0.90; tuning +0.001 |
| 4 | A few countries carry much of the error | Worst 10% of countries = 22% of error |
| 5 | The "pull toward the middle" is regression to the mean | Calibration slope 0.99; corrections made it worse |
| 6 | Developing countries are missed by more, even like for like | +0.41 years (95% range 0.10 to 0.73) |
| 7 | The uncertainty band needed resizing per group | Developing band about 12% wider → 79.9% coverage |
| 8 | A cleaning-step leak was real but tiny | Unseen-country R² 0.902 → 0.901 after fixing |
| 9 | The original notebook had one wrong claim | Polynomial regression does not overfit past degree 3 |
| 10 | The source data had hidden gaps and wrong labels | 606 false zeros fixed; six advanced economies labelled "Developing" |

## 1. Honest accuracy is lower than the headline

**What we were looking for.** How accurate is the model for a country it has never seen, which is the realistic use?

**Key numbers.**

| Test | R² | Average miss |
| --- | --- | --- |
| Random 80/20 split | 0.960 | 1.15 years |
| Random 5-fold cross-validation | 0.960 | 1.18 years |
| Countries held out entirely (5 folds) | 0.901 (± 0.020) | 2.14 years |

The median country is off by 1.96 years; 14.2% of countries are off by more than 3 years.

**What it means.** A random split puts other years of the same country in training, so the model partly recognises the country rather than predicting it. Holding out whole countries is the fair test, and the model is about twice as far off. **0.90 and 2.1 years are the numbers to quote.**

**What's next.** All later findings use the held-out-country setup.

*[Unseen Countries; `model/metadata.json`, `model/diagnostics.json`]*

## 2. Three inputs drive the predictions

**What we were looking for.** Which inputs does the model actually rely on?

**Key numbers.** Scrambling one input at a time and seeing how much worse the model gets (called permutation importance; bigger = more important): HIV/AIDS deaths 0.26, adult mortality 0.25, income-composition index 0.20. The next input, thinness ages 10-19, is 0.02.

**What it means.** Three inputs carry the model; the other ten add little. The Random Forest's built-in importance agrees on the top three. These are associations, not causes.

**What's next.** A simpler model on these three inputs would be a useful comparison for explaining the project.

*[Permutation Importance, Partial Dependence; `model/diagnostics.json`]*

## 3. The algorithm and its settings are not the bottleneck

**What we were looking for.** Would a different algorithm, or better settings, predict unseen countries better?

**Key numbers.** On identical held-out-country folds: Random Forest 0.901, Hist Gradient Boosting 0.901, Gradient Boosting 0.896, Linear Regression 0.825. A 36-setting grid search, scored on held-out countries: tuned 0.904 vs default 0.903, with all settings within 0.002 of each other against fold-to-fold noise of about 0.045.

**What it means.** The three tree models are tied, and tuning changes nothing measurable. Linear regression is clearly worse because the relationships are not straight lines. The Random Forest with default settings stays.

**What's next.** Effort is better spent on the inputs than on the model.

*[Model Showdown, Hyperparameter Tuning; `model/diagnostics.json`, `model/tuning_results.json`]*

## 4. A few countries carry much of the error

**What we were looking for.** Is the error spread evenly, and why are the worst countries hard?

**Key numbers.** The worst 18 countries (10%) carry 22% of all error; without them, R² rises from 0.905 to 0.924. Somalia is missed by 11.5 years, Sierra Leone by 6.2, Angola by 5.5. Six countries are among the worst 18 for every algorithm (Côte d'Ivoire, Ethiopia, Republic of Korea, Sierra Leone, Somalia, Swaziland).

**What it means.** Somalia has no data at all for three inputs, so the model saw an average country; that explains Somalia, but across countries missing data is unrelated to error (rank correlation 0.00). The country average of HIV/AIDS deaths tracks error best (+0.44). Some countries, such as the Republic of Korea, Cuba and Guatemala, live longer than their indicators suggest.

**What's next.** The obvious guess was that these countries vary in ways the 13 inputs do not capture (conflict, epidemics, reporting quality). Finding 11 tested that guess.

*[Hardest Countries; `model/diagnostics.json`]*

## 5. The "pull toward the middle" is regression to the mean

**What we were looking for.** The lowest life expectancies are over-predicted by about 2.5 years and the highest under-predicted. Is that a bias a correction step could remove?

**Key numbers.**

| Version | Average miss | Calibration slope | Lowest 10% of true values | Lowest 10% of predictions |
| --- | --- | --- | --- | --- |
| No correction | 2.139 years | 0.99 | -2.52 | -0.16 |
| Straight-line correction | 2.143 years | 1.00 | -2.68 | -0.35 |
| Curve correction | 2.207 years | 1.00 | -2.75 | -0.33 |

**What it means.** Grouped by what the model predicted, the errors are near zero, and the "calibration slope" (how well predictions line up with reality; 1 is perfect) is 0.99, so the predictions are not squeezed toward the middle. The pattern only appears when rows are grouped by their true value: the most extreme countries are the ones the inputs explain least. Any imperfect model shows this, and correcting it after the fact made predictions worse.

**What's next.** No model change; only better inputs can reduce it.

*[Shrinkage-Correction Test; `model/shrinkage_check.json`]*

## 6. Developing countries are missed by more, even like for like

**What we were looking for.** Is the model worse for developing countries, or are they simply harder cases? (Status is not a model input.)

**Key numbers.** With six mislabelled advanced economies corrected:

| Group | Countries | Average miss | Bias | 80% band covers |
| --- | --- | --- | --- | --- |
| Developed | 38 | 1.81 years | -0.05 | 80.8% |
| Developing | 145 | 2.22 years | -0.07 | 76.3% |

Gap: 0.41 years (95% range 0.10 to 0.73, so very likely real but small). Comparing countries with the same life expectancy: 1.21 vs 1.48 years (70-75 band) and 1.63 vs 2.26 (75-80 band).

**What it means.** The gap is real but modest. It is not because developing countries have more extreme life expectancies (it holds like for like), not missing data (developed countries had more filled-in values), and not a systematic bias (neither group is pushed up or down). The inputs describe developing countries less completely.

**What's next.** Better inputs; in the meantime, an uncertainty band that is honest for both groups (finding 7).

*[Fairness Check; `model/fairness_check.json`]*

## 7. The uncertainty band needed resizing per group

**What we were looking for.** The model's "80%" band (the middle 80% of its 200 trees' predictions) should contain the truth 80% of the time. Does it, for countries the model never saw?

**Key numbers.** On a random split it covers 87.4%, which looks safe but is flattered by same-country overlap. On unseen countries it covers 77.2%: 80.8% for developed and 76.3% for developing countries. Tested on countries not used to size it:

| Version | Developed | Developing |
| --- | --- | --- |
| Raw band | 80.8% / 5.6 years wide | 76.3% / 7.0 years wide |
| One stretch factor for everyone | 82.7% / 6.2 | 78.9% / 7.7 |
| One stretch factor per group | 81.1% / 5.8 | 79.9% / 7.8 |

**What it means.** One shared factor over-covers developed countries and still under-covers developing ones. A separate factor (developing about 12% wider, developed unchanged) brings both to about 80%.

**What's next.** The factors are saved for any future use of the band. Widening is honest, not a cure.

*[Uncertainty Calibration, Uncertainty Band Sized Per Group; `model/interval_calibration.json`]*

## 8. A cleaning-step leak was real but tiny

**What we were looking for.** The original pipeline capped outliers using all rows before splitting (so test rows helped set the caps), and also capped life expectancy itself (28 low values raised; 36.3 years became 45.6). How much did that flatter the scores?

**Key numbers.** Unseen-country R² 0.902 before the fix, 0.901 after; random split 0.962 → 0.960; average miss on unseen countries 2.12 → 2.14 years. Most of the change came from no longer clipping life expectancy, not from the caps.

**What it means.** The leak is fixed (caps are now learned from training rows only, inside every split) and every number in this document is leak-free. The earlier conclusions did not depend on it.

**What's next.** One small leak remains: a country with no data at all for an input gets the all-country median. Low priority.

*[Outlier Clipping Leak Check; `model/leak_check.json`]*

## 9. The original notebook had one wrong claim

**What we were looking for.** Do the six models in the original practicum notebook hold up when re-run?

**Key numbers.** The claim that polynomial regression "overfits past degree 3" was wrong: re-running degrees 1 to 25, the gap between training and test R² stays under 0.02 at every degree, and performance plateaus around R² 0.57. The logistic regression cell scaled its data before splitting (a mild leak); fixing it gives the same 88.05% accuracy.

**What it means.** With a single input (schooling), there isn't enough complexity to overfit; the curve simply stops improving. The documentation was corrected.

*[Auditing the Model Comparison Table; `notebooks/`]*

## 10. The source data had hidden gaps and wrong labels

**What we were looking for.** Is the WHO data as complete and accurate as it looks?

**Key numbers.** Several columns show 0% missing by `.isnull()` but use 0 to mean "not reported". The cleaning rules treat a 0 in infant and under-five deaths as missing. Health-expenditure percentage was also exactly 0 in 606 of 2,928 rows: every country in 2015, and 26 countries in every year. Six advanced economies (Canada, Finland, France, Greece, Israel, Republic of Korea) are labelled "Developing".

**What it means.** No country spends nothing on health, so those zeros meant "not reported". They are now treated as missing (milestone 21) and filled from each country's other years. Because health spending carries under 1% of the model's weight, the fix moved every result by less than a hundredth and changed no conclusion; it was still the right thing to do. The fairness check relabels the six countries for its comparison only. Anyone using this dataset should check for both problems.

**What's next.** Nothing for the zeros. The 26 countries with no real health-spending figure get the all-country median, the same small fallback noted in finding 8.

*[Data Quality: Implicit Missingness; `model/diagnostics.json`]*

## 11. Outside data on conflict and epidemics did not help, and the project's recommendation was wrong

**What we were looking for.** Findings 3, 4 and 5 all ended the same way: the method is not the limit, the inputs are, so bring in outside data on conflict, epidemics and reporting quality. This tests that recommendation on the same honest, country-held-out folds.

**Key numbers.** Base model: average miss 2.139 years on unseen countries. Adding UCDP battle deaths: 2.138 (change −0.001, 95% range −0.010 to +0.010). Adding EM-DAT epidemic events and deaths: 2.135 (−0.004, range −0.014 to +0.006). Adding two reporting-quality signals (share of inputs filled in, year-to-year volatility): 2.195 (+0.056, range +0.004 to +0.114, so reliably worse), with the fairness gap widening from 0.41 to 0.45. All together: 2.203. The 18 hardest countries: 4.76 → 4.75 with conflict data. Somalia: 11.45 → 11.44.

**What it means.** Conflict and epidemic data add nothing because the WHO death statistics already reflect a crisis: adult mortality is far higher in heavy-conflict years, and the model reads it. There *is* a small leftover: in the 121 heaviest conflict years (4% of rows) the model is about 1.4 years too optimistic, but that is too thin a signal for the forest to learn, and the conflict input barely moves it. The reporting signals are weakly related to error (correlations 0.23 and 0.16), so the idea was not wrong, but as per-country constants they act as country fingerprints: a tree can split on them to identify a training country, which helps in training and hurts on unseen ones. The honest test caught that; a random-row test would have rewarded it.

The larger lesson is that the project's own prediction about itself did not survive a fair test, and it is recorded rather than quietly dropped. Three claims have now been tested and rejected: a better algorithm, after-the-fact correction, and outside data of this kind.

**What's next.** Not another national-level data source. Sub-national data (regions within countries) would give the model many more examples of the extreme cases it sees too few of, and is the one idea left standing. Otherwise the limit is reached, and the deliverable is an honest R² 0.90 with an uncertainty band that keeps its promise.

*[Outside Data (Milestone 25); `model/outside_data_check.json`; notebook 8]*

## What would improve the model

Every line of evidence points the same way: the algorithm, its settings, post-hoc corrections, and now outside data on conflict, epidemics and reporting quality are not the limit. The 13 indicators carry what can be carried about a country at this level of detail; the remaining error is country-specific. The one untested idea is a different unit of analysis: sub-national data, which would give the model many more examples of the extreme cases. Short of that, the limit is reached, and the right deliverable is what the project now has: an honest R² 0.90 on unseen countries with an uncertainty band sized per group.
