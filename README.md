# Life Expectancy Predictor

[![tests](https://github.com/schirko/Life-Expectancy-Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/schirko/Life-Expectancy-Predictor/actions/workflows/tests.yml)

A data-science project that predicts a country's life expectancy from WHO
health and economic indicators (2000–2015) with a Random Forest, and tests
honestly how well it does on countries it has never seen.

- Random split: R² 0.96, average miss 1.15 years
- Countries held out entirely (GroupKFold by country): R² 0.90, average miss 2.14 years

The second number is the one to trust: a random split puts a country's other
years in training. The full write-up, with charts, is at
<https://schirko.com/scott/life-expectancy/>. Every conclusion is summarised in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Project structure

```
Life-Expectancy-Predictor/
├── config/
│   └── who_life_expectancy.yaml   # ALL dataset-specific knowledge lives here
├── config_loader.py            # Loads + validates a dataset config against its CSV
├── data_prep.py                # Config-driven cleaning; FoldWinsorizer (leak-free outlier clipping)
├── train_model.py              # Trains, evaluates and saves the model + diagnostics
├── tune_hyperparameters.py     # Optional: country-held-out grid search (~1-3 min)
├── analysis/
│   ├── leak_check.py           # Optional: measures the old outlier-clipping leak
│   ├── fairness_check.py       # Developed vs developing comparison (milestone 18)
│   ├── interval_calibration.py # Uncertainty band sized per group (milestone 19)
│   └── shrinkage_check.py      # Shrinkage-correction test (milestone 20)
├── site_build/                 # Generates the static project page (+ the deck's charts)
├── presentation/               # Exports of the story deck, published to the page
├── tests/                      # 37 automated tests (python -m pytest)
├── notebooks/                  # 01-07: the project explained step by step; plus the original
├── data/
│   ├── Life_Expectancy_WHO_Data.csv
│   └── clean_data.csv          # Written by train_model.py
├── model/                      # Written by the scripts above
└── docs/
    ├── FINDINGS.md             # Every conclusion in one place
    └── DOCUMENTATION.md        # Methodology, findings and build log
```

## Setup

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r site_build/requirements-site.txt   # only to build the project page
```

## Run

```
python train_model.py                 # train, evaluate, save model/ and data/clean_data.csv
python tune_hyperparameters.py        # optional
python -m analysis.leak_check         # optional
python -m analysis.fairness_check     # developed vs developing comparison
python -m analysis.interval_calibration   # size the uncertainty band per group
python -m analysis.shrinkage_check    # can recalibration undo the pull toward the middle?
python -m site_build.build_site       # build the project page
```

`train_model.py` validates the config against the CSV, cleans the data, trains
the Random Forest, scores it on a random split and on held-out countries, and
saves `model/rf_model.joblib`, `model/metadata.json` and
`model/diagnostics.json` (residuals, permutation importance, partial
dependence, interval calibration, unseen-country errors, hardest countries and
a four-model comparison).

## Tests

```
pip install -r requirements-dev.txt
python -m pytest
```

37 tests in `tests/` check the config, the cleaning (including the zero-means-missing rules), the leak-free outlier clipping, that no country is ever on both sides of the honest test, that the README, findings and documentation quote the same numbers as the saved results, and that the page build stays inside its own folder, that every notebook on the page has saved outputs and no errors, and that the presentation section appears only when exports exist. GitHub Actions runs them on every push (badge above).

## Notebooks

`notebooks/01_the_data.ipynb` to `07_what_did_not_help.ipynb` walk through the
project in order, for a reader who is smart but not necessarily an analyst: the
data and its hidden problems, cleaning and the leak, the model and the honest
test, what drives the predictions, where the model fails, fairness and the
uncertainty band, and what didn't help. Every step is "what we are looking for,
the code and its result, how to read it, what it means, what's next and why".
They run in PyCharm with the project's `venv`; run 03 before 05 and 06. The
committed copies include their outputs.

## Presentation

The whole project as a 16-slide story with speaker notes, published on the
project page as PDF, PowerPoint, a web page and slide images. The deck is built
from the notebooks; its charts come from `python -m site_build.presentation_charts`.
`presentation/README.md` explains the export step.

## Project page

`python -m site_build.build_site` redraws the charts from the saved results,
fills in the numbers, exports the eight notebooks and the docs, and
writes the page to `SITE_DIR` from `.env` (copy `.env.example`; on Scott's PC
this is the Schirko portfolio folder served at schirko.com/scott/life-expectancy/).
Without `SITE_DIR` it writes to a git-ignored `site/` folder. After building,
push this repo, then push the Schirko repo so Coolify redeploys.

## Swapping in a different dataset

The code holds no dataset-specific knowledge. To use a similarly shaped
WHO-style CSV (a country-year panel with a numeric target), copy
`config/who_life_expectancy.yaml`, edit `dataset.source_csv`, `target_column`,
`column_renames`, `winsorize_limits`, `drop_columns` and the `predictors` list,
then run `python train_model.py config/my_new_dataset.yaml`. A mismatch with
the CSV fails fast with a list of exactly what's missing.

## History

The project began as an M.S. practicum notebook and later became a Streamlit
app. On 2026-11-15 Streamlit was removed: the project is now presented only as
the static page above. See "Decision: Streamlit Removed" and the build log in
[`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).
