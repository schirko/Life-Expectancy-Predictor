# Life Expectancy Predictor

A Streamlit app built on top of the WHO Life Expectancy data-science project:
a **Dashboard** for exploring trends/correlations/rankings, and a **Predict**
page that uses a trained Random Forest model (test R² ≈ 0.96, 5-fold CV
R² ≈ 0.91) to estimate life expectancy from health and economic indicators,
with SHAP-based per-prediction explanations.

## Project structure

```
life_expectancy_app/
├── Home.py                     # Home page
├── pages/
│   ├── 1_📊_Dashboard.py       # Explore page
│   └── 2_🔮_Predict.py         # Prediction page (SHAP explanations)
├── config/
│   └── who_life_expectancy.yaml   # ALL dataset-specific knowledge lives here
├── config_loader.py            # Loads + validates a dataset config against its CSV
├── data_prep.py                # Generic, config-driven cleaning pipeline
├── train_model.py              # Trains & saves the Random Forest model
├── data/
│   └── Life_Expectancy_WHO_Data.csv
├── model/                      # Created by train_model.py
│   ├── rf_model.joblib
│   └── metadata.json
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model (do this once, or whenever the data/config changes)

```bash
python train_model.py
```

This validates the config against the CSV's actual columns, cleans the raw
CSV, trains the Random Forest, runs 5-fold cross-validation, and saves:
- `model/rf_model.joblib` — the trained model
- `model/metadata.json` — test metrics, CV scores, feature importances,
  feature ranges, and predictor UI metadata (labels/descriptions)
- `data/clean_data.csv` — the cleaned dataset used by the dashboard

## 2. Run the app

```bash
streamlit run Home.py
```

This opens the app in your browser (usually `http://localhost:8501`).

## Swapping in a different dataset

This app is **config-driven**: none of the Python code hardcodes column
names, winsorization limits, or the predictor list. All of that lives in
`config/who_life_expectancy.yaml`.

To point the app at a different, similarly-shaped WHO-style CSV (a country-
year panel with a numeric target and numeric predictors):

1. Copy `config/who_life_expectancy.yaml` to a new file, e.g.
   `config/my_new_dataset.yaml`
2. Edit its values to match the new CSV: `dataset.source_csv`,
   `target_column`, `column_renames`, `winsorize_limits`, `drop_columns`,
   and the `predictors` list (with labels/descriptions/UI type for each)
3. Run training against that config:
   ```bash
   python train_model.py config/my_new_dataset.yaml
   ```
4. If the config doesn't match the CSV's actual columns, this fails fast
   with a clear list of exactly what's missing — check `config_loader.py`'s
   `validate_schema()` for the checks it runs
5. Restart the Streamlit app — the Predict page's sliders/inputs, labels,
   and descriptions are all read from `model/metadata.json`, so they update
   automatically without touching any page code

**Known limitation:** the Dashboard page still assumes `Country`/`Year`/
`Status`-style identifying columns for its filters (this app was designed
around WHO-style country-year panel data specifically). A dataset with a
genuinely different shape would need its own dashboard filters, even though
the cleaning pipeline and Predict page would work unchanged.

## Notes

- The cleaning pipeline in `data_prep.py` mirrors the original notebook:
  header normalization, implicit-null handling, per-entity time-series
  interpolation, Winsorization, and dropping highly-collinear features —
  now entirely driven by the active config rather than hardcoded.
- The Predict page mixes sliders (bounded/percentage-style indicators) and
  direct numeric entry (less intuitive-range indicators), per each
  predictor's `input_type` in the config.
- The "prediction spread" shown on the Predict page comes from the variance
  across the Random Forest's individual trees — a rough, not statistically
  rigorous, uncertainty indicator. SHAP values (also on the Predict page)
  give a more principled per-prediction explanation.
- `train_model.py` reports both a single 80/20 train/test split AND 5-fold
  cross-validation — worth noting the CV score is typically a bit lower and
  more trustworthy than a single lucky/unlucky split.

## Ideas for next steps

- Hyperparameter tuning (GridSearchCV) instead of the current default
  `n_estimators=200`
- Data quality / missingness-before-and-after visualization
- A second example config for a genuinely different WHO dataset, to prove
  out the config-driven claim end-to-end
- Deploy via Streamlit Community Cloud for a shareable public link (done!)

