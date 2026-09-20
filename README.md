# Life Expectancy Explorer

A Streamlit app built on top of the WHO Life Expectancy data-science project:
a **Dashboard** for exploring trends/correlations/rankings, and a **Predict**
page that uses a trained Random Forest model (test R² ≈ 0.96) to estimate
life expectancy from health and economic indicators.

## Project structure

```
life_expectancy_app/
├── app.py                      # Home page
├── pages/
│   ├── 1_📊_Dashboard.py       # Explore page
│   └── 2_🔮_Predict.py         # Prediction page
├── data_prep.py                # Shared cleaning/winsorizing pipeline
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

## 1. Train the model (do this once, or whenever the data changes)

```bash
python train_model.py
```

This cleans the raw CSV, trains the Random Forest, and saves:
- `model/rf_model.joblib` — the trained model
- `model/metadata.json` — test metrics, feature importances, feature ranges
- `data/clean_data.csv` — the cleaned dataset used by the dashboard

## 2. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).

## Notes

- The cleaning pipeline in `data_prep.py` mirrors the notebook: header
  normalization, implicit-null handling, per-country time-series
  interpolation, Winsorization, and dropping highly-collinear features.
- The predictor uses a mix of sliders (bounded/percentage-style indicators)
  and direct numeric entry (less intuitive-range indicators), per your
  preference.
- The "prediction spread" shown on the Predict page comes from the variance
  across the Random Forest's individual trees — a rough, not statistically
  rigorous, uncertainty indicator.

## Ideas for next steps

- Add a per-prediction explanation (e.g. SHAP values) instead of only global
  feature importances
- Add a country-comparison view (select two countries, compare trajectories)
- Deploy via Streamlit Community Cloud for a shareable public link
- Add caching/versioning if you retrain the model with new data later
