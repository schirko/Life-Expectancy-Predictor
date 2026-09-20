"""
Data loading and cleaning for the Life Expectancy app.

This mirrors the cleaning pipeline from the original notebook:
- normalize column headers
- treat implicit nulls (0s that really mean "missing")
- interpolate missing values per-country (time series), with median fallbacks
- Winsorize outliers
- drop low-value / highly-collinear features

Kept as plain functions (not the notebook's global-variable style) so it can
be imported by both the training script and the Streamlit app.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

# Harmless scipy/numpy compatibility warning from winsorize() on masked
# arrays with zero actual missing values -- suppressed as in the notebook.
warnings.filterwarnings("ignore", message="Warning: 'partition' will ignore the 'mask'")

# Features removed in the notebook for being highly collinear with other
# predictors, or (BMI) for being ~50% null.
DROPPED_HIGH_CORR = ["Thinness_5-9_Years", "GDP", "Infant_Deaths", "Population"]
DROPPED_ALWAYS = ["BMI"]

# Winsorization limits per feature, taken from the notebook's tuned values.
# (lower_pct, upper_pct) trimmed from each tail.
WINSOR_LIMITS = {
    "Life_Expectancy": (0.01, 0.0),
    "Adult_Mortality": (0.0, 0.03),
    "Infant_Deaths": (0.0, 0.10),
    "Alcohol": (0.0, 0.01),
    "Pct_Expenditure": (0.0, 0.13),
    "Hepatitis_B": (0.11, 0.0),
    "Measles": (0.0, 0.19),
    "Under-Five_Deaths": (0.0, 0.11),
    "Polio": (0.10, 0.0),
    "Ttl_Expend": (0.0, 0.02),
    "Diphtheria": (0.10, 0.0),
    "HIV_AIDS": (0.0, 0.16),
    "GDP": (0.0, 0.13),
    "Population": (0.0, 0.15),
    "Thinness_10-19_Years": (0.0, 0.04),
    "Thinness_5-9_Years": (0.0, 0.04),
    "Income_Comp_Of_Resources": (0.05, 0.0),
    "Schooling": (0.02, 0.01),
}

# Final predictor set used by the Random Forest model (matches the notebook).
PREDICTOR_COLS = [
    "Adult_Mortality",
    "Alcohol",
    "Pct_Expenditure",
    "Hepatitis_B",
    "Measles",
    "Under-Five_Deaths",
    "Polio",
    "Ttl_Expend",
    "Diphtheria",
    "HIV_AIDS",
    "Thinness_10-19_Years",
    "Income_Comp_Of_Resources",
    "Schooling",
]

TARGET_COL = "Life_Expectancy"


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for col in df.columns:
        new_cols.append(col.strip().title().replace("  ", " ").replace(" ", "_"))
    df = df.copy()
    df.columns = new_cols
    df.rename(
        columns={
            "Thinness_1-19_Years": "Thinness_10-19_Years",
            "Percentage_Expenditure": "Pct_Expenditure",
            "Total_Expenditure": "Ttl_Expend",
            "Bmi": "BMI",
            "Gdp": "GDP",
            "Hiv/Aids": "HIV_AIDS",
            "Income_Composition_Of_Resources": "Income_Comp_Of_Resources",
        },
        inplace=True,
    )
    return df


def _fix_implicit_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Infant_Deaths"] = df["Infant_Deaths"].replace(0, np.nan)
    if "Under-Five_Deaths" in df.columns:
        df["Under-Five_Deaths"] = df["Under-Five_Deaths"].replace(0, np.nan)
    if "BMI" in df.columns:
        df["BMI"] = df.apply(
            lambda x: np.nan if (x.BMI < 10 or x.BMI > 50) else x.BMI, axis=1
        )
    return df


def _interpolate_per_country(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Interpolate within each country's time series, then fall back to
    that country's median, then the overall column median."""
    df = df.sort_values(["Country", "Year"]).copy()

    for col in feature_cols:
        df[col] = df.groupby("Country")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    for col in feature_cols:
        country_median = df.groupby("Country")[col].transform("median")
        df[col] = df[col].fillna(country_median)
        df[col] = df[col].fillna(df[col].median())

    return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Full cleaning pipeline: raw CSV -> cleaned, imputed dataframe
    (NOT yet winsorized)."""
    df = pd.read_csv(csv_path)
    df = _clean_column_names(df)
    df = df[df[TARGET_COL].notnull()].copy()

    if "BMI" in df.columns:
        df.drop(columns="BMI", inplace=True)

    df = _fix_implicit_nulls(df)

    feature_cols = [c for c in df.columns[3:] if c in df.columns]
    df = _interpolate_per_country(df, feature_cols)

    return df.reset_index(drop=True)


def winsorize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Winsorization limits to numeric feature columns."""
    df = df.copy()
    for col, (lo, hi) in WINSOR_LIMITS.items():
        if col in df.columns:
            df[col] = winsorize(df[col].to_numpy(), limits=(lo, hi))
    return df


def get_model_ready_df(csv_path: str) -> pd.DataFrame:
    """Full pipeline: clean -> winsorize -> drop high-corr features."""
    df = load_and_clean(csv_path)
    df = winsorize_df(df)
    to_drop = [c for c in DROPPED_HIGH_CORR if c in df.columns]
    df = df.drop(columns=to_drop)
    return df


def feature_ranges(df: pd.DataFrame) -> dict:
    """Return {col: (min, max, median)} for building sliders/defaults in the app."""
    ranges = {}
    for col in PREDICTOR_COLS:
        if col in df.columns:
            ranges[col] = (
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].median()),
            )
    return ranges
