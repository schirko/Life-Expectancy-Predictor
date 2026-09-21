"""
Generic data loading and cleaning pipeline for the Life Expectancy Predictor.

This module holds NO dataset-specific knowledge -- no hardcoded column
names, winsorization limits, or predictor lists. All of that comes from a
DatasetConfig (see config_loader.py), loaded from a YAML file. This means
the same cleaning logic (normalize headers, fix implicit nulls, interpolate
per-entity, winsorize, drop columns) can be reused for a differently-shaped
WHO-style CSV just by writing a new config -- without touching this code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from config_loader import DatasetConfig

# Harmless scipy/numpy compatibility warning from winsorize() on masked
# arrays with zero actual missing values -- suppressed as in the notebook.
warnings.filterwarnings("ignore", message="Warning: 'partition' will ignore the 'mask'")


def _clean_column_names(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    new_cols = [c.strip().title().replace("  ", " ").replace(" ", "_") for c in df.columns]
    df = df.copy()
    df.columns = new_cols
    df.rename(columns=config.column_renames, inplace=True)
    return df


def _fix_implicit_nulls(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    df = df.copy()
    for col in config.zero_as_null:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    for col, bounds in config.range_as_null.items():
        if col in df.columns:
            lo, hi = bounds
            df[col] = df[col].apply(
                lambda v: np.nan if pd.notnull(v) and (v < lo or v > hi) else v
            )
    return df


def _interpolate_grouped(
    df: pd.DataFrame, config: DatasetConfig, feature_cols: list
) -> pd.DataFrame:
    """Interpolate within each entity's (e.g. country's) time series, then
    fall back to that entity's median, then the overall column median.
    If no group_column is configured, this step is skipped."""
    if not config.group_column or config.group_column not in df.columns:
        return df

    sort_cols = [c for c in config.sort_columns if c in df.columns] or [config.group_column]
    df = df.sort_values(sort_cols).copy()

    for col in feature_cols:
        df[col] = df.groupby(config.group_column)[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    for col in feature_cols:
        group_median = df.groupby(config.group_column)[col].transform("median")
        df[col] = df[col].fillna(group_median)
        df[col] = df[col].fillna(df[col].median())

    return df


def load_and_clean(config: DatasetConfig) -> pd.DataFrame:
    """Full cleaning pipeline: raw CSV -> cleaned, imputed dataframe
    (NOT yet winsorized)."""
    df = pd.read_csv(config.source_csv)
    df = _clean_column_names(df, config)
    df = df[df[config.target_column].notnull()].copy()

    df = _fix_implicit_nulls(df, config)

    non_feature_cols = set(config.id_columns) | {config.target_column}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    df = _interpolate_grouped(df, config, feature_cols)

    return df.reset_index(drop=True)


def winsorize_df(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply configured Winsorization limits to numeric feature columns."""
    df = df.copy()
    for col, bounds in config.winsorize_limits.items():
        if col in df.columns:
            lo, hi = bounds
            df[col] = winsorize(df[col].to_numpy(), limits=(lo, hi))
    return df


def get_model_ready_df(config: DatasetConfig) -> pd.DataFrame:
    """Full pipeline: clean -> winsorize -> drop configured columns."""
    df = load_and_clean(config)
    df = winsorize_df(df, config)
    to_drop = [c for c in config.drop_columns if c in df.columns]
    df = df.drop(columns=to_drop)
    return df


def feature_ranges(df: pd.DataFrame, config: DatasetConfig) -> dict:
    """Return {col: (min, max, median)} for building sliders/defaults in the app."""
    ranges = {}
    for col in config.predictor_columns:
        if col in df.columns:
            ranges[col] = (
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].median()),
            )
    return ranges
