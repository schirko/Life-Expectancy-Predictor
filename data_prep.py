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
from sklearn.base import BaseEstimator, TransformerMixin

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
    """Apply configured Winsorization limits to numeric feature columns, using
    ALL rows. This is the old, leaky way, kept only so analysis/leak_check.py
    can reproduce the old scores; training and evaluation clip inside each
    training split instead (see FoldWinsorizer). The target is never clipped."""
    df = df.copy()
    for col, bounds in config.winsorize_limits.items():
        if col == config.target_column:
            continue
        if col in df.columns:
            lo, hi = bounds
            df[col] = winsorize(df[col].to_numpy(), limits=(lo, hi))
    return df


def get_unclipped_df(config: DatasetConfig) -> pd.DataFrame:
    """Clean -> drop configured columns, with NO outlier clipping.
    Used for training and evaluation: clipping is learned inside each training
    split by FoldWinsorizer, so test rows never influence the clip limits."""
    df = load_and_clean(config)
    to_drop = [c for c in config.drop_columns if c in df.columns]
    return df.drop(columns=to_drop)


class FoldWinsorizer(BaseEstimator, TransformerMixin):
    """Outlier clipping that learns its limits from training rows only.

    The old pipeline winsorized the whole dataset before splitting, so the
    clip limits (e.g. "the top 16% of HIV_AIDS values") were partly set by the
    test rows: a small data leak. This transformer is the fix. fit() finds the
    cut-off values on the training rows using the same per-column percentages
    as the config; transform() clips any rows (train or test) to those
    cut-offs. Put it first in a scikit-learn Pipeline and
    every cross-validation fold learns its own limits from its own training
    rows.

    limits: list of (lower_pct, upper_pct), one per column of X, in order.
    Matches scipy's winsorize(): with n rows, the lowest int(lower_pct * n)
    values are raised to the next value, and likewise at the top.
    """

    def __init__(self, limits=None):
        self.limits = limits

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        lows, highs = [], []
        for j, (lo, hi) in enumerate(self.limits):
            col = np.sort(X[:, j])
            lows.append(col[int(lo * n)] if lo > 0 else -np.inf)
            highs.append(col[n - 1 - int(hi * n)] if hi > 0 else np.inf)
        self.lower_ = np.array(lows)
        self.upper_ = np.array(highs)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def fold_winsorizer(config: DatasetConfig) -> FoldWinsorizer:
    """A FoldWinsorizer set up with the config's limits for each predictor."""
    return FoldWinsorizer(
        limits=[tuple(config.winsorize_limits.get(c, (0.0, 0.0))) for c in config.predictor_columns]
    )


def missingness_report(config: DatasetConfig) -> list:
    """
    Report missing-value rates per feature at three pipeline stages:
      - raw: straight from the CSV, before any cleaning
      - after_implicit_nulls: after converting implicit nulls (e.g. 0-as-missing)
        to real NaN, but before interpolation
      - final: after interpolation/fallback-median imputation

    Saved in model/diagnostics.json to show what the cleaning pipeline actually
    did to missingness, rather than asserting it without evidence.
    """
    raw_df = pd.read_csv(config.source_csv)
    raw_df = _clean_column_names(raw_df, config)
    raw_df = raw_df[raw_df[config.target_column].notnull()].copy()

    implicit_df = _fix_implicit_nulls(raw_df, config)

    non_feature_cols = set(config.id_columns) | {config.target_column}
    feature_cols = [c for c in implicit_df.columns if c not in non_feature_cols]

    final_df = _interpolate_grouped(implicit_df, config, feature_cols)

    n = len(raw_df)
    report = []
    for col in feature_cols:
        raw_pct = 100 * raw_df[col].isnull().sum() / n if col in raw_df.columns else None
        report.append(
            {
                "column": col,
                "raw_missing_pct": round(float(raw_pct), 2) if raw_pct is not None else None,
                "after_implicit_nulls_missing_pct": round(float(100 * implicit_df[col].isnull().sum() / n), 2),
                "final_missing_pct": round(float(100 * final_df[col].isnull().sum() / n), 2),
            }
        )
    return sorted(report, key=lambda r: r["after_implicit_nulls_missing_pct"], reverse=True)
