"""Shared setup for the test suite.

Run from the project folder:   python -m pytest
The tests use the real WHO data in data/ and the saved results in model/.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_loader import load_config  # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "config", "who_life_expectancy.yaml")


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def config():
    return load_config(CONFIG_PATH, base_dir=ROOT)


@pytest.fixture(scope="session")
def raw_csv(config):
    return pd.read_csv(config.source_csv)


@pytest.fixture(scope="session")
def clean_df(config):
    from data_prep import get_unclipped_df
    return get_unclipped_df(config).dropna(subset=config.predictor_columns + [config.target_column]).reset_index(drop=True)
