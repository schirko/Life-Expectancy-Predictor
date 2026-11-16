"""The cleaning pipeline turns placeholder values into gaps, fills every gap,
and never touches the target."""
import numpy as np

from data_prep import _clean_column_names, _fix_implicit_nulls


def test_zero_means_missing(config, raw_csv):
    """A 0 in these columns means "not reported" and must become a gap (milestone 21)."""
    df = _fix_implicit_nulls(_clean_column_names(raw_csv, config), config)
    assert "Pct_Expenditure" in config.zero_as_null
    for col in config.zero_as_null:
        assert (df[col] == 0).sum() == 0, f"{col} still has zeros"


def test_health_expenditure_2015_is_filled_not_zero(clean_df):
    """Every country reported 0% health spending in 2015; after cleaning none should."""
    assert (clean_df.loc[clean_df["Year"] == 2015, "Pct_Expenditure"] == 0).sum() == 0


def test_bmi_outside_plausible_range_becomes_a_gap(config, raw_csv):
    df = _fix_implicit_nulls(_clean_column_names(raw_csv, config), config)
    low, high = config.range_as_null["BMI"]
    bmi = df["BMI"].dropna()
    assert bmi.between(low, high).all()


def test_no_gaps_left_in_the_inputs(config, clean_df):
    assert not clean_df[config.predictor_columns].isna().any().any()
    assert len(clean_df) == 2928
    assert clean_df[config.group_column].nunique() == 183


def test_target_is_never_clipped(config, raw_csv, clean_df):
    """Life expectancy keeps its true recorded values (milestone 16)."""
    raw = _clean_column_names(raw_csv, config)[config.target_column].dropna()
    assert np.isclose(clean_df[config.target_column].min(), raw.min())
    assert np.isclose(clean_df[config.target_column].max(), raw.max())
