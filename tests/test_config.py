"""The config file and the CSV agree, and a mismatch is caught early."""
from config_loader import validate_schema


def test_config_matches_the_csv(config, raw_csv):
    assert validate_schema(config, list(raw_csv.columns)) == []


def test_missing_column_is_reported(config, raw_csv):
    target_header = next(c for c in raw_csv.columns if c.strip().lower() == "life expectancy")
    problems = validate_schema(config, [c for c in raw_csv.columns if c != target_header])
    assert any("Target column" in p for p in problems)


def test_thirteen_predictors(config):
    assert len(config.predictor_columns) == 13
    assert config.target_column not in config.predictor_columns
