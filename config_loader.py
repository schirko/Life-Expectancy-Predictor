"""
Config loader and schema validator for the Life Expectancy Predictor.

Loads a YAML config describing one dataset's shape (column renames, null
handling, winsorization limits, predictors and their labels) so the
cleaning pipeline and training script can work generically against any
similarly-shaped tabular dataset -- not just the one CSV this project
shipped with.

To point the project at a different WHO-style CSV: copy an existing config
file under config/, edit its values to match the new CSV's columns, and
pass its path to train_model.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


class ConfigError(Exception):
    """Raised when a config file is malformed, or doesn't match its CSV."""


@dataclass
class PredictorSpec:
    column: str
    label: str
    description: str = ""


@dataclass
class DatasetConfig:
    name: str
    description: str
    source_csv: str
    id_columns: list = field(default_factory=list)
    target_column: str = ""
    column_renames: dict = field(default_factory=dict)
    zero_as_null: list = field(default_factory=list)
    range_as_null: dict = field(default_factory=dict)
    group_column: str = ""
    sort_columns: list = field(default_factory=list)
    winsorize_limits: dict = field(default_factory=dict)
    drop_columns: list = field(default_factory=list)
    predictors: list = field(default_factory=list)
    model_params: dict = field(default_factory=dict)
    config_path: str = ""

    @property
    def predictor_columns(self) -> list:
        return [p.column for p in self.predictors]


def load_config(path: str, base_dir: str | None = None) -> DatasetConfig:
    """Load and parse a dataset config YAML file.

    `base_dir`, if given, is used to resolve a relative `source_csv` path
    (so the config can say "data/foo.csv" regardless of the current
    working directory the script happens to be run from).
    """
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ConfigError(f"Config file {path} is empty or invalid YAML.")

    try:
        dataset = raw["dataset"]
        source_csv = dataset["source_csv"]
        if base_dir and not os.path.isabs(source_csv):
            source_csv = os.path.join(base_dir, source_csv)

        predictors_raw = raw["predictors"]
        if not predictors_raw:
            raise ConfigError(f"Config file {path} defines no predictors.")

        predictors = [
            PredictorSpec(
                column=p["column"],
                label=p.get("label", p["column"]),
                description=p.get("description", ""),
            )
            for p in predictors_raw
        ]

        winsorize_limits = {
            col: tuple(limits) for col, limits in (raw.get("winsorize_limits") or {}).items()
        }
        implicit_nulls = raw.get("implicit_nulls") or {}
        interpolation = raw.get("interpolation") or {}

        return DatasetConfig(
            name=dataset["name"],
            description=dataset.get("description", ""),
            source_csv=source_csv,
            id_columns=raw.get("id_columns") or [],
            target_column=raw["target_column"],
            column_renames=raw.get("column_renames") or {},
            zero_as_null=implicit_nulls.get("zero_as_null") or [],
            range_as_null=implicit_nulls.get("range_as_null") or {},
            group_column=interpolation.get("group_column", ""),
            sort_columns=interpolation.get("sort_columns") or [],
            winsorize_limits=winsorize_limits,
            drop_columns=raw.get("drop_columns") or [],
            predictors=predictors,
            model_params=raw.get("model") or {},
            config_path=path,
        )
    except KeyError as e:
        raise ConfigError(f"Config file {path} is missing required key: {e}") from e


def _normalize_header(col: str) -> str:
    """Same normalization data_prep applies to raw CSV headers, used here
    to check a config against a CSV's *raw*, unprocessed column names."""
    return col.strip().title().replace("  ", " ").replace(" ", "_")


def validate_schema(config: DatasetConfig, raw_columns: list) -> list:
    """
    Check that a raw CSV's columns, once normalized and renamed the same
    way the cleaning pipeline will, cover everything the config expects.

    Returns a list of human-readable problem descriptions; an empty list
    means the config and CSV are compatible.
    """
    problems = []

    normalized = {_normalize_header(c) for c in raw_columns}
    renamed = {config.column_renames.get(c, c) for c in normalized}

    if not config.target_column:
        problems.append("Config has no target_column set.")
    elif config.target_column not in renamed:
        problems.append(
            f"Target column '{config.target_column}' not found in the CSV after "
            f"cleaning. Check column_renames in the config, or the CSV's headers."
        )

    for col in config.id_columns:
        if col not in renamed:
            problems.append(
                f"ID column '{col}' not found in the CSV after cleaning."
            )

    if not config.predictors:
        problems.append("Config defines no predictors.")

    for spec in config.predictors:
        if spec.column not in renamed:
            problems.append(
                f"Predictor column '{spec.column}' not found in the CSV after "
                f"cleaning (check column_renames in the config)."
            )

    if config.group_column and config.group_column not in renamed:
        problems.append(
            f"Interpolation group_column '{config.group_column}' not found "
            f"in the CSV after cleaning."
        )

    return problems
