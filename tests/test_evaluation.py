"""The honest test really holds whole countries out, and a small model still learns."""
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from train_model import rf_pipeline


def test_no_country_is_in_both_training_and_test(config, clean_df):
    groups = clean_df[config.group_column].values
    x = clean_df[config.predictor_columns].values
    for tr, te in GroupKFold(5).split(x, groups=groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_small_model_predicts_unseen_countries(config, clean_df):
    """A quick 30-tree model trained on 4/5 of countries should still explain most of the variation."""
    groups = clean_df[config.group_column].values
    x = clean_df[config.predictor_columns].values
    y = clean_df[config.target_column].values
    tr, te = next(GroupKFold(5).split(x, y, groups))
    model = rf_pipeline(config, n_estimators=30, random_state=42, n_jobs=-1).fit(x[tr], y[tr])
    assert r2_score(y[te], model.predict(x[te])) > 0.85
