"""Outlier clipping learns its limits from training rows only (milestone 16)."""
import numpy as np
from scipy.stats.mstats import winsorize

from data_prep import FoldWinsorizer, fold_winsorizer, winsorize_df
from train_model import rf_pipeline


def test_matches_the_old_method_on_all_rows(config, clean_df):
    """Fitted on all rows, the new clipper gives exactly the old scipy result."""
    old = winsorize_df(clean_df, config)[config.predictor_columns].values
    new = fold_winsorizer(config).fit_transform(clean_df[config.predictor_columns].values)
    assert np.nanmax(np.abs(old - new)) == 0.0


def test_test_rows_do_not_move_the_limits():
    """Changing the held-out rows must not change limits learned from the training rows."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 1))
    test = rng.normal(size=(50, 1))
    clipper = FoldWinsorizer(limits=[(0.05, 0.05)]).fit(train)
    limits = (clipper.lower_.copy(), clipper.upper_.copy())
    clipper_again = FoldWinsorizer(limits=[(0.05, 0.05)]).fit(train)  # same training rows
    assert np.array_equal(limits[0], clipper_again.lower_) and np.array_equal(limits[1], clipper_again.upper_)
    clipped = clipper.transform(test * 100)  # extreme test values get clipped to the TRAINING limits
    assert clipped.min() >= limits[0][0] and clipped.max() <= limits[1][0]


def test_lower_limit_matches_scipy_definition():
    x = np.arange(100, dtype=float).reshape(-1, 1)
    clipper = FoldWinsorizer(limits=[(0.1, 0.0)]).fit(x)
    expected = np.asarray(winsorize(x[:, 0], limits=(0.1, 0.0)))
    assert np.array_equal(clipper.transform(x)[:, 0], expected)


def test_clipping_is_the_first_pipeline_step(config):
    pipe = rf_pipeline(config, n_estimators=5, random_state=0)
    assert isinstance(pipe.steps[0][1], FoldWinsorizer)
