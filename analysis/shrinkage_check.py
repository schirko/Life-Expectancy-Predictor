"""
Milestone 20: can a correction step undo the "pull toward the middle"?

What we are looking for: on countries it has never seen, the model predicts
too high for the lowest life expectancies and too low for the highest
(milestone 10). Every model in the showdown did the same. Can a simple
correction applied after the model ("recalibration") fix it?

Two ways to measure the pull, and they answer different questions:
  * By TRUE value: group rows by their actual life expectancy and look at the
    average error. Some pull is unavoidable here whenever predictions are
    imperfect (regression to the mean), even for a well-calibrated model.
  * By PREDICTED value (calibration): regress actual on predicted. A slope
    above 1 means the predictions really are squeezed toward the middle and
    stretching them would help; a slope near 1 means they are not.

Corrections compared, each learned honestly (nested): for every outer fold
of countries, the correction is fitted on predictions the model made for the
OTHER training countries (an inner GroupKFold), then applied to the held-out
countries it never saw:
  A. none (raw Random Forest)
  B. linear: actual ~ a + b * prediction
  C. isotonic: a free-form increasing curve from prediction to actual

Run from the project folder:   python -m analysis.shrinkage_check
Saves model/shrinkage_check.json. Takes about a minute.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from analysis.fairness_check import MISLABELLED_AS_DEVELOPING
from config_loader import load_config
from data_prep import get_unclipped_df
from train_model import rf_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "who_life_expectancy.yaml")
OUT_PATH = os.path.join(BASE_DIR, "model", "shrinkage_check.json")


def fit_predict(config, x_tr, y_tr, x_te):
    mp = config.model_params
    m = rf_pipeline(config, mp.get("n_estimators", 200), mp.get("random_state", 42), n_jobs=-1)
    return m.fit(x_tr, y_tr).predict(x_te)


def calibration_slope(actual, pred):
    """Slope of actual regressed on predicted. 1 = well calibrated; >1 = predictions squeezed."""
    return float(LinearRegression().fit(pred.reshape(-1, 1), actual).coef_[0])


def band_bias(actual, pred, n_bands=10):
    """Average error (actual - pred) in the lowest and highest 10% of TRUE values.
    Negative = the model predicted too high."""
    order = np.argsort(actual)
    k = len(actual) // n_bands
    low, high = order[:k], order[-k:]
    return float(np.mean(actual[low] - pred[low])), float(np.mean(actual[high] - pred[high]))


def pred_band_bias(actual, pred, n_bands=10):
    """Average error in the lowest and highest 10% of PREDICTED values (the calibration view)."""
    order = np.argsort(pred)
    k = len(pred) // n_bands
    low, high = order[:k], order[-k:]
    return float(np.mean(actual[low] - pred[low])), float(np.mean(actual[high] - pred[high]))


def evaluate(actual, pred, groups_status):
    lo, hi = band_bias(actual, pred)
    plo, phi = pred_band_bias(actual, pred)
    out = {
        "mae": round(float(mean_absolute_error(actual, pred)), 3),
        "r2": round(float(r2_score(actual, pred)), 4),
        "calibration_slope": round(calibration_slope(actual, pred), 3),
        "bias_lowest_10pct": round(lo, 2),
        "bias_highest_10pct": round(hi, 2),
        "bias_lowest_10pct_of_predictions": round(plo, 2),
        "bias_highest_10pct_of_predictions": round(phi, 2),
    }
    for g in ("Developed", "Developing"):
        sel = groups_status == g
        out[f"mae_{g.lower()}"] = round(float(mean_absolute_error(actual[sel], pred[sel])), 3)
    return out


def main():
    config = load_config(CONFIG_PATH, base_dir=BASE_DIR)
    df = get_unclipped_df(config).dropna(
        subset=config.predictor_columns + [config.target_column]).reset_index(drop=True)
    x = df[config.predictor_columns].values
    y = df[config.target_column].values
    countries = df[config.group_column].values
    status = np.where(np.isin(countries, MISLABELLED_AS_DEVELOPING), "Developed", df["Status"].values)

    raw, lin, iso = (np.zeros(len(y)) for _ in range(3))
    for tr, te in GroupKFold(5).split(x, y, countries):
        # Inner out-of-fold predictions on the training countries, to learn the correction.
        inner = np.zeros(len(tr))
        for itr, ite in GroupKFold(5).split(x[tr], y[tr], countries[tr]):
            inner[ite] = fit_predict(config, x[tr][itr], y[tr][itr], x[tr][ite])
        lin_map = LinearRegression().fit(inner.reshape(-1, 1), y[tr])
        iso_map = IsotonicRegression(out_of_bounds="clip").fit(inner, y[tr])
        # The model itself, trained on all training countries, predicts the held-out ones.
        p = fit_predict(config, x[tr], y[tr], x[te])
        raw[te] = p
        lin[te] = lin_map.predict(p.reshape(-1, 1))
        iso[te] = iso_map.predict(p)

    def by_decile(key):
        order = np.argsort(key)
        return [round(float(np.mean(y[idx] - raw[idx])), 2) for idx in np.array_split(order, 10)]

    results = {
        "decile_bias_raw": {
            "by_true_value": by_decile(y),       # chart: the pull toward the middle
            "by_prediction": by_decile(raw),     # chart: calibration (flat = fine)
        },
        "method": "Nested GroupKFold(5) by Country; corrections fitted on inner out-of-fold predictions only",
        "A_raw": evaluate(y, raw, status),
        "B_linear": evaluate(y, lin, status),
        "C_isotonic": evaluate(y, iso, status),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    for k in ("A_raw", "B_linear", "C_isotonic"):
        r = results[k]
        print(f"{k:<11} MAE {r['mae']:.3f}  R2 {r['r2']:.4f}  slope {r['calibration_slope']:.3f}  "
              f"true-low10 {r['bias_lowest_10pct']:+.2f}  true-high10 {r['bias_highest_10pct']:+.2f}  "
              f"pred-low10 {r['bias_lowest_10pct_of_predictions']:+.2f}  pred-high10 {r['bias_highest_10pct_of_predictions']:+.2f}  "
              f"dev {r['mae_developed']:.2f}  dvg {r['mae_developing']:.2f}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
