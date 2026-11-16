"""
Milestone 15: how much did the outlier-clipping leak change the scores?

Before the fix, the cleaning step winsorized (clipped) every column using ALL
rows, before any train/test split. That had two effects:
  1. Feature clip limits were partly set by the test rows (a data leak).
  2. The target itself (life expectancy) was clipped: the lowest 1% of values
     were raised, so the model was scored against altered "true" values.

This script separates the two effects by scoring four versions on the same
splits as train_model.py (80/20 random split, and 5 country-held-out folds):

  A. old pipeline: features clipped on all rows, target clipped
  B. features clipped per training split, target still clipped
  C. features clipped on all rows, target NOT clipped
  D. new pipeline: features clipped per training split, target NOT clipped

Run from the project folder:   python -m analysis.leak_check
Saves model/leak_check.json.
"""
import json
import os

import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from config_loader import load_config
from data_prep import fold_winsorizer, get_unclipped_df, winsorize_df

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "who_life_expectancy.yaml")
OUT_PATH = os.path.join(BASE_DIR, "model", "leak_check.json")


def main():
    config = load_config(CONFIG_PATH, base_dir=BASE_DIR)
    mp = config.model_params
    rs, n_est, test_size = mp.get("random_state", 42), mp.get("n_estimators", 200), mp.get("test_size", 0.2)
    preds, target = config.predictor_columns, config.target_column

    raw = get_unclipped_df(config).dropna(subset=preds + [target]).reset_index(drop=True)
    groups = raw[config.group_column].values
    x_raw = raw[preds].values
    x_all_clipped = winsorize_df(raw, config)[preds].values  # features clipped on all rows (old)
    y_true = raw[target].values
    lo, hi = config.winsorize_limits.get(target, (0.0, 0.0))
    y_clipped = np.asarray(winsorize(y_true, limits=(lo, hi)), dtype=float)  # old target

    def rf():
        return RandomForestRegressor(n_estimators=n_est, random_state=rs, n_jobs=-1)

    def per_fold():
        return Pipeline([("clip", fold_winsorizer(config)), ("rf", rf())])

    variants = {
        "A_old": (x_all_clipped, y_clipped, rf),
        "B_features_per_fold": (x_raw, y_clipped, per_fold),
        "C_target_unclipped": (x_all_clipped, y_true, rf),
        "D_new": (x_raw, y_true, per_fold),
    }
    results = {}
    for name, (x, y, make) in variants.items():
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=rs)
        m = make().fit(x[tr], y[tr])
        p = m.predict(x[te])
        oof = np.zeros(len(y))
        fold_r2 = []
        for gtr, gte in GroupKFold(5).split(x, y, groups):
            oof[gte] = make().fit(x[gtr], y[gtr]).predict(x[gte])
            fold_r2.append(r2_score(y[gte], oof[gte]))
        results[name] = {
            "random_split_r2": round(float(r2_score(y[te], p)), 4),
            "random_split_mae": round(float(mean_absolute_error(y[te], p)), 3),
            "group_cv_r2": round(float(np.mean(fold_r2)), 4),
            "group_oof_mae": round(float(mean_absolute_error(y, oof)), 3),
        }
        print(name, results[name])

    n_target_changed = int((y_clipped != y_true).sum())
    results["target_rows_changed_by_old_clipping"] = n_target_changed
    results["target_floor_old"] = round(float(y_clipped.min()), 1)
    results["target_min_true"] = round(float(y_true.min()), 1)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Old clipping changed {n_target_changed} target values "
          f"(true minimum {y_true.min():.1f} raised to {y_clipped.min():.1f}).")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
