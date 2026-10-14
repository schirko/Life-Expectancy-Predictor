"""
Hyperparameter tuning for the Random Forest model, via GridSearchCV with
COUNTRY-HELD-OUT folds (GroupKFold).

Why group folds: the data is a panel (each country appears in up to 16 years).
Ordinary random folds put a country's other years in the training folds, so the
search would reward settings that memorize countries. Holding out whole
countries asks the question we actually care about: which settings predict
countries the model has never seen? The final test set is also held out by
country (GroupShuffleSplit), so the tuned-vs-default comparison is made on
unseen countries too.

Kept as a separate, optional script from train_model.py because a full grid
search takes much longer than a normal training run (order of a minute,
depending on grid size and machine) and most of the time you just want to
retrain quickly with known-good settings.

Usage:
    python tune_hyperparameters.py                     # uses the default config
    python tune_hyperparameters.py config/my_other.yaml # uses a different config

After running, `model/tuning_results.json` is saved. The Diagnostics page
shows its results if present, or a note that tuning hasn't been run yet.
This does NOT overwrite the deployed model (model/rf_model.joblib) -- it
only reports what tuning found. Re-run train_model.py with updated
`model.n_estimators` etc. in the config if you want to adopt the results.
"""
import json
import os
import sys
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from config_loader import ConfigError, load_config, validate_schema
from data_prep import get_model_ready_df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "who_life_expectancy.yaml")
TUNING_RESULTS_PATH = os.path.join(BASE_DIR, "model", "tuning_results.json")

# Kept intentionally small so a run finishes in roughly a minute on a
# typical laptop. Widen this if you have time to spare.
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
}


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    print(f"Loading config: {config_path}")
    config = load_config(config_path, base_dir=BASE_DIR)

    raw_columns = list(pd.read_csv(config.source_csv, nrows=0).columns)
    problems = validate_schema(config, raw_columns)
    if problems:
        raise ConfigError(f"Config/schema validation failed: {problems}")

    print("Loading and cleaning data...")
    df = get_model_ready_df(config).dropna(subset=config.predictor_columns + [config.target_column])
    x = df[config.predictor_columns].values
    y = df[config.target_column].values

    mp = config.model_params
    test_size = mp.get("test_size", 0.2)
    random_state = mp.get("random_state", 42)
    default_n_estimators = mp.get("n_estimators", 200)

    group_col = config.group_column
    if not group_col or group_col not in df.columns:
        raise ConfigError("Group-based tuning needs interpolation.group_column in the config.")
    groups = df[group_col].values

    # Hold out ~20% of COUNTRIES (all their years) as the final test set.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]
    print(f"Train: {len(set(g_train))} countries; test: {len(set(groups[test_idx]))} unseen countries")

    # Keep the previous (random-fold) results, if present, to compare rankings.
    random_baseline = None
    if os.path.exists(TUNING_RESULTS_PATH):
        with open(TUNING_RESULTS_PATH) as f:
            prev = json.load(f)
        if "cv_method" not in prev:
            random_baseline = prev

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    print(f"Grid: {PARAM_GRID}")
    print(f"{n_combos} combinations x 5 country-held-out folds = {n_combos * 5} fits. This may take a while...")

    t0 = time.time()
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        PARAM_GRID,
        cv=GroupKFold(n_splits=5),
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train, groups=g_train)
    elapsed = time.time() - t0
    print(f"Grid search finished in {elapsed:.1f}s")

    best_model = grid_search.best_estimator_
    best_test_r2 = r2_score(y_test, best_model.predict(x_test))

    # For comparison: the currently-deployed default configuration, scored
    # the same way, so the improvement (if any) is measured apples-to-apples.
    default_model = RandomForestRegressor(
        n_estimators=default_n_estimators, random_state=random_state
    )
    default_model.fit(x_train, y_train)
    default_test_r2 = r2_score(y_test, default_model.predict(x_test))

    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best grouped-CV R2 (train folds): {grid_search.best_score_:.4f}")
    print(f"Best params' test R2 (unseen countries): {best_test_r2:.4f}")
    print(f"Default params' test R2 ({{'n_estimators': {default_n_estimators}}}): {default_test_r2:.4f}")

    # Full results table, sorted best-first, for transparency (not just the winner).
    all_results = []
    for params, mean_score, std_score in zip(
        grid_search.cv_results_["params"],
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
    ):
        all_results.append(
            {"params": params, "cv_r2_mean": float(mean_score), "cv_r2_std": float(std_score)}
        )
    all_results.sort(key=lambda r: r["cv_r2_mean"], reverse=True)

    # Does the choice of CV method change which settings look best?
    rank_comparison = None
    if random_baseline:
        rand = {json.dumps(r["params"], sort_keys=True): r["cv_r2_mean"] for r in random_baseline["all_results"]}
        pairs = [(rand[json.dumps(r["params"], sort_keys=True)], r["cv_r2_mean"])
                 for r in all_results if json.dumps(r["params"], sort_keys=True) in rand]
        if len(pairs) > 2:
            rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0]
            rank_comparison = {
                "random_cv_best_params": random_baseline["best_params"],
                "random_cv_best_r2": random_baseline["best_cv_r2"],
                "random_cv_worst_r2": min(p[0] for p in pairs),
                "grouped_cv_worst_r2": min(p[1] for p in pairs),
                "spearman_rank_correlation": round(float(rho), 3),
                "same_winner": random_baseline["best_params"] == grid_search.best_params_,
            }

    results = {
        "cv_method": "GroupKFold(5) by " + group_col + "; test set = held-out countries",
        "n_train_countries": int(len(set(g_train))),
        "n_test_countries": int(len(set(groups[test_idx]))),
        "rank_comparison": rank_comparison,
        "param_grid": PARAM_GRID,
        "n_combinations": n_combos,
        "search_time_seconds": round(elapsed, 1),
        "best_params": grid_search.best_params_,
        "best_cv_r2": float(grid_search.best_score_),
        "best_test_r2": float(best_test_r2),
        "default_params": {"n_estimators": default_n_estimators},
        "default_test_r2": float(default_test_r2),
        "improvement_over_default": round(float(best_test_r2 - default_test_r2), 4),
        "all_results": all_results,
    }

    os.makedirs(os.path.dirname(TUNING_RESULTS_PATH), exist_ok=True)
    with open(TUNING_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved tuning results to {TUNING_RESULTS_PATH}")


if __name__ == "__main__":
    main()
