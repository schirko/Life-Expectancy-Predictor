"""
Train the Random Forest life expectancy model and save it to disk for the
Streamlit app to load. Fully config-driven: which CSV, which columns, which
predictors, and how to clean them all come from a YAML config file (see
config/who_life_expectancy.yaml), not from hardcoded values in this script.

Usage:
    python train_model.py                          # uses the default config
    python train_model.py config/my_other.yaml      # uses a different config
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from config_loader import ConfigError, load_config, validate_schema
from data_prep import feature_ranges, get_model_ready_df, missingness_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "who_life_expectancy.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")
DIAGNOSTICS_PATH = os.path.join(BASE_DIR, "model", "diagnostics.json")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data.csv")

# How many top features (by permutation importance) to compute partial
# dependence plots for. Kept small since PDP computation cost grows with
# feature count and grid resolution.
N_PDP_FEATURES = 3
PDP_GRID_RESOLUTION = 20


def compute_cv_comparison(config, df, x, y, n_estimators, random_state, n_splits=5):
    """
    Compare random-fold CV against group-fold CV (whole entities held out),
    and break the group-CV errors down by entity.

    Returns None if the config defines no group column to hold out.
    Everything here is computed from the data -- nothing is hardcoded --
    so it stays correct if the dataset or config changes.
    """
    group_col = config.group_column
    if not group_col or group_col not in df.columns:
        return None
    groups = df[group_col].values

    def make_model():
        return RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    interval = {"covered": np.zeros(len(y), dtype=bool), "width": np.zeros(len(y))}

    def run_cv(splitter, use_groups, track_interval=False):
        """One fit per fold; returns per-fold R2 and out-of-fold predictions.
        If track_interval, also records whether the Predict page's 80%
        tree-spread interval (10th-90th percentile across trees) contained
        the true value for each held-out row."""
        oof = np.zeros(len(y))
        fold_r2 = []
        split_args = (x, y, groups) if use_groups else (x, y)
        for train_idx, test_idx in splitter.split(*split_args):
            m = make_model().fit(x[train_idx], y[train_idx])
            oof[test_idx] = m.predict(x[test_idx])
            fold_r2.append(float(r2_score(y[test_idx], oof[test_idx])))
            if track_interval:
                tree_preds = np.array([t.predict(x[test_idx]) for t in m.estimators_])
                lo, hi = np.percentile(tree_preds, [10, 90], axis=0)
                interval["covered"][test_idx] = (y[test_idx] >= lo) & (y[test_idx] <= hi)
                interval["width"][test_idx] = hi - lo
        return np.array(fold_r2), oof

    def summarize(fold_r2, oof):
        return {
            "fold_r2": [round(float(s), 4) for s in fold_r2],
            "r2_mean": float(fold_r2.mean()),
            "r2_std": float(fold_r2.std()),
            "oof_mae": float(mean_absolute_error(y, oof)),
            "oof_r2": float(r2_score(y, oof)),
        }

    random_r2, random_oof = run_cv(KFold(n_splits, shuffle=True, random_state=random_state), False)
    group_r2, group_oof = run_cv(GroupKFold(n_splits), True, track_interval=True)

    # Why was the original un-shuffled "5-fold CV" so close to the group result?
    # sklearn's cv=5 for a regressor is KFold WITHOUT shuffling. If the rows
    # happen to be sorted by entity, each fold is a contiguous block of
    # entities -- i.e. almost a group split by accident.
    sorted_by_group = bool((groups[:-1] <= groups[1:]).all())
    plain_r2, _ = run_cv(KFold(n_splits, shuffle=False), False)
    straddling = set()
    for train_idx, test_idx in KFold(n_splits, shuffle=False).split(x):
        straddling |= set(groups[train_idx]) & set(groups[test_idx])

    # Per-entity errors under group CV (every row predicted by a model
    # that never saw that entity).
    err_df = pd.DataFrame(
        {"group": groups, "abs_err": np.abs(y - group_oof)}
    )
    if "Status" in df.columns:
        err_df["status"] = df["Status"].values
    by_group = (
        err_df.groupby("group")["abs_err"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    per_group = [
        {"group": str(g), "mean_abs_error": round(float(r["mean"]), 3), "n_rows": int(r["count"])}
        for g, r in by_group.iterrows()
    ]

    by_status = []
    if "status" in err_df.columns:
        for status, sub in err_df.groupby("status"):
            by_status.append(
                {
                    "status": str(status),
                    "mean_abs_error": round(float(sub["abs_err"].mean()), 3),
                    "n_rows": int(len(sub)),
                    "n_groups": int(sub["group"].nunique()),
                }
            )

    return {
        "group_column": group_col,
        "n_groups": int(len(set(groups))),
        "avg_rows_per_group": round(len(y) / len(set(groups)), 1),
        "n_splits": n_splits,
        "random": summarize(random_r2, random_oof),
        "group": summarize(group_r2, group_oof),
        "unshuffled": {
            "r2_mean": float(plain_r2.mean()),
            "data_sorted_by_group": sorted_by_group,
            "groups_straddling_folds": int(len(straddling)),
        },
        # Does the Predict page's "80%" interval still hold for entities the
        # model has never seen? (Compare with calibration.empirical_coverage_pct,
        # which was measured on a random split.)
        "interval_unseen": {
            "stated_pct": 80,
            "coverage_pct": round(float(interval["covered"].mean() * 100), 1),
            "mean_width_years": round(float(interval["width"].mean()), 2),
        },
        "per_group_error": per_group,
        "median_group_error": round(float(by_group["mean"].median()), 3),
        "pct_groups_over_3yrs": round(float((by_group["mean"] > 3).mean() * 100), 1),
        "error_by_status": by_status,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    print(f"Loading config: {config_path}")
    config = load_config(config_path, base_dir=BASE_DIR)
    print(f"Dataset: {config.name}")

    if not os.path.exists(config.source_csv):
        raise FileNotFoundError(
            f"CSV not found at '{config.source_csv}' (from dataset.source_csv "
            f"in {config_path}). Check that path."
        )

    # Validate the config against the CSV's actual (raw, pre-cleaning)
    # columns before doing any real work, so a mismatch fails fast with a
    # clear message instead of a cryptic KeyError partway through cleaning.
    raw_columns = list(pd.read_csv(config.source_csv, nrows=0).columns)
    problems = validate_schema(config, raw_columns)
    if problems:
        print("\nConfig/schema validation FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise ConfigError(
            f"{len(problems)} problem(s) found between the config and the CSV. "
            f"Fix the config or the CSV and try again."
        )
    print("Schema validation passed.")

    print("Loading and cleaning data...")
    df = get_model_ready_df(config)
    df = df.dropna(subset=config.predictor_columns + [config.target_column])
    print(f"Model-ready rows: {len(df)}")

    x = df[config.predictor_columns].values
    y = df[config.target_column].values

    mp = config.model_params
    test_size = mp.get("test_size", 0.2)
    random_state = mp.get("random_state", 42)
    n_estimators = mp.get("n_estimators", 200)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test R2:   {r2:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Cross-validation, done two explicit ways so the difference is visible:
    #   - random folds: rows are shuffled, so different years of the SAME
    #     entity (country) can land in both train and test
    #   - group folds: whole entities are held out, i.e. "unseen countries"
    # The group version is the headline CV number, because it matches how
    # the model would really be used (predicting for a country it hasn't seen).
    print("Running cross-validation (random folds vs. held-out countries)...")
    cv_report = compute_cv_comparison(config, df, x, y, n_estimators, random_state)
    if cv_report is None:
        raise ConfigError(
            "interpolation.group_column must be set in the config (and exist in the "
            "data) to run group cross-validation."
        )
    cv_scores = np.array(cv_report["group"]["fold_r2"])
    print(f"Random-fold CV R2:   mean={cv_report['random']['r2_mean']:.4f}, std={cv_report['random']['r2_std']:.4f}")
    print(f"Group (country) CV R2: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")
    print(f"  group fold scores: {[round(s, 4) for s in cv_scores.tolist()]}")
    print(
        f"  out-of-fold MAE: random={cv_report['random']['oof_mae']:.3f} yrs, "
        f"group={cv_report['group']['oof_mae']:.3f} yrs"
    )

    importances = dict(
        sorted(
            zip(config.predictor_columns, model.feature_importances_.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    predictors_meta = [
        {
            "column": p.column,
            "label": p.label,
            "description": p.description,
            "input_type": p.input_type,
            "show_in_ui": p.show_in_ui,
        }
        for p in config.predictors
    ]

    metadata = {
        "dataset_name": config.name,
        "dataset_description": config.description,
        "config_path": os.path.relpath(config_path, BASE_DIR),
        "target_col": config.target_column,
        "predictor_cols": config.predictor_columns,
        "predictors": predictors_meta,
        "test_r2": r2,
        "test_mae": mae,
        "test_rmse": rmse,
        # Headline CV = group folds (whole countries held out). The random-fold
        # figures are kept alongside for comparison on the Diagnostics page.
        "cv_method": f"GroupKFold(5) by {config.group_column}",
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "cv_r2_folds": [round(float(s), 4) for s in cv_scores.tolist()],
        "cv_random_r2_mean": cv_report["random"]["r2_mean"],
        "cv_random_r2_std": cv_report["random"]["r2_std"],
        "n_rows": len(df),
        "feature_importances": importances,
        "feature_ranges": feature_ranges(df, config),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")

    # ---- Diagnostics: residuals, permutation importance, partial ----
    # ---- dependence, uncertainty calibration, and data quality. ----
    # Kept in a separate file from metadata.json since these are larger
    # and only needed by the Diagnostics page, not on every app boot.
    print("\nComputing diagnostics...")

    residuals = (y_test - y_pred).tolist()

    print("  Running permutation importance (uses the held-out test set)...")
    perm = permutation_importance(
        model, x_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    permutation_importances = sorted(
        [
            {
                "feature": col,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for col, mean, std in zip(
                config.predictor_columns, perm.importances_mean, perm.importances_std
            )
        ],
        key=lambda d: d["importance_mean"],
        reverse=True,
    )

    print(f"  Computing partial dependence for top {N_PDP_FEATURES} features...")
    top_features = [d["feature"] for d in permutation_importances[:N_PDP_FEATURES]]
    partial_dependence_data = {}
    for feat in top_features:
        idx = config.predictor_columns.index(feat)
        pdp = partial_dependence(
            model, x_train, features=[idx], kind="average", grid_resolution=PDP_GRID_RESOLUTION
        )
        partial_dependence_data[feat] = {
            "grid_values": pdp["grid_values"][0].tolist(),
            "predicted": pdp["average"][0].tolist(),
        }

    print("  Checking uncertainty interval calibration...")
    all_tree_preds = np.array([tree.predict(x_test) for tree in model.estimators_])
    lower_80 = np.percentile(all_tree_preds, 10, axis=0)
    upper_80 = np.percentile(all_tree_preds, 90, axis=0)
    covered_80 = (y_test >= lower_80) & (y_test <= upper_80)
    calibration = {
        "stated_interval_pct": 80,
        "empirical_coverage_pct": round(float(covered_80.mean() * 100), 1),
        "note": (
            "Empirical coverage is how often the true value actually fell within "
            "the stated 10th-90th percentile tree-spread band, measured on the "
            "held-out test set. Close to 80% = well-calibrated; higher = "
            "conservative (wider than needed); lower = overconfident."
        ),
    }

    print("  Building missingness report...")
    missingness = missingness_report(config)

    diagnostics = {
        "residuals": {
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "residuals": residuals,
        },
        "permutation_importance": permutation_importances,
        "partial_dependence": partial_dependence_data,
        "calibration": calibration,
        "missingness": missingness,
        "group_cv": cv_report,
    }
    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {DIAGNOSTICS_PATH}")

    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
