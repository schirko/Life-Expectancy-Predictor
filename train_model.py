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
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from config_loader import ConfigError, load_config, validate_schema
from data_prep import (
    _clean_column_names,
    _fix_implicit_nulls,
    feature_ranges,
    get_model_ready_df,
    missingness_report,
)

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
        "hard_cases": compute_hard_country_analysis(config, df, y, group_oof, groups),
    }


def compute_hard_country_analysis(config, df, y, oof, groups, top_n=15):
    """
    Why are some entities (countries) hard to predict? Uses the out-of-fold
    predictions from group CV, so every row was predicted by a model that
    never saw that entity. Everything is computed from the data.

    Reports: how concentrated the error is, whether the model shrinks toward
    the middle (over-predicting the lowest targets, under-predicting the
    highest), which predictors' entity averages track the error, and for the
    hardest entities their signed bias and how many predictors had NO data
    (so were filled with the global median) before imputation.
    """
    target = config.target_column
    preds = config.predictor_columns
    work = df[[config.group_column] + preds].copy()
    work["actual"] = y
    work["pred"] = oof
    work["err"] = y - oof  # positive = model under-predicted
    work["abs_err"] = work["err"].abs()
    if "Status" in df.columns:
        work["status"] = df["Status"].values

    by = work.groupby(config.group_column)
    summ = by.agg(
        mae=("abs_err", "mean"), bias=("err", "mean"), mean_actual=("actual", "mean"),
        n_rows=("err", "size"),
    )
    if "status" in work.columns:
        summ["status"] = by["status"].first()
    summ = summ.sort_values("mae", ascending=False)

    # 1. Concentration of error
    n_top = max(1, int(round(len(summ) * 0.10)))
    top_set = set(summ.index[:n_top])
    keep = ~work[config.group_column].isin(top_set)
    concentration = {
        "top_pct": 10,
        "n_top_groups": n_top,
        "share_of_total_error_pct": round(float(work.loc[~keep, "abs_err"].sum() / work["abs_err"].sum() * 100), 1),
        "r2_all": round(float(r2_score(y, oof)), 3),
        "r2_without_top": round(float(r2_score(y[keep.values], oof[keep.values])), 3),
        "hardest_group": str(summ.index[0]),
        "hardest_mae": round(float(summ["mae"].iloc[0]), 2),
        "r2_without_hardest": round(float(r2_score(
            y[(work[config.group_column] != summ.index[0]).values],
            oof[(work[config.group_column] != summ.index[0]).values])), 3),
    }

    # 2. Shrinkage toward the middle: signed error by band of the true target
    edges = np.unique(np.quantile(y, [0, 0.1, 0.3, 0.7, 0.9, 1.0]))
    bands = pd.cut(work["actual"], edges, include_lowest=True)
    labels = ["lowest 10%", "low", "middle 40%", "high", "highest 10%"]
    by_band = []
    for i, (band, sub) in enumerate(work.groupby(bands, observed=True)):
        by_band.append({
            "band": labels[i] if len(edges) == 6 else str(band),
            "range": f"{band.left:.0f}-{band.right:.0f}",
            "n_rows": int(len(sub)),
            "bias": round(float(sub["err"].mean()), 2),
            "mean_abs_error": round(float(sub["abs_err"].mean()), 2),
        })

    # 3. Which predictors' entity averages track entity error?
    feat_means = by[preds].mean()
    corr_rows = []
    for p in preds:
        rho = pd.concat([feat_means[p], summ["mae"]], axis=1).corr(method="spearman").iloc[0, 1]
        if pd.notnull(rho):
            corr_rows.append({"feature": p, "spearman": round(float(rho), 2)})
    corr_rows.sort(key=lambda r: -abs(r["spearman"]))

    # 4. Missing-data check: predictors with NO data for an entity, before imputation
    raw = _fix_implicit_nulls(_clean_column_names(pd.read_csv(config.source_csv), config), config)
    gcol = config.group_column
    present = [p for p in preds if p in raw.columns]
    empty = raw.groupby(gcol)[present].apply(lambda t: t.isna().all()).sum(axis=1)
    summ["empty_predictors"] = empty.reindex(summ.index).fillna(0).astype(int)
    summ["empty_group"] = pd.cut(summ["empty_predictors"], [-1, 0, 1, 99], labels=["0", "1", "2+"])
    missing_check = [
        {"empty_predictors": str(k), "n_groups": int(len(v)),
         "mean_abs_error": round(float(v["mae"].mean()), 2)}
        for k, v in summ.groupby("empty_group", observed=True)
    ]
    rho_missing = summ[["mae", "empty_predictors"]].corr(method="spearman").iloc[0, 1]

    hardest = []
    for g, r in summ.head(top_n).iterrows():
        hardest.append({
            "group": str(g),
            "mean_abs_error": round(float(r["mae"]), 2),
            "bias": round(float(r["bias"]), 2),
            "mean_actual": round(float(r["mean_actual"]), 1),
            "status": str(r["status"]) if "status" in summ.columns else "",
            "empty_predictors": int(r["empty_predictors"]),
        })
    return {
        "concentration": concentration,
        "by_target_band": by_band,
        "feature_error_correlations": corr_rows[:5],
        "missing_data_check": missing_check,
        "missing_data_spearman": round(float(rho_missing), 2),
        "hardest": hardest,
    }


def compute_model_showdown(config, df, x, y, groups, n_estimators, random_state, n_splits=5):
    """
    Score several model families on the SAME country-held-out folds, so the
    only thing that differs is the algorithm. The point is diagnostic: if a
    different algorithm also misses the same countries by the same amount,
    the cause is the data, not the Random Forest.

    Returns None if no group column is configured.
    """
    group_col = config.group_column
    if not group_col or group_col not in df.columns:
        return None

    factories = {
        "Random Forest (deployed)": lambda: RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=random_state),
        "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(random_state=random_state),
        "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    }
    lo_cut, hi_cut = np.quantile(y, [0.10, 0.90])
    n_top = max(1, int(round(len(set(groups)) * 0.10)))

    rows, country_err = [], {}
    for name, make in factories.items():
        oof = np.zeros(len(y))
        fold_r2 = []
        for tr, te in GroupKFold(n_splits).split(x, y, groups):
            m = make().fit(x[tr], y[tr])
            oof[te] = m.predict(x[te])
            fold_r2.append(float(r2_score(y[te], oof[te])))
        err = y - oof
        per_country = pd.Series(np.abs(err)).groupby(groups).mean().sort_values(ascending=False)
        country_err[name] = per_country
        rows.append({
            "model": name,
            "r2_mean": round(float(np.mean(fold_r2)), 4),
            "r2_std": round(float(np.std(fold_r2)), 4),
            "mae": round(float(np.abs(err).mean()), 3),
            "bias_lowest_10pct": round(float(err[y <= lo_cut].mean()), 2),
            "bias_highest_10pct": round(float(err[y >= hi_cut].mean()), 2),
            "hardest_country": str(per_country.index[0]),
            "hardest_country_mae": round(float(per_country.iloc[0]), 2),
            "top10pct_error_share": round(float(per_country.head(n_top).sum() / per_country.sum() * 100), 1),
        })

    # Do the models agree on WHICH countries are hard? (Spearman on country error.)
    base = country_err["Random Forest (deployed)"]
    for r in rows:
        other = country_err[r["model"]].reindex(base.index)
        r["rank_agreement_with_rf"] = round(float(base.corr(other, method="spearman")), 2)

    # Hardest countries shared by every model (top-n in all lists)
    shared = set.intersection(*[set(c.index[:n_top]) for c in country_err.values()])
    return {
        "n_splits": n_splits,
        "models": rows,
        "n_top_countries": n_top,
        "hard_in_every_model": sorted(shared),
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
    print("Comparing model families on the same country-held-out folds...")
    showdown = compute_model_showdown(config, df, x, y, df[config.group_column].values, n_estimators, random_state)
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
        "model_showdown": showdown,
    }
    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {DIAGNOSTICS_PATH}")

    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
