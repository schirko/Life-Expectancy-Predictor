"""
Milestone 19: a wider uncertainty band for developing countries.

What we are looking for: the model's 80% band (10th-90th percentile of the
forest's 200 trees) held the true value 80% of the time for developed
countries but only about 76.5% for developing ones (milestone 18). Can the
band be re-sized so it means "80%" for both groups?

How: every row is predicted by a model that never saw that country
(GroupKFold(5) by Country, leak-free clipping). The band is stretched around
the prediction by a factor k (k = 1 is the raw tree spread). Three versions:
  A. raw band (k = 1 for everyone)
  B. one factor for everyone, chosen so overall coverage is 80%
  C. one factor per group (developed / developing), each chosen so that
     group's coverage is 80%
To keep this honest, the factors are cross-fitted: for each fold, k is
chosen on the other four folds' countries and then applied to the held-out
fold, so no country helps size its own band.

Groups use the corrected Status labels from analysis/fairness_check.py.
Run from the project folder:   python -m analysis.interval_calibration
Saves model/interval_calibration.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from analysis.fairness_check import MISLABELLED_AS_DEVELOPING
from config_loader import load_config
from data_prep import get_unclipped_df
from train_model import rf_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "who_life_expectancy.yaml")
OUT_PATH = os.path.join(BASE_DIR, "model", "interval_calibration.json")
TARGET = 0.80


def needed_factor(actual, pred, lo, hi, target=TARGET):
    """Smallest stretch k so that [pred - k*(pred-lo), pred + k*(hi-pred)] holds `target` of the rows."""
    below = actual < pred
    half = np.where(below, pred - lo, hi - pred)
    half = np.maximum(half, 1e-9)
    ratio = np.abs(actual - pred) / half  # the k each row needs to be covered
    return float(np.quantile(ratio, target))


def stretch(pred, lo, hi, k):
    return pred - k * (pred - lo), pred + k * (hi - pred)


def summary(sub, lo, hi):
    covered = (sub["actual"] >= lo) & (sub["actual"] <= hi)
    return {"coverage_pct": round(float(covered.mean() * 100), 1),
            "width_years": round(float((hi - lo).mean()), 2)}


def main():
    config = load_config(CONFIG_PATH, base_dir=BASE_DIR)
    df = get_unclipped_df(config).dropna(
        subset=config.predictor_columns + [config.target_column]).reset_index(drop=True)
    x = df[config.predictor_columns].values
    y = df[config.target_column].values
    groups = df[config.group_column].values
    mp = config.model_params

    pred, lo, hi, fold = (np.zeros(len(y)) for _ in range(4))
    for f, (tr, te) in enumerate(GroupKFold(5).split(x, y, groups)):
        m = rf_pipeline(config, mp.get("n_estimators", 200), mp.get("random_state", 42), n_jobs=-1)
        m.fit(x[tr], y[tr])
        trees = np.array([t.predict(m.named_steps["clip"].transform(x[te])) for t in m.named_steps["rf"].estimators_])
        pred[te] = trees.mean(axis=0)
        lo[te], hi[te] = np.percentile(trees, [10, 90], axis=0)
        fold[te] = f

    w = pd.DataFrame({"Country": groups, "actual": y, "pred": pred, "lo": lo, "hi": hi, "fold": fold})
    w["group"] = np.where(w["Country"].isin(MISLABELLED_AS_DEVELOPING), "Developed", df["Status"])

    # Cross-fitted bands: factors chosen on the other folds, applied to this one.
    out = {v: (np.zeros(len(w)), np.zeros(len(w))) for v in ("A_raw", "B_one_factor", "C_per_group")}
    out["A_raw"] = (w["lo"].values.copy(), w["hi"].values.copy())
    for f in range(5):
        te = (w["fold"] == f).values
        cal = w[~te]
        k_all = needed_factor(cal["actual"], cal["pred"], cal["lo"], cal["hi"])
        l, h = stretch(w.loc[te, "pred"], w.loc[te, "lo"], w.loc[te, "hi"], k_all)
        out["B_one_factor"][0][te], out["B_one_factor"][1][te] = l, h
        for g in ("Developed", "Developing"):
            c = cal[cal["group"] == g]
            k_g = needed_factor(c["actual"], c["pred"], c["lo"], c["hi"])
            sel = te & (w["group"] == g).values
            l, h = stretch(w.loc[sel, "pred"], w.loc[sel, "lo"], w.loc[sel, "hi"], k_g)
            out["C_per_group"][0][sel], out["C_per_group"][1][sel] = l, h

    evaluation = {}
    for v, (l, h) in out.items():
        evaluation[v] = {"All": summary(w, l, h)}
        for g in ("Developed", "Developing"):
            sel = (w["group"] == g).values
            evaluation[v][g] = summary(w[sel], l[sel], h[sel])

    # Factors fitted on all countries, for use from now on.
    factors = {"all": round(needed_factor(w["actual"], w["pred"], w["lo"], w["hi"]), 3)}
    for g in ("Developed", "Developing"):
        c = w[w["group"] == g]
        factors[g] = round(needed_factor(c["actual"], c["pred"], c["lo"], c["hi"]), 3)

    results = {
        "method": "Cross-fitted stretch of the 10th-90th percentile tree band, GroupKFold(5) by Country; "
                  "groups use corrected Status labels",
        "target_coverage_pct": TARGET * 100,
        "evaluation": evaluation,
        "factors_fitted_on_all": factors,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    for v, ev in evaluation.items():
        print(v, "  ".join(f"{g}: {s['coverage_pct']}% / {s['width_years']} yrs" for g, s in ev.items()))
    print("Factors (all data):", factors)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
