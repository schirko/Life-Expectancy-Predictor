"""
Milestone 25: does outside data help where the 13 WHO inputs run out?

Run from the project folder, after `python -m analysis.external_data`:

    python -m analysis.outside_data_check

Notebooks 5 and 7 concluded that the hard countries are hard because the inputs
do not contain what makes them different, and named three candidates: conflict,
epidemics, and reporting quality. This script tests each one, and all together,
on exactly the honest test used everywhere else: GroupKFold(5) by country with
leak-free clipping, the same folds for every variant so the comparison is paired.

Variants:
  base        the deployed 13 inputs
  +conflict   + log(1 + UCDP battle-related deaths)
  +epidemic   + EM-DAT epidemic events and log(1 + epidemic deaths)
  +reporting  + share of inputs filled in, year-to-year volatility of inputs
  +all        all of the above (18 inputs)

For each variant: R² (mean over folds), average miss on unseen countries, the
miss on the 18 hardest countries of the base model, the developed/developing
gap with corrected labels, and a paired bootstrap over countries of the change
in average miss versus base (so "+0.03 years" can be told from noise).

Writes model/outside_data_check.json. The hard numbers for the docs are in
"variants" and "paired_bootstrap"; "per_country" has every country's miss under
every variant for the notebook.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config  # noqa: E402
from data_prep import FoldWinsorizer, get_unclipped_df  # noqa: E402
from analysis.fairness_check import MISLABELLED_AS_DEVELOPING  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTSIDE = os.path.join(ROOT, "data", "external", "outside_data_by_country_year.csv")
OUT_JSON = os.path.join(ROOT, "model", "outside_data_check.json")

EXTRA = {
    "+conflict": ["log_battle_deaths"],
    "+epidemic": ["epidemic_events", "log_epidemic_deaths"],
    "+reporting": ["inputs_filled_pct", "input_volatility"],
}
EXTRA["+all"] = sum(EXTRA.values(), [])
N_HARD = 18
N_BOOT = 2000


def pipeline(config, cols):
    limits = [tuple(config.winsorize_limits.get(c, (0.0, 0.0))) for c in cols]  # new inputs: no clipping
    mp = config.model_params
    return Pipeline([
        ("clip", FoldWinsorizer(limits=limits)),
        ("rf", RandomForestRegressor(n_estimators=mp.get("n_estimators", 200),
                                     random_state=mp.get("random_state", 42), n_jobs=-1)),
    ])


def out_of_fold(config, df, cols, folds):
    x, y = df[cols].values, df[config.target_column].values
    pred, fold_r2 = np.zeros(len(y)), []
    for tr, te in folds:
        m = pipeline(config, cols).fit(x[tr], y[tr])
        pred[te] = m.predict(x[te])
        fold_r2.append(r2_score(y[te], pred[te]))
    return pred, fold_r2


def main():
    config = load_config(os.path.join(ROOT, "config", "who_life_expectancy.yaml"), base_dir=ROOT)
    df = get_unclipped_df(config).dropna(subset=config.predictor_columns + [config.target_column]).reset_index(drop=True)
    outside = pd.read_csv(OUTSIDE)
    df = df.merge(outside, how="left", on=["Country", "Year"])
    assert df[EXTRA["+all"]].notna().all().all(), "every WHO country-year must have outside data"
    df["Status_corrected"] = np.where(df["Country"].isin(MISLABELLED_AS_DEVELOPING), "Developed", df["Status"])

    y, groups = df[config.target_column].values, df["Country"].values
    folds = list(GroupKFold(5).split(df, y, groups))
    variants = {"base": [], **EXTRA}

    per_country, summary, preds = {}, {}, {}
    for name, extra in variants.items():
        cols = config.predictor_columns + extra
        pred, fold_r2 = out_of_fold(config, df, cols, folds)
        preds[name] = pred
        miss = pd.Series(np.abs(pred - y)).groupby(groups).mean()
        per_country[name] = miss
        gap = pd.Series(np.abs(pred - y)).groupby(df["Status_corrected"].values).mean()
        summary[name] = {
            "inputs": len(cols),
            "r2_mean": round(float(np.mean(fold_r2)), 4),
            "r2_folds": [round(float(r), 4) for r in fold_r2],
            "oof_mae": round(float(mean_absolute_error(y, pred)), 4),
            "mae_developed": round(float(gap["Developed"]), 3),
            "mae_developing": round(float(gap["Developing"]), 3),
            "fairness_gap": round(float(gap["Developing"] - gap["Developed"]), 3),
        }
        print(f"{name:11s} inputs={len(cols):2d}  R²={summary[name]['r2_mean']:.4f}  "
              f"miss={summary[name]['oof_mae']:.3f}  gap={summary[name]['fairness_gap']:.3f}")

    hardest = per_country["base"].sort_values(ascending=False).head(N_HARD).index.tolist()
    for name in variants:
        summary[name]["mae_hardest_18_of_base"] = round(float(per_country[name][hardest].mean()), 3)
        summary[name]["mae_other_countries"] = round(float(per_country[name].drop(hardest).mean()), 3)

    # Paired bootstrap over countries: change in average miss vs base.
    rng = np.random.default_rng(0)
    countries = per_country["base"].index.to_numpy()
    boot = {}
    for name in EXTRA:
        diff = (per_country[name] - per_country["base"]).to_numpy()
        draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        boot[name] = {"mean_change_years": round(float(diff.mean()), 4),
                      "ci95": [round(float(lo), 4), round(float(hi), 4)],
                      "share_of_draws_below_zero": round(float((draws < 0).mean()), 3),
                      "countries_improved": int((diff < -0.05).sum()),
                      "countries_worsened": int((diff > 0.05).sum())}

    # Where did +all change the most?
    change_all = (per_country["+all"] - per_country["base"]).sort_values()
    biggest = {"most_improved": {k: round(float(v), 2) for k, v in change_all.head(8).items()},
               "most_worsened": {k: round(float(v), 2) for k, v in change_all.tail(8)[::-1].items()}}

    # How much signal is in each outside input? Permutation importance inside the +all model, out of fold.
    cols_all = config.predictor_columns + EXTRA["+all"]
    x_all = df[cols_all].values
    imp = {c: [] for c in EXTRA["+all"]}
    for tr, te in folds:
        m = pipeline(config, cols_all).fit(x_all[tr], y[tr])
        base_mae = mean_absolute_error(y[te], m.predict(x_all[te]))
        for c in EXTRA["+all"]:
            j = cols_all.index(c)
            xs = x_all[te].copy()
            xs[:, j] = rng.permutation(xs[:, j])
            imp[c].append(mean_absolute_error(y[te], m.predict(xs)) - base_mae)
    importance = {c: round(float(np.mean(v)), 4) for c, v in imp.items()}

    results = {
        "method": "GroupKFold(5) by Country, same folds for every variant; leak-free clipping on the 13 WHO inputs, "
                  "no clipping on outside inputs; RandomForest(200, seed 42)",
        "sources": {"conflict": "UCDP Battle-Related Deaths Dataset v26.1, conflict level",
                    "epidemic": "EM-DAT custom request 2026-11-24, Disaster Type = Epidemic, 2000-2015",
                    "reporting": "computed from the WHO file: share of inputs filled in, year-to-year volatility"},
        "variants": summary,
        "hardest_18_of_base": hardest,
        "paired_bootstrap_vs_base": boot,
        "biggest_changes_with_all": biggest,
        "outside_input_importance_in_all": importance,
        "per_country": {name: {k: round(float(v), 3) for k, v in s.items()} for name, s in per_country.items()},
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nHardest 18 (base): miss " + ", ".join(f"{n} {summary[n]['mae_hardest_18_of_base']:.2f}" for n in variants))
    for name, b in boot.items():
        print(f"{name:11s} change vs base {b['mean_change_years']:+.3f} yrs, 95% CI {b['ci95']}, "
              f"improved {b['countries_improved']}, worsened {b['countries_worsened']}")
    print("outside-input importance (+all):", importance)
    print("most improved:", biggest["most_improved"])
    print("most worsened:", biggest["most_worsened"])
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
