"""
Milestone 18: is the model fair to developing countries?

What we are looking for: on countries the model has never seen, developing
countries are missed by more years than developed ones. Is that because the
model treats them worse, or because their life expectancy is harder to
predict (wider range, more extreme values, thinner data)?

How: every row is predicted by a model that never saw that country
(GroupKFold(5) by Country, the same folds and leak-free clipping as
train_model.py). Then, for each group:
  1. average miss, bias (over- or under-prediction) and R² within the group;
  2. the same, comparing like with like: rows are put into life-expectancy
     bands and the groups are compared only inside the same band;
  3. the 80% tree-spread interval: how often it holds the true value, and
     how wide it is;
  4. a bootstrap over countries (resampling whole countries) to see whether
     the gap is bigger than chance.
All of it is run twice: with the source data's Status labels, and with six
clear mislabels corrected (advanced economies tagged "Developing").

Status is NOT a model input; it is only used here to group the results.

Run from the project folder:   python -m analysis.fairness_check
Saves model/fairness_check.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from config_loader import load_config
from data_prep import _clean_column_names, _fix_implicit_nulls, get_unclipped_df
from train_model import rf_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "who_life_expectancy.yaml")
OUT_PATH = os.path.join(BASE_DIR, "model", "fairness_check.json")

# Advanced economies that the source data labels "Developing" (IMF/OECD
# classification). Only these six are relabelled; borderline cases such as
# Qatar or Estonia are left as the source has them.
MISLABELLED_AS_DEVELOPING = ["Canada", "Finland", "France", "Greece", "Israel", "Republic of Korea"]

BANDS = [0, 60, 65, 70, 75, 80, 100]  # life-expectancy bands (years) for like-with-like comparison
N_BOOT = 2000


def out_of_fold(config, df):
    """Predictions and 80% tree-spread intervals, each from a model that never saw that country."""
    x = df[config.predictor_columns].values
    y = df[config.target_column].values
    groups = df[config.group_column].values
    mp = config.model_params
    pred = np.zeros(len(y))
    lo = np.zeros(len(y))
    hi = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        m = rf_pipeline(config, mp.get("n_estimators", 200), mp.get("random_state", 42), n_jobs=-1)
        m.fit(x[tr], y[tr])
        xt = m.named_steps["clip"].transform(x[te])
        trees = np.array([t.predict(xt) for t in m.named_steps["rf"].estimators_])
        pred[te] = trees.mean(axis=0)
        lo[te], hi[te] = np.percentile(trees, [10, 90], axis=0)
    return pred, lo, hi


def group_stats(sub):
    err = sub["actual"] - sub["pred"]
    return {
        "n_countries": int(sub["Country"].nunique()),
        "n_rows": int(len(sub)),
        "mean_life_expectancy": round(float(sub["actual"].mean()), 1),
        "spread_sd_years": round(float(sub["actual"].std()), 2),
        "mae": round(float(err.abs().mean()), 3),
        "bias": round(float(err.mean()), 3),  # + = model under-predicts
        "r2_within_group": round(float(r2_score(sub["actual"], sub["pred"])), 3),
        "interval_coverage_pct": round(float(sub["covered"].mean() * 100), 1),
        "interval_width_years": round(float((sub["hi"] - sub["lo"]).mean()), 2),
    }


def analyse(work, status_col, gaps_by_country):
    groups = {g: group_stats(s) for g, s in work.groupby(status_col)}
    # How much of each group's input data was missing in the source and had to be filled in?
    status_by_country = work.groupby("Country")[status_col].first()
    for g in groups:
        groups[g]["inputs_filled_in_pct"] = round(float(
            gaps_by_country[status_by_country[status_by_country == g].index].mean()), 1)

    # Like with like: compare the groups inside the same life-expectancy band.
    work = work.assign(band=pd.cut(work["actual"], BANDS, right=False))
    by_band = []
    for band, sub in work.groupby("band", observed=True):
        row = {"band": f"{band.left:.0f}-{band.right:.0f}"}
        for g, s in sub.groupby(status_col):
            row[g] = {"n_countries": int(s["Country"].nunique()), "n_rows": int(len(s)),
                      "mae": round(float((s["actual"] - s["pred"]).abs().mean()), 3)}
        by_band.append(row)

    # Developing-country error re-weighted to the developed life-expectancy mix:
    # "what would the developing MAE be if those countries had the same spread
    # of life expectancies as the developed ones?"
    dev = work[work[status_col] == "Developed"]
    dvg = work[work[status_col] == "Developing"]
    w = dev["band"].value_counts(normalize=True)
    band_mae = dvg.assign(ae=(dvg["actual"] - dvg["pred"]).abs()).groupby("band", observed=True)["ae"].mean()
    common = [b for b in w.index if b in band_mae.index and w[b] > 0]
    reweighted = float(sum(w[b] * band_mae[b] for b in common) / sum(w[b] for b in common))

    # Bootstrap over countries: is the developing-minus-developed gap real?
    rng = np.random.default_rng(42)
    ae_by_country = work.assign(ae=(work["actual"] - work["pred"]).abs()).groupby("Country").agg(
        ae=("ae", "sum"), n=("ae", "size"), status=(status_col, "first"))
    dev_c = ae_by_country[ae_by_country["status"] == "Developed"]
    dvg_c = ae_by_country[ae_by_country["status"] == "Developing"]
    gaps = []
    for _ in range(N_BOOT):
        a = dvg_c.sample(len(dvg_c), replace=True, random_state=rng.integers(1e9))
        b = dev_c.sample(len(dev_c), replace=True, random_state=rng.integers(1e9))
        gaps.append(a["ae"].sum() / a["n"].sum() - b["ae"].sum() / b["n"].sum())
    gaps = np.array(gaps)

    return {
        "groups": groups,
        "mae_gap_years": round(groups["Developing"]["mae"] - groups["Developed"]["mae"], 3),
        "mae_gap_95ci": [round(float(np.percentile(gaps, 2.5)), 3), round(float(np.percentile(gaps, 97.5)), 3)],
        "share_of_bootstrap_gaps_above_zero_pct": round(float((gaps > 0).mean() * 100), 1),
        "by_life_expectancy_band": by_band,
        "developing_mae_reweighted_to_developed_mix": round(reweighted, 3),
    }


def main():
    config = load_config(CONFIG_PATH, base_dir=BASE_DIR)
    df = get_unclipped_df(config).dropna(
        subset=config.predictor_columns + [config.target_column]).reset_index(drop=True)
    pred, lo, hi = out_of_fold(config, df)
    work = pd.DataFrame({
        "Country": df[config.group_column], "Status": df["Status"],
        "actual": df[config.target_column], "pred": pred, "lo": lo, "hi": hi,
    })
    work["covered"] = (work["actual"] >= work["lo"]) & (work["actual"] <= work["hi"])
    work["Status_corrected"] = np.where(work["Country"].isin(MISLABELLED_AS_DEVELOPING), "Developed", work["Status"])

    # Share of predictor values missing in the source (before interpolation), per country.
    raw = _fix_implicit_nulls(_clean_column_names(pd.read_csv(config.source_csv), config), config)
    raw = raw[raw[config.target_column].notnull()]
    present = [c for c in config.predictor_columns if c in raw.columns]
    gaps_by_country = raw.groupby(config.group_column)[present].apply(lambda t: t.isna().values.mean() * 100)

    results = {
        "method": "GroupKFold(5) by Country, leak-free clipping; Status is not a model input",
        "overall_mae": round(float(mean_absolute_error(work["actual"], work["pred"])), 3),
        "relabelled_countries": MISLABELLED_AS_DEVELOPING,
        "source_labels": analyse(work, "Status", gaps_by_country),
        "corrected_labels": analyse(work, "Status_corrected", gaps_by_country),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    for key in ("source_labels", "corrected_labels"):
        r = results[key]
        print(f"{key}: MAE gap {r['mae_gap_years']:+.2f} yrs (95% CI {r['mae_gap_95ci'][0]:+.2f} to {r['mae_gap_95ci'][1]:+.2f})")
        for g, st in r["groups"].items():
            print(f"  {g:<10} MAE {st['mae']:.2f}  bias {st['bias']:+.2f}  coverage {st['interval_coverage_pct']}%  "
                  f"filled-in inputs {st['inputs_filled_in_pct']}%")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
