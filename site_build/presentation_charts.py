"""Draw the presentation's charts (milestone 24). Run after retraining:

    python -m site_build.presentation_charts

Writes the charts to docs/presentation_charts/ from the same saved results the
project page and notebooks use (model/*.json, data/clean_data.csv,
notebooks/oof_predictions.csv), in the deck's palette, sized for 16:9 slides.
The deck ("Can you predict how long people live?", a Claude Slides deck) uses
these images; after a retrain, re-run this, update the deck's images and
numbers, and re-export the deck into presentation/ (see presentation/README.md).

Charts: two_scores (random vs unseen-country test), hidden_zeros (health
spending zeros by year), drivers (permutation importance), hiv_curve (partial
dependence for HIV/AIDS, computed as in notebook 4), hardest (worst countries), shrinkage (bias by truth
vs by prediction), band (interval coverage before/after), showdown (four
algorithms), cover_bg (title backdrop: true vs predicted, dark).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_loader import load_config  # noqa: E402

OUT = ROOT / "docs" / "presentation_charts"
INK, PAPER, MUTED = "#1F1B18", "#FAF8F5", "#5F5852"
GREEN, BLUE, RUST, GOLD = "#2E6A4C", "#2F6690", "#9A3B2E", "#B26B16"
GRID_L, GRID_D, LIGHT, DARK_BG = "#E3DAD1", "#3D3631", "#D9D1C9", "#1F1B18"


def results(name):
    return json.loads((ROOT / "model" / f"{name}.json").read_text(encoding="utf-8"))


def style(ax, dark=False, ts=16):
    ax.set_facecolor(DARK_BG if dark else PAPER)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID_D if dark else GRID_L)
        ax.spines[s].set_linewidth(1.5)
    ax.tick_params(colors=LIGHT if dark else MUTED, labelsize=ts, length=0, pad=8)
    ax.grid(color=GRID_D if dark else GRID_L, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.title.set_color(LIGHT if dark else INK)
    ax.xaxis.label.set_color(LIGHT if dark else MUTED)
    ax.yaxis.label.set_color(LIGHT if dark else MUTED)


def save(fig, name, dark=False):
    fig.savefig(OUT / name, dpi=200, facecolor=DARK_BG if dark else PAPER, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def two_scores():
    meta, gcv = results("metadata"), results("diagnostics")["group_cv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, vals, title, unit in [
        (axes[0], [meta["test_r2"], gcv["group"]["r2_mean"]], "R² (1.0 = perfect)", ""),
        (axes[1], [meta["test_mae"], gcv["group"]["oof_mae"]], "Average miss", " yrs"),
    ]:
        style(ax)
        bars = ax.bar(["Random rows\nheld out", "Whole countries\nheld out"], vals, color=[GOLD, GREEN], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}{unit}", ha="center", va="bottom", fontsize=20, color=INK, fontweight="bold")
        ax.set_title(title, fontsize=20, pad=14)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_xticks(range(2)); ax.set_xticklabels(["Random rows\nheld out", "Whole countries\nheld out"], fontsize=16)
    fig.tight_layout(w_pad=3)
    save(fig, "two_scores.png")


def hidden_zeros():
    config = load_config(str(ROOT / "config" / "who_life_expectancy.yaml"), base_dir=str(ROOT))
    raw = pd.read_csv(config.source_csv)
    col = [c for c in raw.columns if c.strip().lower().startswith("percentage expenditure")][0]
    zeros = (raw[col] == 0).groupby(raw["Year"]).sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    style(ax)
    ax.bar(zeros.index, zeros.values, color=RUST, width=0.7)
    ax.set_title("Countries reporting exactly zero health spending, by year", fontsize=20, pad=14)
    ax.set_ylabel("Countries (of 183)", fontsize=15)
    ax.annotate("2015: 181 of 183", xy=(2015, zeros.loc[2015]), xytext=(2009.5, 150), fontsize=17, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xticks(range(2000, 2016, 3))
    fig.tight_layout()
    save(fig, "hidden_zeros.png")


def drivers():
    config = load_config(str(ROOT / "config" / "who_life_expectancy.yaml"), base_dir=str(ROOT))
    labels = {p.column: p.label.split(" (")[0] for p in config.predictors}
    pi = sorted(results("diagnostics")["permutation_importance"], key=lambda r: r["importance_mean"])
    fig, ax = plt.subplots(figsize=(11, 6))
    style(ax)
    names = [labels.get(r["feature"], r["feature"]) for r in pi]
    vals = [r["importance_mean"] for r in pi]
    colors = [GREEN if v > 0.1 else GRID_L for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("Drop in R² when the input is scrambled", fontsize=15)
    ax.set_title("Three inputs do almost all the work", fontsize=20, pad=14)
    ax.tick_params(axis="y", labelsize=14)
    fig.tight_layout()
    save(fig, "drivers.png")


def hiv_curve():
    """Partial dependence computed by hand, exactly as notebook 4 does it (the saved
    sklearn curve runs far past where the data is)."""
    from data_prep import get_unclipped_df
    from train_model import rf_pipeline
    from sklearn.model_selection import train_test_split

    config = load_config(str(ROOT / "config" / "who_life_expectancy.yaml"), base_dir=str(ROOT))
    df = get_unclipped_df(config)
    X = df[config.predictor_columns].to_numpy(); y = df[config.target_column].to_numpy()
    mp = config.model_params
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=mp["test_size"], random_state=mp["random_state"])
    model = rf_pipeline(config, n_estimators=mp["n_estimators"], random_state=mp["random_state"], n_jobs=-1).fit(X_tr, y_tr)
    j = config.predictor_columns.index("HIV_AIDS")
    lo, hi = np.percentile(X_te[:, j], [2, 90])
    grid = np.linspace(lo, hi, 40)
    avg = []
    for v in grid:
        X_mod = X_te.copy(); X_mod[:, j] = v
        avg.append(model.predict(X_mod).mean())
    fig, ax = plt.subplots(figsize=(11, 5))
    style(ax)
    ax.plot(grid, avg, color=BLUE, lw=4)
    ax.set_xlabel("HIV/AIDS deaths per 1,000 (ages 0-4), middle 88% of the data", fontsize=15)
    ax.set_ylabel("Predicted life expectancy (years)", fontsize=15)
    ax.set_title(f"Any serious HIV burden costs about {avg[0] - avg[-1]:.0f} years, most of it at once", fontsize=20, pad=14)
    fig.tight_layout()
    save(fig, "hiv_curve.png")


def hardest():
    per = results("diagnostics")["group_cv"]["per_group_error"][:10]
    oof = pd.read_csv(ROOT / "notebooks" / "oof_predictions.csv")
    signed = (oof["oof_prediction"] - oof["truth"]).groupby(oof["Country"]).mean()
    names = [r["group"] for r in per][::-1]
    vals = [r["mean_abs_error"] for r in per][::-1]
    colors = [RUST if signed[n] > 0 else BLUE for n in names]
    fig, ax = plt.subplots(figsize=(11, 6))
    style(ax)
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=14, color=INK)
    ax.set_xlabel("Average miss over 16 years (years)", fontsize=15)
    ax.set_title("The ten hardest countries   (rust: predicted too high · blue: too low)", fontsize=18, pad=14)
    ax.tick_params(axis="y", labelsize=14)
    fig.tight_layout()
    save(fig, "hardest.png")


def shrinkage():
    shr = results("shrinkage_check")["decile_bias_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, key, title in [(axes[0], "by_true_value", "Sliced by the TRUE value"), (axes[1], "by_prediction", "Sliced by the PREDICTION")]:
        style(ax)
        v = -np.array(shr[key])  # file stores truth minus prediction; show prediction minus truth
        ax.bar(range(10), v, color=[RUST if a > 0 else BLUE for a in v])
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks([0, 9]); ax.set_xticklabels(["shortest\nlives", "longest\nlives"], fontsize=14)
        ax.set_title(title, fontsize=18, pad=12)
    axes[0].set_ylabel("Average miss (years)\nabove 0 = predicted too high", fontsize=14)
    fig.suptitle("The pattern is there by truth, gone by prediction", fontsize=20, color=INK, y=1.02)
    fig.tight_layout(w_pad=3)
    save(fig, "shrinkage.png")


def band():
    ev = results("interval_calibration")["evaluation"]
    fig, ax = plt.subplots(figsize=(11, 5))
    style(ax)
    groups = ["Developed", "Developing"]
    x = np.arange(2)
    before = [ev["A_raw"][g]["coverage_pct"] for g in groups]
    after = [ev["C_per_group"][g]["coverage_pct"] for g in groups]
    ax.bar(x - 0.18, before, 0.34, color=GRID_L, label="Before")
    ax.bar(x + 0.18, after, 0.34, color=GREEN, label="After resizing per group")
    for xi, b, a in zip(x, before, after):
        ax.text(xi - 0.18, b + 0.5, f"{b:.0f}%", ha="center", fontsize=16, color=MUTED)
        ax.text(xi + 0.18, a + 0.5, f"{a:.0f}%", ha="center", fontsize=16, color=INK, fontweight="bold")
    ax.axhline(80, color=RUST, lw=2, ls="--")
    ax.text(-0.42, 80.4, "promised: 80%", color=RUST, fontsize=14, ha="left", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=16)
    ax.set_ylim(70, 86)
    ax.set_ylabel("How often the truth is inside the band", fontsize=14)
    ax.set_title("The uncertainty band was overconfident for developing countries", fontsize=18, pad=14)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    fig.tight_layout()
    save(fig, "band.png")


def showdown():
    models = results("diagnostics")["model_showdown"]["models"]
    names = [m["model"].replace(" (deployed)", "\n(deployed)") for m in models]
    vals = [m["r2_mean"] for m in models]
    fig, ax = plt.subplots(figsize=(11, 5))
    style(ax)
    bars = ax.bar(names, vals, color=[GREEN, BLUE, BLUE, GRID_L], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=17, color=INK, fontweight="bold")
    ax.set_ylim(0.78, 0.94)
    ax.set_ylabel("R² on unseen countries", fontsize=14)
    ax.set_title("Four algorithms, same honest test: the tree models tie", fontsize=18, pad=14)
    ax.tick_params(axis="x", labelsize=14)
    fig.tight_layout()
    save(fig, "showdown.png")


def cover_bg():
    oof = pd.read_csv(ROOT / "notebooks" / "oof_predictions.csv")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.scatter(oof["truth"], oof["oof_prediction"], s=14, alpha=0.35, color="#C9553F", linewidths=0)
    ax.plot([35, 90], [35, 90], color="#C9553F", lw=1.5, alpha=0.6)
    ax.axis("off")
    fig.savefig(OUT / "cover_bg.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    print("  wrote cover_bg.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Drawing presentation charts into {OUT}")
    for f in [two_scores, hidden_zeros, drivers, hiv_curve, hardest, shrinkage, band, showdown, cover_bg]:
        f()


if __name__ == "__main__":
    main()
