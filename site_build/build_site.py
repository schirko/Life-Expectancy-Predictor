"""Build the static project page for the Life Expectancy Predictor.

Run from the project folder:   python -m site_build.build_site

What it does (same recipe as the meat-prices project, decision D36 there):
  1. Redraws every chart as a PNG from the saved model results (model/*.json)
     and the cleaned data (data/clean_data.csv).
  2. Fills the numbers into site_build/site_template.html -> index.html.
  3. Exports the notebook in notebooks/ to HTML (using the outputs saved in it).
  4. Turns docs/DOCUMENTATION.md into an HTML page.
  Every page gets a "back" bar and a noindex tag.

Where it writes: the folder named by SITE_DIR in .env (Scott's PC: the Schirko
portfolio folder, served at schirko.com/scott/life-expectancy/). With no SITE_DIR
it writes to a git-ignored site/ folder inside the project.

This page is the project's public face; the project has no app (Streamlit was
removed on 2026-11-15, milestone 17).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import date
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mistune  # noqa: E402
import numpy as np  # noqa: E402
import nbformat  # noqa: E402
import pandas as pd  # noqa: E402
from nbconvert import HTMLExporter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/schirko/Life-Expectancy-Predictor"
NOINDEX = '<meta name="robots" content="noindex, nofollow">'


def _read_dotenv() -> None:
    """Load KEY=VALUE lines from .env (no extra package needed)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_read_dotenv()
SITE_DIR = Path(os.getenv("SITE_DIR", "").strip() or ROOT / "site")


NOTEBOOKS = {
    "Life_Expectancy_Claude.ipynb": (
        "The original practicum notebook",
        "Cleaning about 2,900 WHO records, exploring them, and comparing the first regression models. "
        "This is where the project started.",
    ),
}
DOC_PAGES = {"DOCUMENTATION.md": "Project documentation"}

# Shared "back to portfolio" bar. The portfolio owns /scott/backbar.css; link to it,
# never copy or restyle the .sx- classes. Absolute /scott/ paths work at any depth.
BACKBAR_CSS = (
    '<link rel="stylesheet" href="/scott/backbar.css">\n'
    '<link rel="icon" href="/scott/favicon.ico" sizes="32x32">\n'
    '<link rel="icon" type="image/png" sizes="16x16" href="/scott/favicon-16.png">\n'
    '<link rel="icon" type="image/png" sizes="32x32" href="/scott/favicon-32.png">\n'
    '<link rel="apple-touch-icon" href="/scott/apple-touch-icon.png">'
)  # stylesheet and favicons are owned by the portfolio
BACK_BAR = (
    '<div class="sx-backbar">'
    '<a class="sx-back" href="/scott/">'
    '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg>'
    " Portfolio</a>"
    '<span class="sx-crumb">Life Expectancy Predictor</span>'
    '<span class="sx-links"><a href="/scott/life-expectancy/">Overview</a>'
    '<a href="/scott/life-expectancy/notebooks/Life_Expectancy_Claude.html">Notebook</a>'
    '<a href="/scott/life-expectancy/docs/DOCUMENTATION.html">Docs</a></span>'
    "</div>"
)

DOC_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{noindex}
{backbar_css}
<title>{title} - Life Expectancy Predictor</title>
<style>
:root{{--bg:#f6f5f1;--ink:#1c211e;--muted:#4a524d;--line:#e2e0d8;--accent:#2E6A4C;--code:#ecebe4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151a17;--ink:#eceae4;--muted:#b0b8b2;--line:#34403a;--accent:#6fc79a;--code:#222b26}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.65 Inter,system-ui,sans-serif}}
a{{color:var(--accent)}}
main{{max-width:860px;margin:0 auto;padding:12px 20px 60px}}
table{{border-collapse:collapse;display:block;overflow-x:auto;margin:16px 0}}
th,td{{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top;font-size:15px}}
th{{background:var(--code)}}
code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.92em}}
pre{{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto}}pre code{{padding:0}}
</style></head><body>
{back}
<main>{body}</main></body></html>"""


# ---------------------------------------------------------------- charts
INK, MUTED, GREEN, WARM, BLUE = "#1c211e", "#5f6762", "#2E6A4C", "#B26B16", "#2F6690"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def _save(fig, name: str) -> None:
    fig.savefig(SITE_DIR / "images" / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def showdown(diag: dict) -> list:
    """(model name, R² on unseen countries), best first, from diagnostics.json."""
    rows = diag["model_showdown"]["models"]
    return sorted(((r["model"], r["r2_mean"]) for r in rows), key=lambda t: -t[1])


def make_charts(diag: dict) -> None:
    (SITE_DIR / "images").mkdir(parents=True, exist_ok=True)
    g = diag["group_cv"]

    # 1. Two honest scores: random split vs unseen countries (per fold)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    folds = range(1, 6)
    ax.plot(folds, g["random"]["fold_r2"], "o-", color=WARM, lw=2.2, label="Random split (country's other years leak in)")
    ax.plot(folds, g["group"]["fold_r2"], "o-", color=GREEN, lw=2.2, label="Unseen countries (honest)")
    ax.set_ylim(0.84, 1.0)
    ax.set_xticks(list(folds))
    ax.set_xlabel("Cross-validation fold")
    ax.set_ylabel("R² (1.0 = perfect)")
    ax.set_title("The same model, scored two ways", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="center right", labelcolor=INK)
    _style(ax)
    _save(fig, "two_scores.png")

    # 2. Actual vs predicted (held-out test rows)
    r = diag["residuals"]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(r["y_test"], r["y_pred"], s=14, alpha=.45, color=GREEN, edgecolors="none")
    lo, hi = min(r["y_test"]), max(r["y_test"])
    ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1.2)
    ax.set_xlabel("Actual life expectancy (years)")
    ax.set_ylabel("Predicted (years)")
    ax.set_title("Predicted vs actual, 586 held-out rows", loc="left", fontweight="bold")
    _style(ax)
    _save(fig, "actual_vs_predicted.png")

    # 3. Permutation importance
    pi = sorted(diag["permutation_importance"], key=lambda d: d["importance_mean"])[-8:]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([p["feature"].replace("_", " ") for p in pi], [p["importance_mean"] for p in pi],
            xerr=[p["importance_std"] for p in pi], color=GREEN, ecolor="#999")
    ax.set_xlabel("Drop in R² when the input is scrambled")
    ax.set_title("What the model leans on (permutation importance)", loc="left", fontweight="bold")
    _style(ax)
    _save(fig, "importance.png")

    # 4. Hardest countries
    hard = sorted(g["per_group_error"], key=lambda d: -d["mean_abs_error"])[:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([h["group"] for h in hard], [h["mean_abs_error"] for h in hard], color=WARM)
    ax.axvline(g["median_group_error"], color=MUTED, ls="--", lw=1.2, label=f"Median country: {g['median_group_error']:.1f} yrs")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.set_xlabel("Average miss when the country is held out (years)")
    ax.set_title("The ten hardest countries to predict", loc="left", fontweight="bold")
    _style(ax)
    _save(fig, "hardest.png")

    # 5. Model showdown
    sd = showdown(diag)
    names = [n for n, _ in sd][::-1]
    vals = [v for _, v in sd][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(names, vals, color=[GREEN if "deployed" in n else "#9fb8aa" for n in names])
    for i, v in enumerate(vals):
        ax.text(v + .004, i, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0.7, 0.96)
    ax.set_xlabel("R² on unseen countries")
    ax.set_title("Four algorithms, same unseen countries", loc="left", fontweight="bold")
    _style(ax)
    _save(fig, "showdown.png")


def make_fairness_chart(fair: dict) -> None:
    """Average miss for developed vs developing countries, overall and inside
    the same life-expectancy bands (corrected Status labels)."""
    r = fair["corrected_labels"]
    labels, dev, dvg = ["All rows"], [r["groups"]["Developed"]["mae"]], [r["groups"]["Developing"]["mae"]]
    for b in r["by_life_expectancy_band"]:
        if b.get("Developed", {}).get("n_countries", 0) >= 5 and b.get("Developing", {}).get("n_countries", 0) >= 5:
            labels.append(f"Life exp. {b['band']}")
            dev.append(b["Developed"]["mae"])
            dvg.append(b["Developing"]["mae"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, dev, 0.4, color=BLUE, label="Developed")
    ax.bar(x + 0.2, dvg, 0.4, color=WARM, label="Developing")
    for i, (a, b) in enumerate(zip(dev, dvg)):
        ax.text(i - 0.2, a + 0.04, f"{a:.2f}", ha="center", color=INK, fontsize=9)
        ax.text(i + 0.2, b + 0.04, f"{b:.2f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average miss, unseen countries (years)")
    ax.set_title("Developing countries are missed by more, even like for like", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    _save(fig, "fairness.png")


# ------------------------------------------------------------ notebooks/docs
def export_notebooks() -> str:
    out = SITE_DIR / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    exporter = HTMLExporter(template_name="lab")
    items = []
    for i, (fname, (title, blurb)) in enumerate(NOTEBOOKS.items(), 1):
        nb = nbformat.read(ROOT / "notebooks" / fname, as_version=4)
        html, _ = exporter.from_notebook_node(nb)
        html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + BACK_BAR, html, count=1)
        html = html.replace("<head>", "<head>" + NOINDEX + BACKBAR_CSS, 1)
        # Let the shared bar run edge to edge: move the notebook template's page margin onto its content
        html = html.replace("</head>", "<style>body.jp-Notebook{margin:0;padding:0}body.jp-Notebook>main{padding:10px 18px}</style></head>", 1)
        (out / (Path(fname).stem + ".html")).write_text(html, encoding="utf-8")
        items.append(
            f'  <li><a href="notebooks/{Path(fname).stem}.html"><span class="num">{i}</span>'
            f'<span class="nbtitle">{title}</span></a><span class="blurb">{blurb}</span></li>'
        )
    return "\n".join(items)


def export_docs() -> None:
    out = SITE_DIR / "docs"
    out.mkdir(parents=True, exist_ok=True)
    md = mistune.create_markdown(plugins=["table", "strikethrough"])
    for fname, title in DOC_PAGES.items():
        body = md((ROOT / "docs" / fname).read_text(encoding="utf-8"))
        body = re.sub(r'href="([^"#]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
        page = DOC_PAGE.format(noindex=NOINDEX, backbar_css=BACKBAR_CSS, back=BACK_BAR, title=title, body=body)
        (out / (Path(fname).stem + ".html")).write_text(page, encoding="utf-8")


# ------------------------------------------------------------------- main
def main() -> None:
    meta = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    diag = json.loads((ROOT / "model" / "diagnostics.json").read_text(encoding="utf-8"))
    clean = pd.read_csv(ROOT / "data" / "clean_data.csv", usecols=["Year"])
    tuning = json.loads((ROOT / "model" / "tuning_results.json").read_text(encoding="utf-8"))
    leak = json.loads((ROOT / "model" / "leak_check.json").read_text(encoding="utf-8"))
    fair = json.loads((ROOT / "model" / "fairness_check.json").read_text(encoding="utf-8"))
    fc = fair["corrected_labels"]
    cal = json.loads((ROOT / "model" / "interval_calibration.json").read_text(encoding="utf-8"))
    cal_c = cal["evaluation"]["C_per_group"]
    g = diag["group_cv"]
    sd = dict(showdown(diag))
    conc = g["hard_cases"]["concentration"]

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    make_charts(diag)
    make_fairness_chart(fair)

    today = date.today()
    values = {
        "years": f"{int(clean['Year'].min())} to {int(clean['Year'].max())}",
        "n_rows": f"{meta['n_rows']:,}",
        "n_countries": str(g["n_groups"]),
        "r2_random": f"{meta['test_r2']:.2f}",
        "r2_group": f"{g['group']['r2_mean']:.2f}",
        "miss_random": f"{g['random']['oof_mae']:.2f}",
        "miss_group": f"{g['group']['oof_mae']:.2f}",
        "cover_random": f"{diag['calibration']['empirical_coverage_pct']:.0f}",
        "cover_group": f"{g['interval_unseen']['coverage_pct']:.0f}",
        "hiv_share": f"{meta['feature_importances']['HIV_AIDS'] * 100:.0f}",
        "top_country": g["per_group_error"][0]["group"],
        "top_country_miss": f"{g['per_group_error'][0]['mean_abs_error']:.1f}",
        "top10_share": f"{conc['share_of_total_error_pct']:.0f}",
        "tuned_r2": f"{tuning['best_test_r2']:.3f}",
        "default_r2": f"{tuning['default_test_r2']:.3f}",
        "sd_rf": f"{sd['Random Forest (deployed)']:.3f}",
        "sd_gb": f"{sd['Gradient Boosting']:.3f}",
        "sd_hgb": f"{sd['Hist Gradient Boosting']:.3f}",
        "sd_lin": f"{sd['Linear Regression']:.3f}",
        "leak_old_group": f"{leak['A_old']['group_cv_r2']:.3f}",
        "leak_new_group": f"{leak['D_new']['group_cv_r2']:.3f}",
        "leak_rows": str(leak["target_rows_changed_by_old_clipping"]),
        "fair_dev_mae": f"{fc['groups']['Developed']['mae']:.2f}",
        "fair_dvg_mae": f"{fc['groups']['Developing']['mae']:.2f}",
        "fair_gap": f"{fc['mae_gap_years']:.1f}",
        "fair_ci_lo": f"{fc['mae_gap_95ci'][0]:.1f}",
        "fair_ci_hi": f"{fc['mae_gap_95ci'][1]:.1f}",
        "fair_cov_dev": f"{fc['groups']['Developed']['interval_coverage_pct']:.0f}",
        "fair_cov_dvg": f"{fc['groups']['Developing']['interval_coverage_pct']:.0f}",
        "cal_factor_dvg": f"{(cal['factors_fitted_on_all']['Developing'] - 1) * 100:.0f}",
        "cal_cov_dev": f"{cal_c['Developed']['coverage_pct']:.0f}",
        "cal_cov_dvg": f"{cal_c['Developing']['coverage_pct']:.1f}",
        "github": GITHUB,
        "notebooks": export_notebooks(),
        "backbar_css": BACKBAR_CSS,
        "back": BACK_BAR,
        "built": f"{today.strftime('%B')} {today.day}, {today.year}",
    }
    template = Template((ROOT / "site_build" / "site_template.html").read_text(encoding="utf-8"))
    (SITE_DIR / "index.html").write_text(template.substitute(values), encoding="utf-8")
    export_docs()
    # Safety check: every generated page must carry the shared back bar.
    pages = [SITE_DIR / "index.html", *(SITE_DIR / "notebooks").glob("*.html"), *(SITE_DIR / "docs").glob("*.html")]
    missing = [str(f) for f in pages if "sx-backbar" not in f.read_text(encoding="utf-8")
               or "/scott/backbar.css" not in f.read_text(encoding="utf-8")
               or "/scott/favicon-32.png" not in f.read_text(encoding="utf-8")]
    if missing:
        raise SystemExit("Back bar or favicons missing from: " + ", ".join(missing))
    print(f"Site written to {SITE_DIR} ({len(pages)} pages, back bar and favicons verified)")


if __name__ == "__main__":
    main()
