"""
Build the outside-data table for milestone 25 (conflict, epidemics, reporting quality).

Run from the project folder:

    python -m analysis.external_data

Reads two files Scott downloaded into data/external/ and the WHO table itself, and
writes data/external/outside_data_by_country_year.csv with one row per WHO
country-year (2000-2015) and these columns:

  battle_deaths        UCDP battle-related deaths in conflicts located in the country
                       that year (best estimate; 0 when UCDP records no conflict)
  log_battle_deaths    log(1 + battle_deaths): the model input, so a 10,000-death
                       year is "much worse" than 100, not 100 times worse
  epidemic_events      EM-DAT epidemic events that started in the country that year
  epidemic_deaths      deaths in those events (0 when none recorded)
  log_epidemic_deaths  log(1 + epidemic_deaths)
  inputs_filled_pct    share of the country's 13 model inputs that were blank or
                       disguised blanks in the WHO file and had to be filled in,
                       over all 16 years (a per-country constant)
  input_volatility     how jumpy the country's inputs are year to year: the median,
                       over inputs, of the standard deviation of year-on-year
                       changes divided by the input's overall standard deviation
                       (a per-country constant)

Sources:
  UCDP Battle-Related Deaths Dataset, conflict level, version 26.1
    (data/external/BattleDeaths_v26_1_conf.csv), Uppsala Conflict Data Program.
    One row per conflict-year; `location_inc` names the country or countries the
    conflict is about (several, comma-separated, for a few conflicts). Deaths of
    a multi-country conflict are credited to every country named.
  EM-DAT, The International Disaster Database, CRED / UCLouvain, custom request
    2026-11-24: Disaster Type = Epidemic, 2000-2015
    (data/external/emdat_epidemics_2000_2015.xlsx).

Country names are mapped to the WHO spellings in NAME_MAP; anything that still
does not match is printed so it can be added. No name in the WHO table is dropped:
a country absent from UCDP or EM-DAT gets zeros, which is the honest reading
("no recorded conflict / epidemic"), and is noted in the documentation.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config  # noqa: E402
from data_prep import _clean_column_names, _fix_implicit_nulls  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")
UCDP_FILE = os.path.join(EXT, "BattleDeaths_v26_1_conf.csv")
EMDAT_FILE = os.path.join(EXT, "emdat_epidemics_2000_2015.xlsx")
OUT_FILE = os.path.join(EXT, "outside_data_by_country_year.csv")
YEARS = range(2000, 2016)

# Source spelling -> WHO spelling (as in data/clean_data.csv).
NAME_MAP = {
    "Cambodia (Kampuchea)": "Cambodia",
    "DR Congo (Zaire)": "Democratic Republic of the Congo",
    "Iran": "Iran (Islamic Republic of)",
    "Ivory Coast": "Côte d'Ivoire",
    "Côte d’Ivoire": "Côte d'Ivoire",
    "Myanmar (Burma)": "Myanmar",
    "North Macedonia": "The former Yugoslav republic of Macedonia",
    "Russia (Soviet Union)": "Russian Federation",
    "Syria": "Syrian Arab Republic",
    "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "Yemen (North Yemen)": "Yemen",
    "Eswatini": "Swaziland",
    "Türkiye": "Turkey",
    "Serbia Montenegro": "Serbia",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Laos": "Lao People's Democratic Republic",
    "Moldova": "Republic of Moldova",
    "Tanzania": "United Republic of Tanzania",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Vietnam": "Viet Nam",
    "South Korea": "Republic of Korea",
    "North Korea": "Democratic People's Republic of Korea",
}


def who_countries(config):
    raw = _clean_column_names(pd.read_csv(config.source_csv), config)
    return sorted(raw[raw[config.target_column].notnull()][config.group_column].unique())


def conflict_table(countries):
    u = pd.read_csv(UCDP_FILE)
    u = u[(u["year"] >= min(YEARS)) & (u["year"] <= max(YEARS))]
    rows = []
    for _, r in u.iterrows():
        for name in str(r["location_inc"]).split(","):
            name = NAME_MAP.get(name.strip(), name.strip())
            rows.append((name, int(r["year"]), int(r["bd_best"])))
    t = pd.DataFrame(rows, columns=["Country", "Year", "battle_deaths"])
    unmatched = sorted(set(t["Country"]) - set(countries))
    if unmatched:
        print("UCDP names not in the WHO table (ignored):", unmatched)
    return t[t["Country"].isin(countries)].groupby(["Country", "Year"], as_index=False)["battle_deaths"].sum()


def epidemic_table(countries):
    e = pd.read_excel(EMDAT_FILE)
    e = e[e["Disaster Type"] == "Epidemic"]
    e = e[(e["Start Year"] >= min(YEARS)) & (e["Start Year"] <= max(YEARS))]
    t = pd.DataFrame({
        "Country": e["Country"].map(lambda n: NAME_MAP.get(n, n)),
        "Year": e["Start Year"].astype(int),
        "epidemic_deaths": e["Total Deaths"].fillna(0).astype(int),
    })
    unmatched = sorted(set(t["Country"]) - set(countries))
    if unmatched:
        print("EM-DAT names not in the WHO table (ignored):", unmatched)
    t = t[t["Country"].isin(countries)]
    return t.groupby(["Country", "Year"]).agg(epidemic_events=("epidemic_deaths", "size"),
                                              epidemic_deaths=("epidemic_deaths", "sum")).reset_index()


def reporting_quality(config):
    """Two per-country signals from the WHO file itself, before any filling."""
    raw = _fix_implicit_nulls(_clean_column_names(pd.read_csv(config.source_csv), config), config)
    raw = raw[raw[config.target_column].notnull()]
    cols = [c for c in config.predictor_columns if c in raw.columns]
    filled = raw.groupby(config.group_column)[cols].apply(lambda t: t.isna().values.mean() * 100)
    overall_sd = raw[cols].std()

    def volatility(t):
        t = t.sort_values("Year")[cols]
        ratios = (t.diff().std() / overall_sd).replace([np.inf, -np.inf], np.nan)
        return float(ratios.median())

    vol = raw.groupby(config.group_column).apply(volatility)
    return pd.DataFrame({"inputs_filled_pct": filled.round(1), "input_volatility": vol.round(3)})


def main():
    config = load_config(os.path.join(ROOT, "config", "who_life_expectancy.yaml"), base_dir=ROOT)
    countries = who_countries(config)
    grid = pd.MultiIndex.from_product([countries, list(YEARS)], names=["Country", "Year"]).to_frame(index=False)
    out = (grid.merge(conflict_table(countries), how="left", on=["Country", "Year"])
               .merge(epidemic_table(countries), how="left", on=["Country", "Year"]))
    for c in ["battle_deaths", "epidemic_events", "epidemic_deaths"]:
        out[c] = out[c].fillna(0).astype(int)
    out["log_battle_deaths"] = np.log1p(out["battle_deaths"]).round(4)
    out["log_epidemic_deaths"] = np.log1p(out["epidemic_deaths"]).round(4)
    out = out.merge(reporting_quality(config), how="left", left_on="Country", right_index=True)
    out.to_csv(OUT_FILE, index=False)

    n_conf = (out["battle_deaths"] > 0).groupby(out["Country"]).any().sum()
    n_epi = (out["epidemic_events"] > 0).groupby(out["Country"]).any().sum()
    print(f"Wrote {OUT_FILE}: {len(out):,} country-years, {len(countries)} countries")
    print(f"  countries with any battle deaths 2000-2015: {n_conf}; with any epidemic event: {n_epi}")
    print(f"  country-years with battle deaths: {(out['battle_deaths'] > 0).sum()}; "
          f"with an epidemic event: {(out['epidemic_events'] > 0).sum()}")
    top = out.groupby("Country")[["battle_deaths", "epidemic_deaths"]].sum()
    print("  most battle deaths:", top["battle_deaths"].nlargest(5).to_dict())
    print("  most epidemic deaths:", top["epidemic_deaths"].nlargest(5).to_dict())


if __name__ == "__main__":
    main()
