"""Saved results are where they should be, and the headline numbers in the
README, findings and documentation match them. A failure here means the docs
need updating after a re-run."""
import json
import os

import pytest

RESULT_FILES = ["metadata.json", "diagnostics.json", "tuning_results.json", "leak_check.json",
                "fairness_check.json", "interval_calibration.json", "shrinkage_check.json"]


def load(root, name):
    with open(os.path.join(root, "model", name), encoding="utf-8") as f:
        return json.load(f)


def read(root, *parts):
    with open(os.path.join(root, *parts), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("name", RESULT_FILES)
def test_every_analysis_has_saved_results(root, name):
    assert os.path.exists(os.path.join(root, "model", name)), f"run the script that writes model/{name}"


def test_scores_are_in_the_expected_range(root):
    meta = load(root, "metadata.json")
    assert 0.95 < meta["test_r2"] < 0.97          # random split
    assert 0.88 < meta["cv_r2_mean"] < 0.92       # countries held out


def test_readme_headline_matches_results(root):
    diag = load(root, "diagnostics.json")
    meta = load(root, "metadata.json")
    readme = read(root, "README.md")
    assert f"average miss {diag['group_cv']['group']['oof_mae']:.2f} years" in readme
    assert f"average miss {meta['test_mae']:.2f} years" in readme


def test_findings_headline_matches_results(root):
    diag = load(root, "diagnostics.json")
    findings = read(root, "docs", "FINDINGS.md")
    assert f"**{diag['group_cv']['group']['oof_mae']:.1f} years on countries it has never seen**" in findings
    assert f"| Countries held out entirely (5 folds) | {diag['group_cv']['group']['r2_mean']:.3f}" in findings


def test_documentation_tables_match_results(root):
    doc = read(root, "docs", "DOCUMENTATION.md")
    fair = load(root, "fairness_check.json")["corrected_labels"]["groups"]
    dev = fair["Developed"]
    assert f"| Corrected | Developed | {dev['n_countries']} | {dev['mae']:.2f} |" in doc
    shr = load(root, "shrinkage_check.json")["A_raw"]
    assert f"| A. No correction | {shr['mae']:.3f} |" in doc
    leak = load(root, "leak_check.json")["D_new"]
    assert f"| {leak['group_cv_r2']:.4f} | {leak['group_oof_mae']:.3f} |" in doc
