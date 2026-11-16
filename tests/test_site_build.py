"""The project-page build writes only inside its own folder and produces
complete pages (milestones 15, 17 and the 2026-11-16 ownership rule)."""
import os
import re
import subprocess
import sys

import pytest


def run_build(root, site_dir):
    env = dict(os.environ, SITE_DIR=str(site_dir))
    return subprocess.run([sys.executable, "-m", "site_build.build_site"], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)


def test_refuses_to_write_the_portfolio_root(root, tmp_path):
    result = run_build(root, tmp_path / "scott")
    assert result.returncode != 0
    assert "portfolio root" in (result.stdout + result.stderr)


def test_builds_all_pages_with_back_bar_and_noindex(root, tmp_path):
    pytest.importorskip("nbconvert")
    pytest.importorskip("mistune")
    pytest.importorskip("matplotlib")
    site = tmp_path / "life-expectancy"
    result = run_build(root, site)
    assert result.returncode == 0, result.stderr
    pages = [site / "index.html", site / "docs" / "FINDINGS.html", site / "docs" / "DOCUMENTATION.html",
             site / "notebooks" / "Life_Expectancy_Claude.html"]
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert "sx-backbar" in html and "/scott/backbar.css" in html, page
        assert 'content="noindex, nofollow"' in html, page
    index = pages[0].read_text(encoding="utf-8")
    assert "${" not in re.sub(r"<!--.*?-->", "", index, flags=re.S)  # every placeholder was filled
    assert "streamlit" not in index.lower()
    for img in ["two_scores.png", "importance.png", "hardest.png", "showdown.png", "fairness.png", "shrinkage.png"]:
        assert (site / "images" / img).exists(), img
