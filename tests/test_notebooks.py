"""The notebooks listed for the project page exist, carry saved outputs and have
no error outputs (milestone 23). The page build exports saved outputs and never
re-runs a notebook, so a notebook committed without outputs would publish empty."""
import json
from pathlib import Path

import pytest

from site_build.build_site import NOTEBOOKS


@pytest.mark.parametrize("fname", list(NOTEBOOKS))
def test_notebook_is_executed_and_clean(root, fname):
    path = Path(root) / "notebooks" / fname
    assert path.exists(), fname
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, fname
    executed = [c for c in code_cells if c.get("outputs")]
    assert len(executed) >= len(code_cells) // 2, f"{fname}: too few cells have saved outputs"
    errors = [o.get("ename") for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{fname}: error outputs {errors}"


def test_explanatory_notebooks_use_the_four_headings():
    """Every numbered notebook follows the reading pattern the documentation promises."""
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "notebooks").glob("0[1-7]_*.ipynb")):
        text = " ".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                        for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "markdown")
        for heading in ["What we are looking for", "How to read it", "What it means", "What's next and why"]:
            assert heading in text, f"{path.name} is missing '{heading}'"
