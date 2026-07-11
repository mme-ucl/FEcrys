"""Sphinx configuration for the FEcrys documentation."""

from __future__ import annotations

import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR))

from generate_api import generate  # noqa: E402


generate(REPOSITORY_ROOT / "O", DOCS_DIR / "api")

project = "FECrys"
author = "FECrys contributors"
copyright = "2026, FECrys contributors"
release = "development"

extensions = ["sphinx.ext.napoleon"]
templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "FECrys documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["figure-1-code-map.html"]
html_theme_options = {
    "description": "Free-energy calculations for molecular crystals",
    "fixed_sidebar": True,
    "github_user": "mme-ucl",
    "github_repo": "FEcrys",
    "github_button": True,
}
