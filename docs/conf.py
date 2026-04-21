"""Sphinx configuration for devinterp documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "devinterp"
copyright = "2024-2026, Timaeus"
author = "Timaeus"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Theme -------------------------------------------------------------------

html_theme = "furo"
html_title = "DevInterp"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82B1FF",
        "color-brand-content": "#82B1FF",
    },
}

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"

# Don't show inherited Pydantic/torch methods
autodoc_inherited_members = False

html_static_path = ["_static"]
html_css_files = ["custom.css"]
