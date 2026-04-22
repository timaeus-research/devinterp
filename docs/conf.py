"""Sphinx configuration for devinterp documentation."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "devinterp"
copyright = "2024-2026, Timaeus"
author = "Timaeus"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx_math_dollar",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosectionlabel_prefix_document = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight

.. role:: bash(code)
    :language: bash
    :class: highlight
"""

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autosummary_generate = True


def run_apidoc(_):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    module_dir = os.path.join(current_dir, "..", "src", "devinterp")
    output_dir = os.path.join(current_dir, "source")
    subprocess.call(["sphinx-apidoc", "-o", output_dir, module_dir, "--force"])


def skip(app, what, name, obj, would_skip, options):
    if name.startswith("_") or not getattr(obj, "__doc__", None):
        return True
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect("builder-inited", run_apidoc)
