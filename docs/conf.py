# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath("../"))

project = "devinterp"
copyright = "2024, Van Wingerden et al."
author = "Van Wingerden et al."
release = "0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_math_dollar",
    "sphinx.ext.autosectionlabel",
]
autosectionlabel_prefix_document = True
mathjax_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    }
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "donate.html",
    ]
}
rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
    
.. role:: bash(code)
    :language: bash
    :class: highlight
"""
html_static_path = ["static"]


def run_apidoc(_):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    module_dir = os.path.join(
        current_dir, "..", "your_module"
    )  # Adjust the path to your module
    output_dir = os.path.join(current_dir, "")  # Adjust the path to your output
    subprocess.call(["sphinx-apidoc", "-o", output_dir, module_dir, "--force"])


def skip(app, what, name, obj, would_skip, options):
    if name.startswith("_") or not getattr(obj, "__doc__", None):
        return True
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect("builder-inited", run_apidoc)

    app.add_js_file("custom.js")
