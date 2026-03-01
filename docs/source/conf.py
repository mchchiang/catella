# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

import os
import sys
import inspect

project = "nucmc"
copyright = "2026, Michael Chiang"
author = "Michael Chiang"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
]

mathjax4_config = {
    "tex": {
        "inlineMath": [ ["$","$"], ["\\(","\\)"] ],
        "displayMath": [ ["$$","$$"], ["\\[","\\]"] ],
        "processEscapes": True,
        "tags": "ams",        # Enable numbering
        "tagSide": "right",   # Place numbers on the RHS
        "tagAlign": "center", # Vertically center the number
    },
}

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autodoc_member_order = "bysource"

# Disable global numbering
numfig = False
math_numfig = False

# Make sure equations are numbered
math_number_all = True

def setup(app):
    app.add_css_file("custom.css")

# Ensure your code is importable
sys.path.insert(0, os.path.abspath(".."))
    
def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    # Get the module (nucmc)
    mod = sys.modules.get(info["module"])
    if mod is None:
        return None

    # Get the actual object (e.g., nucmc.run)
    obj = mod
    for part in info["fullname"].split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        # Find the actual file where the code is defined (api.py)
        # unwrap() handles cases where methods are decorated
        obj = inspect.unwrap(obj)
        filename = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    if not filename:
        return None

    # Convert absolute path to a relative path for GitLab
    # This finds the "nucmc" folder and calculates the path from there
    repo_root = os.path.abspath(".")
    rel_path = os.path.relpath(filename, start=repo_root)

    # GitLab URL Configuration
    project_url = "https://git.ecdf.ed.ac.uk/cchiang2/nucmc-project"
    branch = "main"

    return f"{project_url}/-/blob/{branch}/{rel_path}#L{lineno}"
    
# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
