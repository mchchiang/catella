# Configuration file for the Sphinx documentation builder.

import os
import sys
import inspect
import tomllib
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # docs/source -> docs -> project root

sys.path.insert(0, str(ROOT / "src"))

# -- Project information (read from installed package metadata) ---------------

info = metadata("catella")
project   = info.get("Name") or "catella"
author    = info.get("Author") or "Michael Chiang"
copyright = f"{datetime.now():%Y}, {author}"
version   = info["Version"]
release   = version

project_urls_raw = info.get_all("Project-URL") or []
urls = {}
for pu in project_urls_raw:
    if ", " in pu:
        k, v = pu.split(", ", 1)
        urls[k] = v
repository_url = urls.get("Homepage") or urls.get("Source") or ""

# -- Mock imports (avoid needing compiled extension or all runtime deps) -------

with open(ROOT / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

# numpy and pandas are installed in the docs env so their types resolve
# correctly in Napoleon-parsed docstrings. typer must not be mocked either,
# since sphinx-click imports catella.cli for real to build the CLI
# reference and needs an actual click.Group, not a mock object. pyarrow
# must not be mocked either, since pandas imports it for real at import
# time and chokes on a mocked pyarrow.__version__.
_no_mock = {"numpy", "pandas", "typer", "pyarrow"}
deps = pyproject.get("project", {}).get("dependencies", [])
autodoc_mock_imports = [
    dep.split(">")[0].split("=")[0].split("<")[0].strip()
    for dep in deps
    if dep.split(">")[0].split("=")[0].split("<")[0].strip() not in _no_mock
]
autodoc_mock_imports.append("catella_cpp")

# -- Member visibility (honour __all__) ----------------------------------------

def skip_unexported_members(app, what, name, obj, skip, options):
    module = inspect.getmodule(obj)
    if module is None:
        return skip
    if name.startswith("_"):
        return True
    exported = getattr(module, "__all__", None)
    if exported is not None and name not in exported:
        return True
    return skip

def setup(app):
    app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", skip_unexported_members)

# -- Source linking ------------------------------------------------------------

def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    mod = sys.modules.get(info["module"])
    if mod is None:
        return None
    obj = mod
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    try:
        obj = inspect.unwrap(obj)
        filename = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None
    if not filename:
        return None
    rel_path = os.path.relpath(filename, start=str(ROOT))
    branch = "main"
    return f"{repository_url}/blob/{branch}/{rel_path}#L{lineno}"

# -- General configuration ----------------------------------------------------

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx_click",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # Not ready to publish yet; remove once wired into index.rst's toctree.
    "tutorials/tut03_promoters.ipynb",
    "changelog.md",
    "contributing.md",
]

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_class_signature = "separated"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True

mathjax4_config = {
    "tex": {
        "inlineMath":   [["$", "$"], ["\\(", "\\)"]],
        "displayMath":  [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "tags":     "ams",
        "tagSide":  "right",
        "tagAlign": "center",
    },
}
numfig         = False
math_numfig    = False
math_number_all = True

# -- HTML output --------------------------------------------------------------

html_theme       = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_context = {
    "display_github":  True,
    "github_user":     "mchchiang",
    "github_repo":     "catella",
    "github_version":  "main",
    "conf_py_path":    "/docs/source/",
}
