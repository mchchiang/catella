# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'nucmc'
copyright = '2026, Michael Chiang'
author = 'Michael Chiang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
]

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["$","$"], ["\\(","\\)"] ],
        'displayMath': [ ["$$","$$"], ["\\[","\\]"] ],
        'processEscapes': True,
    },
}

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
