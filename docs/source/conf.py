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

mathjax4_config = {
    'tex': {
        'inlineMath': [ ["$","$"], ["\\(","\\)"] ],
        'displayMath': [ ["$$","$$"], ["\\[","\\]"] ],
        'processEscapes': True,
        'tags': 'ams',        # Enable numbering
        'tagSide': 'right',   # Place numbers on the RHS
        'tagAlign': 'center', # Vertically center the number
    },
}

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autodoc_member_order = 'bysource'

# Disable global numbering
numfig = False
math_numfig = False

# Make sure equations are numbered
math_number_all = True

def setup(app):
    app.add_css_file("custom.css")

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
