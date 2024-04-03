# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.abspath('../src')))
sys.path.insert(0, os.path.abspath(os.path.abspath('..')))

project = 'pyRASP'
copyright = '2024, Joseph S. Beckwith, Bin Fu, Steven F. Lee'
author = 'Joseph S. Beckwith, Bin Fu'
release = 'v0.5.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#  add in the extension names to the empty list variable 'extensions'
extensions = [
      'sphinx.ext.autodoc', 
      'sphinx.ext.napoleon', 
      'autodocsumm', 
      'sphinx.ext.coverage'
]

napoleon_custom_sections = [('Returns', 'params_style')]

# add in this line for the autosummary functionality
auto_doc_default_options = {'autosummary': True}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
