# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Deep Learning Guide Book'
# copyright = 'No Copyright'
html_show_copyright = False
html_logo = "logo/dlguidebook_logo_inverted-nobg.png"
html_favicon = "logo/dlguidebook_logo_inverted-nobg.png"

author = 'Nishant Baheti'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Deep Learning Guide Book"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "DL Guide Book"

# The full version, including alpha/beta/rc tags
# release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','_build', '**.ipynb_checkpoints','docs']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "piccolo_theme"

html_theme_options = {
    # "navigation_depth": 2,  # Controls depth of navigation tree
    # "collapse_navigation": True,  # Collapses sidebar sections by default
    # "show_nav_level": 1,  # Shows navigation level
    # "sidebar_hide_name": False,  # Show project name in the sidebar
}

# html_sidebars = {
#     "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
#     "root-doc-name" : []
# }

# import sphinx_pdj_theme
# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
# html_theme = 'furo'
