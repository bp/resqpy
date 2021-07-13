# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# See also https://wwoods.github.io/2016/06/09/easy-sphinx-documentation-without-the-boilerplate/#setup

# Nb. useful stackexchange thread about the :recursive: option and custom templates:
# https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202

# -- Path setup --------------------------------------------------------------

# Do not do sys.path surgery
# Instead, resqpy package should be pip-installed in the python environment

# Mock imports for things not available
# autodoc_mock_imports = ["h5py"]

# -- Project information -----------------------------------------------------

project = 'resqpy'
copyright = '2021, BP'
author = 'BP'

# Version from git tag
# See https://github.com/pypa/setuptools_scm/#usage-from-sphinx
try:
   from importlib import metadata
   release = metadata.version('resqpy')
except Exception:
   release = '0.0.0-version-not-available'

# Take major/minor
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

autoclass_content = "class"  # Alternatively, "both" to include init method
# autodoc_default_options = {
#     'members': None,
#     'inherited-members': None,
#     'private-members': None,
#     'show-inheritance': None,
# }
autosummary_generate = True  # Make _autosummary files and include them
autosummary_generate_overwrite = True

napoleon_use_rtype = False  # More legible
autodoc_member_order = 'bysource'
# napoleon_numpy_docstring = False  # Force consistency, leave only Google

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.napoleon',
   'sphinx.ext.viewcode',
   'autoclasstoc',
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options custom autoclasstoc sections ------------------------------------

# See https://autoclasstoc.readthedocs.io/en/latest/advanced_usage.html

from autoclasstoc import PublicMethods


class CommomMethods(PublicMethods):
   key = "common-methods"
   title = "Commonly Used Methods:"

   def predicate(self, name, attr, meta):
      return super().predicate(name, attr, meta) and 'common' in meta


class OtherMethods(PublicMethods):
   key = "other-methods"
   title = "Methods:"

   def predicate(self, name, attr, meta):
      return super().predicate(name, attr, meta) and 'common' not in meta


autoclasstoc_sections = [
   "public-attrs",
   "common-methods",
   "other-methods",
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# Override table width restrictions
# See https://rackerlabs.github.io/docs-rackspace/tools/rtd-tables.html
html_static_path = ['_static']
html_context = {
   'css_files': [
      '_static/theme_overrides.css',  # override wide tables in RTD theme
   ],
}
