""" RESQML manipulation library

.. autosummary::
    :toctree: _autosummary
    :caption: API Reference
    :template: custom-module-template.rst
    :recursive:

    model
    crs
    derived_model
    fault
    grid
    grid_surface
    lines
    organize
    property
    rq_import
    surface
    time_series
    unstructured
    weights_and_measures
    well
    olio

"""

try:
   # Version dynamically extracted from git tags when package is built
   from .version import version as __version__  # type: ignore

except ImportError:
   __version__ = "0.0.0-version-not-available"
