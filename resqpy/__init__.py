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
    well
    olio

"""

try:
    # Version dynamically extracted from git tags when package is built
    from .version import version as __version__

except ImportError:
    __version__ = "0.0.0-version-not-available"
