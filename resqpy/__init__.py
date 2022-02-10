"""RESQML manipulation library.

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
    strata
    surface
    time_series
    unstructured
    weights_and_measures
    well
    olio
"""

import logging

__version__ = "0.0.0"  # Set at build time
log = logging.getLogger(__name__)
log.info(f"Imported resqpy version {__version__}")
