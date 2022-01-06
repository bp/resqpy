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
import warnings


def _determine_version():

    # Prod setup: Look for hard-coded file resqpy/version.py
    try:
        from .version import version  # type: ignore
        return version
    except Exception:
        pass

    # Dev setup: Use local git history
    try:
        from setuptools_scm import get_version
        return get_version(root = '..', relative_to = __file__)
    except Exception as e:
        warnings.warn("Unable to determine resqpy version: " + str(e))
        return "0.0.0-version-not-available"


__version__ = _determine_version()
log = logging.getLogger(__name__)
log.info(f"Imported resqpy version {__version__}")
