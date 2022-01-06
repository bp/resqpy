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

# Dynamically get true resqpy version
try:
    # Prod setup: Look for resqpy/version.py
    # This file is created by setuptools_scm when package is pip-installed
    from .version import version as __version__  # type: ignore
except ImportError:
    # Dev setup: Dynamically get version from local git history
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root = '..', relative_to = __file__)
    except Exception:
        __version__ = "0.0.0-version-not-available"
except Exception:
    __version__ = "0.0.0-version-not-available"

log = logging.getLogger(__name__)
log.info(f"Imported resqpy version {__version__}")
