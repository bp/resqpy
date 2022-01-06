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

_DEFAULT_VERSION = "0.0.0-version-not-available"


def _get_dev_version():
    """Lookup version from local git history (dev setup only)"""
    try:
        from setuptools_scm import get_version
        return get_version(root = '..', relative_to = __file__)
    except ImportError:
        warnings.warn("Missing dev dependency 'setuptools_scm'")
        return _DEFAULT_VERSION
    except Exception:
        return _DEFAULT_VERSION


try:
    # Prod setup: Look for hard-coded file resqpy/version.py , created when lib is packaged.
    from .version import version as __version__  # type: ignore
except ImportError:
    # Dev setup: Use local git history
    __version__ = _get_dev_version()
except Exception:
    __version__ = _DEFAULT_VERSION

log = logging.getLogger(__name__)
log.info(f"Imported resqpy version {__version__}")
