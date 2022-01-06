import pytest
from packaging.version import Version

import resqpy
from resqpy import model


def test_empty_model():
    _ = model.Model()
    return


def test_all_imports():
    from resqpy import (crs, derived_model, fault, grid, grid_surface, lines, organize, property, rq_import, surface,
                        time_series, well)

    # The line below prevents IDEs from deleting the above
    _ = (crs, derived_model, fault, grid, grid_surface, lines, organize, property, rq_import, surface, time_series,
         well)

    #    from resqpy.olio import *
    return


def test_version():
    # This is dynamically created when package is built
    # If this fails when running tests locally, ensure you have installed the dev dependencies specified in setup.cfg
    # In particular, try:  pip install setuptools_scm
    version_string = resqpy.__version__

    # Ensure version string is a PEP-440 compliant version
    version = Version(version_string)
    assert version >= Version("1.1.1")
