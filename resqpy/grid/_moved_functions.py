"""Submodule containing some general grid functions that have now been moved. This file will be deleted in a later release."""

import warnings

import resqpy.grid._points_functions as pf
import resqpy.property.property_kind as pk


def establish_zone_property_kind(model):
    """MOVED: Returns zone local property kind object, creating the xml and adding as part if not found in model."""
    warnings.warn(
        'This function has been moved to property/property_kind. Please update your code to the new '
        'location.', DeprecationWarning)

    return pk.establish_zone_property_kind(model)
