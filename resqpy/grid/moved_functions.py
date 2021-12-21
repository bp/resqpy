"""Submodule containing some general grid functions that have now been moved. This file will be deleted in a later release."""

import warnings

import resqpy.olio.grid_functions as gf
import resqpy.property.property_kind as pk


def establish_zone_property_kind(model):
    """MOVED: Returns zone local property kind object, creating the xml and adding as part if not found in model."""
    warnings.warn(
        'This function has been moved to property/property_kind. Please update your code to the new '
        'location.', DeprecationWarning)

    return pk.establish_zone_property_kind(model)


def find_cell_for_x_sect_xz(x_sect, x, z):
    """MOVED: Returns the (k0, j0) or (k0, i0) indices of the cell containing point x,z in the cross section.

    arguments:
       x_sect (numpy float array of shape (nk, nj or ni, 2, 2, 2 or 3): the cross section x,z or x,y,z data
       x (float) x-coordinate of point of interest in the cross section space
       z (float): y-coordinate of  point of interest in the cross section space

    note:
       the x_sect data is in the form returned by x_section_corner_points() or split_gap_x_section_points();
       the 2nd of the returned pair is either a J index or I index, whichever was not the axis specified
       when generating the x_sect data; returns (None, None) if point inclusion not detected; if xyz data is
       provided, the y values are ignored; note that the point of interest x,z coordinates are in the space of
       x_sect, so if rotation has occurred, the x value is no longer an easting and is typically picked off a
       cross section plot
    """
    warnings.warn('This function has been moved to olio/grid_functions. Please update your code to the new location.',
                  DeprecationWarning)

    return gf.find_cell_for_x_sect_xz(x_sect, x, z)
