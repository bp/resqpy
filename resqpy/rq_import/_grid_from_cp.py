"""_grid_from_cp.py: Module to generate a RESQML grid object from an input corner point array."""

version = '15th November 2021'

import logging

log = logging.getLogger(__name__)
log.debug('_grid_from_cp.py version ' + version)

from resqpy.rq_import import GridFromCp


def grid_from_cp(model,
                 cp_array,
                 crs_uuid,
                 active_mask = None,
                 geometry_defined_everywhere = True,
                 treat_as_nan = None,
                 dot_tolerance = 1.0,
                 morse_tolerance = 5.0,
                 max_z_void = 0.1,
                 split_pillars = True,
                 split_tolerance = 0.01,
                 ijk_handedness = 'right',
                 known_to_be_straight = False):
    """Create a resqpy.grid.Grid object from a 7D corner point array.

    Arguments:
        model (resqpy.model.Model): model to which the grid will be added
        cp_array (numpy float array): 7 dimensional numpy array of nexus corner point data, in nexus ordering
        crs_uuid (uuid.UUID): uuid for the coordinate reference system
        active_mask (3d numpy bool array): array indicating which cells are active
        geometry_defined_everywhere (bool, default True): if False then inactive cells are marked as not having geometry
        treat_as_nan (float, default None): if a value is provided corner points with this value will be assigned nan
        dot_tolerance (float, default 1.0): minimum manhatten distance of primary diagonal of cell, below which cell is treated as inactive
        morse_tolerance (float, default 5.0): maximum ratio of i and j face vector lengths, beyond which cells are treated as inactive
        max_z_void (float, default 0.1): maximum z gap between vertically neighbouring corner points. Vertical gaps greater than this will introduce k gaps into resqml grid. Units are corp z units
        split_pillars (bool, default True): if False an unfaulted grid will be generated
        split_tolerance (float, default 0.01): maximum distance between neighbouring corner points before a pillar is considered 'split'. Applies to each of x, y, z differences
        ijk_handedness (str, default 'right'): 'right' or 'left'
        known_to_be_straight (bool, default False): if True pillars are forced to be straight

    notes:
       this function sets up all the geometry arrays in memory but does not write to hdf5 nor create xml: use Grid methods;
       geometry_defined_everywhere is deprecated, use treat_as_nan instead
    """
    grid = GridFromCp(model, cp_array, crs_uuid, active_mask, geometry_defined_everywhere, treat_as_nan, dot_tolerance,
                      morse_tolerance, max_z_void, split_pillars, split_tolerance, ijk_handedness, known_to_be_straight)

    return grid.grid
