"""rq_import.py: Module to import a nexus corp grid & properties, or vdb, or vdb ensemble into resqml format."""

version = '22nd October 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and GOCAD are trademarks of Emerson

import logging

log = logging.getLogger(__name__)
log.debug('rq_import.py version ' + version)

from resqpy.rq_import import GridFromCp


def grid_from_cp(model,
                 cp_array,
                 crs_uuid,
                 active_mask=None,
                 geometry_defined_everywhere=True,
                 treat_as_nan=None,
                 dot_tolerance=1.0,
                 morse_tolerance=5.0,
                 max_z_void=0.1,
                 split_pillars=True,
                 split_tolerance=0.01,
                 ijk_handedness='right',
                 known_to_be_straight=False):
    """Create a resqpy.grid.Grid object from a 7D corner point array.

    notes:
       this function sets up all the geometry arrays in memory but does not write to hdf5 nor create xml: use Grid methods;
       geometry_defined_everywhere is deprecated, use treat_as_nan instead
    """
    grid = GridFromCp(model,
                      cp_array,
                      crs_uuid,
                      active_mask,
                      geometry_defined_everywhere,
                      treat_as_nan,
                      dot_tolerance,
                      morse_tolerance,
                      max_z_void,
                      split_pillars,
                      split_tolerance,
                      ijk_handedness,
                      known_to_be_straight)

    return grid.grid
