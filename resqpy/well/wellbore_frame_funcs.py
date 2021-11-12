"""well_functions.py: resqpy well module providing trajectory, deviation survey, blocked well, wellbore frame and marker frame and md datum classes.

"""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

version = '10th November 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('well_functions.py version ' + version)

import warnings

import lasio
import numpy as np

import resqpy.property as rqp
import resqpy.weights_and_measures as bwam
from .wellbore_frame import WellboreFrame

def add_las_to_trajectory(las: lasio.LASFile, trajectory, realization = None, check_well_name = False):
    """Creates a WellLogCollection and WellboreFrame from a LAS file.

    Note:
       In this current implementation, the first curve in the las object must be
       Measured Depths, not e.g. TVDSS.

    Arguments:
       las: an lasio.LASFile object
       trajectory: an instance of :class:`resqpy.well.Trajectory` .
       realization (integer): if present, the single realisation (within an ensemble)
          that this collection is for
       check_well_name (bool): if True, raise warning if LAS well name does not match
          existing wellborefeature citation title

    Returns:
       collection, well_frame: instances of :class:`resqpy.property.WellLogCollection`
          and :class:`resqpy.well.WellboreFrame`
    """

    # Lookup relevant related resqml parts
    model = trajectory.model
    well_interp = trajectory.wellbore_interpretation
    well_title = well_interp.title

    if check_well_name and well_title != las.well.WELL.value:
        warnings.warn(f'LAS well title {las.well.WELL.value} does not match resqml tite {well_title}')

    # Create a new wellbore frame, using depth data from first curve in las file
    depth_values = np.array(las.index).copy()
    assert isinstance(depth_values, np.ndarray)
    las_depth_uom = bwam.rq_length_unit(las.curves[0].unit)

    # Ensure depth units are correct
    bwam.convert_lengths(depth_values, from_units = las_depth_uom, to_units = trajectory.md_uom)
    assert len(depth_values) > 0

    well_frame = WellboreFrame(
        parent_model = model,
        trajectory = trajectory,
        mds = depth_values,
        represented_interp = well_interp,
    )
    well_frame.write_hdf5()
    well_frame.create_xml()

    # Create a WellLogCollection in which to put logs
    collection = rqp.WellLogCollection(frame = well_frame, realization = realization)

    # Read in data from each curve in turn (skipping first curve which has depths)
    for curve in las.curves[1:]:

        collection.add_log(
            title = curve.mnemonic,
            data = curve.data,
            unit = curve.unit,
            realization = realization,
            write = False,
        )
        collection.write_hdf5_for_imported_list()
        collection.create_xml_for_imported_list_and_add_parts_to_model()

    return collection, well_frame