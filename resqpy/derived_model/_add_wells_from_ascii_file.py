"""High level add_wells_from_ascii_file() function."""

import logging

log = logging.getLogger(__name__)

import os

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.model as rq
import resqpy.olio.xml_et as rqet
import resqpy.well as rqw

import resqpy.derived_model._common as rqdm_c


def add_wells_from_ascii_file(epc_file,
                              crs_uuid,
                              trajectory_file,
                              comment_character = '#',
                              space_separated_instead_of_csv = False,
                              well_col = 'WELL',
                              md_col = 'MD',
                              x_col = 'X',
                              y_col = 'Y',
                              z_col = 'Z',
                              length_uom = 'm',
                              md_domain = None,
                              drilled = False,
                              z_inc_down = True,
                              new_epc_file = None):
    """Adds new md datum, trajectory, interpretation and feature objects for each well in a tabular ascii file..

    arguments:
       epc_file (string): file name to load model resqml model from (and rewrite to if new_epc_file is None)
       crs_uuid (uuid.UUID): the unique identifier of the coordinate reference system applicable to the x,y,z data;
          if None, a default crs will be created, making use of the length_uom and z_inc_down arguments
       trajectory_file (string): the path of the ascii file holding the well trajectory data to be loaded
       comment_character (string, default '#'): character deemed to introduce a comment in the trajectory file
       space_separated_instead_of_csv (boolean, default False): if True, the columns in the trajectory file are space
          separated; if False, comma separated
       well_col (string, default 'WELL'): the heading for the column containing well names
       md_col (string, default 'MD'): the heading for the column containing measured depths
       x_col (string, default 'X'): the heading for the column containing X (usually easting) data
       y_col (string, default 'Y'): the heading for the column containing Y (usually northing) data
       z_col (string, default 'Z'): the heading for the column containing Z (depth or elevation) data
       length_uom (string, default 'm'): the units of measure for the measured depths; should be 'm' or 'ft'
       md_domain (string, optional): the source of the original deviation data; may be 'logger' or 'driller'
       drilled (boolean, default False): True should be used for wells that have been drilled; False otherwise (planned,
          proposed, or a location being studied)
       z_inc_down (boolean, default True): indicates whether z values increase with depth; only used in the creation
          of a default coordinate reference system; ignored if crs_uuid is not None
       new_epc_file (string, optional): if None, the source epc_file is extended with the new property object; if present,
          a new epc file (& associated h5 file) is created to contain a copy of the grid and the new property

    returns:
       int: the number of wells added

    notes:
       ascii file must be table with first line being column headers, with columns for WELL, MD, X, Y & Z;
       actual column names can be set with optional arguments;
       all the objects are added to the model, with array data being written to the hdf5 file for the trajectories;
       the md_domain and drilled values are stored in the RESQML metadata but are only for human information and do not
       generally affect computations
    """

    assert trajectory_file and os.path.exists(trajectory_file)
    if md_domain:
        assert md_domain in ['driller', 'logger']

    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None

    # open up model
    if new_epc_file:
        model = rq.Model(new_epc_file, copy_from = epc_file)
    else:
        model = rq.Model(epc_file)

    # sort out the coordinate reference system
    if crs_uuid is None:
        crs_uuid = model.crs_uuid
    if crs_uuid is None:
        if z_inc_down is None:
            z_inc_down = True
        crs = rqc.Crs(model, xy_units = length_uom, z_units = length_uom, z_inc_down = z_inc_down)
        crs.create_xml()
        crs_uuid = crs.uuid

    # add all the well related objects to the model, based on data in the ascii file
    (feature_list, interpretation_list, trajectory_list, md_datum_list) =  \
       rqw.add_wells_from_ascii_file(model, crs_uuid, trajectory_file, comment_character = comment_character,
                                     space_separated_instead_of_csv = space_separated_instead_of_csv,
                                     well_col = well_col, md_col = md_col, x_col = x_col, y_col = y_col, z_col = z_col,
                                     length_uom = length_uom, md_domain = md_domain, drilled = drilled)

    assert len(feature_list) == len(interpretation_list) == len(trajectory_list) == len(md_datum_list)
    count = len(feature_list)

    log.info('features, interpretations, trajectories and md data added for ' + str(count) + ' well' +
             rqdm_c._pl(count))

    # write or re-write model
    model.h5_release()
    model.store_epc()

    return count
