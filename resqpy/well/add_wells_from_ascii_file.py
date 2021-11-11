"""well_functions.py: resqpy well module providing trajectory, deviation survey, blocked well, wellbore frame and marker frame and md datum classes.

"""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

version = '10th November 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('well_functions.py version ' + version)

import pandas as pd

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo

from .md_datum import MdDatum
from .well_functions import WellboreFrame, DeviationSurvey
from .wellbore_marker_frame import WellboreMarkerFrame
from .blocked_well import BlockedWell
from .trajectory import Trajectory

def add_wells_from_ascii_file(model,
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
                              drilled = False):
    """Creates new md datum, trajectory, interpretation and feature objects for each well in an ascii file.

    arguments:
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

    returns:
       tuple of lists of objects: (feature_list, interpretation_list, trajectory_list, md_datum_list),

    notes:
       ascii file must be table with first line being column headers, with columns for WELL, MD, X, Y & Z;
       actual column names can be set with optional arguments;
       all the objects are added to the model, with array data being written to the hdf5 file for the trajectories;
       the md_domain and drilled values are stored in the RESQML metadata but are only for human information and do not
       generally affect computations
    """

    assert md_col and x_col and y_col and z_col
    md_col = str(md_col)
    x_col = str(x_col)
    y_col = str(y_col)
    z_col = str(z_col)
    if crs_uuid is None:
        crs_uuid = model.crs_uuid
    assert crs_uuid is not None, 'coordinate reference system not found when trying to add wells'

    try:
        df = pd.read_csv(trajectory_file,
                         comment = comment_character,
                         delim_whitespace = space_separated_instead_of_csv)
        if df is None:
            raise Exception
    except Exception:
        log.error('failed to read ascii deviation survey file: ' + str(trajectory_file))
        raise
    if well_col and well_col not in df.columns:
        log.warning('well column ' + str(well_col) + ' not found in ascii trajectory file: ' + str(trajectory_file))
        well_col = None
    if well_col is None:
        for col in df.columns:
            if str(col).upper().startswith('WELL'):
                well_col = str(col)
                break
    else:
        well_col = str(well_col)
    assert well_col
    unique_wells = set(df[well_col])
    if len(unique_wells) == 0:
        log.warning('no well data found in ascii trajectory file: ' + str(trajectory_file))
        # note: empty lists will be returned, below

    feature_list = []
    interpretation_list = []
    trajectory_list = []
    md_datum_list = []

    for well_name in unique_wells:

        log.debug('importing well: ' + str(well_name))
        # create single well data frame (assumes measured depths increasing)
        well_df = df[df[well_col] == well_name]
        # create a measured depth datum for the well and add as part
        first_row = well_df.iloc[0]
        if first_row[md_col] == 0.0:
            md_datum = MdDatum(model,
                               crs_uuid = crs_uuid,
                               location = (first_row[x_col], first_row[y_col], first_row[z_col]))
        else:
            md_datum = MdDatum(model, crs_uuid = crs_uuid,
                               location = (first_row[x_col], first_row[y_col], 0.0))  # sea level datum
        md_datum.create_xml(title = str(well_name))
        md_datum_list.append(md_datum)

        # create a well feature and add as part
        feature = rqo.WellboreFeature(model, feature_name = well_name)
        feature.create_xml()
        feature_list.append(feature)

        # create interpretation and add as part
        interpretation = rqo.WellboreInterpretation(model, is_drilled = drilled, wellbore_feature = feature)
        interpretation.create_xml(title_suffix = None)
        interpretation_list.append(interpretation)

        # create trajectory, write arrays to hdf5 and add as part
        trajectory = Trajectory(model,
                                md_datum = md_datum,
                                data_frame = well_df,
                                length_uom = length_uom,
                                md_domain = md_domain,
                                represented_interp = interpretation,
                                well_name = well_name)
        trajectory.write_hdf5()
        trajectory.create_xml(title = well_name)
        trajectory_list.append(trajectory)

    return (feature_list, interpretation_list, trajectory_list, md_datum_list)


def well_name(well_object, model = None):
    """Returns the 'best' citation title from the object or related well objects.

    arguments:
       well_object (object, uuid or root): Object for which a well name is required. Can be a
          Trajectory, WellboreInterpretation, WellboreFeature, BlockedWell, WellboreMarkerFrame,
          WellboreFrame, DeviationSurvey or MdDatum object
       model (model.Model, optional): required if passing a uuid or root; not recommended otherwise

    returns:
       string being the 'best' citation title to serve as a well name, form the object or some related objects

    note:
       xml and relationships must be established for this function to work
    """

    def better_root(model, root_a, root_b):
        a = rqet.citation_title_for_node(root_a)
        b = rqet.citation_title_for_node(root_b)
        if a is None or len(a) == 0:
            return root_b
        if b is None or len(b) == 0:
            return root_a
        parts_like_a = model.parts(title = a)
        parts_like_b = model.parts(title = b)
        if len(parts_like_a) > 1 and len(parts_like_b) == 1:
            return root_b
        elif len(parts_like_b) > 1 and len(parts_like_a) == 1:
            return root_a
        a_digits = 0
        for c in a:
            if c.isdigit():
                a_digits += 1
        b_digits = 0
        for c in b:
            if c.isdigit():
                b_digits += 1
        if a_digits < b_digits:
            return root_b
        return root_a

    def best_root(model, roots_list):
        if len(roots_list) == 0:
            return None
        if len(roots_list) == 1:
            return roots_list[0]
        if len(roots_list) == 2:
            return better_root(model, roots_list[0], roots_list[1])
        return better_root(model, roots_list[0], best_root(model, roots_list[1:]))

    def best_root_for_object(well_object, model = None):

        if well_object is None:
            return None
        if model is None:
            model = well_object.model
        root_list = []
        obj_root = None
        obj_uuid = None
        obj_type = None
        traj_root = None

        if isinstance(well_object, str):
            obj_uuid = bu.uuid_from_string(well_object)
            assert obj_uuid is not None, 'well_name string argument could not be interpreted as uuid'
            well_object = obj_uuid
        if isinstance(well_object, bu.uuid.UUID):
            obj_uuid = well_object
            obj_root = model.root_for_uuid(obj_uuid)
            assert obj_root is not None, 'uuid not found in model when looking for well name'
            obj_type = rqet.node_type(obj_root)
        elif rqet.is_node(well_object):
            obj_root = well_object
            obj_type = rqet.node_type(obj_root)
            obj_uuid = rqet.uuid_for_part_root(obj_root)
        elif isinstance(well_object, Trajectory):
            obj_type = 'WellboreTrajectoryRepresentation'
            traj_root = well_object.root
        elif isinstance(well_object, rqo.WellboreFeature):
            obj_type = 'WellboreFeature'
        elif isinstance(well_object, rqo.WellboreInterpretation):
            obj_type = 'WellboreInterpretation'
        elif isinstance(well_object, BlockedWell):
            obj_type = 'BlockedWellboreRepresentation'
            if well_object.trajectory is not None:
                traj_root = well_object.trajectory.root
        elif isinstance(well_object, WellboreMarkerFrame):  # note: trajectory might be None
            obj_type = 'WellboreMarkerFrameRepresentation'
            if well_object.trajectory is not None:
                traj_root = well_object.trajectory.root
        elif isinstance(well_object, WellboreFrame):  # note: trajectory might be None
            obj_type = 'WellboreFrameRepresentation'
            if well_object.trajectory is not None:
                traj_root = well_object.trajectory.root
        elif isinstance(well_object, DeviationSurvey):
            obj_type = 'DeviationSurveyRepresentation'
        elif isinstance(well_object, MdDatum):
            obj_type = 'MdDatum'

        assert obj_type is not None, 'argument type not recognized for well_name'
        if obj_type.startswith('obj_'):
            obj_type = obj_type[4:]
        if obj_uuid is None:
            obj_uuid = well_object.uuid
            obj_root = model.root_for_uuid(obj_uuid)

        if obj_type == 'WellboreFeature':
            interp_parts = model.parts(obj_type = 'WellboreInterpretation')
            interp_parts = model.parts_list_filtered_by_related_uuid(interp_parts, obj_uuid)
            all_parts = interp_parts
            all_traj_parts = model.parts(obj_type = 'WellboreTrajectoryRepresentation')
            if interp_parts is not None:
                for part in interp_parts:
                    traj_parts = model.parts_list_filtered_by_related_uuid(all_traj_parts, model.uuid_for_part(part))
                    all_parts += traj_parts
            if all_parts is not None:
                root_list = [model.root_for_part(part) for part in all_parts]
        elif obj_type == 'WellboreInterpretation':
            feat_roots = model.roots(obj_type = 'WellboreFeature', related_uuid = obj_uuid)  # should return one root
            traj_roots = model.roots(obj_type = 'WellboreTrajectoryRepresentation', related_uuid = obj_uuid)
            root_list = feat_roots + traj_roots
        elif obj_type == 'WellboreTrajectoryRepresentation':
            interp_parts = model.parts(obj_type = 'WellboreInterpretation')
            interp_parts = model.parts_list_filtered_by_related_uuid(interp_parts, obj_uuid)
            all_parts = interp_parts
            all_feat_parts = model.parts(obj_type = 'WellboreFeature')
            if interp_parts is not None:
                for part in interp_parts:
                    feat_parts = model.parts_list_filtered_by_related_uuid(all_feat_parts, model.uuid_for_part(part))
                    all_parts += feat_parts
            if all_parts is not None:
                root_list = [model.root_for_part(part) for part in all_parts]
        elif obj_type in [
                'BlockedWellboreRepresentation', 'WellboreMarkerFrameRepresentation', 'WellboreFrameRepresentation'
        ]:
            if traj_root is None:
                traj_root = model.root(obj_type = 'WellboreTrajectoryRepresentation', related_uuid = obj_uuid)
            root_list = [best_root_for_object(traj_root, model = model)]
        elif obj_type == 'DeviationSurveyRepresentation':
            root_list = [best_root_for_object(model.root(obj_type = 'MdDatum', related_uuid = obj_uuid), model = model)]
        elif obj_type == 'MdDatum':
            pass

        root_list.append(obj_root)

        return best_root(model, root_list)

    return rqet.citation_title_for_node(best_root_for_object(well_object, model = model))