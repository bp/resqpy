"""well_functions.py: resqpy well module providing trajectory, deviation survey, blocked well, wellbore frame and marker frame and md datum classes.

"""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

version = '10th November 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('well_functions.py version ' + version)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo

from .md_datum import MdDatum
from .wellbore_marker_frame import WellboreMarkerFrame
from .well_functions import WellboreFrame, DeviationSurvey
from .blocked_well import BlockedWell
from .trajectory import Trajectory

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