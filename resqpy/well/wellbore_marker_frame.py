"""wellbore_marker_frame.py: resqpy well module providing marker frame class.

"""

version = '18th November 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('wellbore_marker_frame.py version ' + version)

import numpy as np
import pandas as pd

import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns

from .trajectory import Trajectory
from .well_utils import load_hdf5_array


class WellboreMarkerFrame(BaseResqpy):
    """Class to handle RESQML WellBoreMarkerFrameRepresentation objects.

    note:
       measured depth data must be in same crs as those for the related trajectory
    """

    resqml_type = 'WellboreMarkerFrameRepresentation'

    def __init__(self,
                 parent_model,
                 wellbore_marker_frame_root = None,
                 uuid = None,
                 trajectory = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Creates a new wellbore marker object and optionally loads it from xml, or trajectory, or Nexus wellspec file.

        arguments:
           parent_model (model.Model object): the model which the new blocked well belongs to
           wellbore_marker_root (DEPRECATED): the root node of an xml tree representing the wellbore marker;
           trajectory (optional, Trajectory object): the trajectory of the well, to be intersected with the grid;
              not used if wellbore_marker_root is not None;
           title (str, optional): the citation title to use for a new wellbore marker frame;
              ignored if uuid or wellbore_marker_frame_root is not None
           originator (str, optional): the name of the person creating the wellbore marker frame, defaults to login id;
              ignored if uuid or wellbore_marker_frame_root is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore marker frame;
              ignored if uuid or wellbore_marker_frame_root is not None

        returns:
           the newly created wellbore framework marker object
        """

        self.trajectory = trajectory
        self.node_count = None  # number of measured depth nodes, each being for a marker
        self.node_mds = None  # node_count measured depths (in same units and datum as trajectory) of markers
        # self.wellbore_marker_list = [
        # ]  # list of markers, each: (marker UUID, geologic boundary, marker citation title, interp. object)
        self.wellbore_marker_list = []

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata,
                         root_node = wellbore_marker_frame_root)

    def load_from_data_frame(self,
                             dataframe,
                             md_col = 'MD',
                             x_col = 'X',
                             y_col = 'Y',
                             z_col = 'Z',
                             boundary_kind_col = 'Type',
                             feature_col = 'Feature_Name',
                             interp_col = 'Interp_Name',
                             well_col = 'Well'):
        """Load wellbore marker frame data from a pandas data frame.

        Args:
           dataframe: a pandas dataframe holding the wellbore marker frame data
           md_col (string, default 'MD'): the name of the column holding measured depth values
           x_col (string, default 'X'): the name of the column holding X values
           y_col (string, default 'Y'): the name of the column holding Y values
           z_col (string, default 'Z'): the name of the column holding Z values
           boundary_kind_col (string, default 'Type'): the name of the column holding boundary feature kind values
           feature_col (string, default 'Feature_Name'): the name of the column holding feature name values
           interp_col (string, default 'Interp_Name'): the name of the column holding feature interpretation name values
           well_col (string, default 'Well'): the name of the column holding well name value
        """

        for col in [md_col, x_col, y_col, z_col, boundary_kind_col, feature_col, interp_col, well_col]:
            assert col in dataframe.columns
        self.node_count = len(dataframe)
        self.node_mds = dataframe[md_col].values
        self.get_wellbore_marker_list(dataframe)

    def get_wellbore_marker_list(self, dataframe):
        """ Create a list of wellbore markers from a pandas dataframe. Each wellbore marker tuple should contain
        (marker UUID, geologic boundary, marker citation title, interp. object).

        Args:
           dataframe: a pandas dataframe holding the wellbore marker frame data
        """

        for i, row in enumerate(dataframe.iterrows()):
            marker_uuid = str(self.model.uuid(title = row[1]['Feature_Name'], title_case_sensitive = False))
            geologic_boundary = row[1]['Type']
            marker_citation_title = row[1]['Feature_Name']
            interp_object = self.get_interpretation_obj(
                interpretation_uuid = self.model.uuid(title = row[1]['Interp_Name'], title_case_sensitive = False))
            self.wellbore_marker_list.append((marker_uuid, geologic_boundary, marker_citation_title, interp_object))

    def get_trajectory_obj(self, trajectory_uuid):
        """Returns a trajectory object.

        arguments:
           trajectory_uuid (string or uuid.UUID): the uuid of the trajectory for which a Trajectory object is required

        returns:
           well.Trajectory object

        note:
           this method is not usually called directly
        """

        if trajectory_uuid is None:
            log.error('no trajectory was found')
            return None
        else:
            # create new trajectory object
            trajectory_root_node = self.model.root_for_uuid(trajectory_uuid)
            assert trajectory_root_node is not None, 'referenced wellbore trajectory missing from model'
            # return Trajectory(self.model, trajectory_root = trajectory_root_node)
            return Trajectory(self.model, uuid = trajectory_uuid)

    def get_interpretation_obj(self, interpretation_uuid, interp_type = None):
        """Creates an interpretation object; returns a horizon or fault interpretation object.

        arguments:
           interpretation_uiud (string or uuid.UUID): the uuid of the required interpretation object
           interp_type (string, optional): 'HorizonInterpretation' or 'FaultInterpretation' (optionally
              prefixed with `obj_`); if None, the type is inferred from the xml for the given uuid

        returns:
           organization.HorizonInterpretation or organization.FaultInterpretation object

        note:
           this method is not usually called directly
        """

        assert interpretation_uuid is not None, 'interpretation uuid argument missing'

        interpretation_root_node = self.model.root_for_uuid(interpretation_uuid)

        if not interp_type:
            interp_type = rqet.node_type(interpretation_root_node)

        if not interp_type.startswith('obj_'):
            interp_type = 'obj_' + interp_type

        if interp_type == 'obj_HorizonInterpretation':
            # create new horizon interpretation object
            # return rqo.HorizonInterpretation(self.model, root_node = interpretation_root_node)
            return rqo.HorizonInterpretation(self.model, uuid = interpretation_uuid)

        elif interp_type == 'obj_FaultInterpretation':
            # create new fault interpretation object
            # return rqo.FaultInterpretation(self.model, root_node = interpretation_root_node)
            return rqo.FaultInterpretation(self.model, uuid = interpretation_uuid)

        elif interp_type == 'obj_GeobodyInterpretation':
            # create new geobody interpretation object
            # return rqo.GeobodyInterpretation(self.model, root_node = interpretation_root_node)
            return rqo.GeobodyInterpretation(self.model, uuid = interpretation_uuid)
        else:
            # No interpretation for the marker
            return None
            # log.error('interpretation type not recognized: ' + str(interp_type))

    def _load_from_xml(self):
        """Loads the wellbore marker frame object from an xml node (and associated hdf5 data).

        note:
           this method is not usually called directly
        """

        wellbore_marker_frame_root = self.root
        assert wellbore_marker_frame_root is not None

        self.trajectory = self.get_trajectory_obj(
            rqet.find_nested_tags_text(wellbore_marker_frame_root, ['Trajectory', 'UUID']))

        # list of Wellbore markers, each: (marker UUID, geologic boundary, marker citation title, interp. object)
        self.wellbore_marker_list = []
        for tag in rqet.list_of_tag(wellbore_marker_frame_root, 'WellboreMarker'):
            interp_tag = rqet.content_type(rqet.find_nested_tags_text(tag, ['Interpretation', 'ContentType']))
            if interp_tag is not None:
                interp_obj = self.get_interpretation_obj(rqet.find_nested_tags_text(tag, ['Interpretation', 'UUID']),
                                                         interp_tag)
            else:
                interp_obj = None
            self.wellbore_marker_list.append(
                (str(rqet.uuid_for_part_root(tag)), rqet.find_tag_text(tag, 'GeologicBoundaryKind'),
                 rqet.find_nested_tags_text(tag, ['Citation', 'Title']), interp_obj))

        self.node_count = rqet.find_tag_int(wellbore_marker_frame_root, 'NodeCount')
        load_hdf5_array(self, rqet.find_tag(wellbore_marker_frame_root, 'NodeMd'), "node_mds", tag = 'Values')
        if self.node_count != len(self.node_mds):
            log.error('node count does not match hdf5 array')

        if len(self.wellbore_marker_list) != self.node_count:
            log.error('wellbore marker list does not contain correct node count')

    def dataframe(self):
        """Returns a pandas dataframe with columns X, Y, Z, MD, Type, Surface, Well."""

        # todo: handle fractures and geobody boundaries as well as horizons and faults

        xyz = np.empty((self.node_count, 3))
        type_list = []
        surface_list = []
        well_list = []

        for i in range(self.node_count):
            _, boundary_kind, title, interp = self.wellbore_marker_list[i]
            if interp:
                if boundary_kind == 'horizon':
                    # feature_name = rqo.GeneticBoundaryFeature(self.model, root_node = interp.feature_root).feature_name
                    feature_name = rqo.GeneticBoundaryFeature(self.model, uuid = interp.uuid).feature_name
                elif boundary_kind == 'fault':
                    # feature_name = rqo.TectonicBoundaryFeature(self.model, root_node = interp.feature_root).feature_name
                    feature_name = rqo.TectonicBoundaryFeature(self.model, uuid = interp.uuid).feature_name
                elif boundary_kind == 'geobody':
                    # feature_name = rqo.GeneticBoundaryFeature(self.model, root_node = interp.feature_root).feature_name
                    feature_name = rqo.GeneticBoundaryFeature(self.model, uuid = interp.uuid).feature_name
                else:
                    assert False, 'unexpected boundary kind'
            else:
                feature_name = title
            boundary_kind = boundary_kind[0].upper() + boundary_kind[1:]
            feature_name = '"' + feature_name + '"'
            xyz[i] = self.trajectory.xyz_for_md(self.node_mds[i])
            type_list.append(boundary_kind)
            surface_list.append(feature_name)
            if self.trajectory.wellbore_interpretation is None:
                well_name = '"' + self.trajectory.title + '"'  # todo: trace through wellbore interp to wellbore feature name
            else:
                well_name = '"' + self.trajectory.wellbore_interpretation.title + '"'  # use wellbore_interpretation title instead, RMS exports have feature_name as "Wellbore feature"
                # well_name = '"' + self.trajectory.wellbore_interpretation.wellbore_feature.feature_name + '"'
            well_list.append(well_name)

        return pd.DataFrame({
            'X': xyz[:, 0],
            'Y': xyz[:, 1],
            'Z': xyz[:, 2],
            'MD': self.node_mds,
            'Type': type_list,
            'Surface': surface_list,
            'Well': well_list
        })

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   title = 'wellbore marker framework',
                   originator = None):

        assert type(add_as_part) is bool

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()

        wbm_node = super().create_xml(originator = originator, add_as_part = False)

        node_count, nodeMd, md_values_node = self.__add_sub_elements_to_root_node(wbm_node = wbm_node)

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'mds', root = md_values_node)

        if self.trajectory is not None:
            traj_root = self.trajectory.root
            self.model.create_ref_node('Trajectory',
                                       rqet.find_tag(rqet.find_tag(traj_root, 'Citation'), 'Title').text,
                                       bu.uuid_from_string(traj_root.attrib['uuid']),
                                       content_type = 'obj_WellboreTrajectoryRepresentation',
                                       root = wbm_node)
        else:
            log.error('trajectory object is missing and must be included')

        # fill wellbore marker
        for marker in self.wellbore_marker_list:

            wbm_node_obj = self.model.new_obj_node('WellboreMarker', is_top_lvl_obj = False)
            wbm_node_obj.set('uuid', marker[0])
            wbm_node.append(wbm_node_obj)
            wbm_gb_node = rqet.SubElement(wbm_node_obj, ns['resqml2'] + 'GeologicBoundaryKind')
            wbm_gb_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
            wbm_gb_node.text = str(marker[1])

            interp = marker[3]
            if interp is not None:
                interp_root = marker[3].root
                if 'HorizonInterpretation' in str(type(marker[3])):
                    self.model.create_ref_node('Interpretation',
                                               rqet.find_tag(rqet.find_tag(interp_root, 'Citation'), 'Title').text,
                                               bu.uuid_from_string(interp_root.attrib['uuid']),
                                               content_type = 'obj_HorizonInterpretation',
                                               root = wbm_node_obj)

                elif 'FaultInterpretation' in str(type(marker[3])):
                    self.model.create_ref_node('Interpretation',
                                               rqet.find_tag(rqet.find_tag(interp_root, 'Citation'), 'Title').text,
                                               bu.uuid_from_string(interp_root.attrib['uuid']),
                                               content_type = 'obj_FaultInterpretation',
                                               root = wbm_node_obj)

        # add as part
        self.__add_as_part_and_add_relationships(wbm_node = wbm_node,
                                                 ext_uuid = ext_uuid,
                                                 add_as_part = add_as_part,
                                                 add_relationships = add_relationships)

        return wbm_node

    def __add_sub_elements_to_root_node(self, wbm_node):
        """Appends sub-elements to the WellboreMarkerFrame object's root node."""

        nodeCount = rqet.SubElement(wbm_node, ns['resqml2'] + 'NodeCount')
        nodeCount.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        nodeCount.text = str(self.node_count)

        nodeMd = rqet.SubElement(wbm_node, ns['resqml2'] + 'NodeMd')
        nodeMd.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
        nodeMd.text = rqet.null_xml_text

        md_values_node = rqet.SubElement(nodeMd, ns['resqml2'] + 'Values')
        md_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
        md_values_node.text = rqet.null_xml_text

        return nodeCount, nodeMd, md_values_node

    def __add_as_part_and_add_relationships(self, wbm_node, ext_uuid, add_as_part, add_relationships):
        """Add the newly created WellboreMarkerFrame object's root node as a part in the model
         and add reciprocal relationships."""

        if add_as_part:
            self.model.add_part('obj_WellboreMarkerFrameRepresentation', self.uuid, wbm_node)

            if add_relationships:
                self.model.create_reciprocal_relationship(wbm_node, 'destinationObject', self.trajectory.root,
                                                          'sourceObject')
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(wbm_node, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

                for marker in self.wellbore_marker_list:
                    self.model.create_reciprocal_relationship(wbm_node, 'destinationObject', marker[3].root,
                                                              'sourceObject')

    def write_hdf5(self, file_name = None, mode = 'a'):
        """Writes the hdf5 array associated with this object (the measured depth data).

        arguments:
           file_name (string): the name of the hdf5 file, or None, in which case the model's default will be used
           mode (string, default 'a'): the write mode for the hdf5, either 'w' or 'a'
        """

        h5_reg = rwh5.H5Register(self.model)
        h5_reg.register_dataset(self.uuid, 'Mds', self.node_mds)
        h5_reg.write(file = file_name, mode = mode)

    def find_marker_from_interp(self, interpetation_obj = None, uuid = None):
        """Find wellbore marker by interpretation; can pass object or uuid.

        arguments:
           interpretation_obj (organize.HorizonInterpretation or organize.FaultInterpretation object, optional):
              if present, the first (smallest md) marker relating to this interpretation object is returned
           uuid (string or uuid.UUID): if present, the uuid of the interpretation object of interest; ignored if
              interpretation_obj is not None

        returns:
           tuple, list of tuples or None; tuple is (marker UUID, geologic boundary, marker citation title, interp. object)

        note:
           if no arguments are passed, then a list of wellbore markers is returned;
           if no marker is found for the interpretation object, None is returned
        """

        if interpetation_obj is None and uuid is None:
            return self.wellbore_marker_list

        if interpetation_obj is not None:
            uuid = interpetation_obj.uuid

        for marker in self.wellbore_marker_list:
            if bu.matching_uuids(marker[3].uuid, uuid):
                return marker

        return None

    def get_marker_count(self):
        """Retruns number of wellbore markers."""

        return len(self.wellbore_marker_list)

    def find_marker_from_index(self, idx):
        """Returns wellbore marker by index."""

        return self.wellbore_marker_list[idx - 1]
