"""WellboreMarkerFrame class."""

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.well
import resqpy.well._trajectory as rqt
import resqpy.well._wellbore_marker as rqwbm
import resqpy.well.well_utils as rqwu
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class WellboreMarkerFrame(BaseResqpy):
    """Class to handle RESQML WellBoreMarkerFrameRepresentation objects.

    note:
       measured depth data must be in same crs as those for the related trajectory
    """

    resqml_type = 'WellboreMarkerFrameRepresentation'

    boundary_feature_dict = {
        'GeologicBoundaryKind': ['fault', 'geobody', 'horizon'],
        'FluidMarker': ['gas down to', 'gas up to', 'oil down to', 'oil up to', 'water down to', 'water up to'],
        'FluidContact': ['free water contact', 'gas oil contact', 'gas water contact', 'seal', 'water oil contact']
    }

    def __init__(self,
                 parent_model,
                 uuid = None,
                 trajectory_uuid = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Creates a new wellbore marker frame object and optionally loads it from xml.

        arguments:
           parent_model (model.Model object): the model which the new wellbore marker frame belongs to
           uuid (uuid.UUID, optional): If given, loads from disk. Else, creates new.
           trajectory_uuid (uuid.UUID, optional): the uuid of the Trajectory object associated with the well;
              not used if uuid is not None
           title (str, optional): the citation title to use for a new wellbore marker frame;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the wellbore marker frame, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore marker frame;
              ignored if uuid is not None

        returns:
           the newly created wellbore marker frame object

        :meta common:
        """
        self.trajectory_uuid = trajectory_uuid
        self.node_count = None
        self.node_mds = None
        self.marker_list = None

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.trajectory_uuid is not None:
            self.trajectory = rqt.Trajectory(parent_model = self.model, uuid = self.trajectory_uuid)

    @classmethod
    def from_dataframe(cls,
                       parent_model,
                       trajectory_uuid,
                       dataframe,
                       md_col = 'MD',
                       boundary_feature_type_col = 'Boundary_Feature_Type',
                       marker_citation_title_col = 'Marker_Citation_Title',
                       interp_citation_title_col = 'Interp_Citation_Title',
                       create_organizing_objects_where_needed = True,
                       title = None,
                       originator = None,
                       extra_metadata = None):
        """Load wellbore marker frame data from a pandas data frame.

        arguments:
           parent_model (model.Model object): the model which the new blocked well belongs to
           dataframe: a pandas dataframe holding the wellbore marker frame data
           trajectory_uuid (uuid.UUID, optional): the uuid of the Trajectory object associated with the well;
           md_col (string, default 'MD'): the name of the column holding measured depth values
           boundary_feature_type_col (string, default 'Boundary_Feature_Type'): the name of the column holding the type of geologic,
            fluid or contact feature;
            e.g. "fault", "geobody", "horizon ", "gas/oil/water down to", "gas/oil/water up to",
            "free water contact", "gas oil contact", "gas water contact", "water oil contact", "seal"
           marker_citation_title_col (string, default 'Marker_Citation_Title'): the name of the column holding the marker citation title
           interp_citation_title_col (string, default 'Interp_Citation_Title'): the name of the column holding the interpretation
           create_organizing_objects_where_needed (bool, default True): if True, interpretation and feature objects will be created
              when needed if they do not already exist; if False, an exception will be raised if a required interpretation is missing
           citation title
           title (str, optional): the citation title to use for a new wellbore marker frame;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the wellbore marker frame, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore marker frame;
              ignored if uuid is not None

        returns:
            the newly created wellbore marker frame object
        """

        # the interpretation_citation_column is not a mandatory column in the dataframe as FluidContact and FluidMarker
        # boundary features do not have corresponding interpretations.
        for col in [md_col, boundary_feature_type_col, marker_citation_title_col]:
            assert col in dataframe.columns
            # assert that none of the values is blank
            assert dataframe[col].isnull().sum() == 0, f'blank value found in {col}'

        # verify that the boundary feature types specified are valid
        for i, boundary_feature in enumerate(dataframe[boundary_feature_type_col]):
            assert boundary_feature.lower() in (["fault", "geobody", "horizon", "gas down to", "oil down to", "water down to",
                                                 "gas up to", "oil up to", "water up to", "free water contact", "gas oil contact",
                                                 "gas water contact", "water oil contact", "seal"]),\
                f"invalid boundary feature type specified in row {i}"

        # create a wellbore marker object for each of the rows of the dataframe
        marker_list = []
        for i, row in dataframe.iterrows():
            row_index = i
            row_marker_type = row[boundary_feature_type_col].lower()
            row_interp_uuid = WellboreMarkerFrame.__parse_dataframe_row_for_interp_uuid(
                parent_model = parent_model,
                row = row,
                row_marker_type = row_marker_type,
                interp_citation_title_col = interp_citation_title_col,
                create_organizing_objects_where_needed = create_organizing_objects_where_needed)
            row_marker_citation_title = row[marker_citation_title_col]
            row_wellbore_marker_object = rqwbm.WellboreMarker(parent_model = parent_model,
                                                              parent_frame = WellboreMarkerFrame,
                                                              marker_index = row_index,
                                                              marker_type = row_marker_type,
                                                              interpretation_uuid = row_interp_uuid,
                                                              title = row_marker_citation_title)
            marker_list.append(row_wellbore_marker_object)

        wellbore_marker_frame = cls(parent_model = parent_model,
                                    trajectory_uuid = trajectory_uuid,
                                    title = title,
                                    originator = originator,
                                    extra_metadata = extra_metadata)
        wellbore_marker_frame.node_count = len(dataframe)
        wellbore_marker_frame.node_mds = np.array(dataframe['MD'])
        wellbore_marker_frame.marker_list = marker_list
        # check that the number of measured depths matches the node count and the number of markers
        assert wellbore_marker_frame.node_count == wellbore_marker_frame.node_mds.shape[0] == len(marker_list)

        return wellbore_marker_frame

    @staticmethod
    def __parse_dataframe_row_for_interp_uuid(parent_model, row, row_marker_type, interp_citation_title_col,
                                              create_organizing_objects_where_needed):
        """Extract the uuid of the interpretation object that the marker relates to from a row in the dataframe."""

        row_marker_type = row_marker_type.lower()
        if row_marker_type in ['horizon', 'fault', 'geobody']:
            row_interp_type = 'obj_' + row_marker_type.capitalize() + 'Interpretation'
        else:
            row_interp_type = None
        try:
            if row_interp_type is not None:
                row_interp_citation_title = row[interp_citation_title_col]
                row_interp_uuid = parent_model.uuid(obj_type = row_interp_type, title = row_interp_citation_title)
                if row_interp_uuid is None:
                    if create_organizing_objects_where_needed:
                        row_interp_uuid = WellboreMarkerFrame._create_feature_and_interpretation(
                            parent_model, row_marker_type, row_interp_citation_title)
                    else:
                        raise ValueError(f'interpretation uuid cannot be found for title {row_interp_citation_title}')
            else:
                # no interpretation exists for the boundary feature
                row_interp_uuid = None
        except KeyError:  # no interpretation column in the dataframe
            row_interp_uuid = None

        return row_interp_uuid

    @staticmethod
    def _create_feature_and_interpretation(model, feature_type, feature_name):
        # includes create_xml() for created organisational objects
        feature_type = feature_type.lower()
        assert feature_type in ['fault', 'horizon', 'geobody']
        if feature_type == 'fault':
            tbf = rqo.TectonicBoundaryFeature(model, kind = 'fault', feature_name = feature_name)
        else:
            tbf = rqo.GeneticBoundaryFeature(model, kind = feature_type, feature_name = feature_name)
        tbf.create_xml(reuse = True)
        if feature_type == 'fault':
            fi = rqo.FaultInterpretation(model, tectonic_boundary_feature = tbf, is_normal = True)
            # todo: set is_normal correctly
        elif feature_type == 'horizon':
            fi = rqo.HorizonInterpretation(model, genetic_boundary_feature = tbf)
            # todo: support boundary relation list and sequence stratigraphy surface
        else:  # geobody boundary
            fi = rqo.GeobodyBoundaryInterpretation(model, genetic_boundary_feature = tbf)
        fi.create_xml(reuse = True)
        return fi.uuid

    def _load_from_xml(self):
        """Loads the wellbore marker frame object from an xml node (and associated hdf5 data).

        note:
           this method is not usually called directly
        """

        wellbore_marker_frame_root = self.root
        assert wellbore_marker_frame_root is not None

        self.trajectory_uuid = bu.uuid_from_string(
            rqet.find_nested_tags_text(wellbore_marker_frame_root, ['Trajectory', 'UUID']))
        self.trajectory = rqt.Trajectory(parent_model = self.model, uuid = self.trajectory_uuid)

        # list of Wellbore markers
        self.marker_list = []
        for i, tag in enumerate(rqet.list_of_tag(wellbore_marker_frame_root, 'WellboreMarker')):
            marker_obj = rqwbm.WellboreMarker(parent_model = self.model,
                                              parent_frame = self,
                                              marker_index = i,
                                              marker_node = tag)
            self.marker_list.append(marker_obj)

        self.node_count = rqet.find_tag_int(wellbore_marker_frame_root, 'NodeCount')
        rqwu.load_hdf5_array(self, rqet.find_tag(wellbore_marker_frame_root, 'NodeMd'), "node_mds", tag = 'Values')

        assert self.node_count == len(self.node_mds), 'node count does not match hdf5 array'
        assert len(self.marker_list) == self.node_count, 'wellbore marker list does not contain correct node count'

    def dataframe(self):
        """Returns a pandas dataframe with columns X, Y, Z, MD, Boundary_Feature_Type, Marker_Citation_Title,

        Interp_Citation_Title.
        """

        xyz = np.empty((self.node_count, 3))
        boundary_feature_type_list = []
        marker_citation_title_list = []
        interp_citation_title_list = []

        for i, marker_obj in enumerate(self.marker_list):
            boundary_feature_type = marker_obj.marker_type
            marker_citation_title = marker_obj.title
            interp_uuid = marker_obj.interpretation_uuid
            if interp_uuid is not None:
                interp_root = self.model.root_for_uuid(interp_uuid)
                interp_citation_title = self.model.title_for_root(root = interp_root)
            else:
                interp_citation_title = None
            xyz[i] = self.trajectory.xyz_for_md(self.node_mds[i])
            boundary_feature_type_list.append(boundary_feature_type)
            marker_citation_title_list.append(marker_citation_title)
            interp_citation_title_list.append(interp_citation_title)

        return pd.DataFrame({
            'X': xyz[:, 0],
            'Y': xyz[:, 1],
            'Z': xyz[:, 2],
            'MD': self.node_mds,
            'Boundary_Feature_Type': boundary_feature_type_list,
            'Marker_Citation_Title': marker_citation_title_list,
            'Interp_Citation_Title': interp_citation_title_list
        })

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   title = 'wellbore marker framework',
                   originator = None):
        """Creates the xml tree for this wellbore marker frame and optionally adds as a part to the model."""

        assert type(add_as_part) is bool

        if not self.title:
            self.title = title

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()

        wbm_node = super().create_xml(originator = originator, add_as_part = False)
        node_count, nodeMd, md_values_node = self.__add_sub_elements_to_root_node(wbm_node = wbm_node)
        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Mds', root = md_values_node)

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
        if self.marker_list is not None:
            for marker in self.marker_list:
                title = self.__get_wbm_node_title(marker)
                wbm_node_obj = marker.create_xml(parent_node = wbm_node, title = title)
                assert wbm_node_obj is not None

        # add as part
        self.__add_as_part_and_add_relationships(wbm_node = wbm_node,
                                                 ext_uuid = ext_uuid,
                                                 add_as_part = add_as_part,
                                                 add_relationships = add_relationships)

        return wbm_node

    def __get_wbm_node_title(self, marker):
        """ Generate the title of the newly created wellbore marker object node."""

        well_name = self.trajectory.well_name
        for k, v in WellboreMarkerFrame.boundary_feature_dict.items():
            if marker.marker_type in v:
                boundary_feature = k
                break
        interp_title = self.model.title_for_root(root = self.model.root_for_uuid(uuid = marker.interpretation_uuid))
        title = ' '.join([x for x in [well_name, boundary_feature, marker.marker_type, interp_title] if x is not None])
        return title

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

        and add reciprocal relationships.
        """

        if add_as_part:
            self.model.add_part('obj_WellboreMarkerFrameRepresentation', self.uuid, wbm_node)

            if add_relationships:
                self.model.create_reciprocal_relationship(wbm_node, 'destinationObject', self.trajectory.root,
                                                          'sourceObject')
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(wbm_node, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')
                if self.marker_list is not None:  # TODO: confirm that you can have an empty wellbore marker frame with no marker list
                    for marker in self.marker_list:
                        interp_root = self.model.root_for_uuid(uuid = marker.interpretation_uuid)
                        self.model.create_reciprocal_relationship(wbm_node, 'destinationObject', interp_root,
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

    def find_marker_index_from_interp(
            self, interpretation_uuid):  # TODO: return marker index from interp uuid, else return None
        """Find wellbore marker index by interpretation uuid.

        arguments:
           interpretation_uuid (uuid.UUID or string): the uuid of the interpretation object of interest

        returns:
           integer indicating the associated marker's position in self.marker_list

        note:
           if no marker is found for the interpretation uuid, None is returned
        """

        if type(interpretation_uuid) is str:
            interpretation_uuid = bu.uuid_from_string(interpretation_uuid)

        for i, marker in enumerate(self.marker_list):
            if bu.matching_uuids(marker.interpretation_uuid, interpretation_uuid):
                return i

        return None

    def find_marker_from_index(self, idx):
        """Returns wellbore marker by index.

        arguments:
           idx (int): position of the marker in the wellbore marker list

        returns:
           marker (WellboreMarker object)
        """
        try:
            found_marker = self.marker_list[idx]
        except IndexError:
            found_marker = None

        return found_marker
