"""_grid_connection_set.py: Module providing RESQML grid connection set class."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np
import pandas as pd

import resqpy.fault
import resqpy.olio.read_nexus_fault as rnf
import resqpy.olio.trademark as tm
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.property as rqp
import resqpy.surface as rqs
import resqpy.crs as rqc
import resqpy.fault._gcs_functions as rqf_gf
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns

valid_interpretation_types = [
    'obj_FaultInterpretation', 'obj_HorizonInterpretation', 'obj_GeobodyBoundaryInterpretation'
]


class GridConnectionSet(BaseResqpy):
    """Class for obj_GridConnectionSetRepresentation holding pairs of connected faces, usually for faults."""

    resqml_type = 'GridConnectionSetRepresentation'

    def __init__(self,
                 parent_model,
                 uuid = None,
                 cache_arrays = False,
                 find_properties = True,
                 grid = None,
                 ascii_load_format = None,
                 ascii_file = None,
                 k_faces = None,
                 j_faces = None,
                 i_faces = None,
                 k_sides = None,
                 j_sides = None,
                 i_sides = None,
                 feature_name = None,
                 feature_type = 'fault',
                 create_organizing_objects_where_needed = False,
                 create_transmissibility_multiplier_property = True,
                 fault_tmult_dict = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Initializes a new GridConnectionSet.
        
        Optionally loads it from xml or a list of simulator format ascii files.

        arguments:
           parent_model (model.Model object): the resqml model that this grid connection set will be part of
           uuid (uuid.UUID, optional): the uuid of an existing RESQML GridConnectionSetRepresentation from which
                 this resqpy object is populated
           find_properties (boolean, default True): if True and uuid is present, the property collection
                 relating to the grid connection set is prepared
           grid (grid.Grid object, optional): If present, the grid object that this connection set relates to;
                 if absent, the main grid for the parent model is assumed; only used if uuid is None; see also notes
           ascii_load_format (string, optional): If present, must be 'nexus'; ignored if loading from xml;
                 otherwise required if ascii_file is present
           ascii_file (string, optional): the full path of an ascii file holding fault definition data in
                 nexus keyword format; ignored if loading from xml; otherwise, if present, ascii_load_format
                 must also be set
           k_faces, j_faces, i_faces (boolean arrays, optional): if present, these arrays are used to identify
                 which faces between logically neighbouring cells to include in the new grid connection set
           k_sides, j_sides, i_sides (boolean arrays, optional): if present, and k_faces etc are present, these
                 arrays are used to determine which side of the cell face should appear as the first in the pairing
           feature_name (string, optional): the feature name to use when setting from faces
           feature_type (string, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
           create_organizing_objects_where_needed (boolean, default False): if True when loading from ascii or
                 face masks, a fault interpretation object and tectonic boundary feature object will be created
                 for any named fault for which such objects do not exist; if False, missing organizational objects
                 will cause an error to be logged; ignored when loading from xml
           create_transmissibility_multiplier_property (boolean, default True): if True when loading from ascii,
                 a transmissibility multiplier property is created for the connection set
           fault_tmult_dict (dict of str: float): optional dictionary mapping fault name to a transmissibility
                 multiplier; only used if initialising from ascii and creating a multiplier property
           title (str, optional): the citation title to use for a new grid connection set;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the new grid connection set, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the grid connection set
              ignored if uuid is not None

        returns:
           a new GridConnectionSet object, initialised from xml or ascii file, or left empty

        notes:
           in the resqml standard, a grid connection set can be used to identify connections between different
           grids (eg. parent grid to LGR); however, many of the methods in this code currently only handle
           connections within a single grid;
           when loading from an ascii file, cell faces are paired on a simplistic logical neighbour basis (as if
           there is no throw on the fault); this is because the simulator input does not include the full
           juxtaposition information; the simple mode is adequate for identifying which faces are involved in a
           fault but not for matters of juxtaposition or absolute transmissibility calculations;
           if uuid is None and ascii_file is None and k_faces, j_faces & i_faces are None,
           then an empty connection set is returned;
           if a transmissibility multiplier property is generated, it will only appear in the property collection
           for the grid connection set after the create_xml() method has been called

        :meta common:
        """

        log.debug('initialising grid connection set')
        self.count = None  #: number of face-juxtaposition pairs in this connection set
        self.cell_index_pairs = None  #: shape (count, 2); dtype int; index normalized for flattened array
        self.cell_index_pairs_null_value = -1  #: integer null value for array above
        self.grid_index_pairs = None  #: shape (count, 2); dtype int; optional; used if more than one grid referenced
        self.face_index_pairs = None  #: shape (count, 2); dtype int32; local to cell, ie. range 0 to 5
        self.face_index_pairs_null_value = -1  #: integer null value for array above
        # NB face index values 0..5 usually mean [K-, K+, J+, I+, J-, I-] respectively but there is some ambiguity
        #    over I & J in the Energistics RESQML Usage Guide; see comments in DevOps backlog item 269001 for more info
        self.grid_list = []  #: ordered list of grid objects, indexed by grid_index_pairs
        self.grid_index_pairs_null_value = -1
        self.feature_indices = None  #: shape (count,); dtype int; optional; which fault interpretation each pair is part of
        # note: resqml data structures allow a face pair to be part of more than one fault but this code restricts to one
        self.feature_list = None  #: ordered list, actually of interpretations, indexed by feature_indices
        # feature list contains tuples: (content_type, uuid, title) for fault features (or other interpretations)
        self.property_collection = None  #: optional property.PropertyCollection

        # NB: RESQML documentation is not clear which order is correct; should be kept consistent with same data in property.py
        # face_index_map maps from (axis, p01) to face index value in range 0..5
        # this is the default as indicated on page 139 (but not p. 180) of the RESQML Usage Gude v2.0.1
        # also assumes K is generally increasing downwards
        # see DevOps backlog item 269001 discussion for more information
        #     self.face_index_map = np.array([[0, 1], [4, 2], [5, 3]], dtype = int)
        self.face_index_map = np.array([[0, 1], [2, 4], [5, 3]], dtype = int)  # order: top, base, J-, I+, J+, I-
        # and the inverse, maps from 0..5 to (axis, p01)
        #     self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 1], [2, 1], [1, 0], [2, 0]], dtype = int)
        self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [1, 1], [2, 0]], dtype = int)
        # note: the rework_face_pairs() method, below, overwrites the face indices based on I, J cell indices
        if not title:
            title = feature_name

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is None:
            log.debug('setting grid for new connection set to default (ROOT)')
            if grid is None:
                grid = self.model.grid(find_properties = False)  # should find only or ROOT grid
                assert grid is not None, 'No ROOT grid found in model'
            self.grid_list = [grid]
            if ascii_load_format and ascii_file:
                faces = None
                if ascii_load_format == 'nexus':
                    log.debug('loading connection set (fault) faces from Nexus format ascii file: ' + ascii_file)
                    tm.log_nexus_tm('debug')
                    faces = rnf.load_nexus_fault_mult_table(ascii_file)
                else:
                    log.warning('ascii format for connection set faces not handled by base resqpy code: ' +
                                ascii_load_format)
                assert faces is not None, 'failed to load fault face information from file: ' + ascii_file
                self.set_pairs_from_faces_df(
                    faces,
                    create_organizing_objects_where_needed = create_organizing_objects_where_needed,
                    create_mult_prop = create_transmissibility_multiplier_property,
                    fault_tmult_dict = fault_tmult_dict,
                    one_based_indexing = True)
                # note: hdf5 write and xml generation elsewhere
                # todo: optionally set face sets in grid object? or add to existing set?
                # grid.make_face_set_from_dataframe(faces) to set from dataframe
                # to access pair of lists for j & i plus faces, from grid attibutes, eg:
                # for fs in grid.face_set_dict.keys():
                #    list_pair = grid.face_set_dict[fs][0:2]
                #    face_sets.append(str(fs) + ': ' + str(len(list_pair[1])) + ' i face cols; ' +
                #                                      str(len(list_pair[0])) + ' j face cols; from ' + origin)
                # note: grid structures assume curtain like set of kelp
            elif k_faces is not None or j_faces is not None or i_faces is not None:
                self.set_pairs_from_face_masks(k_faces,
                                               j_faces,
                                               i_faces,
                                               feature_name,
                                               create_organizing_objects_where_needed,
                                               feature_type = feature_type,
                                               k_sides = k_sides,
                                               j_sides = j_sides,
                                               i_sides = i_sides)
        else:
            if cache_arrays:
                self.cache_arrays()
            if find_properties:
                self.extract_property_collection()

    def _load_from_xml(self):
        root = self.root
        assert root is not None
        self.count = rqet.find_tag_int(root, 'Count')
        assert self.count > 0, 'empty grid connection set'
        self.cell_index_pairs_null_value = rqet.find_nested_tags_int(root, ['CellIndexPairs', 'NullValue'])
        self.face_index_pairs_null_value = rqet.find_nested_tags_int(root, ['LocalFacePerCellIndexPairs', 'NullValue'])
        # postpone loading of hdf5 array data till on-demand load (cell, grid & face index pairs)
        self.grid_list = []
        for child in root:
            if rqet.stripped_of_prefix(child.tag) != 'Grid':
                continue
            grid_type = rqet.content_type(rqet.find_tag_text(child, 'ContentType'))
            grid_uuid = bu.uuid_from_string(rqet.find_tag_text(child, 'UUID'))
            assert grid_type == 'obj_IjkGridRepresentation', 'only IJK grids supported for grid connection sets'
            grid = self.model.grid(uuid = grid_uuid,
                                   find_properties = False)  # centralised list of grid objects for shared use
            self.grid_list.append(grid)
        # following code only needed to handle defective datasets generated by earlier versions!
        if len(self.grid_list) == 0:
            log.warning('no grid nodes found in xml for connection set')
            grid = self.model.grid(find_properties = False)  # should find only or ROOT grid
            assert grid is not None, 'No ROOT grid found in model'
            self.grid_list = [grid]
        # interpretations (referred to here as feature list) are optional
        interp_root = rqet.find_tag(root, 'ConnectionInterpretations')
        if interp_root is None:
            return
        # load ordered feature list
        self.feature_list = []  # ordered list of (content_type, uuid, title) for faults
        for child in interp_root:
            if rqet.stripped_of_prefix(child.tag) != 'FeatureInterpretation':
                continue
            feature_type = rqet.content_type(rqet.find_tag_text(child, 'ContentType'))
            feature_uuid = bu.uuid_from_string(rqet.find_tag_text(child, 'UUID'))
            feature_title = rqet.find_tag_text(child, 'Title')
            assert feature_type in valid_interpretation_types, \
               f'unsupported type {feature_type} for gcs feature interpretation'
            self.feature_list.append((feature_type, feature_uuid, feature_title))
            log.debug(f'connection set references interpretation: {feature_title}; of type: {feature_type}')
        log.debug('number of features referred to in connection set: ' + str(len(self.feature_list)))
        assert len(self.feature_list) > 0, 'list of interpretation references is empty for connection set'
        # leave feature indices till on-demand load

    def extract_property_collection(self):
        """Prepares the property collection for this grid connection set."""

        if self.property_collection is None:
            self.property_collection = rqp.PropertyCollection(support = self)
        return self.property_collection

    def set_pairs_from_kelp(self,
                            kelp_0,
                            kelp_1,
                            feature_name,
                            create_organizing_objects_where_needed,
                            axis = 'K',
                            feature_type = 'fault'):
        """Set cell_index_pairs and face_index_pairs based on j and i face kelp strands.
        
        Uses simple no throw pairing.
        """
        # note: this method has been reworked to allow 'kelp' to be 'horizontal' strands when working in cross section

        if axis == 'K':
            kelp_k = None
            kelp_j = kelp_0
            kelp_i = kelp_1
        elif axis == 'J':
            kelp_k = kelp_0
            kelp_j = None
            kelp_i = kelp_1
        else:
            assert axis == 'I'
            kelp_k = kelp_0
            kelp_j = kelp_1
            kelp_i = None

        if feature_name is None:
            feature_name = 'feature from kelp lists'
        if len(self.grid_list) > 1:
            log.warning('setting grid connection set pairs from kelp for first grid in list only')
        grid = self.grid_list[0]
        if grid.nk > 1 and kelp_k is not None and len(kelp_k) > 0:
            if axis == 'J':
                k_layer = np.zeros((grid.nk - 1, grid.ni), dtype = bool)
            else:
                k_layer = np.zeros((grid.nk - 1, grid.nj), dtype = bool)
            kelp_a = np.array(kelp_k, dtype = int).T
            k_layer[kelp_a[0], kelp_a[1]] = True
            k_faces = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
            if axis == 'J':
                k_faces[:] = k_layer.reshape((grid.nk - 1, 1, grid.ni))
            else:
                k_faces[:] = k_layer.reshape((grid.nk - 1, grid.nj, 1))
        else:
            k_faces = None
        if grid.nj > 1 and kelp_j is not None and len(kelp_j) > 0:
            if axis == 'K':
                j_layer = np.zeros((grid.nj - 1, grid.ni), dtype = bool)
            else:
                j_layer = np.zeros((grid.nk, grid.nj - 1), dtype = bool)
            kelp_a = np.array(kelp_j, dtype = int).T
            j_layer[kelp_a[0], kelp_a[1]] = True
            j_faces = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
            if axis == 'K':
                j_faces[:] = j_layer.reshape((1, grid.nj - 1, grid.ni))
            else:
                j_faces[:] = j_layer.reshape((grid.nk, grid.nj - 1, 1))
        else:
            j_faces = None
        if grid.ni > 1 and kelp_i is not None and len(kelp_i) > 0:
            if axis == 'K':
                i_layer = np.zeros((grid.nj, grid.ni - 1), dtype = bool)
            else:
                i_layer = np.zeros((grid.nk, grid.ni - 1), dtype = bool)
            kelp_a = np.array(kelp_i, dtype = int).T
            i_layer[kelp_a[0], kelp_a[1]] = True
            i_faces = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
            if axis == 'K':
                i_faces[:] = i_layer.reshape((1, grid.nj, grid.ni - 1))
            else:
                i_faces[:] = i_layer.reshape((grid.nk, 1, grid.ni - 1))
        else:
            i_faces = None
        self.set_pairs_from_face_masks(k_faces,
                                       j_faces,
                                       i_faces,
                                       feature_name,
                                       create_organizing_objects_where_needed,
                                       feature_type = feature_type)

    def set_pairs_from_face_masks(
            self,
            k_faces,
            j_faces,
            i_faces,
            feature_name,
            create_organizing_objects_where_needed,
            feature_type = 'fault',  # other feature_type values: 'horizon', 'geobody boundary'
            k_sides = None,
            j_sides = None,
            i_sides = None):
        """Sets cell_index_pairs and face_index_pairs based on triple face masks, using simple no throw pairing."""

        assert feature_type in ['fault', 'horizon', 'geobody boundary']
        if feature_name is None:
            feature_name = 'feature from face masks'  # not sure this default is wise
        if len(self.grid_list) > 1:
            log.warning('setting grid connection set pairs from face masks for first grid in list only')
        grid = self.grid_list[0]
        if feature_type == 'fault':
            feature_flavour = 'TectonicBoundaryFeature'
            interpretation_flavour = 'FaultInterpretation'
        else:
            feature_flavour = 'GeneticBoundaryFeature'
            if feature_type == 'horizon':
                interpretation_flavour = 'HorizonInterpretation'
            else:
                interpretation_flavour = 'GeobodyBoundaryInterpretation'
            # kind differentiates between horizon and geobody boundary
        fi_parts_list = self.model.parts_list_of_type(interpretation_flavour)
        if (fi_parts_list is None or len(fi_parts_list) == 0) and not create_organizing_objects_where_needed:
            log.warning('no interpretation parts found in model for ' + feature_type)
        fi_uuid = None
        for fi_part in fi_parts_list:
            fi_title = self.model.title_for_part(fi_part)
            if fi_title == feature_name or fi_title.split()[0].lower() == feature_name.lower():
                fi_uuid = self.model.uuid_for_part(fi_part)
                break
        if fi_uuid is None:
            if create_organizing_objects_where_needed:
                tbf_parts_list = self.model.parts_list_of_type(feature_flavour)
                tbf = None
                for tbf_part in tbf_parts_list:
                    tbf_title = self.model.title_for_part(tbf_part)
                    if feature_name == tbf_title or feature_name.lower() == tbf_title.split()[0].lower():
                        tbf_uuid = self.model.uuid_for_part(tbf_part)
                        if feature_type == 'fault':
                            tbf = rqo.TectonicBoundaryFeature(self.model, uuid = tbf_uuid)
                        else:
                            tbf = rqo.GeneticBoundaryFeature(self.model, kind = feature_type, uuid = tbf_uuid)
                        break
                if tbf is None:
                    if feature_type == 'fault':
                        tbf = rqo.TectonicBoundaryFeature(self.model, kind = 'fault', feature_name = feature_name)
                    else:
                        tbf = rqo.GeneticBoundaryFeature(self.model, kind = feature_type, feature_name = feature_name)
                    tbf_root = tbf.create_xml()
                else:
                    tbf_root = tbf.root
                if feature_type == 'fault':
                    fi = rqo.FaultInterpretation(
                        self.model, tectonic_boundary_feature = tbf,
                        is_normal = True)  # todo: set is_normal based on fault geometry in grid?
                elif feature_type == 'horizon':
                    fi = rqo.HorizonInterpretation(self.model, genetic_boundary_feature = tbf)
                    # todo: support boundary relation list and sequence stratigraphy surface
                else:  # geobody boundary
                    fi = rqo.GeobodyBoundaryInterpretation(self.model, genetic_boundary_feature = tbf)
                fi_root = fi.create_xml(tbf_root)
                fi_uuid = rqet.uuid_for_part_root(fi_root)
            else:
                log.error('no interpretation found for feature: ' + feature_name)
                return
        self.feature_list = [('obj_' + interpretation_flavour, fi_uuid, str(feature_name))]
        cell_pair_list = []
        face_pair_list = []
        nj_ni = grid.nj * grid.ni
        if k_faces is not None:
            if k_sides is None:
                k_sides = np.zeros(k_faces.shape, dtype = bool)
            for cell_kji0, flip in zip(np.stack(np.where(k_faces)).T, k_sides[np.where(k_faces)]):
                cell = grid.natural_cell_index(cell_kji0)
                if flip:
                    cell_pair_list.append((cell + nj_ni, cell))
                    face_pair_list.append((self.face_index_map[0, 0], self.face_index_map[0, 1]))
                else:
                    cell_pair_list.append((cell, cell + nj_ni))
                    face_pair_list.append((self.face_index_map[0, 1], self.face_index_map[0, 0]))
        if j_faces is not None:
            if j_sides is None:
                j_sides = np.zeros(j_faces.shape, dtype = bool)
            for cell_kji0, flip in zip(np.stack(np.where(j_faces)).T, j_sides[np.where(j_faces)]):
                cell = grid.natural_cell_index(cell_kji0)
                if flip:
                    cell_pair_list.append((cell + grid.ni, cell))
                    face_pair_list.append((self.face_index_map[1, 0], self.face_index_map[1, 1]))
                else:
                    cell_pair_list.append((cell, cell + grid.ni))
                    face_pair_list.append((self.face_index_map[1, 1], self.face_index_map[1, 0]))
        if i_faces is not None:
            if i_sides is None:
                i_sides = np.zeros(i_faces.shape, dtype = bool)
            for cell_kji0, flip in zip(np.stack(np.where(i_faces)).T, i_sides[np.where(i_faces)]):
                cell = grid.natural_cell_index(cell_kji0)
                if flip:
                    cell_pair_list.append((cell + 1, cell))
                    face_pair_list.append((self.face_index_map[2, 0], self.face_index_map[2, 1]))
                else:
                    cell_pair_list.append((cell, cell + 1))
                    face_pair_list.append((self.face_index_map[2, 1], self.face_index_map[2, 0]))
        self.cell_index_pairs = np.array(cell_pair_list, dtype = int)
        self.face_index_pairs = np.array(face_pair_list, dtype = int)
        self.count = len(self.cell_index_pairs)
        self.feature_indices = np.zeros(self.count, dtype = int)
        assert len(self.face_index_pairs) == self.count

    def set_pairs_from_faces_df(self,
                                faces,
                                create_organizing_objects_where_needed = False,
                                create_mult_prop = True,
                                fault_tmult_dict = None,
                                one_based_indexing = True):
        """Sets cell_index_pairs and face_index_pairs based on pandas dataframe, using simple no throw pairing.

        arguments:
           faces (pandas.DataFrame): dataframe with columns 'name', 'face', 'i1', 'i2', 'j1', 'j2', 'k1', 'k2', 'mult'
           create_organizing_objects_where_needed (bool, default False): if True, interpretation and
              feature objects are created (including xml creation) where needed
           create_mult_prop (bool, default True): if True, a transmissibility multiplier property is added to the
              collection for this grid connection set
           fault_tmult_dict (dict of str: float): optional dictionary mapping fault name to a transmissibility
                 multiplier; if present, is combined with multiplier values from the dataframe
           one_based_indexing (bool, default True): if True, the i, j & k values in the dataframe are taken to be
              in simulator protocol and 1 is subtracted to yield the RESQML cell indices

        notes:
           as a side effect, this method will set the cell indices in faces to be zero based;
           this method currently assumes fault interpretation (not horizon or geobody boundary)
        """

        if len(self.grid_list) > 1:
            log.warning('setting grid connection set pairs from dataframe for first grid in list only')
        grid = self.grid_list[0]
        rqf_gf.standardize_face_indicator_in_faces_df(faces)
        if one_based_indexing:
            rqf_gf.zero_base_cell_indices_in_faces_df(faces)
        faces = rqf_gf.remove_external_faces_from_faces_df(faces, self.grid_list[0].extent_kji)
        self.feature_list = []
        cell_pair_list = []
        face_pair_list = []
        fi_list = []
        feature_index = 0
        name_list = faces['name'].unique()
        fi_parts_list = self.model.parts_list_of_type('FaultInterpretation')
        if not create_organizing_objects_where_needed and (fi_parts_list is None or len(fi_parts_list) == 0):
            log.warning('no fault interpretation parts found in model')
        fi_dict = {}  # maps fault name to interpretation uuid
        mult_list = []
        const_mult = True
        for fi_part in fi_parts_list:
            fi_dict[self.model.title_for_part(fi_part).split()[0].lower()] = self.model.uuid_for_part(fi_part)
        if create_organizing_objects_where_needed:
            tbf_parts_list = self.model.parts_list_of_type('TectonicBoundaryFeature')

        for name in name_list:
            success, const_mult = self._set_pairs_from_faces_df_for_named_fault(
                feature_index, faces, name, fault_tmult_dict, fi_dict, create_organizing_objects_where_needed,
                tbf_parts_list, fi_list, cell_pair_list, face_pair_list, const_mult, mult_list, grid, create_mult_prop)
            if success:
                feature_index += 1

        self.feature_indices = np.array(fi_list, dtype = int)
        self.cell_index_pairs = np.array(cell_pair_list, dtype = int)
        self.face_index_pairs = np.array(face_pair_list, dtype = int)
        self.count = len(self.cell_index_pairs)
        assert len(self.face_index_pairs) == self.count
        if create_mult_prop and self.count > 0:
            self._create_multiplier_property(mult_list, const_mult)

    def write_hdf5_and_create_xml_for_new_properties(self):
        """Wites any new property arrays to hdf5, creates xml for the properties and adds them to model.

        note:
           this method is usually called by the create_xml() method for the grid connection set
        """

        if self.property_collection is not None:
            self.property_collection.write_hdf5_for_imported_list()
            self.property_collection.create_xml_for_imported_list_and_add_parts_to_model()

    def append(self, other):
        """Adds the features in other grid connection set to this one."""

        # todo: check that grid is common to both connection sets
        assert len(self.grid_list) == 1 and len(other.grid_list) == 1, 'attempt to merge multi-grid connection sets'
        assert bu.matching_uuids(self.grid_list[0].uuid, other.grid_list[0].uuid),  \
              'attempt to merge grid connection sets from different grids'
        if self.count is None or self.count == 0:
            self.feature_list = other.feature_list.copy()
            self.count = other.count
            self.feature_indices = other.feature_indices.copy()
            self.cell_index_pairs = other.cell_index_pairs.copy()
            self.face_index_pairs = other.face_index_pairs.copy()
        else:
            feature_offset = len(self.feature_list)
            self.feature_list += other.feature_list
            combined_feature_indices = np.concatenate((self.feature_indices, other.feature_indices))
            combined_feature_indices[self.count:] += feature_offset
            combined_cell_index_pairs = np.concatenate((self.cell_index_pairs, other.cell_index_pairs))
            combined_face_index_pairs = np.concatenate((self.face_index_pairs, other.face_index_pairs))
            self.count += other.count
            self.cell_index_pairs = combined_cell_index_pairs
            self.face_index_pairs = combined_face_index_pairs
            self.feature_indices = combined_feature_indices
        self.uuid = bu.new_uuid()

    def single_feature(self, feature_index = None, feature_name = None):
        """Returns a new GridConnectionSet object containing the single feature copied from this set.

        note:
           the single feature connection set is established in memory but this method does not write
           to the hdf5 nor create xml or add as a part to the model

        :meta common:
        """

        assert feature_index is not None or feature_name is not None

        if feature_index is None:
            feature_index, _ = self.feature_index_and_uuid_for_fault_name(feature_name)
        if feature_index is None:
            return None
        if feature_index < 0 or feature_index >= len(self.feature_list):
            return None
        singleton = GridConnectionSet(self.model, grid = self.grid_list[0])
        singleton.cell_index_pairs, singleton.face_index_pairs =  \
           self.raw_list_of_cell_face_pairs_for_feature_index(feature_index)
        singleton.count = singleton.cell_index_pairs.shape[0]
        singleton.feature_indices = np.zeros((singleton.count,), dtype = int)
        singleton.feature_list = [self.feature_list[feature_index]]
        return singleton

    def filtered_by_layer_range(self, min_k0 = None, max_k0 = None, pare_down = True, return_indices = False):
        """Returns a new GridConnectionSet, being a copy with cell faces whittled down to a layer range.

        arguments:
           min_k0 (int, optional): if present, the minimum layer number to be included (zero based)
           max_k0 (int, optional): if present, the maximum layer number to be included (zero based)
           pare_down (bool, default True): if True, any unused features in the new grid connection set will be removed
              and the feature indices adjusted appropriately; if False, unused features will be left in the list for
              the new connection set, meaning that the feature indices will be compatible with those for self
           return_indices (bool, default False): if True, a numpy list of the selected indices is also returned (see notes)

        returns:
           a new GridConnectionSet or (GridConnectionSet, numpy int array of shape (N,)) depending on return_indices argument,
           where the array is a list of selected indices (see notes)

        notes:
           cells in layer max_k0 are included in the filtered set (not pythonesque);
           currently only works for single grid connection sets;
           if return_indices is True, a second item is returned which is a 1D numpy int array holding the indices of
           cell face pairs that have been selected from the original grid connection set; these values can be used to
           select equivalent entries from associated properties
        """

        self.cache_arrays()
        assert len(self.grid_list) == 1, 'attempt to filter multi-grid connection set by layer range'
        grid = self.grid_list[0]
        if min_k0 <= 0:
            min_k0 = None
        if max_k0 >= grid.extent_kji[0] - 1:
            max_k0 = None
        if min_k0 is None and max_k0 is None:
            dupe = GridConnectionSet(self.model, grid = grid)
            dupe.append(self)
            return (dupe, np.arange(self.count)) if return_indices else dupe
        mask = np.zeros(grid.extent_kji, dtype = bool)
        if min_k0 is not None and max_k0 is not None:
            mask[min_k0:max_k0 + 1, :, :] = True
        elif min_k0 is not None:
            mask[min_k0:, :, :] = True
        else:
            mask[:max_k0 + 1, :, :] = True
        return self.filtered_by_cell_mask(mask, pare_down = pare_down, return_indices = return_indices)

    def filtered_by_cell_mask(self, mask, both_cells_required = True, pare_down = True, return_indices = False):
        """Returns a new GridConnectionSet, being a copy with cell faces whittled down by a boolean mask array.

        arguments:
           mask (numpy bool array of shape grid.extent_kji): connections will be kept for cells where this mask is True
           both_cells_required (bool, default True): if True, both cells involved in a connection must have a mask value
              of True to be included; if False, any connection where either cell has a True mask value will be included
           pare_down (bool, default True): if True, any unused features in the new grid connection set will be removed
              and the feature indices adjusted appropriately; if False, unused features will be left in the list for
              the new connection set, meaning that the feature indices will be compatible with those for self
           return_indices (bool, default False): if True, a numpy list of the selected indices is also returned (see notes)

        returns:
           a new GridConnectionSet or (GridConnectionSet, numpy int array of shape (N,)) depending on return_indices argument,
           where the array is a list of selected indices (see notes)

        note:
           currently only works for single grid connection sets;
           if return_indices is True, a second item is returned which is a 1D numpy int array holding the indices of
           cell face pairs that have been selected from the original grid connection set; these values can be used to
           select equivalent entries from associated properties
        """

        assert len(self.grid_list) == 1, 'attempt to filter multi-grid connection set by cell mask'
        grid = self.grid_list[0]
        flat_extent = grid.cell_count()
        flat_mask = mask.reshape((flat_extent,))
        where_0 = flat_mask[self.cell_index_pairs[:, 0]]
        where_1 = flat_mask[self.cell_index_pairs[:, 1]]
        if both_cells_required:
            where_both = np.logical_and(where_0, where_1)
        else:
            where_both = np.logical_or(where_0, where_1)
        indices = np.where(where_both)[0]  # indices into primary axis of original arrays
        if len(indices) == 0:
            log.warning('no connections have passed filtering')
            return (None, None) if return_indices else None
        masked_gcs = GridConnectionSet(self.model, grid = grid)
        masked_gcs.count = len(indices)
        masked_gcs.cell_index_pairs = self.cell_index_pairs[indices, :]
        masked_gcs.face_index_pairs = self.face_index_pairs[indices, :]
        masked_gcs.feature_indices = self.feature_indices[indices]
        masked_gcs.feature_list = self.feature_list.copy()
        if pare_down:
            masked_gcs.clean_feature_list()
        return (masked_gcs, indices) if return_indices else masked_gcs

    def clean_feature_list(self):
        """Removes any features that have no associated connections."""

        keep_list = np.unique(self.feature_indices)
        if len(keep_list) == len(self.feature_list):
            return
        cleaned_list = []
        for i in range(len(keep_list)):
            assert i <= keep_list[i]
            if i != keep_list[i]:
                self.feature_indices[np.where(self.feature_indices == keep_list[i])[0]] = i
            cleaned_list.append(self.feature_list[keep_list[i]])
        self.feature_list = cleaned_list

    def cache_arrays(self):
        """Checks that the connection set array data is loaded and loads from hdf5 if not.

        :meta common:
        """

        if self.cell_index_pairs is None or self.face_index_pairs is None or (self.feature_list is not None and
                                                                              self.feature_indices is None):
            assert self.root is not None

        if self.cell_index_pairs is None:
            log.debug('caching cell index pairs from hdf5')
            cell_index_pairs_node = rqet.find_tag(self.root, 'CellIndexPairs')
            h5_key_pair = self.model.h5_uuid_and_path_for_node(cell_index_pairs_node, tag = 'Values')
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'cell_index_pairs',
                                        dtype = 'int')

        if self.face_index_pairs is None:
            log.debug('caching face index pairs from hdf5')
            face_index_pairs_node = rqet.find_tag(self.root, 'LocalFacePerCellIndexPairs')
            h5_key_pair = self.model.h5_uuid_and_path_for_node(face_index_pairs_node, tag = 'Values')
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'face_index_pairs',
                                        dtype = 'int32')

        if len(self.grid_list) > 1 and self.grid_index_pairs is None:
            grid_index_node = rqet.find_tag(self.root, 'GridIndexPairs')
            assert grid_index_node is not None, 'grid index pair data missing in grid connection set'
            h5_key_pair = self.model.h5_uuid_and_path_for_node(grid_index_node, tag = 'Values')
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'grid_index_pairs',
                                        dtype = 'int')

        if self.feature_list is None:
            return
        interp_root = rqet.find_tag(self.root, 'ConnectionInterpretations')

        if self.feature_indices is None:
            log.debug('caching feature indices')
            elements_node = rqet.find_nested_tags(interp_root, ['InterpretationIndices', 'Elements'])
            #         elements_node = rqet.find_nested_tags(interp_root, ['FaultIndices', 'Elements'])
            h5_key_pair = self.model.h5_uuid_and_path_for_node(elements_node, tag = 'Values')
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'feature_indices',
                                        dtype = 'uint32')
            assert self.feature_indices.shape == (self.count,)

            cl_node = rqet.find_nested_tags(interp_root, ['InterpretationIndices', 'CumulativeLength'])
            #         cl_node = rqet.find_nested_tags(interp_root, ['FaultIndices', 'CumulativeLength'])
            h5_key_pair = self.model.h5_uuid_and_path_for_node(cl_node, tag = 'Values')
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'fi_cl',
                                        dtype = 'uint32')
            assert self.fi_cl.shape == (
                self.count,), 'connection set face pair(s) not assigned to exactly one feature'  # rough check

        # delattr(self, 'fi_cl')  # assumed to be one-to-one mapping, so cumulative length is discarded

    def number_of_grids(self):
        """Returns the number of grids involved in the connection set.

        :meta common:
        """

        return len(self.grid_list)

    def grid_for_index(self, grid_index):
        """Returns the grid object for the given grid_index."""

        assert 0 <= grid_index < len(self.grid_list)
        return self.grid_list[grid_index]

    def number_of_features(self):
        """Returns the number of features (faults) in the connection set.

        :meta common:
        """

        if self.feature_list is None:
            return 0
        return len(self.feature_list)

    def list_of_feature_names(self, strip = True):
        """Returns a list of the feature (fault) names for which the connection set has cell face data.

        :meta common:
        """

        if self.feature_list is None:
            return None
        name_list = []
        for (_, _, title) in self.feature_list:
            if strip:
                name_list.append(title.split()[0])
            else:
                name_list.append(title)
        return name_list

    def list_of_fault_names(self, strip = True):
        """Returns a list of the fault names for which the connection set has cell face data."""

        return self.list_of_feature_names(strip = strip)

    def feature_index_and_uuid_for_fault_name(self, fault_name):
        """Returns the index into the feature (fault) list for the named fault.

        arguments:
           fault_name (string): the name of the fault of interest

        returns:
           (index, uuid) where index is an integer which can be used to index the feature_list and is compatible
              with the elements of feature_indices (connection interpretations elements in xml);
              uuid is the uuid.UUID of the feature interpretation for the fault;
              (None, None) is returned if the named fault is not found in the feature list
        """

        if self.feature_list is None:
            return (None, None)
        # the protocol adopted here is that the citation title must start with the fault name
        index = 0
        for (_, uuid, title) in self.feature_list:
            if title.startswith(fault_name):
                return (index, uuid)
            index += 1
        return (None, None)

    def feature_name_for_feature_index(self, feature_index, strip = True):
        """Returns the fault name corresponding to the given index into the feature (fault) list.

        arguments:
           feature_index (non-negative integer): the index into the ordered feature list (fault interpretation list)
           strip (boolean, default True): if True, the citation title for the feature interpretation is split
              and the first word is returned (stripped of leading and trailing whitespace); if False, the
              citation title is returned unaltered

        returns:
           string being the citation title of the indexed fault interpretation part, optionally stripped
           down to the first word; by convention this is the name of the fault
        """

        if self.feature_list is None:
            return None
        if strip:
            return self.feature_list[feature_index][2].split()[0]
        return self.feature_list[feature_index][2]

    def fault_name_for_feature_index(self, feature_index, strip = True):
        """Returns the fault name corresponding to the given index into the feature (fault) list."""

        return self.feature_name_for_feature_index(feature_index = feature_index, strip = strip)

    def feature_index_for_cell_face(self, cell_kji0, axis, p01):
        """Returns the index into the feature (fault) list for the given face of the given cell, or None.

        note:
           where the cell face appears in more than one feature, the result will arbitrarily be the first
           occurrence in the cell_index_pair ordering of the grid connection set
        """

        self.cache_arrays()
        if self.feature_indices is None:
            return None
        cell = self.grid_list[0].natural_cell_index(cell_kji0)
        face = self.face_index_map[axis, p01]
        cell_matches = np.stack(np.where(self.cell_index_pairs == cell)).T
        for match in cell_matches[:]:
            if self.face_index_pairs[match[0], match[1]] == face:
                return self.feature_indices[match[0]]
        return None

    def indices_for_feature_index(self, feature_index):
        """Returns numpy list of indices into main arrays for elements for the specified feature."""

        self.cache_arrays()
        if self.feature_indices is None:
            return None
        matches = np.where(self.feature_indices == feature_index)[0]
        return matches

    def raw_list_of_cell_face_pairs_for_feature_index(self, feature_index):
        """Returns list of cell face pairs contributing to feature (fault) with given index, in raw form.

        arguments:
           feature_index (non-negative integer): the index into the ordered feature list (fault interpretation list)

        returns:
           (cell_index_pairs, face_index_pairs) or (cell_index_pairs, face_index_pairs, grid_index_pairs)
           where each is an integer numpy array of shape (N, 2);
           if the connection set is for a single grid, the returned value is a pair, otherwise a triplet;
           the returned data is in raw form: normalized cell indices (for flattened array) and face indices
           in range 0..5; grid indices can be used to index the grid_list attribute for relevant Grid object
        """

        matches = self.indices_for_feature_index(feature_index)
        if matches is None:
            return None
        if len(self.grid_list) == 1:
            return self.cell_index_pairs[matches], self.face_index_pairs[matches]
        assert self.grid_index_pairs is not None
        return self.cell_index_pairs[matches], self.face_index_pairs[matches], self.grid_index_pairs[matches]

    def inherit_properties_for_selected_indices(self, other, selected_indices):
        """Adds to imported property list by sampling the properties for another grid connection set.

        arguments:
           other (GridConnectionSet): the source grid connection set whose properties will be sampled
           selected_indices (1D numpy int array): the indices, into the main arrays of other, of the
              cell face pairs for which data is to be inherited

        notes:
           this method is typically called after creating a subset grid connection set using methods
           such as: filtered_by_layer_range(), filtered_by_cell_mask() or single_feature(); for the
           first two of those, set the return_indices argument True to acquire the array to pass as
           selected_indices here; when working with a single_feature() connection set, the indices
           can be acquired by calling indices_for_feature_index() for the source connection set;
           this method only adds the inherited property data to the imported list of the property
           collection for this grid connection set (self); it does not write the data to hdf5 or
           create the xml; those actions will happen when calling create_xml() with the
           write_new_properties argument set True; they can also be triggered by calling the
           write_hdf5_and_create_xml_for_new_properties() method directly;
           the property collection for the other grid connection set must be established before
           calling this method, for example by setting find_properties to True when instantiating
           other
        """

        if other.property_collection is None or other.property_collection.number_of_parts() == 0:
            return
        if self.property_collection is None:
            self.property_collection = rqp.PropertyCollection()
            self.property_collection.set_support(support = self)
        self.property_collection.add_to_imported_list_sampling_other_collection(other.property_collection,
                                                                                selected_indices)

    def list_of_cell_face_pairs_for_feature_index(self, feature_index = None):
        """Returns list of cell face pairs contributing to feature (fault) with given index.

        arguments:
           feature_index (non-negative integer, optional): the index into the ordered feature list
               (fault interpretation list); if None, all cell face pairs are returned

        returns:
           (cell_index_pairs, face_index_pairs) or (cell_index_pairs, face_index_pairs, grid_index_pairs)
           where cell_index_pairs is a numpy int array of shape (N, 2, 3) being the paired kji0 cell indices;
           and face_index_pairs is a numpy int array of shape (N, 2, 2) being the paired face indices with the
           final axis of extent 2 indexed by 0 to give the facial axis (0 = K, 1 = J, 2 = I), and indexed by 1
           to give the facial polarity (0 = minus face, 1 = plus face);
           and grid_index_pairs is a numpy int array of shape (N, 2) the values of which can be used to index
           the grid_list attribute to access relevant Grid objects;
           if the connection set is for a single grid, the return value is a pair; otherwise a triplet
        """

        self.cache_arrays()
        if feature_index is None:
            pairs_tuple = (self.cell_index_pairs, self.face_index_pairs)
        else:
            if self.feature_indices is None:
                return None
            pairs_tuple = self.raw_list_of_cell_face_pairs_for_feature_index(feature_index)
        if len(self.grid_list) == 1:
            raw_cell_pairs, raw_face_pairs = pairs_tuple
            grid_pairs = None
        else:
            raw_cell_pairs, raw_face_pairs, grid_pairs = pairs_tuple

        cell_pairs = self.grid_list[0].denaturalized_cell_indices(raw_cell_pairs)
        face_pairs = self.face_index_inverse_map[raw_face_pairs]

        if grid_pairs is None:
            return cell_pairs, face_pairs
        return cell_pairs, face_pairs, grid_pairs

    def simplified_sets_of_kelp_for_feature_index(self, feature_index):
        """Returns a pair of sets of column indices, one for J+ faces, one for I+ faces, contributing to feature.

        arguments:
           feature_index (non-negative integer): the index into the ordered feature list (fault interpretation list)

        returns:
           (set of numpy pair of integers, set of numpy pair of integers) the first set is those columns where
           the J+ faces are contributing to the connection (or the J- faces of the neighbouring column); the second
           set is where the the I+ faces are contributing to the connection (or the I- faces of the neighbouring column)

        notes:
           K faces are ignored;
           this is compatible with the resqml baffle functionality elsewhere
        """

        cell_pairs, face_pairs = self.list_of_cell_face_pairs_for_feature_index(feature_index)
        # set of (j0, i0) pairs of column indices where J+ faces contribute, as 2 element numpy arrays
        simple_j_set = set()
        # set of (j0, i0) pairs of column indices where I+ faces contribute, as 2 element numpy arrays
        simple_i_set = set()
        for i in range(cell_pairs.shape[0]):
            for ip in range(2):
                cell_kji0 = cell_pairs[i, ip].copy()
                axis = face_pairs[i, ip, 0]
                if axis == 0:
                    continue  # skip k faces
                polarity = face_pairs[i, ip, 1]
                if polarity == 0:
                    cell_kji0[axis] -= 1
                    if cell_kji0[axis] < 0:
                        continue
                if axis == 1:  # J axis
                    simple_j_set.add(tuple(cell_kji0[1:]))
                else:  # axis == 2; ie. I axis
                    simple_i_set.add(tuple(cell_kji0[1:]))
        return (simple_j_set, simple_i_set)

    def rework_face_pairs(self):
        """Overwrites the in-memory array of face pairs to reflect neighbouring I or J columns.

        note:
           the indexing of faces within a cell is not clearly documented in the RESQML guides;
           the constant self.face_index_map and its inverse is the best guess at what this mapping is;
           this function overwrites the faces within cell data where a connected pair are in neighbouring
           I or J columns, using the face within cell values that fit with the neighbourly relationship;
           for some implausibly extreme gridding, this might not give the correct result
        """

        self.cache_arrays()
        if self.feature_indices is None:
            return None

        assert len(self.grid_list) == 1

        cell_pairs = self.grid_list[0].denaturalized_cell_indices(self.cell_index_pairs)

        modifications = 0
        for i in range(self.count):
            kji0_pair = cell_pairs[i]
            new_faces = None
            if kji0_pair[0][1] == kji0_pair[1][1]:  # same J index
                if kji0_pair[0][2] == kji0_pair[1][2] + 1:
                    new_faces = (self.face_index_map[2, 0], self.face_index_map[2, 1])
                elif kji0_pair[0][2] == kji0_pair[1][2] - 1:
                    new_faces = (self.face_index_map[2, 1], self.face_index_map[2, 0])
            elif kji0_pair[0][2] == kji0_pair[1][2]:  # same I index
                if kji0_pair[0][1] == kji0_pair[1][1] + 1:
                    new_faces = (self.face_index_map[1, 0], self.face_index_map[1, 1])
                elif kji0_pair[0][1] == kji0_pair[1][1] - 1:
                    new_faces = (self.face_index_map[1, 1], self.face_index_map[1, 0])
            if new_faces is not None and np.any(self.face_index_pairs[i] != new_faces):
                self.face_index_pairs[i] = new_faces
                modifications += 1
        log.debug('number of face pairs modified during neighbour based reworking: ' + str(modifications))

    def write_hdf5(self, file_name = None, mode = 'a'):
        """Create or append to an hdf5 file, writing datasets for the connection set after caching arrays.

        :meta common:
        """

        # NB: cell pairs, face pairs, and feature indices arrays must be ready, in memory
        # note: grid pairs not yet supported
        # prepare one-to-one mapping of cell face pairs with feaature indices
        if not file_name:
            file_name = self.model.h5_file_name()
        log.debug('gcs write to hdf5 file: ' + file_name + ' mode: ' + mode)

        h5_reg = rwh5.H5Register(self.model)

        # register arrays
        # note: this implementation requires each cell face pair to be assigned to exactly one interpretation
        # uuid/CellIndexPairs  (N, 2)  int64
        h5_reg.register_dataset(self.uuid, 'CellIndexPairs', self.cell_index_pairs)
        if len(self.grid_list) > 1 and self.grid_index_pairs is not None:
            # uuid/GridIndexPairs  (N, 2)  int64
            h5_reg.register_dataset(self.uuid, 'GridIndexPairs', self.grid_index_pairs)
        # uuid/LocalFacePerCellIndexPairs  (N, 2)  int32
        h5_reg.register_dataset(self.uuid, 'LocalFacePerCellIndexPairs', self.face_index_pairs)

        if self.feature_indices is not None:
            # uuid/InterpretationIndices/elements  (N,)  uint32
            h5_reg.register_dataset(self.uuid, 'InterpretationIndices/elements', self.feature_indices)
            # uuid/InterpretationIndices/cumulativeLength  (N,)  uint32
            one_to_one = np.arange(1, self.count + 1, dtype = int)
            h5_reg.register_dataset(self.uuid, 'InterpretationIndices/cumulativeLength', one_to_one)

        h5_reg.write(file_name, mode = mode)

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   write_new_properties = True,
                   title = None,
                   originator = None,
                   extra_metadata = None):
        """Creates a Grid Connection Set (fault faces) xml node.
        
        Optionally adds to parts forest.

        :meta common:
        """

        # NB: only one grid handled for now
        # xml for grid(s) must be created before calling this method

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()
        if not self.title and not title:
            title = 'ROOT'

        gcs = super().create_xml(add_as_part = False,
                                 title = title,
                                 originator = originator,
                                 extra_metadata = extra_metadata)

        c_node = rqet.SubElement(gcs, ns['resqml2'] + 'Count')
        c_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        c_node.text = str(self.count)

        cip_node = rqet.SubElement(gcs, ns['resqml2'] + 'CellIndexPairs')
        cip_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        cip_node.text = '\n'

        cip_null = rqet.SubElement(cip_node, ns['resqml2'] + 'NullValue')
        cip_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        cip_null.text = str(self.cell_index_pairs_null_value)

        cip_values = rqet.SubElement(cip_node, ns['resqml2'] + 'Values')
        cip_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        cip_values.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'CellIndexPairs', root = cip_values)

        if len(self.grid_list) > 1 and self.grid_index_pairs is not None:

            gip_node = rqet.SubElement(gcs, ns['resqml2'] + 'GridIndexPairs')
            gip_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            gip_node.text = '\n'

            gip_null = rqet.SubElement(gip_node, ns['resqml2'] + 'NullValue')
            gip_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            gip_null.text = str(self.grid_index_pairs_null_value)

            gip_values = rqet.SubElement(gip_node, ns['resqml2'] + 'Values')
            gip_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            gip_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'GridIndexPairs', root = gip_values)

        fip_node = rqet.SubElement(gcs, ns['resqml2'] + 'LocalFacePerCellIndexPairs')
        fip_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        fip_node.text = '\n'

        fip_null = rqet.SubElement(fip_node, ns['resqml2'] + 'NullValue')
        fip_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        fip_null.text = str(self.face_index_pairs_null_value)

        fip_values = rqet.SubElement(fip_node, ns['resqml2'] + 'Values')
        fip_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        fip_values.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'LocalFacePerCellIndexPairs', root = fip_values)

        if self.feature_indices is not None and self.feature_list is not None:

            ci_node = rqet.SubElement(gcs, ns['resqml2'] + 'ConnectionInterpretations')
            ci_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ConnectionInterpretations')
            ci_node.text = '\n'

            ii = rqet.SubElement(ci_node, ns['resqml2'] + 'InterpretationIndices')
            #         ii = rqet.SubElement(ci_node, ns['resqml2'] + 'FaultIndices')
            ii.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlJaggedArray')
            ii.text = '\n'

            elements = rqet.SubElement(ii, ns['resqml2'] + 'Elements')
            elements.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            elements.text = '\n'

            el_null = rqet.SubElement(elements, ns['resqml2'] + 'NullValue')
            el_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            el_null.text = '-1'

            el_values = rqet.SubElement(elements, ns['resqml2'] + 'Values')
            el_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            el_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'InterpretationIndices/elements', root = el_values)
            #         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'FaultIndices/elements', root = el_values)

            c_length = rqet.SubElement(ii, ns['resqml2'] + 'CumulativeLength')
            c_length.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            c_length.text = '\n'

            cl_null = rqet.SubElement(c_length, ns['resqml2'] + 'NullValue')
            cl_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            cl_null.text = '-1'

            cl_values = rqet.SubElement(c_length, ns['resqml2'] + 'Values')
            cl_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            cl_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'InterpretationIndices/cumulativeLength',
                                               root = cl_values)
            #         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'FaultIndices/cumulativeLength', root = cl_values)

            # add feature interpretation reference node for each fault in list, NB: ordered list
            for (f_content_type, f_uuid, f_title) in self.feature_list:

                if f_content_type in valid_interpretation_types:
                    fi_part = rqet.part_name_for_object(f_content_type, f_uuid)
                    fi_root = self.model.root_for_part(fi_part)
                    self.model.create_ref_node('FeatureInterpretation',
                                               self.model.title_for_root(fi_root),
                                               f_uuid,
                                               content_type = f_content_type,
                                               root = ci_node)
                else:
                    raise Exception(f'unsupported content type {f_content_type} in grid connection set')

        for grid in self.grid_list:
            self.model.create_ref_node('Grid',
                                       self.model.title_for_root(grid.root),
                                       grid.uuid,
                                       content_type = 'obj_IjkGridRepresentation',
                                       root = gcs)

        if add_as_part:
            self.model.add_part('obj_GridConnectionSetRepresentation', self.uuid, gcs)

            if add_relationships:
                if self.feature_list:
                    for (obj_type, f_uuid, _) in self.feature_list:
                        fi_part = rqet.part_name_for_object(obj_type, f_uuid)
                        fi_root = self.model.root_for_part(fi_part)
                        if fi_root is not None:
                            self.model.create_reciprocal_relationship(gcs, 'destinationObject', fi_root, 'sourceObject')
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(gcs, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')
                for grid in self.grid_list:
                    self.model.create_reciprocal_relationship(gcs, 'destinationObject', grid.root, 'sourceObject')

        if write_new_properties:
            self.write_hdf5_and_create_xml_for_new_properties()

        return gcs

    def write_simulator(self,
                        filename,
                        mode = 'w',
                        simulator = 'nexus',
                        include_both_sides = False,
                        use_minus = False,
                        trans_mult_uuid = None):
        """Creates a Nexus include file holding MULT keywords and data. trans_mult_uuid (optional) is the uuid of a property on the gcs containing transmissibility multiplier values. If not provided values of 1.0 will be used."""
        if trans_mult_uuid is not None:
            self.extract_property_collection()
            assert self.property_collection.part_in_collection(self.model.part_for_uuid(
                trans_mult_uuid)), f'trans_mult_uuid provided is not part of collection {trans_mult_uuid}'
            tmult_array = self.property_collection.cached_part_array_ref(self.model.part_for_uuid(trans_mult_uuid))
            assert tmult_array is not None
        else:
            tmult_array = None

        assert simulator == 'nexus'

        active_nexus_head = None
        row_count = 0

        def write_nexus_header_lines(fp, axis, polarity, fault_name, grid_name = 'ROOT'):
            nonlocal active_nexus_head
            if (axis, polarity, fault_name, grid_name) == active_nexus_head:
                return
            T = 'T' + 'ZYX'[axis]
            plus_minus = ['MINUS', 'PLUS'][polarity]
            fp.write('\nMULT\t' + T + '\tALL\t' + plus_minus + '\tMULT\n')
            fp.write('\tGRID\t' + grid_name + '\n')
            fp.write('\tFNAME\t' + fault_name + '\n')
            if len(fault_name) > 256:
                log.warning('exported fault name longer than Nexus limit of 256 characters: ' + fault_name)
                tm.log_nexus_tm('warning')
            active_nexus_head = (axis, polarity, fault_name, grid_name)

        def write_row(gcs, fp, name, i, j, k1, k2, axis, polarity, tmult):
            nonlocal row_count
            write_nexus_header_lines(fp, axis, polarity, name)
            fp.write('\t{0:1d}\t{1:1d}\t{2:1d}\t{3:1d}\t{4:1d}\t{5:1d}\t{6:.4f}\n'.format(
                i + 1, i + 1, j + 1, j + 1, k1 + 1, k2 + 1, tmult))
            row_count += 1

        log.info('writing fault data in simulator format to file: ' + filename)

        if include_both_sides:
            sides = [0, 1]
        elif use_minus:
            sides = [1]
        else:
            sides = [0]

        with open(filename, mode, newline = '') as fp:
            for feature_index in range(len(self.feature_list)):
                feature_name = self.feature_list[feature_index][2].split()[0].upper()
                cell_index_pairs, face_index_pairs = self.list_of_cell_face_pairs_for_feature_index(feature_index)
                if tmult_array is not None:
                    feature_mask = np.where(self.feature_indices == feature_index, 1, 0)
                    feat_mult_array = np.extract(feature_mask, tmult_array)
                else:
                    feat_mult_array = np.ones(shape = (cell_index_pairs.shape[0],), dtype = float)
                for side in sides:
                    both = np.empty((cell_index_pairs.shape[0], 6), dtype = int)  # axis, polarity, k, j, i, tmult
                    both[:, :2] = face_index_pairs[:, side, :]  # axis, polarity
                    both[:, 2:-1] = cell_index_pairs[:, side, :]  # k, j, i
                    both[:, -1] = feat_mult_array.flatten()
                    df = pd.DataFrame(both, columns = ['axis', 'polarity', 'k', 'j', 'i', 'tmult'])
                    df = df.sort_values(by = ['axis', 'polarity', 'j', 'i', 'k', 'tmult'])
                    both_sorted = np.empty(both.shape, dtype = int)
                    both_sorted[:] = df
                    cell_indices = both_sorted[:, 2:-1]
                    face_indices = np.empty((both_sorted.shape[0], 2), dtype = int)
                    face_indices[:, :] = both_sorted[:, :2]
                    tmult_values = both_sorted[:, -1]
                    del both_sorted
                    del both
                    del df
                    k = None
                    i = j = k2 = axis = polarity = None  # only needed to placate flake8 which whinges incorrectly otherwise
                    for row in range(cell_indices.shape[0]):
                        kp, jp, ip = cell_indices[row]
                        axis_p, polarity_p = face_indices[row]
                        tmult = tmult_values[row]
                        if k is not None:
                            if axis_p != axis or polarity_p != polarity or ip != i or jp != j or kp != k2 + 1:
                                write_row(self, fp, feature_name, i, j, k, k2, axis, polarity, tmult)
                                k = None
                            else:
                                k2 = kp
                        if k is None:
                            i = ip
                            j = jp
                            k = k2 = kp
                            axis = axis_p
                            polarity = polarity_p
                    if k is not None:
                        write_row(self, fp, feature_name, i, j, k, k2, axis, polarity, tmult)

    def get_column_edge_list_for_feature(self, feature, gridindex = 0, min_k = 0, max_k = 0):
        """Extracts a list of cell faces for a given feature index, over a given range of layers in the grid.

        arguments:
           feature - feature index
           gridindex - index of grid to be used in grid connection set gridlist, default 0
           min_k - minimum k layer, default 0
           max_k - maximum k layer, default 0
        returns:
           list of cell faces for the feature (j_col, i_col, face_axis, face_polarity)
        """
        subgcs = self.filtered_by_layer_range(min_k0 = min_k, max_k0 = max_k, pare_down = False)
        if subgcs is None:
            return None

        cell_face_details = subgcs.list_of_cell_face_pairs_for_feature_index(feature)

        if len(cell_face_details) == 2:
            assert gridindex == 0, 'Only one grid present'
            cell_ind_pairs, face_ind_pairs = cell_face_details
            grid_ind_pairs = None
        else:
            cell_ind_pairs, face_ind_pairs, grid_ind_pairs = cell_face_details
        ij_faces = []
        for i in range(len(cell_ind_pairs)):
            for a_or_b in [0, 1]:
                if grid_ind_pairs is None or grid_ind_pairs[i, a_or_b] == gridindex:
                    if face_ind_pairs[i, a_or_b, 0] != 0:
                        entry = ((
                            int(cell_ind_pairs[i, a_or_b, 1]),
                            int(cell_ind_pairs[i, a_or_b, 2]),
                            int(face_ind_pairs[i, a_or_b, 0]) - 1,  # in the outputs j=0, i=1
                            int(face_ind_pairs[i, a_or_b, 1])))  # in the outputs negativeface=0, positiveface=1
                        if entry in ij_faces:
                            continue
                        ij_faces.append(entry)
        ij_faces_np = np.array((ij_faces))
        return ij_faces_np

    def get_column_edge_bool_array_for_feature(self, feature, gridindex = 0, min_k = 0, max_k = 0):
        """Generate a boolean aray defining which column edges are present for a given feature and k-layer range.

        arguments:
           feature - feature index
           gridindex - index of grid to be used in grid connection set gridlist, default 0
           min_k - minimum k layer
           max_k - maximum k layer
        returns:
           boolean fault_by_column_edge_mask array (shape nj,ni,2,2)

        note: the indices for the final two axes define the edges:
           the first defines j or i (0 or 1)
           the second negative or positive face (0 or 1)

           so [[True,False],[False,True]] indicates the -j and +i edges of the column are present
        """
        cell_face_list = self.get_column_edge_list_for_feature(feature, gridindex, min_k, max_k)

        fault_by_column_edge_mask = np.zeros((self.grid_list[gridindex].nj, self.grid_list[gridindex].ni, 2, 2),
                                             dtype = bool)
        if cell_face_list is not None:
            for i in cell_face_list:
                fault_by_column_edge_mask[tuple(i)] = True

        return fault_by_column_edge_mask

    def get_property_by_feature_index_list(self,
                                           feature_index_list = None,
                                           property_name = 'Transmissibility multiplier'):
        """Returns a list of property values by feature based on extra metadata items.

        arguments:
           feature_index_list (list of int, optional): if present, the feature indices for which property values will be included
              in the resulting list; if None, values will be included for each feature in the feature_list for this connection set
           property_name (string, default 'Transmissibility multiplier'): the property name of interest, as used in the features'
              extra metadata as a key (this not a property collection citation title)

        returns:
           list of float being the list of property values for the list of features, in corresponding order

        notes:
           this method does not refer to property collection arrays, it simply looks for a constant extra metadata item
           for each feature; where no such item is found, a NaN is added to the return list;
           currently assumes fault interpretation (not horizon or geobody boundary)
        """

        if feature_index_list is None:
            feature_index_list = range(len(self.feature_list))
        value_list = []
        for feature_index in feature_index_list:
            _, feature_uuid, _ = self.feature_list[feature_index]
            feat = rqo.FaultInterpretation(parent_model = self.model, uuid = feature_uuid)
            if property_name not in feat.extra_metadata.keys():
                log.info(
                    f'Property name {property_name} not found in extra_metadata for {self.model.citation_title_for_part(self.model.part_for_uuid(feature_uuid))}'
                )
                value_list.append(np.NaN)
            else:
                value_list.append(float(feat.extra_metadata[property_name]))
        return value_list

    def get_column_edge_float_array_for_feature(self,
                                                feature,
                                                fault_by_column_edge_mask,
                                                property_name = 'Transmissibility multiplier',
                                                gridindex = 0,
                                                ref_k = None):
        """Generate a float value aray defining the property values for different column edges present for a given feature.

        arguments:
           feature - feature index
           fault_by_column_edge_mask - fault_by_column_edge_mask with True on edges where feature is present
           property_name - name of property, should be present within the FaultInterpreation feature metadata;
              lowercase version is also used as property kind when searching for a property array
           gridindex - index of grid for which column edge data is required
           ref_k - reference k_layer to use where property has variable value for a feature;
              if None, no property array will be used and None will be returned for variable properties

        returns:
           float property_value_by_column_edge array (shape nj,ni,2,2) based on extra metadata

        note: the indices for the final two axes define the edges:
           the first defines j or i (0 or 1)
           the second negative or positive face (0 or 1)

           so [[1,np.nan],[np.nan,np.nan]] indicates the -j edge of the column are present with a value of 1

           this method preferentially uses a constant extra metadata item for the feature, with the property
           collection being used when the extra metadata is absent
        """

        single_grid = (self.number_of_grids() == 1)
        if single_grid:
            assert gridindex == 0
        else:
            assert self.grid_index_pairs is not None

        prop_values = self.get_property_by_feature_index_list(feature_index_list = [feature],
                                                              property_name = property_name)
        if prop_values == [] or np.isnan(prop_values[0]):
            pc = self.extract_property_collection()
            if ref_k is None or pc is None:
                return None
            # use property name as (local) property kind
            pk = property_name.lower()
            prop_array = pc.single_array_ref(property_kind = pk)
            if prop_array is None:
                return None
            # filter down to a single layer and single feature
            feature_gcs = self.single_feature(feature)
            feature_indices = self.indices_for_feature_index(feature)
            feature_prop_array = prop_array[feature_indices]
            layer_gcs, layer_indices = feature_gcs.filtered_by_layer_range(min_k0 = ref_k,
                                                                           max_k0 = ref_k,
                                                                           pare_down = False,
                                                                           return_indices = True)
            assert layer_gcs.count == len(layer_indices)
            if layer_gcs.count == 0:  # feature does not have any faces in reference layer
                return None
            layer_prop_array = feature_prop_array[layer_indices]
            # fill in individual values rather laboriously
            property_value_by_column_edge = np.full(fault_by_column_edge_mask.shape, np.nan)
            for i in range(layer_gcs.count):
                for side in range(2):
                    gi = 0 if single_grid else layer_gcs.grid_index_pairs[i, side]
                    if gi != gridindex:
                        continue
                    cell = layer_gcs.cell_index_pairs[i, side]
                    if cell == layer_gcs.cell_index_pairs_null_value:
                        continue
                    cell_kji = self.grid_list[gi].denaturalized_cell_index(cell)
                    face_index = layer_gcs.face_index_pairs[i, side]
                    if face_index == layer_gcs.face_index_pairs_null_value:
                        continue
                    axis, polarity = self.face_index_inverse_map[face_index]
                    if axis == 0:  # k face
                        continue
                    property_value_by_column_edge[cell_kji[1], cell_kji[2], axis - 1, polarity] = layer_prop_array[i]
        else:
            property_value_by_column_edge = np.where(fault_by_column_edge_mask, prop_values[0], np.nan)
        return property_value_by_column_edge

    def get_combined_fault_mask_index_value_arrays(self,
                                                   gridindex = 0,
                                                   min_k = 0,
                                                   max_k = 0,
                                                   property_name = 'Transmissibility multiplier',
                                                   feature_list = None,
                                                   ref_k = None):
        """Generate combined mask, index and value arrays for all column edges across a k-layer range, for a defined feature_list.

        arguments:
           gridindex - index of grid to be used in grid connection set gridlist, default 0
           min_k - minimum k_layer
           max_k - maximum k_layer
           property_name - name of property, should be present within the FaultInterpreation feature metadata;
              lowercase version is used as property kind when searching for a property array
           feature_list - list of feature index numbers to run for, default to all features
           ref_k - reference k_layer to use where property has variable value for a feature;
              if None defaults to min_k
        returns:
           bool array mask showing all column edges within features (shape nj,ni,2,2)
           int array showing the feature index for all column edges within features (shape nj,ni,2,2)
           float array showing the property value for all column edges within features (shape nj,ni,2,2)
        """
        self.cache_arrays()
        #     if feature_list is None: feature_list = np.unique(self.feature_indices)
        if feature_list is None:
            feature_list = np.arange(len(self.feature_list))
        if ref_k is None:
            ref_k = min_k

        sum_unmasked = None

        for i, feature in enumerate(feature_list):
            fault_by_column_edge_mask = self.get_column_edge_bool_array_for_feature(feature,
                                                                                    gridindex,
                                                                                    min_k = min_k,
                                                                                    max_k = max_k)
            property_value_by_column_edge = self.get_column_edge_float_array_for_feature(feature,
                                                                                         fault_by_column_edge_mask,
                                                                                         property_name = property_name,
                                                                                         gridindex = gridindex,
                                                                                         ref_k = ref_k)
            if i == 0:
                combined_mask = fault_by_column_edge_mask.copy()
                if property_value_by_column_edge is not None:
                    combined_values = property_value_by_column_edge.copy()
                else:
                    combined_values = None
                combined_index = np.full((fault_by_column_edge_mask.shape), -1, dtype = int)
                combined_index = np.where(fault_by_column_edge_mask, feature, combined_index)
                sum_unmasked = np.sum(fault_by_column_edge_mask)
            else:
                combined_mask = np.logical_or(combined_mask, fault_by_column_edge_mask)
                if property_value_by_column_edge is not None:
                    if combined_values is not None:
                        combined_values = np.where(fault_by_column_edge_mask, property_value_by_column_edge,
                                                   combined_values)
                    else:
                        combined_values = property_value_by_column_edge.copy()
                combined_index = np.where(fault_by_column_edge_mask, feature, combined_index)
                sum_unmasked += np.sum(fault_by_column_edge_mask)

        if not np.sum(combined_mask) == sum_unmasked:
            log.warning("One or more features exist across the same column edge!")
        return combined_mask, combined_index, combined_values

    def tr_property_array(self, fa = None, tol_fa = 0.0001, tol_half_t = 1.0e-5, apply_multipliers = False):
        """Return a transmissibility array with one value per pair in the connection set.

        argument:
           fa (numpy float array of shape (count, 2), optional): if present, the fractional area for each pair connection,
              from perspective of each side of connection; if None, a fractional area of one is assumed
           tol_fa (float, default 0.0001): fractional area tolerance – if the fractional area on either side of a
              juxtaposition is less than this, then the corresponding transmissibility is set to zero
           tol_half_t (float, default 1.0e-5): if the half cell transmissibility either side of a juxtaposition is
              less than this, the corresponding transmissibility is set to zero; units are as for returned values (see
              notes)
           apply_multipliers (boolean, default False): if True, a transmissibility multiplier array is fetched from the
              property collection for the connection set, and failing that a multiplier for each feature is
              extracted from the feature extra metadata, and applied to the transmissibility calculation

        returns:
           numpy float array of shape (count,) being the absolute transmissibilities across the connected cell face pairs;
           see notes regarding units

        notes:
           implicit units of measure of returned values will be m3.cP/(kPa.d) if grids' crs length units are metres,
           bbl.cP/(psi.d) if length units are feet; the computation is compatible with the Nexus NEWTRAN formulation;
           multiple grids are assumed to be in the same units and z units must be the same as xy units; this method
           does not add the transmissibility array as a property
        """

        feature_mult_list = None
        mult_array = None
        if apply_multipliers:
            pc = self.extract_property_collection()
            mult_array = pc.single_array_ref(property_kind = 'transmissibility multiplier')
            if mult_array is None:
                feature_mult_list = self.get_property_by_feature_index_list()

        count = self.count
        if fa is not None:
            assert fa.shape == (count, 2)
        f_tr = np.zeros(count)
        half_t_list = []
        for grid in self.grid_list:
            half_t_list.append(grid.half_cell_transmissibility())  # todo: pass realization argument
        single_grid = (self.number_of_grids() == 1)
        kji0 = self.grid_list[0].denaturalized_cell_indices(self.cell_index_pairs) if single_grid else None
        # kji0 shape (count, 2, 3); 2 being m,p; 3 being k,j,i; multi-grid cell indices denaturalised within for loop below
        fa_m = fa_p = 1.0
        gi_m = gi_p = 0  # used below in single grid case
        for e in range(count):
            axis_m, polarity_m = self.face_index_inverse_map[self.face_index_pairs[e, 0]]
            axis_p, polarity_p = self.face_index_inverse_map[self.face_index_pairs[e, 0]]
            if single_grid:
                kji0_m = kji0[e, 0]
                kji0_p = kji0[e, 1]
            else:
                gi_m = self.grid_index_pairs[e, 0]
                gi_p = self.grid_index_pairs[e, 1]
                kji0_m = self.grid_list[gi_m].denaturalized_cell_index(self.cell_index_pairs[e, 0])
                kji0_p = self.grid_list[gi_p].denaturalized_cell_index(self.cell_index_pairs[e, 1])
            half_t_m = half_t_list[gi_m][kji0_m[0], kji0_m[1], kji0_m[2], axis_m, polarity_m]
            half_t_p = half_t_list[gi_p][kji0_p[0], kji0_p[1], kji0_p[2], axis_p, polarity_p]
            if half_t_m < tol_half_t or half_t_p < tol_half_t:
                continue
            if fa is not None:
                fa_m, fa_p = fa[e]
                if fa_m < tol_fa or fa_p < tol_fa:
                    continue
            mult_tr = 1.0
            if apply_multipliers:
                if mult_array is not None:
                    mult_tr = mult_array[e]
                elif feature_mult_list is not None:
                    mult_tr = feature_mult_list[self.feature_indices[e]]
            f_tr[e] = mult_tr / (1.0 / (half_t_m * fa_m) + 1.0 / (half_t_p * fa_p))
        return f_tr

    def inherit_features(self, featured):
        """Inherit features from another connection set, reassigning feature indices for matching faces.

        arguments:
           featured (GridConnectionSet): the other connection set which holds features to be inherited

        notes:
           the list of cell face pairs in this set remains unchanged; the corresponding feature indices
           are updated based on individual cell faces that are found in the featured connection set;
           the feature list is extended with inherited features as required; this method is typically
           used to populate a fault connection set built from grid geometry with named fault features
           from a simplistic connection set, thus combining full geometric juxtaposition information
           with named features; currently restricted to single grid connection sets
        """

        def sorted_paired_cell_face_index_position(cell_face_index, a_or_b):
            # pair one side (a_or_b) of cell_face_index with its position, then sort
            count = len(cell_face_index)
            sp = np.empty((count, 2), dtype = int)
            sp[:, 0] = cell_face_index[:, a_or_b]
            sp[:, 1] = np.arange(count)
            t = [tuple(r) for r in sp]  # could use numpy fields based sort instead of tuple list?
            t.sort()
            return np.array(t)

        def find_in_sorted(paired, v):

            def fis(p, v, a, b):  # recursive binary search
                if a >= b:
                    return None
                m = (a + b) // 2
                s = p[m, 0]
                if s == v:
                    return p[m, 1]
                if s > v:
                    return fis(p, v, a, m)
                return fis(p, v, m + 1, b)

            return fis(paired, v, 0, len(paired))

        assert len(self.grid_list) == 1 and len(featured.grid_list) == 1
        assert tuple(self.grid_list[0].extent_kji) == tuple(featured.grid_list[0].extent_kji)

        # build combined feature list and mapping of feature indices
        featured.cache_arrays()
        original_feature_count = len(self.feature_list)
        featured_index_map = []  # maps from feature index in featured to extended feature list in this set
        feature_uuid_list = [bu.uuid_from_string(u) for _, u, _ in self.feature_list]  # bu call probably not needed
        for featured_triplet in featured.feature_list:
            featured_uuid = bu.uuid_from_string(featured_triplet[1])  # bu call probably not needed
            if featured_uuid in feature_uuid_list:
                featured_index_map.append(feature_uuid_list.index[featured_uuid])
            else:
                featured_index_map.append(len(self.feature_list))
                self.feature_list.append(featured_triplet)
                feature_uuid_list.append(featured_uuid)
                self.model.copy_part_from_other_model(featured.model, featured.model.part(uuid = featured_uuid))

        # convert cell index, face index to a single integer to facilitate sorting and comparison
        cell_face_index_self = self.compact_indices()
        cell_face_index_featured = featured.compact_indices()

        # sort all 4 (2 sides in each of 2 sets) cell_face index data, keeping track of positions in original lists
        cfp_a_self = sorted_paired_cell_face_index_position(cell_face_index_self, 0)
        cfp_b_self = sorted_paired_cell_face_index_position(cell_face_index_self, 1)
        cfp_a_featured = sorted_paired_cell_face_index_position(cell_face_index_featured, 0)
        cfp_b_featured = sorted_paired_cell_face_index_position(cell_face_index_featured, 1)

        # for each cell face in self, look for same in featured and inherit feature index if found
        previous_cell_face_index = previous_feature_index = None
        for (a_or_b, cfp_self) in [(0, cfp_a_self), (1, cfp_b_self)]:  # could risk being lazy and only using one side?
            for (cell_face_index, place) in cfp_self:
                if self.feature_indices[place] >= original_feature_count:
                    continue  # already been updated
                if cell_face_index == previous_cell_face_index:
                    if previous_feature_index is not None:
                        self.feature_indices[place] = previous_feature_index
                elif cell_face_index == self.face_index_pairs_null_value:
                    continue
                else:
                    featured_place = find_in_sorted(cfp_a_featured, cell_face_index)
                    if featured_place is None:
                        featured_place = find_in_sorted(cfp_b_featured, cell_face_index)
                    if featured_place is None:
                        previous_feature_index = None
                    else:
                        featured_index = featured.feature_indices[featured_place]
                        self.feature_indices[place] = previous_feature_index = featured_index_map[featured_index]
                    previous_cell_face_index = cell_face_index

        # clean up by removing any original features no longer in use
        removals = []
        for fi in range(original_feature_count):
            if fi not in self.feature_indices:
                removals.append(fi)
        reduction = len(removals)
        if reduction > 0:
            while len(removals) > 0:
                self.feature_list.pop(removals[-1])
                removals.pop()
            self.feature_indices[:] -= reduction

    def compact_indices(self):
        """Returns numpy int array of shape (count, 2) combining each cell index, face index into a single integer."""
        if self.cell_index_pairs is None or self.face_index_pairs is None:
            return None
        null_mask = np.logical_or(self.cell_index_pairs == self.cell_index_pairs_null_value,
                                  self.face_index_pairs == self.face_index_pairs_null_value)
        combined = 6 * self.cell_index_pairs + self.face_index_pairs
        return np.where(null_mask, self.face_index_pairs_null_value, combined)

    def surface(self, feature_index = None):
        """Returns a Surface object representing the faces of the feature, for an unsplit grid.

        note:
           this method does not write the hdf5 data nor create the xml for the surface
        """
        t = rqf_gf._triangulate_unsplit_grid_connection_set(self, feature_index = feature_index)
        if t is None:
            return None
        p = self.grid_list[0].points_cached.reshape((-1, 3))
        assert p is not None
        t, p = rqs.distill_triangle_points(t, p)
        if feature_index is None:
            feature_index = 0
        title = self.feature_name_for_feature_index(feature_index)
        surf = rqs.Surface(self.model, crs_uuid = self.grid_list[0].crs_uuid, title = title)
        assert surf is not None
        surf.set_from_triangles_and_points(t, p)
        return surf

    def _create_multiplier_property(self, mult_list, const_mult):
        pc = self.extract_property_collection()
        if const_mult:
            mult_array = None
            const_value = mult_list[0]
        else:
            mult_array = np.array(mult_list, dtype = float)
            const_value = None
        pc.add_cached_array_to_imported_list(
            mult_array,
            'dataframe from ascii simulator input file',
            'TMULT',
            uom = 'Euc',  # actually a ratio of transmissibilities
            property_kind = 'transmissibility multiplier',
            local_property_kind_uuid = None,
            realization = None,
            indexable_element = 'faces',
            const_value = const_value)

    def _set_pairs_from_faces_df_for_named_fault(self, feature_index, faces, name, fault_tmult_dict, fi_dict,
                                                 create_organizing_objects_where_needed, tbf_parts_list, fi_list,
                                                 cell_pair_list, face_pair_list, const_mult, mult_list, grid,
                                                 create_mult_prop):
        fault_dict_multiplier = 1.0
        if fault_tmult_dict is not None:
            if name in fault_tmult_dict:
                fault_dict_multiplier = float(fault_tmult_dict[name])
            if name.lower() in fault_tmult_dict:
                fault_dict_multiplier = float(fault_tmult_dict[name.lower()])
        # fetch uuid for fault interpretation object
        if name.lower() in fi_dict:
            fi_uuid = fi_dict[name.lower()]
        elif create_organizing_objects_where_needed:
            tbf = None
            fi_root = None
            for tbf_part in tbf_parts_list:
                if name.lower() == self.model.title_for_part(tbf_part).split()[0].lower():
                    tbf = rqo.TectonicBoundaryFeature(self.model, uuid = self.model.uuid_for_part(tbf_part))
                    break
            if tbf is None:
                tbf = rqo.TectonicBoundaryFeature(self.model, feature_name = name, kind = 'fault')
                tbf.create_xml()
            fi = rqo.FaultInterpretation(self.model, title = name, tectonic_boundary_feature = tbf,
                                         is_normal = True)  # todo: set is_normal based on fault geometry in grid?
            fi_root = fi.create_xml(tectonic_boundary_feature_root = tbf.root)
            fi_uuid = rqet.uuid_for_part_root(fi_root)
            fi_dict[name.lower()] = fi_uuid
        else:
            log.error('no interpretation found for fault: ' + name)
            return False, const_mult
        self.feature_list.append(('obj_FaultInterpretation', fi_uuid, str(name)))
        feature_faces = faces[faces['name'] == name]
        fault_const_mult = True
        fault_mult_value = None
        for i in range(len(feature_faces)):
            entry = feature_faces.iloc[i]
            f = entry['face']
            axis = 'KJI'.index(f[0])
            fp = '-+'.index(f[1])
            multiplier = float(entry['mult']) * fault_dict_multiplier
            if const_mult and len(mult_list):
                const_mult = maths.isclose(multiplier, mult_list[0])
            if fault_const_mult:
                if fault_mult_value is None:
                    fault_mult_value = multiplier
                else:
                    fault_const_mult = maths.isclose(multiplier, fault_mult_value)
            for k0 in range(entry['k1'], entry['k2'] + 1):
                for j0 in range(entry['j1'], entry['j2'] + 1):
                    for i0 in range(entry['i1'], entry['i2'] + 1):
                        neighbour = np.array([k0, j0, i0], dtype = int)
                        if fp:
                            neighbour[axis] += 1
                        else:
                            neighbour[axis] -= 1
                        fi_list.append(feature_index)
                        cell_pair_list.append((grid.natural_cell_index(
                            (k0, j0, i0)), grid.natural_cell_index(neighbour)))
                        face_pair_list.append((self.face_index_map[axis, fp], self.face_index_map[axis, 1 - fp]))
                        if create_mult_prop:
                            mult_list.append(multiplier)
        if fi_root is not None and fault_const_mult and fault_mult_value is not None:
            #patch extra_metadata into xml for new fault interpretation object
            rqet.create_metadata_xml(fi_root, {"Transmissibility multiplier": str(fault_mult_value)})
        return True, const_mult

    def face_surface_normal_vectors(self,
                                    triangle_per_face: np.ndarray,
                                    surface_normal_vectors: np.ndarray,
                                    add_as_property: bool = False,
                                    uom: str = 'm') -> np.ndarray:
        """Returns an array of the surface normal vectors corresponding to each GCS face.

        arguments:
            triangle_per_face (np.ndarray): an array of the surface triangle index corresponding to each face.
            surface_normal_vectors (np.ndarray): an array of the normal vectors for each triangle in the surface.
            add_as_property (bool): if True, face_surface_normal_vectors_array is added as a property to the model.
            uom (str): the unit of measure of the normal vectors. It is used if add_as_property is True.

        returns:
            face_surface_normal_vectors_array (np.ndarray): the surface normal vectors corresponding to each GCS face.

        note:
            returned vectors are sampled from the normal vectors for the surface triangles, which are true
            normals, accounting for any difference between xy & z units for the surface crs
        """
        face_surface_normal_vectors_array = np.empty((triangle_per_face.size, 3), dtype = float)
        face_surface_normal_vectors_array[:] = surface_normal_vectors[triangle_per_face]
        if add_as_property:
            pc = rqp.PropertyCollection()
            pc.set_support(support = self)
            pc.add_cached_array_to_imported_list(face_surface_normal_vectors_array,
                                                 "computed from surface",
                                                 "normal vector",
                                                 uom = uom,
                                                 property_kind = "normal vector",
                                                 indexable_element = "faces",
                                                 points = True)
            pc.write_hdf5_for_imported_list()
            pc.create_xml_for_imported_list_and_add_parts_to_model()
        return face_surface_normal_vectors_array

    def grid_face_arrays(self,
                         property_uuid,
                         default_value = None,
                         feature_index = None,
                         active_only = True,
                         lazy = False,
                         baffle_uuid = None):
        """Creates a triplet of grid face numpy arrays populated from a property for this gcs.

        arguments:
            property_uuid (UUID): the uuid of the gcs property
            default_value (float or int, optional): the value to use in the grid property
                on faces that do not appear in the grid connection set; will default to
                np.NaN for continuous properties, -1 for categorical or discrete
            feature_index (int, optional): if present, only faces for this feature are used
            active_only (bool, default True): if True and an active property exists for the
                grid connection set, then only active faces are used when populating the
                grid face arrays
            lazy (bool, default False): if True, only the first cell & face of a pair is
                used when setting values in the arrays; if False, both left and right are
                used
            baffle_uuid (uuid, optional): if present, the uuid of a discrete (bool) property
                of the gcs holding baffle flags; where True the output face value is set
                to zero regardless of the main property value

        returns:
            triple numpy arrays: identifying the K, J & I direction grid face property values;
                shapes are (nk + 1, nj, ni), (nk, nj + 1, ni), (nk, nj, ni + 1) respectively

        notes:
            can only be used on single grid gcs; gcs property must have indexable of faces;
            at present generates grid properties with indexable 'faces' not 'faces per cell',
            which might not be appropriate for grids with split pillars (structural faults);
            points properties not currently supported; count must be 1
        """

        assert self.number_of_grids() == 1
        (nk, nj, ni) = self.grid_list[0].extent_kji
        active_mask = None
        if active_only:
            pc = self.extract_property_collection()
            active_mask = pc.single_array_ref(property_kind = 'active')
            if active_mask is not None:
                assert active_mask.shape == (self.count,)
        gcs_prop = rqp.Property(self.model, uuid = property_uuid)
        assert gcs_prop is not None
        assert bu.matching_uuids(gcs_prop.collection.support_uuid, self.uuid)
        assert gcs_prop.count() == 1
        assert not gcs_prop.is_points()
        dtype = float if gcs_prop.is_continuous() else int
        if default_value is None:
            default_value = -1 if dtype is int else np.NaN
        gcs_prop_array = gcs_prop.array_ref()
        log.debug(f'preparing grid face arrays from gcs property: {gcs_prop.title}; from gcs:{self.title}')

        baffle_mask = None
        if baffle_uuid is not None:
            baffle_mask = rqp.Property(self.model, uuid = baffle_uuid).array_ref()
            assert baffle_mask is not None

        # note that following arrays include external faces, in line with grid properties for 'faces'
        ak = np.full((nk + 1, nj, ni), default_value, dtype = dtype)
        aj = np.full((nk, nj + 1, ni), default_value, dtype = dtype)
        ai = np.full((nk, nj, ni + 1), default_value, dtype = dtype)
        # mk = np.zeros((nk + 1, nj, ni), dtype = bool)
        # mj = np.zeros((nk, nj + 1, ni), dtype = bool)
        # mi = np.zeros((nk, nj, ni + 1), dtype = bool)

        # populate arrays from faces of gcs, optionally filtered by feature index
        cip, fip = self.list_of_cell_face_pairs_for_feature_index(None)
        assert len(cip) == self.count and len(fip) == self.count
        assert gcs_prop_array.shape == (self.count,)
        if feature_index is None:
            indices = np.arange(self.count, dtype = int)
        else:
            indices = self.indices_for_feature_index(feature_index)

        # opposing_count = 0
        side_list = ([0] if lazy else [0, 1])
        for fi in indices:
            # fi = int(i)
            if active_mask is not None and not active_mask[fi]:
                continue
            value = gcs_prop_array[fi]
            if baffle_mask is not None and baffle_mask[fi]:
                value = 0  # will be cast to float (or bool) if needed when assigned below
            for side in side_list:
                cell_kji0 = cip[fi, side].copy()
                # opposing = cell_kji0.copy()
                axis, polarity = fip[fi, side]
                assert 0 <= axis <= 2 and 0 <= polarity <= 1
                cell_kji0[axis] += polarity
                # opposing[axis] += (1 - polarity)
                if axis == 0:
                    ak[tuple(cell_kji0)] = value
                    # mk[tuple(cell_kji0)] = True
                    # if mk[tuple(opposing)]:
                    #     opposing_count += 1
                elif axis == 1:
                    aj[tuple(cell_kji0)] = value
                    # mj[tuple(cell_kji0)] = True
                    # if mj[tuple(opposing)]:
                    #     opposing_count += 1
                else:
                    ai[tuple(cell_kji0)] = value
                    # mi[tuple(cell_kji0)] = True
                    # if mi[tuple(opposing)]:
                    #     opposing_count += 1

        # if opposing_count:
        #     log.warning(f'{opposing_count} suspicious opposing faces of {len(indices)} detected in gcs: {self.title}')
        # else:
        #     log.debug(f'no suspicious opposing faces detected in gcs: {self.title}')

        return (ak, aj, ai)
