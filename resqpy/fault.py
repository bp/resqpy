"""fault.py: Module providing resqml classes relating to fault representation."""

version = '2nd July 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('fault.py version ' + version)

import numpy as np
import pandas as pd
import os
# import xml.etree.ElementTree as et
# from lxml import etree as et

from resqpy.olio.base import BaseResqpy
import resqpy.olio.read_nexus_fault as rnf
import resqpy.olio.xml_et as rqet
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.uuid as bu
import resqpy.olio.trademark as tm
from resqpy.olio.xml_namespaces import curly_namespace as ns

import resqpy.organize as rqo


class GridConnectionSet(BaseResqpy):
   """Class for obj_GridConnectionSetRepresentation holding pairs of connected faces, usually for faults."""

   resqml_type = 'GridConnectionSetRepresentation'

   def __init__(self,
                parent_model,
                uuid = None,
                connection_set_root = None,
                grid = None,
                ascii_load_format = None,
                ascii_file = None,
                k_faces = None,
                j_faces = None,
                i_faces = None,
                feature_name = None,
                create_organizing_objects_where_needed = False,
                title = None,
                originator = None,
                extra_metadata = None):
      """Initializes a new GridConnectionSet and optionally loads it from xml or a list of simulator format ascii files.

      arguments:
         parent_model (model.Model object): the resqml model that this grid connection set will be part of
         uuid (uuid.UUID, optional): the uuid of an existing RESQML GridConnectionSetRepresentation from which
               this resqpy object is populated
         connection_set_root (DEPRECATED): use uuid instead; the root node of the xml tree for the
               obj_GridConnectionSet part; ignored if uuid is present
         grid (grid.Grid object, optional): If present, the grid object that this connection set relates to;
               if absent, the main grid for the parent model is assumed; only used if connection set root is
               None; see also notes
         ascii_load_format (string, optional): If present, must be 'nexus'; ignored if loading from xml;
               otherwise required if ascii_file is present
         ascii_file (string, optional): the full path of an ascii file holding fault definition data in
               nexus keyword format; ignored if loading from xml; otherwise, if present, ascii_load_format
               must also be set
         k_faces, j_faces, i_faces (boolean arrays, optional): if present, these arrays are used to identify
               which faces between logically neighbouring cells to include in the new grid connection set
         create_organizing_objects_where_needed (boolean, default False): if True when loading from ascii or
               face masks, a fault interpretation object and tectonic boundary feature object will be created
               for any named fault for which such objects do not exist; if False, missing organizational objects
               will cause an error to be logged; ignored when loading from xml
         title (str, optional): the citation title to use for a new grid connection set;
            ignored if uuid or connection_set_root is not None
         originator (str, optional): the name of the person creating the new grid connection set, defaults to login id;
            ignored if uuid or connection_set_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the grid connection set
            ignored if uuid or connection_set_root is not None

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
         if uuid is None and connection_set_root is None and ascii_file is None and k_faces, j_faces & i_faces are None,
         then an empty connection set is returned

      :meta common:
      """

      log.debug('initialising grid connection set')
      self.count = None  # number of face-juxtaposition pairs in this connection set
      self.cell_index_pairs = None  # shape (count, 2); dtype int; index normalized for flattened array
      self.cell_index_pairs_null_value = -1  # integer null value for array above
      self.grid_index_pairs = None  # shape (count, 2); dtype int; optional; used if more than one grid referenced
      self.face_index_pairs = None  # shape (count, 2); dtype int32; local to cell, ie. range 0 to 5
      self.face_index_pairs_null_value = -1  # integer null value for array above
      # NB face index values 0..5 usually mean [K-, K+, J+, I+, J-, I-] respectively but there is some ambiguity
      #    over I & J in the Energistics RESQML Usage Guide; see comments in DevOps backlog item 269001 for more info
      self.grid_list = []  # ordered list of grid objects, indexed by grid_index_pairs
      self.feature_indices = None  # shape (count,); dtype int; optional; which fault interpretation each pair is part of
      # note: resqml data structures allow a face pair to be part of more than one fault but this code restricts to one
      self.feature_list = None  # ordered list, actually of interpretations, indexed by feature_indices
      # feature list contains tuples: (content_type, uuid, title) for fault features (or other interpretations)

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
                       extra_metadata = extra_metadata,
                       root_node = connection_set_root)

      if self.root is None:
         log.debug('setting grid for new connection set to default (ROOT)')
         if grid is None:
            grid = self.model.grid(find_properties = False)  # should find only or ROOT grid
            assert grid is not None, 'No ROOT grid found in model'
         self.grid_list = [grid]
         if ascii_load_format and ascii_file:
            if ascii_load_format == 'nexus':
               log.debug('loading connection set (fault) faces from Nexus format ascii file: ' + ascii_file)
               tm.log_nexus_tm('debug')
               faces = rnf.load_nexus_fault_mult_table(ascii_file)
            else:
               log.debug('ascii format for connection set faces not handled by base resqpy code: ' + ascii_load_format)
            assert faces is not None, 'failed to load fault face information from file: ' + ascii_file
            self.set_pairs_from_faces_df(faces, create_organizing_objects_where_needed)
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
            self.set_pairs_from_face_masks(k_faces, j_faces, i_faces, feature_name,
                                           create_organizing_objects_where_needed)

   def _load_from_xml(self):
      root = self.root
      assert root is not None
      self.count = rqet.find_tag_int(root, 'Count')
      assert self.count > 0, 'empty grid connection set'
      self.cell_index_pairs_null_value = rqet.find_nested_tags_int(root, ['CellIndexPairs', 'NullValue'])
      self.face_index_pairs_null_value = rqet.find_nested_tags_int(root, ['LocalFacePerCellIndexPairs', 'NullValue'])
      # postpone loading of hdf5 array data till on-demand load (cell, grid & face index pairs)
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
         # for now, only accept faults
         assert feature_type in ['obj_FaultInterpretation', 'obj_HorizonInterpretation']
         self.feature_list.append((feature_type, feature_uuid, feature_title))
         log.debug('connection set references fault interpretation: ' + feature_title)
      log.debug('number of faults referred to in connection set: ' + str(len(self.feature_list)))
      assert len(self.feature_list) > 0, 'list of fault interpretation references is empty for connection set'
      # leave feature indices till on-demand load
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
      if len(self.grid_list) == 0:  # this code only needed to handle defective datasets generated by earlier versions!
         log.warning('no grid nodes found in xml for connection set')
         grid = self.model.grid(find_properties = False)  # should find only or ROOT grid
         assert grid is not None, 'No ROOT grid found in model'
         self.grid_list = [grid]

   def set_pairs_from_kelp(self, kelp_0, kelp_1, feature_name, create_organizing_objects_where_needed, axis = 'K'):
      """Sets cell_index_pairs and face_index_pairs based on j and i face kelp strands, using simple no throw pairing."""

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
      self.set_pairs_from_face_masks(k_faces, j_faces, i_faces, feature_name, create_organizing_objects_where_needed)

   def set_pairs_from_face_masks(self,
                                 k_faces,
                                 j_faces,
                                 i_faces,
                                 feature_name,
                                 create_organizing_objects_where_needed,
                                 feature_type = 'fault'):  # other feature_type values: 'horizon', 'geobody boundary'
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
      if fi_parts_list is None or len(fi_parts_list) == 0:
         log.warning('no interpretation parts found in model for ' + feature_type)
      fi_uuid = None
      for fi_part in fi_parts_list:
         if self.model.title_for_part(fi_part).split()[0].lower() == feature_name.lower():
            fi_uuid = self.model.uuid_for_part(fi_part)
            break
      if fi_uuid is None:
         if create_organizing_objects_where_needed:
            tbf_parts_list = self.model.parts_list_of_type(feature_flavour)
            tbf = None
            for tbf_part in tbf_parts_list:
               if feature_name.lower() == self.model.title_for_part(tbf_part).split()[0].lower():
                  tbf_root = self.model.root_for_part(tbf_part)
                  if feature_type == 'fault':
                     tbf = rqo.TectonicBoundaryFeature(self.model, root_node = tbf_root)
                  else:
                     tbf = rqo.GeneticBoundaryFeature(self.model, kind = feature_type, root_node = tbf_root)
                  break
            if tbf is None:
               if feature_type == 'fault':
                  tbf = rqo.TectonicBoundaryFeature(self.model, kind = 'fault', feature_name = feature_name)
               else:
                  tbf = rqo.GeneticBoundaryFeature(self.model, kind = feature_type, feature_name = feature_name)
               tbf_root = tbf.create_xml()
            if feature_type == 'fault':
               fi = rqo.FaultInterpretation(self.model, tectonic_boundary_feature = tbf,
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
      self.feature_list = [('obj_FaultInterpretation', fi_uuid, str(feature_name))]
      cell_pair_list = []
      face_pair_list = []
      nj_ni = grid.nj * grid.ni
      if k_faces is not None:
         for cell_kji0 in np.stack(np.where(k_faces)).T:
            cell = grid.natural_cell_index(cell_kji0)
            cell_pair_list.append((cell, cell + nj_ni))
            face_pair_list.append((self.face_index_map[0, 1], self.face_index_map[0, 0]))
      if j_faces is not None:
         for cell_kji0 in np.stack(np.where(j_faces)).T:
            cell = grid.natural_cell_index(cell_kji0)
            cell_pair_list.append((cell, cell + grid.ni))
            face_pair_list.append((self.face_index_map[1, 1], self.face_index_map[1, 0]))
      if i_faces is not None:
         for cell_kji0 in np.stack(np.where(i_faces)).T:
            cell = grid.natural_cell_index(cell_kji0)
            cell_pair_list.append((cell, cell + 1))
            face_pair_list.append((self.face_index_map[2, 1], self.face_index_map[2, 0]))
      self.cell_index_pairs = np.array(cell_pair_list, dtype = int)
      self.face_index_pairs = np.array(face_pair_list, dtype = int)
      self.count = len(self.cell_index_pairs)
      self.feature_indices = np.zeros(self.count, dtype = int)
      assert len(self.face_index_pairs) == self.count

   def set_pairs_from_faces_df(self, faces, create_organizing_objects_where_needed):
      """Sets cell_index_pairs and face_index_pairs based on pandas dataframe, using simple no throw pairing."""

      if len(self.grid_list) > 1:
         log.warning('setting grid connection set pairs from dataframe for first grid in list only')
      grid = self.grid_list[0]
      standardize_face_indicator_in_faces_df(faces)
      zero_base_cell_indices_in_faces_df(faces)
      faces = remove_external_faces_from_faces_df(faces, self.grid_list[0].extent_kji)
      self.feature_list = []
      cell_pair_list = []
      face_pair_list = []
      fi_list = []
      feature_index = 0
      name_list = faces['name'].unique()
      fi_parts_list = self.model.parts_list_of_type('FaultInterpretation')
      if fi_parts_list is None or len(fi_parts_list) == 0:
         log.warning('no fault interpretation parts found in model')
      fi_dict = {}  # maps fault name to interpretation uuid
      for fi_part in fi_parts_list:
         fi_dict[self.model.title_for_part(fi_part).split()[0].lower()] = self.model.uuid_for_part(fi_part)
      if create_organizing_objects_where_needed:
         tbf_parts_list = self.model.parts_list_of_type('TectonicBoundaryFeature')
      for name in name_list:
         # fetch uuid for fault interpretation object
         if name.lower() in fi_dict:
            fi_uuid = fi_dict[name.lower()]
         elif create_organizing_objects_where_needed:
            tbf = None
            for tbf_part in tbf_parts_list:
               if name.lower() == self.model.title_for_part(tbf_part).split()[0].lower():
                  tbf_root = self.model.root_for_part(tbf_part)
                  tbf = rqo.TectonicBoundaryFeature(self.model, root_node = tbf_root)
                  break
            if tbf is None:
               tbf = rqo.TectonicBoundaryFeature(self.model, feature_name = name)
               tbf_root = tbf.create_xml()
            fi = rqo.FaultInterpretation(self.model, tectonic_boundary_feature = tbf,
                                         is_normal = True)  # todo: set is_normal based on fault geometry in grid?
            fi_root = fi.create_xml(tbf_root)
            fi_uuid = rqet.uuid_for_part_root(fi_root)
            fi_dict[name.lower()] = fi_uuid
         else:
            log.error('no interpretation found for fault: ' + name)
            continue
         self.feature_list.append(('obj_FaultInterpretation', fi_uuid, str(name)))
         feature_faces = faces[faces['name'] == name]
         for i in range(len(feature_faces)):
            entry = feature_faces.iloc[i]
            f = entry['face']
            axis = 'KJI'.index(f[0])
            fp = '-+'.index(f[1])
            for k0 in range(entry['k1'], entry['k2'] + 1):
               for j0 in range(entry['j1'], entry['j2'] + 1):
                  for i0 in range(entry['i1'], entry['i2'] + 1):
                     neighbour = np.array([k0, j0, i0], dtype = int)
                     if fp:
                        neighbour[axis] += 1
                     else:
                        neighbour[axis] -= 1
                     fi_list.append(feature_index)
                     cell_pair_list.append((grid.natural_cell_index((k0, j0, i0)), grid.natural_cell_index(neighbour)))
                     face_pair_list.append((self.face_index_map[axis, fp], self.face_index_map[axis, 1 - fp]))
         feature_index += 1
      self.feature_indices = np.array(fi_list, dtype = int)
      self.cell_index_pairs = np.array(cell_pair_list, dtype = int)
      self.face_index_pairs = np.array(face_pair_list, dtype = int)
      self.count = len(self.cell_index_pairs)
      assert len(self.face_index_pairs) == self.count

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

   def filtered_by_layer_range(self, min_k0 = None, max_k0 = None, pare_down = True):
      """Returns a new GridConnectionSet, being a copy with cell faces whittled down to a layer range.

      arguments:
         min_k0 (int, optional): if present, the minimum layer number to be included (zero based)
         max_k0 (int, optional): if present, the maximum layer number to be included (zero based)
         pare_down (bool, default True): if True, any unused features in the new grid connection set will be removed
            and the feature indices adjusted appropriately; if False, unused features will be left in the list for
            the new connection set, meaning that the feature indices will be compatible with those for self

      returns:
         a new GridConnectionSet

      notes:
         cells in layer max_k0 are included in the filtered set (not pythonesque);
         currently only works for single grid connection sets
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
         return dupe
      mask = np.zeros(grid.extent_kji, dtype = bool)
      if min_k0 is not None and max_k0 is not None:
         mask[min_k0:max_k0 + 1, :, :] = True
      elif min_k0 is not None:
         mask[min_k0:, :, :] = True
      else:
         mask[:max_k0 + 1, :, :] = True
      return self.filtered_by_cell_mask(mask, pare_down = pare_down)

   def filtered_by_cell_mask(self, mask, both_cells_required = True, pare_down = True):
      """Returns a new GridConnectionSet, being a copy with cell faces whittled down by a boolean mask array.

      arguments:
         mask (numpy bool array of shape grid.extent_kji): connections will be kept for cells where this mask is True
         both_cells_required (bool, default True): if True, both cells involved in a connection must have a mask value
            of True to be included; if False, any connection where either cell has a True mask value will be included
         pare_down (bool, default True): if True, any unused features in the new grid connection set will be removed
            and the feature indices adjusted appropriately; if False, unused features will be left in the list for
            the new connection set, meaning that the feature indices will be compatible with those for self

      returns:
         a new GridConnectionSet

      note:
         currently only works for single grid connection sets
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
         return None
      masked_gcs = GridConnectionSet(self.model, grid = grid)
      masked_gcs.count = len(indices)
      masked_gcs.cell_index_pairs = self.cell_index_pairs[indices, :]
      masked_gcs.face_index_pairs = self.face_index_pairs[indices, :]
      masked_gcs.feature_indices = self.feature_indices[indices]
      masked_gcs.feature_list = self.feature_list.copy()
      if pare_down:
         masked_gcs.clean_feature_list()
      return masked_gcs

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
            self.count,), 'connection set face pair(s) not assigned to exactly one fault'  # rough check


#        delattr(self, 'fi_cl')  # assumed to be one-to-one mapping, so cumulative length is discarded

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

      self.cache_arrays()
      if self.feature_indices is None:
         return None
      matches = np.where(self.feature_indices == feature_index)
      if len(self.grid_list) == 1:
         return self.cell_index_pairs[matches], self.face_index_pairs[matches]
      assert self.grid_index_pairs is not None
      return self.cell_index_pairs[matches], self.face_index_pairs[matches], self.grid_index_pairs[matches]

   def list_of_cell_face_pairs_for_feature_index(self, feature_index):
      """Returns list of cell face pairs contributing to feature (fault) with given index.

      arguments:
         feature_index (non-negative integer): the index into the ordered feature list (fault interpretation list)

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
      simple_j_set = set(
      )  # set of (j0, i0) pairs of column indices where J+ faces contribute, as 2 element numpy arrays
      simple_i_set = set(
      )  # set of (j0, i0) pairs of column indices where I+ faces contribute, as 2 element numpy arrays
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

      one_to_one = np.empty(self.feature_indices.shape, dtype = int)
      for i in range(one_to_one.size):
         one_to_one[i] = i + 1

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
      # uuid/FaultIndices/elements  (N,)  uint32
      h5_reg.register_dataset(self.uuid, 'InterpretationIndices/elements', self.feature_indices)
      #      h5_reg.register_dataset(self.uuid, 'FaultIndices/elements', self.feature_indices)
      # uuid/FaultIndices/cumulativeLength  (N,)  uint32
      h5_reg.register_dataset(self.uuid, 'InterpretationIndices/cumulativeLength', one_to_one)
      #      h5_reg.register_dataset(self.uuid, 'FaultIndices/cumulativeLength', one_to_one)

      h5_reg.write(file_name, mode = mode)

   def create_xml(self, ext_uuid = None, add_as_part = True, add_relationships = True, title = None, originator = None):
      """Creates a Grid Connection Set (fault faces) xml node and optionally adds as child of root and/or to parts forest.

      :meta common:
      """

      # NB: only one grid handled for now
      # xml for grid(s) must be created before calling this method

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()
      if not self.title and not title:
         title = 'ROOT'

      gcs = super().create_xml(add_as_part = False, title = title, originator = originator)

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

            # for now, only support connection sets for faults
            if f_content_type == 'obj_FaultInterpretation':
               fi_part = rqet.part_name_for_object('obj_FaultInterpretation', f_uuid)
               fi_root = self.model.root_for_part(fi_part)
               self.model.create_ref_node('FeatureInterpretation',
                                          self.model.title_for_root(fi_root),
                                          f_uuid,
                                          content_type = 'obj_FaultInterpretation',
                                          root = ci_node)
            elif f_content_type == 'obj_HorizonInterpretation':
               fi_part = rqet.part_name_for_object('obj_HorizonInterpretation', f_uuid)
               fi_root = self.model.root_for_part(fi_part)
               self.model.create_ref_node('FeatureInterpretation',
                                          self.model.title_for_root(fi_root),
                                          f_uuid,
                                          content_type = 'obj_HorizonInterpretation',
                                          root = ci_node)
            else:
               raise Exception('unsupported content type in grid connection set')

      for grid in self.grid_list:
         self.model.create_ref_node('Grid',
                                    self.model.title_for_root(grid.root),
                                    grid.uuid,
                                    content_type = 'obj_IjkGridRepresentation',
                                    root = gcs)

      if add_as_part:
         self.model.add_part('obj_GridConnectionSetRepresentation', self.uuid, gcs)

         if add_relationships:
            for (obj_type, f_uuid, _) in self.feature_list:
               fi_part = rqet.part_name_for_object(obj_type, f_uuid)
               fi_root = self.model.root_for_part(fi_part)
               self.model.create_reciprocal_relationship(gcs, 'destinationObject', fi_root, 'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(gcs, 'mlToExternalPartProxy', ext_node, 'externalPartProxyToMl')
            for grid in self.grid_list:
               self.model.create_reciprocal_relationship(gcs, 'destinationObject', grid.root, 'sourceObject')

      return gcs

   def write_simulator(self, filename, mode = 'w', simulator = 'nexus', include_both_sides = False, use_minus = False):
      """Creates a Nexus include file holding FAULTS (or MULT) keyword and data."""

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

      def write_row(gcs, fp, name, i, j, k1, k2, axis, polarity):
         nonlocal row_count
         write_nexus_header_lines(fp, axis, polarity, name)
         fp.write('\t{0:1d}\t{1:1d}\t{2:1d}\t{3:1d}\t{4:1d}\t{5:1d}\t1.0\n'.format(i + 1, i + 1, j + 1, j + 1, k1 + 1,
                                                                                   k2 + 1))
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
            for side in sides:
               both = np.empty((cell_index_pairs.shape[0], 5), dtype = int)
               both[:, :2] = face_index_pairs[:, side, :]  # axis, polarity
               both[:, 2:] = cell_index_pairs[:, side, :]  # k, j, i
               df = pd.DataFrame(both, columns = ['axis', 'polarity', 'k', 'j', 'i'])
               df = df.sort_values(by = ['axis', 'polarity', 'j', 'i', 'k'])
               both_sorted = np.empty(both.shape, dtype = int)
               both_sorted[:] = df
               cell_indices = both_sorted[:, 2:]
               face_indices = np.empty((both_sorted.shape[0], 2), dtype = int)
               face_indices[:, :] = both_sorted[:, :2]
               del both_sorted
               del both
               del df
               k = None
               i = j = k2 = axis = polarity = None  # only needed to placate flake8 which whinges incorrectly otherwise
               for row in range(cell_indices.shape[0]):
                  kp, jp, ip = cell_indices[row]
                  axis_p, polarity_p = face_indices[row]
                  if k is not None:
                     if axis_p != axis or polarity_p != polarity or ip != i or jp != j or kp != k2 + 1:
                        write_row(self, fp, feature_name, i, j, k, k2, axis, polarity)
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
                  write_row(self, fp, feature_name, i, j, k, k2, axis, polarity)

   def get_column_edge_list_for_feature(self, feature, gridindex = 0, min_k = 0, max_k = 0):
      """Extracts a list of cell faces for a given feature index, over a given range of layers in the grid

      Args:
         feature - feature index
         gridindex - index of grid to be used in grid connection set gridlist, default 0
         min_k - minimum k layer, default 0
         max_k - maximum k layer, default 0
      Returns:
         list of cell faces for the feature (j_col, i_col, face_axis, face_polarity)
      """
      subgcs = self.filtered_by_layer_range(min_k0 = min_k, max_k0 = max_k, pare_down = False)

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
      """Generate a boolean aray defining which column edges are present for a given feature and k-layer range

      Args:
         feature - feature index
         gridindex - index of grid to be used in grid connection set gridlist, default 0
         min_k - minimum k layer
         max_k - maximum k layer
      Returns:
         boolean fault_by_column_edge_mask array (shape nj,ni,2,2)

      Note: the indices for the final two axes define the edges:
         the first defines j or i (0 or 1)
         the second negative or positive face (0 or 1)

         so [[True,False],[False,True]] indicates the -j and +i edges of the column are present
      """
      cell_face_list = self.get_column_edge_list_for_feature(feature, gridindex, min_k, max_k)

      fault_by_column_edge_mask = np.zeros((self.grid_list[gridindex].nj, self.grid_list[gridindex].ni, 2, 2),
                                           dtype = bool)
      for i in cell_face_list:
         fault_by_column_edge_mask[tuple(i)] = True

      return fault_by_column_edge_mask

   def get_property_by_feature_index_list(self,
                                          feature_index_list = None,
                                          property_name = 'Transmissibility multiplier'):
      """Returns a list of property values by feature.

      arguments:
         feature_index_list (list of int, optional): if present, the feature indices for which property values will be included
            in the resulting list; if None, values will be included for each feature in the feature_list for this connection set
         property_name (string, default 'Transmissibility multiplier'): the property name of interest, as used in the features'
            extra metadata as a key

      returns:
         list of float being the list of property values for the list of features, in corresponding order
      """

      if feature_index_list is None:
         feature_index_list = range(self.feature_list)
      value_list = []
      for feature_index in feature_index_list:
         _, feature_uuid, _ = self.feature_list[feature_index]
         feat = rqo.FaultInterpretation(parent_model = self.model, root_node = self.model.root(uuid = feature_uuid))
         if property_name not in feat.extra_metadata.keys():
            log.info(
               f'Property name {property_name} not found in extra_metadata for {self.model.citation_title_for_part(self.model.part_for_uuid(feature_uuid))}'
            )
         else:
            value_list.append(float(feat.extra_metadata[property_name]))
      return value_list

   def get_column_edge_float_array_for_feature(self,
                                               feature,
                                               fault_by_column_edge_mask,
                                               property_name = 'Transmissibility multiplier'):
      """Generate a float value aray defining the property valyes for different column edges present for a given feature and k-layer range

      Args:
         feature - feature index
         fault_by_column_edge_mask - fault_by_column_edge_mask with True on edges where feature is present
         property_name - name of property, should be present within the FaultInterpreation feature metadata
      Returns:
         float property_value_by_column_edge array (shape nj,ni,2,2)

      Note: the indices for the final two axes define the edges:
         the first defines j or i (0 or 1)
         the second negative or positive face (0 or 1)

         so [[1,np.nan],[np.nan,np.nan]] indicates the -j edge of the column are present with a value of 1
      """

      prop_values = self.get_property_by_feature_index_list(feature_index_list = [feature],
                                                            property_name = property_name)
      if prop_values == []:
         return None
      property_value_by_column_edge = np.where(fault_by_column_edge_mask, prop_values[0], np.nan)
      return property_value_by_column_edge

   def get_combined_fault_mask_index_value_arrays(self,
                                                  gridindex = 0,
                                                  min_k = 0,
                                                  max_k = 0,
                                                  property_name = 'Transmissibility multiplier',
                                                  feature_list = None):
      """Generate a combined mask, index and value arrays for all column edges across a k-layer range, for a defined feature_list

      Args:
         gridindex - index of grid to be used in grid connection set gridlist, default 0
         min_k - minimum k_layer
         max_k - maximum k_layer
         property_name - name of property, should be present within the FaultInterpreation feature metadata
         feature_list - list of feature index numbers to run for, default to all features
      Returns:
         bool array mask showing all column edges within features (shape nj,ni,2,2)
         int array showing the feature index for all column edges within features (shape nj,ni,2,2)
         float array showing the property value for all column edges within features (shape nj,ni,2,2)
      """
      self.cache_arrays()
      #     if feature_list is None: feature_list = np.unique(self.feature_indices)
      if feature_list is None:
         feature_list = np.arange(len(self.feature_list))
      sum_unmasked = None

      for i, feature in enumerate(feature_list):
         fault_by_column_edge_mask = self.get_column_edge_bool_array_for_feature(feature,
                                                                                 gridindex,
                                                                                 min_k = min_k,
                                                                                 max_k = max_k)
         property_value_by_column_edge = self.get_column_edge_float_array_for_feature(feature,
                                                                                      fault_by_column_edge_mask,
                                                                                      property_name = property_name)
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
                  combined_values = np.where(fault_by_column_edge_mask, property_value_by_column_edge, combined_values)
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
         apply_multipliers (boolean, default False): if True, a transmissibility multiplier for each feature is
            extracted from the feature extra metadata and applied to the transmissibility calculation

      returns:
         numpy float array of shape (count,) being the absolute transmissibilities across the connected cell face pairs;
         see notes regarding units

      notes:
         implicit units of measure of returned values will be m3.cP/(kPa.d) if grids' crs length units are metres,
         bbl.cP/(psi.d) if length units are feet; the computation is compatible with the Nexus NEWTRAN formulation;
         multiple grids are assumed to be in the same units and z units must be the same as xy units
      """

      feature_mult_list = self.get_property_by_feature_index_list() if apply_multipliers else None

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
         mult_tr = feature_mult_list[self.feature_indices[e]] if apply_multipliers else 1.0
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
      assert bu.matching_uuids(self.grid_list[0].uuid, featured.grid_list[0].uuid)

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
            self.model.copy_part_from_other_model(featured, featured.model.part(uuid = featured_uuid))

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


def pinchout_connection_set(grid, skip_inactive = True, feature_name = 'pinchout'):
   """Returns a new GridConnectionSet representing non-standard K face connections across pinchouts.

   arguments:
      grid (grid.Grid): the grid for which a pinchout connection set is required
      skip_inactive (boolean, default True): if True, connections are not included where there is an inactive cell
         above or below the pinchout; if False, such connections are included
      feature_name (string, default 'pinchout'): the name to use as citation title in the feature and interpretation

   notes:
      this function does not write to hdf5, nor create xml for the new grid connection set;
      however, it does create one feature and a corresponding interpretation and creates xml for those
   """

   assert grid is not None

   po = grid.pinched_out()
   dead = grid.extract_inactive_mask() if skip_inactive else None

   cip_list = []  # cell index pair list

   for j in range(grid.nj):
      for i in range(grid.ni):
         ka = 0
         while True:
            while ka < grid.nk - 1 and po[ka, j, i]:
               ka += 1
            while ka < grid.nk - 1 and not po[ka + 1, j, i]:
               ka += 1
            if ka >= grid.nk - 1:
               break
            # ka now in non-pinched out cell above pinchout
            if (skip_inactive and dead[ka, j, i]) or (grid.k_gaps and grid.k_gap_after[ka]):
               ka += 1
               continue
            kb = ka + 1
            while kb < grid.nk and po[kb, j, i]:
               kb += 1
            if kb >= grid.nk:
               break
            if skip_inactive and dead[kb, j, i]:
               ka = kb + 1
               continue
            # kb now beneath pinchout
            cip_list.append((grid.natural_cell_index((ka, j, i)), grid.natural_cell_index((kb, j, i))))
            ka = kb + 1

   log.debug(f'{len(cip_list)} pinchout connections found')

   pcs = _make_k_gcs_from_cip_list(grid, cip_list, feature_name)

   return pcs


def k_gap_connection_set(grid, skip_inactive = True, feature_name = 'k gap connection', tolerance = 0.001):
   """Returns a new GridConnectionSet representing K face connections where a K gap is zero thickness.

   arguments:
      grid (grid.Grid): the grid for which a K gap connection set is required
      skip_inactive (boolean, default True): if True, connections are not included where there is an inactive cell
         above or below the pinchout; if False, such connections are included
      feature_name (string, default 'pinchout'): the name to use as citation title in the feature and interpretation
      tolerance (float, default 0.001): the minimum vertical distance below which a K gap is deemed to be zero
         thickness; units are implicitly the z units of the coordinate reference system used by grid

   notes:
      this function does not write to hdf5, nor create xml for the new grid connection set;
      however, it does create one feature and a corresponding interpretation and creates xml for those;
      note that the entries in the connection set will be for logically K-neighbouring pairs of cells – such pairs
      are omitted from the standard transmissibilities due to the presence of the K gap layer
   """

   assert grid is not None
   if not grid.k_gaps:
      return None

   p = grid.points_ref(masked = False)
   dead = grid.extract_inactive_mask() if skip_inactive else None
   flip_z = (grid.k_direction_is_down != rqet.find_tag_bool(grid.crs_root, 'ZIncreasingDownward'))

   cip_list = []  # cell index pair list

   for k in range(grid.nk - 1):
      if grid.k_gap_after_array[k]:
         k_gap_pillar_z = p[grid.k_raw_index_array[k + 1]][..., 2] - p[grid.k_raw_index_array[k] + 1][..., 2]
         if grid.has_split_coordinate_lines:
            pfc = grid.create_column_pillar_mapping()  # pillars for column
            k_gap_z = 0.25 * np.sum(k_gap_pillar_z[pfc], axis = (2, 3))  # resulting shape (nj, ni)
         else:
            k_gap_z = 0.25 * (k_gap_pillar_z[:-1, :-1] + k_gap_pillar_z[:-1, 1:] + k_gap_pillar_z[1:, :-1] +
                              k_gap_pillar_z[1:, 1:])  # shape (nj, ni)
      if flip_z:
         k_gap_z = -k_gap_z
      layer_mask = np.logical_and(np.logical_not(np.isnan(k_gap_z)), k_gap_z < tolerance)
      if skip_inactive:
         layer_mask = np.logical_and(layer_mask, np.logical_not(np.logical_or(dead[k], dead[k + 1])))
      # layer mask now boolean array of shape (nj, ni) set True where connection needed
      ji_list = np.stack(np.where(layer_mask)).T  # numpy array being list of [j, i] pairs
      for (j, i) in ji_list:
         cip_list.append((grid.natural_cell_index((k, j, i)), grid.natural_cell_index((k + 1, j, i))))

   log.debug(f'{len(cip_list)} k gap connections found')

   kgcs = _make_k_gcs_from_cip_list(grid, cip_list, feature_name)

   return kgcs


def add_connection_set_and_tmults(model, fault_incl, tmult_dict = None):
   """Add a grid connection set to a resqml model, based on a fault include file and a dictionary of fault:tmult pairs.

   Grid connection set added to resqml model, with extra_metadata on the fault interpretation containing the MULTFL values

   Args:
      model: resqml model object
      fault_incl: fullpath to fault include file or list of fullpaths to fault include files
      tmult_dict: dictionary of fault name/transmissibility multiplier pairs (must align with faults in include file).
         Optional, if blank values in the fault.include file will be used instead
   Returns:
      grid connection set uuid
   """
   if tmult_dict is None:
      tmult_dict = {}
   if isinstance(fault_incl, list):
      if len(fault_incl) > 1:
         # Making a concatenated version of the faultincl files
         # TODO: perhaps a better/more unique name and location could be used in future?
         temp_faults = os.path.join(os.path.dirname(model.epc_file), 'faults_combined_temp.txt')
         with open(temp_faults, 'w') as outfile:
            log.debug("combining multiple include files into one")
            for fname in fault_incl:
               with open(fname) as infile:
                  for line in infile:
                     outfile.write(line)
      else:
         temp_faults = fault_incl[0]
   else:
      temp_faults = fault_incl
   faces_df = rnf.load_nexus_fault_mult_table(temp_faults)
   faults = set(faces_df['name'])
   for fault in faults:
      mults = set(faces_df[faces_df['name'] == fault]['mult'])
      if not len(mults) == 1:
         raise NotImplementedError(
            f'Expected a single multiplier value in the fault include files for each fault, cannot write fault {fault}')
      # TODO: Update this to save TX/TY/TZ as three different values
      log.debug(f"Working on {fault}")
      tbf = rqo.TectonicBoundaryFeature(parent_model = model, feature_name = fault, kind = 'fault')
      tbf.create_xml()
      fint = rqo.FaultInterpretation(parent_model = model, tectonic_boundary_feature = tbf, is_normal = True)
      if fault in tmult_dict.keys():
         mult = list(mults)[0] * tmult_dict[fault]  # multiply by our MULTFL from the input dictionary
      else:
         mult = list(mults)[0]
      fint.extra_metadata = {"Transmissibility multiplier": mult}
      fint.create_xml()
      model.store_epc()
      log.info(f"{fault} added to RESQML model")

   log.info("Creating grid connection set")

   gcs = GridConnectionSet(parent_model = model, ascii_load_format = 'nexus', ascii_file = temp_faults)
   gcs.write_hdf5(model.h5_file_name())
   gcs.create_xml(model.h5_uuid())

   model.store_epc()
   if os.path.exists(os.path.join(os.path.dirname(model.epc_file), 'faults_combined_temp.txt')):
      os.remove(os.path.join(os.path.dirname(model.epc_file), 'faults_combined_temp.txt'))
   log.info("Grid connection set added")

   return gcs.uuid


# fault face table pandas dataframe functions
# these functions are for processing dataframes that have been read from (or to be written to) simulator ascii files


def zero_base_cell_indices_in_faces_df(faces, reverse = False):
   """Decrements all the cell indices in the fault face dataframe, in situ (or increments if reverse is True)."""

   if reverse:
      offset = 1
   else:
      offset = -1
   for col in ['i1', 'i2', 'j1', 'j2', 'k1', 'k2']:
      temp = faces[col] + offset
      faces[col] = temp


def standardize_face_indicator_in_faces_df(faces):
   """Sets face indicators to uppercase I, J or K, always with + or - following direction, in situ."""

   # todo: convert XYZ into IJK respectively?
   temp = faces['face'].copy().str.upper()
   temp[faces['face'].str.len() == 1] = faces['face'][faces['face'].str.len() == 1] + '+'
   for u in temp.unique():
      s = str(u)
      assert len(s) == 2, 'incorrect length to face string in fault dataframe: ' + s
      assert s[0] in 'IJK', 'unknown direction (axis) character in fault dataframe: ' + s
      assert s[1] in '+-', 'unknown face polarity character in fault dataframe: ' + s
   faces['face'] = temp


def remove_external_faces_from_faces_df(faces, extent_kji, remove_all_k_faces = False):
   """Returns a subset of the rows of faces dataframe, excluding rows on external faces."""

   # NB: assumes cell indices have been converted to zero based
   # NB: ignores grid column, ie. assumes extent_kji is applicable to all rows
   # NB: assumes single layer of cells is specified in the direction of the face
   filtered = []
   max_k0 = extent_kji[0] - 1
   max_j0 = extent_kji[1] - 1
   max_i0 = extent_kji[2] - 1
   for i in range(len(faces)):
      entry = faces.iloc[i]
      f = entry['face']
      if ((entry['i1'] <= 0 and f == 'I-') or (entry['j1'] <= 0 and f == 'J-') or (entry['k1'] <= 0 and f == 'K-') or
          (entry['i2'] >= max_i0 and f == 'I+') or (entry['j2'] >= max_j0 and f == 'J+') or
          (entry['k2'] >= max_k0 and f == 'K+')):
         continue
      if remove_all_k_faces and f[0] == 'K':
         continue
      filtered.append(i)
   return faces.loc[filtered]


def _make_k_gcs_from_cip_list(grid, cip_list, feature_name):
   # cip (cell index pair) list contains pairs of natural cell indices for which k connection is required
   # first of pair is layer above (lower k to be precise), second is below (higher k)
   # called by pinchout_connection_set() and k_gap_connection_set() functions

   count = len(cip_list)

   if count == 0:
      return None

   pcs = GridConnectionSet(grid.model)
   pcs.grid_list = [grid]
   pcs.count = count
   pcs.grid_index_pairs = np.zeros((count, 2), dtype = int)
   pcs.cell_index_pairs = np.array(cip_list, dtype = int)
   pcs.face_index_pairs = np.zeros((count, 2), dtype = int)  # initialize to top faces
   pcs.face_index_pairs[:, 0] = 1  # bottom face of cells above pinchout

   pcs.feature_indices = np.zeros(count, dtype = int)  # could create seperate features by layer above or below?
   gbf = rqo.GeneticBoundaryFeature(grid.model, kind = 'horizon', feature_name = feature_name)
   gbf_root = gbf.create_xml()
   fi = rqo.HorizonInterpretation(grid.model, genetic_boundary_feature = gbf)
   fi_root = fi.create_xml(gbf_root, title_suffix = None)
   fi_uuid = rqet.uuid_for_part_root(fi_root)

   pcs.feature_list = [('obj_HorizonInterpretation', fi_uuid, str(feature_name))]

   return pcs
