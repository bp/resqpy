"""grid.py: Resqml grid module handling IJK cartesian grids."""

# note: only IJK Grid format supported at present
# see also rq_import.py

version = '19th August 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('grid.py version ' + version)

import pandas as pd
import numpy as np
import numpy.ma as ma
# import xml.etree.ElementTree as et
# from lxml import etree as et

from resqpy.olio.base import BaseResqpy
import resqpy.olio.transmission as rqtr
import resqpy.olio.fine_coarse as fc
import resqpy.olio.vector_utilities as vec
import resqpy.olio.grid_functions as gf
import resqpy.olio.write_data as wd
import resqpy.olio.point_inclusion as pip
import resqpy.olio.volume as vol
import resqpy.olio.uuid as bu
import resqpy.weights_and_measures as bwam
import resqpy.olio.xml_et as rqet
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.trademark as tm
from resqpy.olio.xml_namespaces import curly_namespace as ns

import resqpy.crs as rqc
import resqpy.property as rprop
import resqpy.fault as rqf

always_write_pillar_geometry_is_defined_array = False
always_write_cell_geometry_is_defined_array = False

# 'private' function


def _add_to_kelp_list(extent_kji, kelp_list, face_axis, ji):
   """
      :meta private:
   """
   if isinstance(face_axis, bool):
      face_axis = 'J' if face_axis else 'I'
   # ignore external faces
   if face_axis == 'J':
      if ji[0] < 0 or ji[0] >= extent_kji[1] - 1:
         return
   elif face_axis == 'I':
      if ji[1] < 0 or ji[1] >= extent_kji[2] - 1:
         return
   else:  # ji is actually kj or ki
      assert face_axis == 'K'
      if ji[0] < 0 or ji[0] >= extent_kji[0] - 1:
         return
   pair = ji
   if pair in kelp_list:
      return  # avoid duplication
   kelp_list.append(pair)


class Grid(BaseResqpy):
   """Class for RESQML Grid (extent and geometry) within RESQML model object."""

   resqml_type = 'IjkGridRepresentation'

   def __init__(self,
                parent_model,
                uuid = None,
                grid_root = None,
                find_properties = True,
                geometry_required = True,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Create a Grid object and optionally populate from xml tree.

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         grid_root (DEPRECATED): use uuid instead; the root of the xml tree for the grid part
         find_properties (boolean, default True): if True and uuid (or grid_root) is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         geometry_required (boolean, default True): if True and no geometry node exists in the xml,
            an assertion error is raised; ignored if uuid is None (and grid_root is None)
         title (str, optional): citation title for new grid; ignored if loading from xml
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if loading from xml
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if loading from xml

      returns:
         a newly created Grid object

      notes:
         only IJK grids are handled at the moment (the resqml standard also defines 5 other varieties)

      :meta common:
      """

      # note: currently only handles IJK grids
      # todo: check grid_root, if passed, is for an IJK grid
      self.parent_grid_uuid = None  #: parent grid when this is a local grid
      self.parent_window = None  #: FineCoarse cell index mapping info between self and parent grid
      self.is_refinement = None  #: True indicates self is a refinement wrt. parent; False means coarsening
      self.local_grid_uuid_list = None  #: LGR & LGC children list
      self.grid_representation = None  #: flavour of grid, currently 'IjkGrid' or 'IjkBlockGrid'; not much used
      self.geometry_root = None  #: xml node at root of geometry sub-tree
      self.extent_kji = None  #: size of grid: (nk, nj, ni)
      self.ni = self.nj = self.nk = None  #: duplicated extent information as individual integers
      self.nk_plus_k_gaps = None  #: int: nk + k_gaps
      self.crs_uuid = None  #: uuid of the coordinate reference system used by the grid's geometry
      self.crs_root = None  #: xml root node for the crs used by the grid's geometry
      self.points_cached = None  #: numpy array of raw points data; loaded on demand
      # Following are only relevant to structured grid varieties
      self.grid_is_right_handed = None  #: boolean indicating ijk handedness
      self.k_direction_is_down = None  #: boolean indicating dominant direction of k increase
      self.pillar_shape = None  #: string: often 'curved' is used, even for straight pillars
      self.has_split_coordinate_lines = None  #: boolean; affects dimensionality of points array
      self.split_pillars_count = None  #: int
      self.k_gaps = None  #: int; number of k gaps, or None
      self.k_gap_after_array = None  #: 1D numpy bool array of extent nk-1, or None
      self.k_raw_index_array = None  #: 1D numpy int array of extent nk, or None
      self.geometry_defined_for_all_pillars_cached = None
      self.geometry_defined_for_all_cells_cached = None
      self.xyz_box_cached = None  #: numpy array of shape (2, 3) being (min max, x y z)
      self.property_collection = None  #: GridPropertyCollection object
      self.inactive = None  #: numpy bool array: inactive cell mask (not native resqml - derived from active property)
      self.all_inactive = None  #: numpy bool indicating whether all cells are inactive
      self.active_property_uuid = None  #: uuid of property holding active cell boolean array (used to populate inactive)
      self.pinchout = None  #: numpy bool array: pinchout mask, only set on demand (not native resqml)
      self.grid_skin = None  #: outer skin of grid as a GridSkin object, computed and cached on demand
      self.stratigraphic_column_rank_uuid = None  #: optional reference for interpreting stratigraphic units
      self.stratigraphic_units = None  #: optional array of unit indices (one per layer or K gap)

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = grid_root)

      if not self.title:
         self.title = 'ROOT'

      if (uuid is not None or grid_root is not None):
         if geometry_required:
            assert self.geometry_root is not None, 'grid geometry not present in xml'
         if find_properties:
            self.extract_property_collection()

   def _load_from_xml(self):
      # Extract simple attributes from xml and set as attributes in this resqpy object
      grid_root = self.root
      assert grid_root is not None
      self.grid_representation = 'IjkGrid'  # this attribute not much used
      self.extract_extent_kji()
      self.nk = self.extent_kji[0]  # for convenience available as individual attribs as well as np triplet
      self.nj = self.extent_kji[1]
      self.ni = self.extent_kji[2]
      self.nk_plus_k_gaps = self.nk  # temporarily, set properly by self.extract_k_gaps()
      self.geometry_root = rqet.find_tag(grid_root, 'Geometry')
      if self.geometry_root is None:
         self.geometry_defined_for_all_pillars_cached = True
         self.geometry_defined_for_all_cells_cached = True
         self.pillar_shape = 'straight'
         self.has_split_coordinate_lines = False
         self.k_direction_is_down = True  # arbitrary, as 'down' is rather meaningless without a crs
      else:
         self.extract_crs_root()
         self.extract_crs_uuid()
         self.extract_has_split_coordinate_lines()
         self.extract_grid_is_right_handed()
         self.pillar_geometry_is_defined()  # note: if there is no geometry at all, resqpy sets this True
         self.cell_geometry_is_defined()  # note: if there is no geometry at all, resqpy sets this True
         self.extract_pillar_shape()
         self.extract_k_direction_is_down()
      self.extract_k_gaps()
      if self.geometry_root is None:
         assert not self.k_gaps, 'K gaps present in grid without geometry'
      self.extract_parent()
      self.extract_children()
      #        self.create_column_pillar_mapping()  # mapping now created on demand in other methods
      self.extract_inactive_mask()
      self.extract_stratigraphy()

   @property
   def grid_root(self):
      """Alias for root"""
      return self.root

   def set_modified(self, update_xml = False, update_hdf5 = False):
      """Assigns a new uuid to this grid; also calls set_modified() for parent model.

      arguments:
         update_xml (boolean, default False): if True, the uuid is modified in the xml tree
            for the grid part
         update_hdf5: (boolean, default False): if True, the uuid in the hdf5 internal path names
            for the datasets (arrays) for the grid are updated

      returns:
         the new uuid for this grid object

      notes:
         a resqml object should be thought of as immutable; therefore when modifying an object,
         it is preferable to assign it a new unique identifer which this method does for a grid;
         the hdf5 internal path names held in xml are only updated if both update_xml and update_hdf5
         are True;
         if the grid object has been created using the Model.copy_part() method, it is not
         necessary to call this function as a new uuid will already have been assigned;
         NB: relationships are not updated by this function, including the relationship to the
         hdf5 external part
      """

      old_uuid = self.uuid
      self.uuid = bu.new_uuid()
      if old_uuid is not None:
         log.info('changing uuid for grid from: ' + str(old_uuid) + ' to: ' + str(self.uuid))
      else:
         log.info('setting new uuid for grid: ' + str(self.uuid))
      if update_xml:
         rqet.patch_uuid_in_part_root(self.root, self.uuid)
         self.model.add_part('obj_IjkGridRepresentation', self.uuid, self.root)
         self.model.remove_part(rqet.part_name_for_object('obj_IjkGridRepresentation', old_uuid))
      if update_hdf5:
         hdf5_uuid_list = self.model.h5_uuid_list(self.root)
         for ext_uuid in hdf5_uuid_list:
            hdf5_file = self.model.h5_access(ext_uuid, mode = 'r+')
            rwh5.change_uuid(hdf5_file, old_uuid, self.uuid)
      if update_xml and update_hdf5:
         self.model.change_uuid_in_hdf5_references(self.root, old_uuid, self.uuid)
      self.model.set_modified()
      return self.uuid

   def extract_extent_kji(self):
      """Returns the grid extent; for IJK grids this is a 3 integer numpy array, order is Nk, Nj, Ni.

      returns:
         numpy int array of shape (3,) being number of cells in k, j & i axes respectively;
         the return value is cached in attribute extent_kji, which can alternatively be referenced
         directly by calling code as the value is set from xml on initialisation
      """

      if self.extent_kji is not None:
         return self.extent_kji
      self.extent_kji = np.ones(3, dtype = 'int')  # todo: handle other varieties of grid
      self.extent_kji[0] = int(rqet.find_tag(self.root, 'Nk').text)
      self.extent_kji[1] = int(rqet.find_tag(self.root, 'Nj').text)
      self.extent_kji[2] = int(rqet.find_tag(self.root, 'Ni').text)
      return self.extent_kji

   def cell_count(self, active_only = False, non_pinched_out_only = False, geometry_defined_only = False):
      """Returns number of cells in grid; optionally limited by active, non-pinched-out, or having geometry.

      arguments:
         active_only (boolean, default False): if True, the count of active cells is returned
         non_pinched_out_only (boolean, default False): if True, the count of cells with vertical
            thickness greater than 0.001 (units are crs vertical units) is returned
         geometry_defined_only (boolean, default False): if True, the count of cells which have a
            defined geometry is returned (a zero thickness cell may still have a defined geometry)

      returns:
         integer being the number of cells in the grid
      """

      # todo: elsewhere: setting of active array from boolean array or zero pore volume
      if not (active_only or non_pinched_out_only or geometry_defined_only):
         return np.prod(self.extent_kji)
      if non_pinched_out_only:
         self.pinched_out(cache_pinchout_array = True)
         return self.pinchout.size - np.count_nonzero(self.pinchout)
      if active_only:
         if self.all_inactive:
            return 0
         if self.inactive is not None:
            return self.inactive.size - np.count_nonzero(self.inactive)
         else:
            geometry_defined_only = True
      if geometry_defined_only:
         if self.geometry_defined_for_all_cells(cache_array = True):
            return np.prod(self.extent_kji)
         return np.count_nonzero(self.array_cell_geometry_is_defined)
      return None

   def natural_cell_index(self, cell_kji0):
      """Returns a single integer for the cell, being the index into a flattened array."""

      return (cell_kji0[0] * self.nj + cell_kji0[1]) * self.ni + cell_kji0[2]

   def natural_cell_indices(self, cell_kji0s):
      """Returns a numpy integer array with a value for each of the cells, being the index into a flattened array.

      argument:
         cell_kji0s: numpy integer array of shape (..., 3) being a list of cell indices in kji0 protocol

      returns:
         numpy integer array of shape (...,) being the equivalent natural cell indices (for a flattened array of cells)
      """

      return (cell_kji0s[..., 0] * self.nj + cell_kji0s[..., 1]) * self.ni + cell_kji0s[..., 2]

   def denaturalized_cell_index(self, c0):
      """Returns a 3 element cell_kji0 index (as a tuple) for the cell with given natural index."""

      k0, ji0 = divmod(c0, self.nj * self.ni)
      j0, i0 = divmod(ji0, self.ni)
      return (k0, j0, i0)

   def denaturalized_cell_indices(self, c0s):
      """Returns an integer numpy array of shape (..., 3) holding kji0 indices for the cells with given natural indices.

      argument:
         c0s: numpy integer array of shape (...,) being natural cell indices (for a flattened array)

      returns:
         numpy integer array of shape (..., 3) being the equivalent kji0 protocol cell indices
      """

      k0s, ji0s = divmod(c0s, self.nj * self.ni)
      j0s, i0s = divmod(ji0s, self.ni)
      return np.stack((k0s, j0s, i0s), axis = -1)

   def resolve_geometry_child(self, tag, child_node = None):
      """If xml child node is None, looks for tag amongst children of geometry root.

      arguments:
         tag (string): the tag of the geometry child node of interest
         child_node (optional): the already resolved xml root of the child, or None

      returns:
         xml node of child of geometry node for this grid, which matches tag

      note:
         if child_node argument is not None, it is simply returned;
         if child_node is None, the geometry node for this grid is scanned for a child with matching tag
      """

      if child_node is not None:
         return child_node
      return rqet.find_tag(self.geometry_root, tag)

   def extract_crs_uuid(self):
      """Returns uuid for coordinate reference system, as stored in geometry xml tree.

      returns:
         uuid.UUID object
      """

      if self.crs_uuid is not None:
         return self.crs_uuid
      crs_root = self.resolve_geometry_child('LocalCrs')
      uuid_str = rqet.find_tag_text(crs_root, 'UUID')
      if uuid_str:
         self.crs_uuid = bu.uuid_from_string(uuid_str)
      return self.crs_uuid

   def extract_crs_root(self):
      """Returns root in parent model xml parts forest of coordinate reference system used by this grid geomwtry.

      returns:
         root node in xml tree for coordinate reference system

      note:
         resqml allows a part to refer to another part that is not actually present in the same epc package;
         in practice, the crs is a tiny part and has always been included in datasets encountered so far;
         if the crs is not present, this method will return None (I think)
      """

      if self.crs_root is not None:
         return self.crs_root
      crs_uuid = self.extract_crs_uuid()
      if crs_uuid is None:
         return None
      self.crs_root = self.model.root(uuid = crs_uuid)
      return self.crs_root

   def extract_grid_is_right_handed(self):
      """Returns boolean indicating whether grid IJK axes are right handed, as stored in xml.

      returns:
         boolean: True if grid is right handed; False if left handed

      notes:
         this is the actual handedness of the IJK indexing of grid cells;
         the coordinate reference system has its own implicit handedness for xyz axes;
         Nexus requires the IJK space to be righthanded so if it is not, the handedness of the xyz space is
         falsified when exporting for Nexus (as Nexus allows xyz to be right or lefthanded and it is the
         handedness of the IJK space with respect to the xyz space that matters)
      """

      if self.grid_is_right_handed is not None:
         return self.grid_is_right_handed
      rh_node = self.resolve_geometry_child('GridIsRighthanded')
      if rh_node is None:
         return None
      self.grid_is_right_handed = (rh_node.text.lower() == 'true')
      return self.grid_is_right_handed

   def extract_k_direction_is_down(self):
      """Returns boolean indicating whether increasing K indices are generally for deeper cells, as stored in xml.

      returns:
         boolean: True if increasing K generally indicates increasing depth

      notes:
         resqml allows layers to fold back over themselves, so the relationship between k and depth might not
         be monotonic;
         higher level code sometimes requires k to increase with depth;
         independently of this, z values may increase upwards or downwards in a coordinate reference system
      """

      if self.k_direction_is_down is not None:
         return self.k_direction_is_down
      k_dir_node = self.resolve_geometry_child('KDirection')
      if k_dir_node is None:
         return None
      self.k_direction_is_down = (k_dir_node.text.lower() == 'down')
      return self.k_direction_is_down

   def extract_pillar_shape(self):
      """Returns string indicating whether whether pillars are curved, straight, or vertical as stored in xml.

      returns:
         string: either 'curved', 'straight' or 'vertical'

      note:
         resqml datasets often have 'curved', even when the pillars are actually 'vertical' or 'straight';
         use actual_pillar_shape() method to determine the shape from the actual xyz points data
      """

      if self.pillar_shape is not None:
         return self.pillar_shape
      ps_node = self.resolve_geometry_child('PillarShape')
      if ps_node is None:
         return None
      self.pillar_shape = ps_node.text
      return self.pillar_shape

   def extract_has_split_coordinate_lines(self):
      """Returns boolean indicating whether grid geometry has any split coordinate lines (split pillars, ie. faults).

      returns:
         boolean: True if the grid has one or more split pillars; False if all pillars are unsplit

      notes:
         the return value is based on the array elements present in the xml tree, unless it has already been
         determined;
         resqml ijk grids with split coordinate lines have extra arrays compared to unfaulted grids, and the main
         Points array is indexed differently: [k', pillar_index, xyz] instead of [k', j', i', xyz] (where k', j', i'
         range of nk+k_gaps+1, nj+1, ni+1 respectively)
      """

      if self.has_split_coordinate_lines is not None:
         return self.has_split_coordinate_lines
      split_node = self.resolve_geometry_child('SplitCoordinateLines')
      self.has_split_coordinate_lines = (split_node is not None)
      if split_node is not None:
         self.split_pillars_count = int(rqet.find_tag(split_node, 'Count').text.strip())
      return self.has_split_coordinate_lines

   def extract_k_gaps(self):
      """Returns information about gaps (voids) between layers in the grid.

      returns:
         (int, numpy bool array, numpy int array) being the number of gaps between layers;
         a 1D bool array of extent nk-1 set True where there is a gap below the layer; and
         a 1D int array being the k index to actually use in the points data for each layer k0

      notes:
         all returned elements are stored as attributes in the grid object; int and bool array elements
         will be None if there are no k gaps; each k gap implies an extra element in the points data for
         each pillar; when wanting to index k interfaces (horizons) rather than layers, the last of the
         returned values can be used to index the k axis of the points data to yield the top face of the
         layer and the successor in k will always index the basal face of the same layer
      """

      if self.k_gaps is not None:
         self.nk_plus_k_gaps = self.nk + self.k_gaps
         return self.k_gaps, self.k_gap_after_array, self.k_raw_index_array
      self.k_gaps = rqet.find_nested_tags_int(self.root, ['KGaps', 'Count'])
      if self.k_gaps:
         self.nk_plus_k_gaps = self.nk + self.k_gaps
         k_gap_after_root = rqet.find_nested_tags(self.root, ['KGaps', 'GapAfterLayer'])
         assert k_gap_after_root is not None
         bool_array_type = rqet.node_type(k_gap_after_root)
         assert bool_array_type == 'BooleanHdf5Array'  # could be a constant array but not handled by this code
         h5_key_pair = self.model.h5_uuid_and_path_for_node(k_gap_after_root)
         assert h5_key_pair is not None
         self.model.h5_array_element(h5_key_pair,
                                     index = None,
                                     cache_array = True,
                                     object = self,
                                     array_attribute = 'k_gap_after_array',
                                     dtype = 'bool')
         assert hasattr(self, 'k_gap_after_array')
         assert self.k_gap_after_array.ndim == 1 and self.k_gap_after_array.size == self.nk - 1
         self._set_k_raw_index_array()
      else:
         self.nk_plus_k_gaps = self.nk
         self.k_gap_after_array = None
         self.k_raw_index_array = np.arange(self.nk, dtype = int)
      return self.k_gaps, self.k_gap_after_array, self.k_raw_index_array

   def _set_k_raw_index_array(self):
      """Sets the layering raw index array based on the k gap after boolean array."""
      if self.k_gap_after_array is None:
         self.k_raw_index_array = None
         return
      self.k_raw_index_array = np.empty((self.nk,), dtype = int)
      gap_count = 0
      for k in range(self.nk):
         self.k_raw_index_array[k] = k + gap_count
         if k < self.nk - 1 and self.k_gap_after_array[k]:
            gap_count += 1
      assert gap_count == self.k_gaps, 'inconsistency in k gap data'

   def extract_stratigraphy(self):
      """Loads stratigraphic information from xml."""

      self.stratigraphic_column_rank_uuid = None
      self.stratigraphic_units = None
      strata_node = rqet.find_tag(self.root, 'IntervalStratigraphicUnits')
      if strata_node is None:
         return
      self.stratigraphic_column_rank_uuid =  \
         bu.uuid_from_string(rqet.find_nested_tags_text(strata_node, ['StratigraphicOrganization', 'UUID']))
      assert self.stratigraphic_column_rank_uuid is not None
      unit_indices_node = rqet.find_tag(strata_node, 'UnitIndices')
      h5_key_pair = self.model.h5_uuid_and_path_for_node(unit_indices_node)
      self.model.h5_array_element(h5_key_pair,
                                  index = None,
                                  cache_array = True,
                                  object = self,
                                  array_attribute = 'stratigraphic_units',
                                  dtype = 'int')
      assert len(self.stratigraphic_units) == self.nk_plus_k_gaps

   def extract_parent(self):
      """Loads fine:coarse mapping information between this grid and parent, if any, returning parent grid uuid."""

      class IntervalsInfo:

         def __init__(self):
            pass

      if self.extent_kji is None:
         self.extract_extent_kji()
      if self.parent_grid_uuid is not None:
         return self.parent_grid_uuid
      self.parent_window = None  # FineCoarse cell index mapping info with respect to parent
      self.is_refinement = None
      pw_node = rqet.find_tag(self.root, 'ParentWindow')
      if pw_node is None:
         return None
      # load a FineCoarse object as parent_window attirbute and set parent_grid_uuid attribute
      self.parent_grid_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(pw_node, ['ParentGrid', 'UUID']))
      assert self.parent_grid_uuid is not None
      parent_grid_root = self.model.root(uuid = self.parent_grid_uuid)
      if parent_grid_root is None:
         log.warning('parent grid not present in model, unable to treat as local grid')
         return None
      # etxract parent grid extent directly from xml to avoid risk of circular references
      parent_grid_extent_kji = np.array((rqet.find_tag_int(
         parent_grid_root, 'Nk'), rqet.find_tag_int(parent_grid_root, 'Nj'), rqet.find_tag_int(parent_grid_root, 'Ni')),
                                        dtype = int)
      parent_initials = []
      intervals_count_list = []
      parent_count_list_list = []
      child_count_list_list = []
      child_weight_list_list = []
      refining_flag = None  # gets set True if local grid is a refinement, False if a coarsening
      parent_box = np.zeros((2, 3), dtype = int)
      for axis in range(3):
         regrid_node = rqet.find_tag(pw_node, 'KJI'[axis] + 'Regrid')
         assert regrid_node is not None
         pii = rqet.find_tag_int(regrid_node, 'InitialIndexOnParentGrid')
         assert pii is not None and 0 <= pii < parent_grid_extent_kji[axis]
         parent_initials.append(pii)
         parent_box[0, axis] = pii
         intervals_node = rqet.find_tag(regrid_node, 'Intervals')
         if intervals_node is None:  # implicit one-to-one mapping
            intervals_count_list.append(1)
            parent_count_list_list.append(np.array(self.extent_kji[axis], dtype = int))
            parent_box[1, axis] = parent_box[0, axis] + self.extent_kji[axis] - 1
            assert parent_box[1, axis] < parent_grid_extent_kji[axis]
            child_count_list_list.append(np.array(self.extent_kji[axis], dtype = int))
            child_weight_list_list.append(None)
         else:
            intervals_info = IntervalsInfo()
            intervals_count = rqet.find_tag_int(intervals_node, 'IntervalCount')
            assert intervals_count is not None and intervals_count > 0
            pcpi_node = rqet.find_tag(intervals_node, 'ParentCountPerInterval')
            assert pcpi_node is not None
            h5_key_pair = self.model.h5_uuid_and_path_for_node(pcpi_node)
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        index = None,
                                        cache_array = True,
                                        object = intervals_info,
                                        array_attribute = 'parent_count_per_interval',
                                        dtype = 'int')
            assert hasattr(intervals_info, 'parent_count_per_interval')
            assert intervals_info.parent_count_per_interval.ndim == 1 and intervals_info.parent_count_per_interval.size == intervals_count
            parent_box[1, axis] = parent_box[0, axis] + np.sum(intervals_info.parent_count_per_interval) - 1
            assert parent_box[1, axis] < parent_grid_extent_kji[axis]
            ccpi_node = rqet.find_tag(intervals_node, 'ChildCountPerInterval')
            assert ccpi_node is not None
            h5_key_pair = self.model.h5_uuid_and_path_for_node(ccpi_node)
            assert h5_key_pair is not None
            self.model.h5_array_element(h5_key_pair,
                                        index = None,
                                        cache_array = True,
                                        object = intervals_info,
                                        array_attribute = 'child_count_per_interval',
                                        dtype = 'int')
            assert hasattr(intervals_info, 'child_count_per_interval')
            assert intervals_info.child_count_per_interval.ndim == 1 and intervals_info.child_count_per_interval.size == intervals_count
            assert np.sum(intervals_info.child_count_per_interval) == self.extent_kji[
               axis]  # assumes both local and parent grids are IjkGrids
            for interval in range(intervals_count):
               if intervals_info.child_count_per_interval[interval] == intervals_info.parent_count_per_interval[
                     interval]:
                  continue  # one-to-one
               if refining_flag is None:
                  refining_flag = (intervals_info.child_count_per_interval[interval] >
                                   intervals_info.parent_count_per_interval[interval])
               assert refining_flag == (intervals_info.child_count_per_interval[interval] > intervals_info.parent_count_per_interval[interval]),  \
                  'mixture of refining and coarsening in one local grid – allowed by RESQML but not handled by this code'
               if refining_flag:
                  assert intervals_info.child_count_per_interval[interval] % intervals_info.parent_count_per_interval[interval] == 0,  \
                     'within a single refinement interval, fine and coarse cell boundaries are not obviously aligned'
               else:
                  assert intervals_info.parent_count_per_interval[interval] % intervals_info.child_count_per_interval[interval] == 0,  \
                     'within a single coarsening interval, fine and coarse cell boundaries are not obviously aligned'
            ccw_node = rqet.find_tag(intervals_node, 'ChildCellWeights')
            if ccw_node is None:
               intervals_info.child_cell_weights = None
            else:
               h5_key_pair = self.model.h5_uuid_and_path_for_node(ccw_node)
               assert h5_key_pair is not None
               self.model.h5_array_element(h5_key_pair,
                                           index = None,
                                           cache_array = True,
                                           object = intervals_info,
                                           array_attribute = 'child_cell_weights',
                                           dtype = 'float')
               assert hasattr(intervals_info, 'child_cell_weights')
               assert intervals_info.child_cell_weights.ndim == 1 and intervals_info.child_cell_weights.size == self.extent_kji[
                  axis]
            intervals_count_list.append(intervals_count)
            parent_count_list_list.append(intervals_info.parent_count_per_interval)
            child_count_list_list.append(intervals_info.child_count_per_interval)
            child_weight_list_list.append(intervals_info.child_cell_weights)
      cell_overlap_node = rqet.find_tag(pw_node, 'CellOverlap')
      if cell_overlap_node is not None:
         log.warning('ignoring cell overlap information in grid relationship')
      omit_node = rqet.find_tag(pw_node, 'OmitParentCells')
      if omit_node is not None:
         log.warning('unable to handle parent cell omissions in local grid definition – ignoring')
      # todo: handle omissions

      if refining_flag is None:
         log.warning('local grid has no refinement nor coarsening – treating as a refined grid')
         refining_flag = True
      self.is_refinement = refining_flag

      if refining_flag:  # local grid is a refinement
         self.parent_window = fc.FineCoarse(self.extent_kji,
                                            parent_box[1] - parent_box[0] + 1,
                                            within_coarse_box = parent_box)
         for axis in range(3):
            if intervals_count_list[axis] == 1:
               self.parent_window.set_constant_ratio(axis)
               constant_ratio = self.extent_kji[axis] // (parent_box[1, axis] - parent_box[0, axis] + 1)
               ratio_vector = None
            else:
               constant_ratio = None
               ratio_vector = child_count_list_list[axis] // parent_count_list_list[axis]
               self.parent_window.set_ratio_vector(axis, ratio_vector)
            if child_weight_list_list[axis] is None:
               self.parent_window.set_equal_proportions(axis)
            else:
               proportions_list = []
               place = 0
               for coarse_slice in range(parent_box[1, axis] - parent_box[0, axis] + 1):
                  if ratio_vector is None:
                     proportions_list.append(np.array(child_weight_list_list[axis][place:place + constant_ratio]))
                     place += constant_ratio
                  else:
                     proportions_list.append(
                        np.array(child_weight_list_list[axis][place:place + ratio_vector[coarse_slice]]))
                     place += ratio_vector[coarse_slice]
               self.parent_window.set_proportions_list_of_vectors(axis, proportions_list)

      else:  # local grid is a coarsening
         self.parent_window = fc.FineCoarse(parent_box[1] - parent_box[0] + 1,
                                            self.extent_kji,
                                            within_fine_box = parent_box)
         for axis in range(3):
            if intervals_count_list[axis] == 1:
               self.parent_window.set_constant_ratio(axis)
               constant_ratio = (parent_box[1, axis] - parent_box[0, axis] + 1) // self.extent_kji[axis]
               ratio_vector = None
            else:
               constant_ratio = None
               ratio_vector = parent_count_list_list[axis] // child_count_list_list[axis]
               self.parent_window.set_ratio_vector(axis, ratio_vector)
            if child_weight_list_list[axis] is None:
               self.parent_window.set_equal_proportions(axis)
            else:
               proportions_list = []
               place = 0
               for coarse_slice in range(self.extent_kji[axis]):
                  if ratio_vector is None:
                     proportions_list.append(np.array(child_weight_list_list[axis][place:place + constant_ratio]))
                     place += constant_ratio
                  else:
                     proportions_list.append(
                        np.array(child_weight_list_list[axis][place:place + ratio_vector[coarse_slice]]))
                     place += ratio_vector[coarse_slice]
               self.parent_window.set_proportions_list_of_vectors(axis, proportions_list)

      self.parent_window.assert_valid()

      return self.parent_grid_uuid

   def set_parent(self, parent_grid_uuid, self_is_refinement, parent_window):
      """Set relationship with respect to a parent grid.

      arguments:
         parent_grid_uuid (uuid.UUID): the uuid of the parent grid
         self_is_refinement (boolean): if True, this grid is a refinement of the subset of the parent grid;
            if False, this grid is a coarsening
         parent_window: (olio.fine_coarse.FineCoarse object): slice mapping information in K, j & I axes; note
            that self_is_refinement determines which of the 2 grids is fine and which is coarse
      """

      if self.parent_grid_uuid is not None:
         log.warning('overwriting parent grid information')
      self.parent_grid_uuid = parent_grid_uuid
      if parent_grid_uuid is None:
         self.parent_window = None
         self.is_refinement = None
      else:
         parent_window.assert_valid()
         self.parent_window = parent_window
         self.is_refinement = self_is_refinement

   def extract_children(self):
      assert self.uuid is not None
      if self.local_grid_uuid_list is not None:
         return self.local_grid_uuid_list
      self.local_grid_uuid_list = []
      related_grid_roots = self.model.roots(obj_type = 'IjkGridRepresentation', related_uuid = self.uuid)
      if related_grid_roots is not None:
         for related_root in related_grid_roots:
            parent_uuid = rqet.find_nested_tags_text(related_root, ['ParentWindow', 'ParentGrid', 'UUID'])
            if parent_uuid is None:
               continue
            parent_uuid = bu.uuid_from_string(parent_uuid)
            if bu.matching_uuids(self.uuid, parent_uuid):
               self.local_grid_uuid_list.append(parent_uuid)
      return self.local_grid_uuid_list

   def extract_property_collection(self):
      """Load grid property collection object holding lists of all properties in model that relate to this grid.

      returns:
         resqml_property.GridPropertyCollection object

      note:
         a reference to the grid property collection is cached in this grid object; if the properties change,
         for example by generating some new properties, the property_collection attribute of the grid object
         would need to be reset to None elsewhere before calling this method again
      """

      if self.property_collection is not None:
         return self.property_collection
      self.property_collection = rprop.GridPropertyCollection(grid = self)
      return self.property_collection

   def extract_inactive_mask(self, check_pinchout = False):
      """Returns boolean numpy array indicating which cells are inactive, if (in)active property found in this grid.

      returns:
         numpy array of booleans, of shape (nk, nj, ni) being True for cells which are inactive; False for active

      note:
         RESQML does not have a built-in concept of inactive (dead) cells, though the usage guide advises to use a
         discrete property with a local property kind of 'active'; this resqpy code can maintain an 'inactive'
         attribute for the grid object, which is a boolean numpy array indicating which cells are inactive
      """

      if self.inactive is not None and not check_pinchout:
         return self.inactive
      geom_defined = self.cell_geometry_is_defined_ref()
      if self.inactive is None:
         if geom_defined is None or geom_defined is True:
            self.inactive = np.zeros(tuple(self.extent_kji))  # ie. all active
         else:
            self.inactive = np.logical_not(self.cell_geometry_is_defined_ref())
      if check_pinchout:
         self.inactive = np.logical_or(self.inactive, self.pinched_out())
      gpc = self.extract_property_collection()
      if gpc is None:
         self.all_inactive = np.all(self.inactive)
         return self.inactive
      active_gpc = rprop.GridPropertyCollection()
      # note: use of bespoke (local) property kind 'active' as suggested in resqml usage guide
      active_gpc.inherit_parts_selectively_from_other_collection(other = gpc,
                                                                 property_kind = 'active',
                                                                 continuous = False)
      if active_gpc.number_of_parts() > 0:
         if active_gpc.number_of_parts() > 1:
            log.warning('more than one property found with bespoke kind "active", using last encountered')
         active_part = active_gpc.parts()[-1]
         active_array = active_gpc.cached_part_array_ref(active_part, dtype = 'bool')
         self.inactive = np.logical_or(self.inactive, np.logical_not(active_array))
         self.active_property_uuid = active_gpc.uuid_for_part(active_part)
         active_gpc.uncache_part_array(active_part)
      else:  # for backward compatibility with earlier versions of resqpy
         inactive_gpc = rprop.GridPropertyCollection()
         inactive_gpc.inherit_parts_selectively_from_other_collection(other = gpc,
                                                                      property_kind = 'code',
                                                                      facet_type = 'what',
                                                                      facet = 'inactive')
         if inactive_gpc.number_of_parts() == 1:
            inactive_part = inactive_gpc.parts()[0]
            inactive_array = inactive_gpc.cached_part_array_ref(inactive_part, dtype = 'bool')
            self.inactive = np.logical_or(self.inactive, inactive_array)
            inactive_gpc.uncache_part_array(inactive_part)

      self.all_inactive = np.all(self.inactive)
      return self.inactive

   def cell_geometry_is_defined(self, cell_kji0 = None, cell_geometry_is_defined_root = None, cache_array = True):
      """Returns True if the geometry of the specified cell is defined; can also be used to cache (load) the boolean array.

      arguments:
         cell_kji0 (triplet of integer, optional): if present, the index of the cell of interest, in kji0 protocol;
            if False, None is returned but the boolean array can still be cached
         cell_geometry_is_defined_root (optional): if present, the root of the 'cell geometry is defined' xml tree for
            this grid; this optional argument is to allow for speed optimisation, to save searching for the node
         cache_array (boolean, default True): if True, the 'cell geometry is defined' array is cached in memory, unless
            the xml tree indicates that geometry is defined for all cells, in which case that is noted

      returns:
         if cell_kji0 is not None, a boolean is returned indicating whether geometry is defined for that cell;
         if cell_kji0 is None, None is returned (but the array caching logic will have been applied)
      """

      if self.geometry_defined_for_all_cells_cached:
         return True
      if hasattr(self, 'array_cell_geometry_is_defined') and self.array_cell_geometry_is_defined is None:
         delattr(self, 'array_cell_geometry_is_defined')
      if hasattr(self, 'array_cell_geometry_is_defined'):
         self.geometry_defined_for_all_cells_cached = np.all(self.array_cell_geometry_is_defined)
         if self.geometry_defined_for_all_cells_cached:
            return True
         if cell_kji0 is None:
            return False
         return self.array_cell_geometry_is_defined[tuple(cell_kji0)]
      is_def_root = self.resolve_geometry_child('CellGeometryIsDefined', child_node = cell_geometry_is_defined_root)
      if is_def_root is None:
         points = self.points_ref(masked = False)
         assert points is not None
         self.geometry_defined_for_all_cells_cached = not np.any(np.isnan(points))
         if self.geometry_defined_for_all_cells_cached or cell_kji0 is None:
            return self.geometry_defined_for_all_cells_cached
      is_def_type = rqet.node_type(is_def_root)
      if is_def_type == 'BooleanConstantArray':
         self.geometry_defined_for_all_cells_cached = (rqet.find_tag_text(is_def_root, 'Value').lower() == 'true')
         return self.geometry_defined_for_all_cells_cached
      else:
         assert (is_def_type == 'BooleanHdf5Array')
         h5_key_pair = self.model.h5_uuid_and_path_for_node(is_def_root)
         if h5_key_pair is None:
            return None
         result = self.model.h5_array_element(h5_key_pair,
                                              index = cell_kji0,
                                              cache_array = cache_array,
                                              object = self,
                                              array_attribute = 'array_cell_geometry_is_defined',
                                              dtype = 'bool')
         if self.geometry_defined_for_all_cells_cached is None and cache_array and hasattr(
               self, 'array_cell_geometry_is_defined'):
            self.geometry_defined_for_all_cells_cached = (np.count_nonzero(
               self.array_cell_geometry_is_defined) == self.array_cell_geometry_is_defined.size)
            if self.geometry_defined_for_all_cells_cached:
               delattr(self, 'array_cell_geometry_is_defined')
         return result

   def pillar_geometry_is_defined(self, pillar_ji0 = None, pillar_geometry_is_defined_root = None, cache_array = True):
      """Returns True if the geometry of the specified pillar is defined; False otherwise; can also be used to cache (load) the boolean array.

      arguments:
         pillar_ji0 (pair of integers, optional): if present, the index of the pillar of interest, in ji0 protocol;
            if False, None is returned but the boolean array can still be cached
         pillar_geometry_is_defined_root (optional): if present, the root of the 'pillar geometry is defined' xml tree for
            this grid; this optional argument is to allow for speed optimisation, to save searching for the node
         cache_array (boolean, default True): if True, the 'pillar geometry is defined' array is cached in memory, unless
            the xml tree indicates that geometry is defined for all pillars, in which case that is noted

      returns:
         if pillar_ji0 is not None, a boolean is returned indicating whether geometry is defined for that pillar;
         if pillar_ji0 is None, None is returned unless geometry is defined for all pillars in which case True is returned
      """

      if self.geometry_defined_for_all_pillars_cached:
         return True
      if hasattr(self, 'array_pillar_geometry_is_defined'):
         if pillar_ji0 is None:
            return None  # this option allows caching of array without actually referring to any pillar
         return self.array_pillar_geometry_is_defined[tuple(pillar_ji0)]
      is_def_root = self.resolve_geometry_child('PillarGeometryIsDefined', child_node = pillar_geometry_is_defined_root)
      if is_def_root is None:
         return True  # maybe default should be False?
      is_def_type = rqet.node_type(is_def_root)
      if is_def_type == 'BooleanConstantArray':
         assert rqet.find_tag(is_def_root, 'Value').text.lower() == 'true'
         self.geometry_defined_for_all_pillars_cached = True
         return True
      else:
         assert is_def_type == 'BooleanHdf5Array'
         h5_key_pair = self.model.h5_uuid_and_path_for_node(is_def_root)
         if h5_key_pair is None:
            return None
         result = self.model.h5_array_element(h5_key_pair,
                                              index = pillar_ji0,
                                              cache_array = cache_array,
                                              object = self,
                                              array_attribute = 'array_pillar_geometry_is_defined',
                                              dtype = 'bool')
         if self.geometry_defined_for_all_pillars_cached is None and cache_array and hasattr(
               self, 'array_pillar_geometry_is_defined'):
            self.geometry_defined_for_all_pillars_cached = (np.count_nonzero(
               self.array_pillar_geometry_is_defined) == self.array_pillar_geometry_is_defined.size)
            if self.geometry_defined_for_all_pillars_cached:
               del self.array_pillar_geometry_is_defined  # memory optimisation
         return result

   def geometry_defined_for_all_cells(self, cache_array = True):
      """Returns True if geometry is defined for all cells; False otherwise.

      argument:
         cache_array (boolean, default True): if True, the 'cell geometry is defined' array is cached in memory,
            unless the xml indicates that geometry is defined for all cells, in which case that is noted

      returns:
         boolean: True if geometry is defined for all cells; False otherwise
      """

      if self.geometry_defined_for_all_cells_cached is not None:
         return self.geometry_defined_for_all_cells_cached
      if cache_array:
         self.cell_geometry_is_defined(cache_array = True)
         return self.geometry_defined_for_all_cells_cached
      # loop over all cells (until a False is encountered) – only executes if cache_array is False
      cell_geom_defined_root = self.resolve_geometry_child('CellGeometryIsDefined')
      if cell_geom_defined_root is not None:
         for k0 in range(self.nk):
            for j0 in range(self.nj):
               for i0 in range(self.ni):
                  if not self.cell_geometry_is_defined(cell_kji0 = (k0, j0, i0),
                                                       cell_geometry_is_defined_root = cell_geom_defined_root,
                                                       cache_array = False):
                     self.geometry_defined_for_all_cells_cached = False
                     return False
      self.geometry_defined_for_all_cells_cached = True
      return True

   def geometry_defined_for_all_pillars(self, cache_array = True, pillar_geometry_is_defined_root = None):
      """Returns True if geometry is defined for all pillars; False otherwise.

      arguments:
         cache_array (boolean, default True): if True, the 'pillar geometry is defined' array is cached in memory,
            unless the xml indicates that geometry is defined for all pillars, in which case that is noted
         pillar_geometry_is_defined_root (optional): if present, the root of the 'pillar geometry is defined' xml tree for
            this grid; this optional argument is to allow for speed optimisation, to save searching for the node

      returns:
         boolean: True if the geometry is defined for all pillars; False otherwise
      """

      if self.geometry_defined_for_all_pillars_cached is not None:
         return self.geometry_defined_for_all_pillars_cached
      if cache_array:
         self.pillar_geometry_is_defined(cache_array = cache_array)
         return self.geometry_defined_for_all_pillars_cached
      is_def_root = self.resolve_geometry_child('PillarGeometryIsDefined', child_node = pillar_geometry_is_defined_root)
      self.geometry_defined_for_all_pillars_cached = True
      if is_def_root is not None:
         for pillar_j in range(self.nj):
            for pillar_i in range(self.ni):
               if not self.pillar_geometry_is_defined(
                  [pillar_j, pillar_i], pillar_geometry_is_defined_root = is_def_root, cache_array = False):
                  self.geometry_defined_for_all_pillars_cached = False
                  break
            if not self.geometry_defined_for_all_pillars_cached:
               break
      return self.geometry_defined_for_all_pillars_cached

   def cell_geometry_is_defined_ref(self):
      """Returns an in-memory numpy array containing the boolean data indicating which cells have geometry defined.

      returns:
         numpy array of booleans of shape (nk, nj, ni); True value indicates cell has geometry defined; False
         indicates that the cell's geometry (points xyz values) cannot be used

      note:
         if geometry is flagged in the xml as being defined for all cells, then this function returns None;
         geometry_defined_for_all_cells() can be used to test for that situation
      """

      # todo: treat this array like any other property?; handle constant array seamlessly?
      self.cell_geometry_is_defined(cache_array = True)
      if hasattr(self, 'array_cell_geometry_is_defined'):
         return self.array_cell_geometry_is_defined
      return None  # can happen, if geometry is defined for all cells

   def pillar_geometry_is_defined_ref(self):
      """Returns an in-memory numpy array containing the boolean data indicating which pillars have geometry defined.

      returns:
         numpy array of booleans of shape (nj + 1, ni + 1); True value indicates pillar has geometry defined (at
         least for some points); False indicates that the pillar's geometry (points xyz values) cannot be used;
         the resulting array only covers primary pillars; extra pillars for split pillars always have geometry
         defined

      note:
         if geometry is flagged in the xml as being defined for all pillars, then this function returns None
      """

      # todo: double-check behaviour in presence of split pillars
      # todo: treat this array like any other property?; handle constant array seamlessly?
      self.pillar_geometry_is_defined(cache_array = True)
      if hasattr(self, 'array_pillar_geometry_is_defined'):
         return self.array_pillar_geometry_is_defined
      return None  # can happen, if geometry is defined for all pillars

   def set_geometry_is_defined(self,
                               treat_as_nan = None,
                               treat_dots_as_nan = False,
                               complete_partial_pillars = False,
                               nullify_partial_pillars = False,
                               complete_all = False):
      """Sets cached flags and/or arrays indicating which primary pillars have any points defined and which cells all points.

      arguments:
         treat_as_nan (float, optional): if present, any point with this value as x, y or z is changed
            to hold NaN values, which is the correct RESQML representation of undefined values
         treat_dots_as_nan (boolean, default False): if True, the points around any inactive cell which has zero length along
            all its I and J edges will be set to NaN (which can intentionally invalidate the geometry of neighbouring cells)
         complete_partial_pillars (boolean, default False): if True, pillars which have some but not all points defined will
            have values generated for the undefined (NaN) points
         nullify_partial_pillars (boolean, default False): if True, pillars which have some undefined (NaN) points will be
            treated as if all the points on the pillar are undefined
         complete_all (boolean, default False): if True, values will be generated for all undefined points (includes
            completion of partial pillars if both partial pillar arguments are False)

      notes:
         this method discards any previous information about which pillars and cells have geometry defined; the new settings
         are based solely on where points data is NaN (or has the value supplied as treat_as_nan etc.);
         the inactive attribute is also updated by this method, though any cells previously flagged as inactive will still be
         inactive;
         if points are generated due to either complete... argument being set True, the inactive mask is set prior to
         generating points, so all cells making use of generated points will be inactive; however, the geometry will show
         as defined where points have been generated;
         at most one of complete_partial_pillars and nullify_partial_pillars may be True;
         although the method modifies the cached (attribute) copies of various arrays, they are not written to hdf5 here
      """

      def infill_partial_pillar(grid, pillar_index):
         points = grid.points_ref(masked = False).reshape((grid.nk_plus_k_gaps + 1, -1, 3))
         nan_mask = np.isnan(points[:, pillar_index, 0])
         first_k = 0
         while first_k < grid.nk_plus_k_gaps + 1 and nan_mask[first_k]:
            first_k += 1
         assert first_k < grid.nk_plus_k_gaps + 1
         if first_k > 0:
            points[:first_k, pillar_index] = points[first_k, pillar_index]
         last_k = grid.nk_plus_k_gaps
         while nan_mask[last_k]:
            last_k -= 1
         if last_k < grid.nk_plus_k_gaps:
            points[last_k + 1:, pillar_index] = points[last_k, pillar_index]
         while True:
            while first_k < last_k and not nan_mask[first_k]:
               first_k += 1
            if first_k >= last_k:
               break
            scan_k = first_k + 1
            while nan_mask[scan_k]:
               scan_k += 1
            points[first_k - 1 : scan_k, pillar_index] =  \
               np.linspace(points[first_k - 1, pillar_index], points[scan_k, pillar_index],
                           num = scan_k - first_k + 1, endpoint = False)
            first_k = scan_k

      def create_surround_masks(top_nan_mask):
         assert top_nan_mask.ndim == 2
         nj1, ni1 = top_nan_mask.shape  # nj + 1, ni + 1
         surround_mask = np.zeros(top_nan_mask.shape, dtype = bool)
         coastal_mask = np.zeros(top_nan_mask.shape, dtype = bool)
         for j in range(nj1):
            i = 0
            while i < ni1 and top_nan_mask[j, i]:
               coastal_mask[j, i] = True
               i += 1
            if i < ni1:
               i = ni1 - 1
               while top_nan_mask[j, i]:
                  coastal_mask[j, i] = True
                  i -= 1
            else:
               surround_mask[j] = True
               coastal_mask[j] = False
         for i in range(ni1):
            j = 0
            while j < nj1 and top_nan_mask[j, i]:
               coastal_mask[j, i] = True
               j += 1
            if j < nj1:
               j = nj1 - 1
               while top_nan_mask[j, i]:
                  coastal_mask[j, i] = True
                  j -= 1
            else:
               surround_mask[:, i] = True
               coastal_mask[:, i] = False
         return surround_mask, coastal_mask

      def fill_holes(grid, holes_mask):
         log.debug(f'filling {np.count_nonzero(holes_mask)} pillars for holes')
         points = grid.points_ref(masked = False).reshape(grid.nk_plus_k_gaps + 1, -1, 3)
         ni_plus_1 = grid.ni + 1
         mask_01 = np.empty(holes_mask.shape, dtype = int)
         while np.any(holes_mask):
            flat_holes_mask = holes_mask.flatten()
            mask_01[:] = np.where(holes_mask, 0, 1)
            modified = False
            # fix isolated NaN pillars with 4 neighbours
            neighbours = np.zeros(holes_mask.shape, dtype = int)
            neighbours[:-1, :] += mask_01[1:, :]
            neighbours[1:, :] += mask_01[:-1, :]
            neighbours[:, :-1] += mask_01[:, 1:]
            neighbours[:, 1:] += mask_01[:, :-1]
            foursomes = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 4))[0]
            if len(foursomes) > 0:
               interpolated = 0.25 * (points[:, foursomes - 1, :] + points[:, foursomes + 1, :] +
                                      points[:, foursomes - ni_plus_1, :] + points[:, foursomes + ni_plus_1, :])
               points[:, foursomes, :] = interpolated
               flat_holes_mask[foursomes] = False
               modified = True
            # fix NaN pillars with defined opposing neighbours in -J and +J
            neighbours[:] = 0
            neighbours[:-1, :] += mask_01[1:, :]
            neighbours[1:, :] += mask_01[:-1, :]
            twosomes = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
            if len(twosomes) > 0:
               interpolated = 0.5 * (points[:, twosomes - ni_plus_1, :] + points[:, twosomes + ni_plus_1, :])
               points[:, twosomes, :] = interpolated
               flat_holes_mask[twosomes] = False
               modified = True
            # fix NaN pillars with defined opposing neighbours in -I and +I
            neighbours[:] = 0
            neighbours[:, :-1] += mask_01[:, 1:]
            neighbours[:, 1:] += mask_01[:, :-1]
            twosomes = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
            if len(twosomes) > 0:
               interpolated = 0.5 * (points[:, twosomes - 1, :] + points[:, twosomes + 1, :])
               points[:, twosomes, :] = interpolated
               flat_holes_mask[twosomes] = False
               modified = True
            # fix NaN pillars with defined cornering neighbours in J- and I-
            neighbours[:] = 0
            neighbours[1:, :] += mask_01[:-1, :]
            neighbours[:, 1:] += mask_01[:, :-1]
            corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
            neighbours[1:, 1:] += mask_01[:-1, :-1]
            pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
            if len(corners) > 0:
               interpolated = 0.5 * (points[:, corners - ni_plus_1, :] + points[:, corners - 1, :])
               points[:, corners, :] = interpolated
               pushed = 2.0 * points[:, pushable, :] - points[:, pushable - ni_plus_1 - 1, :]
               points[:, pushable, :] = pushed
               flat_holes_mask[corners] = False
               modified = True
            # fix NaN pillars with defined cornering neighbours in J- and I+
            neighbours[:] = 0
            neighbours[1:, :] += mask_01[:-1, :]
            neighbours[:, :-1] += mask_01[:, 1:]
            corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
            neighbours[1:, :-1] += mask_01[:-1, 1:]
            pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
            if len(corners) > 0:
               interpolated = 0.5 * (points[:, corners - ni_plus_1, :] + points[:, corners + 1, :])
               points[:, corners, :] = interpolated
               pushed = 2.0 * points[:, pushable, :] - points[:, pushable - ni_plus_1 + 1, :]
               points[:, pushable, :] = pushed
               flat_holes_mask[corners] = False
               modified = True
            # fix NaN pillars with defined cornering neighbours in J+ and I-
            neighbours[:] = 0
            neighbours[:-1, :] += mask_01[1:, :]
            neighbours[:, 1:] += mask_01[:, :-1]
            corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
            neighbours[:-1, 1:] += mask_01[1:, :-1]
            pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
            if len(corners) > 0:
               interpolated = 0.5 * (points[:, corners + ni_plus_1, :] + points[:, corners - 1, :])
               points[:, corners, :] = interpolated
               pushed = 2.0 * points[:, pushable, :] - points[:, pushable + ni_plus_1 - 1, :]
               points[:, pushable, :] = pushed
               flat_holes_mask[corners] = False
               modified = True
            # fix NaN pillars with defined cornering neighbours in J+ and I+
            neighbours[:] = 0
            neighbours[:-1, :] += mask_01[1:, :]
            neighbours[:, :-1] += mask_01[:, 1:]
            corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
            neighbours[:-1, :-1] += mask_01[1:, 1:]
            pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
            if len(corners) > 0:
               interpolated = 0.5 * (points[:, corners + ni_plus_1, :] + points[:, corners + 1, :])
               points[:, corners, :] = interpolated
               pushed = 2.0 * points[:, pushable, :] - points[:, pushable + ni_plus_1 + 1, :]
               points[:, pushable, :] = pushed
               flat_holes_mask[corners] = False
               modified = True
            holes_mask = flat_holes_mask.reshape((grid.nj + 1, ni_plus_1))
            if not modified:
               log.warning('failed to fill all holes in grid geometry')
               break

      def fill_surround(grid, surround_mask):
         # note: only fills x,y; based on bottom layer of points; assumes surround mask is a regularly shaped frame of columns
         log.debug(f'filling {np.count_nonzero(surround_mask)} pillars for surround')
         points = grid.points_ref(masked = False)
         points_view = points[-1, :, :2].reshape((-1, 2))[:(grid.nj + 1) * (grid.ni + 1), :].reshape(
            (grid.nj + 1, grid.ni + 1, 2))
         modified = False
         if grid.nj > 1:
            j_xy_vector = np.nanmean(points_view[1:, :] - points_view[:-1, :])
            j = 0
            while j < grid.nj and np.all(surround_mask[j, :]):
               j += 1
            assert j < grid.nj
            while j > 0:
               points_view[j - 1, :] = points_view[j, :] - j_xy_vector
               modified = True
               j -= 1
            j = grid.nj - 1
            while j >= 0 and np.all(surround_mask[j, :]):
               j -= 1
            assert j >= 0
            while j < grid.nj - 1:
               points_view[j + 1, :] = points_view[j, :] + j_xy_vector
               modified = True
               j += 1
         if grid.ni > 1:
            i_xy_vector = np.nanmean(points_view[:, 1:] - points_view[:, :-1])
            i = 0
            while i < grid.ni and np.all(surround_mask[:, i]):
               i += 1
            assert i < grid.ni
            while i > 0:
               points_view[:, i - 1] = points_view[:, i] - i_xy_vector
               modified = True
               i -= 1
            i = grid.ni - 1
            while i >= 0 and np.all(surround_mask[:, i]):
               i -= 1
            assert i >= 0
            while i < grid.ni - 1:
               points_view[:, i + 1] = points_view[:, i] + i_xy_vector
               modified = True
               i += 1
         if modified:
            points.reshape((grid.nk_plus_k_gaps + 1, -1, 3))[:-1, :(grid.nj + 1) * (grid.ni + 1), :2][:, surround_mask.flatten(), :] =  \
               points_view[surround_mask, :].reshape((1, -1, 2))

      assert not (complete_partial_pillars and nullify_partial_pillars)
      if complete_all and not nullify_partial_pillars:
         complete_partial_pillars = True

      points = self.points_ref(masked = False)

      if treat_as_nan is not None:
         nan_mask = np.any(np.logical_or(np.isnan(points), points == treat_as_nan), axis = -1)
      else:
         nan_mask = np.any(np.isnan(points), axis = -1)

      if treat_dots_as_nan:
         areal_dots = self.point_areally()
         some_areal_dots = np.any(areal_dots)
      else:
         areal_dots = None
         some_areal_dots = False

      self.geometry_defined_for_all_pillars_cached = None
      if hasattr(self, 'array_pillar_geometry_is_defined'):
         del self.array_pillar_geometry_is_defined
      self.geometry_defined_for_all_cells_cached = None
      if hasattr(self, 'array_cell_geometry_is_defined'):
         del self.array_cell_geometry_is_defined

      if not np.any(nan_mask) and not some_areal_dots:
         self.geometry_defined_for_all_pillars_cached = True
         self.geometry_defined_for_all_cells_cached = True
         return

      if some_areal_dots:
         # inject NaNs into the pillars around any cell that has zero length in I and J
         if self.k_gaps:
            dot_mask = np.zeros((self.nk_plus_k_gaps + 1, self.nj + 1, self.ni + 1), dtype = bool)
            dot_mask[self.k_raw_index_array, :-1, :-1] = areal_dots
            dot_mask[self.k_raw_index_array + 1, :-1, :-1] = np.logical_or(
               dot_mask[self.k_raw_index_array + 1, :-1, :-1], areal_dots)
         else:
            dot_mask = np.zeros((self.nk + 1, self.nj + 1, self.ni + 1), dtype = bool)
            dot_mask[:-1, :-1, :-1] = areal_dots
            dot_mask[1:, :-1, :-1] = np.logical_or(dot_mask[:-1, :-1, :-1], areal_dots)
         dot_mask[:, 1:, :-1] = np.logical_or(dot_mask[:, :-1, :-1], dot_mask[:, 1:, :-1])
         dot_mask[:, :, 1:] = np.logical_or(dot_mask[:, :, :-1], dot_mask[:, :, 1:])
         if self.has_split_coordinate_lines:
            # only set points in primary pillars to NaN; todo: more thorough to consider split pillars too
            primaries = (self.nj + 1) * (self.ni + 1)
            nan_mask[:, :primaries] = np.logical_or(nan_mask[:, :primaries], dot_mask.reshape((-1, primaries)))
         else:
            nan_mask = np.where(dot_mask, np.NaN, nan_mask)

      assert not np.all(nan_mask), 'grid does not have any geometry defined'

      points[:] = np.where(np.repeat(np.expand_dims(nan_mask, axis = nan_mask.ndim), 3, axis = -1), np.NaN, points)

      surround_z = self.xyz_box(lazy = False)[1 if self.z_inc_down() else 0, 2]

      pillar_defined_mask = np.logical_not(np.all(nan_mask, axis = 0)).flatten()
      primary_count = (self.nj + 1) * (self.ni + 1)
      if np.all(pillar_defined_mask):
         self.geometry_defined_for_all_pillars_cached = True
      else:
         self.geometry_defined_for_all_pillars_cached = False
         self.array_pillar_geometry_is_defined = pillar_defined_mask[:primary_count].reshape((self.nj + 1, self.ni + 1))
      if pillar_defined_mask.size > primary_count and not np.all(pillar_defined_mask[primary_count:]):
         log.warning('at least one split pillar has geometry undefined')

      self.geometry_defined_for_all_cells_cached = False

      primary_nan_mask =  \
         nan_mask.reshape((self.nk_plus_k_gaps + 1, -1))[:, :primary_count].reshape((self.nk_plus_k_gaps + 1, self.nj + 1, self.ni + 1))
      column_nan_mask = np.logical_or(np.logical_or(primary_nan_mask[:, :-1, :-1], primary_nan_mask[:, :-1, 1:]),
                                      np.logical_or(primary_nan_mask[:, 1:, :-1], primary_nan_mask[:, 1:, 1:]))
      if self.k_gaps:
         self.array_cell_geometry_is_defined = np.logical_not(
            np.logical_or(column_nan_mask[self.k_raw_index_array], column_nan_mask[self.k_raw_index_array + 1]))
      else:
         self.array_cell_geometry_is_defined = np.logical_not(np.logical_or(column_nan_mask[:-1], column_nan_mask[1:]))

      if hasattr(self, 'inactive') and self.inactive is not None:
         self.inactive = np.logical_or(self.inactive, np.logical_not(self.array_cell_geometry_is_defined))
      else:
         self.inactive = np.logical_not(self.array_cell_geometry_is_defined)
      self.all_inactive = np.all(self.inactive)

      if self.geometry_defined_for_all_cells_cached:
         return

      cells_update_needed = False

      if nullify_partial_pillars:
         partial_pillar_mask = np.logical_and(pillar_defined_mask, np.any(nan_mask, axis = 0).flatten())
         if np.any(partial_pillar_mask):
            points.reshape((self.nk_plus_k_gaps + 1, -1, 3))[:, partial_pillar_mask, :] = np.NaN
            cells_update_needed = True
      elif complete_partial_pillars:
         partial_pillar_mask = np.logical_and(pillar_defined_mask, np.any(nan_mask, axis = 0).flatten())
         if np.any(partial_pillar_mask):
            log.warning('completing geometry for partially defined pillars')
            for pillar_index in np.where(partial_pillar_mask)[0]:
               infill_partial_pillar(self, pillar_index)
            cells_update_needed = True

      if complete_all:
         # note: each pillar is either fully defined or fully undefined at this point
         top_nan_mask = np.isnan(points[0, ..., 0].flatten()[:(self.nj + 1) * (self.ni + 1)].reshape(
            (self.nj + 1, self.ni + 1)))
         surround_mask, coastal_mask = create_surround_masks(top_nan_mask)
         holes_mask = np.logical_and(top_nan_mask, np.logical_not(surround_mask))
         if np.any(holes_mask):
            fill_holes(self, holes_mask)
         if np.any(surround_mask):
            fill_surround(self, surround_mask)
         # set z values for coastal and surround to max z for grid
         surround_mask = np.logical_or(surround_mask, coastal_mask).flatten()
         if np.any(surround_mask):
            points.reshape(self.nk_plus_k_gaps + 1, -1, 3)[:, :(self.nj + 1) * (self.ni + 1)][:, surround_mask,
                                                                                              2] = surround_z
         self.geometry_defined_for_all_pillars_cached = True
         if hasattr(self, 'array_pillar_geometry_is_defined'):
            del self.array_pillar_geometry_is_defined
         cells_update_needed = False
         assert not np.any(np.isnan(points))
         self.geometry_defined_for_all_cells_cached = True
         if hasattr(self, 'array_cell_geometry_is_defined'):
            del self.array_cell_geometry_is_defined

      if cells_update_needed:
         # note: each pillar is either fully defined or fully undefined at this point
         if self.geometry_defined_for_all_pillars_cached:
            self.geometry_defined_for_all_cells_cached = True
            if hasattr(self, 'array_cell_geometry_is_defined'):
               del self.array_cell_geometry_is_defined
         else:
            top_nan_mask = np.isnan(points[0, ..., 0].flatten()[:(self.nj + 1) * (self.ni + 1)].reshape(
               (self.nj + 1, self.ni + 1)))
            column_nan_mask = np.logical_or(np.logical_or(top_nan_mask[:-1, :-1], top_nan_mask[:-1, 1:]),
                                            np.logical_or(top_nan_mask[1:, :-1], top_nan_mask[1:, 1:]))
            self.array_cell_geometry_is_defined = np.repeat(np.expand_dims(column_nan_mask, 0), self.nk, axis = 0)
            self.geometry_defined_for_all_cells_cached = np.all(self.array_cell_geometry_is_defined)
            if self.geometry_defined_for_all_cells_cached:
               del self.array_cell_geometry_is_defined

   def actual_pillar_shape(self, patch_metadata = False, tolerance = 0.001):
      """Returns actual shape of pillars.

      arguments:
         patch_metadata (boolean, default False): if True, the actual shape replaces whatever was in the metadata
         tolerance (float, default 0.001): a length value (in units of grid xy units) used as a Manhattan distance
            limit in the xy plane when considering whether a point lies 'on' a straight line

      returns:
         string: 'vertical', 'straight' or 'curved'

      note:
         setting patch_metadata True will affect the attribute in this Grid object; however, it will not be
         preserved unless the create_xml() method is called, followed at some point with model.store_epc()
      """

      pillar_shape = gf.actual_pillar_shape(self.points_ref(masked = False), tolerance = tolerance)
      if patch_metadata:
         self.pillar_shape = pillar_shape
      return pillar_shape

   def cache_all_geometry_arrays(self):
      """Loads from hdf5 into memory all the arrays defining the grid geometry.

      returns:
         None

      notes:
         call this method if much grid geometry processing is coming up, to save having to worry about
         individual caching arguments to many other methods;
         this method does not create a column to pillar mapping which will often also be needed;
         the arrays are cached as direct attributes to this grid object;
         the names, shapes and types of the attributes are:
            array_cell_geometry_is_defined   (nk, nj, ni)  bool
            array_pillar_geometry_is_defined (nj + 1, ni + 1)  bool
            points_cached  (nk + 1, nj + 1, ni + 1, 3) or (nk + 1, np, 3)  float  (np = number of primary pillars)
            split_pillar_indices_cached  (nps)  int  (nps = number of primary pillars that are split)
            cols_for_split_pillars  (npxc)  int  (npxc = number of column corners using extra pillars due to splitting)
            cols_for_split_pillars_cl  (npx)  int  (npx = number of extra pillars due to splitting)
         the last 3 are only present when the grid has one or more split pillars;
         the split pillar data includes the use of a 'jagged' array (effectively an array of lists represented as
         a linear array and a 'cumulative length' index array)

      :meta common:
      """

      # todo: recheck the description of split pillar arrays given in the doc string
      self.cell_geometry_is_defined(cache_array = True)
      self.pillar_geometry_is_defined(cache_array = True)
      self.point(cache_array = True)
      if self.has_split_coordinate_lines:
         split_root = None
         if not hasattr(self, 'split_pillar_indices_cached'):
            split_root = self.resolve_geometry_child('SplitCoordinateLines')
            # assert(rqet.node_type(split_root) == 'ColumnLayerSplitCoordinateLines')
            pillar_indices_root = rqet.find_tag(split_root, 'PillarIndices')
            h5_key_pair = self.model.h5_uuid_and_path_for_node(pillar_indices_root)
            self.model.h5_array_element(h5_key_pair,
                                        index = None,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'split_pillar_indices_cached',
                                        dtype = 'int')
         if not hasattr(self, 'cols_for_split_pillars'):
            if split_root is None:
               split_root = self.resolve_geometry_child('SplitCoordinateLines')
            cpscl_root = rqet.find_tag(split_root, 'ColumnsPerSplitCoordinateLine')
            cpscl_elements_root = rqet.find_tag(cpscl_root, 'Elements')
            h5_key_pair = self.model.h5_uuid_and_path_for_node(cpscl_elements_root)
            self.model.h5_array_element(h5_key_pair,
                                        index = None,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'cols_for_split_pillars',
                                        dtype = 'int')
            cpscl_cum_length_root = rqet.find_tag(cpscl_root, 'CumulativeLength')
            h5_key_pair = self.model.h5_uuid_and_path_for_node(cpscl_cum_length_root)
            self.model.h5_array_element(h5_key_pair,
                                        index = None,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'cols_for_split_pillars_cl',
                                        dtype = 'int')

   def column_is_inactive(self, col_ji0):
      """Returns True if all the cells in the specified column are inactive.

      arguments:
         col_ji0 (int pair): the (j0, i0) column indices

      returns:
         boolean: True if all the cells in the column are inactive; False if at least one cell is active
      """

      self.extract_inactive_mask()
      if self.inactive is None:
         return False  # no inactive mask indicates all cells are active
      return np.all(self.inactive[:, col_ji0[0], col_ji0[1]])

   def create_column_pillar_mapping(self):
      """Creates an array attribute holding set of 4 pillar indices for each I, J column of cells.

      returns:
         numpy integer array of shape (nj, ni, 2, 2) where the last two indices are jp, ip;
         the array contains the pillar index for each of the 4 corners of each column of cells

      notes:
         the array is also cached as an attribute of the grid object: self.pillars_for_column
         for grids with split coordinates lines (faults), this array allows for fast access to
         the correct pillar data for the corner of a column of cells;
         here and elsewhere, ip & jp (& kp) refer to a 0 or 1 index which determines the side
         of a cell, ip & jp together select one of the four corners of a column;
         the pillar index is a single integer, which is used as the second index into the points
         array for a grid geometry with split pillars;
         for unsplit grid geometries, such a pillar index must be converted back into a j', i'
         pair of indices (or the points array must be reshaped to combine the two indices into one)

      :meta common:
      """

      if hasattr(self, 'pillars_for_column') and self.pillars_for_column is not None:
         return self.pillars_for_column

      self.cache_all_geometry_arrays()

      self.pillars_for_column = np.empty((self.nj, self.ni, 2, 2), dtype = int)
      ni_plus_1 = self.ni + 1

      for j in range(self.nj):
         self.pillars_for_column[j, :, 0, 0] = np.arange(j * ni_plus_1, (j + 1) * ni_plus_1 - 1, dtype = int)
         self.pillars_for_column[j, :, 0, 1] = np.arange(j * ni_plus_1 + 1, (j + 1) * ni_plus_1, dtype = int)
         self.pillars_for_column[j, :, 1, 0] = np.arange((j + 1) * ni_plus_1, (j + 2) * ni_plus_1 - 1, dtype = int)
         self.pillars_for_column[j, :, 1, 1] = np.arange((j + 1) * ni_plus_1 + 1, (j + 2) * ni_plus_1, dtype = int)

      if self.has_split_coordinate_lines:
         unsplit_pillar_count = (self.nj + 1) * ni_plus_1
         extras_count = len(self.split_pillar_indices_cached)
         for extra_index in range(extras_count):
            primary = self.split_pillar_indices_cached[extra_index]
            primary_ji0 = divmod(primary, self.ni + 1)
            extra_pillar_index = unsplit_pillar_count + extra_index
            if extra_index == 0:
               start = 0
            else:
               start = self.cols_for_split_pillars_cl[extra_index - 1]
            for cpscl_index in range(start, self.cols_for_split_pillars_cl[extra_index]):
               col = self.cols_for_split_pillars[cpscl_index]
               j, i = divmod(col, self.ni)
               jp = primary_ji0[0] - j
               ip = primary_ji0[1] - i
               assert (jp == 0 or jp == 1) and (ip == 0 or ip == 1)
               self.pillars_for_column[j, i, jp, ip] = extra_pillar_index

      return self.pillars_for_column

   def pillar_foursome(self, ji0, none_if_unsplit = False):
      """Returns a numpy int array of shape (2, 2) being the natural pillar indices applicable to each column around primary.

      arguments:
         ji0 (pair of ints): the pillar indices (j0, i0) of the primary pillar of interest
         none_if_unsplit (boolean, default False): if True and the primary pillar is unsplit, None is returned; if False,
            a foursome is returned full of the natural index of the primary pillar

      returns:
         numpy int array of shape (2, 2) being the natural pillar indices (second axis index in raw points array)
         applicable to each of the four columns around the primary pillar; axes of foursome are (jp, ip); if the
         primary pillar is unsplit, None is returned if none_if_unsplit is set to True, otherwise the foursome as
         usual
      """

      j0, i0 = ji0

      self.cache_all_geometry_arrays()

      primary = (self.ni + 1) * j0 + i0
      foursome = np.full((2, 2), primary, dtype = int)  # axes are: jp, ip
      if not self.has_split_coordinate_lines:
         return None if none_if_unsplit else foursome
      extras = np.where(self.split_pillar_indices_cached == primary)[0]
      if len(extras) == 0:
         return None if none_if_unsplit else foursome

      primary_count = (self.nj + 1) * (self.ni + 1)
      assert len(self.cols_for_split_pillars) == self.cols_for_split_pillars_cl[-1]
      for cpscl_index in extras:
         if cpscl_index == 0:
            start_index = 0
         else:
            start_index = self.cols_for_split_pillars_cl[cpscl_index - 1]
         for csp_index in range(start_index, self.cols_for_split_pillars_cl[cpscl_index]):
            natural_col = self.cols_for_split_pillars[csp_index]
            col_j0_e, col_i0_e = divmod(natural_col, self.ni)
            col_j0_e -= (j0 - 1)
            col_i0_e -= (i0 - 1)
            assert col_j0_e in [0, 1] and col_i0_e in [0, 1]
            foursome[col_j0_e, col_i0_e] = primary_count + cpscl_index

      return foursome

   def is_split_column_face(self, j0, i0, axis, polarity):
      """Returns True if the I or J column face is split; False otherwise."""

      if not self.has_split_coordinate_lines:
         return False
      assert axis in (1, 2)
      if axis == 1:  # J
         ip = i0
         if polarity:
            if j0 == self.nj - 1:
               return False
            jp = j0 + 1
         else:
            if j0 == 0:
               return False
            jp = j0 - 1
      else:  # I
         jp = j0
         if polarity:
            if i0 == self.ni - 1:
               return False
            ip = i0 + 1
         else:
            if i0 == 0:
               return False
            ip = i0 - 1
      cpm = self.create_column_pillar_mapping()
      if axis == 1:
         return ((cpm[j0, i0, polarity, 0] != cpm[jp, ip, 1 - polarity, 0]) or
                 (cpm[j0, i0, polarity, 1] != cpm[jp, ip, 1 - polarity, 1]))
      else:
         return ((cpm[j0, i0, 0, polarity] != cpm[jp, ip, 0, 1 - polarity]) or
                 (cpm[j0, i0, 1, polarity] != cpm[jp, ip, 1, 1 - polarity]))

   def split_column_faces(self):
      """Returns a pair of numpy boolean arrays indicating which internal column faces (column edges) are split."""

      if not self.has_split_coordinate_lines:
         return None, None
      if (hasattr(self, 'array_j_column_face_split') and self.array_j_column_face_split is not None and
          hasattr(self, 'array_i_column_face_split') and self.array_i_column_face_split is not None):
         return self.array_j_column_face_split, self.array_i_column_face_split
      if self.nj == 1:
         self.array_j_column_face_split = None
      else:
         self.array_j_column_face_split = np.zeros((self.nj - 1, self.ni),
                                                   dtype = bool)  # NB. internal faces only, index for +ve face
      if self.ni == 1:
         self.array_i_column_face_split = None
      else:
         self.array_i_column_face_split = np.zeros((self.nj, self.ni - 1),
                                                   dtype = bool)  # NB. internal faces only, index for +ve face
      self.create_column_pillar_mapping()
      for spi in self.split_pillar_indices_cached:
         j_p, i_p = divmod(spi, self.ni + 1)
         if j_p > 0 and j_p < self.nj:
            if i_p > 0 and self.is_split_column_face(j_p, i_p - 1, 1, 0):
               self.array_j_column_face_split[j_p - 1, i_p - 1] = True
            if i_p < self.ni - 1 and self.is_split_column_face(j_p, i_p, 1, 0):
               self.array_j_column_face_split[j_p - 1, i_p] = True
         if i_p > 0 and i_p < self.ni:
            if j_p > 0 and self.is_split_column_face(j_p - 1, i_p, 2, 0):
               self.array_i_column_face_split[j_p - 1, i_p - 1] = True
            if j_p < self.nj - 1 and self.is_split_column_face(j_p, i_p, 2, 0):
               self.array_i_column_face_split[j_p, i_p - 1] = True
      return self.array_j_column_face_split, self.array_i_column_face_split

   def find_faults(self, set_face_sets = False, create_organizing_objects_where_needed = False):
      """Searches for column-faces that are faulted and assigns fault ids; creates list of column-faces per fault id.

      note:
         this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
         GridConnectionSet class
      """

      # note:the logic to group kelp into distinct fault ids is simplistic and won't always give the right grouping

      if set_face_sets:
         self.clear_face_sets()

      if hasattr(self, 'fault_dict') and self.fault_dict is not None and len(self.fault_dict.keys()) > 0:
         if set_face_sets:
            for f, (j_list, i_list) in self.fault_dict.items():
               self.face_set_dict[f] = (j_list, i_list, 'K')
            self.set_face_set_gcs_list_from_dict(self.fault_dict, create_organizing_objects_where_needed)
         return None

      log.info('looking for faults in grid')
      self.create_column_pillar_mapping()
      if not self.has_split_coordinate_lines:
         log.info('grid does not have split coordinate lines, ie. is unfaulted')
         self.fault_dict = None
         return None

      # note: if Ni or Nj is 1, the kelp array has zero size, but that seems to be handled okay
      kelp_j = np.zeros((self.extent_kji[1] - 1, self.extent_kji[2]), dtype = 'int')  # fault id between cols j, j+1
      kelp_i = np.zeros((self.extent_kji[1], self.extent_kji[2] - 1), dtype = 'int')  # fault id between cols i, i+1

      last_fault_id = 0

      # look for splits affecting j faces
      for j in range(self.extent_kji[1] - 1):
         for i in range(self.extent_kji[2]):
            if i == 0 and (self.pillars_for_column[j, i, 1, 0] != self.pillars_for_column[j + 1, i, 0, 0] or
                           self.pillars_for_column[j, i, 1, 1] != self.pillars_for_column[j + 1, i, 0, 1]):
               last_fault_id += 1
               kelp_j[j, i] = last_fault_id
            elif self.pillars_for_column[j, i, 1, 1] != self.pillars_for_column[j + 1, i, 0, 1]:
               if i > 0 and kelp_j[j, i - 1] > 0:
                  kelp_j[j, i] = kelp_j[j, i - 1]
               else:
                  last_fault_id += 1
                  kelp_j[j, i] = last_fault_id

      # look for splits affecting i faces
      for i in range(self.extent_kji[2] - 1):
         for j in range(self.extent_kji[1]):
            if j == 0 and (self.pillars_for_column[j, i, 0, 1] != self.pillars_for_column[j, i + 1, 0, 0] or
                           self.pillars_for_column[j, i, 1, 1] != self.pillars_for_column[j, i + 1, 1, 0]):
               last_fault_id += 1
               kelp_i[j, i] = last_fault_id
            elif self.pillars_for_column[j, i, 1, 1] != self.pillars_for_column[j, i + 1, 1, 0]:
               if j > 0 and kelp_i[j - 1, i] > 0:
                  kelp_i[j, i] = kelp_i[j - 1, i]
               else:
                  last_fault_id += 1
                  kelp_i[j, i] = last_fault_id

      # make pass over kelp to reduce distinct ids: combine where pillar has exactly 2 kelps, one in each of i and j
      if kelp_j.size and kelp_i.size:
         for j in range(self.extent_kji[1] - 1):
            for i in range(self.extent_kji[2] - 1):
               if (bool(kelp_j[j, i]) != bool(kelp_j[j, i + 1])) and (bool(kelp_i[j, i]) != bool(kelp_i[j + 1, i])):
                  j_id = kelp_j[j, i] + kelp_j[j, i + 1]  # ie. the non-zero value
                  i_id = kelp_i[j, i] + kelp_i[j + 1, i]
                  if j_id == i_id:
                     continue
                  #                  log.debug('merging fault id {} into {}'.format(i_id, j_id))
                  kelp_i = np.where(kelp_i == i_id, j_id, kelp_i)
                  kelp_j = np.where(kelp_j == i_id, j_id, kelp_j)

      fault_id_list = np.unique(np.concatenate(
         (np.unique(kelp_i.flatten()), np.unique(kelp_j.flatten()))))[1:]  # discard zero from list
      log.info('number of distinct faults: ' + str(fault_id_list.size))
      # for each fault id, make pair of tuples of kelp locations
      self.fault_dict = {}  # maps fault_id to pair (j faces, i faces) of array of [j, i] kelp indices for that fault_id
      for fault_id in fault_id_list:
         self.fault_dict[fault_id] = (np.stack(np.where(kelp_j == fault_id),
                                               axis = 1), np.stack(np.where(kelp_i == fault_id), axis = 1))
      self.fault_id_j = kelp_j.copy()  # fault_id for each internal j kelp, zero is none; extent nj-1, ni
      self.fault_id_i = kelp_i.copy()  # fault_id for each internal i kelp, zero is none; extent nj, ni-1
      if set_face_sets:
         for f, (j_list, i_list) in self.fault_dict.items():
            self.face_set_dict[f] = (j_list, i_list, 'K')
         self.set_face_set_gcs_list_from_dict(self.fault_dict, create_organizing_objects_where_needed)
      return (self.fault_id_j, self.fault_id_i)

   def fault_throws(self):
      """Finds mean throw of each J and I face; adds throw arrays as attributes to this grid and returns them.

      note:
         this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
         GridConnectionSet class
      """

      if hasattr(self, 'fault_throw_j') and self.fault_throw_j is not None and hasattr(
            self, 'fault_throw_i') and self.fault_throw_i is not None:
         return (self.fault_throw_j, self.fault_throw_i)
      if not self.has_split_coordinate_lines:
         return None
      if not hasattr(self, 'fault_id_j') or self.fault_id_j is None or not hasattr(
            self, 'fault_id_i') or self.fault_id_i is None:
         self.find_faults()
         if not hasattr(self, 'fault_id_j'):
            return None
      log.debug('computing fault throws (deprecated method)')
      cp = self.corner_points(cache_cp_array = True)
      self.fault_throw_j = np.zeros((self.nk, self.nj - 1, self.ni))
      self.fault_throw_i = np.zeros((self.nk, self.nj, self.ni - 1))
      self.fault_throw_j = np.where(self.fault_id_j == 0, 0.0,
                                    0.25 * np.sum(cp[:, 1:, :, :, 0, :, 2] - cp[:, :-1, :, :, 1, :, 2], axis = (3, 4)))
      self.fault_throw_i = np.where(self.fault_id_i == 0, 0.0,
                                    0.25 * np.sum(cp[:, :, 1:, :, :, 0, 2] - cp[:, :, -1:, :, :, 1, 2], axis = (3, 4)))
      return (self.fault_throw_j, self.fault_throw_i)

   def fault_throws_per_edge_per_column(self, mode = 'maximum', simple_z = False, axis_polarity_mode = True):
      """Returns numpy array of shape (nj, ni, 2, 2) or (nj, ni, 4) holding max, mean or min throw based on split node separations.

      arguments:
         mode (string, default 'maximum'): one of 'minimum', 'mean', 'maximum'; determines how to resolve variation in throw for
            each column edge
         simple_z (boolean, default False): if True, the returned throw values are vertical offsets; if False, the displacement
            in xyz space between split points is the basis of the returned values and may include a lateral offset component as
            well as xy displacement due to sloping pillars
         axis_polarity (boolean, default True): determines shape and ordering of returned array; if True, the returned array has
            shape (nj, ni, 2, 2); if False the shape is (nj, ni, 4); see return value notes for more information

      returns:
         numpy float array of shape (nj, ni, 2, 2) or (nj, ni, 4) holding fault throw values for each column edge; units are
            z units of crs for this grid; if simple_z is False, xy units and z units must be the same; positive values indicate
            greater depth if z is increasing downwards (or shallower if z is increasing upwards); negative values indicate the
            opposite; the shape and ordering of the returned array is determined by axis_polarity_mode; if axis_polarity_mode is
            True, the returned array has shape (nj, ni, 2, 2) with the third index being axis (0 = J, 1 = I) and the final index
            being polarity (0 = minus face edge, 1 = plus face edge); if axis_polarity_mode is False, the shape is (nj, ni, 4)
            and the face edges are ordered I-, J+, I+, J-, as required by the resqml standard for a property with indexable
            element 'edges per column'

      notes:
         the throws calculated by this method are based merely on grid geometry and do not refer to grid connection sets;
         NB: the same absolute value is returned, with opposite sign, for the edges on opposing sides of a fault; either one of
         these alone indicates the full throw;
         the property module contains a pair of reformatting functions for moving an array between the two axis polarity modes;
         minimum and maximum modes work on the absolute throws
      """

      assert mode in ['maximum', 'mean', 'minimum']
      if not simple_z:
         assert self.z_units() == self.xy_units()

      log.debug('computing fault throws per edge per column based on corner point geometry')
      if not self.has_split_coordinate_lines:  # note: no NaNs returned in this situation
         if axis_polarity_mode:
            return np.zeros((self.nj, self.ni, 2, 2))
         return np.zeros((self.nj, self.ni, 4))
      self.create_column_pillar_mapping()
      i_pillar_throws = (
         self.points_cached[:, self.pillars_for_column[:, :-1, :, 1], 2]
         -  # (nk+1, nj, ni-1, jp) +ve dz I- cell > I+ cell
         self.points_cached[:, self.pillars_for_column[:, 1:, :, 0], 2])
      j_pillar_throws = (self.points_cached[:, self.pillars_for_column[:-1, :, 1, :], 2] -
                         self.points_cached[:, self.pillars_for_column[1:, :, 0, :], 2])
      if not simple_z:
         i_pillar_throws = np.sign(
            i_pillar_throws)  # note: will return zero if displacement is purely horizontal wrt. z axis
         j_pillar_throws = np.sign(j_pillar_throws)
         i_pillar_throws *= vec.naive_lengths(self.points_cached[:, self.pillars_for_column[:, :-1, :, 1], :] -
                                              self.points_cached[:, self.pillars_for_column[:, 1:, :, 0], :])
         j_pillar_throws *= vec.naive_lengths(self.points_cached[:, self.pillars_for_column[:-1, :, 1, :], :] -
                                              self.points_cached[:, self.pillars_for_column[1:, :, 0, :], :])

      if mode == 'mean':
         i_edge_throws = np.nanmean(i_pillar_throws, axis = (0, -1))  # (nj, ni-1)
         j_edge_throws = np.nanmean(j_pillar_throws, axis = (0, -1))  # (nj-1, ni)
      else:
         min_i_edge_throws = np.nanmean(np.nanmin(i_pillar_throws, axis = 0), axis = -1)
         max_i_edge_throws = np.nanmean(np.nanmax(i_pillar_throws, axis = 0), axis = -1)
         min_j_edge_throws = np.nanmean(np.nanmin(j_pillar_throws, axis = 0), axis = -1)
         max_j_edge_throws = np.nanmean(np.nanmax(j_pillar_throws, axis = 0), axis = -1)
         i_flip_mask = (np.abs(min_i_edge_throws) > np.abs(max_i_edge_throws))
         j_flip_mask = (np.abs(min_j_edge_throws) > np.abs(max_j_edge_throws))
         if mode == 'maximum':
            i_edge_throws = np.where(i_flip_mask, min_i_edge_throws, max_i_edge_throws)
            j_edge_throws = np.where(j_flip_mask, min_j_edge_throws, max_j_edge_throws)
         elif mode == 'minimum':
            i_edge_throws = np.where(i_flip_mask, max_i_edge_throws, min_i_edge_throws)
            j_edge_throws = np.where(j_flip_mask, max_j_edge_throws, min_j_edge_throws)
         else:
            raise Exception('code failure')

      # positive values indicate column has greater z values, ie. downthrown if z increases with depth
      if axis_polarity_mode:
         throws = np.zeros((self.nj, self.ni, 2, 2))  # first 2 is I (0) or J (1); final 2 is -ve or +ve face
         throws[1:, :, 0, 0] = -j_edge_throws  # J-
         throws[:-1, :, 0, 1] = j_edge_throws  # J+
         throws[:, 1:, 1, 0] = -i_edge_throws  # I-
         throws[:, :-1, 1, 1] = i_edge_throws  # I+

      else:  # resqml protocol
         # order I-, J+, I+, J- as required for properties with 'edges per column' indexable element
         throws = np.zeros((self.nj, self.ni, 4))
         throws[:, 1:, 0] = -i_edge_throws  # I-
         throws[:-1, :, 1] = j_edge_throws  # J+
         throws[:, :-1, 2] = i_edge_throws  # I+
         throws[1:, :, 3] = -j_edge_throws  # J-

      return throws

   def clear_face_sets(self):
      """Discard face sets."""
      # following maps face_set_id to (j faces, i faces, 'K') of array of [j, i] kelp indices for that face_set_id
      # or equivalent for axes 'J' or 'I'
      self.face_set_dict = {}
      self.face_set_gcs_list = []

   def set_face_set_gcs_list_from_dict(self, face_set_dict = None, create_organizing_objects_where_needed = False):
      """Creates a grid connection set for each feature in the face set dictionary, based on kelp list pairs."""

      if face_set_dict is None:
         face_set_dict = self.face_set_dict
      self.face_set_gcs_list = []
      for feature in face_set_dict:
         gcs = rqf.GridConnectionSet(self.model, grid = self)
         kelp_j, kelp_i, axis = face_set_dict[feature]
         log.debug(f'creating gcs for: {feature} {axis}')
         gcs.set_pairs_from_kelp(kelp_j, kelp_i, feature, create_organizing_objects_where_needed, axis = axis)
         self.face_set_gcs_list.append(gcs)

   # TODO: make separate curtain and K-face versions of following function
   def make_face_set_from_dataframe(self, df):
      """Creates a curtain face set for each named fault in dataframe.

      note:
         this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
         GridConnectionSet class
      """

      # df columns: name, i1, i2, j1, j2, k1, k2, face
      self.clear_face_sets()
      names = pd.unique(df.name)
      count = 0
      box_kji0 = np.zeros((2, 3), dtype = int)
      k_warning_given = False
      for fs_name in names:
         i_kelp_list = []
         j_kelp_list = []
         # ignore k faces for now
         fs_ds = df[df.name == fs_name]
         for row in range(len(fs_ds)):
            face = fs_ds.iloc[row]['face']
            fl = face[0].upper()
            if fl in 'IJK':
               axis = 'KJI'.index(fl)
            elif fl in 'XYZ':
               axis = 'ZYX'.index(fl)
            else:
               raise ValueError('fault data face not recognized: ' + face)
            if axis == 0:
               continue  # ignore k faces for now
            box_kji0[0, 0] = fs_ds.iloc[row]['k1'] - 1  # k1
            box_kji0[1, 0] = fs_ds.iloc[row]['k2'] - 1  # k2
            box_kji0[0, 1] = fs_ds.iloc[row]['j1'] - 1  # j1
            box_kji0[1, 1] = fs_ds.iloc[row]['j2'] - 1  # j2
            box_kji0[0, 2] = fs_ds.iloc[row]['i1'] - 1  # i1
            box_kji0[1, 2] = fs_ds.iloc[row]['i2'] - 1  # i2
            box_kji0[1, 0] = min(box_kji0[1, 0], self.extent_kji[0] - 1)
            if not k_warning_given and (box_kji0[0, 0] != 0 or box_kji0[1, 0] != self.extent_kji[0] - 1):
               log.warning(
                  'one or more entries in face set dataframe does not cover entire layer range: extended to all layers')
               k_warning_given = True
            if len(face) > 1 and face[1] == '-':  # treat negative faces as positive faces of neighbouring cell
               box_kji0[0, axis] = max(box_kji0[0, axis] - 1, 0)
               box_kji0[1, axis] -= 1
            else:
               box_kji0[1, axis] = min(box_kji0[1, axis], self.extent_kji[axis] - 2)
            if box_kji0[1, axis] < box_kji0[0, axis]:
               continue  # faces are all on edge of grid
            # for now ignore layer range and create curtain of j and i kelp
            for j in range(box_kji0[0, 1], box_kji0[1, 1] + 1):
               for i in range(box_kji0[0, 2], box_kji0[1, 2] + 1):
                  if axis == 1:
                     _add_to_kelp_list(self.extent_kji, j_kelp_list, True, (j, i))
                  elif axis == 2:
                     _add_to_kelp_list(self.extent_kji, i_kelp_list, False, (j, i))
         self.face_set_dict[fs_name] = (j_kelp_list, i_kelp_list, 'K')
         count += 1
      log.info(str(count) + ' face sets extracted from dataframe')

   def make_face_sets_from_pillar_lists(self,
                                        pillar_list_list,
                                        face_set_id,
                                        axis = 'K',
                                        ref_slice0 = 0,
                                        plus_face = False,
                                        projection = 'xy'):
      """Creates a curtain face set for each pillar (or rod) list.

      returns:
         (face_set_dict, full_pillar_list_dict)

      note:
         'xz' and 'yz' projections currently only supported for unsplit grids
      """

      # NB. this code was originally written for axis K and projection xy, working with horizon points
      # it has since been reworked for the cross sectional cases, so variables named 'pillar...' may refer to rods
      # and i_... and j_... may actually represent i,j or i,k or j,k

      assert axis in ['K', 'J', 'I']
      assert projection in ['xy', 'xz', 'yz']

      local_face_set_dict = {}
      full_pillar_list_dict = {}

      if not hasattr(self, 'face_set_dict'):
         self.clear_face_sets()

      if axis.upper() == 'K':
         assert projection == 'xy'
         pillar_xy = self.horizon_points(ref_k0 = ref_slice0, kp = 1 if plus_face else 0)[:, :, 0:2]
         kelp_axes = 'JI'
      else:
         if projection == 'xz':
            pillar_xy = self.unsplit_x_section_points(axis, ref_slice0 = ref_slice0,
                                                      plus_face = plus_face)[:, :, 0:3:2]  # x,z
         else:  # projection == 'yz'
            pillar_xy = self.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)[:, :,
                                                                                                            1:]  # y,z
         if axis.upper() == 'J':
            kelp_axes = 'KI'
         else:
            kelp_axes = 'KJ'
      kelp_axes_int = np.empty((2,), dtype = int)
      for a in range(2):
         kelp_axes_int[a] = 'KJI'.index(kelp_axes[a])
#     self.clear_face_sets()  # now accumulating sets
      face_set_count = 0
      here = np.zeros(2, dtype = int)
      side_step = np.zeros(2, dtype = int)
      for pillar_list in pillar_list_list:
         if len(pillar_list) < 2:
            continue
         full_pillar_list = [pillar_list[0]]
         face_set_count += 1
         if len(pillar_list_list) > 1:
            id_suffix = '_line_' + str(face_set_count)
         else:
            id_suffix = ''
         # i,j are as stated if axis is 'K'; for axis 'J', i,j are actually i,k; for axis 'I', i,j are actually j,k
         i_kelp_list = []
         j_kelp_list = []
         for p in range(len(pillar_list) - 1):
            ji_0 = pillar_list[p]
            ji_1 = pillar_list[p + 1]
            if np.all(ji_0 == ji_1):
               continue
            # xy might actually be xy, xz or yz depending on projection
            xy_0 = pillar_xy[tuple(ji_0)]
            xy_1 = pillar_xy[tuple(ji_1)]
            if vec.isclose(xy_0, xy_1):
               continue
            dj = ji_1[0] - ji_0[0]
            di = ji_1[1] - ji_0[1]
            abs_dj = abs(dj)
            abs_di = abs(di)
            if dj < 0:
               j_sign = -1
            else:
               j_sign = 1
            if di < 0:
               i_sign = -1
            else:
               i_sign = 1
            here[:] = ji_0
            while np.any(here != ji_1):
               previous = here.copy()  # debug
               if abs_dj >= abs_di:
                  j = here[0]
                  if j != ji_1[0]:
                     jp = j + j_sign
                     _add_to_kelp_list(self.extent_kji, i_kelp_list, kelp_axes[1], (min(j, jp), here[1] - 1))
                     here[0] = jp
                     full_pillar_list.append(tuple(here))
                  if di != 0:
                     divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(here)], xy_0, xy_1)
                     side_step[:] = here
                     side_step[1] += i_sign
                     if side_step[1] >= 0 and side_step[1] <= self.extent_kji[kelp_axes_int[1]]:
                        stepped_divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(side_step)], xy_0, xy_1)
                        if stepped_divergence < divergence:
                           here[:] = side_step
                           _add_to_kelp_list(self.extent_kji, j_kelp_list, kelp_axes[0],
                                             (here[0] - 1, min(here[1], here[1] - i_sign)))
                           full_pillar_list.append(tuple(here))
               else:
                  i = here[1]
                  if i != ji_1[1]:
                     ip = i + i_sign
                     _add_to_kelp_list(self.extent_kji, j_kelp_list, kelp_axes[0], (here[0] - 1, min(i, ip)))
                     here[1] = ip
                     full_pillar_list.append(tuple(here))
                  if dj != 0:
                     divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(here)], xy_0, xy_1)
                     side_step[:] = here
                     side_step[0] += j_sign
                     if side_step[0] >= 0 and side_step[0] <= self.extent_kji[kelp_axes_int[0]]:
                        stepped_divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(side_step)], xy_0, xy_1)
                        if stepped_divergence < divergence:
                           here[:] = side_step
                           _add_to_kelp_list(self.extent_kji, i_kelp_list, kelp_axes[1],
                                             (min(here[0], here[0] - j_sign), here[1] - 1))
                           full_pillar_list.append(tuple(here))
               assert np.any(here != previous), 'failed to move'
         self.face_set_dict[face_set_id + id_suffix] = (j_kelp_list, i_kelp_list, axis)
         local_face_set_dict[face_set_id + id_suffix] = (j_kelp_list, i_kelp_list, axis)
         full_pillar_list_dict[face_set_id + id_suffix] = full_pillar_list.copy()

      return local_face_set_dict, full_pillar_list_dict

   def check_top_and_base_cell_edge_directions(self):
      """checks grid top face I & J edge vectors (in x,y) against basal equivalents: max 90 degree angle tolerated

      returns: boolean: True if all checks pass; False if one or more checks fail

      notes:
         similarly checks cell edge directions in neighbouring cells in top (and separately in base)
         currently requires geometry to be defined for all pillars
         logs a warning if a check is not passed
      """

      log.debug('deriving cell edge vectors at top and base (for checking)')
      self.point(cache_array = True)
      good = True
      if self.has_split_coordinate_lines:
         # build top and base I & J cell edge vectors
         self.create_column_pillar_mapping()  # pillar indices for 4 columns around interior pillars
         top_j_edge_vectors_p = np.zeros((self.nj, self.ni, 2, 2))  # third axis is ip
         top_i_edge_vectors_p = np.zeros((self.nj, 2, self.ni, 2))  # second axis is jp
         base_j_edge_vectors_p = np.zeros((self.nj, self.ni, 2, 2))  # third axis is ip
         base_i_edge_vectors_p = np.zeros((self.nj, 2, self.ni, 2))  # second axis is jp
         # todo: rework as numpy operations across nj & ni
         for j in range(self.nj):
            for i in range(self.ni):
               for jip in range(2):  # serves as either jp or ip
                  top_j_edge_vectors_p[j, i,
                                       jip, :] = (self.points_cached[0, self.pillars_for_column[j, i, 1, jip], :2] -
                                                  self.points_cached[0, self.pillars_for_column[j, i, 0, jip], :2])
                  base_j_edge_vectors_p[j, i, jip, :] = (
                     self.points_cached[self.nk_plus_k_gaps, self.pillars_for_column[j, i, 1, jip], :2] -
                     self.points_cached[self.nk_plus_k_gaps, self.pillars_for_column[j, i, 0, jip], :2])
                  top_i_edge_vectors_p[j, jip,
                                       i, :] = (self.points_cached[0, self.pillars_for_column[j, i, jip, 1], :2] -
                                                self.points_cached[0, self.pillars_for_column[j, i, jip, 0], :2])
                  base_i_edge_vectors_p[j, jip, i, :] = (
                     self.points_cached[self.nk_plus_k_gaps, self.pillars_for_column[j, i, jip, 1], :2] -
                     self.points_cached[self.nk_plus_k_gaps, self.pillars_for_column[j, i, jip, 0], :2])
         # reshape to allow common checking code with unsplit grid vectors (below)
         top_j_edge_vectors = top_j_edge_vectors_p.reshape((self.nj, 2 * self.ni, 2))
         top_i_edge_vectors = top_i_edge_vectors_p.reshape((2 * self.nj, self.ni, 2))
         base_j_edge_vectors = base_j_edge_vectors_p.reshape((self.nj, 2 * self.ni, 2))
         base_i_edge_vectors = base_i_edge_vectors_p.reshape((2 * self.nj, self.ni, 2))
      else:
         top_j_edge_vectors = (self.points_cached[0, 1:, :, :2] - self.points_cached[0, :-1, :, :2]).reshape(
            (self.nj, self.ni + 1, 2))
         top_i_edge_vectors = (self.points_cached[0, :, 1:, :2] - self.points_cached[0, :, :-1, :2]).reshape(
            (self.nj + 1, self.ni, 2))
         base_j_edge_vectors = (self.points_cached[-1, 1:, :, :2] - self.points_cached[-1, :-1, :, :2]).reshape(
            (self.nj, self.ni + 1, 2))
         base_i_edge_vectors = (self.points_cached[-1, :, 1:, :2] - self.points_cached[-1, :, :-1, :2]).reshape(
            (self.nj + 1, self.ni, 2))
      log.debug('checking relative direction of top and base edges')
      # check direction of top edges against corresponding base edges, tolerate upto 90 degree difference
      dot_j = np.sum(top_j_edge_vectors * base_j_edge_vectors, axis = 2)
      dot_i = np.sum(top_i_edge_vectors * base_i_edge_vectors, axis = 2)
      if not np.all(dot_j >= 0.0) and np.all(dot_i >= 0.0):
         log.warning('one or more columns of cell edges flip direction: this grid is probably unusable')
         good = False
      log.debug('checking relative direction of edges in neighbouring cells at top of grid (and base)')
      # check direction of similar edges on neighbouring cells, tolerate upto 90 degree difference
      dot_jp = np.sum(top_j_edge_vectors[1:, :, :] * top_j_edge_vectors[:-1, :, :], axis = 2)
      dot_ip = np.sum(top_i_edge_vectors[1:, :, :] * top_i_edge_vectors[:-1, :, :], axis = 2)
      if not np.all(dot_jp >= 0.0) and np.all(dot_ip >= 0.0):
         log.warning('top cell edges for neighbouring cells flip direction somewhere: this grid is probably unusable')
         good = False
      dot_jp = np.sum(base_j_edge_vectors[1:, :, :] * base_j_edge_vectors[:-1, :, :], axis = 2)
      dot_ip = np.sum(base_i_edge_vectors[1:, :, :] * base_i_edge_vectors[:-1, :, :], axis = 2)
      if not np.all(dot_jp >= 0.0) and np.all(dot_ip >= 0.0):
         log.warning('base cell edges for neighbouring cells flip direction somewhere: this grid is probably unusable')
         good = False
      return good

   def point_raw(self, index = None, points_root = None, cache_array = True):
      """Returns element from points data, indexed as in the hdf5 file; can optionally be used to cache points data.

      arguments:
         index (2 or 3 integers, optional): if not None, the index into the raw points data for the point of interest
         points_root (optional): the xml node holding the points data
         cache_array (boolean, default True): if True, the raw points data is cached in memory as a side effect

      returns:
         (x, y, z) of selected point as a 3 element numpy vector, or None if index is None

      notes:
         this function is typically called either to cache the points data in memory, or to fetch the coordinates of
         a single point; the details of the indexing depend upon whether the grid has split coordinate lines: if not,
         the index should be a triple kji0 with axes ranging over the shared corners nk+k_gaps+1, nj+1, ni+1; if there
         are split pillars, index should be a pair, being the k0 in range nk+k_gaps+1 and a pillar index; note that if
         index is passed, the k0 element must already have been mapped to the raw index value taking into consideration
         any k gaps; if the grid object does not include geometry then None is returned
      """

      # NB: shape of index depends on whether grid has split pillars
      if index is not None and not self.geometry_defined_for_all_pillars(cache_array = cache_array):
         if len(index) == 3:
            ji = tuple(index[1:])
         else:
            ji = tuple(divmod(index[1], self.ni))
         if ji[0] < self.nj and not self.pillar_geometry_is_defined(ji, cache_array = cache_array):
            return None
      if self.points_cached is not None:
         if index is None:
            return self.points_cached
         return self.points_cached[tuple(index)]
      p_root = self.resolve_geometry_child('Points', child_node = points_root)
      if p_root is None:
         log.debug('point_raw() returning None as geometry not present')
         return None  # geometry not present
      assert rqet.node_type(p_root) == 'Point3dHdf5Array'
      h5_key_pair = self.model.h5_uuid_and_path_for_node(p_root, tag = 'Coordinates')
      if h5_key_pair is None:
         return None
      if self.has_split_coordinate_lines:
         required_shape = None
      else:
         required_shape = (self.nk_plus_k_gaps + 1, self.nj + 1, self.ni + 1, 3)
      try:
         value = self.model.h5_array_element(h5_key_pair,
                                             index = index,
                                             cache_array = cache_array,
                                             object = self,
                                             array_attribute = 'points_cached',
                                             required_shape = required_shape)
      except Exception:
         log.error('hdf5 points failure for index: ' + str(index))
         raise
      if index is None:
         return self.points_cached
      return value

   def point(self, cell_kji0 = None, corner_index = np.zeros(3, dtype = 'int'), points_root = None, cache_array = True):
      """Return a cell corner point xyz; can optionally be used to cache points data.

      arguments:
         cell_kji0 (3 integers, optional): if not None, the index of the cell for the point of interest, in kji0 protocol
         corner_index (3 integers, default zeros): the kp, jp, ip corner-within-cell indices (each 0 or 1)
         points_root (optional): the xml node holding the points data
         cache_array (boolean, default True): if True, the raw points data is cached in memory as a side effect

      returns:
         (x, y, z) of selected point as a 3 element numpy vector, or None if cell_kji0 is None

      note:
         if cell_kji0 is passed, the k0 value should be the layer index before adjustment for k_gaps, which this
         method will apply
      """

      if cache_array and self.points_cached is None:
         self.point_raw(points_root = points_root, cache_array = True)
      if cell_kji0 is None:
         return None
      if self.k_raw_index_array is None:
         self.extract_k_gaps()
      if not self.geometry_defined_for_all_cells():
         if not self.cell_geometry_is_defined(cell_kji0, cache_array = cache_array):
            return None
      p_root = self.resolve_geometry_child('Points', child_node = points_root)
      #      if p_root is None: return None  # geometry not present
      index = np.zeros(3, dtype = int)
      index[:] = cell_kji0
      index[0] = self.k_raw_index_array[index[0]]  # adjust for k gaps
      if self.has_split_coordinate_lines:
         self.create_column_pillar_mapping()
         pillar_index = self.pillars_for_column[index[1], index[2], corner_index[1], corner_index[2]]
         return self.point_raw(index = (index[0] + corner_index[0], pillar_index),
                               points_root = p_root,
                               cache_array = cache_array)
      else:
         index[:] += corner_index
         return self.point_raw(index = index, points_root = p_root, cache_array = cache_array)

   def points_ref(self, masked = True):
      """Returns an in-memory numpy array containing the xyz data for points used in the grid geometry.

      argument:
         masked (boolean, default True): if True, a masked array is returned with NaN points masked out;
            if False, a simple (unmasked) numpy array is returned

      returns:
         numpy array or masked array of float, of shape (nk + k_gaps + 1, nj + 1, ni + 1, 3) or (nk + k_gaps + 1, np, 3)
         where np is the total number of pillars (primary pillars + extras for split pillars)

      notes:
         this is the usual way to get at the actual grid geometry points data in the native resqml layout;
         the has_split_coordinate_lines boolean attribute can be used to determine which shape to expect;
         the shape is (nk + k_gaps + 1, nj + 1, ni + 1, 3) if there are no split coordinate lines (unfaulted);
         otherwise it is (nk + k_gaps + 1, np, 3), where np > (nj + 1) * (ni + 1), due to extra pillar data for
         the split pillars

      :meta common:
      """

      if self.points_cached is None:
         self.point(cache_array = True)
         if self.points_cached is None:
            return None
      if not masked:
         return self.points_cached
      return ma.masked_invalid(self.points_cached)

   def uncache_points(self):
      """Frees up memory by removing the cached copy of the grid's points data.

      note:
         the memory will only actually become free when any other references to it pass out of scope
         or are deleted
      """

      if self.points_cached is not None:
         del self.points_cached
         self.points_cached = None

   def unsplit_points_ref(self, cache_array = False, masked = False):
      """Returns a copy of the points array that has split pillars merged back into an unsplit configuration.

      arguments:
         cache_array (boolean, default False): if True, a copy of the unsplit points array is added as
            attribute array_unsplit_points to this grid object
         masked (boolean, default False): if True, a masked array is returned with NaN points masked out;
            if False, a simple (unmasked) numpy array is returned

      returns:
         numpy array of float of shape (nk + k_gaps + 1, nj + 1, ni + 1, 3)

      note:
         for grids without split pillars, this function simply returns the points array in its native form;
         for grids with split pillars, an unsplit equivalent points array is calculated as the average of
         contributions to each pillar from the surrounding cell columns
      """

      if hasattr(self, 'array_unsplit_points'):
         return self.array_unsplit_points
      points = self.points_ref(masked = masked)
      if not self.has_split_coordinate_lines:
         if cache_array:
            self.array_unsplit_points = points.copy()
            return self.array_unsplit_points
         return points
      # todo: finish version that copies primaries and only modifies split pillars?
      # njkp1 = (self.nj + 1) * (self.ni + 1)
      # merged_points = np.empty((self.nk + 1, njkp1, 3))   # shaped somewhat like split points array
      # merged_points[:, :, :] = points[:, :njkp1, :]       # copy primary data
      result = np.empty((self.nk_plus_k_gaps + 1, self.nj + 1, self.ni + 1, 3))
      # todo: if not geometry defined for all cells, take nanmean of four points?
      # compute for internal pillars
      self.create_column_pillar_mapping()  # pillar indices for 4 columns around interior pillars
      pfc_11 = self.pillars_for_column[:-1, :-1, 1, 1]
      pfc_10 = self.pillars_for_column[:-1, 1:, 1, 0]
      pfc_01 = self.pillars_for_column[1:, :-1, 0, 1]
      pfc_00 = self.pillars_for_column[1:, 1:, 0, 0]
      result[:, 1:-1, 1:-1, :] = 0.25 * (points[:, pfc_11, :] + points[:, pfc_10, :] + points[:, pfc_01, :] +
                                         points[:, pfc_00, :])
      # edges
      # todo: use numpy array operations instead of for loops (see lines above for example code)
      for j in range(1, self.nj):
         result[:, j, 0, :] = 0.5 * (points[:, self.pillars_for_column[j - 1, 0, 1, 0], :] +
                                     points[:, self.pillars_for_column[j, 0, 0, 0], :])
         result[:, j, self.ni, :] = 0.5 * (points[:, self.pillars_for_column[j - 1, self.ni - 1, 1, 1], :] +
                                           points[:, self.pillars_for_column[j, self.ni - 1, 0, 1], :])
      for i in range(1, self.ni):
         result[:, 0, i, :] = 0.5 * (points[:, self.pillars_for_column[0, i - 1, 0, 1], :] +
                                     points[:, self.pillars_for_column[0, i, 0, 0], :])
         result[:, self.nj, i, :] = 0.5 * (points[:, self.pillars_for_column[self.nj - 1, i - 1, 1, 1], :] +
                                           points[:, self.pillars_for_column[self.nj - 1, i, 1, 0], :])
      # corners (could optimise as these should always be primaries
      result[:, 0, 0, :] = points[:, self.pillars_for_column[0, 0, 0, 0], :]
      result[:, 0, self.ni, :] = points[:, self.pillars_for_column[0, self.ni - 1, 0, 1], :]
      result[:, self.nj, 0, :] = points[:, self.pillars_for_column[self.nj - 1, 0, 1, 0], :]
      result[:, self.nj, self.ni, :] = points[:, self.pillars_for_column[self.nj - 1, self.ni - 1, 1, 1], :]
      if cache_array:
         self.array_unsplit_points = result
         return self.array_unsplit_points
      return result

   def xyz_box(self, points_root = None, lazy = True, local = False):
      """Returns the minimum and maximum xyz for the grid geometry.

      arguments:
         points_root (optional): if not None, the xml root node for the points data (speed optimization)
         lazy (boolean, default True): if True, only the 8 outermost logical corners of the grid are used
            to determine the ranges of xyz; if False, all the points in the entire grid are scanned to
            determine the xyz ranges in an exhaustive manner
         local (boolean, default False): if True, the xyz ranges that are returned are in the local
            coordinate space, otherwise the global (crs parent) coordinate space

      returns:
         numpy array of float of shape (2, 3); the first axis is minimum, maximum; the second axis is x, y, z

      note:
         if the lazy argument is True, the results are likely to under-report the ranges, especially for z

      :meta common:
      """

      if self.xyz_box_cached is None or (not lazy and not self.xyz_box_cached_thoroughly):
         self.xyz_box_cached = np.zeros((2, 3))
         if lazy:
            eight_corners = np.zeros((2, 2, 2, 3))
            for kp in [0, 1]:
               for jp in [0, 1]:
                  for ip in [0, 1]:
                     eight_corners[kp, jp, ip] = self.point(cell_kji0 = [
                        kp * (self.extent_kji[0] - 1), jp * (self.extent_kji[1] - 1), ip * (self.extent_kji[2] - 1)
                     ],
                                                            corner_index = [kp, jp, ip],
                                                            points_root = points_root,
                                                            cache_array = False)
            self.xyz_box_cached[0, :] = np.nanmin(eight_corners, axis = (0, 1, 2))
            self.xyz_box_cached[1, :] = np.nanmax(eight_corners, axis = (0, 1, 2))
         else:
            ps = self.points_ref()
            if self.has_split_coordinate_lines:
               self.xyz_box_cached[0, :] = np.nanmin(ps, axis = (0, 1))
               self.xyz_box_cached[1, :] = np.nanmax(ps, axis = (0, 1))
            else:
               self.xyz_box_cached[0, :] = np.nanmin(ps, axis = (0, 1, 2))
               self.xyz_box_cached[1, :] = np.nanmax(ps, axis = (0, 1, 2))
         self.xyz_box_cached_thoroughly = not lazy
      if local:
         return self.xyz_box_cached
      global_xyz_box = self.xyz_box_cached.copy()
      self.local_to_global_crs(global_xyz_box, self.crs_root)
      return global_xyz_box

   def xyz_box_centre(self, points_root = None, lazy = False, local = False):
      """Returns the (x,y,z) point (as 3 element numpy) at the centre of the xyz box for the grid.

      arguments:
         points_root (optional): if not None, the xml root node for the points data (speed optimization)
         lazy (boolean, default True): if True, only the 8 outermost logical corners of the grid are used
            to determine the ranges of xyz and hence the centre; if False, all the points in the entire
            grid are scanned to determine the xyz ranges in an exhaustive manner
         local (boolean, default False): if True, the xyz values that are returned are in the local
            coordinate space, otherwise the global (crs parent) coordinate space

      returns:
         numpy array of float of shape (3,) being the x, y, z coordinates of the centre of the grid

      note:
         the centre point returned is simply the midpoint of the x, y & z ranges of the grid
      """

      return np.nanmean(self.xyz_box(points_root = points_root, lazy = lazy, local = local), axis = 0)

   def horizon_points(self, ref_k0 = 0, heal_faults = False, kp = 0):
      """Returns reference to a points layer array of shape ((nj + 1), (ni + 1), 3) based on primary pillars.

      arguments:
         ref_k0 (integer): the horizon layer number, in the range 0 to nk (or layer number in range 0..nk-1
            in the case of grids with k gaps)
         heal_faults (boolean, default False): if True and the grid has split coordinate lines, an unsplit
            equivalent of the grid points is generated first and the returned points are based on that data;
            otherwise, the primary pillar data is used, which effectively gives a point from one side or
            another of any faults, rather than an averaged point
         kp (integer, default 0): set to 1 to specify the base of layer ref_k0, in case of grids with k gaps

      returns:
         a numpy array of floats of shape ((nj + 1), (ni + 1), 3) being the (shared) cell corner point
         locations for the plane of points, based on the primary pillars or unsplit equivalent pillars

      notes:
         the primary pillars are the 'first' set of points for a pillar; a split pillar will have one to
         three other sets of point data but those are ignored by this function unless heal_faults is True,
         in which case an averaged point will be used for the split pillars;
         to get full unhealed representation of split horizon points, use split_horizon_points() function
         instead;
         for grids without k gaps, ref_k0 can be used alone, in the range 0..nk, to identify the horizon;
         alternatively, or for grids with k gaps, the ref_k0 can specify the layer in the range 0..nk-1,
         with kp being passed the value 0 (default) for the top of the layer, or 1 for the base of the layer
      """

      # note: if heal_faults is False, primary pillars only are used
      pe_j = self.nj + 1
      pe_i = self.ni + 1
      if self.k_gaps:
         ref_k0 = self.k_raw_index_array[ref_k0]
      ref_k0 += kp
      if self.has_split_coordinate_lines:
         if heal_faults:
            points = self.unsplit_points_ref()  # expensive operation: would be better to cache the unsplit points
            return points[ref_k0, :, :, :].reshape((pe_j, pe_i, 3))
         else:
            points = self.points_ref(masked = False)
            return points[ref_k0, :pe_j * pe_i, :].reshape((pe_j, pe_i, 3))
      # unfaulted grid
      points = self.points_ref(masked = False)
      return points[ref_k0, :, :, :].reshape((pe_j, pe_i, 3))

   def split_horizon_points(self, ref_k0 = 0, masked = False, kp = 0):
      """Returns reference to a corner points for a horizon, of shape (nj, ni, 2, 2, 3).

      arguments:
         ref_k0 (integer): the horizon layer number, in the range 0 to nk (or layer number in range 0..nk-1
            in the case of grids with k gaps)
         masked (boolean, default False): if True, a masked array is returned with NaN points masked out;
            if False, a simple (unmasked) numpy array is returned
         kp (integer, default 0): set to 1 to specify the base of layer ref_k0, in case of grids with k gaps

      returns:
         numpy array of shape (nj, ni, 2, 2, 3) being corner point x,y,z values for cell corners (j, i, jp, ip)

      notes:
         if split points are needed for a range of horizons, it is more efficient to call split_horizons_points()
         than repeatedly call this function;
         for grids without k gaps, ref_k0 can be used alone, in the range 0..nk, to identify the horizon;
         alternatively, or for grids with k gaps, the ref_k0 can specify the layer in the range 0..nk-1,
         with kp being passed the value 0 (default) for the top of the layer, or 1 for the base of the layer
      """

      if self.k_gaps:
         ref_k0 = self.k_raw_index_array[ref_k0]
      ref_k0 += kp
      points = self.points_ref(masked = masked)
      hp = np.empty((self.nj, self.ni, 2, 2, 3))
      if self.has_split_coordinate_lines:
         assert points.ndim == 3
         # todo: replace for loops with numpy slice indexing
         self.create_column_pillar_mapping()
         assert self.pillars_for_column.ndim == 4 and self.pillars_for_column.shape == (self.nj, self.ni, 2, 2)
         for j in range(self.nj):
            for i in range(self.ni):
               hp[j, i, 0, 0, :] = points[ref_k0, self.pillars_for_column[j, i, 0, 0], :]
               hp[j, i, 1, 0, :] = points[ref_k0, self.pillars_for_column[j, i, 1, 0], :]
               hp[j, i, 0, 1, :] = points[ref_k0, self.pillars_for_column[j, i, 0, 1], :]
               hp[j, i, 1, 1, :] = points[ref_k0, self.pillars_for_column[j, i, 1, 1], :]
      else:
         assert points.ndim == 4
         hp[:, :, 0, 0, :] = points[ref_k0, :-1, :-1, :]
         hp[:, :, 1, 0, :] = points[ref_k0, 1:, :-1, :]
         hp[:, :, 0, 1, :] = points[ref_k0, :-1, 1:, :]
         hp[:, :, 1, 1, :] = points[ref_k0, 1:, 1:, :]
      return hp

   def split_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
      """Returns an array of points representing cell corners from an I or J interface slice for a faulted grid.

      arguments:
         axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
         ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
         plus_face (boolean, default False): if False, negative face is used; if True, positive
         masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out

      returns:
         a numpy array of shape (nk + 1, nj, 2, 3) or (nk + 1, ni, 2, 3) being the xyz points of the cell corners
         on the interfacial cross section; 3rd axis is jp or ip; final axis is xyz

      note:
         this function will only work for grids with no k gaps; it is intended for split grids though will also
         function for unsplit grids; use split_gap_x_section_points() if k gaps are present
      """

      log.debug(f'x-sect: axis {axis}; ref_slice0 {ref_slice0}; plus_face {plus_face}; masked {masked}')
      assert axis.upper() in ['I', 'J']
      assert not self.k_gaps, 'split_x_section_points() method is for grids without k gaps; use split_gap_x_section_points()'

      points = self.points_ref(masked = masked)
      cpm = self.create_column_pillar_mapping()

      ij_p = 1 if plus_face else 0

      if axis.upper() == 'I':
         return points[:, cpm[:, ref_slice0, :, ij_p], :]
      else:
         return points[:, cpm[ref_slice0, :, ij_p, :], :]

   def split_gap_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
      """Returns an array of points representing cell corners from an I or J interface slice for a faulted grid with k gaps.

      arguments:
         axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
         ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
         plus_face (boolean, default False): if False, negative face is used; if True, positive
         masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out

      returns:
         a numpy array of shape (nk, nj, 2, 2, 3) or (nk, ni, 2, 2, 3) being the xyz points of the cell corners
         on the interfacial cross section; 3rd axis is kp; 4th axis is jp or ip; final axis is xyz

      note:
         this function is intended for split grids with k gaps though will also function for split grids
         without k gaps
      """

      log.debug(f'k gap x-sect: axis {axis}; ref_slice0 {ref_slice0}; plus_face {plus_face}; masked {masked}')
      assert axis.upper() in ['I', 'J']

      if self.has_split_coordinate_lines:
         points = self.points_ref(masked = masked)
      else:
         points = self.points_ref(masked = masked).reshape((self.nk_plus_k_gaps, (self.nj + 1) * (self.ni + 1), 3))
      cpm = self.create_column_pillar_mapping()

      ij_p = 1 if plus_face else 0

      if self.k_gaps:
         top_points = points[self.k_raw_index_array]
         base_points = points[self.k_raw_index_array + 1]
         if axis.upper() == 'I':
            top = top_points[:, cpm[:, ref_slice0, :, ij_p], :]
            base = base_points[:, cpm[:, ref_slice0, :, ij_p], :]
         else:
            top = top_points[:, cpm[ref_slice0, :, ij_p, :], :]
            base = base_points[:, cpm[ref_slice0, :, ij_p, :], :]
      else:
         p = self.split_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)
         top = p[:-1]
         base = p[1:]
      return np.stack((top, base), axis = 2)

   def unsplit_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
      """Returns a 2D (+1 for xyz) array of points representing cell corners from an I or J interface slice.

      arguments:
         axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
         ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
         plus_face (boolean, default False): if False, negative face is used; if True, positive
         masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out

      returns:
         a 2+1D numpy array being the xyz points of the cell corners on the interfacial cross section;
         the 2D axes are K,J or K,I - whichever does not involve axis; shape is (nk + 1, nj + 1, 3) or
         (nk + 1, ni + 1, 3)

      note:
         restricted to unsplit grids with no k gaps; use split_x_section_points() for split grids with no k gaps
         or split_gap_x_section_points() for split grids with k gaps or x_section_corner_points() for any grid
      """

      log.debug(f'x-sect: axis {axis}; ref_slice0 {ref_slice0}; plus_face {plus_face}; masked {masked}')
      assert axis.upper() in ['I', 'J']
      assert not self.has_split_coordinate_lines, 'cross sectional points for unsplit grids require split_x_section_points()'
      assert not self.k_gaps, 'cross sectional points with k gaps require split_gap_x_section_points()'

      if plus_face:
         ref_slice0 += 1

      points = self.points_ref(masked = masked)

      if axis.upper() == 'I':
         return points[:, :, ref_slice0, :]
      else:
         return points[:, ref_slice0, :, :]

   def x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
      """Deprecated: please use unsplit_x_section_points() instead."""

      return self.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

   def x_section_corner_points(self,
                               axis,
                               ref_slice0 = 0,
                               plus_face = False,
                               masked = False,
                               rotate = False,
                               azimuth = None):
      """Returns a fully expanded array of points representing cell corners from an I or J interface slice.

      arguments:
         axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
         ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
         plus_face (boolean, default False): if False, negative face is used; if True, positive
         masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out
         rotate (boolean, default False): if True, the cross section points are rotated around the z axis so that
            an azimuthal direction is mapped onto the positive x axis
         aximuth (float, optional): the compass bearing in degrees to map onto the positive x axis if rotating;
            if None, the mean direction of the cross sectional points, along axis, is used; ignored if rotate
            is False

      returns:
         a numpy float array of shape (nk, nj, 2, 2, 3) or (nk, ni, 2, 2, 3) being the xyz points of the cell
         corners on the interfacial cross section; the 3rd index (1st 2) is kp, the 4th index is jp or ip

      note:
         this method will work for unsplit or split grids, with or without k gaps; use rotate argument to yield
         points with predominant variation in xz, suitable for plotting cross sections; if rotate is True then
         the absolute values of x & y will not be very meaningful though the units will still be the grid's xy
         units for relative purposes
      """

      assert axis.upper() in ['I', 'J']
      nj_or_ni = self.nj if axis.upper() == 'I' else self.ni

      if self.k_gaps:
         x_sect = self.split_gap_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)
      else:
         if self.has_split_coordinate_lines:
            no_k_gap_xs = self.split_x_section_points(axis,
                                                      ref_slice0 = ref_slice0,
                                                      plus_face = plus_face,
                                                      masked = masked)
            x_sect = np.empty((self.nk, nj_or_ni, 2, 2, 3))
            x_sect[:, :, 0, :, :] = no_k_gap_xs[:-1, :, :, :]
            x_sect[:, :, 1, :, :] = no_k_gap_xs[1:, :, :, :]
         else:
            simple_xs = self.unsplit_x_section_points(axis,
                                                      ref_slice0 = ref_slice0,
                                                      plus_face = plus_face,
                                                      masked = masked)
            x_sect = np.empty((self.nk, nj_or_ni, 2, 2, 3))
            x_sect[:, :, 0, 0, :] = simple_xs[:-1, :-1, :]
            x_sect[:, :, 1, 0, :] = simple_xs[1:, :-1, :]
            x_sect[:, :, 0, 1, :] = simple_xs[:-1, 1:, :]
            x_sect[:, :, 1, 1, :] = simple_xs[1:, 1:, :]

      if rotate:
         if azimuth is None:
            direction = vec.points_direction_vector(x_sect, axis = 1)
         else:
            direction = vec.unit_vector_from_azimuth(azimuth)
         x_sect = vec.rotate_xyz_array_around_z_axis(x_sect, direction)
         x_sect[..., 0] -= np.nanmin(x_sect[..., 0])
         x_sect[..., 1] -= np.nanmin(x_sect[..., 1])

      return x_sect

   def pixel_map_for_split_horizon_points(self, horizon_points, origin, width, height, dx, dy = None):
      """Makes a mapping from pixels to cell j, i indices, based on split horizon points for a single horizon.

      args:
         horizon_points (numpy array of shape (nj, ni, 2, 2, 2+)): corner point x,y,z values for cell
            corners (j, i, jp, ip); as returned by split_horizon_points()
         origin (float pair): x, y of south west corner of area covered by pixel rectangle, in local crs
         width (int): the width of the pixel rectangle (number of pixels)
         height (int): the height of the pixel rectangle (number of pixels)
         dx (float): the size (west to east) of a pixel, in locel crs
         dx (float, optional): the size (south to north) of a pixel, in locel crs; defaults to dx

      returns:
         numpy int array of shape (height, width, 2), being the j, i indices of cells that the pixel centres lie within;
         values of -1 are used as null (ie. pixel not within any cell)
      """

      if dy is None:
         dy = dx
      half_dx = 0.5 * dx
      half_dy = 0.5 * dy
      d = np.array((dx, dy))
      half_d = np.array((half_dx, half_dy))

      #     north_east = np.array(origin) + np.array((width * dx, height * dy))

      p_map = np.full((height, width, 2), -1, dtype = int)

      # switch from logical corner ordering to polygon ordering
      poly_points = horizon_points[..., :2].copy()
      poly_points[:, :, 1, 1] = horizon_points[:, :, 1, 0, :2]
      poly_points[:, :, 1, 0] = horizon_points[:, :, 1, 1, :2]
      poly_points = poly_points.reshape(horizon_points.shape[0], horizon_points.shape[1], 4, 2)

      poly_box = np.empty((2, 2))
      patch_p_origin = np.empty((2,), dtype = int)  # NB. ordering is (ncol, nrow)
      patch_origin = np.empty((2,))
      patch_extent = np.empty((2,), dtype = int)  # NB. ordering is (ncol, nrow)

      for j in range(poly_points.shape[0]):
         for i in range(poly_points.shape[1]):
            if np.any(np.isnan(poly_points[j, i])):
               continue
            poly_box[0] = np.min(poly_points[j, i], axis = 0) - half_d
            poly_box[1] = np.max(poly_points[j, i], axis = 0) + half_d
            patch_p_origin[:] = (poly_box[0] - origin) / (dx, dy)
            if patch_p_origin[0] < 0 or patch_p_origin[1] < 0:
               continue
            patch_extent[:] = np.ceil((poly_box[1] - poly_box[0]) / (dx, dy))
            if patch_p_origin[0] + patch_extent[0] > width or patch_p_origin[1] + patch_extent[1] > height:
               continue
            patch_origin = origin + d * patch_p_origin + half_d
            scan_mask = pip.scan(patch_origin, patch_extent[0], patch_extent[1], dx, dy, poly_points[j, i])
            patch_mask = np.stack((scan_mask, scan_mask), axis = -1)
            old_patch = p_map[patch_p_origin[1]:patch_p_origin[1] + patch_extent[1],
                              patch_p_origin[0]:patch_p_origin[0] + patch_extent[0], :].copy()
            new_patch = np.empty(old_patch.shape, dtype = int)
            new_patch[:, :] = (j, i)
            p_map[patch_p_origin[1]:patch_p_origin[1] + patch_extent[1], patch_p_origin[0]:patch_p_origin[0] + patch_extent[0], :] =  \
               np.where(patch_mask, new_patch, old_patch)
      return p_map

   def pixel_maps(self, origin, width, height, dx, dy = None, k0 = None, vertical_ref = 'top'):
      """Makes a mapping from pixels to cell j, i indices, based on split horizon points for a single horizon.

      args:
         origin (float pair): x, y of south west corner of area covered by pixel rectangle, in local crs
         width (int): the width of the pixel rectangle (number of pixels)
         height (int): the height of the pixel rectangle (number of pixels)
         dx (float): the size (west to east) of a pixel, in locel crs
         dy (float, optional): the size (south to north) of a pixel, in locel crs; defaults to dx
         k0 (int, default None): if present, the single layer to create a 2D pixel map for; if None, a 3D map
            is created with one layer per layer of the grid
         vertical_ref (string, default 'top'): 'top' or 'base'

      returns:
         numpy int array of shape (height, width, 2), or (nk, height, width, 2), being the j, i indices of cells
         that the pixel centres lie within; values of -1 are used as null (ie. pixel not within any cell)
      """

      if len(origin) == 3:
         origin = tuple(origin[0:2])
      assert len(origin) == 2
      assert width > 0 and height > 0
      if dy is None:
         dy = dx
      assert dx > 0.0 and dy > 0.0
      if k0 is not None:
         assert 0 <= k0 < self.nk
      assert vertical_ref in ['top', 'base']

      kp = 0 if vertical_ref == 'top' else 1
      if k0 is not None:
         hp = self.split_horizon_points(ref_k0 = k0, masked = False, kp = kp)
         p_map = self.pixel_map_for_split_horizon_points(hp, origin, width, height, dx, dy = dy)
      else:
         _, _, raw_k = self.extract_k_gaps()
         hp = self.split_horizons_points(masked = False)
         p_map = np.empty((self.nk, height, width, 2), dtype = int)
         for k0 in range(self.nk):
            rk0 = raw_k[k0] + kp
            p_map[k0] = self.pixel_map_for_split_horizon_points(hp[rk0], origin, width, height, dx, dy = dy)
      return p_map

   def split_horizons_points(self, min_k0 = None, max_k0 = None, masked = False):
      """Returns reference to a corner points layer of shape (nh, nj, ni, 2, 2, 3) where nh is number of horizons.

      arguments:
         min_k0 (integer): the lowest horizon layer number to be included, in the range 0 to nk + k_gaps; defaults to zero
         max_k0 (integer): the highest horizon layer number to be included, in the range 0 to nk + k_gaps; defaults to nk + k_gaps
         masked (boolean, default False): if True, a masked array is returned with NaN points masked out;
            if False, a simple (unmasked) numpy array is returned

      returns:
         numpy array of shape (nh, nj, ni, 2, 2, 3) where nh = max_k0 - min_k0 + 1, being corner point x,y,z values
         for horizon corners (h, j, i, jp, ip) where h is the horizon (layer interface) index in the range
         0 .. max_k0 - min_k0

      notes:
         data for horizon max_k0 is included in the result (unlike with python ranges);
         in the case of a grid with k gaps, the horizons points returned will follow the k indexing of the points data
         and calling code will need to keep track of the min_k0 offset when using k_raw_index_array to select a slice
         of the horizons points array
      """

      if min_k0 is None:
         min_k0 = 0
      else:
         assert min_k0 >= 0 and min_k0 <= self.nk_plus_k_gaps
      if max_k0 is None:
         max_k0 = self.nk_plus_k_gaps
      else:
         assert max_k0 >= min_k0 and max_k0 <= self.nk_plus_k_gaps
      end_k0 = max_k0 + 1
      points = self.points_ref(masked = False)
      hp = np.empty((end_k0 - min_k0, self.nj, self.ni, 2, 2, 3))
      if self.has_split_coordinate_lines:
         self.create_column_pillar_mapping()
         for j in range(self.nj):
            for i in range(self.ni):
               hp[:, j, i, 0, 0, :] = points[min_k0:end_k0, self.pillars_for_column[j, i, 0, 0], :]
               hp[:, j, i, 1, 0, :] = points[min_k0:end_k0, self.pillars_for_column[j, i, 1, 0], :]
               hp[:, j, i, 0, 1, :] = points[min_k0:end_k0, self.pillars_for_column[j, i, 0, 1], :]
               hp[:, j, i, 1, 1, :] = points[min_k0:end_k0, self.pillars_for_column[j, i, 1, 1], :]
      else:
         hp[:, :, :, 0, 0, :] = points[min_k0:end_k0, :-1, :-1, :]
         hp[:, :, :, 1, 0, :] = points[min_k0:end_k0, 1:, :-1, :]
         hp[:, :, :, 0, 1, :] = points[min_k0:end_k0, :-1, 1:, :]
         hp[:, :, :, 1, 1, :] = points[min_k0:end_k0, 1:, 1:, :]
      return hp

   def pillar_distances_sqr(self, xy, ref_k0 = 0, kp = 0, horizon_points = None):
      """Returns array of the square of the distances of primary pillars in x,y plane to point xy.

      arguments:
         xy (float pair): the xy coordinate to compute the pillar distances to
         ref_k0 (int, default 0): the horizon layer number to use
         horizon_points (numpy array, optional): if present, should be array as returned by
            horizon_points() method; pass for efficiency in case of multiple calls
      """

      # note: currently works with unmasked data and using primary pillars only
      pe_j = self.extent_kji[1] + 1
      pe_i = self.extent_kji[2] + 1
      if horizon_points is None:
         horizon_points = self.horizon_points(ref_k0 = ref_k0, kp = kp)
      pillar_xy = horizon_points[:, :, 0:2]
      dxy = pillar_xy - xy
      dxy2 = dxy * dxy
      return (dxy2[:, :, 0] + dxy2[:, :, 1]).reshape((pe_j, pe_i))

   def nearest_pillar(self, xy, ref_k0 = 0, kp = 0):
      """Returns the (j0, i0) indices of the primary pillar with point closest in x,y plane to point xy."""

      # note: currently works with unmasked data and using primary pillars only
      pe_i = self.extent_kji[2] + 1
      sum_dxy2 = self.pillar_distances_sqr(xy, ref_k0 = ref_k0, kp = kp)
      ji = np.nanargmin(sum_dxy2)
      j, i = divmod(ji, pe_i)
      return (j, i)

   def nearest_rod(self, xyz, projection, axis, ref_slice0 = 0, plus_face = False):
      """Returns the (k0, j0) or (k0 ,i0) indices of the closest point(s) to xyz(s); projection is 'xy', 'xz' or 'yz'.

      note:
         currently only for unsplit grids
      """

      x_sect = self.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)
      if type(xyz) is np.ndarray and xyz.ndim > 1:
         assert xyz.shape[-1] == 3
         result_shape = list(xyz.shape)
         result_shape[-1] = 2
         nearest = np.empty(tuple(result_shape), dtype = int).reshape((-1, 2))
         for i, p in enumerate(xyz.reshape((-1, 3))):
            nearest[i] = vec.nearest_point_projected(p, x_sect, projection)
         return nearest.reshape(tuple(result_shape))
      else:
         return vec.nearest_point_projected(xyz, x_sect, projection)

   def coordinate_line_end_points(self):
      """Returns xyz of top and bottom of each primary pillar.

      returns:
         numpy float array of shape (nj + 1, ni + 1, 2, 3)
      """

      points = self.points_ref(masked = False).reshape((self.nk + 1, -1, 3))
      primary_pillar_count = (self.nj + 1) * (self.ni + 1)
      result = np.empty((self.nj + 1, self.ni + 1, 2, 3))
      result[:, :, 0, :] = points[0, :primary_pillar_count, :].reshape((self.nj + 1, self.ni + 1, 3))
      result[:, :, 1, :] = points[-1, :primary_pillar_count, :].reshape((self.nj + 1, self.ni + 1, 3))
      return result

   def z_corner_point_depths(self, order = 'cellular'):
      """Returns the z (depth) values of each corner of each cell.

      arguments:
         order (string, default 'cellular'): either 'cellular' or 'linear'; if 'cellular' the resulting array has
            shape (nk, nj, ni, 2, 2, 2); if 'linear', the shape is (nk, 2, nj, 2, ni, 2)

      returns:
         numpy array of shape (nk, nj, ni, 2, 2, 2) or (nk, 2, nj, 2, ni, 2); for the cellular ordering, the
         result can be indexed with [k, j, i, kp, jp, ip] (where kp, for example, is 0 for the K- face and 1 for K+);
         for the linear ordering, the equivalent indexing is [k, kp, j, jp, i, ip], as used by some common simulator
         keyword formats
      """

      assert order in ['cellular', 'linear']

      z_cp = np.empty((self.nk, self.nj, self.ni, 2, 2, 2))
      points = self.points_ref()
      if self.has_split_coordinate_lines:
         self.create_column_pillar_mapping()
         # todo: replace j,i for loops with numpy broadcasting
         if self.k_gaps:
            for j in range(self.nj):
               for i in range(self.ni):
                  z_cp[:, j, i, 0, 0, 0] = points[self.k_raw_index_array, self.pillars_for_column[j, i, 0, 0], 2]
                  z_cp[:, j, i, 1, 0, 0] = points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 0, 0], 2]
                  z_cp[:, j, i, 0, 1, 0] = points[self.k_raw_index_array, self.pillars_for_column[j, i, 1, 0], 2]
                  z_cp[:, j, i, 1, 1, 0] = points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 1, 0], 2]
                  z_cp[:, j, i, 0, 0, 1] = points[self.k_raw_index_array, self.pillars_for_column[j, i, 0, 1], 2]
                  z_cp[:, j, i, 1, 0, 1] = points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 0, 1], 2]
                  z_cp[:, j, i, 0, 1, 1] = points[self.k_raw_index_array, self.pillars_for_column[j, i, 1, 1], 2]
                  z_cp[:, j, i, 1, 1, 1] = points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 1, 1], 2]
         else:
            for j in range(self.nj):
               for i in range(self.ni):
                  z_cp[:, j, i, 0, 0, 0] = points[:-1, self.pillars_for_column[j, i, 0, 0], 2]
                  z_cp[:, j, i, 1, 0, 0] = points[1:, self.pillars_for_column[j, i, 0, 0], 2]
                  z_cp[:, j, i, 0, 1, 0] = points[:-1, self.pillars_for_column[j, i, 1, 0], 2]
                  z_cp[:, j, i, 1, 1, 0] = points[1:, self.pillars_for_column[j, i, 1, 0], 2]
                  z_cp[:, j, i, 0, 0, 1] = points[:-1, self.pillars_for_column[j, i, 0, 1], 2]
                  z_cp[:, j, i, 1, 0, 1] = points[1:, self.pillars_for_column[j, i, 0, 1], 2]
                  z_cp[:, j, i, 0, 1, 1] = points[:-1, self.pillars_for_column[j, i, 1, 1], 2]
                  z_cp[:, j, i, 1, 1, 1] = points[1:, self.pillars_for_column[j, i, 1, 1], 2]
      else:
         if self.k_gaps:
            z_cp[:, :, :, 0, 0, 0] = points[self.k_raw_index_array, :-1, :-1, 2]
            z_cp[:, :, :, 1, 0, 0] = points[self.k_raw_index_array + 1, :-1, :-1, 2]
            z_cp[:, :, :, 0, 1, 0] = points[self.k_raw_index_array, 1:, :-1, 2]
            z_cp[:, :, :, 1, 1, 0] = points[self.k_raw_index_array + 1, 1:, :-1, 2]
            z_cp[:, :, :, 0, 0, 1] = points[self.k_raw_index_array, :-1, 1:, 2]
            z_cp[:, :, :, 1, 0, 1] = points[self.k_raw_index_array + 1, :-1, 1:, 2]
            z_cp[:, :, :, 0, 1, 1] = points[self.k_raw_index_array, 1:, 1:, 2]
            z_cp[:, :, :, 1, 1, 1] = points[self.k_raw_index_array + 1, 1:, 1:, 2]
         else:
            z_cp[:, :, :, 0, 0, 0] = points[:-1, :-1, :-1, 2]
            z_cp[:, :, :, 1, 0, 0] = points[1:, :-1, :-1, 2]
            z_cp[:, :, :, 0, 1, 0] = points[:-1, 1:, :-1, 2]
            z_cp[:, :, :, 1, 1, 0] = points[1:, 1:, :-1, 2]
            z_cp[:, :, :, 0, 0, 1] = points[:-1, :-1, 1:, 2]
            z_cp[:, :, :, 1, 0, 1] = points[1:, :-1, 1:, 2]
            z_cp[:, :, :, 0, 1, 1] = points[:-1, 1:, 1:, 2]
            z_cp[:, :, :, 1, 1, 1] = points[1:, 1:, 1:, 2]

      if order == 'linear':
         return np.transpose(z_cp, axes = (0, 3, 1, 4, 2, 5))
      return z_cp

   def corner_points(self, cell_kji0 = None, points_root = None, cache_resqml_array = True, cache_cp_array = False):
      """Returns a numpy array of corner points for a single cell or the whole grid.

      notes:
         if cell_kji0 is not None, a 4D array of shape (2, 2, 2, 3) holding single cell corner points in logical order
         [kp, jp, ip, xyz] is returned; if cell_kji0 is None, a pagoda style 7D array [k, j, i, kp, jp, ip, xyz] is
         cached and returned;
         the ordering of the corner points is in the logical order, which is not the same as that used by Nexus CORP data;
         olio.grid_functions.resequence_nexus_corp() can be used to switch back and forth between this pagoda ordering
         and Nexus corp ordering;
         this is the usual way to access full corner points for cells where working with native resqml data is undesirable

      :meta common:
      """

      # note: this function returns a derived object rather than a native resqml object

      def one_cell_cp(grid, cell_kji0, points_root, cache_array):
         cp = np.full((2, 2, 2, 3), np.NaN)
         if not grid.geometry_defined_for_all_cells():
            if not grid.cell_geometry_is_defined(cell_kji0, cache_array = cache_array):
               return cp
         corner_index = np.zeros(3, dtype = 'int')
         for kp in range(2):
            corner_index[0] = kp
            for jp in range(2):
               corner_index[1] = jp
               for ip in range(2):
                  corner_index[2] = ip
                  one_point = self.point(cell_kji0,
                                         corner_index = corner_index,
                                         points_root = points_root,
                                         cache_array = cache_array)
                  if one_point is not None:
                     cp[kp, jp, ip] = one_point
         return cp

      if cell_kji0 is None:
         cache_cp_array = True
      if hasattr(self, 'array_corner_points'):
         if cell_kji0 is None:
            return self.array_corner_points
         return self.array_corner_points[tuple(cell_kji0)]
      points_root = self.resolve_geometry_child('Points', child_node = points_root)
      #      if points_root is None: return None  # geometry not present
      if cache_resqml_array:
         self.point_raw(points_root = points_root, cache_array = True)
      if cache_cp_array:
         self.array_corner_points = np.zeros((self.nk, self.nj, self.ni, 2, 2, 2, 3))
         points = self.points_ref()
         if points is None:
            return None  # geometry not present
         if self.has_split_coordinate_lines:
            self.create_column_pillar_mapping()
            # todo: replace j,i for loops with numpy broadcasting
            if self.k_gaps:
               for j in range(self.nj):
                  for i in range(self.ni):
                     self.array_corner_points[:, j, i, 0, 0, 0, :] = points[self.k_raw_index_array,
                                                                            self.pillars_for_column[j, i, 0, 0], :]
                     self.array_corner_points[:, j, i, 1, 0, 0, :] = points[self.k_raw_index_array + 1,
                                                                            self.pillars_for_column[j, i, 0, 0], :]
                     self.array_corner_points[:, j, i, 0, 1, 0, :] = points[self.k_raw_index_array,
                                                                            self.pillars_for_column[j, i, 1, 0], :]
                     self.array_corner_points[:, j, i, 1, 1, 0, :] = points[self.k_raw_index_array + 1,
                                                                            self.pillars_for_column[j, i, 1, 0], :]
                     self.array_corner_points[:, j, i, 0, 0, 1, :] = points[self.k_raw_index_array,
                                                                            self.pillars_for_column[j, i, 0, 1], :]
                     self.array_corner_points[:, j, i, 1, 0, 1, :] = points[self.k_raw_index_array + 1,
                                                                            self.pillars_for_column[j, i, 0, 1], :]
                     self.array_corner_points[:, j, i, 0, 1, 1, :] = points[self.k_raw_index_array,
                                                                            self.pillars_for_column[j, i, 1, 1], :]
                     self.array_corner_points[:, j, i, 1, 1, 1, :] = points[self.k_raw_index_array + 1,
                                                                            self.pillars_for_column[j, i, 1, 1], :]
            else:
               for j in range(self.nj):
                  for i in range(self.ni):
                     self.array_corner_points[:, j, i, 0, 0, 0, :] = points[:-1, self.pillars_for_column[j, i, 0, 0], :]
                     self.array_corner_points[:, j, i, 1, 0, 0, :] = points[1:, self.pillars_for_column[j, i, 0, 0], :]
                     self.array_corner_points[:, j, i, 0, 1, 0, :] = points[:-1, self.pillars_for_column[j, i, 1, 0], :]
                     self.array_corner_points[:, j, i, 1, 1, 0, :] = points[1:, self.pillars_for_column[j, i, 1, 0], :]
                     self.array_corner_points[:, j, i, 0, 0, 1, :] = points[:-1, self.pillars_for_column[j, i, 0, 1], :]
                     self.array_corner_points[:, j, i, 1, 0, 1, :] = points[1:, self.pillars_for_column[j, i, 0, 1], :]
                     self.array_corner_points[:, j, i, 0, 1, 1, :] = points[:-1, self.pillars_for_column[j, i, 1, 1], :]
                     self.array_corner_points[:, j, i, 1, 1, 1, :] = points[1:, self.pillars_for_column[j, i, 1, 1], :]
         else:
            if self.k_gaps:
               self.array_corner_points[:, :, :, 0, 0, 0, :] = points[self.k_raw_index_array, :-1, :-1, :]
               self.array_corner_points[:, :, :, 1, 0, 0, :] = points[self.k_raw_index_array + 1, :-1, :-1, :]
               self.array_corner_points[:, :, :, 0, 1, 0, :] = points[self.k_raw_index_array, 1:, :-1, :]
               self.array_corner_points[:, :, :, 1, 1, 0, :] = points[self.k_raw_index_array + 1, 1:, :-1, :]
               self.array_corner_points[:, :, :, 0, 0, 1, :] = points[self.k_raw_index_array, :-1, 1:, :]
               self.array_corner_points[:, :, :, 1, 0, 1, :] = points[self.k_raw_index_array + 1, :-1, 1:, :]
               self.array_corner_points[:, :, :, 0, 1, 1, :] = points[self.k_raw_index_array, 1:, 1:, :]
               self.array_corner_points[:, :, :, 1, 1, 1, :] = points[self.k_raw_index_array + 1, 1:, 1:, :]
            else:
               self.array_corner_points[:, :, :, 0, 0, 0, :] = points[:-1, :-1, :-1, :]
               self.array_corner_points[:, :, :, 1, 0, 0, :] = points[1:, :-1, :-1, :]
               self.array_corner_points[:, :, :, 0, 1, 0, :] = points[:-1, 1:, :-1, :]
               self.array_corner_points[:, :, :, 1, 1, 0, :] = points[1:, 1:, :-1, :]
               self.array_corner_points[:, :, :, 0, 0, 1, :] = points[:-1, :-1, 1:, :]
               self.array_corner_points[:, :, :, 1, 0, 1, :] = points[1:, :-1, 1:, :]
               self.array_corner_points[:, :, :, 0, 1, 1, :] = points[:-1, 1:, 1:, :]
               self.array_corner_points[:, :, :, 1, 1, 1, :] = points[1:, 1:, 1:, :]
      if cell_kji0 is None:
         return self.array_corner_points
      if not self.geometry_defined_for_all_cells():
         if not self.cell_geometry_is_defined(cell_kji0, cache_array = cache_resqml_array):
            return None
      if hasattr(self, 'array_corner_points'):
         return self.array_corner_points[tuple(cell_kji0)]
      cp = one_cell_cp(self, cell_kji0, points_root = points_root, cache_array = cache_resqml_array)
      return cp

   def invalidate_corner_points(self):
      """Deletes cached copy of corner points, if present; use if any pillar geometry changes, or to reclaim memory."""

      if hasattr(self, 'array_corner_points'):
         delattr(self, 'array_corner_points')

   def centre_point(self, cell_kji0 = None, cache_centre_array = False):
      """Returns centre point of a cell or array of centre points of all cells; optionally cache centre points for all cells.

      arguments:
         cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
            centre point is required; zero based indexing
         cache_centre_array (boolean, default False): If True, or cell_kji0 is None, an array of centre points
            is generated and added as an attribute of the grid, with attribute name array_centre_point

      returns:
         (x, y, z) 3 element numpy array of floats holding centre point of cell;
         or numpy 3+1D array if cell_kji0 is None

      note:
         resulting coordinates are in the same (local) crs as the grid points

      :meta common:
      """

      if cell_kji0 is None:
         cache_centre_array = True

      # note: this function returns a derived object rather than a native resqml object
      if hasattr(self, 'array_centre_point'):
         if cell_kji0 is None:
            return self.array_centre_point
         return self.array_centre_point[tuple(cell_kji0)]  # could check for nan here and return None
      if cache_centre_array:
         # todo: turn off nan warnings
         self.array_centre_point = np.empty((self.nk, self.nj, self.ni, 3))
         points = self.points_ref(masked = False)  # todo: think about masking
         if hasattr(self, 'array_corner_points'):
            self.array_centre_point = 0.125 * np.sum(self.array_corner_points,
                                                     axis = (3, 4, 5))  # mean of eight corner points for each cell
         elif self.has_split_coordinate_lines:
            # todo: replace j,i for loops with numpy broadcasting
            self.create_column_pillar_mapping()
            if self.k_gaps:
               for j in range(self.nj):
                  for i in range(self.ni):
                     self.array_centre_point[:, j, i, :] = 0.125 * (
                        points[self.k_raw_index_array, self.pillars_for_column[j, i, 0, 0], :] +
                        points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 0, 0], :] +
                        points[self.k_raw_index_array, self.pillars_for_column[j, i, 1, 0], :] +
                        points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 1, 0], :] +
                        points[self.k_raw_index_array, self.pillars_for_column[j, i, 0, 1], :] +
                        points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 0, 1], :] +
                        points[self.k_raw_index_array, self.pillars_for_column[j, i, 1, 1], :] +
                        points[self.k_raw_index_array + 1, self.pillars_for_column[j, i, 1, 1], :])
            else:
               for j in range(self.nj):
                  for i in range(self.ni):
                     self.array_centre_point[:, j,
                                             i, :] = 0.125 * (points[:-1, self.pillars_for_column[j, i, 0, 0], :] +
                                                              points[1:, self.pillars_for_column[j, i, 0, 0], :] +
                                                              points[:-1, self.pillars_for_column[j, i, 1, 0], :] +
                                                              points[1:, self.pillars_for_column[j, i, 1, 0], :] +
                                                              points[:-1, self.pillars_for_column[j, i, 0, 1], :] +
                                                              points[1:, self.pillars_for_column[j, i, 0, 1], :] +
                                                              points[:-1, self.pillars_for_column[j, i, 1, 1], :] +
                                                              points[1:, self.pillars_for_column[j, i, 1, 1], :])
         else:
            if self.k_gaps:
               self.array_centre_point[:, :, :, :] = 0.125 * (
                  points[self.k_raw_index_array, :-1, :-1, :] + points[self.k_raw_index_array, :-1, 1:, :] +
                  points[self.k_raw_index_array, 1:, :-1, :] + points[self.k_raw_index_array, 1:, 1:, :] +
                  points[self.k_raw_index_array + 1, :-1, :-1, :] + points[self.k_raw_index_array + 1, :-1, 1:, :] +
                  points[self.k_raw_index_array + 1, 1:, :-1, :] + points[self.k_raw_index_array + 1, 1:, 1:, :])
            else:
               self.array_centre_point[:, :, :, :] = 0.125 * (points[:-1, :-1, :-1, :] + points[:-1, :-1, 1:, :] +
                                                              points[:-1, 1:, :-1, :] + points[:-1, 1:, 1:, :] +
                                                              points[1:, :-1, :-1, :] + points[1:, :-1, 1:, :] +
                                                              points[1:, 1:, :-1, :] + points[1:, 1:, 1:, :])
         if cell_kji0 is None:
            return self.array_centre_point
         return self.array_centre_point[cell_kji0[0], cell_kji0[1],
                                        cell_kji0[2]]  # could check for nan here and return None
      cp = self.corner_points(cell_kji0 = cell_kji0, cache_cp_array = False)
      if cp is None:
         return None
      centre = np.zeros(3)
      for axis in range(3):
         centre[axis] = np.mean(cp[:, :, :, axis])
      return centre

   def centre_point_list(self, cell_kji0s):
      """Returns centre points for a list of cells; caches centre points for all cells.

      arguments:
         cell_kji0s (numpy int array of shape (N, 3)): the (k, j, i) indices of the individual cells for which the
            centre points are required; zero based indexing

      returns:
         numpy float array of shape (N, 3) being the (x, y, z) centre points of the cells

      note:
         resulting coordinates are in the same (local) crs as the grid points
      """

      assert cell_kji0s.ndim == 2 and cell_kji0s.shape[1] == 3
      centres_list = np.empty(cell_kji0s.shape)
      for cell in range(len(cell_kji0s)):
         centres_list[cell] = self.centre_point(cell_kji0 = cell_kji0s[cell], cache_centre_array = True)
      return centres_list

   def thickness(self,
                 cell_kji0 = None,
                 points_root = None,
                 cache_resqml_array = True,
                 cache_cp_array = False,
                 cache_thickness_array = True,
                 property_collection = None):
      """Returns vertical (z) thickness of cell and/or caches thicknesses for all cells.

      arguments:
         cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
                               thickness is required; zero based indexing
         cache_resqml_array (boolean, default True): If True, the raw points array from the hdf5 file
                               is cached in memory, but only if it is needed to generate the thickness
         cache_cp_array (boolean, default True): If True, an array of corner points is generated and
                               added as an attribute of the grid, with attribute name corner_points, but only
                               if it is needed in order to generate the thickness
         cache_thickness_array (boolean, default False): if True, thicknesses are generated for all cells in
                               the grid and added as an attribute named array_thickness
         property_collection (property:GridPropertyCollection, optional): If not None, this collection
                               is probed for a suitable thickness or cell length property which is used
                               preferentially to calculating thickness; if no suitable property is found,
                               the calculation is made as if the collection were None

      returns:
         float, being the thickness of cell identified by cell_kji0; or numpy float array if cell_kji0 is None

      notes:
         the function can be used to find the thickness of a single cell, or cache thickness for all cells, or both;
         if property_collection is not None, a suitable thickness or cell length property will be used if present;
         if calculated, thickness is defined as z difference between centre points of top and base faces (TVT);
         at present, assumes K increases with same polarity as z; if not, negative thickness will be calculated;
         units of result are implicitly those of z coordinates in grid's coordinate reference system, or units of
         measure of property array if the result is based on a suitable property

      :meta common:
      """

      def load_from_property(collection):
         if collection is None:
            return None
         parts = collection.selective_parts_list(property_kind = 'thickness', facet_type = 'netgross', facet = 'gross')
         if len(parts) == 1:
            return collection.cached_part_array_ref(parts[0])
         parts = collection.selective_parts_list(property_kind = 'thickness')
         if len(parts) == 1 and collection.facet_for_part(parts[0]) is None:
            return collection.cached_part_array_ref(parts[0])
         parts = collection.selective_parts_list(property_kind = 'cell length', facet_type = 'direction', facet = 'K')
         if len(parts) == 1:
            return collection.cached_part_array_ref(parts[0])
         return None

      # note: this function optionally looks for a suitable thickness property, otherwise calculates from geometry
      # note: for some geometries, thickness might need to be defined as length of vector between -k & +k face centres (TST)
      # todo: give more control over source of data through optional args; offer TST or TVT option
      # todo: if cp array is not already cached, compute directly from points without generating cp
      # todo: cache uom
      assert cache_thickness_array or (cell_kji0 is not None)

      if hasattr(self, 'array_thickness'):
         if cell_kji0 is None:
            return self.array_thickness
         return self.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None

      thick = load_from_property(property_collection)
      if thick is not None:
         log.debug('thickness array loaded from property')
         if cache_thickness_array:
            self.array_thickness = thick.copy()
         if cell_kji0 is None:
            return self.array_thickness
         return self.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None

      points_root = self.resolve_geometry_child('Points', child_node = points_root)
      if cache_thickness_array:
         if cache_cp_array:
            self.corner_points(points_root = points_root, cache_cp_array = True)
         if hasattr(self, 'array_corner_points'):
            self.array_thickness = np.abs(
               np.mean(self.array_corner_points[:, :, :, 1, :, :, 2] - self.array_corner_points[:, :, :, 0, :, :, 2],
                       axis = (3, 4)))
            if cell_kji0 is None:
               return self.array_thickness
            return self.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None
         self.array_thickness = np.empty(tuple(self.extent_kji))
         points = self.point_raw(cache_array = True)  # cache points regardless
         if points is None:
            return None  # geometry not present
         if self.k_gaps:
            pillar_thickness = points[self.k_raw_index_array + 1, ..., 2] - points[self.k_raw_index_array, ..., 2]
         else:
            pillar_thickness = points[1:, ..., 2] - points[:-1, ..., 2]
         if self.has_split_coordinate_lines:
            pillar_for_col = self.create_column_pillar_mapping()
            self.array_thickness = np.abs(
               0.25 *
               (pillar_thickness[:, pillar_for_col[:, :, 0, 0]] + pillar_thickness[:, pillar_for_col[:, :, 0, 1]] +
                pillar_thickness[:, pillar_for_col[:, :, 1, 0]] + pillar_thickness[:, pillar_for_col[:, :, 1, 1]]))
         else:
            self.array_thickness = np.abs(0.25 * (pillar_thickness[:, :-1, :-1] + pillar_thickness[:, :-1, 1:] +
                                                  pillar_thickness[:, 1:, :-1] + pillar_thickness[:, 1:, 1:]))
         if cell_kji0 is None:
            return self.array_thickness
         return self.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None

      cp = self.corner_points(cell_kji0 = cell_kji0,
                              points_root = points_root,
                              cache_resqml_array = cache_resqml_array,
                              cache_cp_array = cache_cp_array)
      if cp is None:
         return None
      return abs(np.mean(cp[1, :, :, 2]) - np.mean(cp[0, :, :, 2]))

   def point_areally(self, tolerance = 0.001):
      """Returns a numpy boolean array of shape extent_kji indicating which cells are reduced to a point in both I & J axes.

      Note:
         Any NaN point values will yield True for a cell
      """

      points = self.points_ref(masked = False)
      # todo: turn off NaN warning for numpy > ?
      if self.has_split_coordinate_lines:
         pillar_for_col = self.create_column_pillar_mapping()
         j_pair_vectors = points[:, pillar_for_col[:, :, 1, :], :] - points[:, pillar_for_col[:, :, 0, :], :]
         i_pair_vectors = points[:, pillar_for_col[:, :, :, 1], :] - points[:, pillar_for_col[:, :, :, 0], :]
         j_pair_nans = np.isnan(j_pair_vectors)
         i_pair_nans = np.isnan(i_pair_vectors)
         any_nans = np.any(np.logical_or(j_pair_nans, i_pair_nans), axis = (3, 4))
         j_pair_extant = np.any(np.abs(j_pair_vectors) > tolerance, axis = -1)
         i_pair_extant = np.any(np.abs(i_pair_vectors) > tolerance, axis = -1)
         any_extant = np.any(np.logical_or(j_pair_extant, i_pair_extant), axis = 3)
      else:
         j_vectors = points[:, 1:, :, :] - points[:, :-1, :, :]
         i_vectors = points[:, :, 1:, :] - points[:, :, :-1, :]
         j_nans = np.any(np.isnan(j_vectors), axis = -1)
         i_nans = np.any(np.isnan(i_vectors), axis = -1)
         j_pair_nans = np.logical_or(j_nans[:, :, :-1], j_nans[:, :, 1:])
         i_pair_nans = np.logical_or(i_nans[:, :-1, :], i_nans[:, 1:, :])
         any_nans = np.logical_or(j_pair_nans, i_pair_nans)
         j_extant = np.any(np.abs(j_vectors) > tolerance, axis = -1)
         i_extant = np.any(np.abs(i_vectors) > tolerance, axis = -1)
         j_pair_extant = np.logical_or(j_extant[:, :, :-1], j_extant[:, :, 1:])
         i_pair_extant = np.logical_or(i_extant[:, :-1, :], i_extant[:, 1:, :])
         any_extant = np.logical_or(j_pair_extant, i_pair_extant)
      layered = np.logical_or(any_nans, np.logical_not(any_extant))
      if self.k_gaps:
         return np.logical_and(layered[self.k_raw_index_array], layered[self.k_raw_index_array + 1])
      return np.logical_and(layered[:-1], layered[1:])

   def volume(self,
              cell_kji0 = None,
              points_root = None,
              cache_resqml_array = True,
              cache_cp_array = False,
              cache_centre_array = False,
              cache_volume_array = True,
              property_collection = None):
      """Returns bulk rock volume of cell or numpy array of bulk rock volumes for all cells.

      arguments:
         cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
                               volume is required; zero based indexing
         cache_resqml_array (boolean, default True): If True, the raw points array from the hdf5 file
                               is cached in memory, but only if it is needed to generate the volume
         cache_cp_array (boolean, default False): If True, an array of corner points is generated and
                               added as an attribute of the grid, with attribute name corner_points, but only
                               if it is needed in order to generate the volume
         cache_volume_array (boolean, default False): if True, volumes are generated for all cells in
                               the grid and added as an attribute named array_volume
         property_collection (property:GridPropertyCollection, optional): If not None, this collection
                               is probed for a suitable volume property which is used preferentially
                               to calculating volume; if no suitable property is found,
                               the calculation is made as if the collection were None

      returns:
         float, being the volume of cell identified by cell_kji0;
         or numpy float array of shape (nk, nj, ni) if cell_kji0 is None

      notes:
         the function can be used to find the volume of a single cell, or cache volumes for all cells, or both;
         if property_collection is not None, a suitable volume property will be used if present;
         if calculated, volume is computed using 6 tetras each with a non-planar bilinear base face;
         at present, grid's coordinate reference system must use same units in z as xy (projected);
         units of result are implicitly those of coordinates in grid's coordinate reference system, or units of
         measure of property array if the result is based on a suitable property

      :meta common:
      """

      def load_from_property(collection):
         if collection is None:
            return None
         parts = collection.selective_parts_list(property_kind = 'rock volume',
                                                 facet_type = 'netgross',
                                                 facet = 'gross')
         if len(parts) == 1:
            return collection.cached_part_array_ref(parts[0])
         parts = collection.selective_parts_list(property_kind = 'rock volume')
         if len(parts) == 1 and collection.facet_for_part(parts[0]) is None:
            return collection.cached_part_array_ref(parts[0])
         return None

      # note: this function optionally looks for a suitable volume property, otherwise calculates from geometry
      # todo: modify z units if needed, to match xy units
      # todo: give control over source with optional arguments
      # todo: cache uom
      assert (cache_volume_array is not None) or (cell_kji0 is not None)

      if hasattr(self, 'array_volume'):
         if cell_kji0 is None:
            return self.array_volume
         return self.array_volume[tuple(cell_kji0)]  # could check for nan here and return None

      vol_array = load_from_property(property_collection)
      if vol_array is not None:
         if cache_volume_array:
            self.array_volume = vol_array.copy()
         if cell_kji0 is None:
            return vol_array
         return vol_array[tuple(cell_kji0)]  # could check for nan here and return None

      cache_cp_array = cache_cp_array or cell_kji0 is None
      cache_volume_array = cache_volume_array or cell_kji0 is None

      off_hand = self.off_handed()

      points_root = self.resolve_geometry_child('Points', child_node = points_root)
      if points_root is None:
         return None  # geometry not present
      centre_array = None
      if cache_volume_array or cell_kji0 is None:
         self.corner_points(points_root = points_root, cache_cp_array = True)
         if cache_centre_array:
            self.centre_point(cache_centre_array = True)
            centre_array = self.array_centre_point
         vol_array = vol.tetra_volumes(self.array_corner_points, centres = centre_array, off_hand = off_hand)
         if cache_volume_array:
            self.array_volume = vol_array
         if cell_kji0 is None:
            return vol_array
         return vol_array[tuple(cell_kji0)]  # could check for nan here and return None

      cp = self.corner_points(cell_kji0 = cell_kji0,
                              points_root = points_root,
                              cache_resqml_array = cache_resqml_array,
                              cache_cp_array = cache_cp_array)
      if cp is None:
         return None
      return vol.tetra_cell_volume(cp, off_hand = off_hand)

   def pinched_out(self,
                   cell_kji0 = None,
                   tolerance = 0.001,
                   points_root = None,
                   cache_resqml_array = True,
                   cache_cp_array = False,
                   cache_thickness_array = False,
                   cache_pinchout_array = None):
      """Returns boolean or boolean array indicating whether cell is pinched out; ie. has a thickness less than tolerance.

      :meta common:
      """

      # note: this function returns a derived object rather than a native resqml object
      # note: returns True for cells without geometry
      # todo: check behaviour in response to NaNs and undefined geometry
      if cache_pinchout_array is None:
         cache_pinchout_array = (cell_kji0 is None)

      if self.pinchout is not None:
         if cell_kji0 is None:
            return self.pinchout
         return self.pinchout[tuple(cell_kji0)]

      if points_root is None:
         points_root = self.resolve_geometry_child('Points', child_node = points_root)
#         if points_root is None: return None  # geometry not present

      thick = self.thickness(
         cell_kji0,
         points_root = points_root,
         cache_resqml_array = cache_resqml_array,
         cache_cp_array = cache_cp_array,  # deprecated
         cache_thickness_array = cache_thickness_array or cache_pinchout_array)
      if cache_pinchout_array:
         self.pinchout = np.where(np.isnan(self.array_thickness), True,
                                  np.logical_not(self.array_thickness > tolerance))
         if cell_kji0 is None:
            return self.pinchout
         return self.pinchout[tuple(cell_kji0)]
      if thick is not None:
         return thick <= tolerance
      return None

   def half_cell_transmissibility(self, use_property = True, realization = None, tolerance = 1.0e-6):
      """Returns (and caches if realization is None) half cell transmissibilities for this grid.

      arguments:
         use_property (boolean, default True): if True, the grid's property collection is inspected for
            a possible half cell transmissibility array and if found, it is used instead of calculation
         realization (int, optional) if present, only a property with this realization number will be used
         tolerance (float, default 1.0e-6): minimum half axis length below which the transmissibility
            will be deemed uncomputable (for the axis in question); NaN values will be returned (not Inf);
            units are implicitly those of the grid's crs length units

      returns:
         numpy float array of shape (nk, nj, ni, 3, 2) where the 3 covers K,J,I and the 2 covers the
            face polarity: - (0) and + (1); units will depend on the length units of the coordinate reference
            system for the grid; the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m
            and ft respectively

      notes:
         the returned array is in the logical resqpy arrangement; it must be discombobulated before being
         added as a property; this method does not write to hdf5, nor create a new property or xml;
         if realization is None, a grid attribute cached array will be used; tolerance will only be
         used if the half cell transmissibilities are actually computed
      """

      # todo: allow passing of property uuids for ntg, k_k, j, i

      if realization is None and hasattr(self, 'array_half_cell_t'):
         return self.array_half_cell_t

      half_t = None

      if use_property:
         pc = self.property_collection
         half_t_resqml = pc.single_array_ref(property_kind = 'transmissibility',
                                             realization = realization,
                                             continuous = True,
                                             count = 1,
                                             indexable = 'faces per cell')
         if half_t_resqml:
            assert half_t_resqml.shape == (self.nk, self.nj, self.ni, 6)
            half_t = pc.combobulate(half_t_resqml)

      if half_t is None:
         half_t = rqtr.half_cell_t(
            self, realization = realization)  # note: properties must be identifiable in property_collection

      if realization is None:
         self.array_half_cell_t = half_t

      return half_t

   def transmissibility(self, tolerance = 1.0e-6, use_tr_properties = True, realization = None, modifier_mode = None):
      """Returns transmissibilities for standard (IJK neighbouring) connections within this grid.

      arguments:
         tolerance (float, default 1.0e-6): the minimum half cell transmissibility below which zero inter-cell
            transmissibility will be set; units are as for returned values (see notes)
         use_tr_properties (boolean, default True): if True, the grid's property collection is inspected for
            possible transmissibility arrays and if found, they are used instead of calculation; note that
            when this argument is False, the property collection is still used for the feed arrays to the
            calculation
         realization (int, optional) if present, only properties with this realization number will be used;
            applies to pre-computed transmissibility properties or permeability and net to gross ratio
            properties when computing
         modifier_mode (string, optional): if None, no transmissibility modifiers are applied; other
            options are: 'faces multiplier', for which directional transmissibility properties with indexable
            element of 'faces' will be used; 'faces per cell multiplier', in which case a transmissibility
            property with 'faces per cell' as the indexable element will be used to modify the half cell
            transmissibilities prior to combination; or 'absolute' in which case directional properties
            of local property kind 'fault transmissibility' (or 'mat transmissibility') and indexable
            element of 'faces' will be used as a third transmissibility term along with the two half
            cell transmissibilities at each face; see also the notes below

      returns:
         3 numpy float arrays of shape (nk + 1, nj, ni), (nk, nj + 1, ni), (nk, nj, ni + 1) being the
         neighbourly transmissibilities in K, J & I axes respectively

      notes:
         the 3 permeability arrays (and net to gross ratio if in use) must be identifiable in the property
         collection as they are used for the calculation;
         implicit units of measure of returned values will be m3.cP/(kPa.d) if grid crs length units are metres,
         bbl.cP/(psi.d) if length units are feet; the computation is compatible with the Nexus NEWTRAN formulation;
         values will be zero at pinchouts, and at column edges where there is a split pillar, even if there is
         juxtapostion of faces; the same is true of K gap faces (even where the gap is zero); NaNs in any of
         the feed properties also result in transmissibility values of zero;
         outer facing values will always be zero (included to be in accordance with RESQML faces properties);
         array caching in the grid object will only be used if realization is None; if a modifier mode of
         'faces multiplier' or 'faces per cell multiplier' is specified, properties will be searched for with
         local property kind 'transmissibility multiplier' and the appropriate indexable element (and direction
         facet in the case of 'faces multiplier'); the modifier mode of 'absolute' can be used to model the
         effect of faults and thin shales, tar mats etc. in a way which is independent of cell size;
         for 'aboslute' directional properties with indexable element of 'faces' and local property kind
         'fault transmissibility' (or 'mat transmissibility') will be used; such absolute faces transmissibilities
         should have a value of np.inf or np.nan where no modification is required; note that this method is only
         dealing with logically neighbouring cells and will not compute values for faces with a split pillar,
         which should be handled elsewhere
      """

      # todo: improve handling of units: check uom for half cell transmissibility property and for absolute modifiers

      k_tr = j_tr = i_tr = None

      if realization is None:
         if hasattr(self, 'array_k_transmissibility') and self.array_k_transmissibility is not None:
            k_tr = self.array_k_transmissibility
         if hasattr(self, 'array_j_transmissibility') and self.array_j_transmissibility is not None:
            j_tr = self.array_j_transmissibility
         if hasattr(self, 'array_i_transmissibility') and self.array_i_transmissibility is not None:
            i_tr = self.array_i_transmissibility

      if use_tr_properties and (k_tr is None or j_tr is None or i_tr is None):

         pc = self.extract_property_collection()

         if k_tr is None:
            k_tr = pc.single_array_ref(property_kind = 'transmissibility',
                                       realization = realization,
                                       continuous = True,
                                       count = 1,
                                       indexable = 'faces',
                                       facet_type = 'direction',
                                       facet = 'K')
            if k_tr is not None:
               assert k_tr.shape == (self.nk + 1, self.nj, self.ni)
               if realization is None:
                  self.array_k_transmissibility = k_tr

         if j_tr is None:
            j_tr = pc.single_array_ref(property_kind = 'transmissibility',
                                       realization = realization,
                                       continuous = True,
                                       count = 1,
                                       indexable = 'faces',
                                       facet_type = 'direction',
                                       facet = 'J')
            if j_tr is not None:
               assert j_tr.shape == (self.nk, self.nj + 1, self.ni)
               if realization is None:
                  self.array_j_transmissibility = j_tr

         if i_tr is None:
            i_tr = pc.single_array_ref(property_kind = 'transmissibility',
                                       realization = realization,
                                       continuous = True,
                                       count = 1,
                                       indexable = 'faces',
                                       facet_type = 'direction',
                                       facet = 'I')
            if i_tr is not None:
               assert i_tr.shape == (self.nk, self.nj, self.ni + 1)
               if realization is None:
                  self.array_i_transmissibility = i_tr

      if k_tr is None or j_tr is None or i_tr is None:

         half_t = self.half_cell_transmissibility(use_property = use_tr_properties, realization = realization)

         if modifier_mode == 'faces per cell multiplier':
            pc = self.extract_property_collection()
            half_t_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                              realization = realization,
                                              continuous = True,
                                              count = 1,
                                              indexable = 'faces per cell')
            if half_t_mult is None:
               log.warning('no faces per cell transmissibility multiplier found when calculating transmissibilities')
            else:
               log.debug('applying faces per cell transmissibility multipliers')
               half_t = np.where(np.isnan(half_t_mult), half_t, half_t * half_t_mult)

         if self.has_split_coordinate_lines and (j_tr is None or i_tr is None):
            split_column_edges_j, split_column_edges_i = self.split_column_faces()
         else:
            split_column_edges_j, split_column_edges_i = None, None

         np.seterr(divide = 'ignore')

         if k_tr is None:
            k_tr = np.zeros((self.nk + 1, self.nj, self.ni))
            slice_a = half_t[:-1, :, :, 0, 1]  # note: internal faces only
            slice_b = half_t[1:, :, :, 0, 0]
            internal_zero_mask = np.logical_or(np.logical_or(np.isnan(slice_a), slice_a < tolerance),
                                               np.logical_or(np.isnan(slice_b), slice_b < tolerance))
            if self.k_gaps:  # todo: scan K gaps for zero thickness gaps and allow transmission there
               internal_zero_mask[self.k_gap_after_array, :, :] = True
            tr_mult = None
            if modifier_mode == 'faces multiplier':
               tr_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                             realization = realization,
                                             facet_type = 'direction',
                                             facet = 'K',
                                             continuous = True,
                                             count = 1,
                                             indexable = 'faces')
               if tr_mult is not None:
                  assert tr_mult.shape == (self.nk + 1, self.nj, self.ni)
                  internal_zero_mask = np.logical_or(internal_zero_mask, np.isnan(tr_mult[1:-1, :, :]))
            if tr_mult is None:
               tr_mult = 1.0
            tr_abs_r = 0.0
            if modifier_mode == 'absolute':
               tr_abs = pc.single_array_ref(property_kind = 'mat transmissibility',
                                            realization = realization,
                                            facet_type = 'direction',
                                            facet = 'K',
                                            continuous = True,
                                            count = 1,
                                            indexable = 'faces')
               if tr_abs is None:
                  tr_abs = pc.single_array_ref(property_kind = 'mat transmissibility',
                                               realization = realization,
                                               continuous = True,
                                               count = 1,
                                               indexable = 'faces')
               if tr_abs is None:
                  tr_abs = pc.single_array_ref(property_kind = 'fault transmissibility',
                                               realization = realization,
                                               facet_type = 'direction',
                                               facet = 'K',
                                               continuous = True,
                                               count = 1,
                                               indexable = 'faces')
               if tr_abs is not None:
                  log.debug('applying absolute K face transmissibility modification')
                  assert tr_abs.shape == (self.nk + 1, self.nj, self.ni)
                  internal_zero_mask = np.logical_or(internal_zero_mask, tr_abs[1:-1, :, :] <= 0.0)
                  tr_abs_r = np.where(np.logical_or(np.isinf(tr_abs), np.isnan(tr_abs)), 0.0, 1.0 / tr_abs)
            k_tr[1:-1, :, :] = np.where(internal_zero_mask, 0.0,
                                        tr_mult / ((1.0 / slice_a) + tr_abs_r + (1.0 / slice_b)))
            if realization is None:
               self.array_k_transmissibility = k_tr

         if j_tr is None:
            j_tr = np.zeros((self.nk, self.nj + 1, self.ni))
            slice_a = half_t[:, :-1, :, 1, 1]  # note: internal faces only
            slice_b = half_t[:, 1:, :, 1, 0]
            internal_zero_mask = np.logical_or(np.logical_or(np.isnan(slice_a), slice_a < tolerance),
                                               np.logical_or(np.isnan(slice_b), slice_b < tolerance))
            if split_column_edges_j is not None:
               internal_zero_mask[:, split_column_edges_j] = True
            tr_mult = None
            if modifier_mode == 'faces multiplier':
               tr_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                             realization = realization,
                                             facet_type = 'direction',
                                             facet = 'J',
                                             continuous = True,
                                             count = 1,
                                             indexable = 'faces')
               if tr_mult is None:
                  log.warning(
                     'no J direction faces transmissibility multiplier found when calculating transmissibilities')
               else:
                  assert tr_mult.shape == (self.nk, self.nj + 1, self.ni)
                  internal_zero_mask = np.logical_or(internal_zero_mask, np.isnan(tr_mult[:, 1:-1, :]))
            if tr_mult is None:
               tr_mult = 1.0
            tr_abs_r = 0.0
            if modifier_mode == 'absolute':
               tr_abs = pc.single_array_ref(property_kind = 'fault transmissibility',
                                            realization = realization,
                                            facet_type = 'direction',
                                            facet = 'J',
                                            continuous = True,
                                            count = 1,
                                            indexable = 'faces')
               if tr_abs is not None:
                  log.debug('applying absolute J face transmissibility modification')
                  assert tr_abs.shape == (self.nk, self.nj + 1, self.ni)
                  internal_zero_mask = np.logical_or(internal_zero_mask, tr_abs[:, 1:-1, :] <= 0.0)
                  tr_abs_r = np.where(np.logical_or(np.isinf(tr_abs), np.isnan(tr_abs)), 0.0, 1.0 / tr_abs)
            j_tr[:, 1:-1, :] = np.where(internal_zero_mask, 0.0,
                                        tr_mult / ((1.0 / slice_a) + tr_abs_r + (1.0 / slice_b)))
            if realization is None:
               self.array_j_transmissibility = j_tr

         if i_tr is None:
            i_tr = np.zeros((self.nk, self.nj, self.ni + 1))
            slice_a = half_t[:, :, :-1, 2, 1]  # note: internal faces only
            slice_b = half_t[:, :, 1:, 2, 0]
            internal_zero_mask = np.logical_or(np.logical_or(np.isnan(slice_a), slice_a < tolerance),
                                               np.logical_or(np.isnan(slice_b), slice_b < tolerance))
            if split_column_edges_i is not None:
               internal_zero_mask[:, split_column_edges_i] = True
            tr_mult = None
            if modifier_mode == 'faces multiplier':
               tr_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                             realization = realization,
                                             facet_type = 'direction',
                                             facet = 'I',
                                             continuous = True,
                                             count = 1,
                                             indexable = 'faces')
               if tr_mult is None:
                  log.warning(
                     'no I direction faces transmissibility multiplier found when calculating transmissibilities')
               else:
                  assert tr_mult.shape == (self.nk, self.nj, self.ni + 1)
                  internal_zero_mask = np.logical_or(internal_zero_mask, np.isnan(tr_mult[:, :, 1:-1]))
            if tr_mult is None:
               tr_mult = 1.0
            tr_abs_r = 0.0
            if modifier_mode == 'absolute':
               tr_abs = pc.single_array_ref(property_kind = 'fault transmissibility',
                                            realization = realization,
                                            facet_type = 'direction',
                                            facet = 'I',
                                            continuous = True,
                                            count = 1,
                                            indexable = 'faces')
               if tr_abs is not None:
                  log.debug('applying absolute I face transmissibility modification')
                  assert tr_abs.shape == (self.nk, self.nj, self.ni + 1)
                  internal_zero_mask = np.logical_or(internal_zero_mask, tr_abs[:, :, 1:-1] <= 0.0)
                  tr_abs_r = np.where(np.logical_or(np.isinf(tr_abs), np.isnan(tr_abs)), 0.0, 1.0 / tr_abs)
            i_tr[:, :, 1:-1] = np.where(internal_zero_mask, 0.0,
                                        tr_mult / ((1.0 / slice_a) + tr_abs_r + (1.0 / slice_b)))

         np.seterr(divide = 'warn')

      return k_tr, j_tr, i_tr

   def fault_connection_set(self,
                            skip_inactive = True,
                            compute_transmissibility = False,
                            add_to_model = False,
                            realization = None,
                            inherit_features_from = None,
                            title = 'fault juxtaposition set'):
      """Returns (and caches) a GridConnectionSet representing juxtaposition across faces with split pillars.

      arguments:
         skip_inactive (boolean, default True): if True, then cell face pairs involving an inactive cell will
            be omitted from the results
         compute_transmissibilities (boolean, default False): if True, then transmissibilities will be computed
            for the cell face pairs (unless already existing as a cached attribute of the grid)
         add_to_model (boolean, default False): if True, the connection set is written to hdf5 and xml is created;
            if compute_transmissibilty is True then the transmissibility property is also added
         realization (int, optional): if present, is used as the realization number when adding transmissibility
            property to model; ignored if compute_transmissibility is False
         inherit_features_from (GridConnectionSet, optional): if present, the features (named faults) are
            inherited from this grid connection set based on a match of either cell face in a juxtaposed pair
         title (string, default 'fault juxtaposition set'): the citation title to use if adding to model

      returns:
         GridConnectionSet, numpy float array of shape (count,) transmissibilities (or None), where count is the
         number of cell face pairs in the grid connection set, which contains entries for all juxtaposed faces
         with a split pillar as an edge; if the grid does not have split pillars (ie. is unfaulted) or there
         are no qualifying connections, (None, None) is returned
      """

      if not hasattr(self, 'fgcs') or self.fgcs_skip_inactive != skip_inactive:
         self.fgcs, self.fgcs_fractional_area = rqtr.fault_connection_set(self, skip_inactive = skip_inactive)
         self.fgcs_skip_inactive = skip_inactive

      if self.fgcs is None:
         return None, None

      new_tr = False
      if compute_transmissibility and not hasattr(self, 'array_fgcs_transmissibility'):
         self.array_fgcs_transmissibility = self.fgcs.tr_property_array(self.fgcs_fractional_area)
         new_tr = True

      tr = self.array_fgcs_transmissibility if hasattr(self, 'array_fgcs_transmissibility') else None

      if inherit_features_from is not None:
         self.fgcs.inherit_features(inherit_features_from)

      if add_to_model:
         if self.model.uuid(uuid = self.fgcs.uuid) is None:
            self.fgcs.write_hdf5()
            self.fgcs.create_xml(title = title)
         if new_tr:
            tr_pc = rprop.PropertyCollection()
            tr_pc.set_support(support = self.fgcs)
            tr_pc.add_cached_array_to_imported_list(
               self.array_fgcs_transmissibility,
               'computed for faces with split pillars',
               'fault transmissibility',
               discrete = False,
               uom = 'm3.cP/(kPa.d)' if self.xy_units() == 'm' else 'bbl.cP/(psi.d)',
               property_kind = 'transmissibility',
               realization = realization,
               indexable_element = 'faces',
               count = 1)
            tr_pc.write_hdf5_for_imported_list()
            tr_pc.create_xml_for_imported_list_and_add_parts_to_model()

      return self.fgcs, tr

   def pinchout_connection_set(self,
                               skip_inactive = True,
                               compute_transmissibility = False,
                               add_to_model = False,
                               realization = None):
      """Returns (and caches) a GridConnectionSet representing juxtaposition across pinched out cells.

      arguments:
         skip_inactive (boolean, default True): if True, then cell face pairs involving an inactive cell will
            be omitted from the results
         compute_transmissibilities (boolean, default False): if True, then transmissibilities will be computed
            for the cell face pairs (unless already existing as a cached attribute of the grid)
         add_to_model (boolean, default False): if True, the connection set is written to hdf5 and xml is created;
            if compute_transmissibilty is True then the transmissibility property is also added
         realization (int, optional): if present, is used as the realization number when adding transmissibility
            property to model; ignored if compute_transmissibility is False

      returns:
         GridConnectionSet, numpy float array of shape (count,) transmissibilities (or None), where count is the
         number of cell face pairs in the grid connection set, which contains entries for all juxtaposed K faces
         separated logically by pinched out (zero thickness) cells; if there are no pinchouts (or no qualifying
         connections) then (None, None) will be returned
      """

      if not hasattr(self, 'pgcs') or self.pgcs_skip_inactive != skip_inactive:
         self.pgcs = rqf.pinchout_connection_set(self, skip_inactive = skip_inactive)
         self.pgcs_skip_inactive = skip_inactive

      if self.pgcs is None:
         return None, None

      new_tr = False
      if compute_transmissibility and not hasattr(self, 'array_pgcs_transmissibility'):
         self.array_pgcs_transmissibility = self.pgcs.tr_property_array()
         new_tr = True

      tr = self.array_pgcs_transmissibility if hasattr(self, 'array_pgcs_transmissibility') else None

      if add_to_model:
         if self.model.uuid(uuid = self.pgcs.uuid) is None:
            self.pgcs.write_hdf5()
            self.pgcs.create_xml()
         if new_tr:
            tr_pc = rprop.PropertyCollection()
            tr_pc.set_support(support = self.pgcs)
            tr_pc.add_cached_array_to_imported_list(
               tr,
               'computed for faces across pinchouts',
               'pinchout transmissibility',
               discrete = False,
               uom = 'm3.cP/(kPa.d)' if self.xy_units() == 'm' else 'bbl.cP/(psi.d)',
               property_kind = 'transmissibility',
               realization = realization,
               indexable_element = 'faces',
               count = 1)
            tr_pc.write_hdf5_for_imported_list()
            tr_pc.create_xml_for_imported_list_and_add_parts_to_model()

      return self.pgcs, tr

   def k_gap_connection_set(self,
                            skip_inactive = True,
                            compute_transmissibility = False,
                            add_to_model = False,
                            realization = None,
                            tolerance = 0.001):
      """Returns (and caches) a GridConnectionSet representing juxtaposition across zero thickness K gaps.

      arguments:
         skip_inactive (boolean, default True): if True, then cell face pairs involving an inactive cell will
            be omitted from the results
         compute_transmissibilities (boolean, default False): if True, then transmissibilities will be computed
            for the cell face pairs (unless already existing as a cached attribute of the grid)
         add_to_model (boolean, default False): if True, the connection set is written to hdf5 and xml is created;
            if compute_transmissibilty is True then the transmissibility property is also added
         realization (int, optional): if present, is used as the realization number when adding transmissibility
            property to model; ignored if compute_transmissibility is False
         tolerance (float, default 0.001): the maximum K gap thickness that will be 'bridged' by a connection;
            units are implicitly those of the z units in the grid's coordinate reference system

      returns:
         GridConnectionSet, numpy float array of shape (count,) transmissibilities (or None), where count is the
         number of cell face pairs in the grid connection set, which contains entries for all juxtaposed K faces
         separated logically by pinched out (zero thickness) cells; if there are no pinchouts (or no qualifying
         connections) then (None, None) will be returned

      note:
         if cached values are found they are returned regardless of the specified tolerance
      """

      if not hasattr(self, 'kgcs') or self.kgcs_skip_inactive != skip_inactive:
         self.kgcs = rqf.k_gap_connection_set(self, skip_inactive = skip_inactive, tolerance = tolerance)
         self.kgcs_skip_inactive = skip_inactive

      if self.kgcs is None:
         return None, None

      new_tr = False
      if compute_transmissibility and not hasattr(self, 'array_kgcs_transmissibility'):
         self.array_kgcs_transmissibility = self.kgcs.tr_property_array()
         new_tr = True

      tr = self.array_kgcs_transmissibility if hasattr(self, 'array_kgcs_transmissibility') else None

      if add_to_model:
         if self.model.uuid(uuid = self.kgcs.uuid) is None:
            self.kgcs.write_hdf5()
            self.kgcs.create_xml()
         if new_tr:
            tr_pc = rprop.PropertyCollection()
            tr_pc.set_support(support = self.kgcs)
            tr_pc.add_cached_array_to_imported_list(
               tr,
               'computed for faces across zero thickness K gaps',
               'K gap transmissibility',
               discrete = False,
               uom = 'm3.cP/(kPa.d)' if self.xy_units() == 'm' else 'bbl.cP/(psi.d)',
               property_kind = 'transmissibility',
               realization = realization,
               indexable_element = 'faces',
               count = 1)
            tr_pc.write_hdf5_for_imported_list()
            tr_pc.create_xml_for_imported_list_and_add_parts_to_model()

      return self.kgcs, tr

   def cell_inactive(self, cell_kji0, pv_array = None, pv_tol = 0.01):
      """Returns True if the cell is inactive."""

      if self.inactive is not None:
         return self.inactive[tuple(cell_kji0)]
      self.extract_inactive_mask()
      if self.inactive is not None:
         return self.inactive[tuple(cell_kji0)]
      if pv_array is not None:  # fabricate an inactive mask from pore volume data
         self.inactive = not (pv_array > pv_tol)  # NaN in pv array will end up inactive
         return self.inactive[tuple(cell_kji0)]
      return (not self.cell_geometry_is_defined(cell_kji0)) or self.pinched_out(cell_kji0, cache_pinchout_array = True)

   def bounding_box(self, cell_kji0, points_root = None, cache_cp_array = False):
      """Returns the xyz box which envelopes the specified cell, as a numpy array of shape (2, 3)."""

      result = np.zeros((2, 3))
      cp = self.corner_points(cell_kji0, points_root = points_root, cache_cp_array = cache_cp_array)
      result[0] = np.min(cp, axis = (0, 1, 2))
      result[1] = np.max(cp, axis = (0, 1, 2))
      return result

   def composite_bounding_box(self, bounding_box_list):
      """Returns the xyz box which envelopes all the boxes in the list, as a numpy array of shape (2, 3)."""

      result = bounding_box_list[0]
      for box in bounding_box_list[1:]:
         result[0] = np.minimum(result[0], box[0])
         result[1] = np.maximum(result[1], box[1])
      return result

   def interpolated_point(self,
                          cell_kji0,
                          interpolation_fraction,
                          points_root = None,
                          cache_resqml_array = True,
                          cache_cp_array = False):
      """Returns xyz point interpolated from corners of cell depending on 3 interpolation fractions in range 0 to 1."""

      # todo: think about best ordering of axes operations given high aspect ratio of cells (for best accuracy)
      fp = np.empty(3)
      fm = np.empty(3)
      for axis in range(3):
         fp[axis] = max(min(interpolation_fraction[axis], 1.0), 0.0)
         fm[axis] = 1.0 - fp[axis]
      cp = self.corner_points(cell_kji0,
                              points_root = points_root,
                              cache_resqml_array = cache_resqml_array,
                              cache_cp_array = cache_cp_array)
      c00 = (cp[0, 0, 0] * fm[0] + cp[1, 0, 0] * fp[0])
      c01 = (cp[0, 0, 1] * fm[0] + cp[1, 0, 1] * fp[0])
      c10 = (cp[0, 1, 0] * fm[0] + cp[1, 1, 0] * fp[0])
      c11 = (cp[0, 1, 1] * fm[0] + cp[1, 1, 1] * fp[0])
      c0 = c00 * fm[1] + c10 * fp[1]
      c1 = c01 * fm[1] + c11 * fp[1]
      c = c0 * fm[2] + c1 * fp[2]
      return c

   def interpolated_points(self,
                           cell_kji0,
                           interpolation_fractions,
                           points_root = None,
                           cache_resqml_array = True,
                           cache_cp_array = False):
      """Returns xyz points interpolated from corners of cell depending on 3 interpolation fraction numpy vectors, each value in range 0 to 1.

      arguments:
         cell_kji0 (triple int): indices of individual cell whose corner points are to be interpolated
         interpolation_fractions (list of three numpy vectors of floats): k, j & i interpolation fraction vectors, each element in range 0 to 1
         points_root (xml node, optional): for efficiency when making multiple calls, this can be set to the xml node of the points data
         cache_resqml_array (boolean, default True): if True, the resqml points data will be cached as an attribute of this grid object
         cache_cp_array (boolean, default False): if True a fully expanded 7D corner points array will be established for this grid and
            cached as an attribute (recommended if looping over many or all the cells and if memory space allows)

      returns:
         4D numpy float array of shape (nik, nij, nii, 3) being the interpolated points; nik is the number of elements in the first of the
         interpolation fraction lists (ie. for k); similarly for nij and nii; the final axis covers xyz

      notea:
         this method returns a lattice of trilinear interpolations of the corner point of the host cell; the returned points are in 'shared'
         arrangement (like resqml points data for an IjkGrid without split pillars or k gaps), not a fully expanded 7D array; calling code
         must redistribute to corner points of individual fine cells if that is the intention
      """

      assert len(interpolation_fractions) == 3
      fp = interpolation_fractions
      fm = []
      for axis in range(3):
         fm.append(1.0 - fp[axis])

      cp = self.corner_points(cell_kji0,
                              points_root = points_root,
                              cache_resqml_array = cache_resqml_array,
                              cache_cp_array = cache_cp_array)

      c00 = (np.outer(fm[2], cp[0, 0, 0]) + np.outer(fp[2], cp[0, 0, 1]))
      c01 = (np.outer(fm[2], cp[0, 1, 0]) + np.outer(fp[2], cp[0, 1, 1]))
      c10 = (np.outer(fm[2], cp[1, 0, 0]) + np.outer(fp[2], cp[1, 0, 1]))
      c11 = (np.outer(fm[2], cp[1, 1, 0]) + np.outer(fp[2], cp[1, 1, 1]))
      c0 = (np.multiply.outer(fm[1], c00) + np.multiply.outer(fp[1], c01))
      c1 = (np.multiply.outer(fm[1], c10) + np.multiply.outer(fp[1], c11))
      c = (np.multiply.outer(fm[0], c0) + np.multiply.outer(fp[0], c1))

      return c

   def face_centre(self,
                   cell_kji0,
                   axis,
                   zero_or_one,
                   points_root = None,
                   cache_resqml_array = True,
                   cache_cp_array = False):
      """Returns xyz location of the centre point of a face of the cell (or all cells)."""

      # todo: optionally compute for all cells and cache
      cp = self.corner_points(cell_kji0,
                              points_root = points_root,
                              cache_resqml_array = cache_resqml_array,
                              cache_cp_array = cache_cp_array)
      if cell_kji0 is None:
         if axis == 0:
            return 0.25 * np.sum(cp[:, :, :, zero_or_one, :, :], axis = (3, 4))
         elif axis == 1:
            return 0.25 * np.sum(cp[:, :, :, :, zero_or_one, :], axis = (3, 4))
         else:
            return 0.25 * np.sum(cp[:, :, :, :, :, zero_or_one], axis = (3, 4))
      else:
         if axis == 0:
            return 0.25 * np.sum(cp[zero_or_one, :, :], axis = (0, 1))
         elif axis == 1:
            return 0.25 * np.sum(cp[:, zero_or_one, :], axis = (0, 1))
         else:
            return 0.25 * np.sum(cp[:, :, zero_or_one], axis = (0, 1))

   def face_centres_kji_01(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
      """Returns an array of shape (3, 2, 3) being (axis, 0 or 1, xyz) of face centre points for cell."""

      assert cell_kji0 is not None
      result = np.zeros((3, 2, 3))
      for axis in range(3):
         for zero_or_one in range(2):
            result[axis, zero_or_one] = self.face_centre(cell_kji0,
                                                         axis,
                                                         zero_or_one,
                                                         points_root = points_root,
                                                         cache_resqml_array = cache_resqml_array,
                                                         cache_cp_array = cache_cp_array)
      return result

   def interface_vector(self, cell_kji0, axis, points_root = None, cache_resqml_array = True, cache_cp_array = False):
      """Returns an xyz vector between centres of an opposite pair of faces of the cell (or vectors for all cells)."""

      face_0_centre = self.face_centre(cell_kji0,
                                       axis,
                                       0,
                                       points_root = points_root,
                                       cache_resqml_array = cache_resqml_array,
                                       cache_cp_array = cache_cp_array)
      face_1_centre = self.face_centre(cell_kji0,
                                       axis,
                                       1,
                                       points_root = points_root,
                                       cache_resqml_array = cache_resqml_array,
                                       cache_cp_array = cache_cp_array)
      return face_1_centre - face_0_centre

   def interface_length(self, cell_kji0, axis, points_root = None, cache_resqml_array = True, cache_cp_array = False):
      """Returns the length between centres of an opposite pair of faces of the cell.

      note:
         assumes that x,y and z units are the same
      """

      assert cell_kji0 is not None
      return vec.naive_length(
         self.interface_vector(cell_kji0,
                               axis,
                               points_root = points_root,
                               cache_resqml_array = cache_resqml_array,
                               cache_cp_array = cache_cp_array))

   def interface_vectors_kji(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
      """Returns 3 interface centre point difference vectors for axes k, j, i."""

      result = np.zeros((3, 3))
      for axis in range(3):
         result[axis] = self.interface_vector(cell_kji0,
                                              axis,
                                              points_root = points_root,
                                              cache_resqml_array = cache_resqml_array,
                                              cache_cp_array = cache_cp_array)
      return result

   def interface_lengths_kji(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
      """Returns 3 interface centre point separation lengths for axes k, j, i.

      note:
         assumes that x,y and z units are the same
      """
      result = np.zeros(3)
      for axis in range(3):
         result[axis] = self.interface_length(cell_kji0,
                                              axis,
                                              points_root = points_root,
                                              cache_resqml_array = cache_resqml_array,
                                              cache_cp_array = cache_cp_array)
      return result

   def local_to_global_crs(self,
                           a,
                           crs_root = None,
                           global_xy_units = None,
                           global_z_units = None,
                           global_z_increasing_downward = None):
      """Converts array of points in situ from local coordinate system to global one."""

      # todo: replace with crs module calls

      if crs_root is None:
         crs_root = self.crs_root
         if crs_root is None:
            return
      flat_a = a.reshape((-1, 3))  # flattened view of array a as vector of (x, y, z) points, in situ
      x_offset = float(rqet.find_tag(crs_root, 'XOffset').text)
      y_offset = float(rqet.find_tag(crs_root, 'YOffset').text)
      z_offset = float(rqet.find_tag(crs_root, 'ZOffset').text)
      areal_rotation = float(rqet.find_tag(crs_root, 'ArealRotation').text)
      assert (areal_rotation == 0.0)
      # todo: check resqml definition for order of rotation and translation
      # todo: apply rotation
      if global_z_increasing_downward is not None:  # note: here negation is made in local crs; if z_offset is not zero, this might not be what is intended
         crs_z_increasing_downward_text = rqet.find_tag(crs_root, 'ZIncreasingDownward').text
         if crs_z_increasing_downward_text in ['true', 'false']:  # todo: otherwise could raise exception
            crs_z_increasing_downward = (crs_z_increasing_downward_text == 'true')
            if global_z_increasing_downward != crs_z_increasing_downward:
               negated_z = np.negative(flat_a[:, 2])
               flat_a[:, 2] = negated_z
      flat_a[:, 0] += x_offset
      flat_a[:, 1] += y_offset
      if z_offset != 0.0:
         flat_a[:, 2] += z_offset
      if global_xy_units is not None:
         crs_xy_units_text = rqet.find_tag(crs_root, 'ProjectedUom').text
         if crs_xy_units_text in ['ft', 'm']:  # todo: else raise exception
            bwam.convert_lengths(flat_a[:, 0], crs_xy_units_text, global_xy_units)  # x
            bwam.convert_lengths(flat_a[:, 1], crs_xy_units_text, global_xy_units)  # y
      if global_z_units is not None:
         crs_z_units_text = rqet.find_tag(crs_root, 'VerticalUom').text
         if crs_z_units_text in ['ft', 'm']:  # todo: else raise exception
            bwam.convert_lengths(flat_a[:, 2], crs_z_units_text, global_z_units)  # z

   def z_inc_down(self):
      """Returns True if z increases downwards in the coordinate reference system used by the grid geometry, False otherwise.

      :meta common:
      """

      assert self.crs_root is not None
      return rqet.find_tag_bool(self.crs_root, 'ZIncreasingDownward')

   def global_to_local_crs(self,
                           a,
                           crs_root = None,
                           global_xy_units = None,
                           global_z_units = None,
                           global_z_increasing_downward = None):
      """Converts array of points in situ from global coordinate system to established local one."""

      # todo: replace with crs module calls

      if crs_root is None:
         crs_root = self.crs_root
         if crs_root is None:
            return
      flat_a = a.reshape((-1, 3))  # flattened view of array a as vector of (x, y, z) points, in situ
      x_offset = float(rqet.find_tag(crs_root, 'XOffset').text)
      y_offset = float(rqet.find_tag(crs_root, 'YOffset').text)
      z_offset = float(rqet.find_tag(crs_root, 'ZOffset').text)
      areal_rotation = float(rqet.find_tag(crs_root, 'ArealRotation').text)
      assert (areal_rotation == 0.0)
      # todo: check resqml definition for order of rotation and translation and apply rotation if not zero
      if global_xy_units is not None:
         crs_xy_units_text = rqet.find_tag(crs_root, 'ProjectedUom').text
         if crs_xy_units_text in ['ft', 'm']:  # todo: else raise exception
            bwam.convert_lengths(flat_a[:, 0], global_xy_units, crs_xy_units_text)  # x
            bwam.convert_lengths(flat_a[:, 1], global_xy_units, crs_xy_units_text)  # y
      if global_z_units is not None:
         crs_z_units_text = rqet.find_tag(crs_root, 'VerticalUom').text
         if crs_z_units_text in ['ft', 'm']:  # todo: else raise exception
            bwam.convert_lengths(flat_a[:, 2], global_z_units, crs_z_units_text)  # z
      flat_a[:, 0] -= x_offset
      flat_a[:, 1] -= y_offset
      if z_offset != 0.0:
         flat_a[:, 2] -= z_offset
      if global_z_increasing_downward is not None:  # note: here negation is made in local crs; if z_offset is not zero, this might not be what is intended
         crs_z_increasing_downward = self.z_inc_down()
         assert crs_z_increasing_downward is not None
         if global_z_increasing_downward != crs_z_increasing_downward:
            negated_z = np.negative(flat_a[:, 2])
            flat_a[:, 2] = negated_z

   def write_hdf5_from_caches(self,
                              file = None,
                              mode = 'a',
                              geometry = True,
                              imported_properties = None,
                              write_active = None,
                              stratigraphy = True):
      """Create or append to an hdf5 file, writing datasets for the grid geometry (and parent grid mapping) and properties from cached arrays."""

      # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
      # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
      # xml is not created here for property objects

      if write_active is None:
         write_active = geometry

      self.cache_all_geometry_arrays()

      if not file:
         file = self.model.h5_file_name()
      h5_reg = rwh5.H5Register(self.model)

      if stratigraphy and self.stratigraphic_units is not None:
         h5_reg.register_dataset(self.uuid, 'unitIndices', self.stratigraphic_units, dtype = 'uint32')

      if geometry:
         if always_write_pillar_geometry_is_defined_array or not self.geometry_defined_for_all_pillars(
               cache_array = True):
            if not hasattr(self, 'array_pillar_geometry_is_defined') or self.array_pillar_geometry_is_defined is None:
               self.array_pillar_geometry_is_defined = np.full((self.nj + 1, self.ni + 1), True, dtype = bool)
            h5_reg.register_dataset(self.uuid,
                                    'PillarGeometryIsDefined',
                                    self.array_pillar_geometry_is_defined,
                                    dtype = 'uint8')
         if always_write_cell_geometry_is_defined_array or not self.geometry_defined_for_all_cells(cache_array = True):
            if not hasattr(self, 'array_cell_geometry_is_defined') or self.array_cell_geometry_is_defined is None:
               self.array_cell_geometry_is_defined = np.full((self.nk, self.nj, self.ni), True, dtype = bool)
            h5_reg.register_dataset(self.uuid,
                                    'CellGeometryIsDefined',
                                    self.array_cell_geometry_is_defined,
                                    dtype = 'uint8')
         # todo: PillarGeometryIsDefined ?
         h5_reg.register_dataset(self.uuid, 'Points', self.points_cached)
         if self.has_split_coordinate_lines:
            h5_reg.register_dataset(self.uuid, 'PillarIndices', self.split_pillar_indices_cached, dtype = 'uint32')
            h5_reg.register_dataset(self.uuid,
                                    'ColumnsPerSplitCoordinateLine/elements',
                                    self.cols_for_split_pillars,
                                    dtype = 'uint32')
            h5_reg.register_dataset(self.uuid,
                                    'ColumnsPerSplitCoordinateLine/cumulativeLength',
                                    self.cols_for_split_pillars_cl,
                                    dtype = 'uint32')
         if self.k_gaps:
            assert self.k_gap_after_array is not None
            h5_reg.register_dataset(self.uuid, 'GapAfterLayer', self.k_gap_after_array, dtype = 'uint8')
         if self.parent_window is not None:
            for axis in range(3):
               if self.parent_window.fine_extent_kji[axis] == self.parent_window.coarse_extent_kji[axis]:
                  continue  # one-to-noe mapping
               # reconstruct hdf5 arrays from FineCoarse object and register for write
               if self.parent_window.constant_ratios[axis] is not None:
                  if self.is_refinement:
                     pcpi = np.array([self.parent_window.coarse_extent_kji[axis]],
                                     dtype = int)  # ParentCountPerInterval
                     ccpi = np.array([self.parent_window.fine_extent_kji[axis]], dtype = int)  # ChildCountPerInterval
                  else:
                     pcpi = np.array([self.parent_window.fine_extent_kji[axis]], dtype = int)
                     ccpi = np.array([self.parent_window.coarse_extent_kji[axis]], dtype = int)
               else:
                  if self.is_refinement:
                     interval_count = self.parent_window.coarse_extent_kji[axis]
                     pcpi = np.ones(interval_count, dtype = int)
                     ccpi = np.array(self.parent_window.vector_ratios[axis], dtype = int)
                  else:
                     interval_count = self.parent_window.fine_extent_kji[axis]
                     pcpi = np.array(self.parent_window.vector_ratios[axis], dtype = int)
                     ccpi = np.ones(interval_count, dtype = int)
               h5_reg.register_dataset(self.uuid, 'KJI'[axis] + 'Regrid/ParentCountPerInterval', pcpi)
               h5_reg.register_dataset(self.uuid, 'KJI'[axis] + 'Regrid/ChildCountPerInterval', ccpi)
               if self.is_refinement and not self.parent_window.equal_proportions[axis]:
                  child_cell_weights = np.concatenate(self.parent_window.vector_proportions[axis])
                  h5_reg.register_dataset(self.uuid, 'KJI'[axis] + 'Regrid/ChildCellWeights', child_cell_weights)

      if write_active and self.inactive is not None:
         if imported_properties is None:
            imported_properties = rprop.PropertyCollection()
            imported_properties.set_support(support = self)
         else:
            filtered_list = []
            for entry in imported_properties.imported_list:
               if entry[2].upper() == 'ACTIVE' or entry[10] == 'active':
                  continue  # keyword or property kind
               filtered_list.append(entry)
            imported_properties.imported_list = filtered_list  # might have unintended side effects elsewhere
         active_mask = np.logical_not(self.inactive)
         imported_properties.add_cached_array_to_imported_list(active_mask,
                                                               'active cell mask',
                                                               'ACTIVE',
                                                               discrete = True,
                                                               property_kind = 'active')

      if imported_properties is not None and imported_properties.imported_list is not None:
         for entry in imported_properties.imported_list:
            if hasattr(imported_properties, entry[3]):  # otherwise constant array
               h5_reg.register_dataset(entry[0], 'values_patch0', imported_properties.__dict__[entry[3]])
            if entry[10] == 'active':
               self.active_property_uuid = entry[0]
      h5_reg.write(file, mode = mode)

   def write_hdf5(self):
      """Writes grid geometry arrays to hdf5 (thin wrapper around write_hdf5_from_caches().

      :meta common:
      """

      self.write_hdf5_from_caches(mode = 'a',
                                  geometry = True,
                                  imported_properties = None,
                                  write_active = True,
                                  stratigraphy = True)

   def off_handed(self):
      """Returns False if IJK and xyz have same handedness, True if they differ."""

      ijk_right_handed = self.extract_grid_is_right_handed()
      assert rqet.find_tag_text(self.crs_root, 'ProjectedAxisOrder').lower() == 'easting northing'
      # note: if z increases downwards, xyz is left handed
      return ijk_right_handed == self.z_inc_down()

   def write_nexus_corp(self,
                        file_name,
                        local_coords = False,
                        global_xy_units = None,
                        global_z_units = None,
                        global_z_increasing_downward = True,
                        write_nx_ny_nz = False,
                        write_units_keyword = False,
                        write_rh_keyword_if_needed = False,
                        write_corp_keyword = False,
                        use_binary = False,
                        binary_only = False,
                        nan_substitute_value = None):
      """Write grid geometry to file in Nexus CORP ordering."""

      log.info('caching Nexus corner points')
      tm.log_nexus_tm('info')
      self.corner_points(cache_cp_array = True)
      log.debug('duplicating Nexus corner points')
      cp = self.array_corner_points.copy()
      log.debug('resequencing duplicated Nexus corner points')
      gf.resequence_nexus_corp(cp, eight_mode = False, undo = True)
      corp_extent = np.zeros(3, dtype = 'int')
      corp_extent[0] = self.cell_count()  # total number of cells in grid
      corp_extent[1] = 8  # 8 corners of cell: k -/+; j -/+; i -/+
      corp_extent[2] = 3  # x, y, z
      ijk_right_handed = self.extract_grid_is_right_handed()
      if ijk_right_handed is None:
         log.warning('ijk handedness not known')
      elif not ijk_right_handed:
         log.warning('ijk axes are left handed; inverted (fake) xyz handedness required')
      crs_root = self.extract_crs_root()
      if not local_coords:
         if not global_z_increasing_downward:
            log.warning('global z is not increasing with depth as expected by Nexus')
            tm.log_nexus_tm('warning')
         if crs_root is not None:  # todo: otherwise raise exception?
            log.info('converting corner points from local to global reference system')
            self.local_to_global_crs(cp,
                                     crs_root,
                                     global_xy_units = global_xy_units,
                                     global_z_units = global_z_units,
                                     global_z_increasing_downward = global_z_increasing_downward)
      log.info('writing simulator corner point file ' + file_name)
      with open(file_name, 'w') as header:
         header.write('! Nexus corner point data written by resqml_grid module\n')
         header.write('! Nexus is a registered trademark of the Halliburton Company\n\n')
         if write_units_keyword:
            if local_coords:
               if crs_root is not None:
                  crs_xy_units_text = rqet.find_tag(crs_root, 'ProjectedUom').text
                  crs_z_units_text = rqet.find_tag(crs_root, 'VerticalUom').text
                  if crs_xy_units_text == 'm' and crs_z_units_text == 'm':
                     header.write('METRIC\n\n')
                  elif crs_xy_units_text == 'ft' and crs_z_units_text == 'ft':
                     header.write('ENGLISH\n\n')
                  else:
                     header.write('! local coordinates mixed (or not recognized)\n\n')
               else:
                  header.write('! local coordinates unknown\n\n')
            elif global_xy_units is not None and global_z_units is not None and global_xy_units == global_z_units:
               if global_xy_units in ['m', 'metre', 'metres']:
                  header.write('METRIC\n\n')
               elif global_xy_units in ['ft', 'feet', 'foot']:
                  header.write('ENGLISH\n\n')
               else:
                  header.write('! globsl coordinates not recognized\n\n')
            else:
               header.write('! global units unknown or mixed\n\n')
         if write_nx_ny_nz:
            header.write('NX      NY      NZ\n')
            header.write('{0:<7d} {1:<7d} {2:<7d}\n\n'.format(self.extent_kji[2], self.extent_kji[1],
                                                              self.extent_kji[0]))
         if write_rh_keyword_if_needed:
            if ijk_right_handed is None or crs_root is None:
               log.warning('unable to determine whether RIGHTHANDED keyword is needed')
            else:
               xy_axes = rqet.find_tag(crs_root, 'ProjectedAxisOrder').text
               if local_coords:
                  z_inc_down = self.z_inc_down()
                  if not z_inc_down:
                     log.warning('local z is not increasing with depth as expected by Nexus')
                     tm.log_nexus_tm('warning')
               else:
                  z_inc_down = global_z_increasing_downward
               xyz_handedness = rqet.xyz_handedness(xy_axes, z_inc_down)
               if xyz_handedness == 'unknown':
                  log.warning('xyz handedness is not known; unable to determine whether RIGHTHANDED keyword is needed')
               else:
                  if ijk_right_handed == (xyz_handedness == 'right'):  # if either both True or both False
                     header.write('RIGHTHANDED\n\n')
      if write_corp_keyword:
         keyword = 'CORP VALUE'
      else:
         keyword = None
      wd.write_array_to_ascii_file(file_name,
                                   corp_extent,
                                   cp.reshape(tuple(corp_extent)),
                                   target_simulator = 'nexus',
                                   keyword = keyword,
                                   columns = 3,
                                   blank_line_after_i_block = False,
                                   blank_line_after_j_block = True,
                                   append = True,
                                   use_binary = use_binary,
                                   binary_only = binary_only,
                                   nan_substitute_value = nan_substitute_value)

   def xy_units(self):
      """Returns the projected view (x, y) units of measure of the coordinate reference system for the grid.

      :meta common:
      """

      crs_root = self.extract_crs_root()
      if crs_root is None:
         return None
      return rqet.find_tag(crs_root, 'ProjectedUom').text

   def z_units(self):
      """Returns the vertical (z) units of measure of the coordinate reference system for the grid.

      :meta common:
      """

      crs_root = self.extract_crs_root()
      if crs_root is None:
         return None
      return rqet.find_tag(crs_root, 'VerticalUom').text

   def poly_line_for_cell(self, cell_kji0, vertical_ref = 'top'):
      """Returns a numpy array of shape (4, 3) being the 4 corners in order J-I-, J-I+, J+I+, J+I-; from the top or base face."""

      if vertical_ref == 'top':
         kp = 0
      elif vertical_ref == 'base':
         kp = 1
      else:
         raise ValueError('vertical reference not catered for: ' + vertical_ref)
      poly = np.empty((4, 3))
      cp = self.corner_points(cell_kji0 = cell_kji0)
      if cp is None:
         return None
      poly[0] = cp[kp, 0, 0]
      poly[1] = cp[kp, 0, 1]
      poly[2] = cp[kp, 1, 1]
      poly[3] = cp[kp, 1, 0]
      return poly

   def find_cell_for_point_xy(self, x, y, k0 = 0, vertical_ref = 'top', local_coords = True):
      """Searches in 2D for a cell containing point x,y in layer k0; return (j0, i0) or (None, None)."""

      # find minimum of manhatten distances from xy to each corner point
      # then check the four cells around that corner point
      a = np.array([[x, y, 0.0]])  # extra axis needed to keep global_to_local_crs happy
      if not local_coords:
         self.global_to_local_crs(a)
      if a is None:
         return (None, None)
      a[0, 2] = 0.0  # discard z
      kp = 1 if vertical_ref == 'base' else 0
      (pillar_j0, pillar_i0) = self.nearest_pillar(a[0, :2], ref_k0 = k0, kp = kp)
      if pillar_j0 > 0 and pillar_i0 > 0:
         cell_kji0 = np.array((k0, pillar_j0 - 1, pillar_i0 - 1))
         poly = self.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
         if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
      if pillar_j0 > 0 and pillar_i0 < self.ni:
         cell_kji0 = np.array((k0, pillar_j0 - 1, pillar_i0))
         poly = self.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
         if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
      if pillar_j0 < self.nj and pillar_i0 > 0:
         cell_kji0 = np.array((k0, pillar_j0, pillar_i0 - 1))
         poly = self.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
         if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
      if pillar_j0 < self.nj and pillar_i0 < self.ni:
         cell_kji0 = np.array((k0, pillar_j0, pillar_i0))
         poly = self.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
         if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
      return (None, None)

   def find_cell_for_x_sect_xz(self, x_sect, x, z):
      """Returns the (k0, j0) or (k0, i0) indices of the cell containing point x,z in the cross section.

      arguments:
         x_sect (numpy float array of shape (nk, nj or ni, 2, 2, 2 or 3): the cross section x,z or x,y,z data
         x, z (floats): the point of interest in the cross section space

      note:
         the x_sect data is in the form returned by x_section_corner_points() or split_gap_x_section_points();
         the 2nd of the returned pair is either a J index or I index, whichever was not the axis specified
         when generating the x_sect data; returns (None, None) if point inclusion not detected; if xyz data is
         provided, the y values are ignored; note that the point of interest x,z coordinates are in the space of
         x_sect, so if rotation has occurred, the x value is no longer an easting and is typically picked off a
         cross section plot
      """

      def test_cell(p, x_sect, k0, ji0):
         poly = np.array([
            x_sect[k0, ji0, 0, 0, 0:3:2], x_sect[k0, ji0, 0, 1, 0:3:2], x_sect[k0, ji0, 1, 1, 0:3:2], x_sect[k0, ji0, 1,
                                                                                                             0, 0:3:2]
         ])
         if np.any(np.isnan(poly)):
            return False
         return pip.pip_cn(p, poly)

      assert x_sect.ndim == 5 and x_sect.shape[2] == 2 and x_sect.shape[3] == 2 and 2 <= x_sect.shape[4] <= 3
      n_k = x_sect.shape[0]
      n_j_or_i = x_sect.shape[1]
      tolerance = 1.0e-3

      if x_sect.shape[4] == 3:
         diffs = x_sect[:, :, :, :, 0:3:2].copy()  # x,z points only
      else:
         diffs = x_sect.copy()
      diffs -= np.array((x, z))
      diffs = np.sum(diffs * diffs, axis = -1)  # square of distance of each point from given x,z
      flat_index = np.nanargmin(diffs)
      min_dist_sqr = diffs.flatten()[flat_index]
      cell_flat_k0_ji0, flat_k_ji_p = divmod(flat_index, 4)
      found_k0, found_ji0 = divmod(cell_flat_k0_ji0, n_j_or_i)
      found_kp, found_jip = divmod(flat_k_ji_p, 2)

      found = test_cell((x, z), x_sect, found_k0, found_ji0)
      if found:
         return found_k0, found_ji0
      # check cells below whilst still close to point
      while found_k0 < n_k - 1:
         found_k0 += 1
         if np.nanmin(diffs[found_k0, found_ji0]) > min_dist_sqr + tolerance:
            break
         found = test_cell((x, z), x_sect, found_k0, found_ji0)
         if found:
            return found_k0, found_ji0

      # try neighbouring column (in case of fault or point exactly on face)
      ji_neighbour = 1 if found_jip == 1 else -1
      found_ji0 += ji_neighbour
      if 0 <= found_ji0 < n_j_or_i:
         col_diffs = diffs[:, found_ji0]
         flat_index = np.nanargmin(col_diffs)
         if col_diffs.flatten()[flat_index] <= min_dist_sqr + tolerance:
            found_k0 = flat_index // 4
            found = test_cell((x, z), x_sect, found_k0, found_ji0)
            if found:
               return found_k0, found_ji0
            # check cells below whilst still close to point
            while found_k0 < n_k - 1:
               found_k0 += 1
               if np.nanmin(diffs[found_k0, found_ji0]) > min_dist_sqr + tolerance:
                  break
               found = test_cell((x, z), x_sect, found_k0, found_ji0)
               if found:
                  return found_k0, found_ji0

      return None, None

   def skin(self, use_single_layer_tactics = False):
      """Returns a GridSkin composite surface object reoresenting the outer surface of the grid."""

      import resqpy.grid_surface as rqgs

      # could cache 2 versions (with and without single layer tactics)
      if self.grid_skin is None or self.grid_skin.use_single_layer_tactics != use_single_layer_tactics:
         self.grid_skin = rqgs.GridSkin(self, use_single_layer_tactics = use_single_layer_tactics)
      return self.grid_skin

   def create_xml(self,
                  ext_uuid = None,
                  add_as_part = True,
                  add_relationships = True,
                  set_as_grid_root = True,
                  title = None,
                  originator = None,
                  write_active = True,
                  write_geometry = True,
                  extra_metadata = {}):
      """Creates an IJK grid node from a grid object and optionally adds as child of root and/or to parts forest.

      arguments:
         ext_uuid (uuid.UUID): the uuid of the hdf5 external part holding the array data for the grid geometry
         add_as_part (boolean, default True): if True, the newly created xml node is added as a part
            in the model
         add_relationships (boolean, default True): if True, relationship xml parts are created relating the
            new grid part to: the crs, and the hdf5 external part
         set_as_grid_root (boolean, default True): if True, the new grid node is noted as being the 'main' grid
            for the model
         title (string, default 'ROOT'): used as the citation Title text; careful consideration should be given
            to this argument when dealing with multiple grids in one model, as it is the means by which a
            human will distinguish them
         originator (string, optional): the name of the human being who created the ijk grid part;
            default is to use the login name
         write_active (boolean, default True): if True, xml for an active cell property is also generated, but
            only if the active_property_uuid is set and no part exists in the model for that uuid
         write_geometry (boolean, default True): if False, the geometry node is omitted from the xml
         extra_metadata (dict): any key value pairs in this dictionary are added as extra metadata xml nodes

      returns:
         the newly created ijk grid xml node

      notes:
         this code has the concept of a 'main' grid for a model, which resqml does not; it is vaguely
            equivalent to a 'root' grid in a simulation model
         the write_active argument should generally be set to the same value as that passed to the write_hdf5... method;
         the RESQML standard allows the geometry to be omitted for a grid, controlled here by the write_geometry argument;
         the explicit geometry may be omitted for regular grids, in which case the arrays should not be written to the hdf5
         file either

      :meta common:
      """

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()
      if title:
         self.title = title
      if not self.title:
         self.title = 'ROOT'

      ijk = super().create_xml(add_as_part = False, originator = originator, extra_metadata = extra_metadata)

      if self.grid_representation and not write_geometry:
         rqet.create_metadata_xml(node = ijk, extra_metadata = {'grid_flavour': self.grid_representation})

      ni_node = rqet.SubElement(ijk, ns['resqml2'] + 'Ni')
      ni_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      ni_node.text = str(self.extent_kji[2])

      nj_node = rqet.SubElement(ijk, ns['resqml2'] + 'Nj')
      nj_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      nj_node.text = str(self.extent_kji[1])

      nk_node = rqet.SubElement(ijk, ns['resqml2'] + 'Nk')
      nk_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      nk_node.text = str(self.extent_kji[0])

      if self.k_gaps:

         kg_node = rqet.SubElement(ijk, ns['resqml2'] + 'KGaps')
         kg_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'KGaps')
         kg_node.text = '\n'

         kgc_node = rqet.SubElement(kg_node, ns['resqml2'] + 'Count')
         kgc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         kgc_node.text = str(self.k_gaps)

         assert self.k_gap_after_array.ndim == 1 and self.k_gap_after_array.size == self.nk - 1

         kgal_node = rqet.SubElement(kg_node, ns['resqml2'] + 'GapAfterLayer')
         kgal_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
         kgal_node.text = '\n'

         kgal_values = rqet.SubElement(kgal_node, ns['resqml2'] + 'Values')
         kgal_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         kgal_values.text = '\n'

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'GapAfterLayer', root = kgal_values)

      if self.stratigraphic_column_rank_uuid is not None and self.stratigraphic_units is not None:

         assert self.model.type_of_uuid(
            self.stratigraphic_column_rank_uuid) == 'obj_StratigraphicColumnRankInterpretation'

         strata_node = rqet.SubElement(ijk, ns['resqml2'] + 'IntervalStratigraphicUnits')
         strata_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntervalStratigraphicUnits')
         strata_node.text = '\n'

         ui_node = rqet.SubElement(strata_node, ns['resqml2'] + 'UnitIndices')
         ui_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
         ui_node.text = '\n'

         ui_null = rqet.SubElement(ui_node, ns['resqml2'] + 'NullValue')
         ui_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
         ui_null.text = '-1'

         ui_values = rqet.SubElement(ui_node, ns['resqml2'] + 'Values')
         ui_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         ui_values.text = '\n'

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'unitIndices', root = ui_values)

         self.model.create_ref_node('StratigraphicOrganization',
                                    self.model.title(uuid = self.stratigraphic_column_rank_uuid),
                                    self.stratigraphic_column_rank_uuid,
                                    content_type = 'StratigraphicColumnRankInterpretation',
                                    root = strata_node)

      if self.parent_window is not None:

         pw_node = rqet.SubElement(ijk, ns['resqml2'] + 'ParentWindow')
         pw_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IjkParentWindow')
         pw_node.text = '\n'

         assert self.parent_grid_uuid is not None
         parent_grid_root = self.model.root(uuid = self.parent_grid_uuid)
         if parent_grid_root is None:
            pg_title = 'ParentGrid'
         else:
            pg_title = rqet.citation_title_for_node(parent_grid_root)
         self.model.create_ref_node('ParentGrid',
                                    pg_title,
                                    self.parent_grid_uuid,
                                    content_type = 'obj_IjkGridRepresentation',
                                    root = pw_node)

         for axis in range(3):

            regrid_node = rqet.SubElement(pw_node, 'KJI'[axis] + 'Regrid')
            regrid_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Regrid')
            regrid_node.text = '\n'

            if self.is_refinement:
               if self.parent_window.within_coarse_box is None:
                  iiopg = 0  # InitialIndexOnParentGrid
               else:
                  iiopg = self.parent_window.within_coarse_box[0, axis]
            else:
               if self.parent_window.within_fine_box is None:
                  iiopg = 0
               else:
                  iiopg = self.parent_window.within_fine_box[0, axis]
            iiopg_node = rqet.SubElement(regrid_node, ns['resqml2'] + 'InitialIndexOnParentGrid')
            iiopg_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
            iiopg_node.text = str(iiopg)

            if self.parent_window.fine_extent_kji[axis] == self.parent_window.coarse_extent_kji[axis]:
               continue  # one-to-noe mapping

            intervals_node = rqet.SubElement(regrid_node, ns['resqml2'] + 'Intervals')
            intervals_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Intervals')
            intervals_node.text = '\n'

            if self.parent_window.constant_ratios[axis] is not None:
               interval_count = 1
            else:
               if self.is_refinement:
                  interval_count = self.parent_window.coarse_extent_kji[axis]
               else:
                  interval_count = self.parent_window.fine_extent_kji[axis]
            ic_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'IntervalCount')
            ic_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            ic_node.text = str(interval_count)

            pcpi_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'ParentCountPerInterval')
            pcpi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            pcpi_node.text = '\n'

            pcpi_values = rqet.SubElement(pcpi_node, ns['resqml2'] + 'Values')
            pcpi_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            pcpi_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'KJI'[axis] + 'Regrid/ParentCountPerInterval',
                                               root = pcpi_values)

            ccpi_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'ChildCountPerInterval')
            ccpi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            ccpi_node.text = '\n'

            ccpi_values = rqet.SubElement(ccpi_node, ns['resqml2'] + 'Values')
            ccpi_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            ccpi_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'KJI'[axis] + 'Regrid/ChildCountPerInterval',
                                               root = ccpi_values)

            if self.is_refinement and not self.parent_window.equal_proportions[axis]:

               ccw_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'ChildCellWeights')
               ccw_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
               ccw_node.text = rqet.null_xml_text

               ccw_values_node = rqet.SubElement(ccw_node, ns['resqml2'] + 'Values')
               ccw_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
               ccw_values_node.text = rqet.null_xml_text

               self.model.create_hdf5_dataset_ref(ext_uuid,
                                                  self.uuid,
                                                  'KJI'[axis] + 'Regrid/ChildCellWeights',
                                                  root = ccw_values_node)

         # todo: handle omit and cell overlap functionality as part of parent window refining or coarsening

      if write_geometry:

         geom = rqet.SubElement(ijk, ns['resqml2'] + 'Geometry')
         geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'IjkGridGeometry')
         geom.text = '\n'

         # the remainder of this function is populating the geometry node
         self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

         k_dir = rqet.SubElement(geom, ns['resqml2'] + 'KDirection')
         k_dir.set(ns['xsi'] + 'type', ns['resqml2'] + 'KDirection')
         if self.k_direction_is_down:
            k_dir.text = 'down'
         else:
            k_dir.text = 'up'

         handed = rqet.SubElement(geom, ns['resqml2'] + 'GridIsRighthanded')
         handed.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
         handed.text = str(self.grid_is_right_handed).lower()

         p_shape = rqet.SubElement(geom, ns['resqml2'] + 'PillarShape')
         p_shape.set(ns['xsi'] + 'type', ns['resqml2'] + 'PillarShape')
         p_shape.text = self.pillar_shape

         points_node = rqet.SubElement(geom, ns['resqml2'] + 'Points')
         points_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
         points_node.text = '\n'

         coords = rqet.SubElement(points_node, ns['resqml2'] + 'Coordinates')
         coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         coords.text = '\n'

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Points', root = coords)

         if always_write_pillar_geometry_is_defined_array or not self.geometry_defined_for_all_pillars(
               cache_array = True):

            pillar_def = rqet.SubElement(geom, ns['resqml2'] + 'PillarGeometryIsDefined')
            pillar_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
            pillar_def.text = '\n'

            pd_values = rqet.SubElement(pillar_def, ns['resqml2'] + 'Values')
            pd_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            pd_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'PillarGeometryIsDefined', root = pd_values)

         else:

            pillar_def = rqet.SubElement(geom, ns['resqml2'] + 'PillarGeometryIsDefined')
            pillar_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanConstantArray')
            pillar_def.text = '\n'

            pd_value = rqet.SubElement(pillar_def, ns['resqml2'] + 'Value')
            pd_value.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
            pd_value.text = 'true'

            pd_count = rqet.SubElement(pillar_def, ns['resqml2'] + 'Count')
            pd_count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            pd_count.text = str((self.extent_kji[1] + 1) * (self.extent_kji[2] + 1))

         if always_write_cell_geometry_is_defined_array or not self.geometry_defined_for_all_cells(cache_array = True):

            cell_def = rqet.SubElement(geom, ns['resqml2'] + 'CellGeometryIsDefined')
            cell_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
            cell_def.text = '\n'

            cd_values = rqet.SubElement(cell_def, ns['resqml2'] + 'Values')
            cd_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            cd_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'CellGeometryIsDefined', root = cd_values)

         else:

            cell_def = rqet.SubElement(geom, ns['resqml2'] + 'CellGeometryIsDefined')
            cell_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanConstantArray')
            cell_def.text = '\n'

            cd_value = rqet.SubElement(cell_def, ns['resqml2'] + 'Value')
            cd_value.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
            cd_value.text = 'true'

            cd_count = rqet.SubElement(cell_def, ns['resqml2'] + 'Count')
            cd_count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            cd_count.text = str(self.nk * self.nj * self.ni)

         if self.has_split_coordinate_lines:

            scl = rqet.SubElement(geom, ns['resqml2'] + 'SplitCoordinateLines')
            scl.set(ns['xsi'] + 'type', ns['resqml2'] + 'ColumnLayerSplitCoordinateLines')
            scl.text = '\n'

            scl_count = rqet.SubElement(scl, ns['resqml2'] + 'Count')
            scl_count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            scl_count.text = str(self.split_pillars_count)

            pi_node = rqet.SubElement(scl, ns['resqml2'] + 'PillarIndices')
            pi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            pi_node.text = '\n'

            pi_null = rqet.SubElement(pi_node, ns['resqml2'] + 'NullValue')
            pi_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            pi_null.text = str((self.extent_kji[1] + 1) * (self.extent_kji[2] + 1))

            pi_values = rqet.SubElement(pi_node, ns['resqml2'] + 'Values')
            pi_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            pi_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'PillarIndices', root = pi_values)

            cpscl = rqet.SubElement(scl, ns['resqml2'] + 'ColumnsPerSplitCoordinateLine')
            cpscl.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlJaggedArray')
            cpscl.text = '\n'

            elements = rqet.SubElement(cpscl, ns['resqml2'] + 'Elements')
            elements.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            elements.text = '\n'

            el_null = rqet.SubElement(elements, ns['resqml2'] + 'NullValue')
            el_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            el_null.text = str(self.extent_kji[1] * self.extent_kji[2])

            el_values = rqet.SubElement(elements, ns['resqml2'] + 'Values')
            el_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            el_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'ColumnsPerSplitCoordinateLine/elements',
                                               root = el_values)

            c_length = rqet.SubElement(cpscl, ns['resqml2'] + 'CumulativeLength')
            c_length.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            c_length.text = '\n'

            cl_null = rqet.SubElement(c_length, ns['resqml2'] + 'NullValue')
            cl_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            cl_null.text = '0'

            cl_values = rqet.SubElement(c_length, ns['resqml2'] + 'Values')
            cl_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            cl_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'ColumnsPerSplitCoordinateLine/cumulativeLength',
                                               root = cl_values)

      if add_as_part:
         self.model.add_part('obj_IjkGridRepresentation', self.uuid, ijk)
         if add_relationships:
            if self.stratigraphic_column_rank_uuid is not None and self.stratigraphic_units is not None:
               self.model.create_reciprocal_relationship(ijk, 'destinationObject',
                                                         self.model.root_for_uuid(self.stratigraphic_column_rank_uuid),
                                                         'sourceObject')
            if write_geometry:
               # create 2 way relationship between IjkGrid and Crs
               self.model.create_reciprocal_relationship(ijk, 'destinationObject', self.crs_root, 'sourceObject')
               # create 2 way relationship between IjkGrid and Ext
               ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
               ext_node = self.model.root_for_part(ext_part)
               self.model.create_reciprocal_relationship(ijk, 'mlToExternalPartProxy', ext_node,
                                                         'externalPartProxyToMl')
            # create relationship with parent grid
            if self.parent_window is not None and self.parent_grid_uuid is not None:
               self.model.create_reciprocal_relationship(ijk, 'destinationObject',
                                                         self.model.root_for_uuid(self.parent_grid_uuid),
                                                         'sourceObject')

      if write_active and self.active_property_uuid is not None and self.model.part(
            uuid = self.active_property_uuid) is None:
         active_collection = rprop.PropertyCollection()
         active_collection.set_support(support = self)
         active_collection.create_xml(None,
                                      None,
                                      'ACTIVE',
                                      'active',
                                      p_uuid = self.active_property_uuid,
                                      discrete = True,
                                      add_min_max = False,
                                      find_local_property_kinds = True)

      return ijk


# end of Grid class


class RegularGrid(Grid):
   """Class for completely regular block grids aligned with xyz axes."""

   # For now generate a standard unsplit pillar grid
   # todo: use RESQML lattice like geometry specification

   def __init__(self,
                parent_model,
                extent_kji = None,
                dxyz = None,
                dxyz_dkji = None,
                origin = (0.0, 0.0, 0.0),
                crs_uuid = None,
                use_vertical = False,
                mesh = None,
                mesh_dz_dk = 1.0,
                uuid = None,
                set_points_cached = False,
                find_properties = True,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Creates a regular grid object based on dxyz, or derived from a Mesh object.

      arguments:
         parent_model (model.Model object): the model to which the new grid will be assigned
         extent_kji (triple positive integers, optional): the number of cells in the grid (nk, nj, ni);
            required unless grid_root is present
         dxyz (triple float, optional): use when the I,J,K axes align with the x,y,z axes (possible with inverted
            directions); the size of each cell (dx, dy, dz); values may be negative
         dxyz_dkji (numpy float array of shape (3, 3), optional): how x,y,z values increase with each step in each
            direction K,J,I; first index is KJI, second index is xyz; only one of dxyz, dxyz_dkji and mesh should be
            present; NB axis ordering is different to that used in Mesh class for dxyz_dij
         origin (triple float, default (0.0, 0.0, 0.0)): the location in the local coordinate space of the crs of
            the 'first' corner point of the grid
         crs_uuid (uuid.UUID, optional): the uuid of the coordinate reference system for the grid
         use_vertical (boolean, default False): if True and the pillars of the regular grid are vertical then a
            pillar shape of 'vertical' is used; if False (or the pillars are not vertical), then a pillar shape of
            'straight' is used
         mesh (surface.Mesh object, optional): if present, the I,J layout of the grid is based on the mesh, which
            must be regular, and the K cell size is given by the mesh_dz_dk argument; if present, then dxyz and
            dxyz_dkji must be None
         mesh_dz_dk (float, default 1.0): the size of cells in the K axis, which is aligned with the z axis, when
            starting from a mesh; ignored if mesh is None
         uuid (optional): the root of the xml tree for the grid part; if present, the RegularGrid object is
            based on existing data or a mix of that data and other arguments where present
         set_points_cached (boolean, default False): if True, an explicit geometry is created for the regular grid
            in the form of the cached points array
         find_properties (boolean, default True): if True and grid_root is not None, a grid property collection is
            instantiated as an attribute, holding properties for which this grid is the supporting representation
         title (str, optional): citation title for new grid; ignored if loading from xml
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if loading from xml
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if loading from xml

      returns:
         a newly created RegularGrid object with inheritance from the Grid class

      notes:
         the RESQML standard allows for regular grid geometry pillars to be stored as parametric lines
         but that is not yet supported by this code base; however, constant dx, dy, dz arrays are supported;
         alternatively, regular meshes (Grid2d) may be stored in parameterized form and used to generate a
         regular grid here;
         if root_grid, dxyz, dxyz_dkji and mesh arguments are all None then unit cube cells aligned with
         the x,y,z axes will be generated;
         to store the geometry explicitly use the following methods: make_regular_points_cached(), write_hdf5(),
         create_xml(..., write_geometry = True);
         otherwise, avoid write_hdf5() and call create_xml(..., write_geometry = False)

      :meta common:
      """

      if uuid is None:
         super().__init__(parent_model)
         self.grid_representation = 'IjkBlockGrid'  # this is not RESQML and might cause issues elsewhere; revert to IjkGrid if needed
         self.extent_kji = np.array(extent_kji).copy()
         self.nk, self.nj, self.ni = self.extent_kji
         self.k_direction_is_down = True  # assumed direction
         self.grid_is_right_handed = False  # todo: work it out from dxyz_dkji and crs xyz handedness
         self.has_split_coordinate_lines = False
         self.k_gaps = None
         self.k_gap_after_array = None
         self.k_raw_index_array = None
         self.inactive = None
         self.all_inactive = None
         self.geometry_defined_for_all_cells_cached = True
         self.geometry_defined_for_all_pillars_cached = True
         self.array_cell_geometry_is_defined = np.full(tuple(self.extent_kji), True, dtype = bool)
      else:
         assert is_regular_grid(parent_model.root_for_uuid(uuid))
         super().__init__(parent_model,
                          uuid = uuid,
                          find_properties = find_properties,
                          geometry_required = False,
                          title = title,
                          originator = originator,
                          extra_metadata = extra_metadata)
         self.grid_representation = 'IjkBlockGrid'
         if dxyz is None and dxyz_dkji is None:
            # find cell length properties and populate dxyz from those values
            assert self.property_collection is not None
            dxi_part = self.property_collection.singleton(property_kind = 'cell length',
                                                          facet_type = 'direction',
                                                          facet = 'I')
            dyj_part = self.property_collection.singleton(property_kind = 'cell length',
                                                          facet_type = 'direction',
                                                          facet = 'J')
            dzk_part = self.property_collection.singleton(property_kind = 'cell length',
                                                          facet_type = 'direction',
                                                          facet = 'K')
            assert dxi_part is not None and dyj_part is not None and dzk_part is not None
            dxi = float(self.property_collection.constant_value_for_part(dxi_part))
            dyj = float(self.property_collection.constant_value_for_part(dyj_part))
            dzk = float(self.property_collection.constant_value_for_part(dzk_part))
            assert dxi is not None and dyj is not None and dzk is not None
            dxyz = (dxi, dyj, dzk)
         if crs_uuid is None:
            self.crs_uuid

      if mesh is not None:
         assert mesh.flavour == 'regular'
         assert dxyz is None and dxyz_dkji is None
         origin = mesh.regular_origin
         dxyz_dkji = np.empty((3, 3))
         dxyz_dkji[0, :] = mesh_dz_dk
         dxyz_dkji[1, :] = mesh.regular_dxyz_dij[1]  # J axis
         dxyz_dkji[2, :] = mesh.regular_dxyz_dij[0]  # I axis
         if crs_uuid is None:
            crs_uuid = mesh.crs_uuid
         else:
            assert bu.matching_uuids(crs_uuid, mesh.crs_uuid)

      assert dxyz is None or dxyz_dkji is None
      if dxyz is None and dxyz_dkji is None:
         dxyz = (1.0, 1.0, 1.0)
      if dxyz_dkji is None:
         dxyz_dkji = np.array([[0.0, 0.0, dxyz[2]], [0.0, dxyz[1], 0.0], [dxyz[0], 0.0, 0.0]])
      self.block_origin = np.array(origin).copy()
      self.block_dxyz_dkji = np.array(dxyz_dkji).copy()
      if use_vertical and dxyz_dkji[0][0] == 0.0 and dxyz_dkji[0][1] == 0.0:  # ie. no x,y change with k
         self.pillar_shape = 'vertical'
      else:
         self.pillar_shape = 'straight'

      if set_points_cached:
         self.make_regular_points_cached()

      if crs_uuid is None:
         new_crs = rqc.Crs(parent_model)
         self.crs_uuid = new_crs.uuid
         self.crs_root = new_crs.create_xml(reuse = True)
      else:
         self.crs_uuid = crs_uuid
         self.crs_root = parent_model.root_for_uuid(crs_uuid)

      if self.uuid is None:
         self.uuid = bu.new_uuid()

   def make_regular_points_cached(self):
      """Set up the cached points array as an explicit representation of the regular grid geometry."""

      if hasattr(self, 'points_cached') and self.points_cached is not None:
         return
      self.points_cached = np.zeros((self.nk + 1, self.nj + 1, self.ni + 1, 3))
      # todo: replace for loops with linspace
      for k in range(self.nk):
         self.points_cached[k + 1, 0, 0] = self.points_cached[k, 0, 0] + self.block_dxyz_dkji[0]
      for j in range(self.nj):
         self.points_cached[:, j + 1, 0] = self.points_cached[:, j, 0] + self.block_dxyz_dkji[1]
      for i in range(self.ni):
         self.points_cached[:, :, i + 1] = self.points_cached[:, :, i] + self.block_dxyz_dkji[2]
      self.points_cached[:, :, :] += self.block_origin

   def axial_lengths_kji(self):
      """Returns a triple float being lengths of primary axes (K, J, I) for each cell."""

      return vec.naive_lengths(self.block_dxyz_dkji)

   # override of Grid methods

   def point_raw(self, index = None, points_root = None, cache_array = True):
      """Returns element from points data, indexed as corner point (k0, j0, i0); can optionally be used to cache points data.

      arguments:
         index (3 integers, optional): if not None, the index into the raw points data for the point of interest
         points_root (ignored)
         cache_array (boolean, default True): if True, the raw points data is cached in memory as a side effect

      returns:
         (x, y, z) of selected point as a 3 element numpy vector, or None if index is None

      notes:
         this function is typically called either to cache the points data in memory, or to fetch the coordinates of
         a single corner point;
         the index should be a triple kji0 with axes ranging over the shared corners nk+1, nj+1, ni+1
      """

      assert cache_array or index is not None

      if cache_array:
         self.make_regular_points_cached()
         if index is None:
            return None
         return self.points_cached[tuple(index)]

      return self.block_origin + np.sum(np.repeat(np.array(index).reshape((3, 1)), 3, axis = -1) * self.block_dxyz_dkji,
                                        axis = 0)

   def half_cell_transmissibility(self, use_property = None, realization = None, tolerance = None):
      """Returns (and caches if realization is None) half cell transmissibilities for this regular grid.

      arguments:
         use_property (ignored)
         realization (int, optional) if present, only a property with this realization number will be used
         tolerance (ignored)

      returns:
         numpy float array of shape (nk, nj, ni, 3, 2) where the 3 covers K,J,I and the 2 covers the
            face polarity: - (0) and + (1); units will depend on the length units of the coordinate reference
            system for the grid; the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m
            and ft respectively

      notes:
         the values for - and + polarity will always be equal, the data is duplicated for compatibility with
         parent Grid class;
         the returned array is in the logical resqpy arrangement; it must be discombobulated before being
         added as a property; this method does not write to hdf5, nor create a new property or xml;
         if realization is None, a grid attribute cached array will be used
      """

      # todo: allow passing of property uuids for ntg, k_k, j, i

      if realization is None and hasattr(self, 'array_half_cell_t'):
         return self.array_half_cell_t

      half_t = rqtr.half_cell_t(
         self, realization = realization)  # note: properties must be identifiable in property_collection

      # introduce facial polarity axis for compatibility with parent Grid class
      assert half_t.ndim == 4
      half_t = np.expand_dims(half_t, -1)
      half_t = np.repeat(half_t, 2, axis = -1)

      if realization is None:
         self.array_half_cell_t = half_t

      return half_t

   def centre_point(self, cell_kji0 = None):
      """Returns centre point of a cell or array of centre points of all cells.

      arguments:
         cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
            centre point is required; zero based indexing
         cache_centre_array (boolean, default False): If True, or cell_kji0 is None, an array of centre points
            is generated and added as an attribute of the grid, with attribute name array_centre_point

      returns:
         (x, y, z) 3 element numpy array of floats holding centre point of cell;
         or numpy 3+1D array if cell_kji0 is None

      note:
         resulting coordinates are in the same (local) crs as the grid points
      """

      if cell_kji0 is not None:
         float_kji0 = np.array(cell_kji0, dtype = float) + 0.5
         centre = self.block_origin + np.sum(
            self.block_dxyz_dkji * np.expand_dims(float_kji0, axis = -1).repeat(3, axis = -1), axis = 0)
         return centre

      centres = np.zeros((self.nk, self.nj, self.ni, 3))
      # todo: replace for loops with linspace
      for k in range(self.nk - 1):
         centres[k + 1, 0, 0] = centres[k, 0, 0] + self.block_dxyz_dkji[0]
      for j in range(self.nj - 1):
         centres[:, j + 1, 0] = centres[:, j, 0] + self.block_dxyz_dkji[1]
      for i in range(self.ni - 1):
         centres[:, :, i + 1] = centres[:, :, i] + self.block_dxyz_dkji[2]
      centres += self.block_origin + 0.5 * np.sum(self.block_dxyz_dkji, axis = 0)
      return centres

   def volume(self, cell_kji0 = None):
      """Returns bulk rock volume of cell or numpy array of bulk rock volumes for all cells.

      arguments:
         cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
                               volume is required; zero based indexing

      returns:
         float, being the volume of cell identified by cell_kji0;
         or numpy float array of shape (nk, nj, ni) if cell_kji0 is None

      notes:
         the function can be used to find the volume of a single cell, or all cells;
         grid's coordinate reference system must use same units in z as xy (projected);
         units of result are implicitly determined by coordinates in grid's coordinate reference system;
         the method currently assumes that the primary i, j, k axes are mutually orthogonal
      """

      vol = np.product(vec.naive_lengths(self.block_dxyz_dkji))
      if cell_kji0 is not None:
         return vol
      return np.full((self.nk, self.nj, self.ni), vol)

   def thickness(self, cell_kji0 = None, **kwargs):
      """Returns cell thickness (K axial length) for a single cell or full array.

      arguments:
         cell_kji0 (triple int, optional): if present, the thickness for a single cell is returned;
            if None, an array is returned
         all other arguments ignored; present for compatibility with same method in Grid()

      returns:
         float, or numpy float array filled with a constant
      """

      thick = self.axial_lengths_kji()[0]
      if cell_kji0 is not None:
         return thick
      return np.full((self.nk, self.nj, self.ni), thick, dtype = float)

   def pinched_out(self, cell_kji0 = None, **kwargs):
      """Returns pinched out boolean (always False) for a single cell or full array.

      arguments:
         cell_kji0 (triple int, optional): if present, the pinched out flag for a single cell is returned;
            if None, an array is returned
         all other arguments ignored; present for compatibility with same method in Grid()

      returns:
         False, or numpy array filled with False
      """

      if cell_kji0 is not None:
         return False
      return np.full((self.nk, self.nj, self.ni), False, dtype = bool)

   def actual_pillar_shape(self, patch_metadata = False, tolerance = 0.001):
      """Returns actual shape of pillars.

      arguments:
         patch_metadata (boolean, default False): if True, the actual shape replaces whatever was in the metadata
         tolerance (float, ignored)

      returns:
         string: 'vertical', 'straight' or 'curved'

      note:
         setting patch_metadata True will affect the attribute in this Grid object; however, it will not be
         preserved unless the create_xml() method is called, followed at some point with model.store_epc()
      """

      if np.all(self.block_dxyz_dkji[0, :2] == 0.0):
         return 'vertical'
      return 'straight'

   def create_xml(self,
                  ext_uuid = None,
                  add_as_part = True,
                  add_relationships = True,
                  set_as_grid_root = True,
                  root = None,
                  title = None,
                  originator = None,
                  write_active = True,
                  write_geometry = False,
                  extra_metadata = {},
                  add_cell_length_properties = True):
      """Creates xml for this RegularGrid object; by default the explicit geometry is not included.

      see docstring for Grid.create_xml()

      additional argument:
         add_cell_length_properties (boolean, default True): if True, 3 constant property arrays with cells as
            indexable element are created to hold the lengths of the primary axes of the cells; the xml is
            created for the properties and they are added to the model (no hdf5 write needed)

      :meta common:
      """

      node = super().create_xml(ext_uuid = ext_uuid,
                                add_as_part = add_as_part,
                                add_relationships = add_relationships,
                                set_as_grid_root = set_as_grid_root,
                                title = title,
                                originator = originator,
                                write_active = write_active,
                                write_geometry = write_geometry,
                                extra_metadata = extra_metadata)

      if add_cell_length_properties:
         axes_lengths_kji = self.axial_lengths_kji()
         dpc = rprop.GridPropertyCollection()
         dpc.set_grid(self)
         for axis in range(3):
            dpc.add_cached_array_to_imported_list(None,
                                                  'regular grid',
                                                  'D' + 'ZYX'[axis],
                                                  discrete = False,
                                                  uom = self.xy_units(),
                                                  property_kind = 'cell length',
                                                  facet_type = 'direction',
                                                  facet = 'KJI'[axis],
                                                  indexable_element = 'cells',
                                                  count = 1,
                                                  const_value = axes_lengths_kji[axis])
         dpc.create_xml_for_imported_list_and_add_parts_to_model()
         if self.property_collection is None:
            self.property_collection = dpc
         else:
            if self.property_collection.support is None:
               self.property_collection.set_support(support = self)
            self.property_collection.inherit_parts_from_other_collection(dpc)

      return node


def establish_zone_property_kind(model):
   """Returns zone local property kind object, creating the xml and adding as part if not found in model."""

   zone_pk_root = model.root(obj_type = 'LocalPropertyKind', title = 'zone')
   if zone_pk_root is None:
      zone_pk = rprop.PropertyKind(model, title = 'zone', parent_property_kind = 'discrete')
      zone_pk.create_xml()
   else:
      zone_pk = rprop.PropertyKind(model, root_node = zone_pk_root)
   return zone_pk


def extent_kji_from_root(root_node):
   """Returns kji extent as stored in xml."""

   return (rqet.find_tag_int(root_node, 'Nk'), rqet.find_tag_int(root_node, 'Nj'), rqet.find_tag_int(root_node, 'Ni'))


def grid_flavour(grid_root):
   """Returns a string indicating type of grid geometry, currently 'IjkGrid' or 'IjkBlockGrid'."""

   if grid_root is None:
      return None
   em = rqet.load_metadata_from_xml(grid_root)
   flavour = em.get('grid_flavour')
   if flavour is None:
      node_type = rqet.node_type(grid_root, strip_obj = True)
      if node_type == 'IjkGridRepresentation':
         if rqet.find_tag(grid_root, 'Geometry') is not None:
            flavour = 'IjkGrid'
         else:
            flavour = 'IjkBlockGrid'  # this might cause issues
      elif node_type == 'UnstructuredGridRepresentation':
         cell_shape = rqet.find_nested_tags_text(grid_root, ['Geometry', 'CellShape'])
         if cell_shape is None or cell_shape == 'polyhedral':
            flavour = 'UnstructuredGrid'
         elif cell_shape == 'tetrahedral':
            flavour = 'TetraGrid'
         elif cell_shape == 'hexahedral':
            flavour = 'HexaGrid'
         elif cell_shape == 'pyramidal':
            flavour = 'PyramidGrid'
         elif cell_shape == 'prism':
            flavour = 'PrismGrid'
   return flavour


def is_regular_grid(grid_root):
   """Returns True if the xml root node is for a RegularGrid."""

   return grid_flavour(grid_root) == 'IjkBlockGrid'


def any_grid(parent_model, grid_root = None, uuid = None, find_properties = True):
   """Returns a Grid or RegularGrid or UnstructuredGrid object depending on the extra metadata in the xml."""

   import resqpy.unstructured as rug

   if uuid is None and grid_root is not None:
      uuid = rqet.uuid_for_part_root(grid_root)
   flavour = grid_flavour(parent_model.root_for_uuid(uuid))
   if flavour is None:
      return None
   if flavour == 'IjkGrid':
      return Grid(parent_model, uuid = uuid, find_properties = find_properties)
   if flavour == 'IjkBlockGrid':
      return RegularGrid(parent_model, extent_kji = None, uuid = uuid, find_properties = find_properties)
   if flavour == 'UnstructuredGrid':
      return rug.UnstructuredGrid(parent_model, uuid = uuid, find_properties = find_properties)
   if flavour == 'TetraGrid':
      return rug.TetraGrid(parent_model, uuid = uuid, find_properties = find_properties)
   if flavour == 'HexaGrid':
      return rug.HexaGrid(parent_model, uuid = uuid, find_properties = find_properties)
   if flavour == 'PyramidGrid':
      return rug.PyramidGrid(parent_model, uuid = uuid, find_properties = find_properties)
   if flavour == 'PrismGrid':
      return rug.PrismGrid(parent_model, uuid = uuid, find_properties = find_properties)
   return None
