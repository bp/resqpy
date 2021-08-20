"""unstructured.py: resqpy unstructured grid module."""

version = '20th August 2021'

import logging

log = logging.getLogger(__name__)
log.debug('unstructured.py version ' + version)

import numpy as np

from resqpy.olio.base import BaseResqpy
import resqpy.olio.uuid as bu
import resqpy.weights_and_measures as bwam
import resqpy.olio.xml_et as rqet
import resqpy.olio.write_hdf5 as rwh5
from resqpy.olio.xml_namespaces import curly_namespace as ns

import resqpy.crs as rqc
import resqpy.property as rprop

valid_cell_shapes = ['polyhedral', 'tetrahedral', 'pyramidal', 'prism', 'hexahedral']


class UnstructuredGrid(BaseResqpy):
   """Class for RESQML Unstructured Grid objects."""

   resqml_type = 'UnstructuredGridRepresentation'

   def __init__(self,
                parent_model,
                uuid = None,
                find_properties = True,
                geometry_required = True,
                cache_geometry = False,
                cell_shape = 'polyhedral',
                title = None,
                originator = None,
                extra_metadata = {}):
      """Create an Unstructured Grid object and optionally populate from xml tree.

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         find_properties (boolean, default True): if True and uuid is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         geometry_required (boolean, default True): if True and no geometry node exists in the xml,
            an assertion error is raised; ignored if uuid is None
         cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
            are loaded into attributes of the new grid object
         cell_shape (str, optional): one of 'polyhedral', 'tetrahedral', 'pyramidal', 'prism', 'hexahedral';
            ignored if uuid is present
         title (str, optional): citation title for new grid; ignored if uuid is present
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if uuid is present
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if uuid is present

      returns:
         a newly created Unstructured Grid object

      notes:
         if not instantiating from an existing RESQML object, then pass a cell_shape here (if geometry is
         going to be used), then set the cell count and, for geometry, build the points array and other
         arrays before writing to the hdf5 file and creating the xml;
         hinge node faces and subnode topology not yet supported at all;
         setting cache_geometry True is equivalent to calling the cache_all_geometry_arrays() method

      :meta common:
      """

      if cell_shape is not None:
         assert cell_shape in valid_cell_shapes, f'invalid cell shape {cell_shape} for unstructured grid'

      self.cell_count = None  #: the number of cells in the grid
      self.cell_shape = cell_shape  #: the shape of cells withing the grid
      self.crs_uuid = None  #: uuid of the coordinate reference system used by the grid's geometry
      self.points_cached = None  #: numpy array of raw points data; loaded on demand
      self.node_count = None  #: number of distinct points used in geometry
      self.face_count = None  #: number of distinct faces used in geometry
      self.nodes_per_face = None
      self.nodes_per_face_cl = None
      self.faces_per_cell = None
      self.cell_face_is_right_handed = None
      self.faces_per_cell_cl = None
      self.inactive = None  #: numpy boolean array indicating which cells are inactive in flow simulation
      self.all_inactive = None  #: boolean indicating whether all cells are inactive
      self.active_property_uuid = None  #: uuid of property holding active cell boolean array (used to populate inactive)
      self.grid_representation = None  #: flavour of grid, 'UnstructuredGrid'; not much used
      self.geometry_root = None  #: xml node at root of geometry sub-tree, if present

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      if not self.title:
         self.title = 'ROOT'

      if uuid is not None:
         if geometry_required:
            assert self.geometry_root is not None, 'unstructured grid geometry not present in xml'
         if cache_geometry and self.geometry_root is not None:
            self.cache_all_geometry_arrays()
         if find_properties:
            self.extract_property_collection()

   def _load_from_xml(self):
      # Extract simple attributes from xml and set as attributes in this resqpy object
      grid_root = self.root
      assert grid_root is not None
      self.grid_representation = 'UnstructuredGrid'  # this attribute not much used
      self.cell_count = rqet.find_tag_int(grid_root, 'CellCount')
      assert self.cell_count > 0
      self.geometry_root = rqet.find_tag(grid_root, 'Geometry')
      if self.geometry_root is None:
         self.cell_shape = None
      else:
         self.cell_shape = rqet.find_tag_text(grid_root, 'CellShape')
         assert self.cell_shape in valid_cell_shapes
         self.node_count = rqet.find_tag_int(grid_root, 'NodeCount')
         assert self.node_count > 3
         self.face_count = rqet.find_tag_int(grid_root, 'FaceCount')
         assert self.face_count > 3
         # note: arrays not loaded until demanded; see cache_all_geometry_arrays()

   def set_cell_count(self, n: int):
      """Set the number of cells in the grid.

      arguments:
         n (int): the number of cells in the unstructured grid

      note:
         only call this method when creating a new grid, not when working from an existing RESQML grid
      """
      assert self.cell_count is None or self.cell_count == n
      self.cell_count = n

   def active_cell_count(self):
      """Returns the number of cells deemed to be active for flow simulation purposes."""
      if self.inactive is None:
         return self.cell_count
      return self.cell_count - np.count_nonzero(self.inactive)

   def cache_all_geometry_arrays(self):
      """Loads from hdf5 into memory all the arrays defining the grid geometry.

      returns:
         None

      notes:
         call this method if much grid geometry processing is coming up;
         the arrays are cached as direct attributes to this grid object;
         the node and face indices make use of 'jagged' arrays (effectively an array of lists represented as
         a linear array and a 'cumulative length' index array)

      :meta common:
      """

      assert self.node_count is not None and self.face_count is not None

      self.points_ref()

      if self.nodes_per_face is None:
         self._load_jagged_array('NodesPerFace', 'nodes_per_face')
         assert len(self.nodes_per_face_cl) == self.face_count

      if self.faces_per_cell is None:
         self._load_jagged_array('FacesPerCell', 'faces_per_cell')
         assert len(self.faces_per_cell_cl) == self.cell_count

      if self.cell_face_is_right_handed is None:
         assert self.geometry_root is not None
         cfirh_node = rqet.find_tag(self.geometry_root, 'CellFaceIsRightHanded')
         assert cfirh_node is not None
         h5_key_pair = self.model.h5_uuid_and_path_for_node(cfirh_node)
         self.model.h5_array_element(h5_key_pair,
                                     index = None,
                                     cache_array = True,
                                     object = self,
                                     array_attribute = 'CellFaceIsRightHanded',
                                     required_shape = (len(self.faces_per_cell),),
                                     dtype = 'bool')

   def _load_jagged_array(self, tag, main_attribute):
      assert self.geometry_root is not None
      root_node = rqet.find_tag(self.geometry_root, tag)
      assert root_node is not None
      elements_root = rqet.find_tag(root_node, 'Elements')
      h5_key_pair = self.model.h5_uuid_and_path_for_node(elements_root)
      self.model.h5_array_element(h5_key_pair,
                                  index = None,
                                  cache_array = True,
                                  object = self,
                                  array_attribute = main_attribute,
                                  dtype = 'int')
      cum_length_root = rqet.find_tag(root_node, 'CumulativeLength')
      h5_key_pair = self.model.h5_uuid_and_path_for_node(cum_length_root)
      self.model.h5_array_element(h5_key_pair,
                                  index = None,
                                  cache_array = True,
                                  object = self,
                                  array_attribute = main_attribute + '_cl',
                                  dtype = 'int')

   def extract_property_collection(self):
      """Load grid property collection object holding lists of all properties in model that relate to this grid.

      returns:
         resqml_property.PropertyCollection object

      note:
         a reference to the grid property collection is cached in this grid object; if the properties change,
         for example by generating some new properties, the property_collection attribute of the grid object
         would need to be reset to None elsewhere before calling this method again
      """

      if self.property_collection is not None:
         return self.property_collection
      self.property_collection = rprop.PropertyCollection(support = self)
      return self.property_collection

   def points_ref(self):
      """Returns an in-memory numpy array containing the xyz data for points used in the grid geometry.

      returns:
         numpy array of shape (node_count,)

      notes:
         this is the usual way to get at the actual grid geometry points data in the native RESQML layout;
         the array is cached as an attribute of the grid object

      :meta common:
      """

      if self.points_cached is None:

         assert self.node_count is not None

         p_root = rqet.find_tag(self.geometry_root, 'Points')
         if p_root is None:
            log.debug('points_ref() returning None as geometry not present')
            return None  # geometry not present

         assert rqet.node_type(p_root) == 'Point3dHdf5Array'
         h5_key_pair = self.model.h5_uuid_and_path_for_node(p_root, tag = 'Coordinates')
         if h5_key_pair is None:
            return None

         self.model.h5_array_element(h5_key_pair,
                                     index = None,
                                     cache_array = True,
                                     object = self,
                                     array_attribute = 'points_cached',
                                     required_shape = (self.node_count, 3))

      return self.points_cached

   def centre_point(self, cell = None, cache_centre_array = False):
      """Returns centre point of a cell or array of centre points of all cells; optionally cache centre points for all cells.

      arguments:
         cell (optional): if present, the cell number of the individual cell for which the
            centre point is required; zero based indexing
         cache_centre_array (boolean, default False): If True, or cell is None, an array of centre points
            is generated and added as an attribute of the grid, with attribute name array_centre_point

      returns:
         (x, y, z) 3 element numpy array of floats holding centre point of cell;
         or numpy 2D array of shape (cell_count, 3) if cell is None

      notes:
         a simple mean of the distinct contributing nodes is used to calculate the centre point of a cell;
         resulting coordinates are in the same (local) crs as the grid points

      :meta common:
      """

      if cell is None:
         cache_centre_array = True

      if hasattr(self, 'array_centre_point'):
         if cell is None:
            return self.array_centre_point
         return self.array_centre_point[cell]  # could check for nan here and return None

      if cache_centre_array:
         # TODO
         pass
      else:
         # only calculate for the specified cell
         # TODO
         pass

      return None

   def write_hdf5(self):
      # TODO
      pass

   def create_xml(self):
      # TODO
      pass
