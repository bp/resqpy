"""unstructured.py: resqpy unstructured grid module."""

version = '7th September 2021'

import logging

log = logging.getLogger(__name__)
log.debug('unstructured.py version ' + version)

import numpy as np

from resqpy.olio.base import BaseResqpy
import resqpy.olio.uuid as bu
import resqpy.weights_and_measures as bwam
import resqpy.olio.vector_utilities as vec
import resqpy.olio.triangulation as tri
import resqpy.olio.volume as vol
import resqpy.olio.intersection as meet
import resqpy.olio.xml_et as rqet
import resqpy.olio.write_hdf5 as rwh5
from resqpy.olio.xml_namespaces import curly_namespace as ns

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.surface as rqs
import resqpy.property as rqp

valid_cell_shapes = ['polyhedral', 'tetrahedral', 'pyramidal', 'prism', 'hexahedral']  #: valid cell shapes


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
      self.cell_shape = cell_shape  #: the shape of cells within the grid
      self.crs_uuid = None  #: uuid of the coordinate reference system used by the grid's geometry
      self.points_cached = None  #: numpy array of raw points data; loaded on demand
      self.node_count = None  #: number of distinct points used in geometry; None if no geometry present
      self.face_count = None  #: number of distinct faces used in geometry; None if no geometry present
      self.nodes_per_face = None
      self.nodes_per_face_cl = None
      self.faces_per_cell = None
      self.cell_face_is_right_handed = None
      self.faces_per_cell_cl = None
      self.inactive = None  #: numpy boolean array indicating which cells are inactive in flow simulation
      self.all_inactive = None  #: boolean indicating whether all cells are inactive
      self.active_property_uuid = None  #: uuid of property holding active cell boolean array (used to populate inactive)
      self.grid_representation = 'UnstructuredGrid'  #: flavour of grid, 'UnstructuredGrid'; not much used
      self.geometry_root = None  #: xml node at root of geometry sub-tree, if present
      self.property_collection = None  #: collection of properties for which this grid is the supporting representation
      self.crs_is_right_handed = None  #: cached boolean indicating handedness of crs axes
      self.cells_per_face = None  #: numpy int array of shape (face_count, 2) holding cells for faces; -1 is null value

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
      self.cell_count = rqet.find_tag_int(grid_root, 'CellCount')
      assert self.cell_count > 0
      self.geometry_root = rqet.find_tag(grid_root, 'Geometry')
      if self.geometry_root is None:
         self.cell_shape = None
      else:
         self.extract_crs_uuid()
         self.cell_shape = rqet.find_tag_text(self.geometry_root, 'CellShape')
         assert self.cell_shape in valid_cell_shapes
         self.node_count = rqet.find_tag_int(self.geometry_root, 'NodeCount')
         assert self.node_count > 3
         self.face_count = rqet.find_tag_int(self.geometry_root, 'FaceCount')
         assert self.face_count > 3
      self.extract_inactive_mask()
      # note: geometry arrays not loaded until demanded; see cache_all_geometry_arrays()

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
                                     array_attribute = 'cell_face_is_right_handed',
                                     required_shape = (len(self.faces_per_cell),),
                                     dtype = 'bool')

   def _load_jagged_array(self, tag, main_attribute):
      # jagged arrays are used by RESQML to efficiantly pack arrays of lists of numbers
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

   def extract_crs_uuid(self):
      """Returns uuid for coordinate reference system, as stored in geometry xml tree.

      returns:
         uuid.UUID object
      """

      if self.crs_uuid is not None:
         return self.crs_uuid
      if self.geometry_root is None:
         return None
      uuid_str = rqet.find_nested_tags_text(self.geometry_root, ['LocalCrs', 'UUID'])
      if uuid_str:
         self.crs_uuid = bu.uuid_from_string(uuid_str)
         self._set_crs_handedness()
      return self.crs_uuid

   def extract_inactive_mask(self):
      """Returns boolean numpy array indicating which cells are inactive, if (in)active property found for this grid.

      returns:
         numpy array of booleans, of shape (cell_count,) being True for cells which are inactive; False for active

      note:
         RESQML does not have a built-in concept of inactive (dead) cells, though the usage guide advises to use a
         discrete property with a local property kind of 'active'; this resqpy code can maintain an 'inactive'
         attribute for the grid object, which is a boolean numpy array indicating which cells are inactive
      """

      if self.inactive is not None:
         return self.inactive
      self.inactive = np.zeros((self.cell_count,), dtype = bool)  # ie. all active
      self.all_inactive = False
      gpc = self.extract_property_collection()
      if gpc is None:
         return self.inactive
      active_gpc = rqp.PropertyCollection()
      # note: use of bespoke (local) property kind 'active' as suggested in resqml usage guide
      active_gpc.inherit_parts_selectively_from_other_collection(other = gpc,
                                                                 property_kind = 'active',
                                                                 continuous = False)
      if active_gpc.number_of_parts() > 0:
         if active_gpc.number_of_parts() > 1:
            log.warning('more than one property found with bespoke kind "active", using last encountered')
         active_part = active_gpc.parts()[-1]
         active_array = active_gpc.cached_part_array_ref(active_part, dtype = 'bool')
         self.inactive = np.logical_not(active_array)
         self.active_property_uuid = active_gpc.uuid_for_part(active_part)
         active_gpc.uncache_part_array(active_part)
         self.all_inactive = np.all(self.inactive)

      return self.inactive

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
      self.property_collection = rqp.PropertyCollection(support = self)
      return self.property_collection

   def set_cells_per_face(self, check_all_faces_used = True):
      """Sets and returns the cells_per_face array showing which cells are using each face.

      arguments:
         check_all_faces_used (boolean, default True): if True, an assertion error is raised if there are any
            faces which do not appear in any cells

      returns:
         numpy int array of shape (face_count, 2) showing upto 2 cell indices for each face index; -1 is a
         null value; if only one cell uses a face, its index is always in position 0 of the second axis
      """

      if self.cells_per_face is not None:
         return self.cells_per_face
      assert self.face_count is not None
      self.cells_per_face = -np.ones((self.face_count, 2), dtype = int)  # -1 is used here as a null value
      for cell in range(self.cell_count):
         for face in self.face_indices_for_cell(cell):
            if self.cells_per_face[face, 0] == -1:
               self.cells_per_face[face, 0] = cell
            else:
               assert self.cells_per_face[face, 1] == -1, f'more than two cells use face with index {face}'
               self.cells_per_face[face, 1] = cell
      if check_all_faces_used:
         assert np.all(self.cells_per_face[:, 0] >= 0), 'not all faces used by cells'
      return self.cells_per_face

   def masked_cells_per_face(self, exclude_cell_mask):
      """Sets and returns the cells_per_face array showing which cells are using each face.

      arguments:
         exclude_cell_mask (numpy bool array of shape (cell_count,)): cells with a value True in this array
         are excluded when populating the result

      returns:
         numpy int array of shape (face_count, 2) showing upto 2 cell indices for each face index; -1 is a
         null value; if only one cell uses a face, its index is always in position 0 of the second axis

      note:
         this method recomputes the result on every call - nothing extra is cached in the grid
      """

      assert self.face_count is not None
      result = -np.ones((self.face_count, 2), dtype = int)  # -1 is used here as a null value
      for cell in range(self.cell_count):
         if exclude_cell_mask[cell]:
            continue
         for face in self.face_indices_for_cell(cell):
            if result[face, 0] == -1:
               result[face, 0] = cell
            else:
               assert result[face, 1] == -1, f'more than two cells use face with index {face}'
               result[face, 1] = cell
      return result

   def external_face_indices(self):
      """Returns a numpy int vector listing the indices of faces which are only used by one cell.

      note:
         resulting array is ordered by face index
      """

      self.set_cells_per_face()
      return np.where(self.cells_per_face[:, 1] == -1)[0]

   def external_face_indices_for_masked_cells(self, exclude_cell_mask):
      """Returns a numpy int vector listing the indices of faces which are used by exactly one of masked cells.

      arguments:
         exclude_cell_mask (numpy bool array of shape (cell_count,)): cells with a value True in this array
         are excluded when populating the result

      note:
         resulting array is ordered by face index
      """

      cpf = self.masked_cells_per_face(exclude_cell_mask)
      return np.where(np.logical_and(cpf[:, 0] >= 0, cpf[:, 1] == -1))[0]

   def internal_face_indices_for_masked_cells(self, exclude_cell_mask):
      """Returns a numpy int vector listing the indices of faces which are used by two of masked cells.

      arguments:
         exclude_cell_mask (numpy bool array of shape (cell_count,)): cells with a value True in this array
         are excluded when populating the result

      note:
         resulting array is ordered by face index
      """

      cpf = self.masked_cells_per_face(exclude_cell_mask)
      return np.where(np.logical_and(cpf[:, 0] >= 0, cpf[:, 1] >= 0))[0]

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

   def face_centre_point(self, face_index):
      """Returns a nominal centre point for a single face calculated as the mean position of its nodes.

      arguments:
         face_index (int): the index of the face (as used in faces_per_cell and implicitly in nodes_per_face)

      returns:
         numpy float array of shape (3,) being the xyz location of the centre point of the face

      note:
         this returns a nominal centre point for a face - the mean position of its nodes - which is not generally
         its barycentre
      """

      return np.mean(self.points_cached[self.node_indices_for_face(face_index)], axis = 0)

   def face_count_for_cell(self, cell):
      """Returns the number of faces for a particular cell."""

      self.cache_all_geometry_arrays()
      start = 0 if cell == 0 else self.faces_per_cell_cl[cell - 1]
      return self.faces_per_cell_cl[cell] - start

   def max_face_count_for_any_cell(self):
      """Returns the largest number of faces in use by any one cell."""

      self.cache_all_geometry_arrays()
      return max(self.faces_per_cell_cl[0], np.max(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1]))

   def node_count_for_face(self, face_index):
      """Returns the number of nodes for a particular face."""

      self.cache_all_geometry_arrays()
      start = 0 if face_index == 0 else self.nodes_per_face_cl[face_index - 1]
      return self.nodes_per_face_cl[face_index] - start

   def max_node_count_for_any_face(self):
      """Returns the largest number of nodes in use by any one face."""

      self.cache_all_geometry_arrays()
      return max(self.nodes_per_face_cl[0], np.max(self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1]))

   def node_indices_for_face(self, face_index):
      """Returns numpy list of node indices for a single face.

      arguments:
         face_index (int): the index of the face (as used in faces_per_cell and implicitly in nodes_per_face)

      returns:
         numpy int array of shape (N,) being the node indices identifying the vertices of the face

      note:
         the node indices are used to index the points data (points_cached attribute);
         ordering of returned nodes is clockwise or anticlockwise when viewed from within the cell,
         as indicated by the entry in the cell_face_is_right_handed array
      """

      self.cache_all_geometry_arrays()
      start = 0 if face_index == 0 else self.nodes_per_face_cl[face_index - 1]
      return self.nodes_per_face[start:self.nodes_per_face_cl[face_index]].copy()

   def distinct_node_indices_for_cell(self, cell):
      """Returns a numpy list of distinct node indices used by the faces of a single cell.

      arguments:
         cell (int): the index of the cell for which distinct node indices are required

      returns:
         numpy int array of shape (N, ) being the indices of N distinct nodes used by the cell's faces

      note:
         the returned array is sorted by increasing node index
      """

      face_indices = self.face_indices_for_cell(cell)
      node_set = self.node_indices_for_face(face_indices[0])
      for face_index in face_indices[1:]:
         node_set = np.union1d(node_set, self.node_indices_for_face(face_index))
      return node_set

   def face_indices_for_cell(self, cell):
      """Returns numpy list of face indices for a single cell.

      arguments:
         cell (int): the index of the cell for which face indices are required

      returns:
         numpy int array of shape (F,) being the face indices of each of the F faces for the cell

      note:
         the face indices are used when accessing the nodes per face data and can also be used to identify
         shared faces
      """

      self.cache_all_geometry_arrays()
      start = 0 if cell == 0 else self.faces_per_cell_cl[cell - 1]
      return self.faces_per_cell[start:self.faces_per_cell_cl[cell]].copy()

   def face_indices_and_handedness_for_cell(self, cell):
      """Returns numpy list of face indices for a single cell, and numpy boolean list of face right handedness.

      arguments:
         cell (int): the index of the cell for which face indices are required

      returns:
         numpy int array of shape (F,), numpy boolean array of shape (F, ):
         being the face indices of each of the F faces for the cell, and the right handedness (clockwise order)
         of the face nodes when viewed from within the cell

      note:
         the face indices are used when accessing the nodes per face data and can also be used to identify
         shared faces; the handedness (clockwise or anti-clockwise ordering of nodes) is significant for
         some processing of geometry such as volume calculations
      """

      self.cache_all_geometry_arrays()
      start = 0 if cell == 0 else self.faces_per_cell_cl[cell - 1]
      return (self.faces_per_cell[start:self.faces_per_cell_cl[cell]].copy(),
              self.cell_face_is_right_handed[start:self.faces_per_cell_cl[cell]].copy())

   def edges_for_face(self, face_index):
      """Returns numpy list of pairs of node indices, each pair being one edge of the face.

      arguments:
         face_index (int): the index of the face (as used in faces_per_cell and implicitly in nodes_per_face)

      returns:
         numpy int array of shape (N, 2) being the node indices identifying the N edges of the face

      notes:
         the order of the pairs follows the order of the nodes for the face; within each pair, the
         order of the two node indices also follows the order of the nodes for the face
      """

      face_nodes = self.node_indices_for_face(face_index)
      return np.array([(face_nodes[i - 1], face_nodes[i]) for i in range(len(face_nodes))], dtype = int)

   def edges_for_face_with_node_indices_ordered_within_pairs(self, face_index):
      """Returns numpy list of pairs of node indices, each pair being one edge of the face.

      arguments:
         face_index (int): the index of the face (as used in faces_per_cell and implicitly in nodes_per_face)

      returns:
         numpy int array of shape (N, 2) being the node indices identifying the N edges of the face

      notes:
         the order of the pairs follows the order of the nodes for the face; within each pair, the
         two node indices are ordered with the lower index first
      """

      edges = self.edges_for_face(face_index)
      for i in range(len(edges)):
         a, b = edges[i]
         if b < a:
            edges[i] = (b, a)
      return edges

   def distinct_edges_for_cell(self, cell):
      """Returns numpy list of pairs of node indices, each pair being one distinct edge of the cell.

      arguments:
         cell (int): the index of the cell

      returns:
         numpy int array of shape (E, 2) being the node indices identifying the E edges of the cell

      note:
         within each pair, the two node indices are ordered with the lower index first
      """

      edge_list = []
      for face_index in self.face_indices_for_cell(cell):
         for a, b in self.edges_for_face(face_index):
            if b < a:
               a, b = b, a
            if (a, b) not in edge_list:
               edge_list.append((a, b))
      return np.array(edge_list, dtype = int)

   def cell_face_centre_points(self, cell):
      """Returns a numpy array of centre points of the faces for a single cell.

      arguments:
         cell (int): the index of the cell for which face centre points are required

      returns:
         numpy array of shape (F, 3) being the xyz location of each of the F faces for the cell

      notes:
         the order of the returned face centre points matches the faces_per_cell for the cell;
         the returned values are nominal centre points for the faces - the mean position of their nodes - which
         are not generally their barycentres
      """

      face_indices = self.face_indices_for_cell(cell)
      face_centres = np.empty((len(face_indices), 3))
      for fi, face_index in enumerate(face_indices):  # todo: vectorise?
         face_centres[fi] = self.face_centre_point(face_index)
      return face_centres

   def face_normal(self, face_index):
      """Returns a unit vector normal to a planar approximation of the face.

      arguments:
         face_index (int): the index of the face (as used in faces_per_cell and implicitly in nodes_per_face)

      returns:
         numpy float array of shape (3,) being the xyz components of a unit length vector normal to the face

      note:
         in the case of a degenerate face, a zero length vector is returned;
         the direction of the normal will be into or out of the cell depending on the handedness of the
         cell face
      """

      self.cache_all_geometry_arrays()
      vertices = self.points_cached[self.node_indices_for_face(face_index)]
      centre = self.face_centre_point(face_index)
      normal_sum = np.zeros(3)

      for e in range(len(vertices)):
         edge = vertices[e] - vertices[e - 1]
         radial = centre - 0.5 * (vertices[e] + vertices[e - 1])
         weight = vec.naive_length(edge)
         if weight == 0.0:
            continue
         edge_normal = vec.unit_vector(vec.cross_product(edge, radial))
         normal_sum += weight * edge_normal

      return vec.unit_vector(normal_sum)

   def planar_face_points(self, face_index, xy_plane = False):
      """Returns points for a planar approximation of a face.

      arguments:
         face_index (int): the index of the face for which a planar approximation is required
         xy_plane (boolean, default False): if True, the returned points lie in a horizontal plane with z = 0.0;
            if False, the plane is located approximately in the position of the original face, with the same
            normal direction

      returns:
         numpy float array of shape (N, 3) being the xyz points of the planar face nodes corresponding to the
         N nodes of the original face, in the same order
      """

      self.cache_all_geometry_arrays()
      face_centre = self.face_centre_point(face_index)
      assert np.all(self.node_indices_for_face(face_index) >= 0)  # debug
      assert np.all(self.node_indices_for_face(face_index) < self.node_count)  # debug
      face_points = self.points_cached[self.node_indices_for_face(face_index), :].copy()
      normal = self.face_normal(face_index)
      az = vec.azimuth(normal)
      incl = vec.inclination(normal)
      vec.tilt_points(face_centre, az, -incl, face_points)  # modifies face_points in situ
      if xy_plane:
         face_points[..., 2] = 0.0
      else:
         face_points[..., 2] = face_centre[2]
         vec.tilt_points(face_centre, az, incl, face_points)  # tilt back to original average normal
      return face_points

   def face_triangulation(self, face_index, local_nodes = False):
      """Returns a Delauney triangulation of (a planar approximation of) a face.

      arguments:
         face_index (int): the index of the face for which a triangulation is required
         local_nodes (boolean, default False): if True, the returned node indices are local to the face nodes,
            ie. can index into node_indices_for_face(); if False, the returned node indices are the global
            node indices in use by the grid

      returns:
         numpy int array of shape (N - 2, 3) being the node indices of the triangulation, where N is the number
         of nodes defining the face
      """

      face_points = self.planar_face_points(face_index, xy_plane = True)
      local_triangulation = tri.dt(face_points)  # returns int array of shape (M, 3)
      assert len(local_triangulation) == len(face_points) - 2, 'face triangulation failure (concave edges when planar?)'
      if local_nodes:
         return local_triangulation
      return self.node_indices_for_face(face_index)[local_triangulation]

   def area_of_face(self, face_index, in_plane = False):
      """Returns the area of a face.

      arguments:
         face_index (int): the index of the face for which the area is required
         in_plane (boolean, default False): if True, the area returned is the area of the planar approximation
            of the face; if False, the area is the sum of the areas of the triangulation of the face, which
            need not be planar

      returns:
         float being the area of the face

      notes:
         units of measure of the area is implied by the units of the crs in use by the grid
      """

      if in_plane:
         face_points = self.planar_face_points(face_index, xy_plane = True)
         local_triangulation = tri.dt(face_points)
         triangulated_points = face_points[local_triangulation]
      else:
         global_triangulation = self.face_triangulation(face_index)
         triangulated_points = self.points_cached[global_triangulation]
      assert triangulated_points.ndim == 3 and triangulated_points.shape[1:] == (3, 3)
      area = 0.0
      for tp in triangulated_points:
         area += vec.area_of_triangle(tp[0], tp[1], tp[2])
      return area

   def cell_centre_point(self, cell):
      """Returns centre point of a single cell calculated as the mean position of the centre points of its faces.

      arguments:
         cell (int): the index of the cell for which the centre point is required

      returns:
         numpy float array of shape (3,) being the xyz location of the centre point of the cell

      note:
         this is a nominal centre point - the mean of the nominal face centres - which is not generally
         the barycentre of the cell
      """

      return np.mean(self.cell_face_centre_points(cell), axis = 0)

   def centre_point(self, cell = None, cache_centre_array = False):
      """Returns centre point of a cell or array of centre points of all cells; optionally cache centre points for all cells.

      arguments:
         cell (optional): if present, the cell number of the individual cell for which the
            centre point is required; zero based indexing
         cache_centre_array (boolean, default False): If True, or cell is None, an array of centre points
            is generated and added as an attribute of the grid, with attribute name array_centre_point

      returns:
         (x, y, z) 3 element numpy array of floats holding centre point of a single cell;
         or numpy 2D array of shape (cell_count, 3) if cell is None

      notes:
         a simple mean of the nominal centres of the faces is used to calculate the centre point of a cell;
         this is not generally the barycentre of the cell;
         resulting coordinates are in the same (local) crs as the grid points

      :meta common:
      """

      if cell is None:
         cache_centre_array = True

      if hasattr(self, 'array_centre_point') and self.array_centre_point is not None:
         if cell is None:
            return self.array_centre_point
         return self.array_centre_point[cell]  # could check for nan here and return None

      if self.node_count is None:  # no geometry present
         return None

      if cache_centre_array:  # calculate for all cells and cache
         self.array_centre_point = np.empty((self.cell_count, 3))
         for cell_index in range(self.cell_count):  # todo: vectorise
            self.array_centre_point[cell_index] = self.cell_centre_point(cell_index)
         if cell is None:
            return self.array_centre_point
         else:
            return self.array_centre_point[cell]

      else:
         return self.cell_centre_point(cell)

   def volume(self, cell):
      """Returns the volume of a single cell.

      arguments:
         cell (int): the index of the cell for which the volume is required

      returns:
         float being the volume of the cell; units of measure is implied by crs units

      note:
         this is a computationally expensive method
      """

      tetra = TetraGrid.from_unstructured_cell(self, cell, set_handedness = False)
      assert tetra is not None
      return tetra.grid_volume()

   def check_indices(self):
      """Asserts that all node and face indices are within range."""

      self.cache_all_geometry_arrays()
      assert self.cell_count > 0
      assert self.face_count >= 4
      assert self.node_count >= 4
      assert np.all(self.faces_per_cell >= 0) and np.all(self.faces_per_cell < self.face_count)
      assert self.faces_per_cell_cl[0] >= 4
      assert np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] >= 4)
      assert len(self.faces_per_cell_cl) == self.cell_count
      assert np.all(self.nodes_per_face >= 0) and np.all(self.nodes_per_face < self.node_count)
      assert self.nodes_per_face_cl[0] >= 3
      assert np.all(self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1] >= 3)
      assert len(self.nodes_per_face_cl) == self.face_count

   def write_hdf5(self, file = None, geometry = True, imported_properties = None, write_active = None):
      """Write to an hdf5 file the datasets for the grid geometry and optionally properties from cached arrays.

      :meta common:
      """

      # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
      # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
      # xml is not created here for property objects

      if geometry:
         assert self.node_count > 0 and self.face_count > 0, 'geometry not present when writing unstructured grid to hdf5'

      if write_active is None:
         write_active = geometry

      self.cache_all_geometry_arrays()

      if not file:
         file = self.model.h5_file_name()
      h5_reg = rwh5.H5Register(self.model)

      if geometry:
         h5_reg.register_dataset(self.uuid, 'Points', self.points_cached)
         h5_reg.register_dataset(self.uuid, 'NodesPerFace/elements', self.nodes_per_face, dtype = 'uint32')
         h5_reg.register_dataset(self.uuid, 'NodesPerFace/cumulativeLength', self.nodes_per_face_cl, dtype = 'uint32')
         h5_reg.register_dataset(self.uuid, 'FacesPerCell/elements', self.faces_per_cell, dtype = 'uint32')
         h5_reg.register_dataset(self.uuid, 'FacesPerCell/cumulativeLength', self.faces_per_cell_cl, dtype = 'uint32')
         h5_reg.register_dataset(self.uuid, 'CellFaceIsRightHanded', self.cell_face_is_right_handed, dtype = 'uint8')

      if write_active and self.inactive is not None:
         if imported_properties is None:
            imported_properties = rqp.PropertyCollection()
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

      h5_reg.write(file, mode = 'a')

   def create_xml(self,
                  ext_uuid = None,
                  add_as_part = True,
                  add_relationships = True,
                  title = None,
                  originator = None,
                  write_active = True,
                  write_geometry = True,
                  extra_metadata = {}):
      """Creates an unstructured grid node and optionally adds as a part in the model.

      arguments:
         ext_uuid (uuid.UUID, optional): the uuid of the hdf5 external part holding the array data for the grid geometry
         add_as_part (boolean, default True): if True, the newly created xml node is added as a part
            in the model
         add_relationships (boolean, default True): if True, relationship xml parts are created relating the
            new grid part to: the crs, and the hdf5 external part
         title (string): used as the citation title text; careful consideration should be given
            to this argument when dealing with multiple grids in one model, as it is the means by which a
            human will distinguish them
         originator (string, optional): the name of the human being who created the unstructured grid part;
            default is to use the login name
         write_active (boolean, default True): if True, xml for an active cell property is also generated, but
            only if the active_property_uuid is set and no part exists in the model for that uuid
         write_geometry (boolean, default True): if False, the geometry node is omitted from the xml
         extra_metadata (dict): any key value pairs in this dictionary are added as extra metadata xml nodes

      returns:
         the newly created unstructured grid xml node

      notes:
         the write_active argument should generally be set to the same value as that passed to the write_hdf5... method;
         the RESQML standard allows the geometry to be omitted for a grid, controlled here by the write_geometry argument;
         the explicit geometry may be omitted for unstructured grids, in which case the arrays should not be written to
         the hdf5 file either

      :meta common:
      """

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()
      if title:
         self.title = title
      if not self.title:
         self.title = 'ROOT'

      ug = super().create_xml(add_as_part = False, originator = originator, extra_metadata = extra_metadata)

      if self.grid_representation and not write_geometry:
         rqet.create_metadata_xml(node = ug, extra_metadata = {'grid_flavour': self.grid_representation})

      cc_node = rqet.SubElement(ug, ns['resqml2'] + 'CellCount')
      cc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      cc_node.text = str(self.cell_count)

      if write_geometry:

         geom = rqet.SubElement(ug, ns['resqml2'] + 'Geometry')
         geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'UnstructuredGridGeometry')
         geom.text = '\n'

         # the remainder of this function is populating the geometry node
         self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

         points_node = rqet.SubElement(geom, ns['resqml2'] + 'Points')
         points_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
         points_node.text = '\n'

         coords = rqet.SubElement(points_node, ns['resqml2'] + 'Coordinates')
         coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         coords.text = '\n'

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Points', root = coords)

         shape_node = rqet.SubElement(geom, ns['resqml2'] + 'CellShape')
         shape_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'CellShape')
         shape_node.text = self.cell_shape

         nc_node = rqet.SubElement(geom, ns['resqml2'] + 'NodeCount')
         nc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         nc_node.text = str(self.node_count)

         fc_node = rqet.SubElement(geom, ns['resqml2'] + 'FaceCount')
         fc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         fc_node.text = str(self.face_count)

         self._create_jagged_array_xml(geom, 'NodesPerFace', ext_uuid)

         self._create_jagged_array_xml(geom, 'FacesPerCell', ext_uuid)

         cfirh_node = rqet.SubElement(geom, ns['resqml2'] + 'CellFaceIsRightHanded')
         cfirh_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
         cfirh_node.text = '\n'

         cfirh_values = rqet.SubElement(cfirh_node, ns['resqml2'] + 'Values')
         cfirh_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         cfirh_values.text = '\n'

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'CellFaceIsRightHanded', root = cfirh_values)

         self.geometry_root = geom

      if add_as_part:
         self.model.add_part('obj_UnstructuredGridRepresentation', self.uuid, ug)
         if add_relationships:
            if write_geometry:
               # create 2 way relationship between UnstructuredGrid and Crs
               self.model.create_reciprocal_relationship(ug, 'destinationObject', self.model.root(uuid = self.crs_uuid),
                                                         'sourceObject')
               # create 2 way relationship between UnstructuredGrid and Ext
               ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
               ext_node = self.model.root_for_part(ext_part)
               self.model.create_reciprocal_relationship(ug, 'mlToExternalPartProxy', ext_node, 'externalPartProxyToMl')

      if write_active and self.active_property_uuid is not None and self.model.part(
            uuid = self.active_property_uuid) is None:
         active_collection = rqp.PropertyCollection()
         active_collection.set_support(support = self)
         active_collection.create_xml(None,
                                      None,
                                      'ACTIVE',
                                      'active',
                                      p_uuid = self.active_property_uuid,
                                      discrete = True,
                                      add_min_max = False,
                                      find_local_property_kinds = True)

      return ug

   def _create_jagged_array_xml(self, parent_node, tag, ext_uuid, null_value = -1):

      j_node = rqet.SubElement(parent_node, ns['resqml2'] + tag)
      j_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlJaggedArray')
      j_node.text = '\n'

      elements = rqet.SubElement(j_node, ns['resqml2'] + 'Elements')
      elements.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
      elements.text = '\n'

      el_null = rqet.SubElement(elements, ns['resqml2'] + 'NullValue')
      el_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
      el_null.text = str(null_value)

      el_values = rqet.SubElement(elements, ns['resqml2'] + 'Values')
      el_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      el_values.text = '\n'

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, tag + '/elements', root = el_values)

      c_length = rqet.SubElement(j_node, ns['resqml2'] + 'CumulativeLength')
      c_length.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
      c_length.text = '\n'

      cl_null = rqet.SubElement(c_length, ns['resqml2'] + 'NullValue')
      cl_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
      cl_null.text = '0'

      cl_values = rqet.SubElement(c_length, ns['resqml2'] + 'Values')
      cl_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      cl_values.text = '\n'

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, tag + '/cumulativeLength', root = cl_values)

   def _set_crs_handedness(self):
      if self.crs_is_right_handed is not None:
         return
      assert self.crs_uuid is not None
      crs = rqc.Crs(self.model, uuid = self.crs_uuid)
      self.crs_is_right_handed = crs.is_right_handed_xyz()


class TetraGrid(UnstructuredGrid):
   """Class for unstructured grids where every cell is a tetrahedron."""

   def __init__(self,
                parent_model,
                uuid = None,
                find_properties = True,
                cache_geometry = False,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Creates a new resqpy TetraGrid object (RESQML UnstructuredGrid with cell shape tetrahedral)

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         find_properties (boolean, default True): if True and uuid is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
            are loaded into attributes of the new grid object
         title (str, optional): citation title for new grid; ignored if uuid is present
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if uuid is present
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if uuid is present

      returns:
         a newly created TetraGrid object
      """

      super().__init__(parent_model = parent_model,
                       uuid = uuid,
                       find_properties = find_properties,
                       geometry_required = True,
                       cache_geometry = cache_geometry,
                       cell_shape = 'tetrahedral',
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      if self.root is not None:
         assert grr.grid_flavour(self.root) == 'TetraGrid'
         self.check_tetra()

      self.grid_representation = 'TetraGrid'  #: flavour of grid; not much used

   def check_tetra(self):
      """Checks that each cell has 4 faces and each face has 3 nodes."""

      assert self.cell_shape == 'tetrahedral'
      self.cache_all_geometry_arrays()
      assert self.faces_per_cell_cl is not None and self.nodes_per_face_cl is not None
      assert self.faces_per_cell_cl[0] == 4 and np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] == 4)
      assert self.nodes_per_face_cl[0] == 3 and np.all(self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1] == 3)

   def face_centre_point(self, face_index):
      """Returns a nominal centre point for a single face calculated as the mean position of its nodes.

      note:
         this is a nominal centre point for a face and not generally its barycentre
      """

      self.cache_all_geometry_arrays()
      start = 0 if face_index == 0 else self.nodes_per_face_cl[face_index - 1]
      return np.mean(self.points_cached[self.nodes_per_face[start:start + 3]], axis = 0)

   def volume(self, cell):
      """Returns the volume of a single cell.

      arguments:
         cell (int): the index of the cell for which the volume is required

      returns:
         float being the volume of the tetrahedral cell; units of measure is implied by crs units
      """

      self.cache_all_geometry_arrays()
      abcd = self.points_cached[self.distinct_node_indices_for_cell(cell)]
      assert abcd.shape == (4, 3)
      return vol.tetrahedron_volume(abcd[0], abcd[1], abcd[2], abcd[3])

   def grid_volume(self):
      """Returns the sum of the volumes of all the cells in the grid.

      returns:
         float being the total volume of the grid; units of measure is implied by crs units
      """

      v = 0.0
      for cell in range(self.cell_count):
         v += self.volume(cell)
      return v

   @classmethod
   def from_unstructured_cell(cls, u_grid, cell, title = None, extra_metadata = {}, set_handedness = False):
      """Instantiates a small TetraGrid representing a single cell from an UnstructuredGrid as a set of tetrahedra."""

      def _min_max(a, b):
         if a < b:
            return (a, b)
         else:
            return (b, a)

      if not title:
         title = str(u_grid.title) + f'_cell_{cell}'

      assert u_grid.cell_shape in valid_cell_shapes
      u_grid.cache_all_geometry_arrays()
      u_cell_faces = u_grid.face_indices_for_cell(cell)
      u_cell_nodes = u_grid.distinct_node_indices_for_cell(cell)

      # create an empty TetreGrid
      tetra = cls(u_grid.model, title = title, extra_metadata = extra_metadata)
      tetra.crs_uuid = u_grid.crs_uuid

      u_cell_node_count = len(u_cell_nodes)
      assert u_cell_node_count >= 4
      u_cell_face_count = len(u_cell_faces)
      assert u_cell_face_count >= 4

      # build attributes, depending on the shape of the individual unstructured cell

      if u_cell_node_count == 4:  # cell is tetrahedral

         assert u_cell_face_count == 4
         tetra.set_cell_count(1)
         tetra.face_count = 4
         tetra.faces_per_cell_cl = np.array((4,), dtype = int)
         tetra.faces_per_cell = np.arange(4, dtype = int)
         tetra.node_count = 4
         tetra.nodes_per_face_cl = np.arange(3, 3 * 4 + 1, 3, dtype = int)
         tetra.nodes_per_face = np.array((0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0), dtype = int)
         tetra.cell_face_is_right_handed = np.ones(4, dtype = bool)
         tetra.points_cached = u_grid.points_cached[u_cell_nodes].copy()

      # todo: add optimised code for pyramidal (and hexahedral?) cells

      else:  # generic case: add a node at centre of unstructured cell and divide faces into triangles

         tetra.node_count = u_cell_node_count + 1
         tetra.points_cached = np.empty((tetra.node_count, 3))
         tetra.points_cached[:-1] = u_grid.points_cached[u_cell_nodes].copy()
         tetra.points_cached[-1] = u_grid.centre_point(cell = cell)
         centre_node = tetra.node_count - 1

         u_cell_nodes = list(u_cell_nodes)  # to allow simple index usage below

         # build list of distinct edges used by cell
         u_cell_edge_list = u_grid.distinct_edges_for_cell(cell)
         u_edge_count = len(u_cell_edge_list)
         assert u_edge_count >= 4

         t_cell_list = []  # list of 4-tuples of ints, being local face indices for tetra cells

         # create an internal tetra face for each edge, using centre point as third node
         # note: u_a, u_b are a sorted pair, and u_cell_nodes is also aorted, so t_a, t_b are a sorted pair
         t_face_list = []  # list of triple ints each triplet being local node indices for a triangular face
         for u_a, u_b in u_cell_edge_list:
            t_a, t_b = u_cell_nodes.index(u_a), u_cell_nodes.index(u_b)
            t_face_list.append((t_a, t_b, centre_node))

         # for each unstructured face, create a Delauney triangulation; create a tetra face for each
         # triangle in the triangulation; create internal tetra faces for each of the internal edges in
         # the triangulation; and create a tetra cell for each triangle in the triangulation
         # note: the resqpy Delauney triangulation is for a 2D system, so here the unstructured face
         # is projected onto a planar approximation defined by the face centre point and an average
         # normal vector
         for fi in u_cell_faces:
            triangulated_face = u_grid.face_triangulation(fi)
            for u_a, u_b, u_c in triangulated_face:
               t_a, t_b, t_c = u_cell_nodes.index(u_a), u_cell_nodes.index(u_b), u_cell_nodes.index(u_c)
               t_cell_faces = [len(t_face_list)]  # local face index for this triangle
               t_face_list.append((t_a, t_b, t_c))
               tri_edges = np.array([_min_max(t_a, t_b), _min_max(t_b, t_c), _min_max(t_c, t_a)], dtype = int)
               for e_a, e_b in tri_edges:
                  try:
                     pos = t_face_list.index((e_a, e_b, centre_node))
                  except ValueError:
                     pos = len(t_face_list)
                     t_face_list.append((e_a, e_b, centre_node))
                  t_cell_faces.append(pos)
               t_cell_list.append(tuple(t_cell_faces))

         # everything is now ready to populate the tetra grid attributes (apart from handedness)
         tetra.set_cell_count(len(t_cell_list))
         tetra.face_count = len(t_face_list)
         tetra.faces_per_cell_cl = np.arange(4, 4 * tetra.cell_count + 1, 4, dtype = int)
         tetra.faces_per_cell = np.array(t_cell_list, dtype = int).flatten()
         tetra.nodes_per_face_cl = np.arange(3, 3 * tetra.face_count + 1, 3, dtype = int)
         tetra.nodes_per_face = np.array(t_face_list, dtype = int).flatten()

         tetra.cell_face_is_right_handed = np.ones(len(tetra.faces_per_cell), dtype = bool)
         if set_handedness:
            # TODO: set handedness correctly and make default for set_handedness True
            raise NotImplementedError('code not written to set handedness for tetra from unstructured cell')

      tetra.check_tetra()

      return tetra

   # todo: add tetra specific method for centre_point()


class PyramidGrid(UnstructuredGrid):
   """Class for unstructured grids where every cell is a quadrilateral pyramid."""

   def __init__(self,
                parent_model,
                uuid = None,
                find_properties = True,
                cache_geometry = False,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Creates a new resqpy PyramidGrid object (RESQML UnstructuredGrid with cell shape pyramidal)

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         find_properties (boolean, default True): if True and uuid is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
            are loaded into attributes of the new grid object
         title (str, optional): citation title for new grid; ignored if uuid is present
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if uuid is present
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if uuid is present

      returns:
         a newly created PyramidGrid object
      """

      super().__init__(parent_model = parent_model,
                       uuid = uuid,
                       find_properties = find_properties,
                       geometry_required = True,
                       cache_geometry = cache_geometry,
                       cell_shape = 'pyramidal',
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      if self.root is not None:
         assert grr.grid_flavour(self.root) == 'PyramidGrid'
         self.check_pyramidal()

      self.grid_representation = 'PyramidGrid'  #: flavour of grid; not much used

   def check_pyramidal(self):
      """Checks that each cell has 5 faces and each face has 3 or 4 nodes.

      note:
         currently only performs a cursory check, without checking nodes are shared or that there is exactly one
         quadrilateral face
      """

      assert self.cell_shape == 'pyramidal'
      self.cache_all_geometry_arrays()
      assert self.faces_per_cell_cl is not None and self.nodes_per_face_cl is not None
      assert self.faces_per_cell_cl[0] == 5 and np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] == 5)
      nodes_per_face_count = np.empty(self.face_count)
      nodes_per_face_count[0] = self.nodes_per_face_cl[0]
      nodes_per_face_count[1:] = self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1]
      assert np.all(np.logical_or(nodes_per_face_count == 3, nodes_per_face_count == 4))

   def face_indices_for_cell(self, cell):
      """Returns numpy list of face indices for a single cell.

      arguments:
         cell (int): the index of the cell for which face indices are required

      returns:
         numpy int array of shape (5,) being the face indices of each of the 5 faces for the cell; the first
         index in the array is for the quadrilateral face

      note:
         the face indices are used when accessing the nodes per face data and can also be used to identify
         shared faces
      """

      faces = super().face_indices_for_cell(cell)
      assert len(faces) == 5
      result = -np.ones(5, dtype = int)
      i = 1
      for f in range(5):
         nc = self.node_count_for_face(faces[f])
         if nc == 3:
            assert i < 5, 'too many triangular faces for cell in pyramid grid'
            result[i] = faces[f]
            i += 1
         else:
            assert nc == 4, 'pyramid grid includes a face that is neither triangle nor quadrilateral'
            assert result[0] == -1, 'more than one quadrilateral face for cell in pyramid grid'
            result[0] = faces[f]
      return result

   def face_indices_and_handedness_for_cell(self, cell):
      """Returns numpy list of face indices for a single cell, and numpy boolean list of face right handedness.

      arguments:
         cell (int): the index of the cell for which face indices are required

      returns:
         numpy int array of shape (5,), numpy boolean array of shape (5, ):
         being the face indices of each of the 5 faces for the cell, and the right handedness (clockwise order)
         of the face nodes when viewed from within the cell; the first entry in each list is for the
         quadrilateral face

      note:
         the face indices are used when accessing the nodes per face data and can also be used to identify
         shared faces; the handedness (clockwise or anti-clockwise ordering of nodes) is significant for
         some processing of geometry such as volume calculations
      """

      faces, handednesses = super().face_indices_and_handedness_for_cell(cell)
      assert len(faces) == 5 and len(handednesses) == 5
      f_result = -np.ones(5, dtype = int)
      h_result = np.empty(5, dtype = bool)
      i = 1
      for f in range(5):
         nc = self.node_count_for_face(faces[f])
         if nc == 3:
            assert i < 5, 'too many triangular faces for cell in pyramid grid'
            f_result[i] = faces[f]
            h_result[i] = handednesses[f]
            i += 1
         else:
            assert nc == 4, 'pyramid grid includes a face that is neither triangle nor quadrilateral'
            assert f_result[0] == -1, 'more than one quadrilateral face for cell in pyramid grid'
            f_result[0] = faces[f]
            h_result[0] = handednesses[f]
      return f_result, h_result

   def volume(self, cell):
      """Returns the volume of a single cell.

      arguments:
         cell (int): the index of the cell for which the volume is required

      returns:
         float being the volume of the pyramidal cell; units of measure is implied by crs units
      """

      self._set_crs_handedness()
      self.cache_all_geometry_arrays()
      faces, hands = self.face_indices_and_handedness_for_cell(cell)
      nodes = self.distinct_node_indices_for_cell(cell)
      base_nodes = self.node_indices_for_face(faces[0])
      for node in nodes:
         if node not in base_nodes:
            apex_node = node
            break
      else:
         raise Exception('apex node not found for cell in pyramid grid')
      apex = self.points_cached[apex_node]
      abcd = self.points_cached[base_nodes]

      return vol.pyramid_volume(apex,
                                abcd[0],
                                abcd[1],
                                abcd[2],
                                abcd[3],
                                crs_is_right_handed = (self.crs_is_right_handed == hands[0]))

   # todo: add pyramidal specific method for centre_point()


class PrismGrid(UnstructuredGrid):
   """Class for unstructured grids where every cell is a triangular prism.

   note:
      prism cells are not constrained to have a fixed cross-section, though in practice they often will
   """

   def __init__(self,
                parent_model,
                uuid = None,
                find_properties = True,
                cache_geometry = False,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Creates a new resqpy PrismGrid object (RESQML UnstructuredGrid with cell shape trisngular prism)

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         find_properties (boolean, default True): if True and uuid is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
            are loaded into attributes of the new grid object
         title (str, optional): citation title for new grid; ignored if uuid is present
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if uuid is present
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if uuid is present

      returns:
         a newly created PrismGrid object
      """

      super().__init__(parent_model = parent_model,
                       uuid = uuid,
                       find_properties = find_properties,
                       geometry_required = True,
                       cache_geometry = cache_geometry,
                       cell_shape = 'prism',
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      if self.root is not None:
         assert grr.grid_flavour(self.root) in ['PrismGrid', 'VerticalPrismGrid']
         self.check_prism()

      self.grid_representation = 'PrismGrid'  #: flavour of grid; not much used

   def check_prism(self):
      """Checks that each cell has 5 faces and each face has 3 or 4 nodes.

      note:
         currently only performs a cursory check, without checking nodes are shared or that there are exactly two
         triangular faces without shared nodes
      """

      assert self.cell_shape == 'prism'
      self.cache_all_geometry_arrays()
      assert self.faces_per_cell_cl is not None and self.nodes_per_face_cl is not None
      assert self.faces_per_cell_cl[0] == 5 and np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] == 5)
      nodes_per_face_count = np.empty(self.face_count)
      nodes_per_face_count[0] = self.nodes_per_face_cl[0]
      nodes_per_face_count[1:] = self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1]
      assert np.all(np.logical_or(nodes_per_face_count == 3, nodes_per_face_count == 4))

   # todo: add prism specific methods for centre_point(), volume()


class VerticalPrismGrid(PrismGrid):
   """Class for unstructured grids where every cell is a vertical triangular prism.

   notes:
      vertical prism cells are constrained to have a fixed triangular cross-section, though top and base triangular
      faces need not be horizontal; edges not involved in the triangular faces must be vertical;
      this is not a native RESQML sub-class but is a resqpy concoction to allow optimisation of some methods;
      face ordering within a cell is also constrained to be top, base, then the three vertical planar quadrilateral
      faces; node ordering within triangular faces is constrained to ensure correspondence of nodes in triangles
      within a column
   """

   def __init__(self,
                parent_model,
                uuid = None,
                find_properties = True,
                cache_geometry = False,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Creates a new resqpy VerticalPrismGrid object.

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         find_properties (boolean, default True): if True and uuid is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
            are loaded into attributes of the new grid object
         title (str, optional): citation title for new grid; ignored if uuid is present
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if uuid is present
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if uuid is present

      returns:
         a newly created VerticalPrismGrid object
      """

      self.layer_count = None  #: number of layers when constructed as a layered grid

      super().__init__(parent_model = parent_model,
                       uuid = uuid,
                       find_properties = find_properties,
                       cache_geometry = cache_geometry,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      if self.root is not None:
         assert grr.grid_flavour(self.root) in ['VerticalPrismGrid', 'PrismGrid']
         self.check_prism()
         if 'layer count' in self.extra_metadata:
            self.layer_count = int(self.extra_metadata['layer count'])

      self.grid_representation = 'VerticalPrismGrid'  #: flavour of grid; not much used

   @classmethod
   def from_surfaces(cls,
                     parent_model,
                     surfaces,
                     title = None,
                     originator = None,
                     extra_metadata = {},
                     set_handedness = False):
      """Create a layered vertical prism grid from an ordered list of untorn surfaces.

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         surfaces (list of surface.Surface): list of two or more untorn surfaces ordered from
            shallowest to deepest; see notes
         title (str, optional): citation title for the new grid
         originator (str, optional): name of person creating the grid; defaults to login id
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid

      returns:
         a newly created VerticalPrismGrid object

      notes:
         this method will not work for torn (faulted) surfaces, nor for surfaces with recumbent folds;
         the surfaces may not cross each other, ie. the depth ordering must be consistent over the area;
         the first, shallowest, surface is used as a master and determines the triangular pattern of
         the columns; where a gravity vector from a node above does not intersect a surface, the
         point is inherited as a copy of the node above;
         the Surface class has methods for creating a Surface from a PointSet or a Mesh (RESQML
         Grid2dRepresentation), or for a horizontal plane;
         this class is represented in RESQML as an UnstructuredGridRepresentation – when a resqpy
         class is written for ColumnLayerGridRepresentation, a method will be added to that class to
         convert from a resqpy VerticalPrismGrid
      """

      def find_pair(a, pair):
         # for sorted array a of shape (N, 2) returns index in first axis of a pair

         def frp(a, pair, b, c):
            m = b + ((c - b) // 2)
            assert m < len(a), 'pair not found in sorted array'
            if np.all(a[m] == pair):
               return m
            assert c > b, 'pair not found in sorted array'
            if a[m, 0] < pair[0]:
               return frp(a, pair, m + 1, c)
            elif a[m, 0] > pair[0]:
               return frp(a, pair, b, m)
            elif a[m, 1] < pair[1]:
               return frp(a, pair, m + 1, c)
            else:
               return frp(a, pair, b, m)

         return frp(a, pair, 0, len(a))

      assert len(surfaces) > 1
      for s in surfaces:
         assert isinstance(s, rqs.Surface)

      vpg = cls(parent_model, title = title, originator = originator, extra_metadata = extra_metadata)
      assert vpg is not None

      top = surfaces[0]

      # set and check consistency of crs
      vpg.crs_uuid = top.crs_uuid
      for s in surfaces[1:]:
         if not bu.matching_uuids(vpg.crs_uuid, s.crs_uuid):
            # check for equivalence
            assert rqc.Crs(parent_model, uuid = vpg.crs_uuid) == rqc.Crs(parent_model,
                                                                         uuid = s.crs_uuid), 'mismatching surface crs'

      # fetch the data for the top surface, to be used as the master for the triangular pattern
      top_triangles, top_points = top.triangles_and_points()
      assert top_triangles.ndim == 2 and top_triangles.shape[1] == 3
      assert top_points.ndim == 2 and top_points.shape[1] == 3
      assert len(top_triangles) > 0
      bad_points = np.zeros(top_points.shape[0], dtype = bool)

      # setup size of arrays for the vertical prism grid
      column_count = top_triangles.shape[0]
      surface_count = len(surfaces)
      layer_count = surface_count - 1
      column_edges = top.distinct_edges()  # ordered pairs of node indices
      column_edge_count = len(column_edges)
      vpg.cell_count = column_count * layer_count
      vpg.node_count = len(top_points) * surface_count
      vpg.face_count = column_count * surface_count + column_edge_count * layer_count
      vpg.layer_count = layer_count
      if vpg.extra_metadata is None:
         vpg.extra_metadata = {}
      vpg.extra_metadata['layer count'] = vpg.layer_count

      # setup points with copies of points for top surface, deeper z values to be updated later
      points = np.empty((surface_count, top_points.shape[0], 3))
      points[:] = top_points

      # arrange faces with all triangles first, followed by the vertical quadrilaterals
      vpg.nodes_per_face_cl = np.zeros(vpg.face_count, dtype = int)
      vpg.nodes_per_face_cl[:column_count * surface_count] =  \
         np.arange(3, 3 * column_count * surface_count + 1, 3, dtype = int)
      quad_start = vpg.nodes_per_face_cl[column_count * surface_count - 1] + 4
      vpg.nodes_per_face_cl[column_count * surface_count:] =  \
         np.arange(quad_start, quad_start + 4 * column_edge_count * layer_count, 4)
      assert vpg.nodes_per_face_cl[-1] == 3 * column_count * surface_count + 4 * column_edge_count * layer_count
      # populate nodes per face for triangular faces
      vpg.nodes_per_face = np.zeros(vpg.nodes_per_face_cl[-1], dtype = int)
      for surface_index in range(surface_count):
         vpg.nodes_per_face[surface_index * 3 * column_count : (surface_index + 1) * 3 * column_count] =  \
            top_triangles.flatten() + surface_index * top_points.shape[0]
      # populate nodes per face for quadrilateral faces
      quad_nodes = np.empty((layer_count, column_edge_count, 2, 2), dtype = int)
      for layer in range(layer_count):
         quad_nodes[layer, :, 0, :] = column_edges + layer * top_points.shape[0]
         # reverse order of base pairs to maintain cyclic ordering of nodes per face
         quad_nodes[layer, :, 1, 0] = column_edges[:, 1] + (layer + 1) * top_points.shape[0]
         quad_nodes[layer, :, 1, 1] = column_edges[:, 0] + (layer + 1) * top_points.shape[0]
      vpg.nodes_per_face[3 * surface_count * column_count:] = quad_nodes.flatten()
      assert vpg.nodes_per_face[-1] > 0

      # set up faces per cell
      vpg.faces_per_cell = np.zeros(5 * vpg.cell_count)
      vpg.faces_per_cell_cl = np.arange(5, 5 * vpg.cell_count + 1, 5, dtype = int)
      assert len(vpg.faces_per_cell_cl) == vpg.cell_count
      # set cell top triangle indices
      for layer in range(layer_count):
         # top triangular faces of cells
         vpg.faces_per_cell[5 * layer * column_count : (layer + 1) * 5 * column_count : 5] =  \
            layer * column_count + np.arange(column_count)
         # base triangular faces of cells
         vpg.faces_per_cell[layer * 5 * column_count + 1: (layer + 1) * 5 * column_count : 5] =  \
            (layer + 1) * column_count + np.arange(column_count)
      # todo: some clever numpy indexing to irradicate the following for loop
      for col in range(column_count):
         t_nodes = top_triangles[col]
         for t_edge in range(3):
            a, b = t_nodes[t_edge - 1], t_nodes[t_edge]
            if b < a:
               a, b = b, a
            edge = find_pair(column_edges, (a, b))  # returns index into first axis of column edges
            # set quadrilateral faces of cells in column, for this edge
            vpg.faces_per_cell[5 * col + t_edge + 2 : 5 * vpg.cell_count : 5 * column_count] =  \
               np.arange(column_count * surface_count + edge, vpg.face_count, column_edge_count)
      # check full population of faces_per_cell (face zero is a top triangle, only used once)
      assert np.count_nonzero(vpg.faces_per_cell) == vpg.faces_per_cell.size - 1

      vpg.cell_face_is_right_handed = np.ones(len(vpg.faces_per_cell), dtype = bool)
      if set_handedness:
         # TODO: set handedness correctly and make default for set_handedness True
         raise NotImplementedError('code not written to set handedness for vertical prism grid from surfaces')

      # instersect gravity vectors from top surface points with other surfaces, and update z values in points
      gravity = np.zeros((top_points.shape[0], 3))
      gravity[:, 2] = 1.0  # up/down does not matter for the intersection function used below
      for layer in range(layer_count):
         base_triangles, base_points = surfaces[layer + 1].triangles_and_points()  # surface at base of layer
         intersects = meet.line_set_triangles_intersects(top_points, gravity, base_points[base_triangles])
         single_intersects = meet.last_intersects(intersects)  # will be triple NaN where no intersection occurs
         # inherit point from surface above where no intersection has occurred
         nan_lines = np.isnan(single_intersects[:, 0])
         single_intersects[nan_lines] = points[layer][nan_lines]
         # populate z values for layer of points
         points[layer + 1, :, 2] = single_intersects[:, 2]

      vpg.points_cached = points.reshape((-1, 3))
      assert np.all(vpg.nodes_per_face < len(vpg.points_cached))

      return vpg


class HexaGrid(UnstructuredGrid):
   """Class for unstructured grids where every cell is hexahedral (faces may be degenerate)."""

   def __init__(self,
                parent_model,
                uuid = None,
                find_properties = True,
                cache_geometry = False,
                title = None,
                originator = None,
                extra_metadata = {}):
      """Creates a new resqpy HexaGrid object (RESQML UnstructuredGrid with cell shape hexahedral)

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
         find_properties (boolean, default True): if True and uuid is present, a
            grid property collection is instantiated as an attribute, holding properties for which
            this grid is the supporting representation
         cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
            are loaded into attributes of the new grid object
         title (str, optional): citation title for new grid; ignored if uuid is present
         originator (str, optional): name of person creating the grid; defaults to login id;
            ignored if uuid is present
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
            ignored if uuid is present

      returns:
         a newly created HexaGrid object
      """

      super().__init__(parent_model = parent_model,
                       uuid = uuid,
                       find_properties = find_properties,
                       geometry_required = True,
                       cache_geometry = cache_geometry,
                       cell_shape = 'hexahedral',
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      if self.root is not None:
         assert grr.grid_flavour(self.root) == 'HexaGrid'
         self.check_hexahedral()

      self.grid_representation = 'HexaGrid'  #: flavour of grid; not much used

   @classmethod
   def from_unsplit_grid(cls,
                         parent_model,
                         grid_uuid,
                         inherit_properties = True,
                         title = None,
                         extra_metadata = {},
                         write_active = None):
      """Creates a new (unstructured) HexaGrid from an existing resqpy unsplit (IJK) Grid without K gaps.

      arguments:
         parent_model (model.Model object): the model which this grid is part of
         grid_uuid (uuid.UUID): the uuid of an IjkGridRepresentation from which the hexa grid will be created
         inherit_properties (boolean, default True): if True, properties will be created for the new grid
         title (str, optional): citation title for the new grid
         extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid
         write_active (boolean, optional): if True (or None and inactive property is established) then an
            active cell property is created (in addition to any inherited properties)

      returns:
         a newly created HexaGrid object

      note:
         this method includes the writing of hdf5 data, creation of xml for the new grid and adding it as a part
      """

      import resqpy.grid as grr

      # establish existing IJK grid
      ijk_grid = grr.Grid(parent_model, uuid = grid_uuid, find_properties = inherit_properties)
      assert ijk_grid is not None
      assert not ijk_grid.has_split_coordinate_lines, 'IJK grid has split coordinate lines (faults)'
      assert not ijk_grid.k_gaps, 'IJK grid has K gaps'
      ijk_grid.cache_all_geometry_arrays()
      ijk_points = ijk_grid.points_ref(masked = False)
      if title is None:
         title = ijk_grid.title

      # make empty unstructured hexa grid
      hexa_grid = cls(parent_model, title = title, extra_metadata = extra_metadata)

      # derive hexa grid attributes from ijk grid
      hexa_grid.crs_uuid = ijk_grid.crs_uuid
      hexa_grid.set_cell_count(ijk_grid.cell_count())
      if ijk_grid.inactive is not None:
         hexa_grid.inactive = ijk_grid.inactive.reshape((hexa_grid.cell_count,))
         hexa_grid.all_inactive = np.all(hexa_grid.inactive)
         if hexa_grid.all_inactive:
            log.warning(f'all cells marked as inactive for unstructured hexa grid {hexa_grid.title}')
      else:
         hexa_grid.all_inactive = False

      # inherit points (nodes) in IJK grid order, ie. K cycling fastest, then I, then J
      hexa_grid.points_cached = ijk_points.reshape((-1, 3))

      # setup faces per cell
      # ordering of faces (in nodes per face): all K faces, then all J faces, then all I faces
      # within J faces, ordering is all of J- faces for J = 0 first, then increasing planes in J
      # similarly for I faces
      nk_plus_1 = ijk_grid.nk + 1
      nj_plus_1 = ijk_grid.nj + 1
      ni_plus_1 = ijk_grid.ni + 1
      k_face_count = nk_plus_1 * ijk_grid.nj * ijk_grid.ni
      j_face_count = ijk_grid.nk * nj_plus_1 * ijk_grid.ni
      i_face_count = ijk_grid.nk * ijk_grid.nj * ni_plus_1
      kj_face_count = k_face_count + j_face_count
      hexa_grid.face_count = k_face_count + j_face_count + i_face_count
      hexa_grid.faces_per_cell_cl = 6 * (1 + np.arange(hexa_grid.cell_count, dtype = int))  # 6 faces per cell
      hexa_grid.faces_per_cell = np.empty(6 * hexa_grid.cell_count, dtype = int)
      arange = np.arange(hexa_grid.cell_count, dtype = int)
      hexa_grid.faces_per_cell[0::6] = arange  # K- faces
      hexa_grid.faces_per_cell[1::6] = ijk_grid.nj * ijk_grid.ni + arange  # K+ faces
      nki = ijk_grid.nk * ijk_grid.ni
      nkj = ijk_grid.nk * ijk_grid.nj
      # todo: vectorise following for loop
      for cell in range(hexa_grid.cell_count):
         k, j, i = ijk_grid.denaturalized_cell_index(cell)
         j_minus_face = k_face_count + nki * j + ijk_grid.ni * k + i
         hexa_grid.faces_per_cell[6 * cell + 2] = j_minus_face  # J- face
         hexa_grid.faces_per_cell[6 * cell + 3] = j_minus_face + nki  # J+ face
         i_minus_face = kj_face_count + nkj * i + ijk_grid.nj * k + j
         hexa_grid.faces_per_cell[6 * cell + 4] = i_minus_face  # I- face
         hexa_grid.faces_per_cell[6 * cell + 5] = i_minus_face + nkj  # I+ face

      # setup nodes per face, clockwise when viewed from negative side of face if ijk handedness matches xyz handedness
      # ordering of nodes in points array is as for the IJK grid
      hexa_grid.node_count = hexa_grid.points_cached.shape[0]
      assert hexa_grid.node_count == (ijk_grid.nk + 1) * (ijk_grid.nj + 1) * (ijk_grid.ni + 1)
      hexa_grid.nodes_per_face_cl = 4 * (1 + np.arange(hexa_grid.face_count, dtype = int))  # 4 nodes per face
      hexa_grid.nodes_per_face = np.empty(4 * hexa_grid.face_count, dtype = int)
      # todo: vectorise for loops
      # K faces
      face_base = 0
      for k in range(nk_plus_1):
         for j in range(ijk_grid.nj):
            for i in range(ijk_grid.ni):
               hexa_grid.nodes_per_face[face_base] = (k * nj_plus_1 + j) * ni_plus_1 + i  # ip 0, jp 0
               hexa_grid.nodes_per_face[face_base + 1] = (k * nj_plus_1 + j + 1) * ni_plus_1 + i  # ip 0, jp 1
               hexa_grid.nodes_per_face[face_base + 2] = (k * nj_plus_1 + j + 1) * ni_plus_1 + i + 1  # ip 1, jp 1
               hexa_grid.nodes_per_face[face_base + 3] = (k * nj_plus_1 + j) * ni_plus_1 + i + 1  # ip 1, jp 0
               face_base += 4
      # J faces
      assert face_base == 4 * k_face_count
      for j in range(nj_plus_1):
         for k in range(ijk_grid.nk):
            for i in range(ijk_grid.ni):
               hexa_grid.nodes_per_face[face_base] = (k * nj_plus_1 + j) * ni_plus_1 + i  # ip 0, kp 0
               hexa_grid.nodes_per_face[face_base + 1] = (k * nj_plus_1 + j) * ni_plus_1 + i + 1  # ip 1, kp 0
               hexa_grid.nodes_per_face[face_base + 2] = ((k + 1) * nj_plus_1 + j) * ni_plus_1 + i + 1  # ip 1, kp 1
               hexa_grid.nodes_per_face[face_base + 3] = ((k + 1) * nj_plus_1 + j) * ni_plus_1 + i  # ip 0, kp 1
               face_base += 4
      # I faces
      assert face_base == 4 * kj_face_count
      for i in range(ni_plus_1):
         for k in range(ijk_grid.nk):
            for j in range(ijk_grid.nj):
               hexa_grid.nodes_per_face[face_base] = (k * nj_plus_1 + j) * ni_plus_1 + i  # jp 0, kp 0
               hexa_grid.nodes_per_face[face_base + 1] = ((k + 1) * nj_plus_1 + j) * ni_plus_1 + i  # jp 0, kp 1
               hexa_grid.nodes_per_face[face_base + 2] = ((k + 1) * nj_plus_1 + j + 1) * ni_plus_1 + i  # jp 1, kp 1
               hexa_grid.nodes_per_face[face_base + 3] = (k * nj_plus_1 + j + 1) * ni_plus_1 + i  # jp 1, kp 0
               face_base += 4
      assert face_base == 4 * hexa_grid.face_count

      # set cell face is right handed
      # todo: check Energistics documents for meaning of cell face is right handed
      # here the assumption is clockwise ordering of nodes viewed from within cell means 'right handed'
      hexa_grid.cell_face_is_right_handed = np.zeros(6 * hexa_grid.cell_count,
                                                     dtype = bool)  # initially set to left handed
      # if IJK grid's ijk handedness matches the xyz handedness, then set +ve faces to right handed; else -ve faces
      if ijk_grid.off_handed():
         hexa_grid.cell_face_is_right_handed[0::2] = True  # negative faces are right handed
      else:
         hexa_grid.cell_face_is_right_handed[1::2] = True  # positive faces are right handed

      hexa_grid.write_hdf5(write_active = write_active)
      hexa_grid.create_xml(write_active = write_active)

      if inherit_properties:
         ijk_pc = ijk_grid.extract_property_collection()
         hexa_pc = rqp.PropertyCollection(support = hexa_grid)
         for part in ijk_pc.parts():
            count = ijk_pc.count_for_part(part)
            hexa_part_shape = (hexa_grid.cell_count,) if count == 1 else (hexa_grid.cell_count, count)
            hexa_pc.add_cached_array_to_imported_list(ijk_pc.cached_part_array_ref(part).reshape(hexa_part_shape),
                                                      'inherited from grid ' + str(ijk_grid.title),
                                                      ijk_pc.citation_title_for_part(part),
                                                      discrete = not ijk_pc.continuous_for_part(part),
                                                      uom = ijk_pc.uom_for_part(part),
                                                      time_index = ijk_pc.time_index_for_part(part),
                                                      null_value = ijk_pc.null_value_for_part(part),
                                                      property_kind = ijk_pc.property_kind_for_part(part),
                                                      local_property_kind_uuid = ijk_pc.local_property_kind_uuid(part),
                                                      facet_type = ijk_pc.facet_type_for_part(part),
                                                      facet = ijk_pc.facet_for_part(part),
                                                      realization = ijk_pc.realization_for_part(part),
                                                      indexable_element = ijk_pc.indexable_for_part(part),
                                                      count = count,
                                                      const_value = ijk_pc.constant_value_for_part(part))
            # todo: patch min & max values if present in ijk part
            hexa_pc.write_hdf5_for_imported_list()
            hexa_pc.create_xml_for_imported_list_and_add_parts_to_model(
               support_uuid = hexa_grid.uuid,
               time_series_uuid = ijk_pc.time_series_uuid_for_part(part),
               string_lookup_uuid = ijk_pc.string_lookup_uuid_for_part(part),
               extra_metadata = ijk_pc.extra_metadata_for_part(part))

      return hexa_grid

   def check_hexahedral(self):
      """Checks that each cell has 6 faces and each face has 4 nodes.

      notes:
         currently only performs a cursory check, without checking nodes are shared;
         assumes that degenerate faces still have four nodes identified
      """

      assert self.cell_shape == 'hexahedral'
      self.cache_all_geometry_arrays()
      assert self.faces_per_cell_cl is not None and self.nodes_per_face_cl is not None
      assert self.faces_per_cell_cl[0] == 6 and np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] == 6)
      assert self.nodes_per_face_cl[0] == 4 and np.all(self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1] == 4)

   def corner_points(self, cell):
      """Returns corner points (nodes) of a single cell.

      arguments:
         cell (int): the index of the cell for which the corner points are required

      returns:
         numpy float array of shape (8, 3) being the xyz points of 8 nodes defining a single hexahedral cell

      note:
         if this hexa grid has been created using the from_unsplit_grid class method, then the result can be
         reshaped to (2, 2, 2, 3) for corner points compatible with those used by the Grid class
      """

      self.cache_all_geometry_arrays()
      return self.points_cached[self.distinct_node_indices_for_cell(cell)]

   def volume(self, cell):
      """Returns the volume of a single cell.

      arguments:
         cell (int): the index of the cell for which the volume is required

      returns:
         float being the volume of the hexahedral cell; units of measure is implied by crs units
      """

      self._set_crs_handedness()
      apex = self.cell_centre_point(cell)
      v = 0.0
      faces, handednesses = self.face_indices_and_handedness_for_cell(cell)
      for face_index, handedness in zip(faces, handednesses):
         nodes = self.node_indices_for_face(face_index)
         abcd = self.points_cached[nodes]
         assert abcd.shape == (4, 3)
         v += vol.pyramid_volume(apex,
                                 abcd[0],
                                 abcd[1],
                                 abcd[2],
                                 abcd[3],
                                 crs_is_right_handed = (self.crs_is_right_handed == handedness))
      return v

   # todo: add hexahedral specific method for centre_point()?
   # todo: also add other methods equivalent to those in Grid class
