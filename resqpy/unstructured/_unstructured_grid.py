"""UnstructuredGrid class module."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rqc
import resqpy.olio.triangulation as tri
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
import resqpy.weights_and_measures as wam

from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns

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
        self.xyz_box_cached = None

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
           numpy array of shape (node_count, 3)

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

    def xyz_box(self):
        """Returns the minimum and maximum xyz for the grid geometry.

        returns:
           numpy array of float of shape (2, 3); the first axis is minimum, maximum; the second axis is x, y, z

        :meta common:
        """

        if self.xyz_box_cached is None:
            self.xyz_box_cached = np.empty((2, 3), dtype = float)
            p = self.points_ref()
            self.xyz_box_cached[0] = np.min(p, axis = 0)
            self.xyz_box_cached[1] = np.max(p, axis = 0)

        return self.xyz_box_cached

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
           cell face;
           the direction of the vector is a true normal, accounting for any difference in xy & z units
        """

        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        self.cache_all_geometry_arrays()
        vertices = self.points_cached[self.node_indices_for_face(face_index)]
        centre = self.face_centre_point(face_index)
        if crs.xy_units != crs.z_units:
            vertices = vertices.copy()
            wam.convert_lengths(vertices[:, 2], crs.z_units, crs.xy_units)
            centre[2] = wam.convert_lengths(centre[2], crs.z_units, crs.xy_units)
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

        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        self.cache_all_geometry_arrays()
        face_centre = self.face_centre_point(face_index)
        z_centre = face_centre[2]
        assert np.all(self.node_indices_for_face(face_index) >= 0)  # debug
        assert np.all(self.node_indices_for_face(face_index) < self.node_count)  # debug
        face_points = self.points_cached[self.node_indices_for_face(face_index), :].copy()
        if crs.xy_units != crs.z_units:
            wam.convert_lengths(face_points[:, 2], crs.z_units, crs.xy_units)
            face_centre[2] = wam.convert_lengths(face_centre[2], crs.z_units, crs.xy_units)
        normal = self.face_normal(face_index)
        az = vec.azimuth(normal)
        incl = vec.inclination(normal)
        vec.tilt_points(face_centre, az, -incl, face_points)  # modifies face_points in situ
        face_points[..., 2] = 0.0
        if not xy_plane:
            face_centre[2] = 0.0
            vec.tilt_points(face_centre, az, incl, face_points)  # tilt back to original average normal
            if crs.xy_units != crs.z_units:
                wam.convert_lengths(face_points[:, 2], crs.xy_units, crs.z_units)
            face_points[..., 2] += z_centre
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
        local_triangulation = tri.dt(face_points, algorithm = 'scipy')  # returns int array of shape (M, 3)
        assert len(
            local_triangulation) == len(face_points) - 2, 'face triangulation failure (concave edges when planar?)'
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
        """Returns centre point of a cell or array of centre points of all cells.
        
        Optionally cache centre points for all cells.

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

        from resqpy.unstructured._tetra_grid import TetraGrid

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

    def xy_units(self):
        """Returns the projected view (x, y) units of measure of the coordinate reference system for the grid.

        :meta common:
        """

        return rqet.find_tag(self._crs_root(), 'ProjectedUom').text

    def z_units(self):
        """Returns the vertical (z) units of measure of the coordinate reference system for the grid.

        :meta common:
        """

        return rqet.find_tag(self._crs_root(), 'VerticalUom').text

    def adjusted_volume(self, v, required_uom = None):
        """Returns volume adjusted for differing xy & z units in crs, and/or converted to required uom.

        note:
            if required_uom is not specified, units of returned value will be cube of crs units if xy & z are
            the same and either 'm' or 'ft', otherwise 'm3' will be used
        """

        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        if not required_uom:
            if crs.z_units in ['m', 'ft'] and crs.xy_units == crs.z_units:
                return v
            required_uom = 'm3'
        factor = 1.0
        if crs.xy_units != 'm':
            factor = wam.convert_lengths(1.0, crs.xy_units, 'm')
            factor *= factor
        if crs.z_units != 'm':
            factor = wam.convert_lengths(factor, crs.z_units, 'm')
        return wam.convert_volumes(v * factor, 'm3', required_uom)

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
            h5_reg.register_dataset(self.uuid,
                                    'NodesPerFace/cumulativeLength',
                                    self.nodes_per_face_cl,
                                    dtype = 'uint32')
            h5_reg.register_dataset(self.uuid, 'FacesPerCell/elements', self.faces_per_cell, dtype = 'uint32')
            h5_reg.register_dataset(self.uuid,
                                    'FacesPerCell/cumulativeLength',
                                    self.faces_per_cell_cl,
                                    dtype = 'uint32')
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
                    self.model.create_reciprocal_relationship(ug, 'destinationObject',
                                                              self.model.root(uuid = self.crs_uuid), 'sourceObject')
                    # create 2 way relationship between UnstructuredGrid and Ext
                    ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                    ext_node = self.model.root_for_part(ext_part)
                    self.model.create_reciprocal_relationship(ug, 'mlToExternalPartProxy', ext_node,
                                                              'externalPartProxyToMl')

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

    def _crs_root(self):
        """Returns xml root node for the crs object referred to by this grid."""

        if self.crs_uuid is None:
            return None
        return self.model.root(uuid = self.crs_uuid)
