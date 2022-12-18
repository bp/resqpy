"""TetraGrid class module."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.grid as grr
import resqpy.olio.volume as vol
import resqpy.unstructured
import resqpy.unstructured._unstructured_grid as rug


class TetraGrid(rug.UnstructuredGrid):
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

    def volume(self, cell, required_uom = None):
        """Returns the volume of a single cell.

        arguments:
           cell (int): the index of the cell for which the volume is required

        returns:
           float being the volume of the tetrahedral cell

        note:
            if required_uom is not specified, returned units will be cube of crs units if xy & z are the same
            and either 'm' or 'ft', otherwise 'm3' will be used
        """

        self.cache_all_geometry_arrays()
        abcd = self.points_cached[self.distinct_node_indices_for_cell(cell)]
        assert abcd.shape == (4, 3)
        v = vol.tetrahedron_volume(abcd[0], abcd[1], abcd[2], abcd[3])
        return self.adjusted_volume(v, required_uom = required_uom)

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

        assert u_grid.cell_shape in rug.valid_cell_shapes
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
