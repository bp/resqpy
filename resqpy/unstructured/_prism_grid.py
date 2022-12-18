"""PrismGrid and VerticalPrismGrid class module."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.olio.intersection as meet
import resqpy.olio.transmission as rqtr
import resqpy.olio.triangulation as tri
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.property as rqp
import resqpy.surface as rqs
import resqpy.weights_and_measures as wam
import resqpy.unstructured
import resqpy.unstructured._unstructured_grid as rug


class PrismGrid(rug.UnstructuredGrid):
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
       vertical prism cells are constrained to have a fixed triangular horzontal cross-section, though top and base
       triangular faces need not be horizontal; edges not involved in the triangular faces must be vertical;
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

        self.nk = None  #: number of layers when constructed as a layered grid

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
                self.nk = int(self.extra_metadata['layer count'])

        self.grid_representation = 'VerticalPrismGrid'  #: flavour of grid; not much used

    @classmethod
    def from_surfaces(cls,
                      parent_model,
                      surfaces,
                      column_points = None,
                      column_triangles = None,
                      title = None,
                      originator = None,
                      extra_metadata = {},
                      set_handedness = False):
        """Create a layered vertical prism grid from an ordered list of untorn surfaces.

        arguments:
           parent_model (model.Model object): the model which this grid is part of
           surfaces (list of surface.Surface): list of two or more untorn surfaces ordered from
              shallowest to deepest; see notes
           column_points (2D numpy float array, optional): if present, the xy points to use for
              the grid's triangulation; see notes
           column_triangles (numpy int array of shape (M, 3), optional): if present, indices into the
              first dimension of column_points giving the xy triangulation to use for the grid; see notes
           title (str, optional): citation title for the new grid
           originator (str, optional): name of person creating the grid; defaults to login id
           extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid

        returns:
           a newly created VerticalPrismGrid object

        notes:
           this method will not work for torn (faulted) surfaces, nor for surfaces with recumbent folds;
           the surfaces may not cross each other, ie. the depth ordering must be consistent over the area;
           the triangular pattern of the columns (in the xy plane) can be specified with the column_points
           and column_triangles arguments;
           if those arguments are None, the first, shallowest, surface is used as a master and determines
           the triangular pattern of the columns;
           where a gravity vector from a node above does not intersect a surface, the point is inherited
           as a copy of the node above and will be NaNs if no surface above has an intersection;
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

        assert (column_points is None) == (column_triangles is None)
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
                assert rqc.Crs(parent_model,
                               uuid = vpg.crs_uuid) == rqc.Crs(parent_model,
                                                               uuid = s.crs_uuid), 'mismatching surface crs'

        # fetch the data for the top surface, to be used as the master for the triangular pattern
        if column_triangles is None:
            top_triangles, top_points = top.triangles_and_points()
            column_edges = top.distinct_edges()  # ordered pairs of node indices
        else:
            top_triangles = column_triangles
            if column_points.shape[1] == 3:
                top_points = column_points
            else:
                top_points = np.zeros((len(column_points), 3))
                top_points[:, :column_points.shape[1]] = column_points
            column_surf = rqs.Surface(parent_model, crs_uuid = vpg.crs_uuid)
            column_surf.set_from_triangles_and_points(column_triangles, column_points)
            column_edges = column_surf.distinct_edges()
        assert top_triangles.ndim == 2 and top_triangles.shape[1] == 3
        assert top_points.ndim == 2 and top_points.shape[1] in [2, 3]
        assert len(top_triangles) > 0
        p_count = len(top_points)
        bad_points = np.zeros(p_count, dtype = bool)

        # setup size of arrays for the vertical prism grid
        column_count = top_triangles.shape[0]
        surface_count = len(surfaces)
        layer_count = surface_count - 1
        column_edge_count = len(column_edges)
        vpg.cell_count = column_count * layer_count
        vpg.node_count = p_count * surface_count
        vpg.face_count = column_count * surface_count + column_edge_count * layer_count
        vpg.nk = layer_count
        if vpg.extra_metadata is None:
            vpg.extra_metadata = {}
        vpg.extra_metadata['layer count'] = vpg.nk

        # setup points with copies of points for top surface, z values to be updated later
        points = np.zeros((surface_count, p_count, 3))
        points[:, :, :] = top_points

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
               top_triangles.flatten() + surface_index * p_count
        # populate nodes per face for quadrilateral faces
        quad_nodes = np.empty((layer_count, column_edge_count, 2, 2), dtype = int)
        for layer in range(layer_count):
            quad_nodes[layer, :, 0, :] = column_edges + layer * p_count
            # reverse order of base pairs to maintain cyclic ordering of nodes per face
            quad_nodes[layer, :, 1, 0] = column_edges[:, 1] + (layer + 1) * p_count
            quad_nodes[layer, :, 1, 1] = column_edges[:, 0] + (layer + 1) * p_count
        vpg.nodes_per_face[3 * surface_count * column_count:] = quad_nodes.flatten()
        assert vpg.nodes_per_face[-1] > 0

        # set up faces per cell
        vpg.faces_per_cell = np.zeros(5 * vpg.cell_count, dtype = int)
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

        # instersect gravity vectors from column points with other surfaces, and update z values in points
        gravity = np.zeros((p_count, 3))
        gravity[:, 2] = 1.0  # up/down does not matter for the intersection function used below
        start = 1 if top_triangles is None else 0
        for surf in range(start, surface_count):
            surf_triangles, surf_points = surfaces[surf].triangles_and_points()
            intersects = meet.line_set_triangles_intersects(top_points, gravity, surf_points[surf_triangles])
            single_intersects = meet.last_intersects(intersects)  # will be triple NaN where no intersection occurs
            # inherit point from surface above where no intersection has occurred
            nan_lines = np.isnan(single_intersects[:, 0])
            if surf == 0:
                # allow NaN entries to handle unused distant circumcentres in Voronoi graph data
                # assert not np.any(nan_lines), 'top surface does not cover all column points'
                single_intersects[nan_lines] = np.NaN
            else:
                single_intersects[nan_lines] = points[surf - 1][nan_lines]
            # populate z values for layer of points
            points[surf, :, 2] = single_intersects[:, 2]

        vpg.points_cached = points.reshape((-1, 3))
        assert np.all(vpg.nodes_per_face < len(vpg.points_cached))

        return vpg

    @classmethod
    def from_seed_points_and_surfaces(cls,
                                      parent_model,
                                      seed_xy,
                                      surfaces,
                                      area_of_interest,
                                      title = None,
                                      originator = None,
                                      extra_metadata = {},
                                      set_handedness = False):
        """Create a layered vertical prism grid from seed points and an ordered list of untorn surfaces.

        arguments:
           parent_model (model.Model object): the model which this grid is part of
           seed_xy (numpy float array of shape (N, 2 or 3)): the xy locations of seed points
           surfaces (list of surface.Surface): list of two or more untorn surfaces ordered from
              shallowest to deepest; see notes
           area_of_interest (closed convex Polyline): the frontier polygon in the xy plane
           title (str, optional): citation title for the new grid
           originator (str, optional): name of person creating the grid; defaults to login id
           extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid

        returns:
           a newly created VerticalPrismGrid object

        notes:
           the triangular pattern of the columns (in the xy plane) is constructed as a re-triangulation
           of the Voronoi diagram of the Delauney triangulation of the seed points;
           the seed points may be xy or xyz data but z is ignored;
           this method will not work for torn (faulted) surfaces, nor for surfaces with recumbent folds;
           the surfaces may not cross each other, ie. the depth ordering must be consistent over the area;
           all the seed points must lie wholly within the area of interest;
           where a gravity vector from a node above does not intersect a surface, the point is inherited
           as a copy of the node above (the topmost surface must cover the entire area of interest);
           the Surface class has methods for creating a Surface from a PointSet or a Mesh (RESQML
           Grid2dRepresentation), or for a horizontal plane
        """

        assert seed_xy.ndim == 2 and seed_xy.shape[1] in [2, 3]
        assert area_of_interest.isclosed and area_of_interest.is_convex()

        # compute Delauney triangulation
        delauney_t, hull_indices = tri.dt(seed_xy, return_hull = True, algorithm = 'scipy')

        # construct Voronoi graph
        voronoi_points, voronoi_indices = tri.voronoi(seed_xy, delauney_t, hull_indices, area_of_interest)

        # re-triangulate Voronoi cells
        points, triangles = tri.triangulated_polygons(voronoi_points, voronoi_indices, centres = seed_xy[:, :2])

        vpg = cls.from_surfaces(parent_model,
                                surfaces,
                                column_points = points,
                                column_triangles = triangles,
                                title = title,
                                originator = originator,
                                extra_metadata = extra_metadata,
                                set_handedness = set_handedness)

        # TODO: store information relating columns to seed points, and seed points to top layer points?

        return vpg

    def column_count(self):
        """Returns the number of columns in the grid."""

        assert self.nk > 1, 'no layer information set for vertical prism grid'
        n_col, remainder = divmod(self.cell_count, self.nk)
        assert remainder == 0, 'code failure for vertical prism grid: cell and layer counts not compatible'
        return n_col

    def cell_centre_point(self, cell = None):
        """Returns centre point of a single cell (or all cells) calculated as the mean position of its 6 nodes.

        arguments:
           cell (int): the index of the cell for which the centre point is required

        returns:
           numpy float array of shape (3,) being the xyz location of the centre point of the cell
        """

        cp = self.corner_points(cell = cell)
        return np.mean(cp.reshape((-1, 6, 3)), axis = (1))

    def corner_points(self, cell = None):
        """Returns corner points for all cells or for a single cell.

        arguments:
           cell (int, optional): cell index of single cell for which corner points are required;
              if None, corner points are returned for all cells

        returns:
           numpy float array of shape (2, 3, 3) being xyz points for top & base points for one cell, or
           numpy float array of shape (N, 2, 3, 3) being xyz points for top & base points for all cells
        """

        if hasattr(self, 'array_corner_points'):
            if cell is None:
                return self.array_corner_points
            return self.array_corner_points[cell]

        p = self.points_ref()
        if cell is None:
            cp = np.empty((self.cell_count, 2, 3, 3), dtype = float)
            top_fi = self.faces_per_cell.reshape((-1, 5))[:, 0]
            top_npfi_start = np.where(top_fi == 0, 0, self.nodes_per_face_cl[top_fi - 1])
            top_npfi_end = self.nodes_per_face_cl[top_fi]
            base_fi = self.faces_per_cell.reshape((-1, 5))[:, 1]
            base_npfi_start = self.nodes_per_face_cl[base_fi - 1]
            base_npfi_end = self.nodes_per_face_cl[base_fi]
            for cell in range(self.cell_count):
                cp[cell, 0] = p[self.nodes_per_face[top_npfi_start[cell]:top_npfi_end[cell]]]
                cp[cell, 1] = p[self.nodes_per_face[base_npfi_start[cell]:base_npfi_end[cell]]]
            self.array_corner_points = cp
            return cp

        top_fi, base_fi = self.faces_per_cell.reshape((-1, 5))[cell, :2]
        top_npfi_start = 0 if top_fi == 0 else self.nodes_per_face_cl[top_fi - 1]
        base_npfi_start = self.nodes_per_face_cl[base_fi - 1]
        cp = np.empty((2, 3, 3), dtype = float)
        cp[0] = p[self.nodes_per_face[top_npfi_start:self.nodes_per_face_cl[top_fi]]]
        cp[1] = p[self.nodes_per_face[base_npfi_start:self.nodes_per_face_cl[base_fi]]]
        return cp

    def thickness(self, cell = None):
        """Returns array of thicknesses of all cells or a single cell.

        note:
           units are z units of crs used by this grid
        """

        if hasattr(self, 'array_thickness'):
            if cell is None:
                return self.array_thickness
            return self.array_thickness[cell]

        cp = self.corner_points(cell = cell)

        if cell is None:
            thick = np.mean(cp[:, 1, :, 2] - cp[:, 0, :, 2], axis = -1)
            self.array_thickness = thick
        else:
            thick = np.mean(cp[1, :, 2] - cp[0, :, 2])

        return thick

    def top_faces(self):
        """Returns the global face indices for the top triangular faces of the top layer of cells."""

        return self.faces_per_cell[:self.column_count() * 5].reshape(-1, 5)[:, 0]

    def triangulation(self):
        """Returns triangulation used by this vertical prism grid.

        returns:
           numpy int array of shape (M, 3) being the indices into the grid points of the triangles forming
           the top faces of the top layer of cells of the grid

        notes:
           use points_ref() to access the full points data;
           the order of the first axis of the returned values will match the order of the columns
           within cell related data
        """

        n_col = self.column_count()
        top_face_indices = self.top_faces()
        top_nodes = np.empty((n_col, 3), dtype = int)
        for column in range(n_col):  # also used as cell number
            top_nodes[column] = self.node_indices_for_face(top_face_indices[column])
        return top_nodes

    def triple_horizontal_permeability(self, primary_k, orthogonal_k, primary_azimuth = 0.0):
        """Returns array of triple horizontal permeabilities derived from a pair of permeability properties.

        arguments:
           primary_k (numpy float array of shape (N,) being the primary horizontal permeability for each cell
           orthogonal_k (numpy float array of shape (N,) being the horizontal permeability for each cell in a
              direction orthogonal to the primary permeability
           primary_azimuth (float or numpy float array of shape (N,), default 0.0): the azimuth(s) of the
              primary permeability, in degrees compass bearing

        returns:
           numpy float array of shape (N, 3) being the triple horizontal permeabilities applicable to half
           cell horizontal transmissibilities

        notes:
           this method should be used to generate horizontal permeabilities for use in transmissibility
           calculations if starting from an anisotropic permeability defined by a pair of locally
           orthogonal values; resulting values will be locally bounded by the pair of source values;
           the order of the 3 values per cell follows the node indices for the top triangle, for the
           opposing vertical face; no net to gross ratio modification is applied here;
        """

        assert primary_k.size == self.cell_count and orthogonal_k.size == self.cell_count
        azimuth_is_constant = isinstance(primary_azimuth, float) or isinstance(primary_azimuth, int)
        if not azimuth_is_constant:
            assert primary_azimuth.size == self.cell_count

        p = self.points_ref()
        t = self.triangulation()
        # mid point of triangle edges
        m = np.empty((t.shape[0], 3, 3), dtype = float)
        for e in range(3):
            m[:, e, :] = 0.5 * (p[t[:, e]] + p[t[:, e - 1]])
        # cell centre points
        c = self.centre_point()
        # todo: decide which directions to use: cell centre to face centre; or face normal?
        # vectors with direction cell centre to face centre (for one layer only)
        v = m - c.reshape((self.nk, -1, 1, 3))[0]
        # compass directions of those vectors
        a = vec.azimuths(v)
        # vector directions relative to primary permeability direction (per cell)
        if azimuth_is_constant:
            # work with one layer only
            ap_rad = np.radians(a - primary_azimuth)
        else:
            # value per face per cell in whole grid
            ap_rad = np.radians(
                np.repeat(a.reshape((1, -1, 3)), self.nk, axis = 0).reshape((-1, 3)) - primary_azimuth.reshape((-1, 1)))
        cos_ap = np.cos(ap_rad)
        sin_ap = np.sin(ap_rad)
        cos_sqr_ap = cos_ap * cos_ap
        sin_sqr_ap = sin_ap * sin_ap
        if azimuth_is_constant:
            cos_sqr_ap = np.repeat(cos_sqr_ap.reshape(1, -1, 3), self.nk, axis = 0).reshape((-1, 3))
            sin_sqr_ap = np.repeat(sin_sqr_ap.reshape(1, -1, 3), self.nk, axis = 0).reshape((-1, 3))
        # local elliptical permeability projected in directions of azimuths
        k = np.sqrt((primary_k * primary_k).reshape((-1, 1)) * cos_sqr_ap +
                    (orthogonal_k * orthogonal_k).reshape((-1, 1)) * sin_sqr_ap)

        return k

    def half_cell_transmissibility(self, use_property = True, realization = None, tolerance = 1.0e-6):
        """Returns (and caches if realization is None) half cell transmissibilities for this vertical prism grid.

        arguments:
           use_property (boolean, default True): if True, the grid's property collection is inspected for
              a possible half cell transmissibility array and if found, it is used instead of calculation
           realization (int, optional) if present, only a property with this realization number will be used
           tolerance (float, default 1.0e-6): minimum half axis length below which the transmissibility
              will be deemed uncomputable (for the axis in question); NaN values will be returned (not Inf);
              units are implicitly those of the grid's crs length units

        returns:
           numpy float array of shape (N, 5) where N is the number of cells in the grid and the 5 covers
           the faces of a cell in the order of the faces_per_cell data;
           units will depend on the length units of the coordinate reference system for the grid;
           the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m and ft respectively

        notes:
           this method does not write to hdf5, nor create a new property or xml;
           if realization is None, a grid attribute cached array will be used;
           tolerance will only be used if the half cell transmissibilities are actually computed
        """

        if realization is None and hasattr(self, 'array_half_cell_t'):
            return self.array_half_cell_t

        half_t = None
        pc = self.property_collection

        if use_property:
            half_t = pc.single_array_ref(property_kind = 'transmissibility',
                                         realization = realization,
                                         continuous = True,
                                         count = 1,
                                         indexable = 'faces per cell')
            if half_t:
                assert half_t.size == 5 * self.cell_count
                half_t = half_t.reshape(self.cell_count, 5)

        if half_t is None:
            # note: properties must be identifiable in property_collection
            # TODO: gather property arrays, deriving tri-permeablities
            # make sub-collection of permeability properties
            ppc = rqp.selective_version_of_collection(pc,
                                                      continuous = True,
                                                      realization = realization,
                                                      property_kind = 'permeability rock')
            assert ppc.number_of_parts() > 0, 'no permeability properties available for vertical prism grid'

            # look for a triple permeability; if present, assume to be per face horizontal permeabilities
            triple_perm_horizontal = ppc.single_array_ref(count = 3, indexable = 'cells')

            if triple_perm_horizontal is None:
                # look for horizontal isotropic permeability
                iso_h_part = None
                if ppc.number_of_parts() == 1:
                    iso_h_part = ppc.parts()[0]
                    assert ppc.indexable_for_part(
                        iso_h_part) == 'cells', 'single permeability property is not for cells'
                else:
                    iso_h_part = ppc.singleton(facet_type = 'direction',
                                               facet = 'horizontal',
                                               count = 1,
                                               indexable = 'cells')
                if iso_h_part is not None:
                    triple_perm_horizontal = np.repeat(ppc.cached_part_array_ref(iso_h_part), 3).reshape((-1, 3))
                else:
                    # look for horizontal permeability field properties and derive triple perm array
                    primary_k = ppc.single_array_ref(facet_type = 'direction',
                                                     facet = 'primary',
                                                     count = 1,
                                                     indexable = 'cells')
                    orthogonal_k = ppc.single_array_ref(facet_type = 'direction',
                                                        facet = 'orthogonal',
                                                        count = 1,
                                                        indexable = 'cells')
                    assert primary_k is not None and orthogonal_k is not None,  \
                       'failed to identify horizontal permeability properties for vertical prism grid'
                    # todo: add extra metadata or further faceting to be sure of finding correct property
                    azimuth_part = pc.singleton(property_kind = 'plane angle',
                                                facet_type = 'direction',
                                                facet = 'primary',
                                                realization = realization,
                                                continuous = True,
                                                count = 1,
                                                indexable = 'cells')
                    if azimuth_part is None:
                        primary_azimuth = 0.0  # default to north
                    else:
                        primary_azimuth = pc.cached_part_array_ref(azimuth_part)
                        azi_uom = pc.uom_for_part(azimuth_part)
                        if azi_uom is not None and azi_uom != 'dega':
                            primary_azimuth = wam.convert(primary_azimuth, azi_uom, 'dega', quantity = 'plane angle')
                    triple_perm_horizontal = self.triple_horizontal_permeability(primary_k, orthogonal_k,
                                                                                 primary_azimuth)

            perm_k = ppc.single_array_ref(facet_type = 'direction', facet = 'K', count = 1, indexable = 'cells')

            ntg = pc.single_array_ref(property_kind = 'net to gross ratio',
                                      realization = realization,
                                      continuous = True,
                                      count = 1,
                                      indexable = 'cells')

            half_t = rqtr.half_cell_t_vertical_prism(self,
                                                     triple_perm_horizontal = triple_perm_horizontal,
                                                     perm_k = perm_k,
                                                     ntg = ntg)

        if realization is None:
            self.array_half_cell_t = half_t

        return half_t
