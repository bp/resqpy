"""Submodule containing the RegularGrid class."""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.crs as rqc
import resqpy.grid
import resqpy.olio.transmission as rqtr
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
from resqpy.olio.xml_namespaces import curly_namespace as ns

import resqpy.grid._grid as grr_g
import resqpy.grid._grid_types as grr_gt
import resqpy.grid._create_grid_xml as grr_xml

# for a regular grid, the ...is_defined xml is created as a constant bool array, if the following set True
always_write_pillar_geometry_is_defined = True  # required True for Fesapi interoperability (xsd compliance)
always_write_cell_geometry_is_defined = False


class RegularGrid(grr_g.Grid):
    """Class for completely regular block grids, usually aligned with xyz axes."""

    # WIP: use RESQML lattice like geometry specification

    def __init__(self,
                 parent_model,
                 uuid = None,
                 extent_kji = None,
                 dxyz = None,
                 dxyz_dkji = None,
                 origin = (0.0, 0.0, 0.0),
                 crs_uuid = None,
                 use_vertical = False,
                 mesh = None,
                 mesh_dz_dk = 1.0,
                 set_points_cached = False,
                 as_irregular_grid = False,
                 find_properties = True,
                 title = None,
                 originator = None,
                 extra_metadata = {}):
        """Creates a regular grid object based on dxyz, or derived from a Mesh object.

        arguments:
           parent_model (model.Model object): the model to which the new grid will be assigned
           uuid (optional): the uuid for an existing grid part; if present, the RegularGrid object is
              based on existing data or a mix of that data and other arguments where present
           extent_kji (triple positive integers, optional): the number of cells in the grid (nk, nj, ni);
              required unless uuid is present
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
           set_points_cached (boolean, default False): if True, an explicit geometry is created for the regular grid
              in the form of the cached points array; will be treated as True if as_irregular_grid is True
           as_irregular_grid (boolean, default False): if True, the grid is setup such that it will appear as a Grid
              object when next loaded from disc
           find_properties (boolean, default True): if True and uuid is not None, a grid property collection is
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
           if uuid, dxyz, dxyz_dkji and mesh arguments are all None then unit cube cells aligned with
           the x,y,z axes will be generated;
           to store the geometry explicitly set as_irregular_grid True and use the following methods:
           make_regular_points_cached(), write_hdf5(), create_xml(..., write_geometry = True);
           otherwise, avoid write_hdf5() and call create_xml(..., write_geometry = False);
           if geometry is not stored explicitly, the uuid of the crs is stored as extra metadata
           if origin is not triple zero, a new crs will be created with the origin moved appropriately

        :meta common:
        """

        if as_irregular_grid:
            set_points_cached = True
            self.is_aligned = False
        else:
            self.is_aligned = None  #: boolean indicating alignment of IJK axes with +/- xyz respectively

        if uuid is None:

            assert extent_kji is not None and len(extent_kji) == 3
            super().__init__(parent_model, title = title, originator = originator, extra_metadata = extra_metadata)
            self.grid_representation = 'IjkGrid' if as_irregular_grid else 'IjkBlockGrid'
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
            self.block_origin = np.array(origin).copy() if uuid is None else np.zeros(3)
            self.block_dxyz_dkji = np.array(dxyz_dkji).copy()
            if self.is_aligned is None:
                self._set_is_aligned()
            if use_vertical and dxyz_dkji[0][0] == 0.0 and dxyz_dkji[0][1] == 0.0:  # ie. no x,y change with k
                self.pillar_shape = 'vertical'
            else:
                self.pillar_shape = 'straight'

            new_crs = None
            shift_origin = np.any(origin != 0.0) and uuid is None and not as_irregular_grid
            if crs_uuid is None and self.extra_metadata is not None:
                crs_uuid = bu.uuid_from_string(self.extra_metadata.get('crs uuid'))
                shift_origin = shift_origin and crs_uuid is None
            if crs_uuid is None:
                crs_uuid = parent_model.crs_uuid
            if crs_uuid is None:
                new_crs = rqc.Crs(parent_model, x_offset = origin[0], y_offset = origin[1], z_offset = origin[2])
                shift_origin = False
                new_crs.create_xml(reuse = True)
                crs_uuid = new_crs.uuid
            if shift_origin:
                new_crs = rqc.Crs(parent_model, uuid = crs_uuid)
                new_crs.uuid = bu.new_uuid()
                new_crs.x_offset += origin[0]
                new_crs.y_offset += origin[1]
                new_crs.z_offset += origin[2]
                new_crs.create_xml(reuse = True)
                crs_uuid = new_crs.uuid
            self.crs_uuid = crs_uuid
            self.crs = rqc.Crs(parent_model, uuid = crs_uuid) if new_crs is None else new_crs

        else:

            assert bu.is_uuid(uuid)
            assert grr_gt.is_regular_grid(parent_model.root_for_uuid(uuid))
            super().__init__(parent_model,
                             uuid = uuid,
                             find_properties = find_properties,
                             geometry_required = False,
                             title = title,
                             originator = originator,
                             extra_metadata = extra_metadata)

        self.geometry_defined_for_all_cells_cached = True
        self.geometry_defined_for_all_pillars_cached = True
        # self.array_cell_geometry_is_defined = np.full(tuple(self.extent_kji), True, dtype = bool)
        self.array_cell_geometry_is_defined = None

        if set_points_cached:
            self.make_regular_points_cached()

        if self.uuid is None:
            self.uuid = bu.new_uuid()

    def _load_regular_grid_from_xml(self):
        """Load from xml, called from super() init."""

        def get_point3d(node):
            assert node is not None and rqet.node_type(node) == 'Point3d'
            xyz = []
            for c in '123':
                xyz.append(rqet.find_tag_float(node, f'Coordinate{c}'))
                assert xyz[-1] is not None
            return tuple(xyz)

        self.grid_representation = 'IjkBlockGrid'
        self.is_aligned = True  # currently only aligned regular grids supported for lattice mode
        self.geometry_defined_for_all_pillars_cached = True
        self.geometry_defined_for_all_cells_cached = True
        self.pillar_shape = 'vertical'
        self.has_split_coordinate_lines = False
        self.block_origin = np.zeros(3, dtype = float)
        self.block_dxyz_dkji = np.zeros((3, 3), dtype = float)
        if self.geometry_root is None:
            log.debug('loading from xml without geometry node')
            # find cell length properties and populate dxyz from those values
            pc = self.extract_property_collection()
            assert pc is not None
            dxi_part = pc.singleton(property_kind = 'cell length',
                                    facet_type = 'direction',
                                    facet = 'I',
                                    indexable = 'cells',
                                    const_value = '*')
            dyj_part = pc.singleton(property_kind = 'cell length',
                                    facet_type = 'direction',
                                    facet = 'J',
                                    indexable = 'cells',
                                    const_value = '*')
            dzk_part = pc.singleton(property_kind = 'cell length',
                                    facet_type = 'direction',
                                    facet = 'K',
                                    indexable = 'cells',
                                    const_value = '*')
            assert dxi_part is not None and dyj_part is not None and dzk_part is not None
            dxi = self.property_collection.constant_value_for_part(dxi_part)
            dyj = self.property_collection.constant_value_for_part(dyj_part)
            dzk = self.property_collection.constant_value_for_part(dzk_part)
            assert dxi is not None and dyj is not None and dzk is not None
            self.block_dxyz_dkji[2, 0] = dxi
            self.block_dxyz_dkji[1, 1] = dyj
            self.block_dxyz_dkji[0, 2] = dzk
        else:
            log.debug('loading from xml with lattice geometry node')
            p_node = rqet.find_tag(self.geometry_root, 'Points')
            assert p_node is not None and rqet.node_type(p_node) == 'Point3dLatticeArray'
            assert rqet.find_tag_bool(p_node, 'AllDimensionsAreOrthogonal',
                                      must_exist = True), 'only aligned regular grids catered for'
            origin_node = rqet.find_tag(p_node, 'Origin')
            assert origin_node is not None, 'origin missing in xml for regular mesh (lattice)'
            self.block_origin[:] = get_point3d(origin_node)
            # following constraint is for aligned regular grids
            assert np.all(self.block_origin == 0.0),  \
                   'aligned regular grid origin must be coincident with local crs origin'
            offset_nodes = rqet.list_of_tag(p_node, 'Offset')
            assert len(offset_nodes) == 3, 'missing (or too many) offset nodes in xml for regular grid (lattice)'
            for axis, main_o_node in enumerate(offset_nodes):  # order K, J, I; ie. z, y, x
                d_xyz = get_point3d(rqet.find_tag(main_o_node, 'Offset'))
                self.block_dxyz_dkji[axis] = d_xyz
                # following constraint is for aligned regular grids
                for ax in range(3):
                    # note: these constraints allow for a negative cell length, ie. xyz axis reverse of IJK
                    is_non_zero = not maths.isclose(d_xyz[2 - ax], 0.0)
                    assert (ax == axis) == is_non_zero, 'unaligned lattice for regular grid not supported'
                spacing_node = rqet.find_tag(main_o_node, 'Spacing')
                assert spacing_node is not None
                spacing_value = rqet.find_tag_float(spacing_node, 'Value')
                assert spacing_value is not None and maths.isclose(spacing_value, 1.0)
                spacing_count = rqet.find_tag_int(spacing_node, 'Count')
                assert spacing_count is not None and spacing_count == self.extent_kji[axis]
            # log.debug(f'block dxyz_dkji from lattice: {self.block_dxyz_dkji}')

    def cell_geometry_is_defined(self, cell_kji0 = None, cell_geometry_is_defined_root = None, cache_array = True):
        """Override of Grid method indicating whether a cell has defined geometry, here always True."""

        return True

    def pillar_geometry_is_defined(self, pillar_ji0 = None, pillar_geometry_is_defined_root = None, cache_array = True):
        """Override of Grid method indicating whether a pillar has defined geometry, here always True."""

        return True

    def make_regular_points_cached(self, apply_origin_offset = True):
        """Set up the cached points array as an explicit representation of the regular grid geometry.

        arguments:
           apply_origin_offset (boolean, default True): if True, this method includes the regular grid
           origin in the calculated points data

        note:
           if apply_origin_offset is True, the related crs must not store the origin as offset data
        """

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
        if apply_origin_offset:
            self.points_cached[:, :, :] += self.block_origin

    def axial_lengths_kji(self):
        """Returns a triple float being lengths of primary axes (K, J, I) for each cell."""

        return vec.naive_lengths(self.block_dxyz_dkji)

    # override of Grid methods

    def point_raw(self, index = None, points_root = None, cache_array = True):
        """Returns element from points data, indexed as corner point (k0, j0, i0).

        Can optionally be used to cache points data.

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

        return self.block_origin + np.sum(np.repeat(np.array(index).reshape(
            (3, 1)), 3, axis = -1) * self.block_dxyz_dkji,
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

    def centre_point(self, cell_kji0 = None, cache_centre_array = False, use_origin = True):
        """Returns centre point of a cell or array of centre points of all cells.

        arguments:
           cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
              centre point is required; zero based indexing
           cache_centre_array (bool, default False): If True, or cell_kji0 is None, an array of centre points
              is generated and added as an attribute of the grid, with attribute name array_centre_point
           use_origin (bool, default True): if True, the x, y & z offsets (local origin) are added to the
              computed cell centre points

        returns:
           (x, y, z) 3 element numpy array of floats holding centre point of cell;
           or numpy 3+1D array if cell_kji0 is None

        note:
           resulting coordinates are in the same (local) crs as the grid points if use_origin is True
        """

        if cell_kji0 is None:
            cache_centre_array = True

        if cache_centre_array and (not hasattr(self, 'array_centre_point') or self.array_centre_point is None or
                                   not use_origin):
            centres = np.zeros((self.nk, self.nj, self.ni, 3))
            if self.is_aligned:
                centres[:, :, :, 0] = np.linspace(0.0,
                                                  self.block_dxyz_dkji[2, 0] * self.ni,
                                                  num = self.ni,
                                                  endpoint = False).reshape((1, 1, self.ni))
                centres[:, :, :, 1] = np.linspace(0.0,
                                                  self.block_dxyz_dkji[1, 1] * self.nj,
                                                  num = self.nj,
                                                  endpoint = False).reshape((1, self.nj, 1))
                centres[:, :, :, 2] = np.linspace(0.0,
                                                  self.block_dxyz_dkji[0, 2] * self.nk,
                                                  num = self.nk,
                                                  endpoint = False).reshape((self.nk, 1, 1))
            else:
                # todo: replace for loops with linspace
                for k in range(self.nk - 1):
                    centres[k + 1, 0, 0] = centres[k, 0, 0] + self.block_dxyz_dkji[0]
                for j in range(self.nj - 1):
                    centres[:, j + 1, 0] = centres[:, j, 0] + self.block_dxyz_dkji[1]
                for i in range(self.ni - 1):
                    centres[:, :, i + 1] = centres[:, :, i] + self.block_dxyz_dkji[2]
            centres += 0.5 * np.sum(self.block_dxyz_dkji, axis = 0)
            if use_origin:
                centres += self.block_origin
                self.array_centre_point = centres
        else:
            centres = None

        if cell_kji0 is not None:
            if centres is not None:
                return centres[tuple(cell_kji0)]
            if hasattr(self, 'array_centre_point') and self.array_centre_point is not None and use_origin:
                return self.array_centre_point[tuple(cell_kji0)]
            float_kji0 = np.array(cell_kji0, dtype = float) + 0.5
            centre = np.sum(self.block_dxyz_dkji * np.expand_dims(float_kji0, axis = -1).repeat(3, axis = -1), axis = 0)
            if use_origin:
                centre += self.block_origin
            return centre

        if centres is not None:
            return centres

        return self.array_centre_point

    def aligned_column_centres(self):
        """For an aligned grid, returns an array of column centres in xy, of shape (nj, ni, 2)."""

        assert self.is_aligned
        centres = np.zeros((self.nj, self.ni, 2))
        centres[:, :, 0] = np.linspace(0.0, self.block_dxyz_dkji[2, 0] * self.ni, num = self.ni,
                                       endpoint = False).reshape((1, self.ni))
        centres[:, :, 1] = np.linspace(0.0, self.block_dxyz_dkji[1, 1] * self.nj, num = self.nj,
                                       endpoint = False).reshape((self.nj, 1))
        centres += self.block_origin[:2] + 0.5 * np.sum(self.block_dxyz_dkji[1:, :2], axis = 0)
        return centres

    def volume(self, cell_kji0 = None, required_uom = None):
        """Returns bulk rock volume of cell or numpy array of bulk rock volumes for all cells.

        arguments:
           cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
                                 volume is required; zero based indexing
           required_uom (str, optional): if present, the RESQML unit of measure (for quantity volume) that
                                 the volumes will be returned in; if None, the grid's CRS z units cubed
                                 will be used

        returns:
           float, being the volume of cell identified by cell_kji0;
           or numpy float array of shape (nk, nj, ni) if cell_kji0 is None

        notes:
           the function can be used to find the volume of a single cell, or all cells;
           the method currently assumes that the primary i, j, k axes are mutually orthogonal
        """

        required_uom = self.get_volume_uom(required_uom)
        conversion_factor = self.get_volume_conversion_factor(required_uom)
        vol = np.product(vec.naive_lengths(self.block_dxyz_dkji))
        if conversion_factor is not None:
            vol *= conversion_factor
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

    def xyz_box(grid, points_root = None, lazy = True, local = False):
        """Returns the minimum and maximum xyz for the grid geometry.

        arguments:
           points_root (ignored): for compatibility with Grid method signature
           lazy (ignored): for compatibility with Grid method signature
           local (boolean, default False): if True, the xyz ranges that are returned are in the local
              coordinate space, otherwise the global (crs parent) coordinate space

        returns:
           numpy array of float of shape (2, 3); the first axis is minimum, maximum; the second axis is x, y, z

        :meta common:
        """

        if grid.xyz_box_cached is None:
            grid.xyz_box_cached = np.empty((2, 3))
            if grid.is_aligned:
                dxyz = np.array([grid.block_dxyz_dkji[2 - axis, axis] for axis in range(3)])
                temp_box = np.zeros((2, 3))
                temp_box[1] = np.array((grid.ni, grid.nj, grid.nk), dtype = float) * dxyz
                grid.xyz_box_cached[0] = np.amin(temp_box, axis = 0)
                grid.xyz_box_cached[1] = np.amax(temp_box, axis = 0)
            else:
                # generate points for outer grid corners, find min & max by axis
                grid_cp = np.zeros((2, 2, 2, 3))
                for k in [0, 1]:
                    lcp_k = float(k * grid.nk) * grid.block_dxyz_dkji[0]
                    for j in [0, 1]:
                        lcp_j = lcp_k + float(j * grid.nj) * grid.block_dxyz_dkji[1]
                        for i in [0, 1]:
                            grid_cp[k, j, i] = lcp_j + float(i * grid.ni) * grid.block_dxyz_dkji[2]
                grid_cp = grid_cp.reshape((8, 3))
                grid.xyz_box_cached[0] = np.amin(grid_cp, axis = 0)
                grid.xyz_box_cached[1] = np.amax(grid_cp, axis = 0)
            grid.xyz_box_cached_thoroughly = True
        if local:
            return grid.xyz_box_cached
        global_xyz_box = grid.xyz_box_cached.copy()
        grid.local_to_global_crs(global_xyz_box, crs_uuid = grid.crs_uuid)
        g_xyz_box = np.empty((2, 3), dtype = float)
        g_xyz_box[0] = np.amin(global_xyz_box, axis = 0)
        g_xyz_box[1] = np.amax(global_xyz_box, axis = 0)
        return g_xyz_box

    def find_cell_for_point_xy(self, x, y, k0 = 0, vertical_ref = 'top', local_coords = True):
        """Searches in 2D for a cell containing point x,y in layer k0; return (j0, i0) or (None, None)."""

        if x is None or y is None:
            return (None, None)
        if not local_coords and self.crs_uuid is not None:
            a = np.array([[x, y, 0.0]])  # extra axis needed to keep global_to_local_crs happy
            self.global_to_local_crs(a, crs_uuid = self.crs_uuid)
            x = a[0, 0]
            y = a[0, 1]
        log.debug(f'find for local x: {x}; y: {y}')
        if self.is_aligned:
            i0 = maths.floor(x / self.block_dxyz_dkji[2, 0])
            j0 = maths.floor(y / self.block_dxyz_dkji[1, 1])
        else:
            # TODO
            raise NotImplementedError('finding cell for x,y in unaligned regular grid')
            i0 = j0 = -1
        log.debug(f'found j0: {j0}; i0: {i0}')
        if 0 <= i0 < self.ni and 0 <= j0 < self.nj:
            return (j0, i0)
        return (None, None)

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   set_as_grid_root = True,
                   title = None,
                   originator = None,
                   write_active = True,
                   write_geometry = None,
                   extra_metadata = {},
                   expand_const_arrays = False,
                   add_cell_length_properties = True,
                   use_lattice = True):
        """Creates xml for this RegularGrid object; by default the explicit geometry is not included.

        see docstring for Grid.create_xml()

        additional argument:
           add_cell_length_properties (boolean, default True): if True, 3 constant property arrays with cells as
              indexable element are created to hold the lengths of the primary axes of the cells; the xml is
              created for the properties and they are added to the model (no hdf5 write needed)
           use_lattice (boolean, default True): if True, a geometry node is created with a Point3dLatticeArray;

        note:
           if the regular grid has been instatiated using the as_irregular_grid mode, a lattice will not be used,
           regardless of the use_lattice argument;
           if using a lattice representation, the write_geomety argument is ignored and a lattice geometry will
           be created

        :meta common:
        """

        if extra_metadata is None:
            extra_metadata = {}

        if write_geometry is None:
            write_geometry = (self.grid_representation == 'IjkGrid')

        if self.grid_representation == 'IjkGrid':
            use_lattice = False

        if write_geometry is None:
            write_geometry = (self.grid_representation == 'IjkGrid')
        else:
            extra_metadata = extra_metadata.copy()

        if write_geometry and not use_lattice:
            extra_metadata['grid_flavour'] = 'IjkGrid'
        if write_geometry or use_lattice:
            if 'crs uuid' in extra_metadata:
                extra_metadata.pop('crs uuid')
        elif self.crs_uuid is not None:
            extra_metadata['crs uuid'] = str(self.crs_uuid)

        node = super().create_xml(ext_uuid = ext_uuid,
                                  add_as_part = add_as_part,
                                  add_relationships = add_relationships,
                                  set_as_grid_root = set_as_grid_root,
                                  title = title,
                                  originator = originator,
                                  write_active = write_active,
                                  write_geometry = write_geometry or use_lattice,
                                  extra_metadata = extra_metadata,
                                  use_lattice = use_lattice)
        assert node is not None

        # patch in constant ...is_defined arrays if required
        if write_geometry or use_lattice:
            geom = rqet.find_tag(node, 'Geometry')
            assert geom is not None
            if always_write_pillar_geometry_is_defined and rqet.find_tag(geom, 'PillarGeometryIsDefined') is None:
                grr_xml._add_constant_pillar_geometry_is_defined(geom, self.extent_kji)
            if always_write_cell_geometry_is_defined and rqet.find_tag(geom, 'CellGeometryIsDefined') is None:
                grr_xml._add_constant_cell_geometry_is_defined(geom, self.extent_kji)

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
            if expand_const_arrays:
                dpc.write_hdf5_for_imported_list(expand_const_arrays = True)
            dpc.create_xml_for_imported_list_and_add_parts_to_model(expand_const_arrays = expand_const_arrays)
            if self.property_collection is None:
                self.property_collection = dpc
            else:
                if self.property_collection.support is None:
                    self.property_collection.set_support(support = self)
                self.property_collection.inherit_parts_from_other_collection(dpc)

            if add_relationships and self.crs_uuid is not None:
                self.model.create_reciprocal_relationship_uuids(self.uuid, 'destinationObject', self.crs_uuid,
                                                                'sourceObject')

        return node

    def _set_is_aligned(self):
        """Sets is_aligned attribute True if IJK axes align with +/- xyz respectively."""
        if self.block_dxyz_dkji is None:
            self.is_aligned = None
        self.is_aligned = (np.count_nonzero(self.block_dxyz_dkji * np.array([[1.0, 1, 0], [1, 0, 1], [0, 1, 1]])) == 0)

    def aligned_dxyz(self):
        """Returns triple float dx, dy, dz cell dimensions for aligned grids only."""
        assert self.is_aligned
        return (self.block_dxyz_dkji[2, 0], self.block_dxyz_dkji[1, 1], self.block_dxyz_dkji[0, 2])

    def slice_points(self, axis = 0, ref_slice = 0, local = False):
        """Returns a slice of points data for a layer or a cross section.

        arguments:
           axis (int, default 0): the axis of slicing 0 for K, 1 for J, 2 for I
           ref_slice (int, default 0): the slice index in the direction of axis
           local (bool, default False): if True, returned points are in the local crs; if False in global

        returns:
           numpy float array of points data of shape (Na, Nb, 3); (Na, Nb) are (NJ+1, NI+1), (NK+1, NI+1) or
           (NK+1, NJ+1) for axis values of 0, 1, or 2 respectively
        """
        assert 0 <= axis < 3
        assert 0 <= ref_slice <= self.extent_kji[axis]
        other_axes = np.array([[1, 2], [0, 2], [0, 1]], dtype = int)[axis]
        shape = (self.extent_kji[other_axes[0]] + 1, self.extent_kji[other_axes[1]] + 1, 3)
        dxyz_da = self.block_dxyz_dkji[other_axes[0]]
        dxyz_db = self.block_dxyz_dkji[other_axes[1]]
        p = np.zeros(shape, dtype = float)
        if ref_slice:
            p[0, 0] = ref_slice * self.block_dxyz_dkji[axis]
        for a in range(1, shape[0]):
            p[a, 0] = p[0, 0] + a * dxyz_da
        for b in range(1, shape[1]):
            p[:, b] = p[:, 0] + b * dxyz_db
        if not local:
            assert self.crs is not None
            self.crs.local_to_global_array(p)
        return p

    def _add_geom_points_xml(self, geom_node, ext_uuid):
        # RegularGrid override method using lattice representation, called from Grid.create_xml()

        def make_axial_offset(model, p_node, axis, d, n):
            dxyz = np.zeros(3, dtype = float)
            dxyz[2 - axis] = d
            o_node = rqet.SubElement(p_node, ns['resqml2'] + 'Offset')
            o_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dOffset')
            o_node.text = '\n'
            model.create_solitary_point3d('Offset', o_node, dxyz)
            space_node = rqet.SubElement(o_node, ns['resqml2'] + 'Spacing')
            space_node.set(ns['xsi'] + 'type',
                           ns['resqml2'] + 'DoubleConstantArray')  # nothing else catered for just now
            space_node.text = '\n'
            ov_node = rqet.SubElement(space_node, ns['resqml2'] + 'Value')
            ov_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
            ov_node.text = '1'
            oc_node = rqet.SubElement(space_node, ns['resqml2'] + 'Count')
            oc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            oc_node.text = str(n)

        assert self.crs_uuid is not None

        if self.grid_representation == 'IjkBlockGrid':

            # todo: handle non aligned regular grids
            assert self.is_aligned

            points_node = rqet.SubElement(geom_node, ns['resqml2'] + 'Points')
            points_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dLatticeArray')
            points_node.text = '\n'

            adao = rqet.SubElement(points_node, ns['resqml2'] + 'AllDimensionsAreOrthogonal')
            adao.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
            adao.text = 'true'

            self.model.create_solitary_point3d('Origin', points_node, (0.0, 0.0, 0.0))

            for axis in range(3):  # note: 1st Offset is for K axis, 2nd for J, then I
                make_axial_offset(self.model, points_node, axis, self.aligned_dxyz()[2 - axis], self.extent_kji[axis])

        else:  # generate geometry as irregular grid

            super()._add_geom_points_xml(geom_node, ext_uuid)
