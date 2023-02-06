"""RESQML grid module handling IJK cartesian grids."""

# note: only IJK Grid format supported at present
# see also rq_import.py

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.grid as grr
import resqpy.grid_surface as rqgs
import resqpy.fault as rqf
import resqpy.property as rqp
import resqpy.olio.grid_functions as gf
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.crs as rqc
from resqpy.olio.base import BaseResqpy

from ._transmissibility import transmissibility, half_cell_transmissibility
from ._extract_functions import extract_grid_parent, extract_extent_kji, extract_grid_is_right_handed, \
    extract_k_direction_is_down, extract_geometry_time_index, extract_crs_uuid, extract_k_gaps, \
    extract_pillar_shape, extract_has_split_coordinate_lines, extract_children, extract_stratigraphy, \
    extract_inactive_mask, extract_property_collection, set_k_direction_from_points
from ._grid_types import grid_flavour
from ._write_hdf5_from_caches import _write_hdf5_from_caches
from ._write_nexus_corp import write_nexus_corp
from ._defined_geometry import pillar_geometry_is_defined, cell_geometry_is_defined, geometry_defined_for_all_cells, \
    set_geometry_is_defined, geometry_defined_for_all_pillars, cell_geometry_is_defined_ref, \
    pillar_geometry_is_defined_ref
from ._faults import find_faults, fault_throws, fault_throws_per_edge_per_column
from ._face_functions import clear_face_sets, make_face_sets_from_pillar_lists, make_face_set_from_dataframe, \
    set_face_set_gcs_list_from_dict, is_split_column_face, split_column_faces, face_centre, face_centres_kji_01
from ._points_functions import point_areally, point, points_ref, point_raw, unsplit_points_ref, corner_points, \
    invalidate_corner_points, interpolated_points, x_section_corner_points, split_x_section_points, \
    unsplit_x_section_points, uncache_points, horizon_points, split_horizon_points, \
    centre_point_list, interpolated_point, split_gap_x_section_points, \
    centre_point, z_corner_point_depths, coordinate_line_end_points, set_cached_points_from_property, \
    find_cell_for_point_xy, split_horizons_points
from ._create_grid_xml import _create_grid_xml, _add_pillar_points_xml
from ._pillars import create_column_pillar_mapping, pillar_foursome, pillar_distances_sqr, nearest_pillar, nearest_rod
from ._cell_properties import thickness, volume, _get_volume_uom, _get_volume_conversion_factor, pinched_out, \
    cell_inactive, interface_length, interface_vector, interface_lengths_kji, interface_vectors_kji, poly_line_for_cell
from ._connection_sets import fault_connection_set, pinchout_connection_set, k_gap_connection_set
from ._xyz import xyz_box, xyz_box_centre, bounding_box, composite_bounding_box, z_inc_down, \
    check_top_and_base_cell_edge_directions, _local_to_global_crs, _global_to_local_crs
from ._pixel_maps import pixel_maps, pixel_map_for_split_horizon_points

import warnings


class Grid(BaseResqpy):
    """Class for RESQML Grid (extent and geometry) within RESQML model object."""

    resqml_type = 'IjkGridRepresentation'

    @property
    def nk_plus_k_gaps(self):
        """Returns the number of layers including any K gaps."""
        if self.nk is None:
            return None
        if self.k_gaps is None:
            return self.nk
        return self.nk + self.k_gaps

    def __init__(self,
                 parent_model,
                 uuid = None,
                 find_properties = True,
                 geometry_required = True,
                 title = None,
                 originator = None,
                 extra_metadata = {}):
        """Create a Grid object and optionally populate from xml tree.

        arguments:
           parent_model (model.Model object): the model which this grid is part of
           uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
           find_properties (boolean, default True): if True and uuid is present, a grid property collection
              is instantiated as an attribute, holding properties for which this grid is the supporting
              representation
           geometry_required (boolean, default True): if True and no geometry node exists in the xml,
              an assertion error is raised; ignored if uuid is None
           title (str, optional): citation title for new grid; ignored if loading from xml
           originator (str, optional): name of person creating the grid; defaults to login id;
              ignored if loading from xml
           extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
              ignored if loading from xml

        returns:
           a newly created Grid object

        note:
           only IJK grids are handled here; see also resqpy unstructured

        :meta common:
        """

        # note: currently only handles IJK grids
        self.parent_grid_uuid = None  #: parent grid when this is a local grid
        self.parent_window = None  #: FineCoarse cell index mapping info between self and parent grid
        self.is_refinement = None  #: True indicates self is a refinement wrt. parent; False means coarsening
        self.local_grid_uuid_list = None  #: LGR & LGC children list
        self.grid_representation = None  #: flavour of grid, currently 'IjkGrid' or 'IjkBlockGrid'; not much used
        self.geometry_root = None  #: xml node at root of geometry sub-tree
        self.extent_kji = None  #: size of grid: (nk, nj, ni)
        self.ni = self.nj = self.nk = None  #: duplicated extent information as individual integers
        self.crs_uuid = None  #: uuid of the coordinate reference system used by the grid's geometry
        self.crs = None  #: Crs object
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
        self.time_index = None  #: optional time index for dynamic geometry
        self.time_series_uuid = None  #: optional time series for dynamic geometry
        self.represented_interpretation_uuid = None  #: optional represented interpretation uuid for the grid - EarthModelInterpretation object expected

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if not self.title:
            self.title = 'ROOT'

        if uuid is not None:
            if geometry_required:
                assert self.geometry_root is not None, 'grid geometry not present in xml'
            if find_properties:
                self.extract_property_collection()

    def _load_from_xml(self):
        # Extract simple attributes from xml and set as attributes in this resqpy object
        grid_root = self.root
        assert grid_root is not None
        flavour = grid_flavour(grid_root)
        assert flavour in ['IjkGrid', 'IjkBlockGrid'], 'attempt to initialise IjkGrid from xml for something else'
        self.grid_representation = flavour  # this attribute not much used
        self.extract_extent_kji()
        self.nk = self.extent_kji[0]  # for convenience available as individual attribs as well as np triplet
        self.nj = self.extent_kji[1]
        self.ni = self.extent_kji[2]
        self.geometry_root = rqet.find_tag(grid_root, 'Geometry')
        self.extract_crs_uuid()
        self.set_crs()
        if isinstance(self, grr.RegularGrid):
            self._load_regular_grid_from_xml()
        else:
            geom_type = rqet.node_type(rqet.find_tag(self.geometry_root, 'Points'))
            assert geom_type == 'Point3dHdf5Array'
            self.extract_has_split_coordinate_lines()
            self.pillar_geometry_is_defined()  # note: if there is no geometry at all, resqpy sets this True
            self.cell_geometry_is_defined()  # note: if there is no geometry at all, resqpy sets this True
            self.extract_geometry_time_index()
        if self.geometry_root is not None:
            self.extract_grid_is_right_handed()
            self.extract_pillar_shape()
            self.extract_k_direction_is_down()
        self.extract_k_gaps()
        assert not self.k_gaps or flavour == 'IjkGrid', 'K gaps present in regular grid'
        self.extract_parent()
        self.extract_children()
        # self.create_column_pillar_mapping()  # mapping now created on demand in other methods
        self.extract_inactive_mask()
        self.extract_stratigraphy()
        self.get_represented_interpretation()

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
            if geometry_defined_for_all_cells(self, cache_array = True):
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
        """Returns an integer array holding kji0 indices for the cells with given natural indices.

        argument:
           c0s: numpy integer array of shape (...,) being natural cell indices (for a flattened array)

        returns:
           numpy integer array of shape (..., 3) being the equivalent kji0 protocol cell indices
        """

        k0s, ji0s = divmod(c0s.flatten(), self.nj * self.ni)
        j0s, i0s = divmod(ji0s, self.ni)
        return np.stack((k0s, j0s, i0s), axis = -1).reshape(tuple(list(c0s.shape) + [3]))

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

        notes:
            call this method if much grid geometry processing is coming up, to save having to worry about
            individual caching arguments to many other methods;
            this method does not create a column to pillar mapping which will often also be needed;
            the arrays are cached as direct attributes to this grid object;
            the names, shapes and types of the attributes are:
            - array_cell_geometry_is_defined  (nk, nj, ni)  bool;
            - array_pillar_geometry_is_defined  (nj + 1, ni + 1)  bool;
            - points_cached  (nk + 1, nj + 1, ni + 1, 3) or (nk + 1, np, 3)  float  (np = number of primary pillars);
            - split_pillar_indices_cached  (nps)  int  (nps = number of primary pillars that are split);
            - cols_for_split_pillars  (npxc)  int  (npxc = number of column corners using extra pillars due to splitting);
            - cols_for_split_pillars_cl  (npx)  int  (npx = number of extra pillars due to splitting);
            the last 3 are only present when the grid has one or more split pillars;
            the split pillar data includes the use of a 'jagged' array (effectively an array of lists represented as
            a linear array and a 'cumulative length' index array)

        :meta common:
        """

        # todo: recheck the description of split pillar arrays given in the doc string
        cell_geometry_is_defined(self, cache_array = True)
        pillar_geometry_is_defined(self, cache_array = True)
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

    def write_hdf5_from_caches(self,
                               file = None,
                               mode = 'a',
                               geometry = True,
                               imported_properties = None,
                               write_active = None,
                               stratigraphy = True,
                               expand_const_arrays = False,
                               use_int32 = None):
        """Create or append to an hdf5 file.
        
        Writes datasets for the grid geometry (and parent grid mapping) and properties from cached arrays.
        """
        # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
        # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
        # xml is not created here for property objects

        _write_hdf5_from_caches(self, file, mode, geometry, imported_properties, write_active, stratigraphy,
                                expand_const_arrays, use_int32)

    def write_hdf5(self, expand_const_arrays = False):
        """Writes grid geometry arrays to hdf5 (thin wrapper around write_hdf5_from_caches().

        :meta common:
        """

        self.write_hdf5_from_caches(mode = 'a',
                                    geometry = True,
                                    imported_properties = None,
                                    write_active = True,
                                    stratigraphy = True,
                                    expand_const_arrays = expand_const_arrays)

    def off_handed(self):
        """Returns False if IJK and xyz have same handedness, True if they differ."""

        ijk_right_handed = self.extract_grid_is_right_handed()
        if self.crs is None:
            assert self.crs_uuid is not None
            self.crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        assert self.crs.axis_order == 'easting northing'
        # note: if z increases downwards, xyz is left handed
        return ijk_right_handed == self.z_inc_down()

    def write_nexus_corp(self,
                         file_name,
                         local_coords = False,
                         global_xy_units = None,
                         global_z_units = None,
                         global_z_increasing_downward = True,
                         nexus_unit_system = None,
                         write_nx_ny_nz = False,
                         write_units_keyword = False,
                         write_rh_keyword_if_needed = False,
                         write_corp_keyword = False,
                         use_binary = False,
                         binary_only = False,
                         nan_substitute_value = None):
        """Write grid geometry to file in Nexus CORP ordering.

        arguments:
            file_name (str): the path of the file to generate
            local_coords (bool, default False): if True, CORP data is written in local coordinates
            global_xy_units (str, optional): RESQML length uom to use for global xy coords;
                required if local_coords is False
            global_z_units (str, optional): RESQML length uom to use for global z coords;
                required if local_coords is False
            global_z_increasing_downward (bool, default True): whether global z values increase downwards;
                ignored if local_coords is True
            nexus_unit_system (str, optional): the target Nexus unit system for the CORP data; if present,
                one of: 'METRIC', 'METBAR', 'METKG/CM2', 'LAB' or 'ENGLISH'; if None, will be guessed
                based on local or global z units
            write_nx_ny_nz (bool, default False): if True, NX NY NZ keywords and values are written
            write_units_keyword (bool, default False): if True, the Nexus unit system keyword is written
            write_rh_keyword_if_needed (bool, default False): if True, the RIGHTHANDED keyword is written
                if needed based on xyz and IJK handedness
            write_corp_keyword (bool, default False): if True, the CORP keyword is written before the values
            use_binary (bool, default False): if True, a pure binary file with the corp array is written
            binary_only (bool, default False): if True, no ascii file is created, only a pure binary file
            nan_substitute_value (float, optional): if present, a value to use in place of NaNs
        """

        write_nexus_corp(self,
                         file_name,
                         local_coords = local_coords,
                         global_xy_units = global_xy_units,
                         global_z_units = global_z_units,
                         global_z_increasing_downward = global_z_increasing_downward,
                         nexus_unit_system = nexus_unit_system,
                         write_nx_ny_nz = write_nx_ny_nz,
                         write_units_keyword = write_units_keyword,
                         write_rh_keyword_if_needed = write_rh_keyword_if_needed,
                         write_corp_keyword = write_corp_keyword,
                         use_binary = use_binary,
                         binary_only = binary_only,
                         nan_substitute_value = nan_substitute_value)

    def xy_units(self):
        """Returns the projected view (x, y) units of measure of the coordinate reference system for the grid.

        :meta common:
        """

        if self.crs is None:
            if self.crs_uuid is None:
                return None
            self.crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        return self.crs.xy_units

    def z_units(self):
        """Returns the vertical (z) units of measure of the coordinate reference system for the grid.

        :meta common:
        """

        if self.crs is None:
            if self.crs_uuid is None:
                return None
            self.crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        return self.crs.z_units

    def skin(self, use_single_layer_tactics = False, is_regular = False):
        """Returns a GridSkin composite surface object reoresenting the outer surface of the grid."""

        # could cache 2 versions (with and without single layer tactics)
        if self.grid_skin is None or self.grid_skin.use_single_layer_tactics != use_single_layer_tactics:
            self.grid_skin = rqgs.GridSkin(self,
                                           use_single_layer_tactics = use_single_layer_tactics,
                                           is_regular = is_regular)
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
                   use_lattice = False,
                   extra_metadata = {}):
        """Creates an IJK grid node from a grid object and optionally adds to parts forest.

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
           use_lattice (boolean, default False): if True and write_geometry is True, a lattice representation is
              used for the geometry (only for RegularGrid objects)
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

        return _create_grid_xml(self, ijk, ext_uuid, add_as_part, add_relationships, write_active, write_geometry,
                                use_lattice)

    def x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
        """Deprecated: please use `unsplit_x_section_points` instead."""
        warnings.warn('Deprecated: please use `unsplit_x_section_points` instead.', DeprecationWarning)

        return unsplit_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

    # The implementations of the below functions have been moved to separate modules.

    def cell_geometry_is_defined(self, cell_kji0 = None, cell_geometry_is_defined_root = None, cache_array = True):
        """Returns True if the geometry of the specified cell is defined.

        Can also be used to cache (load) the boolean array.

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
        return cell_geometry_is_defined(self,
                                        cell_kji0 = cell_kji0,
                                        cell_geometry_is_defined_root = cell_geometry_is_defined_root,
                                        cache_array = cache_array)

    def pillar_geometry_is_defined(self, pillar_ji0 = None, pillar_geometry_is_defined_root = None, cache_array = True):
        """Returns True if the geometry of the specified pillar is defined; False otherwise.

        Can also be used to cache (load) the boolean array.

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
        return pillar_geometry_is_defined(self,
                                          pillar_ji0 = pillar_ji0,
                                          pillar_geometry_is_defined_root = pillar_geometry_is_defined_root,
                                          cache_array = cache_array)

    def geometry_defined_for_all_cells(self, cache_array = True):
        """Returns True if geometry is defined for all cells; False otherwise.

        argument:
           cache_array (boolean, default True): if True, the 'cell geometry is defined' array is cached in memory,
              unless the xml indicates that geometry is defined for all cells, in which case that is noted

        returns:
           boolean: True if geometry is defined for all cells; False otherwise
        """
        return geometry_defined_for_all_cells(self, cache_array = cache_array)

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
        return geometry_defined_for_all_pillars(self,
                                                cache_array = cache_array,
                                                pillar_geometry_is_defined_root = pillar_geometry_is_defined_root)

    def cell_geometry_is_defined_ref(self):
        """Returns an in-memory numpy array containing the boolean data indicating which cells have geometry defined.

        returns:
           numpy array of booleans of shape (nk, nj, ni); True value indicates cell has geometry defined; False
           indicates that the cell's geometry (points xyz values) cannot be used

        note:
           if geometry is flagged in the xml as being defined for all cells, then this function returns None;
           geometry_defined_for_all_cells() can be used to test for that situation
        """
        return cell_geometry_is_defined_ref(self)

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
        return pillar_geometry_is_defined_ref(self)

    def set_geometry_is_defined(self,
                                treat_as_nan = None,
                                treat_dots_as_nan = False,
                                complete_partial_pillars = False,
                                nullify_partial_pillars = False,
                                complete_all = False):
        """Set cached flags and/or arrays indicating which primary pillars have any points defined and which cells all points.

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
        return set_geometry_is_defined(self,
                                       treat_as_nan = treat_as_nan,
                                       treat_dots_as_nan = treat_dots_as_nan,
                                       complete_partial_pillars = complete_partial_pillars,
                                       nullify_partial_pillars = nullify_partial_pillars,
                                       complete_all = complete_all)

    def find_faults(self, set_face_sets = False, create_organizing_objects_where_needed = False):
        """Searches for column-faces that are faulted and assigns fault ids; creates list of column-faces per fault id.

        note:
           this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
           GridConnectionSet class
        """
        return find_faults(self,
                           set_face_sets = set_face_sets,
                           create_organizing_objects_where_needed = create_organizing_objects_where_needed)

    def fault_throws(self):
        """Finds mean throw of each J and I face; adds throw arrays as attributes to this grid and returns them.

        note:
           this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
           GridConnectionSet class
        """
        return fault_throws(self)

    def fault_throws_per_edge_per_column(self, mode = 'maximum', simple_z = False, axis_polarity_mode = True):
        """Return array holding max, mean or min throw based on split node separations.

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
        return fault_throws_per_edge_per_column(self,
                                                mode = mode,
                                                simple_z = simple_z,
                                                axis_polarity_mode = axis_polarity_mode)

    def extract_parent(self):
        """Returns the uuid of the parent grid for the supplied grid"""
        return extract_grid_parent(self)

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
        return transmissibility(self, tolerance, use_tr_properties, realization, modifier_mode)

    def extract_extent_kji(self):
        """Returns the grid extent; for IJK grids this is a 3 integer numpy array, order is Nk, Nj, Ni.

        returns:
           numpy int array of shape (3,) being number of cells in k, j & i axes respectively;
           the return value is cached in attribute extent_kji, which can alternatively be referenced
           directly by calling code as the value is set from xml on initialisation
        """
        return extract_extent_kji(self)

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
        return extract_grid_is_right_handed(self)

    def extract_k_direction_is_down(self):
        """Returns boolean indicating whether increasing K indices are generally for deeper cells, as stored in xml.

        returns:
           boolean: True if increasing K generally indicates increasing depth

        notes:
           resqml allows layers to fold back over themselves, so the relationship between k and depth might not
           be monotonic;
           higher level code sometimes requires k to increase with depth;
           independently of this, z values may increase upwards or downwards in a coordinate reference system;
           this method does not modify the grid_is_righthanded indicator
        """
        return extract_k_direction_is_down(self)

    def extract_geometry_time_index(self):
        """Returns integer time index, or None, for the grid geometry, as stored in xml for dynamic geometries.

        notes:
           if the value is not None, it represents the time index as stored in the xml, or the time index as
           updated when setting the node points from a points property
        """
        return extract_geometry_time_index(self)

    def extract_crs_uuid(self):
        """Returns uuid for coordinate reference system, as stored in geometry xml tree.

        returns:
           uuid.UUID object
        """
        return extract_crs_uuid(self)

    def set_crs(self, crs_uuid = None):
        """Establish crs attribute if not already set"""
        if self.crs is not None:
            return
        if crs_uuid is None:
            crs_uuid = self.extract_crs_uuid()
        assert crs_uuid is not None
        self.crs = rqc.Crs(self.model, uuid = crs_uuid)

    def get_represented_interpretation(self):
        """Establishes the represented interpretation uuid, as stored in the xml tree, if present"""
        self.represented_interpretation_uuid = rqet.find_nested_tags_text(self.root,
                                                                          ['RepresentedInterpretation', 'UUID'])

    def extract_pillar_shape(self):
        """Returns string indicating whether whether pillars are curved, straight, or vertical as stored in xml.

        returns:
           string: either 'curved', 'straight' or 'vertical'

        note:
           resqml datasets often have 'curved', even when the pillars are actually 'vertical' or 'straight';
           use actual_pillar_shape() method to determine the shape from the actual xyz points data
        """
        return extract_pillar_shape(self)

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
        return extract_has_split_coordinate_lines(self)

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
        return extract_k_gaps(self)

    def extract_stratigraphy(self):
        """Loads stratigraphic information from xml."""
        return extract_stratigraphy(self)

    def extract_children(self):
        """Looks for LGRs related to this grid and sets the local_grid_uuid_list attribute."""
        return extract_children(self)

    def extract_property_collection(self):
        """Load grid property collection object holding lists of all properties in model that relate to this grid.

        returns:
           resqml_property.GridPropertyCollection object

        note:
           a reference to the grid property collection is cached in this grid object; if the properties change,
           for example by generating some new properties, the property_collection attribute of the grid object
           would need to be reset to None elsewhere before calling this method again
        """
        return extract_property_collection(self)

    def extract_inactive_mask(self, check_pinchout = False):
        """Returns boolean numpy array indicating which cells are inactive, if (in)active property found in this grid.

        returns:
           numpy array of booleans, of shape (nk, nj, ni) being True for cells which are inactive; False for active

        note:
           RESQML does not have a built-in concept of inactive (dead) cells, though the usage guide advises to use a
           discrete property with a local property kind of 'active'; this resqpy code can maintain an 'inactive'
           attribute for the grid object, which is a boolean numpy array indicating which cells are inactive
        """
        return extract_inactive_mask(self, check_pinchout = check_pinchout)

    def is_split_column_face(self, j0, i0, axis, polarity):
        """Returns True if the I or J column face is split; False otherwise."""
        return is_split_column_face(self, j0, i0, axis, polarity)

    def split_column_faces(self):
        """Returns a pair of numpy boolean arrays indicating which internal column faces (column edges) are split."""
        return split_column_faces(self)

    def clear_face_sets(self):
        """Discard face sets."""
        return clear_face_sets(self)

    def set_face_set_gcs_list_from_dict(self, face_set_dict = None, create_organizing_objects_where_needed = False):
        """Creates a grid connection set for each feature in the face set dictionary, based on kelp list pairs."""
        return set_face_set_gcs_list_from_dict(
            self,
            face_set_dict = face_set_dict,
            create_organizing_objects_where_needed = create_organizing_objects_where_needed)

    def make_face_set_from_dataframe(self, df):
        """Creates a curtain face set for each named fault in dataframe.

        note:
           this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
           GridConnectionSet class
        """
        return make_face_set_from_dataframe(self, df)

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
        return make_face_sets_from_pillar_lists(self,
                                                pillar_list_list,
                                                face_set_id,
                                                axis = axis,
                                                ref_slice0 = ref_slice0,
                                                plus_face = plus_face,
                                                projection = projection)

    def face_centre(self,
                    cell_kji0,
                    axis,
                    zero_or_one,
                    points_root = None,
                    cache_resqml_array = True,
                    cache_cp_array = False):
        """Returns xyz location of the centre point of a face of the cell (or all cells)."""
        return face_centre(self,
                           cell_kji0,
                           axis,
                           zero_or_one,
                           points_root = points_root,
                           cache_resqml_array = cache_resqml_array,
                           cache_cp_array = cache_cp_array)

    def face_centres_kji_01(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """Returns an array of shape (3, 2, 3) being (axis, 0 or 1, xyz) of face centre points for cell."""
        return face_centres_kji_01(self,
                                   cell_kji0,
                                   points_root = points_root,
                                   cache_resqml_array = cache_resqml_array,
                                   cache_cp_array = cache_cp_array)

    def set_cached_points_from_property(self,
                                        points_property_uuid = None,
                                        property_collection = None,
                                        realization = None,
                                        time_index = None,
                                        set_grid_time_index = True,
                                        set_inactive = True,
                                        active_property_uuid = None,
                                        active_collection = None):
        """Modifies the cached points (geometry), setting the values from a points property.

        arguments:
           points_property_uuid (uuid, optional): the uuid of the points property; if present the
              remaining arguments are ignored except for inactive & active arguments
           property_collection (PropertyCollection, optional): defaults to property collection
              for the grid; should only contain one set of points properties (but may also contain
              other non-points properties)
           realization (int, optional): if present, the property in the collection with this
              realization number is used
           time_index (int, optional): if present, the property in the collection with this
              time index is used
           set_grid_time_index (bool, default True): if True, the grid's time index will be set
              to the time_index argument and the grid's time series uuid will be set to that
              referred to by the points property; if False, the grid's time index will not be
              modified
           set_inactive (bool, default True): if True, the grid's inactive mask will be set
              based on an active cell property
           active_property_uuid (uuid, optional): if present, the uuid of an active cell property
              to base the inactive mask on; ignored if set_inactive is False
           active_collection (uuid, optional): default's to property_collection if present, or
              the grid's property collection otherwise; only used if set_inactive is True and
              active_property_uuid is None

        notes:
           the points property must have indexable element 'nodes' and be the same shape as the
           official points array for the grid;
           note that the shape of the points array is quite different between grids with split
           pillars and those without;
           the uom of the points property must be a length uom and match that used by the grid's crs;
           the inactive mask of the grid will only be updated if the set_inactive argument is True;
           if points_property_uuid has been provided, and set_inactive is True, the active property
           must be identified with the active_property_uuid argument;
           if set_inactive is True and active_property_uuid is None and points_property_uuid is None and
           realization and/or time_index is in use, the active property collection must contain one
           series of active properties with the same variants (realizations and time indices) as the
           points property series;
           the active cell properties should be discrete and have a local property kind titled 'active';
           various cached data are invalidated and cleared by this method
        """
        return set_cached_points_from_property(self,
                                               points_property_uuid = points_property_uuid,
                                               property_collection = property_collection,
                                               realization = realization,
                                               time_index = time_index,
                                               set_grid_time_index = set_grid_time_index,
                                               set_inactive = set_inactive,
                                               active_property_uuid = active_property_uuid,
                                               active_collection = active_collection)

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
        return point_raw(self, index = index, points_root = points_root, cache_array = cache_array)

    def point(self,
              cell_kji0 = None,
              corner_index = np.zeros(3, dtype = 'int'),
              points_root = None,
              cache_array = True):
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
        return point(self,
                     cell_kji0 = cell_kji0,
                     corner_index = corner_index,
                     points_root = points_root,
                     cache_array = cache_array)

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
        return points_ref(self, masked = masked)

    def uncache_points(self):
        """Frees up memory by removing the cached copy of the grid's points data.

        note:
           the memory will only actually become free when any other references to it pass out of scope
           or are deleted
        """
        return uncache_points(self)

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
        return unsplit_points_ref(self, cache_array = cache_array, masked = masked)

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
        return horizon_points(self, ref_k0 = ref_k0, heal_faults = heal_faults, kp = kp)

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
        return split_horizon_points(self, ref_k0 = ref_k0, masked = masked, kp = kp)

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
        return split_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

    def split_gap_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
        """Return array of points representing cell corners from an I or J interface slice for a faulted grid.

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
        return split_gap_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

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
        return unsplit_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

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
        return x_section_corner_points(self,
                                       axis,
                                       ref_slice0 = ref_slice0,
                                       plus_face = plus_face,
                                       masked = masked,
                                       rotate = rotate,
                                       azimuth = azimuth)

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
        return pixel_map_for_split_horizon_points(self, horizon_points, origin, width, height, dx, dy = dy)

    def coordinate_line_end_points(self):
        """Returns xyz of top and bottom of each primary pillar.

        returns:
           numpy float array of shape (nj + 1, ni + 1, 2, 3)
        """
        return coordinate_line_end_points(self)

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
        return z_corner_point_depths(self, order = order)

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
        return corner_points(self,
                             cell_kji0 = cell_kji0,
                             points_root = points_root,
                             cache_resqml_array = cache_resqml_array,
                             cache_cp_array = cache_cp_array)

    def invalidate_corner_points(self):
        """Deletes cached copy of corner points, if present.

        Use if any pillar geometry changes, or to reclaim memory.
        """
        return invalidate_corner_points(self)

    def centre_point(self, cell_kji0 = None, cache_centre_array = False):
        """Returns centre point of a cell or array of centre points of all cells.

        Optionally cache centre points for all cells.

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
        return centre_point(self, cell_kji0 = cell_kji0, cache_centre_array = cache_centre_array)

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
        return centre_point_list(self, cell_kji0s)

    def point_areally(self, tolerance = 0.001):
        """Returns array indicating which cells are reduced to a point in both I & J axes.

        Returns:
            numopy bool array of shape extent_kji

        Note:
           Any NaN point values will yield True for a cell
        """
        return point_areally(self, tolerance = tolerance)

    def interpolated_point(self,
                           cell_kji0,
                           interpolation_fraction,
                           points_root = None,
                           cache_resqml_array = True,
                           cache_cp_array = False):
        """Returns xyz point interpolated from corners of cell.

        Depends on 3 interpolation fractions in range 0 to 1.
        """
        return interpolated_point(self,
                                  cell_kji0,
                                  interpolation_fraction,
                                  points_root = points_root,
                                  cache_resqml_array = cache_resqml_array,
                                  cache_cp_array = cache_cp_array)

    def interpolated_points(self,
                            cell_kji0,
                            interpolation_fractions,
                            points_root = None,
                            cache_resqml_array = True,
                            cache_cp_array = False):
        """Returns xyz points interpolated from corners of cell.

        Depending on 3 interpolation fraction numpy vectors, each value in range 0 to 1.

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
        return interpolated_points(self,
                                   cell_kji0,
                                   interpolation_fractions,
                                   points_root = points_root,
                                   cache_resqml_array = cache_resqml_array,
                                   cache_cp_array = cache_cp_array)

    def set_k_direction_from_points(self):
        """Sets the K direction indicator based on z direction and mean z values for top and base.

        note:
           this method does not modify the grid_is_righthanded indicator
        """
        return set_k_direction_from_points(self)

    def find_cell_for_point_xy(self, x, y, k0 = 0, vertical_ref = 'top', local_coords = True):
        """Searches in 2D for a cell containing point x,y in layer k0; return (j0, i0) or (None, None)."""
        return find_cell_for_point_xy(self, x, y, k0, vertical_ref = vertical_ref, local_coords = local_coords)

    def half_cell_transmissibility(grid, use_property = True, realization = None, tolerance = 1.0e-6):
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
        return half_cell_transmissibility(grid,
                                          use_property = use_property,
                                          realization = realization,
                                          tolerance = tolerance)

    def create_column_pillar_mapping(self):
        """Creates an array attribute holding set of 4 pillar indices for each I, J column of cells.

        returns:
           numpy integer array of shape (nj, ni, 2, 2) where the last two indices are jp, ip;
           the array contains the pillar index for each of the 4 corners of each column of cells

        notes:
           the array is also cached as an attribute of the grid object: grid.pillars_for_column
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
        return create_column_pillar_mapping(self)

    def pillar_foursome(self, ji0, none_if_unsplit = False):
        """Returns an int array of the natural pillar indices applicable to each column around primary.

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
        return pillar_foursome(self, ji0, none_if_unsplit = none_if_unsplit)

    def pillar_distances_sqr(self, xy, ref_k0 = 0, kp = 0, horizon_points = None):
        """Returns array of the square of the distances of primary pillars in x,y plane to point xy.

        arguments:
           xy (float pair): the xy coordinate to compute the pillar distances to
           ref_k0 (int, default 0): the horizon layer number to use
           horizon_points (numpy array, optional): if present, should be array as returned by
              horizon_points() method; pass for efficiency in case of multiple calls
        """
        return pillar_distances_sqr(self, xy, ref_k0 = ref_k0, kp = ref_k0, horizon_points = horizon_points)

    def nearest_pillar(self, xy, ref_k0 = 0, kp = 0):
        """Returns the (j0, i0) indices of the primary pillar with point closest in x,y plane to point xy."""
        return nearest_pillar(self, xy, ref_k0 = ref_k0, kp = kp)

    def nearest_rod(self, xyz, projection, axis, ref_slice0 = 0, plus_face = False):
        """Returns the (k0, j0) or (k0 ,i0) indices of the closest point(s) to xyz(s); projection is 'xy', 'xz' or 'yz'.

        note:
           currently only for unsplit grids
        """
        return nearest_rod(self, xyz, projection, axis, ref_slice0 = ref_slice0, plus_face = plus_face)

    def poly_line_for_cell(self, cell_kji0, vertical_ref = 'top'):
        """Returns a numpy array of shape (4, 3) being the 4 corners.

        Corners are in order J-I-, J-I+, J+I+, J+I-; from the top or base face.
        """
        return poly_line_for_cell(self, cell_kji0, vertical_ref = vertical_ref)

    def interface_lengths_kji(self,
                              cell_kji0,
                              points_root = None,
                              cache_resqml_array = True,
                              cache_cp_array = False,
                              required_uom = None):
        """Returns 3 interface centre point separation lengths for axes k, j, i.

        note:
           if required_uom is not specified, units of returned length are the grid's crs xy units
        """
        return interface_lengths_kji(self,
                                     cell_kji0,
                                     points_root = points_root,
                                     cache_resqml_array = cache_resqml_array,
                                     cache_cp_array = cache_cp_array,
                                     required_uom = required_uom)

    def interface_vectors_kji(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """Returns 3 interface centre point difference vectors for axes k, j, i.

        note:
            units are implicitly those of the grid's crs; differing xy & z units would imply that the direction of the
            vectors are not true directions
        """
        return interface_vectors_kji(self,
                                     cell_kji0,
                                     points_root = points_root,
                                     cache_resqml_array = cache_resqml_array,
                                     cache_cp_array = cache_cp_array)

    def interface_length(self,
                         cell_kji0,
                         axis,
                         points_root = None,
                         cache_resqml_array = True,
                         cache_cp_array = False,
                         required_uom = None):
        """Returns the length between centres of an opposite pair of faces of the cell.

        note:
           if required_uom is not specified, units of returned length are the grid's crs xy units
        """
        return interface_length(self,
                                cell_kji0,
                                axis,
                                points_root = points_root,
                                cache_resqml_array = cache_resqml_array,
                                cache_cp_array = cache_cp_array,
                                required_uom = required_uom)

    def interface_vector(self, cell_kji0, axis, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """Returns an xyz vector between centres of an opposite pair of faces of the cell (or vectors for all cells).

        note:
            units are implicitly those of the grid's crs; differing xy & z units would imply that the direction of the
            vector is not a true direction
        """
        return interface_vector(self,
                                cell_kji0,
                                axis,
                                points_root = points_root,
                                cache_resqml_array = cache_resqml_array,
                                cache_cp_array = cache_cp_array)

    def cell_inactive(self, cell_kji0, pv_array = None, pv_tol = 0.01):
        """Returns True if the cell is inactive."""
        return cell_inactive(self, cell_kji0, pv_array = pv_array, pv_tol = pv_tol)

    def pinched_out(self,
                    cell_kji0 = None,
                    tolerance = 0.001,
                    points_root = None,
                    cache_resqml_array = True,
                    cache_cp_array = False,
                    cache_thickness_array = False,
                    cache_pinchout_array = None):
        """Returns boolean or boolean array indicating whether cell is pinched out.

        Pinched out means cell has a thickness less than tolerance.

        :meta common:
        """
        return pinched_out(self,
                           cell_kji0 = cell_kji0,
                           tolerance = tolerance,
                           points_root = points_root,
                           cache_resqml_array = cache_resqml_array,
                           cache_cp_array = cache_cp_array,
                           cache_thickness_array = cache_thickness_array,
                           cache_pinchout_array = cache_pinchout_array)

    def volume(self,
               cell_kji0 = None,
               points_root = None,
               cache_resqml_array = True,
               cache_cp_array = False,
               cache_centre_array = False,
               cache_volume_array = True,
               property_collection = None,
               required_uom = None):
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
           required_uom (str, optional): if present, the RESQML unit of measure (for quantity volume) that
                                 the volumes will be returned (and cached) in; if None, the grid's CRS
                                 z units cubed will be used

        returns:
           float, being the volume of cell identified by cell_kji0;
           or numpy float array of shape (nk, nj, ni) if cell_kji0 is None

        notes:
           the function can be used to find the volume of a single cell, or cache volumes for all cells, or both;
           if property_collection is not None, a suitable volume property will be used if present;
           if calculated, volume is computed using 6 tetras each with a non-planar bilinear base face;

        :meta common:
        """
        return volume(self,
                      cell_kji0 = cell_kji0,
                      points_root = points_root,
                      cache_resqml_array = cache_resqml_array,
                      cache_cp_array = cache_cp_array,
                      cache_centre_array = cache_centre_array,
                      cache_volume_array = cache_volume_array,
                      property_collection = property_collection,
                      required_uom = required_uom)

    def get_volume_uom(self, required_uom):
        """Returns a RESQML unit of measure string to use for volume quantities for the grid.

        arguments:
           required_uom (str, optional): if present, the RESQML unit of measure (for quantity volume)
               to use

        returns:
           string holding a valid RESQML uom for quantity class volume
        """
        return _get_volume_uom(self, required_uom)

    def get_volume_conversion_factor(self, required_uom):
        """Returns a factor for converting volumes calculated in the grid's CRS to the required uom."""
        return _get_volume_conversion_factor(self, required_uom)

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
        return thickness(self,
                         cell_kji0 = cell_kji0,
                         points_root = points_root,
                         cache_resqml_array = cache_resqml_array,
                         cache_cp_array = cache_cp_array,
                         cache_thickness_array = cache_thickness_array,
                         property_collection = property_collection)

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
        return fault_connection_set(self,
                                    skip_inactive = skip_inactive,
                                    compute_transmissibility = compute_transmissibility,
                                    add_to_model = add_to_model,
                                    realization = realization,
                                    inherit_features_from = inherit_features_from,
                                    title = title)

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
        return pinchout_connection_set(self,
                                       skip_inactive = skip_inactive,
                                       compute_transmissibility = compute_transmissibility,
                                       add_to_model = add_to_model,
                                       realization = realization)

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
        return k_gap_connection_set(self,
                                    skip_inactive = skip_inactive,
                                    compute_transmissibility = compute_transmissibility,
                                    add_to_model = add_to_model,
                                    realization = realization,
                                    tolerance = tolerance)

    def check_top_and_base_cell_edge_directions(self):
        """Check grid top face I & J edge vectors (in x,y) against basal equivalents.

        Max 90 degree angle tolerated.

        returns:
            boolean: True if all checks pass; False if one or more checks fail

        notes:
           similarly checks cell edge directions in neighbouring cells in top (and separately in base)
           currently requires geometry to be defined for all pillars
           logs a warning if a check is not passed
        """
        return check_top_and_base_cell_edge_directions(self)

    def global_to_local_crs(self,
                            a,
                            crs_uuid,
                            global_xy_units = None,
                            global_z_units = None,
                            global_z_increasing_downward = None):
        """Converts array of points in situ from global coordinate system to established local one."""
        assert crs_uuid is not None
        # todo: change function name to be different from method name
        return _global_to_local_crs(self,
                                    a,
                                    crs_uuid = crs_uuid,
                                    global_xy_units = global_xy_units,
                                    global_z_units = global_z_units,
                                    global_z_increasing_downward = global_z_increasing_downward)

    def z_inc_down(self):
        """Return True if z increases downwards in the coordinate reference system used by the grid geometry

        :meta common:
        """
        return z_inc_down(self)

    def local_to_global_crs(self,
                            a,
                            crs_uuid,
                            global_xy_units = None,
                            global_z_units = None,
                            global_z_increasing_downward = None):
        """Converts array of points in situ from local coordinate system to global one."""
        assert crs_uuid is not None
        # todo: change function name to be different from method name
        return _local_to_global_crs(self,
                                    a,
                                    crs_uuid = crs_uuid,
                                    global_xy_units = global_xy_units,
                                    global_z_units = global_z_units,
                                    global_z_increasing_downward = global_z_increasing_downward)

    def composite_bounding_box(self, bounding_box_list):
        """Returns the xyz box which envelopes all the boxes in the list, as a numpy array of shape (2, 3)."""
        return composite_bounding_box(self, bounding_box_list)

    def bounding_box(self, cell_kji0, points_root = None, cache_cp_array = False):
        """Returns the xyz box which envelopes the specified cell, as a numpy array of shape (2, 3)."""
        return bounding_box(self, cell_kji0, points_root = points_root, cache_cp_array = cache_cp_array)

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
        return xyz_box_centre(self, points_root = points_root, lazy = lazy, local = local)

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
        return xyz_box(self, points_root = points_root, lazy = lazy, local = local)

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
        return pixel_maps(self, origin, width, height, dx, dy = dy, k0 = k0, vertical_ref = vertical_ref)

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
        return split_horizons_points(self, min_k0 = min_k0, max_k0 = max_k0, masked = masked)

    def combined_tr_mult_properties_from_gcs_mults(self,
                                                   gcs_uuid_list = None,
                                                   realization = None,
                                                   merge_mode = 'minimum',
                                                   sided = None,
                                                   fill_value = 1.0,
                                                   active_only = True,
                                                   apply_baffles = False,
                                                   baffle_triplet = None):
        """Add triplet of transmissibility multiplier properties by combining gcs properties.

        arguments:
            gcs_uuid_list (list of UUID, optional): if None, all grid connection sets related to this grid will
                be used
            realization (int, optional): if present, is used to filter tranmissibility multiplier input
                properties and is added to the output properties
            merge_mode (str, default 'minimum'): one of 'minimum', 'multiply', 'maximum', 'exception'; how to
                handle multiple values applicable to the same grid face
            sided (bool, optional): whether to apply values on both sides of each gcs cell-face pair; if None,
                will default to False if merge mode is multiply, True otherwise
            fill_value (float, optional, default 1.0): the value to use for grid faces not present in any of
                the gcs'es; if None, NaN will be used
            active_only (bool, default True): if True and an active property exists for a grid connection set,
                then only active faces are used when combining to make the grid face properties
            apply_baffles (bool, default False): if True, where a baffle property exists for a grid connection
                set, a transmissibility multiplier of zero will be used for faces marked as True, overriding the
                multiplier property values at such faces
            baffle_triplet (triplet of numpy bool arrays, optional): if present, boolean masks over the grid
                internal faces; where True, a value of zero will be enforced for the multipliers regardless
                of the grid connection set properties

        returns:
            list of 3 uuids, one for each of the newly created transmissibility multiplier properties

        notes:
            each grid connection set must refer to this grid only;
            the generated properties have indexable element 'faces', not 'faces per cell', so may not be
            suitable for faulted grids;
            the baffle_triplet arrays, if provided, must be for internal faces only, so have extents of
            (nk - 1, nj, ni), (nk, nj - 1, ni), (nk, nj, ni -1); note that this is a different protocol
            than the indexable element of faces, which includes external faces
        """

        if not gcs_uuid_list:
            gcs_uuid_list = self.model.uuids(obj_type = 'GridConnectionSetRepresentation', related_uuid = self.uuid)
        assert gcs_uuid_list, 'no grid connections sets identified for transmissibility multiplier combining'

        if baffle_triplet is not None:
            assert len(baffle_triplet) == 3
            assert (baffle_triplet[0].shape == (self.nk - 1, self.nj, self.ni) and
                    baffle_triplet[1].shape == (self.nk, self.nj - 1, self.ni) and
                    baffle_triplet[2].shape == (self.nk, self.nj, self.ni - 1))

        tr_mult_uuid_list = []
        for gcs_uuid in gcs_uuid_list:
            gcs_pc = rqp.PropertyCollection(support = rqf.GridConnectionSet(self.model, uuid = gcs_uuid))
            assert gcs_pc is not None
            tr_mult_part = gcs_pc.singleton(property_kind = 'transmissibility multiplier',
                                            realization = realization,
                                            continuous = True,
                                            indexable = 'faces')
            if tr_mult_part is None:
                log.warning(f'no transmissibility multiplier found for gcs uuid: {gcs_uuid}')
            else:
                tr_mult_uuid_list.append(self.model.uuid_for_part(tr_mult_part))
        log.info(f'{len(tr_mult_uuid_list)} gcs transmissibility multiplier sets being combined')
        assert len(tr_mult_uuid_list) > 0, 'no gcs multipliers found for combining'

        trm_k, trm_j, trm_i = rqf.combined_tr_mult_from_gcs_mults(self.model,
                                                                  tr_mult_uuid_list,
                                                                  merge_mode = merge_mode,
                                                                  sided = sided,
                                                                  fill_value = fill_value,
                                                                  active_only = active_only,
                                                                  apply_baffles = apply_baffles)
        assert trm_k is not None and trm_j is not None and trm_i is not None
        pc = self.extract_property_collection()

        if baffle_triplet is not None:
            trm_k[1:-1] = np.where(baffle_triplet[0], 0.0, trm_k[1:-1])
            trm_j[:, 1:-1] = np.where(baffle_triplet[1], 0.0, trm_j[:, 1:-1])
            trm_i[:, :, 1:-1] = np.where(baffle_triplet[2], 0.0, trm_i[:, :, 1:-1])

        for axis, trm in enumerate((trm_k, trm_j, trm_i)):
            axis_ch = 'KJI'[axis]
            pc.add_cached_array_to_imported_list(trm,
                                                 'combined from gcs tr mults',
                                                 'TMULT' + axis_ch,
                                                 discrete = False,
                                                 uom = 'Euc',
                                                 property_kind = 'transmissibility multiplier',
                                                 facet_type = 'direction',
                                                 facet = axis_ch,
                                                 realization = realization,
                                                 indexable_element = 'faces')

        pc.write_hdf5_for_imported_list()
        return pc.create_xml_for_imported_list_and_add_parts_to_model()

    def _add_geom_points_xml(self, geom_node, ext_uuid):
        """Generate geometry points node in xml, called during create_xml, overridden for RegularGrid."""
        return _add_pillar_points_xml(self, geom_node, ext_uuid)
