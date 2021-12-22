"""RESQML grid module handling IJK cartesian grids."""

# note: only IJK Grid format supported at present
# see also rq_import.py

version = '21st December 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('grid.py version ' + version)

import numpy as np

import resqpy.olio.grid_functions as gf
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from .transmissibility import transmissibility, half_cell_transmissibility
from .extract_functions import extract_grid_parent, extract_extent_kji, extract_grid_is_right_handed, \
    extract_k_direction_is_down, extract_geometry_time_index, extract_crs_uuid, extract_crs_root, extract_k_gaps, \
    extract_pillar_shape, extract_has_split_coordinate_lines, extract_children, extract_stratigraphy, \
    extract_inactive_mask, extract_property_collection, set_k_direction_from_points

from .write_hdf5_from_caches import _write_hdf5_from_caches
from .write_nexus_corp import write_nexus_corp
from .defined_geometry import pillar_geometry_is_defined, cell_geometry_is_defined, geometry_defined_for_all_cells, \
    set_geometry_is_defined, geometry_defined_for_all_pillars, cell_geometry_is_defined_ref, \
    pillar_geometry_is_defined_ref
from .faults import find_faults, fault_throws, fault_throws_per_edge_per_column
from .face_functions import clear_face_sets, make_face_sets_from_pillar_lists, make_face_set_from_dataframe, \
    set_face_set_gcs_list_from_dict, is_split_column_face, split_column_faces, face_centre, face_centres_kji_01

from .points_functions import point_areally, point, points_ref, point_raw, unsplit_points_ref, corner_points, \
    invalidate_corner_points, interpolated_points, x_section_corner_points, split_x_section_points, \
    unsplit_x_section_points, uncache_points, horizon_points, split_horizon_points, \
    centre_point_list, interpolated_point, split_gap_x_section_points, \
    centre_point, z_corner_point_depths, coordinate_line_end_points, set_cached_points_from_property, \
    find_cell_for_point_xy, split_horizons_points

from ._create_grid_xml import _create_grid_xml

from .pillars import create_column_pillar_mapping, pillar_foursome, pillar_distances_sqr, nearest_pillar, nearest_rod
from .cell_properties import thickness, volume, pinched_out, cell_inactive, interface_length, interface_vector, \
    interface_lengths_kji, interface_vectors_kji, poly_line_for_cell
from .connection_sets import fault_connection_set, pinchout_connection_set, k_gap_connection_set
from .xyz import xyz_box, xyz_box_centre, bounding_box, composite_bounding_box, z_inc_down, \
    check_top_and_base_cell_edge_directions, local_to_global_crs, global_to_local_crs
from .pixel_maps import pixel_maps, pixel_map_for_split_horizon_points

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
        self.time_index = None  #: optional time index for dynamic geometry
        self.time_series_uuid = None  #: optional time series for dynamic geometry

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
            pillar_geometry_is_defined(self)  # note: if there is no geometry at all, resqpy sets this True
            cell_geometry_is_defined(self)  # note: if there is no geometry at all, resqpy sets this True
            self.extract_pillar_shape()
            self.extract_k_direction_is_down()
            self.extract_geometry_time_index()
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
        """Alias for root."""
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
           c0s: numpy integer array of shape (..., 3) being natural cell indices (for a flattened array)

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
                               expand_const_arrays = False):
        """Create or append to an hdf5 file.
        
        Writes datasets for the grid geometry (and parent grid mapping) and properties from cached arrays.
        """
        # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
        # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
        # xml is not created here for property objects

        _write_hdf5_from_caches(self, file, mode, geometry, imported_properties, write_active, stratigraphy,
                                expand_const_arrays)

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

        write_nexus_corp(self,
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
                         nan_substitute_value = None)

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

        return _create_grid_xml(self, ijk, ext_uuid, add_as_part, add_relationships, write_active, write_geometry)

    def x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
        """Deprecated: please use `unsplit_x_section_points` instead."""
        warnings.warn('Deprecated: please use `unsplit_x_section_points` instead.', DeprecationWarning)

        return unsplit_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

    # The implementations of the below functions have been moved to separate modules.

    def cell_geometry_is_defined(self, cell_kji0 = None, cell_geometry_is_defined_root = None, cache_array = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return cell_geometry_is_defined(self,
                                        cell_kji0 = cell_kji0,
                                        cell_geometry_is_defined_root = cell_geometry_is_defined_root,
                                        cache_array = cache_array)

    def pillar_geometry_is_defined(self, pillar_ji0 = None, pillar_geometry_is_defined_root = None, cache_array = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return pillar_geometry_is_defined(self,
                                          pillar_ji0 = pillar_ji0,
                                          pillar_geometry_is_defined_root = pillar_geometry_is_defined_root,
                                          cache_array = cache_array)

    def geometry_defined_for_all_cells(self, cache_array = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return geometry_defined_for_all_cells(self, cache_array = cache_array)

    def geometry_defined_for_all_pillars(self, cache_array = True, pillar_geometry_is_defined_root = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return geometry_defined_for_all_pillars(self,
                                                cache_array = cache_array,
                                                pillar_geometry_is_defined_root = pillar_geometry_is_defined_root)

    def cell_geometry_is_defined_ref(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return cell_geometry_is_defined_ref(self)

    def pillar_geometry_is_defined_ref(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return pillar_geometry_is_defined_ref(self)

    def set_geometry_is_defined(self,
                                treat_as_nan = None,
                                treat_dots_as_nan = False,
                                complete_partial_pillars = False,
                                nullify_partial_pillars = False,
                                complete_all = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return set_geometry_is_defined(self,
                                       treat_as_nan = treat_as_nan,
                                       treat_dots_as_nan = treat_dots_as_nan,
                                       complete_partial_pillars = complete_partial_pillars,
                                       nullify_partial_pillars = nullify_partial_pillars,
                                       complete_all = complete_all)

    def find_faults(self, set_face_sets = False, create_organizing_objects_where_needed = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return find_faults(self,
                           set_face_sets = set_face_sets,
                           create_organizing_objects_where_needed = create_organizing_objects_where_needed)

    def fault_throws(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return fault_throws(self)

    def fault_throws_per_edge_per_column(self, mode = 'maximum', simple_z = False, axis_polarity_mode = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return fault_throws_per_edge_per_column(self,
                                                mode = mode,
                                                simple_z = simple_z,
                                                axis_polarity_mode = axis_polarity_mode)

    def extract_parent(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_grid_parent(self)

    def transmissibility(self, tolerance = 1.0e-6, use_tr_properties = True, realization = None, modifier_mode = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return transmissibility(self, tolerance, use_tr_properties, realization, modifier_mode)

    def extract_extent_kji(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_extent_kji(self)

    def extract_grid_is_right_handed(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_grid_is_right_handed(self)

    def extract_k_direction_is_down(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_k_direction_is_down(self)

    def extract_geometry_time_index(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_geometry_time_index(self)

    def extract_crs_uuid(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_crs_uuid(self)

    def extract_crs_root(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_crs_root(self)

    def extract_pillar_shape(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_pillar_shape(self)

    def extract_has_split_coordinate_lines(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_has_split_coordinate_lines(self)

    def extract_k_gaps(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_k_gaps(self)

    def extract_stratigraphy(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_stratigraphy(self)

    def extract_children(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_children(self)

    def extract_property_collection(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_property_collection(self)

    def extract_inactive_mask(self, check_pinchout = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return extract_inactive_mask(self, check_pinchout = check_pinchout)

    def is_split_column_face(self, j0, i0, axis, polarity):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return is_split_column_face(self, j0, i0, axis, polarity)

    def split_column_faces(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return split_column_faces(self)

    def clear_face_sets(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return clear_face_sets(self)

    def set_face_set_gcs_list_from_dict(self, face_set_dict = None, create_organizing_objects_where_needed = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return set_face_set_gcs_list_from_dict(
            self,
            face_set_dict = face_set_dict,
            create_organizing_objects_where_needed = create_organizing_objects_where_needed)

    def make_face_set_from_dataframe(self, df):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return make_face_set_from_dataframe(self, df)

    def make_face_sets_from_pillar_lists(self,
                                         pillar_list_list,
                                         face_set_id,
                                         axis = 'K',
                                         ref_slice0 = 0,
                                         plus_face = False,
                                         projection = 'xy'):
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return face_centre(self,
                           cell_kji0,
                           axis,
                           zero_or_one,
                           points_root = points_root,
                           cache_resqml_array = cache_resqml_array,
                           cache_cp_array = cache_cp_array)

    def face_centres_kji_01(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return point_raw(self, index = index, points_root = points_root, cache_array = cache_array)

    def point(self,
              cell_kji0 = None,
              corner_index = np.zeros(3, dtype = 'int'),
              points_root = None,
              cache_array = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return point(self,
                     cell_kji0 = cell_kji0,
                     corner_index = corner_index,
                     points_root = points_root,
                     cache_array = cache_array)

    def points_ref(self, masked = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return points_ref(self, masked = masked)

    def uncache_points(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return uncache_points(self)

    def unsplit_points_ref(self, cache_array = False, masked = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return unsplit_points_ref(self, cache_array = cache_array, masked = masked)

    def horizon_points(self, ref_k0 = 0, heal_faults = False, kp = 0):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return horizon_points(self, ref_k0 = ref_k0, heal_faults = heal_faults, kp = kp)

    def split_horizon_points(self, ref_k0 = 0, masked = False, kp = 0):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return split_horizon_points(self, ref_k0 = ref_k0, masked = masked, kp = kp)

    def split_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return split_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

    def split_gap_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return split_gap_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

    def unsplit_x_section_points(self, axis, ref_slice0 = 0, plus_face = False, masked = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return unsplit_x_section_points(self, axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)

    def x_section_corner_points(self,
                                axis,
                                ref_slice0 = 0,
                                plus_face = False,
                                masked = False,
                                rotate = False,
                                azimuth = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return x_section_corner_points(self,
                                       axis,
                                       ref_slice0 = ref_slice0,
                                       plus_face = plus_face,
                                       masked = masked,
                                       rotate = rotate,
                                       azimuth = azimuth)

    def pixel_map_for_split_horizon_points(self, horizon_points, origin, width, height, dx, dy = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return pixel_map_for_split_horizon_points(self, horizon_points, origin, width, height, dx, dy = dy)

    def coordinate_line_end_points(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return coordinate_line_end_points(self)

    def z_corner_point_depths(self, order = 'cellular'):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return z_corner_point_depths(self, order = order)

    def corner_points(self, cell_kji0 = None, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return corner_points(self,
                             cell_kji0 = cell_kji0,
                             points_root = points_root,
                             cache_resqml_array = cache_resqml_array,
                             cache_cp_array = cache_cp_array)

    def invalidate_corner_points(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return invalidate_corner_points(self)

    def centre_point(self, cell_kji0 = None, cache_centre_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return centre_point(self, cell_kji0 = cell_kji0, cache_centre_array = cache_centre_array)

    def centre_point_list(self, cell_kji0s):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return centre_point_list(self, cell_kji0s)

    def point_areally(self, tolerance = 0.001):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return point_areally(self, tolerance = tolerance)

    def interpolated_point(self,
                           cell_kji0,
                           interpolation_fraction,
                           points_root = None,
                           cache_resqml_array = True,
                           cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return interpolated_points(self,
                                   cell_kji0,
                                   interpolation_fractions,
                                   points_root = points_root,
                                   cache_resqml_array = cache_resqml_array,
                                   cache_cp_array = cache_cp_array)

    def set_k_direction_from_points(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return set_k_direction_from_points(self)

    def find_cell_for_point_xy(self, x, y, k0 = 0, vertical_ref = 'top', local_coords = True):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return find_cell_for_point_xy(self, x, y, k0, vertical_ref = vertical_ref, local_coords = local_coords)

    def half_cell_transmissibility(grid, use_property = True, realization = None, tolerance = 1.0e-6):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return half_cell_transmissibility(grid,
                                          use_property = use_property,
                                          realization = realization,
                                          tolerance = tolerance)

    def create_column_pillar_mapping(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return create_column_pillar_mapping(self)

    def pillar_foursome(self, ji0, none_if_unsplit = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return pillar_foursome(self, ji0, none_if_unsplit = none_if_unsplit)

    def pillar_distances_sqr(self, xy, ref_k0 = 0, kp = 0, horizon_points = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return pillar_distances_sqr(self, xy, ref_k0 = ref_k0, kp = ref_k0, horizon_points = horizon_points)

    def nearest_pillar(self, xy, ref_k0 = 0, kp = 0):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return nearest_pillar(self, xy, ref_k0 = ref_k0, kp = kp)

    def nearest_rod(self, xyz, projection, axis, ref_slice0 = 0, plus_face = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return nearest_rod(self, xyz, projection, axis, ref_slice0 = ref_slice0, plus_face = plus_face)

    def poly_line_for_cell(self, cell_kji0, vertical_ref = 'top'):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return poly_line_for_cell(self, cell_kji0, vertical_ref = vertical_ref)

    def interface_lengths_kji(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return interface_lengths_kji(self,
                                     cell_kji0,
                                     points_root = points_root,
                                     cache_resqml_array = cache_resqml_array,
                                     cache_cp_array = cache_cp_array)

    def interface_vectors_kji(self, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return interface_vectors_kji(self,
                                     cell_kji0,
                                     points_root = points_root,
                                     cache_resqml_array = cache_resqml_array,
                                     cache_cp_array = cache_cp_array)

    def interface_length(self, cell_kji0, axis, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return interface_length(self,
                                cell_kji0,
                                axis,
                                points_root = points_root,
                                cache_resqml_array = cache_resqml_array,
                                cache_cp_array = cache_cp_array)

    def interface_vector(self, cell_kji0, axis, points_root = None, cache_resqml_array = True, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return interface_vector(self,
                                cell_kji0,
                                axis,
                                points_root = points_root,
                                cache_resqml_array = cache_resqml_array,
                                cache_cp_array = cache_cp_array)

    def cell_inactive(self, cell_kji0, pv_array = None, pv_tol = 0.01):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return cell_inactive(self, cell_kji0, pv_array = pv_array, pv_tol = pv_tol)

    def pinched_out(self,
                    cell_kji0 = None,
                    tolerance = 0.001,
                    points_root = None,
                    cache_resqml_array = True,
                    cache_cp_array = False,
                    cache_thickness_array = False,
                    cache_pinchout_array = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
               property_collection = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return volume(self,
                      cell_kji0 = cell_kji0,
                      points_root = points_root,
                      cache_resqml_array = cache_resqml_array,
                      cache_cp_array = cache_cp_array,
                      cache_centre_array = cache_centre_array,
                      cache_volume_array = cache_volume_array,
                      property_collection = property_collection)

    def thickness(self,
                  cell_kji0 = None,
                  points_root = None,
                  cache_resqml_array = True,
                  cache_cp_array = False,
                  cache_thickness_array = True,
                  property_collection = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
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
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return k_gap_connection_set(self,
                                    skip_inactive = skip_inactive,
                                    compute_transmissibility = compute_transmissibility,
                                    add_to_model = add_to_model,
                                    realization = realization,
                                    tolerance = tolerance)

    def check_top_and_base_cell_edge_directions(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return check_top_and_base_cell_edge_directions(self)

    def global_to_local_crs(self,
                            a,
                            crs_root = None,
                            global_xy_units = None,
                            global_z_units = None,
                            global_z_increasing_downward = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return global_to_local_crs(self,
                                   a,
                                   crs_root = crs_root,
                                   global_xy_units = global_xy_units,
                                   global_z_units = global_z_units,
                                   global_z_increasing_downward = global_z_increasing_downward)

    def z_inc_down(self):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return z_inc_down(self)

    def local_to_global_crs(self,
                            a,
                            crs_root = None,
                            global_xy_units = None,
                            global_z_units = None,
                            global_z_increasing_downward = None):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return local_to_global_crs(self,
                                   a,
                                   crs_root = crs_root,
                                   global_xy_units = global_xy_units,
                                   global_z_units = global_z_units,
                                   global_z_increasing_downward = global_z_increasing_downward)

    def composite_bounding_box(self, bounding_box_list):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return composite_bounding_box(self, bounding_box_list)

    def bounding_box(self, cell_kji0, points_root = None, cache_cp_array = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return bounding_box(self, cell_kji0, points_root = points_root, cache_cp_array = cache_cp_array)

    def xyz_box_centre(self, points_root = None, lazy = False, local = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return xyz_box_centre(self, points_root = points_root, lazy = lazy, local = local)

    def xyz_box(self, points_root = None, lazy = True, local = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return xyz_box(self, points_root = points_root, lazy = lazy, local = local)

    def pixel_maps(self, origin, width, height, dx, dy = None, k0 = None, vertical_ref = 'top'):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return pixel_maps(self, origin, width, height, dx, dy = dy, k0 = k0, vertical_ref = vertical_ref)

    def split_horizons_points(self, min_k0 = None, max_k0 = None, masked = False):
        """This method has now been moved to a new function elsewhere in the Grid module"""
        return split_horizons_points(self, min_k0 = min_k0, max_k0 = max_k0, masked = masked)
