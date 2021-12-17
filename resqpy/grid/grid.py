"""RESQML grid module handling IJK cartesian grids."""

# note: only IJK Grid format supported at present
# see also rq_import.py

version = '20th October 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('grid.py version ' + version)

import numpy as np

import resqpy.fault as rqf
import resqpy.olio.grid_functions as gf
import resqpy.olio.point_inclusion as pip
import resqpy.olio.transmission as rqtr
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.volume as vol
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
import resqpy.weights_and_measures as bwam
from resqpy.olio.base import BaseResqpy
from .transmissibility import transmissibility
from .extract_functions import extract_grid_parent, extract_extent_kji, extract_grid_is_right_handed, \
    extract_k_direction_is_down, extract_geometry_time_index, extract_crs_uuid, extract_crs_root, extract_k_gaps, \
    extract_pillar_shape, extract_has_split_coordinate_lines, extract_children, extract_stratigraphy, \
    extract_inactive_mask, extract_property_collection
from .create_grid_xml import create_grid_xml
from .write_functions import write_hdf5_from_caches, write_nexus_corp
from .defined_geometry import pillar_geometry_is_defined, cell_geometry_is_defined, geometry_defined_for_all_cells, \
    set_geometry_is_defined, geometry_defined_for_all_pillars, cell_geometry_is_defined_ref, \
    pillar_geometry_is_defined_ref
from .faults import find_faults, fault_throws, fault_throws_per_edge_per_column
from .face_functions import clear_face_sets, make_face_sets_from_pillar_lists, make_face_set_from_dataframe, \
    set_face_set_gcs_list_from_dict, is_split_column_face, split_column_faces, face_centre, face_centres_kji_01

from .points_functions import point_areally, point, points_ref, point_raw, unsplit_points_ref, corner_points, \
    invalidate_corner_points, interpolated_points, x_section_corner_points, split_x_section_points, \
    unsplit_x_section_points, uncache_points, horizon_points, split_horizon_points, \
    pixel_map_for_split_horizon_points, centre_point_list, interpolated_point, split_gap_x_section_points, \
    centre_point, z_corner_point_depths, coordinate_line_end_points, set_cached_points_from_property

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

    def set_k_direction_from_points(self):
        """Sets the K direction indicator based on z direction and mean z values for top and base.

        note:
           this method does not modify the grid_is_righthanded indicator
        """

        p = self.points_ref(masked = False)
        self.k_direction_is_down = True  # arbitrary default
        if p is not None:
            diff = np.nanmean(p[-1] - p[0])
            if not np.isnan(diff):
                self.k_direction_is_down = ((diff >= 0.0) == self.z_inc_down())
        return self.k_direction_is_down

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
                        top_j_edge_vectors_p[j, i, jip, :] = (
                            self.points_cached[0, self.pillars_for_column[j, i, 1, jip], :2] -
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
            log.warning(
                'top cell edges for neighbouring cells flip direction somewhere: this grid is probably unusable')
            good = False
        dot_jp = np.sum(base_j_edge_vectors[1:, :, :] * base_j_edge_vectors[:-1, :, :], axis = 2)
        dot_ip = np.sum(base_i_edge_vectors[1:, :, :] * base_i_edge_vectors[:-1, :, :], axis = 2)
        if not np.all(dot_jp >= 0.0) and np.all(dot_ip >= 0.0):
            log.warning(
                'base cell edges for neighbouring cells flip direction somewhere: this grid is probably unusable')
            good = False
        return good

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
                                kp * (self.extent_kji[0] - 1), jp * (self.extent_kji[1] - 1),
                                ip * (self.extent_kji[2] - 1)
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
            parts = collection.selective_parts_list(property_kind = 'thickness',
                                                    facet_type = 'netgross',
                                                    facet = 'gross')
            if len(parts) == 1:
                return collection.cached_part_array_ref(parts[0])
            parts = collection.selective_parts_list(property_kind = 'thickness')
            if len(parts) == 1 and collection.facet_for_part(parts[0]) is None:
                return collection.cached_part_array_ref(parts[0])
            parts = collection.selective_parts_list(property_kind = 'cell length',
                                                    facet_type = 'direction',
                                                    facet = 'K')
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
                    np.mean(self.array_corner_points[:, :, :, 1, :, :, 2] -
                            self.array_corner_points[:, :, :, 0, :, :, 2],
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
        """Returns boolean or boolean array indicating whether cell is pinched out.
        
        Pinched out means cell has a thickness less than tolerance.

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
            # note: properties must be identifiable in property_collection
            half_t = rqtr.half_cell_t(self, realization = realization)

        if realization is None:
            self.array_half_cell_t = half_t

        return half_t

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
        return (not cell_geometry_is_defined(self, cell_kji0 = cell_kji0)) or self.pinched_out(
            cell_kji0, cache_pinchout_array = True)

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
        """Return True if z increases downwards in the coordinate reference system used by the grid geometry

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
                               stratigraphy = True,
                               expand_const_arrays = False):
        """Create or append to an hdf5 file.
        
        Writes datasets for the grid geometry (and parent grid mapping) and properties from cached arrays.
        """
        # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
        # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
        # xml is not created here for property objects

        write_hdf5_from_caches(self, file, mode, geometry, imported_properties, write_active, stratigraphy,
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

    def poly_line_for_cell(self, cell_kji0, vertical_ref = 'top'):
        """Returns a numpy array of shape (4, 3) being the 4 corners.
        
        Corners are in order J-I-, J-I+, J+I+, J+I-; from the top or base face.
        """
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

        return create_grid_xml(self, ijk, ext_uuid, add_as_part, add_relationships, write_active, write_geometry)

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
