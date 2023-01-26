"""BlockedWell class."""

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np
import pandas as pd
from functools import partial

import resqpy.crs as crs
import resqpy.grid as grr
import resqpy.olio.keyword_files as kf
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.wellspec_keywords as wsk
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.property as rqp
import resqpy.time_series as rqts
import resqpy.weights_and_measures as wam
import resqpy.well as rqw
import resqpy.well.well_utils as rqwu
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class BlockedWell(BaseResqpy):
    """Class for RESQML Blocked Wellbore Representation (Wells), ie cells visited by wellbore.

    notes:
       RESQML documentation:
       The information that allows you to locate, on one or several grids (existing or planned),
       the intersection of volume (cells) and surface (faces) elements with a wellbore trajectory
       (existing or planned)
    """

    resqml_type = 'BlockedWellboreRepresentation'
    well_name = rqo.alias_for_attribute("title")

    def __init__(self,
                 parent_model,
                 uuid = None,
                 grid = None,
                 trajectory = None,
                 wellspec_file = None,
                 cellio_file = None,
                 column_ji0 = None,
                 well_name = None,
                 check_grid_name = False,
                 use_face_centres = False,
                 represented_interp = None,
                 originator = None,
                 extra_metadata = None,
                 add_wellspec_properties = False,
                 usa_date_format = False):
        """Creates a new blocked well object and optionally loads it from xml, or trajectory, or Nexus wellspec file.

        arguments:
           parent_model (model.Model object): the model which the new blocked well belongs to
           uuid (optional): if present, the uuid of an existing blocked wellbore, in which case remaining
              arguments are ignored
           grid (optional, grid.Grid object): required if intialising from a trajectory or wellspec file;
              not used if uuid is not None
           trajectory (optional, Trajectory object): the trajectory of the well, to be intersected with the grid;
              not used if uuid is not None
           wellspec_file (optional, string): filename of an ascii file holding the Nexus wellspec data;
              ignored if uuid is not None or trajectory is not None
           cellio_file (optional, string): filename of an ascii file holding the RMS exported blocked well data;
              ignored if uuid is not None or trajectory is not None or wellspec_file is not None
           column_ji0 (optional, pair of ints): column indices (j0, i0) for a 'vertical' well; ignored if
              uuid is not None or trajectory is not None or wellspec_file is not None or
              cellio_file is not None
           well_name (string): the well name as given in the wellspec or cellio file; required if loading from
              one of those files; or the name to be used as citation title for a column well
           check_grid_name (boolean, default False): if True, the GRID column of the wellspec data will be checked
              for a match with the citation title of the grid object; perforations for other grids will be skipped;
              if False, all wellspec data is assumed to relate to the grid; only relevant when loading from wellspec
           use_face_centres (boolean, default False): if True, cell face centre points are used for the entry and
              exit points when constructing the simulation trajectory; if False and ANGLA & ANGLV data are available
              then entry and exit points are constructed based on a straight line at those angles passing through
              the centre of the cell; only relevant when loading from wellspec
           represented_interp (wellbore interpretation object, optional): if present, is noted as the wellbore
              interpretation object which this frame relates to; ignored if uuid is not None
           originator (str, optional): the name of the person creating the blocked well, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the blocked well;
              ignored if uuid is not None
           add_wellspec_properties (boolean or list of str, default False): if not False, and initialising from
              a wellspec file, the blocked well has its hdf5 data written and xml created and properties are
              fully created; if a list is provided the elements must be numerical wellspec column names;
              if True, all numerical columns other than the cell indices are added as properties
           usa_date_format (boolean, optional): specifies whether MM/DD/YYYY (True) or DD/MM/YYYY (False) is used 
              in wellspec file

        returns:
           the newly created blocked well object

        notes:
           if starting from a wellspec file or column indices, a 'simulation' trajectory and md datum objects are
           constructed to go with the blocked well;
           column wells might not be truly vertical - the trajectory will consist of linear segments joining the
           centres of the k faces in the column;
           optional RESQML attributes are not handled by this code (WITSML log reference, interval stratigraphic units,
           cell fluid phase units);
           multiple grids are currently only supported when loading an existing blocked well from xml;
           mysterious RESQML WellboreFrameIndexableElements is not used in any other RESQML classes and is therefore
           not used here;
           measured depth data must be in same crs as those for the related trajectory

        :meta common:
        """

        self.trajectory = trajectory  #: trajectory object associated with the wellbore
        self.trajectory_to_be_written = False
        self.feature_to_be_written = False
        self.interpretation_to_be_written = False
        self.node_count = None  #: number of measured depth nodes, each being an entry or exit point of trajectory with a cell
        self.node_mds = None  #: node_count measured depths (in same units and datum as trajectory) of cell entry and/or exit points
        self.cell_count = None  #: number of blocked intervals (<= node_count - 1)
        self.cell_indices = None  #: cell_count natural cell indices, paired with non-null grid_indices
        self.grid_indices = None  #: node_count-1 indices into grid list for each interval in node_mds; -1 for unblocked interval
        self.face_pair_indices = None  #: entry, exit face per cell indices, -1 for Target Depth termination within a cell
        self.grid_list = []  #: list of grid objects indexed by grid_indices
        self.wellbore_interpretation = None  #: associated wellbore interpretation object
        self.wellbore_feature = None  #: associated wellbore feature object
        self.well_name = None  #: name of well to import from ascii file formats

        self.cell_interval_map = None  # maps from cell index to interval (ie. node) index; populated on demand

        #: All logs associated with the blockedwellbore; an instance of :class:`resqpy.property.WellIntervalPropertyCollection`
        self.logs = None
        self.cellind_null = -1
        self.gridind_null = -1
        self.facepair_null = -1

        # face_index_map maps from (axis, p01) to face index value in range 0..5
        # this is the default as indicated on page 139 (but not p. 180) of the RESQML Usage Gude v2.0.1
        # also assumes K is generally increasing downwards
        # see DevOps backlog item 269001 discussion for more information
        #     self.face_index_map = np.array([[0, 1], [4, 2], [5, 3]], dtype = int)
        self.face_index_map = np.array([[0, 1], [2, 4], [5, 3]], dtype = int)  # order: top, base, J-, I+, J+, I-
        # and the inverse, maps from 0..5 to (axis, p01)
        #     self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 1], [2, 1], [1, 0], [2, 0]], dtype = int)
        self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [1, 1], [2, 0]], dtype = int)
        # note: the rework_face_pairs() method, below, overwrites the face indices based on I, J cell indices

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = well_name,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is None:
            self.wellbore_interpretation = represented_interp
            grid = self.__set_grid(grid = grid,
                                   wellspec_file = wellspec_file,
                                   cellio_file = cellio_file,
                                   column_ji0 = column_ji0)

            # Using dictionary mapping to replicate a switch statement. The init_function is chosen based on the
            # data source and the correct function is then called based on the init_function_dict
            init_function_dict = {
                'trajectory':
                    partial(self.compute_from_trajectory, self.trajectory, grid),
                'wellspec_file':
                    partial(self.derive_from_wellspec,
                            wellspec_file,
                            well_name,
                            grid,
                            check_grid_name = check_grid_name,
                            use_face_centres = use_face_centres,
                            add_properties = add_wellspec_properties,
                            usa_date_format = usa_date_format),
                'cellio_file':
                    partial(self.__check_cellio_init_okay,
                            cellio_file = cellio_file,
                            well_name = well_name,
                            grid = grid),
                'column_ji0':
                    partial(self.set_for_column, well_name, grid, column_ji0)
            }
            chosen_init_method = BlockedWell.__choose_init_data_source(trajectory = self.trajectory,
                                                                       wellspec_file = wellspec_file,
                                                                       cellio_file = cellio_file,
                                                                       column_ji0 = column_ji0)
            try:
                init_function_dict[chosen_init_method]()
            except KeyError:
                pass
            self.gridind_null = -1
            self.facepair_null = -1
            self.cellind_null = -1
            if not self.title:
                self.title = well_name
        # else an empty object is returned

    def __set_grid(self, grid, wellspec_file, cellio_file, column_ji0):
        """Set the grid to which the blocked well belongs."""

        if grid is None and (self.trajectory is not None or wellspec_file is not None or cellio_file is not None or
                             column_ji0 is not None):
            grid_final = self.model.grid()
        else:
            grid_final = grid
        return grid_final

    def __check_cellio_init_okay(self, cellio_file, well_name, grid):
        """Checks if BlockedWell object initialization from a cellio file is okay."""

        okay = self.import_from_rms_cellio(cellio_file, well_name, grid)
        if not okay:
            self.node_count = 0

    @staticmethod
    def __choose_init_data_source(trajectory, wellspec_file, cellio_file, column_ji0):
        """Specify the data source from which the BlockedWell object will be initialized."""
        if trajectory is not None:
            return "trajectory"
        elif wellspec_file is not None:
            return "wellspec_file"
        elif cellio_file is not None:
            return "cellio_file"
        elif column_ji0 is not None:
            return "column_ji0"

    def _load_from_xml(self):
        """Loads the blocked wellbore object from an xml node (and associated hdf5 data)."""

        node = self.root
        assert node is not None

        self.__find_trajectory_uuid(node = node)

        self.node_count = rqet.find_tag_int(node, 'NodeCount')
        assert self.node_count is not None and self.node_count >= 2, 'problem with blocked well node count'

        mds_node = rqet.find_tag(node, 'NodeMd')
        assert mds_node is not None, 'blocked well node measured depths hdf5 reference not found in xml'
        rqwu.load_hdf5_array(self, mds_node, 'node_mds')

        # Statement below has no effect, is this a bug?
        self.node_mds is not None and self.node_mds.ndim == 1 and self.node_mds.size == self.node_count

        self.cell_count = rqet.find_tag_int(node, 'CellCount')
        assert self.cell_count is not None and self.cell_count > 0

        # TODO: remove this if block once RMS export issue resolved
        if self.cell_count == self.node_count:
            extended_mds = np.empty((self.node_mds.size + 1,))
            extended_mds[:-1] = self.node_mds
            extended_mds[-1] = self.node_mds[-1] + 1.0
            self.node_mds = extended_mds
            self.node_count += 1

        assert self.cell_count < self.node_count

        self.__find_ci_node_and_load_hdf5_array(node = node)

        self.__find_fi_node_and_load_hdf5_array(node)

        unique_grid_indices = self.__find_gi_node_and_load_hdf5_array(node = node)

        self.__find_grid_node(node = node, unique_grid_indices = unique_grid_indices)

        interp_uuid = rqet.find_nested_tags_text(node, ['RepresentedInterpretation', 'UUID'])
        if interp_uuid is None:
            self.wellbore_interpretation = None
        else:
            self.wellbore_interpretation = rqo.WellboreInterpretation(self.model, uuid = interp_uuid)

        # Create blocked well log collection of all log data
        self.logs = rqp.WellIntervalPropertyCollection(frame = self)

        # Set up matches between cell_indices and grid_indices
        self.cell_grid_link = self.map_cell_and_grid_indices()

    def __find_trajectory_uuid(self, node):
        """Find and verify the uuid of the trajectory associated with the BlockedWell object."""

        trajectory_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['Trajectory', 'UUID']))
        assert trajectory_uuid is not None, 'blocked well trajectory reference not found in xml'
        if self.trajectory is None:
            self.trajectory = rqw.Trajectory(self.model, uuid = trajectory_uuid)
        else:
            assert bu.matching_uuids(self.trajectory.uuid, trajectory_uuid), 'blocked well trajectory uuid mismatch'

    def __find_ci_node_and_load_hdf5_array(self, node):
        """Find the BlockedWell object's cell indices hdf5 reference node and load the array."""

        ci_node = rqet.find_tag(node, 'CellIndices')
        assert ci_node is not None, 'blocked well cell indices hdf5 reference not found in xml'
        rqwu.load_hdf5_array(self, ci_node, 'cell_indices', dtype = int)
        assert (self.cell_indices is not None and self.cell_indices.ndim == 1 and
                self.cell_indices.size == self.cell_count), 'mismatch in number of cell indices for blocked well'
        self.cellind_null = rqet.find_tag_int(ci_node, 'NullValue')
        if self.cellind_null is None:
            self.cellind_null = -1  # if no Null found assume -1 default

    def __find_fi_node_and_load_hdf5_array(self, node):
        """Find the BlockedWell object's face indices hdf5 reference node and load the array."""

        fi_node = rqet.find_tag(node, 'LocalFacePairPerCellIndices')
        assert fi_node is not None, 'blocked well face indices hdf5 reference not found in xml'
        rqwu.load_hdf5_array(self, fi_node, 'raw_face_indices', dtype = 'int')
        assert self.raw_face_indices is not None, 'failed to load face indices for blocked well'
        assert self.raw_face_indices.size == 2 * self.cell_count, 'mismatch in number of cell faces for blocked well'
        if self.raw_face_indices.ndim > 1:
            self.raw_face_indices = self.raw_face_indices.reshape((self.raw_face_indices.size,))
        mask = np.where(self.raw_face_indices == -1)
        self.raw_face_indices[mask] = 0
        self.face_pair_indices = self.face_index_inverse_map[self.raw_face_indices]
        self.face_pair_indices[mask] = (-1, -1)
        self.face_pair_indices = self.face_pair_indices.reshape((-1, 2, 2))
        del self.raw_face_indices
        self.facepair_null = rqet.find_tag_int(fi_node, 'NullValue')
        if self.facepair_null is None:
            self.facepair_null = -1

    def __find_gi_node_and_load_hdf5_array(self, node):
        """Find the BlockedWell object's grid indices hdf5 reference node and load the array."""

        gi_node = rqet.find_tag(node, 'GridIndices')
        assert gi_node is not None, 'blocked well grid indices hdf5 reference not found in xml'
        rqwu.load_hdf5_array(self, gi_node, 'grid_indices', dtype = 'int')
        assert self.grid_indices is not None and self.grid_indices.ndim == 1 and self.grid_indices.size == self.node_count - 1
        unique_grid_indices = np.unique(self.grid_indices)  # sorted list of unique values
        self.gridind_null = rqet.find_tag_int(gi_node, 'NullValue')
        if self.gridind_null is None:
            self.gridind_null = -1  # if no Null found assume -1 default
        return unique_grid_indices

    def __find_grid_node(self, node, unique_grid_indices):
        """Find the BlockedWell object's grid reference node(s)."""
        grid_node_list = rqet.list_of_tag(node, 'Grid')
        assert len(grid_node_list) > 0, 'blocked well grid reference(s) not found in xml'
        assert unique_grid_indices[0] >= -1 and unique_grid_indices[-1] < len(
            grid_node_list), 'blocked well grid index out of range'
        assert np.count_nonzero(
            self.grid_indices >= 0) == self.cell_count, 'mismatch in number of blocked well intervals'
        self.grid_list = []
        for grid_ref_node in grid_node_list:
            grid_node = self.model.referenced_node(grid_ref_node)
            assert grid_node is not None, 'grid referenced in blocked well xml is not present in model'
            grid_uuid = rqet.uuid_for_part_root(grid_node)
            grid_obj = self.model.grid(uuid = grid_uuid, find_properties = False)
            self.grid_list.append(grid_obj)

    def map_cell_and_grid_indices(self):
        """Returns a list of index values linking the grid_indices to cell_indices.

        note:
           length will match grid_indices, and will show -1 where cell is unblocked
        """

        indexmap = []
        j = 0
        for i in self.grid_indices:
            if i == -1:
                indexmap.append(-1)
            else:
                indexmap.append(j)
                j += 1
        return indexmap

    def compressed_grid_indices(self):
        """Returns a list of grid indices excluding the -1 elements (unblocked intervals).

        note:
           length will match that of cell_indices
        """

        compressed = []
        for i in self.grid_indices:
            if i >= 0:
                compressed.append(i)
        assert len(compressed) == self.cell_count
        return compressed

    def number_of_grids(self):
        """Returns the number of grids referenced by the blocked well object."""

        if self.grid_list is None:
            return 0
        return len(self.grid_list)

    def single_grid(self):
        """Asserts that exactly one grid is being referenced and returns a grid object for that grid."""

        assert len(self.grid_list) == 1, 'blocked well is not referring to exactly one grid'
        return self.grid_list[0]

    def grid_uuid_list(self):
        """Returns a list of the uuids of the grids referenced by the blocked well object.

        :meta common:
        """

        uuid_list = []
        if self.grid_list is None:
            return uuid_list
        for g in self.grid_list:
            uuid_list.append(g.uuid)
        return uuid_list

    def interval_for_cell(self, cell_index):
        """Returns the interval index for a given cell index (identical if there are no unblocked intervals)."""
        assert 0 <= cell_index < self.cell_count
        if self.node_count == self.cell_count + 1:
            return cell_index
        if self.cell_interval_map is None:
            self._set_cell_interval_map()
        return self.cell_interval_map[cell_index]

    def entry_and_exit_mds(self, cell_index):
        """Returns entry and exit measured depths for a blocked cell.

        arguments:
            cell_index (int): the index of the cell in the blocked cells list; 0 <= cell_index < cell_count

        returns:
            (float, float) being the entry and exit measured depths for the cell, along the trajectory;
            uom is held in trajectory object
        """
        interval = self.interval_for_cell(cell_index)
        return (self.node_mds[interval], self.node_mds[interval + 1])

    def _set_cell_interval_map(self):
        """Sets up an index mapping from blocked cell index to interval index, accounting for unblocked intervals."""
        self.cell_interval_map = np.zeros(self.cell_count, dtype = int)
        ci = 0
        for ii in range(self.node_count - 1):
            if self.grid_indices[ii] < 0:
                continue
            self.cell_interval_map[ci] = ii
            ci += 1
        assert ci == self.cell_count

    def cell_indices_kji0(self):
        """Returns a numpy int array of shape (N, 3) of cells visited by well, for a single grid situation.

        :meta common:
        """

        grid = self.single_grid()
        return grid.denaturalized_cell_indices(self.cell_indices)

    def cell_indices_and_grid_list(self):
        """Returns a numpy int array of shape (N, 3) of cells visited by well, and a list of grid objects of length N.

        :meta common:
        """

        grid_for_cell_list = []
        grid_indices = self.compressed_grid_indices()
        assert len(grid_indices) == self.cell_count
        cell_indices = np.empty((self.cell_count, 3), dtype = int)
        for cell_number in range(self.cell_count):
            grid = self.grid_list[grid_indices[cell_number]]
            grid_for_cell_list.append(grid)
            cell_indices[cell_number] = grid.denaturalized_cell_index(self.cell_indices[cell_number])
        return cell_indices, grid_for_cell_list

    def cell_indices_for_grid_uuid(self, grid_uuid):
        """Returns a numpy int array of shape (N, 3) of cells visited by well in specified grid.

        :meta common:
        """

        if isinstance(grid_uuid, str):
            grid_uuid = bu.uuid_from_string(grid_uuid)
        ci_list, grid_list = self.cell_indices_and_grid_list()
        mask = np.zeros((len(ci_list),), dtype = bool)
        for cell_number in range(len(ci_list)):
            mask[cell_number] = bu.matching_uuids(grid_list[cell_number].uuid, grid_uuid)
        ci_selected = ci_list[mask]
        return ci_selected

    def box(self, grid_uuid = None):
        """Returns the KJI box containing the cells visited by the well, for single grid if grid_uuid is None."""

        if grid_uuid is None:
            cells_kji0 = self.cell_indices_kji0()
        else:
            cells_kji0 = self.cell_indices_for_grid_uuid(grid_uuid)

        if cells_kji0 is None or len(cells_kji0) == 0:
            return None
        well_box = np.empty((2, 3), dtype = int)
        well_box[0] = np.min(cells_kji0, axis = 0)
        well_box[1] = np.max(cells_kji0, axis = 0)
        return well_box

    def face_pair_array(self):
        """Returns numpy int array of shape (N, 2, 2) being pairs of face (axis, polarity) pairs, to go with cell_kji0_array().

        notes:
           each of the N rows in the returned array is of the form:
              ((entry_face_axis, entry_face_polarity), (exit_face_axis, exit_face_polarity))
           where the axis values are in the range 0 to 2 for k, j & i respectively, and
           the polarity values are zero for the 'negative' face and 1 for the 'positive' face;
           exit values may be -1 to indicate TD within the cell (ie. no exit point)
        """

        return self.face_pair_indices

    def compute_from_trajectory(self,
                                trajectory,
                                grid,
                                active_only = False,
                                quad_triangles = True,
                                use_single_layer_tactics = True):
        """Populate this blocked wellbore object based on intersection of trajectory with cells of grid.

        arguments:
           trajectory (Trajectory object): the trajectory to intersect with the grid; control_points and crs_uuid attributes must
              be populated
           grid (grid.Grid object): the grid with which to intersect the trajectory
           active_only (boolean, default False): if True, only active cells are included as blocked intervals
           quad_triangles (boolean, default True): if True, 4 triangles per cell face are used for the intersection calculations;
              if False, only 2 triangles per face are used
           use_single_layer_tactics (boolean, default True): if True and the grid does not have k gaps, initial intersection
              calculations with fault planes or the outer IK & JK skin of the grid are calculated as if the grid is a single
              layer (and only after an intersection is thus found is the actual layer identified); this significantly speeds up
              computation but may cause failure in the presence of significantly non-straight pillars and could (rarely) cause
              problems where a fault plane is significantly skewed (non-planar) even if individual pillars are straight

        note:
           this method is computationally intensive and might take ~30 seconds for a tyipical grid and trajectory; large grids,
           grids with k gaps, or setting use_single_layer_tactics False will typically result in significantly longer processing time
        """

        import resqpy.grid_surface as rgs  # was causing circular import issue when at global level

        # note: see also extract_box_for_well code
        assert trajectory is not None and grid is not None
        flavour = grr.grid_flavour(grid.root)
        if not flavour.startswith('Ijk'):
            raise NotImplementedError('well blocking only implemented for IjkGridRepresentation')
        is_regular = (flavour == 'IjkBlockGrid') and hasattr(grid, 'is_aligned') and grid.is_aligned
        if not is_regular and np.any(np.isnan(grid.points_ref(masked = False))):
            log.warning('grid does not have geometry defined everywhere: attempting fill')
            import resqpy.derived_model as rqdm
            fill_grid = rqdm.copy_grid(grid)
            fill_grid.set_geometry_is_defined(nullify_partial_pillars = True, complete_all = True)
            # note: may need to write hdf5 and create xml for fill_grid, depending on use in populate_blocked_well_from_trajectory()
            # fill_grid.write_hdf_from_caches()
            # fill_grid.create_xml
            grid = fill_grid
        assert trajectory.control_points is not None and trajectory.crs_uuid is not None and grid.crs_uuid is not None
        assert len(trajectory.control_points)

        self.trajectory = trajectory
        if not self.well_name:
            self.well_name = rqw.well_name(trajectory)
        if not self.title:
            self.title = self.well_name
        bw = rgs.populate_blocked_well_from_trajectory(self,
                                                       grid,
                                                       active_only = active_only,
                                                       quad_triangles = quad_triangles,
                                                       lazy = False,
                                                       use_single_layer_tactics = use_single_layer_tactics)
        if bw is None:
            log.error(f'failed to generate blocked well from trajectory with uuid: {trajectory.uuid}')
            self.node_count = None
            self.cell_count = None
        else:
            assert bw is self

    def set_for_column(self, well_name, grid, col_ji0, skip_inactive = True, length_uom = None):
        """Populates empty blocked well for a 'vertical' well in given column; creates simulation trajectory and md datum."""

        if well_name:
            self.well_name = well_name
        col_list = ['IW', 'JW', 'L', 'ANGLA', 'ANGLV']  # NB: L is Layer, ie. k
        pinch_col = grid.pinched_out(cache_cp_array = True, cache_pinchout_array = True)[:, col_ji0[0], col_ji0[1]]
        if skip_inactive and grid.inactive is not None:
            inactive_col = grid.inactive[:, col_ji0[0], col_ji0[1]]
        else:
            inactive_col = np.zeros(grid.nk, dtype = bool)
        data = {'IW': [], 'JW': [], 'L': []}
        for k0 in range(grid.nk):
            if pinch_col[k0] or inactive_col[k0]:
                continue
            # note: leaving ANGLA & ANGLV columns as NA will cause K face centres to be used when deriving from dataframe
            data['IW'].extend([col_ji0[1] + 1])
            data['JW'].extend([col_ji0[0] + 1])
            data['L'].extend([k0 + 1])
        df = pd.DataFrame(data, columns = col_list)

        return self.derive_from_dataframe(df, self.well_name, grid, use_face_centres = True, length_uom = length_uom)

    def derive_from_wellspec(self,
                             wellspec_file,
                             well_name,
                             grid,
                             check_grid_name = False,
                             use_face_centres = False,
                             add_properties = True,
                             usa_date_format = False,
                             last_data_only = False,
                             length_uom = None):
        """Populates empty blocked well from Nexus WELLSPEC data; creates simulation trajectory and md datum.

        arguments:
           wellspec_file (string): path of Nexus ascii file holding WELLSPEC keyword
           well_name (string): the name of the well as used in the wellspec data
           grid (grid.Grid object): the grid object which the cell indices in the wellspec data relate to
           check_grid_name (boolean, default False): if True, the GRID column of the wellspec data will be checked
              for a match with the citation title of the grid object; perforations for other grids will be skipped;
              if False, all wellspec data is assumed to relate to the grid
           use_face_centres (boolean, default False): if True, cell face centre points are used for the entry and
              exit points when constructing the simulation trajectory; if False and ANGLA & ANGLV data are available
              then entry and exit points are constructed based on a straight line at those angles passing through
              the centre of the cell
           add_properties (bool or list of str, default True): if True, WELLSPEC columns (other than IW, JW, L & GRID)
              are added as property parts for the blocked well; if a list is passed, it must contain a subset of the
              columns in the WELLSPEC data
           usa_date_format (bool, default False): if True, dates in the WELLSPEC file are interpreted as being in USA
              format (MM/DD/YYYY); otherwise European format (DD/MM/YYYY)
           last_data_only (bool, default False): If True, only the last entry of well data in the file is used and
              no time series or time index is used if properties are being added
           length_uom (str, optional): if present, the target length units for MD data in generated objects; if None,
              will default to z units of grid crs

        returns:
           self if successful; None otherwise

        note:
           if add_properties is True or present as a list, this method will write the hdf5, create the xml and add
           parts to the model for this blocked well and the properties
        """

        if not add_properties:
            last_data_only = True

        well_name = self.__derive_from_wellspec_check_well_name(well_name = well_name)

        col_list = rqwu._derive_from_wellspec_verify_col_list(add_properties = add_properties)

        name_for_check, col_list = rqwu._derive_from_wellspec_check_grid_name(check_grid_name = check_grid_name,
                                                                              grid = grid,
                                                                              col_list = col_list)

        wellspec_dict, dates_list = wsk.load_wellspecs(wellspec_file,
                                                       well = well_name,
                                                       column_list = col_list,
                                                       usa_date_format = usa_date_format,
                                                       last_data_only = last_data_only,
                                                       return_dates_list = True)

        assert len(wellspec_dict) == 1, 'no wellspec data found in file ' + wellspec_file + ' for well ' + well_name

        df = wellspec_dict[well_name]
        assert len(df) > 0, 'no rows of perforation data found in wellspec for well ' + well_name

        if 'DATE' not in df.columns or not dates_list:
            last_data_only = True

        if last_data_only:
            # name_for_check = grid_name if check_grid_name else None
            return self.derive_from_dataframe(df,
                                              well_name,
                                              grid,
                                              grid_name_to_check = name_for_check,
                                              use_face_centres = use_face_centres,
                                              add_as_properties = add_properties,
                                              length_uom = length_uom)

        # handle multiple times; note: dates already converted to iso format and sorted into chronological order
        # create time series from dates_list
        time_series = rqts.time_series_from_list(dates_list, parent_model = self.model, title = 'wellspec time series')
        time_series.create_xml(reuse = True)

        # select subset df for first timestamp (or None/NA if first entry for well is before first timestamp)
        if any(pd.isna(df['DATE'])):
            initial_df = df[pd.isna(df['DATE'])]  # todo: not sure this is valid pandas syntax
            next_date_index = 0
        else:
            initial_df = df[df['DATE'] == dates_list[0]]  # both should be in iso format
            next_date_index = 1

        success = self.derive_from_dataframe(initial_df,
                                             well_name,
                                             grid,
                                             grid_name_to_check = name_for_check,
                                             use_face_centres = use_face_centres,
                                             add_as_properties = add_properties,
                                             length_uom = length_uom,
                                             time_index = 0 if next_date_index else None,
                                             time_series_uuid = time_series.uuid if next_date_index else None)
        if not success:
            return None

        # for each (remaining) date in dates_list, if present in well df, creete subset df, compare row count with init, add props
        length_uom = grid.z_units()
        for date_index in range(next_date_index, len(dates_list)):
            date_df = df[df['DATE'] == dates_list[date_index]]
            if len(date_df) == 0:
                continue
            assert len(date_df) == len(initial_df),  \
                f'mismatch in number of rows (cells) at time {dates_list[date_index]} for well {well_name}'
            # note: dataframe rows assumed to be in unchanging order and refer to same cells!
            self.__add_as_properties_if_specified(add_properties,
                                                  date_df,
                                                  length_uom,
                                                  time_index = date_index,
                                                  time_series_uuid = time_series.uuid)

        return self

    def __derive_from_wellspec_check_well_name(self, well_name):
        """Set the well name to be used in the wellspec file."""
        if well_name:
            self.well_name = well_name
        else:
            well_name = self.well_name
        if not self.title:
            self.title = well_name
        return well_name

    def derive_from_cell_list(self, cell_kji0_list, well_name, grid, length_uom = None):
        """Populate empty blocked well from numpy int array of shape (N, 3) being list of cells."""

        df = pd.DataFrame(columns = ['IW', 'JW', 'L'])
        df['IW'] = cell_kji0_list[:, 2] + 1
        df['JW'] = cell_kji0_list[:, 1] + 1
        df['L'] = cell_kji0_list[:, 0] + 1

        return self.derive_from_dataframe(df, well_name, grid, use_face_centres = True, length_uom = length_uom)

    def derive_from_dataframe(self,
                              df,
                              well_name,
                              grid,
                              grid_name_to_check = None,
                              use_face_centres = True,
                              add_as_properties = False,
                              length_uom = None):
        """Populate empty blocked well from WELLSPEC-like dataframe; first columns must be IW, JW, L (i, j, k).

        note:
           if add_as_properties is True or present as a list of wellspec column names, both the blocked well and
           the properties will have their hdf5 data written, xml created and be added as parts to the model
        """

        if well_name:
            self.well_name = well_name
        else:
            well_name = self.well_name
        if not self.title:
            self.title = well_name

        assert len(df) > 0, 'empty dataframe for blocked well ' + str(well_name)

        if not length_uom:
            length_uom = grid.z_units()
        # assert grid.xy_units() == length_uom, 'mixed length units in grid crs'

        previous_xyz = None
        trajectory_mds = []
        trajectory_points = []  # entries paired with trajectory_mds
        blocked_intervals = []  # will have one fewer entries than trajectory nodes
        # blocked_intervals values; 0 = blocked, -1 = not blocked (for grid indices)
        blocked_cells_kji0 = []  # will have length equal to number of 0's in blocked intervals
        blocked_face_pairs = []  # same length as blocked_cells_kji0
        # blocked_face_pairs is list of ((entry axis, entry polarity), (exit axis, exit polarity))

        log.debug('wellspec dataframe for well ' + str(well_name) + ' has ' + str(len(df)) + ' row' + rqwu._pl(len(df)))

        skipped_warning_grid = None

        angles_present = ('ANGLV' in df.columns and 'ANGLA' in df.columns and not pd.isnull(df.iloc[0]['ANGLV']) and
                          not pd.isnull(df.iloc[0]['ANGLA']))

        if not angles_present and not use_face_centres:
            log.warning(f'ANGLV and/or ANGLA data unavailable for well {well_name}: using face centres')
            use_face_centres = True

        for i in range(len(df)):  # for each row in the dataframe for this well

            cell_kji0 = BlockedWell.__cell_kji0_from_df(df, i)
            if cell_kji0 is None:
                log.error('missing cell index in wellspec data for well ' + str(well_name) + ' row ' + str(i + 1))
                continue

            row = df.iloc[i]

            skipped_warning_grid, skip_row = BlockedWell.__verify_grid_name(grid_name_to_check = grid_name_to_check,
                                                                            row = row,
                                                                            skipped_warning_grid = skipped_warning_grid,
                                                                            well_name = well_name)
            if skip_row:
                continue

            cp = grid.corner_points(cell_kji0 = cell_kji0, cache_resqml_array = False)
            assert not np.any(np.isnan(cp)), 'missing geometry for perforation cell for well ' + str(well_name)

            entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz =  \
                BlockedWell.__calculate_entry_and_exit_axes_polarities_and_points(
                    angles_present = angles_present,
                    row = row,
                    cp = cp,
                    well_name = well_name,
                    df = df,
                    i = i,
                    cell_kji0 = cell_kji0,
                    blocked_cells_kji0 = blocked_cells_kji0,
                    use_face_centres = use_face_centres,
                    xy_units = grid.crs.xy_units,
                    z_units = grid.crs.z_units)
            log.debug(
                f'cell: {cell_kji0}; entry axis: {entry_axis}; polarity {entry_polarity}; exit axis: {exit_axis}; polarity {exit_polarity}'
            )

            previous_xyz, trajectory_mds, trajectory_points, blocked_intervals, blocked_cells_kji0, blocked_face_pairs =  \
                BlockedWell.__add_interval(
                    previous_xyz = previous_xyz,
                    entry_axis = entry_axis,
                    entry_polarity = entry_polarity,
                    entry_xyz = entry_xyz,
                    exit_axis = exit_axis,
                    exit_polarity = exit_polarity,
                    exit_xyz = exit_xyz,
                    cell_kji0 = cell_kji0,
                    trajectory_mds = trajectory_mds,
                    trajectory_points = trajectory_points,
                    blocked_intervals = blocked_intervals,
                    blocked_cells_kji0 = blocked_cells_kji0,
                    blocked_face_pairs = blocked_face_pairs,
                    xy_units = grid.crs.xy_units,
                    z_units = grid.crs.z_units,
                    length_uom = length_uom)

        blocked_count = len(blocked_cells_kji0)
        BlockedWell.__check_number_of_blocked_well_intervals(blocked_cells_kji0 = blocked_cells_kji0,
                                                             well_name = well_name,
                                                             grid_name = grid_name_to_check)

        self.node_count = len(trajectory_mds)
        self.node_mds = np.array(trajectory_mds)
        self.cell_count = len(blocked_cells_kji0)
        self.grid_indices = np.array(blocked_intervals, dtype = int)  # NB. only supporting one grid at the moment
        self.cell_indices = grid.natural_cell_indices(np.array(blocked_cells_kji0))
        self.face_pair_indices = np.array(blocked_face_pairs, dtype = int)
        self.grid_list = [grid]

        trajectory_points, trajectory_mds = BlockedWell.__add_tail_to_trajectory_if_necessary(
            blocked_count = blocked_count,
            exit_axis = exit_axis,
            exit_polarity = exit_polarity,
            cell_kji0 = cell_kji0,
            grid = grid,
            trajectory_points = trajectory_points,
            trajectory_mds = trajectory_mds)

        self.create_md_datum_and_trajectory(grid, trajectory_mds, trajectory_points, length_uom, well_name)
        self.__add_as_properties_if_specified(add_as_properties = add_as_properties, df = df, length_uom = length_uom)

        return self

    @staticmethod
    def __cell_kji0_from_df(df, df_row):
        row = df.iloc[df_row]
        if pd.isna(row[0]) or pd.isna(row[1]) or pd.isna(row[2]):
            return None
        cell_kji0 = np.empty((3,), dtype = int)
        cell_kji0[:] = row[2], row[1], row[0]
        cell_kji0[:] -= 1
        return cell_kji0

    @staticmethod
    def __verify_grid_name(grid_name_to_check, row, skipped_warning_grid, well_name):
        """Check whether the grid associated with a row of the dataframe matches the expected grid name."""
        skip_row = False
        if grid_name_to_check and pd.notna(row['GRID']) and grid_name_to_check != str(row['GRID']).upper():
            other_grid = str(row['GRID'])
            if skipped_warning_grid != other_grid:
                log.warning('skipping perforation(s) in grid ' + other_grid + ' for well ' + str(well_name))
                skipped_warning_grid = other_grid
                skip_row = True
        return skipped_warning_grid, skip_row

    @staticmethod
    def __calculate_entry_and_exit_axes_polarities_and_points(angles_present, row, cp, well_name, df, i, cell_kji0,
                                                              blocked_cells_kji0, use_face_centres, xy_units, z_units):
        if angles_present:
            entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz =  \
            BlockedWell.__calculate_entry_and_exit_axes_polarities_and_points_using_angles(
                row = row, cp = cp, well_name = well_name, xy_units = xy_units, z_units = z_units)
        else:
            # fabricate entry and exit axes and polarities based on indices alone
            # note: could use geometry but here a cheap rough-and-ready approach is used
            log.debug('row ' + str(i) + ': using cell moves')
            entry_axis, entry_polarity, exit_axis, exit_polarity = BlockedWell.__calculate_entry_and_exit_axes_polarities_and_points_using_indices(
                df = df, i = i, cell_kji0 = cell_kji0, blocked_cells_kji0 = blocked_cells_kji0)

        entry_xyz, exit_xyz = BlockedWell.__override_vector_based_xyz_entry_and_exit_points_if_necessary(
            use_face_centres = use_face_centres,
            entry_axis = entry_axis,
            exit_axis = exit_axis,
            entry_polarity = entry_polarity,
            exit_polarity = exit_polarity,
            cp = cp)

        return entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz

    @staticmethod
    def __calculate_entry_and_exit_axes_polarities_and_points_using_angles(row, cp, well_name, xy_units, z_units):
        """Calculate entry and exit axes, polarities and points using azimuth and inclination angles."""

        angla = row['ANGLA']
        inclination = row['ANGLV']
        if inclination < 0.001:
            azimuth = 0.0
        else:
            i_vector = np.sum(cp[:, :, 1] - cp[:, :, 0], axis = (0, 1))
            azimuth = vec.azimuth(i_vector) - angla  # see Nexus keyword reference doc
        well_vector = vec.unit_vector_from_azimuth_and_inclination(azimuth, inclination) * 10000.0
        if xy_units != z_units:
            well_vector[2] = wam.convert_lengths(well_vector[2], xy_units, z_units)
            well_vector = vec.unit_vector(well_vector) * 10000.0
        # todo: the following might be producing NaN's when vector passes precisely through an edge
        (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
            rqwu.find_entry_and_exit(cp, -well_vector, well_vector, well_name)
        return entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz

    def __calculate_entry_and_exit_axes_polarities_and_points_using_indices(df, i, cell_kji0, blocked_cells_kji0):

        entry_axis, entry_polarity = BlockedWell.__fabricate_entry_axis_and_polarity_using_indices(
            i, cell_kji0, blocked_cells_kji0)
        exit_axis, exit_polarity = BlockedWell.__fabricate_exit_axis_and_polarity_using_indices(
            i, cell_kji0, entry_axis, entry_polarity, df)

        return entry_axis, entry_polarity, exit_axis, exit_polarity

    @staticmethod
    def __fabricate_entry_axis_and_polarity_using_indices(i, cell_kji0, blocked_cells_kji0):
        """Fabricate entry and exit axes and polarities based on indices alone.

        note:
            could use geometry but here a cheap rough-and-ready approach is used
        """

        if i == 0:
            entry_axis, entry_polarity = 0, 0  # K-
        else:
            entry_move = cell_kji0 - blocked_cells_kji0[-1]
            log.debug(f'entry move: {entry_move}')
            if entry_move[1] == 0 and entry_move[2] == 0:  # K move
                entry_axis = 0
                entry_polarity = 0 if entry_move[0] >= 0 else 1
            elif abs(entry_move[1]) > abs(entry_move[2]):  # J dominant move
                entry_axis = 1
                entry_polarity = 0 if entry_move[1] >= 0 else 1
            else:  # I dominant move
                entry_axis = 2
                entry_polarity = 0 if entry_move[2] >= 0 else 1
        return entry_axis, entry_polarity

    @staticmethod
    def __fabricate_exit_axis_and_polarity_using_indices(i, cell_kji0, entry_axis, entry_polarity, df):
        if i == len(df) - 1:
            exit_axis, exit_polarity = entry_axis, 1 - entry_polarity
        else:
            next_cell_kji0 = BlockedWell.__cell_kji0_from_df(df, i + 1)
            if next_cell_kji0 is None:
                exit_axis, exit_polarity = entry_axis, 1 - entry_polarity
            else:
                exit_move = next_cell_kji0 - cell_kji0
                log.debug(f'exit move: {exit_move}')
                if exit_move[1] == 0 and exit_move[2] == 0:  # K move
                    exit_axis = 0
                    exit_polarity = 1 if exit_move[0] >= 0 else 0
                elif abs(exit_move[1]) > abs(exit_move[2]):  # J dominant move
                    exit_axis = 1
                    exit_polarity = 1 if exit_move[1] >= 0 else 0
                else:  # I dominant move
                    exit_axis = 2
                    exit_polarity = 1 if exit_move[2] >= 0 else 0
        return exit_axis, exit_polarity

    @staticmethod
    def __override_vector_based_xyz_entry_and_exit_points_if_necessary(use_face_centres, entry_axis, exit_axis,
                                                                       entry_polarity, exit_polarity, cp):
        """Override the vector based xyz entry and exit with face centres."""

        if use_face_centres:  # override the vector based xyz entry and exit points with face centres
            if entry_axis == 0:
                entry_xyz = np.mean(cp[entry_polarity, :, :], axis = (0, 1))
            elif entry_axis == 1:
                entry_xyz = np.mean(cp[:, entry_polarity, :], axis = (0, 1))
            else:
                entry_xyz = np.mean(cp[:, :, entry_polarity], axis = (0, 1))  # entry_axis == 2, ie. I
            if exit_axis == 0:
                exit_xyz = np.mean(cp[exit_polarity, :, :], axis = (0, 1))
            elif exit_axis == 1:
                exit_xyz = np.mean(cp[:, exit_polarity, :], axis = (0, 1))
            else:
                exit_xyz = np.mean(cp[:, :, exit_polarity], axis = (0, 1))  # exit_axis == 2, ie. I
            return entry_xyz, exit_xyz

    @staticmethod
    def __add_interval(previous_xyz, entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz,
                       cell_kji0, trajectory_mds, trajectory_points, blocked_intervals, blocked_cells_kji0,
                       blocked_face_pairs, xy_units, z_units, length_uom):
        if previous_xyz is None:  # first entry
            log.debug('adding mean sea level trajectory start')
            previous_xyz = entry_xyz.copy()
            previous_xyz[2] = 0.0  # use depth zero as md datum
            trajectory_mds.append(0.0)
            trajectory_points.append(previous_xyz)
        if not vec.isclose(previous_xyz, entry_xyz, tolerance = 0.05):  # add an unblocked interval
            log.debug('adding unblocked interval')
            trajectory_points.append(entry_xyz)
            new_md = trajectory_mds[-1] + BlockedWell._md_length(entry_xyz - previous_xyz, xy_units, z_units,
                                                                 length_uom)
            trajectory_mds.append(new_md)
            blocked_intervals.append(-1)  # unblocked interval
            previous_xyz = entry_xyz
        log.debug('adding blocked interval for cell kji0: ' + str(cell_kji0))
        trajectory_points.append(exit_xyz)
        new_md = trajectory_mds[-1] + BlockedWell._md_length(exit_xyz - previous_xyz, xy_units, z_units, length_uom)
        trajectory_mds.append(new_md)
        blocked_intervals.append(0)  # blocked interval
        previous_xyz = exit_xyz
        blocked_cells_kji0.append(cell_kji0)
        blocked_face_pairs.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

        return previous_xyz, trajectory_mds, trajectory_points, blocked_intervals, blocked_cells_kji0, blocked_face_pairs

    @staticmethod
    def _md_length(xyz_vector, xy_units, z_units, length_uom):
        if length_uom == xy_units and length_uom == z_units:
            return vec.naive_length(xyz_vector)
        x = wam.convert_lengths(xyz_vector[0], xy_units, length_uom)
        y = wam.convert_lengths(xyz_vector[1], xy_units, length_uom)
        z = wam.convert_lengths(xyz_vector[2], z_units, length_uom)
        return vec.naive_length((x, y, z))

    @staticmethod
    def __add_tail_to_trajectory_if_necessary(blocked_count, exit_axis, exit_polarity, cell_kji0, grid,
                                              trajectory_points, trajectory_mds):
        """Add tail to trajcetory if last segment terminates at bottom face in bottom layer."""

        if blocked_count > 0 and exit_axis == 0 and exit_polarity == 1 and cell_kji0[
                0] == grid.nk - 1 and grid.k_direction_is_down:
            tail_length = 10.0  # metres or feet
            tail_xyz = trajectory_points[-1].copy()
            tail_xyz[2] += tail_length * (1.0 if grid.z_inc_down() else -1.0)
            trajectory_points.append(tail_xyz)
            new_md = trajectory_mds[-1] + tail_length
            trajectory_mds.append(new_md)

        return trajectory_points, trajectory_mds

    def __add_as_properties_if_specified(self,
                                         add_as_properties,
                                         df,
                                         length_uom,
                                         time_index = None,
                                         time_series_uuid = None):
        # if add_as_properties is True and present as a list of wellspec column names, both the blocked well and
        # the properties will have their hdf5 data written, xml created and be added as parts to the model

        if add_as_properties and len(df.columns) > 3:
            # NB: atypical writing of hdf5 data and xml creation in order to support related properties
            self.write_hdf5()
            self.create_xml()
            if isinstance(add_as_properties, list):
                for col in add_as_properties:
                    assert col in df.columns[3:]  # could just skip missing columns
                property_columns = add_as_properties
            else:
                property_columns = df.columns[3:]
            self.add_df_properties(df,
                                   property_columns,
                                   length_uom = length_uom,
                                   time_index = time_index,
                                   time_series_uuid = time_series_uuid)

    def import_from_rms_cellio(self,
                               cellio_file,
                               well_name,
                               grid,
                               include_overburden_unblocked_interval = False,
                               set_tangent_vectors = False):
        """Populates empty blocked well from RMS cell I/O data; creates simulation trajectory and md datum.

        arguments:
           cellio_file (string): path of RMS ascii export file holding blocked well cell I/O data; cell entry and
              exit points are expected
           well_name (string): the name of the well as used in the cell I/O file
           grid (grid.Grid object): the grid object which the cell indices in the cell I/O data relate to
           set_tangent_vectors (boolean, default False): if True, tangent vectors will be computed from the well
              trajectory's control points

        returns:
           self if successful; None otherwise
        """

        if well_name:
            self.well_name = well_name
        else:
            well_name = self.well_name
        if not self.title:
            self.title = well_name

        grid_name = rqet.citation_title_for_node(grid.root)
        length_uom = grid.z_units()
        grid_z_inc_down = crs.Crs(grid.model, uuid = grid.crs_uuid).z_inc_down
        log.debug('grid z increasing downwards: ' + str(grid_z_inc_down) + '(type: ' + str(type(grid_z_inc_down)) + ')')
        cellio_z_inc_down = None

        try:
            assert ' ' not in well_name, 'cannot import for well name containing spaces'
            with open(cellio_file, 'r') as fp:
                BlockedWell.__verify_header_lines_in_cellio_file(fp = fp,
                                                                 well_name = well_name,
                                                                 cellio_file = cellio_file)
                previous_xyz = None
                trajectory_mds = []
                trajectory_points = []  # entries paired with trajectory_mds
                blocked_intervals = [
                ]  # will have one fewer entries than trajectory nodes; 0 = blocked, -1 = not blocked (for grid indices)
                blocked_cells_kji0 = []  # will have length equal to number of 0's in blocked intervals
                blocked_face_pairs = [
                ]  # same length as blocked_cells_kji0; each is ((entry axis, entry polarity), (exit axis, exit polarity))

                while not kf.blank_line(fp):
                    line = fp.readline()
                    cell_kji0, entry_xyz, exit_xyz = BlockedWell.__parse_non_blank_line_in_cellio_file(
                        line = line,
                        grid = grid,
                        cellio_z_inc_down = cellio_z_inc_down,
                        grid_z_inc_down = grid_z_inc_down)

                    cp, cell_centre, entry_vector, exit_vector = BlockedWell.__calculate_cell_cp_center_and_vectors(
                        grid = grid,
                        cell_kji0 = cell_kji0,
                        entry_xyz = entry_xyz,
                        exit_xyz = exit_xyz,
                        well_name = well_name)

                    # let's hope everything is in the same coordinate reference system!
                    (entry_axis, entry_polarity, facial_entry_xyz, exit_axis, exit_polarity,
                     facial_exit_xyz) = rqwu.find_entry_and_exit(cp, entry_vector, exit_vector, well_name)

                    if previous_xyz is None:  # first entry
                        previous_xyz = entry_xyz.copy()
                        if include_overburden_unblocked_interval:
                            log.debug('adding mean sea level trajectory start')
                            previous_xyz[2] = 0.0  # use depth zero as md datum
                        trajectory_mds.append(previous_xyz[2])
                        trajectory_points.append(previous_xyz)

                    if not vec.isclose(previous_xyz, entry_xyz, tolerance = 0.05):  # add an unblocked interval
                        log.debug('adding unblocked interval')
                        trajectory_points.append(entry_xyz)
                        new_md = trajectory_mds[-1] + vec.naive_length(
                            entry_xyz - previous_xyz)  # assumes x, y & z units are same
                        trajectory_mds.append(new_md)
                        blocked_intervals.append(-1)  # unblocked interval
                        previous_xyz = entry_xyz

                    log.debug('adding blocked interval for cell kji0: ' + str(cell_kji0))
                    trajectory_points.append(exit_xyz)
                    new_md = trajectory_mds[-1] + vec.naive_length(
                        exit_xyz - previous_xyz)  # assumes x, y & z units are same
                    trajectory_mds.append(new_md)
                    blocked_intervals.append(0)  # blocked interval
                    previous_xyz = exit_xyz
                    blocked_cells_kji0.append(cell_kji0)
                    blocked_face_pairs.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

                BlockedWell.__check_number_of_blocked_well_intervals(blocked_cells_kji0 = blocked_cells_kji0,
                                                                     well_name = well_name,
                                                                     grid_name = grid_name)

                self.create_md_datum_and_trajectory(grid,
                                                    trajectory_mds,
                                                    trajectory_points,
                                                    length_uom,
                                                    well_name,
                                                    set_depth_zero = True,
                                                    set_tangent_vectors = set_tangent_vectors)

                self.node_count = len(trajectory_mds)
                self.node_mds = np.array(trajectory_mds)
                self.cell_count = len(blocked_cells_kji0)
                self.grid_indices = np.array(blocked_intervals,
                                             dtype = int)  # NB. only supporting one grid at the moment
                self.cell_indices = grid.natural_cell_indices(np.array(blocked_cells_kji0))
                self.face_pair_indices = np.array(blocked_face_pairs)
                self.grid_list = [grid]

        except Exception:
            log.exception('failed to import info for blocked well ' + str(well_name) + ' from cell I/O file ' +
                          str(cellio_file))
            return None

        return self

    @staticmethod
    def __verify_header_lines_in_cellio_file(fp, well_name, cellio_file):
        """Find and verify the information in the header lines for the specified well in the RMS cellio file."""
        while True:
            kf.skip_blank_lines_and_comments(fp)
            line = fp.readline()  # file format version number?
            assert line, 'well ' + str(well_name) + ' not found in file ' + str(cellio_file)
            fp.readline()  # 'Undefined'
            words = fp.readline().split()
            assert len(words), 'missing header info in cell I/O file'
            if words[0].upper() == well_name.upper():
                break
            while not kf.blank_line(fp):
                fp.readline()  # skip to block of data for next well
        header_lines = int(fp.readline().strip())
        for _ in range(header_lines):
            fp.readline()

    @staticmethod
    def __parse_non_blank_line_in_cellio_file(line, grid, cellio_z_inc_down, grid_z_inc_down):
        """Parse each non-blank line in the RMS cellio file for the relevant parameters."""

        words = line.split()
        assert len(words) >= 9, 'not enough items on data line in cell I/O file, minimum 9 expected'
        i1, j1, k1 = int(words[0]), int(words[1]), int(words[2])
        cell_kji0 = np.array((k1 - 1, j1 - 1, i1 - 1), dtype = int)
        assert np.all(0 <= cell_kji0) and np.all(
            cell_kji0 < grid.extent_kji), 'cell I/O cell index not within grid extent'
        entry_xyz = np.array((float(words[3]), float(words[4]), float(words[5])))
        exit_xyz = np.array((float(words[6]), float(words[7]), float(words[8])))
        if cellio_z_inc_down is None:
            cellio_z_inc_down = bool(entry_xyz[2] + exit_xyz[2] > 0.0)
        if cellio_z_inc_down != grid_z_inc_down:
            entry_xyz[2] = -entry_xyz[2]
            exit_xyz[2] = -exit_xyz[2]
        return cell_kji0, entry_xyz, exit_xyz

    @staticmethod
    def __calculate_cell_cp_center_and_vectors(grid, cell_kji0, entry_xyz, exit_xyz, well_name):
        # calculate the i,j,k coordinates that represent the corner points and center of a perforation cell
        # calculate the entry and exit vectors for the perforation cell

        cp = grid.corner_points(cell_kji0 = cell_kji0, cache_resqml_array = False)
        assert not np.any(np.isnan(
            cp)), 'missing geometry for perforation cell(kji0) ' + str(cell_kji0) + ' for well ' + str(well_name)
        cell_centre = np.mean(cp, axis = (0, 1, 2))
        # let's hope everything is in the same coordinate reference system!
        entry_vector = 100.0 * (entry_xyz - cell_centre)
        exit_vector = 100.0 * (exit_xyz - cell_centre)
        return cp, cell_centre, entry_vector, exit_vector

    @staticmethod
    def __check_number_of_blocked_well_intervals(blocked_cells_kji0, well_name, grid_name):
        """Check that at least one interval is blocked for the specified well."""

        blocked_count = len(blocked_cells_kji0)
        if blocked_count == 0:
            log.warning(f"No intervals blocked for well {well_name} in grid"
                        f"{f' {grid_name}' if grid_name is not None else ''}.")
            return None
        else:
            log.info(f"{blocked_count} interval{rqwu._pl(blocked_count)} blocked for well {well_name} in"
                     f" grid{f' {grid_name}' if grid_name is not None else ''}.")

    def dataframe(self,
                  i_col = 'IW',
                  j_col = 'JW',
                  k_col = 'L',
                  one_based = True,
                  extra_columns_list = [],
                  ntg_uuid = None,
                  perm_i_uuid = None,
                  perm_j_uuid = None,
                  perm_k_uuid = None,
                  satw_uuid = None,
                  sato_uuid = None,
                  satg_uuid = None,
                  region_uuid = None,
                  radw = None,
                  skin = None,
                  stat = None,
                  active_only = False,
                  min_k0 = None,
                  max_k0 = None,
                  k0_list = None,
                  min_length = None,
                  min_kh = None,
                  max_depth = None,
                  max_satw = None,
                  min_sato = None,
                  max_satg = None,
                  perforation_list = None,
                  region_list = None,
                  depth_inc_down = None,
                  set_k_face_intervals_vertical = False,
                  anglv_ref = 'normal ij down',
                  angla_plane_ref = None,
                  length_mode = 'MD',
                  length_uom = None,
                  use_face_centres = False,
                  preferential_perforation = True,
                  add_as_properties = False,
                  use_properties = False,
                  property_time_index = None,
                  time_series_uuid = None):
        """Returns a pandas data frame containing Nexus WELLSPEC style data.

        arguments:
           i_col (string, default 'IW'): the column name to use for cell I index values
           j_col (string, default 'JW'): the column name to use for cell J index values
           k_col (string, default 'L'): the column name to use for cell K index values
           one_based (boolean, default True): if True, simulator protocol i, j & k values are placed in I, J & K columns;
              if False, resqml zero based values; this does not affect the interpretation of min_k0 & max_k0 arguments
           extra_columns_list (list of string, optional): list of WELLSPEC column names to include in the dataframe, from currently
              recognised values: 'GRID', 'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'RADW', 'SKIN', 'PPERF', 'RADB', 'WI', 'WBC'
           ntg_uuid (uuid.UUID, optional): the uuid of the net to gross ratio property; if present is used to downgrade the i & j
              permeabilities in the calculation of KH; ignored if 'KH' not in the extra column list and min_kh is not specified;
              the argument may also be a dictionary mapping from grid uuid to ntg uuid; if no net to gross data is provided, it
              is effectively assumed to be one (or, equivalently, the I & J permeability data is applicable to the gross rock); see
              also preferential_perforation argument which can cause adjustment of effective ntg in partially perforated cells
           perm_i_uuid (uuid.UUID or dictionary, optional): the uuid of the permeability property in the I direction;
              required if 'KH' is included in the extra columns list and min_kh is not specified; ignored otherwise;
              the argument may also be a dictionary mapping from grid uuid to perm I uuid
           perm_j_uuid (uuid.UUID, optional): the uuid (or dict) of the permeability property in the J direction;
              defaults to perm_i_uuid
           perm_k_uuid (uuid.UUID, optional): the uuid (or dict) of the permeability property in the K direction;
              defaults to perm_i_uuid
           satw_uuid (uuid.UUID, optional): the uuid of a water saturation property; required if max_satw is specified; may also
              be a dictionary mapping from grid uuid to satw uuid; ignored if max_satw is None
           sato_uuid (uuid.UUID, optional): the uuid of an oil saturation property; required if min_sato is specified; may also
              be a dictionary mapping from grid uuid to sato uuid; ignored if min_sato is None
           satg_uuid (uuid.UUID, optional): the uuid of a gas saturation property; required if max_satg is specified; may also
              be a dictionary mapping from grid uuid to satg uuid; ignored if max_satg is None
           region_uuid (uuid.UUID, optional): the uuid of a discrete or categorical property, required if region_list is not None;
              may also be a dictionary mapping from grid uuid to region uuid; ignored if region_list is None
           radw (float, optional): if present, the wellbore radius used for all perforations; must be in correct units for intended
              use of the WELLSPEC style dataframe; will default to 0.25 if 'RADW' is included in the extra column list
           skin (float, optional): if present, a skin column is included with values set to this constant
           stat (string, optional): if present, should be 'ON' or 'OFF' and is used for all perforations; will default to 'ON' if
              'STAT' is included in the extra column list
           active_only (boolean, default False): if True, only cells that are flagged in the grid object as active are included;
              if False, cells are included whether active or not
           min_k0 (int, optional): if present, perforations in layers above this are excluded (layer number will be applied
              naively to all grids – not recommended when working with more than one grid with different layering)
           max_k0 (int, optional): if present, perforations in layers below this are excluded (layer number will be applied
              naively to all grids – not recommended when working with more than one grid with different layering)
           k0_list (list of int, optional): if present, only perforations in cells in these layers are included (layer numbers
              will be applied naively to all grids – not recommended when working with more than one grid with different layering)
           min_length (float, optional): if present, a minimum length for an individual perforation interval to be included;
              units are the length units of the trajectory object unless length_uom argument is set
           min_kh (float, optional): if present, the minimum permeability x length value for which an individual interval is
              included; permeabilty uuid(s) must be supplied for the kh calculation; units of the length component are those
              of the trajectory object unless length_uom argument is set
           max_depth (float, optional): if present, rows are excluded for cells with a centre point depth greater than this value;
              max_depth should be positive downwards, with units of measure those of the grid z coordinates
           max_satw (float, optional): if present, perforations in cells where the water saturation exceeds this value will
              be excluded; satw_uuid must be supplied if this argument is present
           min_sato (float, optional): if present, perforations in cells where the oil saturation is less than this value will
              be excluded; sato_uuid must be supplied if this argument is present
           max_satg (float, optional): if present, perforations in cells where the gas saturation exceeds this value will
              be excluded; satg_uuid must be supplied if this argument is present
           perforation_list (list of (float, float), optional): if present, a list of perforated intervals; each entry is the
              start and end measured depths for a perforation; these do not need to align with cell boundaries
           region_list (list of int, optional): if present, a list of region numbers for which rows are to be included; the
              property holding the region data is identified by the region_uuid argument
           depth_inc_down (boolean, optional): if present and True, the depth values will increase with depth; if False or None,
              the direction of the depth values will be determined by the z increasing downwards indicator in the trajectory crs
           set_k_face_intervals_vertical (boolean, default False): if True, intervals with entry through K- and exit through K+
              will have angla and anglv set to 0.0 (vertical); if False angles will be computed depending on geometry
           anglv_ref (string, default 'normal ij down'): either 'gravity', 'z down' (same as gravity), 'z+', 'k down', 'k+',
              'normal ij', or 'normal ij down';
              the ANGLV angles are relative to a local (per cell) reference vector selected by this keyword
           angla_plane_ref (string, optional): string indicating normal vector defining plane onto which trajectory and I axis are
              projected for the calculation of ANGLA; options as for anglv_ref, or 'normal well i+' which results in no projection;
              defaults to the same as anglv_ref
           length_mode (string, default 'MD'): 'MD' or 'straight' indicating which length to use; 'md' takes measured depth
              difference between exit and entry; 'straight' uses a naive straight line length between entry and exit;
              this will affect values for LENGTH, KH, DEPTH, X & Y
           length_uom (string, optional): if present, either 'm' or 'ft': the length units to use for the LENGTH, KH, MD, DEPTH,
              X & Y columns if they are present in extra_columns_list; also used to interpret min_length and min_kh; if None, the
              length units of the trajectory attribute are used LENGTH, KH & MD and those of the grid are used for DEPTH, X & Y;
              RADW value, if present, is assumed to be in the correct units and is not changed; also used implicitly to determine
              conversion constant used in calculation of wellbore constant (WBC)
           use_face_centres (boolean, default False): if True, the centre points of the entry and exit faces will determine the
              vector used as the basis of ANGLA and ANGLV calculations; if False, the trajectory locations for the entry and exit
              measured depths will be used
           preferential_perforation (boolean, default True): if perforation_list is given, and KH is requested or a min_kh given,
              the perforated intervals are assumed to penetrate pay rock preferentially: an effective ntg weighting is computed
              to account for any residual non-pay perforated interval; ignored if perforation_list is None or kh values are not
              being computed
           add_as_properties (boolean or list of str, default False): if True, each column in the extra_columns_list (excluding
              GRID) is added as a property with the blocked well as supporting representation and 'cells' as the
              indexable element; any cell that is excluded from the dataframe will have corresponding entries of NaN in all the
              properties; if a list is provided it must be a subset of extra_columns_list
           use_properties (boolean or list of str, default False): if True, each column in the extra_columns_list (excluding
              GRID) is populated from a property with citation title matching the column name, if it exists
           property_time_index (int, optional): if present and use_properties is True, the time index to select properties for;
              if add_as_properties is True, the time index to tag this set of properties with
           time_series_uuid (UUID, optional): the uuid of the time series for time dependent properties being added

        notes:
           units of length along wellbore will be those of the trajectory's length_uom (also applies to K.H values) unless
           the length_uom argument is used;
           the constraints are applied independently for each row and a row is excluded if it fails any constraint;
           the min_k0 and max_k0 arguments do not stop later rows within the layer range from being included;
           the min_length and min_kh limits apply to individual cell intervals and thus depend on cell size;
           the water and oil saturation limits are for saturations at a single time and affect whether the interval
           is included in the dataframe – there is no functionality to support turning perforations off and on over time;
           the saturation limits do not stop deeper intervals with qualifying saturations from being included;
           the k0_list, perforation_list and region_list arguments should be set to None to disable the corresponding functionality,
           if set to an empty list, no rows will be included in the dataframe;
           if add_as_properties is True, the blocked well must already have been added as a part to the model;
           add_as_properties and use_properties cannot both be True;
           add_as_properties and use_properties are only currently functional for single grid blocked wells;
           at present, unit conversion is not handled when using properties

        :meta common:
        """

        assert length_mode in ['MD', 'straight']
        assert length_uom is None or length_uom in ['m', 'ft']

        anglv_ref, angla_plane_ref = BlockedWell.__verify_angle_references(anglv_ref, angla_plane_ref)
        column_list = [i_col, j_col, k_col]

        column_list, add_as_properties, use_properties, skin, stat, radw = BlockedWell.__verify_extra_properties_to_be_added_to_dataframe(
            extra_columns_list = extra_columns_list,
            column_list = column_list,
            add_as_properties = add_as_properties,
            use_properties = use_properties,
            skin = skin,
            stat = stat,
            radw = radw)

        pc = None
        if use_properties:
            pc = rqp.PropertyCollection(support = self)
            if property_time_index is not None:
                pc = rqp.selective_version_of_collection(pc, time_index = property_time_index)
            if pc is None or pc.number_of_parts() == 0:
                log.error(
                    f'no blocked well properties found for time index {property_time_index} for well {self.title}')
                pc = None
        pc_titles = [] if pc is None else pc.titles()

        max_satw, min_sato, max_satg = BlockedWell.__verify_saturation_ranges_and_property_uuids(
            max_satw, min_sato, max_satg, satw_uuid, sato_uuid, satg_uuid)

        min_kh, doing_kh = BlockedWell.__verify_perm_i_uuid_for_kh(min_kh = min_kh,
                                                                   column_list = column_list,
                                                                   perm_i_uuid = perm_i_uuid,
                                                                   pc_titles = pc_titles)

        do_well_inflow = BlockedWell.__verify_perm_i_uuid_for_well_inflow(column_list = column_list,
                                                                          perm_i_uuid = perm_i_uuid,
                                                                          pc_titles = pc_titles)

        perm_j_uuid, perm_k_uuid, isotropic_perm = BlockedWell.__verify_perm_j_k_uuids_for_kh_and_well_inflow(
            doing_kh = doing_kh,
            do_well_inflow = do_well_inflow,
            perm_i_uuid = perm_i_uuid,
            perm_j_uuid = perm_j_uuid,
            perm_k_uuid = perm_k_uuid)

        if min_length is not None and min_length <= 0.0:
            min_length = None

        if region_list is not None:
            assert region_uuid is not None, 'region list specified without region property array'

        BlockedWell.__check_perforation_properties_to_be_added(column_list = column_list,
                                                               perforation_list = perforation_list)

        BlockedWell.__verify_k_layers_to_be_included(min_k0 = min_k0, max_k0 = max_k0, k0_list = k0_list)

        doing_angles, doing_xyz, doing_entry_exit = BlockedWell.__verify_if_angles_xyz_and_length_to_be_added(
            column_list = column_list,
            pc_titles = pc_titles,
            doing_kh = doing_kh,
            do_well_inflow = do_well_inflow,
            length_mode = length_mode)

        grid_crs_list = self.__verify_number_of_grids_and_crs_units(column_list = column_list)

        k_face_check = np.zeros((2, 2), dtype = int)
        k_face_check[1, 1] = 1  # now represents entry, exit of K-, K+
        k_face_check_end = k_face_check.copy()
        k_face_check_end[1] = -1  # entry through K-, terminating (TD) within cell

        traj_crs, traj_z_inc_down = self.__get_trajectory_crs_and_z_inc_down()

        df = pd.DataFrame(columns = column_list)
        df = df.astype({i_col: int, j_col: int, k_col: int})

        ci = -1
        row_ci_list = []
        interval_count = self.__get_interval_count()

        for interval in range(interval_count):

            if self.grid_indices[interval] < 0:
                continue  # unblocked interval

            ci += 1
            row_dict = {}
            grid = self.grid_list[self.grid_indices[interval]]
            grid_crs = grid_crs_list[self.grid_indices[interval]]
            grid_name = rqet.citation_title_for_node(grid.root).replace(' ', '_')
            natural_cell = self.cell_indices[ci]
            cell_kji0 = grid.denaturalized_cell_index(natural_cell)
            tuple_kji0 = tuple(cell_kji0)

            skip_interval = BlockedWell.__skip_interval_check(max_depth = max_depth,
                                                              grid = grid,
                                                              cell_kji0 = cell_kji0,
                                                              grid_crs = grid_crs,
                                                              active_only = active_only,
                                                              tuple_kji0 = tuple_kji0,
                                                              min_k0 = min_k0,
                                                              max_k0 = max_k0,
                                                              k0_list = k0_list,
                                                              region_list = region_list,
                                                              region_uuid = region_uuid,
                                                              max_satw = max_satw,
                                                              satw_uuid = satw_uuid,
                                                              min_sato = min_sato,
                                                              sato_uuid = sato_uuid,
                                                              max_satg = max_satg,
                                                              satg_uuid = satg_uuid)
            if skip_interval:
                continue

            skip_interval_due_to_perforations, part_perf_fraction = self.__get_part_perf_fraction_for_interval(
                pc = pc, pc_titles = pc_titles, perforation_list = perforation_list, ci = ci, interval = interval)

            if skip_interval_due_to_perforations:
                continue

            entry_xyz, exit_xyz, ee_crs = self.__get_entry_exit_xyz_and_crs_for_interval(
                doing_entry_exit = doing_entry_exit,
                use_face_centres = use_face_centres,
                grid = grid,
                cell_kji0 = cell_kji0,
                interval = interval,
                ci = ci,
                grid_crs = grid_crs,
                traj_crs = traj_crs)

            skip_interval_due_to_invalid_length, length = self.__get_length_of_interval(
                length_mode = length_mode,
                interval = interval,
                length_uom = length_uom,
                entry_xyz = entry_xyz,
                exit_xyz = exit_xyz,
                ee_crs = ee_crs,
                perforation_list = perforation_list,
                part_perf_fraction = part_perf_fraction,
                min_length = min_length)

            if skip_interval_due_to_invalid_length:
                continue

            md = 0.5 * (self.node_mds[interval + 1] + self.node_mds[interval])

            anglv, sine_anglv, cosine_anglv, angla, sine_angla, cosine_angla = self.__get_angles_for_interval(
                pc = pc,
                pc_titles = pc_titles,
                doing_angles = doing_angles,
                set_k_face_intervals_vertical = set_k_face_intervals_vertical,
                ci = ci,
                k_face_check = k_face_check,
                k_face_check_end = k_face_check_end,
                entry_xyz = entry_xyz,
                exit_xyz = exit_xyz,
                ee_crs = ee_crs,
                traj_z_inc_down = traj_z_inc_down,
                grid = grid,
                grid_crs = grid_crs,
                cell_kji0 = cell_kji0,
                anglv_ref = anglv_ref,
                angla_plane_ref = angla_plane_ref)

            ntg_is_one, k_i, k_j, k_k = BlockedWell.__get_ntg_and_directional_perm_for_interval(
                doing_kh = doing_kh,
                do_well_inflow = do_well_inflow,
                ntg_uuid = ntg_uuid,
                grid = grid,
                tuple_kji0 = tuple_kji0,
                isotropic_perm = isotropic_perm,
                preferential_perforation = preferential_perforation,
                part_perf_fraction = part_perf_fraction,
                perm_i_uuid = perm_i_uuid,
                perm_j_uuid = perm_j_uuid,
                perm_k_uuid = perm_k_uuid)

            skip_interval_due_to_min_kh, kh = BlockedWell.__get_kh_for_interval(doing_kh = doing_kh,
                                                                                isotropic_perm = isotropic_perm,
                                                                                ntg_is_one = ntg_is_one,
                                                                                length = length,
                                                                                perm_i_uuid = perm_i_uuid,
                                                                                grid = grid,
                                                                                tuple_kji0 = tuple_kji0,
                                                                                k_i = k_i,
                                                                                k_j = k_j,
                                                                                k_k = k_k,
                                                                                anglv = anglv,
                                                                                sine_anglv = sine_anglv,
                                                                                cosine_anglv = cosine_anglv,
                                                                                sine_angla = sine_angla,
                                                                                cosine_angla = cosine_angla,
                                                                                min_kh = min_kh,
                                                                                pc = pc,
                                                                                pc_titles = pc_titles,
                                                                                ci = ci)

            if skip_interval_due_to_min_kh:
                continue

            length, radw, skin, radb, wi, wbc = BlockedWell.__get_pc_arrays_for_interval(pc = pc,
                                                                                         pc_titles = pc_titles,
                                                                                         ci = ci,
                                                                                         length = length,
                                                                                         radw = radw,
                                                                                         skin = skin)

            radb, wi, wbc = BlockedWell.__get_well_inflow_parameters_for_interval(do_well_inflow = do_well_inflow,
                                                                                  isotropic_perm = isotropic_perm,
                                                                                  ntg_is_one = ntg_is_one,
                                                                                  k_i = k_i,
                                                                                  k_j = k_j,
                                                                                  k_k = k_k,
                                                                                  sine_anglv = sine_anglv,
                                                                                  cosine_anglv = cosine_anglv,
                                                                                  sine_angla = sine_angla,
                                                                                  cosine_angla = cosine_angla,
                                                                                  grid = grid,
                                                                                  cell_kji0 = cell_kji0,
                                                                                  radw = radw,
                                                                                  radb = radb,
                                                                                  wi = wi,
                                                                                  wbc = wbc,
                                                                                  skin = skin,
                                                                                  kh = kh,
                                                                                  length_uom = length_uom,
                                                                                  column_list = column_list)

            xyz = self.__get_xyz_for_interval(doing_xyz = doing_xyz,
                                              length_mode = length_mode,
                                              length_uom = length_uom,
                                              md = md,
                                              traj_crs = traj_crs,
                                              depth_inc_down = depth_inc_down,
                                              traj_z_inc_down = traj_z_inc_down,
                                              entry_xyz = entry_xyz,
                                              exit_xyz = exit_xyz,
                                              ee_crs = ee_crs,
                                              pc = pc,
                                              pc_titles = pc_titles,
                                              ci = ci)

            md = self.__get_md_array_in_correct_units_for_interval(md = md,
                                                                   length_uom = length_uom,
                                                                   pc = pc,
                                                                   pc_titles = pc_titles,
                                                                   ci = ci)

            df = BlockedWell.__append_interval_data_to_dataframe(df = df,
                                                                 grid_name = grid_name,
                                                                 radw = radw,
                                                                 skin = skin,
                                                                 angla = angla,
                                                                 anglv = anglv,
                                                                 length = length,
                                                                 kh = kh,
                                                                 xyz = xyz,
                                                                 md = md,
                                                                 stat = stat,
                                                                 part_perf_fraction = part_perf_fraction,
                                                                 radb = radb,
                                                                 wi = wi,
                                                                 wbc = wbc,
                                                                 column_list = column_list,
                                                                 one_based = one_based,
                                                                 row_dict = row_dict,
                                                                 cell_kji0 = cell_kji0,
                                                                 row_ci_list = row_ci_list,
                                                                 ci = ci)

        self.__add_as_properties(df = df,
                                 add_as_properties = add_as_properties,
                                 extra_columns_list = extra_columns_list,
                                 length_uom = length_uom,
                                 time_index = property_time_index,
                                 time_series_uuid = time_series_uuid)

        return df

    def __get_interval_count(self):
        """Get the number of intervals to be added to the dataframe."""

        if self.node_count is None or self.node_count < 2:
            interval_count = 0
        else:
            interval_count = self.node_count - 1

        return interval_count

    @staticmethod
    def __prop_array(uuid_or_dict, grid):
        assert uuid_or_dict is not None and grid is not None
        if isinstance(uuid_or_dict, dict):
            prop_uuid = uuid_or_dict[grid.uuid]
        else:
            prop_uuid = uuid_or_dict  # uuid either in form of string or uuid.UUID
        return grid.property_collection.single_array_ref(uuid = prop_uuid)

    @staticmethod
    def __get_ref_vector(grid, grid_crs, cell_kji0, mode):
        # returns unit vector with true direction, ie. accounts for differing xy & z units in grid's crs
        # gravity = np.array((0.0, 0.0, 1.0))
        if mode == 'normal well i+':
            return None  # ANGLA only: option for no projection onto a plane
        ref_vector = None
        # options for anglv or angla reference: 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down'
        if mode == 'z+':
            ref_vector = np.array((0.0, 0.0, 1.0))
        elif mode == 'z down':
            if grid_crs.z_inc_down:
                ref_vector = np.array((0.0, 0.0, 1.0))
            else:
                ref_vector = np.array((0.0, 0.0, -1.0))
        else:
            cell_axial_vectors = grid.interface_vectors_kji(cell_kji0)
            if grid_crs.xy_units != grid_crs.z_units:
                wam.convert_lengths(cell_axial_vectors[..., 2], grid_crs.z_units, grid_crs.xy_units)
            if mode in ['k+', 'k down']:
                ref_vector = vec.unit_vector(cell_axial_vectors[0])
                if mode == 'k down' and not grid.k_direction_is_down:
                    ref_vector = -ref_vector
            else:  # normal to plane of ij axes
                ref_vector = vec.unit_vector(vec.cross_product(cell_axial_vectors[1], cell_axial_vectors[2]))
                if mode == 'normal ij down':
                    if grid_crs.z_inc_down:
                        if ref_vector[2] < 0.0:
                            ref_vector = -ref_vector
                    else:
                        if ref_vector[2] > 0.0:
                            ref_vector = -ref_vector
            if ref_vector is None or ref_vector[2] == 0.0:
                if grid_crs.z_inc_down:
                    ref_vector = np.array((0.0, 0.0, 1.0))
                else:
                    ref_vector = np.array((0.0, 0.0, -1.0))
        return ref_vector

    @staticmethod
    def __verify_angle_references(anglv_ref, angla_plane_ref):
        """Verify that the references for anglv and angla are one of the acceptable options."""

        assert anglv_ref in ['gravity', 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down']
        if anglv_ref == 'gravity':
            anglv_ref = 'z down'
        if angla_plane_ref is None:
            angla_plane_ref = anglv_ref
        assert angla_plane_ref in [
            'gravity', 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down', 'normal well i+'
        ]
        if angla_plane_ref == 'gravity':
            angla_plane_ref = 'z down'
        return anglv_ref, angla_plane_ref

    @staticmethod
    def __verify_saturation_ranges_and_property_uuids(max_satw, min_sato, max_satg, satw_uuid, sato_uuid, satg_uuid):
        # verify that the fluid saturation limits fall within 0.0 to 1.0 and that the uuid of the required
        # saturation property array has been specified.

        if max_satw is not None and max_satw >= 1.0:
            max_satw = None
        if min_sato is not None and min_sato <= 0.0:
            min_sato = None
        if max_satg is not None and max_satg >= 1.0:
            max_satg = None

        phase_list = ['water', 'oil', 'gas']
        phase_saturation_limits_list = [max_satw, min_sato, max_satg]
        uuids_list = [satw_uuid, sato_uuid, satg_uuid]

        for phase, phase_limit, uuid in zip(phase_list, phase_saturation_limits_list, uuids_list):
            if phase_limit is not None:
                assert uuid is not None, f'{phase} saturation limit specified without saturation property array'

        return max_satw, min_sato, max_satg

    @staticmethod
    def __verify_extra_properties_to_be_added_to_dataframe(extra_columns_list, column_list, add_as_properties,
                                                           use_properties, skin, stat, radw):
        """Determine which extra columns, if any, should be added as properties to the dataframe.

        note:
            if skin, stat or radw are None, default values are specified
        """

        if extra_columns_list:
            for extra in extra_columns_list:
                assert extra.upper() in [
                    'GRID', 'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'SKIN', 'RADW', 'PPERF', 'RADB',
                    'WI', 'WBC', 'STAT'
                ]
                column_list.append(extra.upper())
        else:
            add_as_properties = use_properties = False
        assert not (add_as_properties and use_properties)

        column_list, skin, stat, radw = BlockedWell.__check_skin_stat_radw_to_be_added_as_properties(
            skin = skin, stat = stat, radw = radw, column_list = column_list)

        return column_list, add_as_properties, use_properties, skin, stat, radw

    @staticmethod
    def __check_perforation_properties_to_be_added(column_list, perforation_list):

        if all(['LENGTH' in column_list, 'PPERF' in column_list, 'KH' not in column_list, perforation_list
                is not None]):
            log.warning(
                'both LENGTH and PPERF will include effects of partial perforation; only one should be used in WELLSPEC'
            )
        elif all([
                perforation_list is not None, 'LENGTH' not in column_list, 'PPERF' not in column_list, 'KH'
                not in column_list, 'WBC' not in column_list
        ]):
            log.warning('perforation list supplied but no use of LENGTH, KH, PPERF nor WBC')

        if perforation_list is not None and len(perforation_list) == 0:
            log.warning('empty perforation list specified for blocked well dataframe: no rows will be included')

    @staticmethod
    def __check_skin_stat_radw_to_be_added_as_properties(skin, stat, radw, column_list):
        """Verify whether skin should be added as a property in the dataframe."""

        if skin is not None and 'SKIN' not in column_list:
            column_list.append('SKIN')
        if skin is None:
            skin = 0.0

        if stat is not None:
            assert str(stat).upper() in ['ON', 'OFF']
            stat = str(stat).upper()
            if 'STAT' not in column_list:
                column_list.append('STAT')
        else:
            stat = 'ON'

        if radw is not None and 'RADW' not in column_list:
            column_list.append('RADW')
        if radw is None:
            radw = 0.25

        return column_list, skin, stat, radw

    @staticmethod
    def __verify_perm_i_uuid_for_well_inflow(column_list, perm_i_uuid, pc_titles):
        # Verify that the I direction permeability has been specified if well inflow properties are to be added
        # to the dataframe.

        do_well_inflow = (('WI' in column_list and 'WI' not in pc_titles) or
                          ('WBC' in column_list and 'WBC' not in pc_titles) or
                          ('RADB' in column_list and 'RADB' not in pc_titles))
        if do_well_inflow:
            assert perm_i_uuid is not None, 'WI, RADB or WBC requested without I direction permeabilty being specified'

        return do_well_inflow

    @staticmethod
    def __verify_perm_i_uuid_for_kh(min_kh, column_list, perm_i_uuid, pc_titles):
        # verify that the I direction permeability has been specified if permeability thickness and
        # wellbore constant properties are to be added to the dataframe.

        if min_kh is not None and min_kh <= 0.0:
            min_kh = None
        doing_kh = False
        if ('KH' in column_list or min_kh is not None) and 'KH' not in pc_titles:
            assert perm_i_uuid is not None, 'KH requested (or minimum specified) without I direction permeabilty being specified'
            doing_kh = True
        if 'WBC' in column_list and 'WBC' not in pc_titles:
            assert perm_i_uuid is not None, 'WBC requested without I direction permeabilty being specified'
            doing_kh = True

        return min_kh, doing_kh

    @staticmethod
    def __verify_perm_j_k_uuids_for_kh_and_well_inflow(doing_kh, do_well_inflow, perm_i_uuid, perm_j_uuid, perm_k_uuid):
        # verify that the J and K direction permeabilities have been specified if well inflow properties or
        # permeability thickness properties are to be added to the dataframe.

        isotropic_perm = None
        if doing_kh or do_well_inflow:
            if perm_j_uuid is None and perm_k_uuid is None:
                isotropic_perm = True
            else:
                if perm_j_uuid is None:
                    perm_j_uuid = perm_i_uuid
                if perm_k_uuid is None:
                    perm_k_uuid = perm_i_uuid
                # following line assumes arguments are passed in same form; if not, some unnecessary maths might be done
                isotropic_perm = (bu.matching_uuids(perm_i_uuid, perm_j_uuid) and
                                  bu.matching_uuids(perm_i_uuid, perm_k_uuid))

        return perm_j_uuid, perm_k_uuid, isotropic_perm

    @staticmethod
    def __verify_k_layers_to_be_included(min_k0, max_k0, k0_list):
        # verify that the k layers to be included in the dataframe exist within the appropriate range

        if min_k0 is None:
            min_k0 = 0
        else:
            assert min_k0 >= 0
        if max_k0 is not None:
            assert min_k0 <= max_k0
        if k0_list is not None and len(k0_list) == 0:
            log.warning('no layers included for blocked well dataframe: no rows will be included')

    @staticmethod
    def __verify_if_angles_xyz_and_length_to_be_added(column_list, pc_titles, doing_kh, do_well_inflow, length_mode):
        # determine if angla, anglv, x, y, z and length data are to be added as properties to the dataframe

        doing_angles = any([('ANGLA' in column_list and 'ANGLA' not in pc_titles),
                            ('ANGLV' in column_list and 'ANGLV' not in pc_titles), (doing_kh), (do_well_inflow)])
        doing_xyz = any([('X' in column_list and 'X' not in pc_titles), ('Y' in column_list and 'Y' not in pc_titles),
                         ('DEPTH' in column_list and 'DEPTH' not in pc_titles)])
        doing_entry_exit = any([(doing_angles),
                                ('LENGTH' in column_list and 'LENGTH' not in pc_titles and length_mode == 'straight')])

        # doing_angles = (('ANGLA' in column_list and 'ANGLA' not in pc_titles) or
        #                 ('ANGLV' in column_list and 'ANGLV' not in pc_titles) or doing_kh or do_well_inflow)
        # doing_xyz = (('X' in column_list and 'X' not in pc_titles) or (
        #             'Y' in column_list and 'Y' not in pc_titles) or
        #              ('DEPTH' in column_list and 'DEPTH' not in pc_titles))
        # doing_entry_exit = doing_angles or ('LENGTH' in column_list and 'LENGTH' not in pc_titles and
        #                                     length_mode == 'straight')

        return doing_angles, doing_xyz, doing_entry_exit

    def __verify_number_of_grids_and_crs_units(self, column_list):
        # verify that a GRID column is included in the dataframe if the well intersects more than one grid
        # verify that each grid's crs units are consistent in all directions

        if 'GRID' not in column_list and self.number_of_grids() > 1:
            log.error('creating blocked well dataframe without GRID column for well that intersects more than one grid')
        grid_crs_list = []
        for grid in self.grid_list:
            grid_crs = crs.Crs(self.model, uuid = grid.crs_uuid)
            grid_crs_list.append(grid_crs)
        return grid_crs_list

    def __get_trajectory_crs_and_z_inc_down(self):

        if self.trajectory is None or self.trajectory.crs_uuid is None:
            traj_crs = None
            traj_z_inc_down = None
        else:
            traj_crs = crs.Crs(self.trajectory.model, uuid = self.trajectory.crs_uuid)
            traj_z_inc_down = traj_crs.z_inc_down

        return traj_crs, traj_z_inc_down

    @staticmethod
    def __check_cell_depth(max_depth, grid, cell_kji0, grid_crs):
        """Check whether the maximum depth specified has been exceeded with the current interval."""

        max_depth_exceeded = False
        if max_depth is not None:
            cell_depth = grid.centre_point(cell_kji0)[2]
            if not grid_crs.z_inc_down:
                cell_depth = -cell_depth
            if cell_depth > max_depth:
                max_depth_exceeded = True
        return max_depth_exceeded

    @staticmethod
    def __skip_interval_check(max_depth, grid, cell_kji0, grid_crs, active_only, tuple_kji0, min_k0, max_k0, k0_list,
                              region_list, region_uuid, max_satw, satw_uuid, min_sato, sato_uuid, max_satg, satg_uuid):
        """Check whether any conditions are met that mean the interval should be skipped."""

        max_depth_exceeded = BlockedWell.__check_cell_depth(max_depth = max_depth,
                                                            grid = grid,
                                                            cell_kji0 = cell_kji0,
                                                            grid_crs = grid_crs)
        inactive_grid = active_only and grid.inactive is not None and grid.inactive[tuple_kji0]
        out_of_bounds_layer_1 = (min_k0 is not None and cell_kji0[0] < min_k0) or (max_k0 is not None and
                                                                                   cell_kji0[0] > max_k0)
        out_of_bounds_layer_2 = k0_list is not None and cell_kji0[0] not in k0_list
        out_of_bounds_region = (region_list is not None and
                                BlockedWell.__prop_array(region_uuid, grid)[tuple_kji0] not in region_list)
        saturation_limit_exceeded_1 = (max_satw is not None and
                                       BlockedWell.__prop_array(satw_uuid, grid)[tuple_kji0] > max_satw)
        saturation_limit_exceeded_2 = (min_sato is not None and
                                       BlockedWell.__prop_array(sato_uuid, grid)[tuple_kji0] < min_sato)
        saturation_limit_exceeded_3 = (max_satg is not None and
                                       BlockedWell.__prop_array(satg_uuid, grid)[tuple_kji0] > max_satg)
        skip_interval = any([
            max_depth_exceeded, inactive_grid, out_of_bounds_layer_1, out_of_bounds_layer_2, out_of_bounds_region,
            saturation_limit_exceeded_1, saturation_limit_exceeded_2, saturation_limit_exceeded_3
        ])

        return skip_interval

    def __get_part_perf_fraction_for_interval(self, pc, pc_titles, ci, perforation_list, interval, length_tol = 0.01):
        """Get the partial perforation fraction for the interval."""

        skip_interval = False
        if 'PPERF' in pc_titles:
            part_perf_fraction = pc.single_array_ref(citation_title = 'PPERF')[ci]
        else:
            part_perf_fraction = 1.0
            if perforation_list is not None:
                perf_length = 0.0
                for perf_start, perf_end in perforation_list:
                    if perf_end <= self.node_mds[interval] or perf_start >= self.node_mds[interval + 1]:
                        continue
                    if perf_start <= self.node_mds[interval]:
                        if perf_end >= self.node_mds[interval + 1]:
                            perf_length += self.node_mds[interval + 1] - self.node_mds[interval]
                            break
                        else:
                            perf_length += perf_end - self.node_mds[interval]
                    else:
                        if perf_end >= self.node_mds[interval + 1]:
                            perf_length += self.node_mds[interval + 1] - perf_start
                        else:
                            perf_length += perf_end - perf_start
                if perf_length < length_tol:
                    skip_interval = True
                    perf_length = 0.0
                part_perf_fraction = min(1.0, perf_length / (self.node_mds[interval + 1] - self.node_mds[interval]))

        return skip_interval, part_perf_fraction

    def __get_entry_exit_xyz_and_crs_for_interval(self, doing_entry_exit, use_face_centres, grid, cell_kji0, interval,
                                                  ci, grid_crs, traj_crs):
        # calculate the entry and exit points for the interval and set the entry and exit coordinate reference system

        entry_xyz = None
        exit_xyz = None
        ee_crs = None
        if doing_entry_exit:
            assert self.trajectory is not None
            if use_face_centres:
                entry_xyz = grid.face_centre(cell_kji0, self.face_pair_indices[ci, 0, 0], self.face_pair_indices[ci, 0,
                                                                                                                 1])
                if self.face_pair_indices[ci, 1, 0] >= 0:
                    exit_xyz = grid.face_centre(cell_kji0, self.face_pair_indices[ci, 1, 0],
                                                self.face_pair_indices[ci, 1, 1])
                else:
                    exit_xyz = grid.face_centre(cell_kji0, self.face_pair_indices[ci, 0, 0],
                                                1 - self.face_pair_indices[ci, 0, 1])
                ee_crs = grid_crs
            else:
                entry_xyz = self.trajectory.xyz_for_md(self.node_mds[interval])
                exit_xyz = self.trajectory.xyz_for_md(self.node_mds[interval + 1])
                ee_crs = traj_crs

        return entry_xyz, exit_xyz, ee_crs

    def __get_length_of_interval(self, length_mode, interval, length_uom, entry_xyz, exit_xyz, ee_crs, perforation_list,
                                 part_perf_fraction, min_length):
        """Calculate the length of the interval."""

        skip_interval = False
        if length_mode == 'MD':
            length = self.node_mds[interval + 1] - self.node_mds[interval]
            if length_uom is not None and self.trajectory is not None and length_uom != self.trajectory.md_uom:
                length = wam.convert_lengths(length, self.trajectory.md_uom, length_uom)
        else:  # use straight line length between entry and exit
            entry_xyz, exit_xyz = BlockedWell._single_uom_entry_exit_xyz(entry_xyz, exit_xyz, ee_crs)
            length = vec.naive_length(exit_xyz - entry_xyz)
            if length_uom is not None:
                length = wam.convert_lengths(length, ee_crs.z_units, length_uom)
            elif self.trajectory is not None:
                length = wam.convert_lengths(length, ee_crs.z_units, self.trajectory.md_uom)
        if perforation_list is not None:
            length *= part_perf_fraction
        if min_length is not None and length < min_length:
            skip_interval = True

        return skip_interval, length

    @staticmethod
    def _single_uom_xyz(xyz, crs, required_uom):
        xyz = np.array(xyz, dtype = float)
        if crs.xy_units != required_uom:
            xyz[0] = wam.convert_lengths(xyz[0], crs.xy_units, required_uom)
            xyz[1] = wam.convert_lengths(xyz[1], crs.xy_units, required_uom)
        if crs.z_units != required_uom:
            xyz[2] = wam.convert_lengths(xyz[2], crs.z_units, required_uom)
        return xyz

    @staticmethod
    def _single_uom_entry_exit_xyz(entry_xyz, exit_xyz, ee_crs):
        return (BlockedWell._single_uom_xyz(entry_xyz, ee_crs, ee_crs.z_units),
                BlockedWell._single_uom_xyz(exit_xyz, ee_crs, ee_crs.z_units))

    def __get_angles_for_interval(self, pc, pc_titles, doing_angles, set_k_face_intervals_vertical, ci, k_face_check,
                                  k_face_check_end, entry_xyz, exit_xyz, ee_crs, traj_z_inc_down, grid, grid_crs,
                                  cell_kji0, anglv_ref, angla_plane_ref):
        """Calculate angla, anglv and related trigonometirc transforms for the interval."""

        sine_anglv = sine_angla = 0.0
        cosine_anglv = cosine_angla = 1.0
        anglv = pc.single_array_ref(citation_title = 'ANGLV')[ci] if 'ANGLV' in pc_titles else None
        angla = pc.single_array_ref(citation_title = 'ANGLA')[ci] if 'ANGLA' in pc_titles else None

        if doing_angles and not (set_k_face_intervals_vertical and
                                 (np.all(self.face_pair_indices[ci] == k_face_check) or
                                  np.all(self.face_pair_indices[ci] == k_face_check_end))):
            anglv, sine_anglv, cosine_anglv, vector, a_ref_vector = BlockedWell.__get_anglv_for_interval(
                anglv = anglv,
                entry_xyz = entry_xyz,
                exit_xyz = exit_xyz,
                ee_crs = ee_crs,
                traj_z_inc_down = traj_z_inc_down,
                grid = grid,
                grid_crs = grid_crs,
                cell_kji0 = cell_kji0,
                anglv_ref = anglv_ref,
                angla_plane_ref = angla_plane_ref)
            if anglv != 0.0:
                angla, sine_angla, cosine_angla = BlockedWell.__get_angla_for_interval(angla = angla,
                                                                                       grid = grid,
                                                                                       cell_kji0 = cell_kji0,
                                                                                       vector = vector,
                                                                                       a_ref_vector = a_ref_vector)
        else:
            if angla is None:
                angla = 0.0
            if anglv is None:
                anglv = 0.0

        return anglv, sine_anglv, cosine_anglv, angla, sine_angla, cosine_angla

    @staticmethod
    def __get_angla_for_interval(angla, grid, cell_kji0, vector, a_ref_vector):
        """Calculate angla and related trigonometric transforms for the interval."""

        # project well vector and i-axis vector onto plane defined by normal vector a_ref_vector
        i_axis = grid.interface_vector(cell_kji0, 2)
        if grid.crs.xy_units != grid.crs.z_units:
            i_axis[2] = wam.convert_lengths(i_axis[2], grid.crs.z_units, grid.crs.xy_units)
        i_axis = vec.unit_vector(i_axis)
        if a_ref_vector is not None:  # project vector and i axis onto a plane
            vector -= vec.dot_product(vector, a_ref_vector) * a_ref_vector
            vector = vec.unit_vector(vector)
            # log.debug('i axis unit vector: ' + str(i_axis))
            i_axis -= vec.dot_product(i_axis, a_ref_vector) * a_ref_vector
            i_axis = vec.unit_vector(i_axis)
        # log.debug('i axis unit vector in reference plane: ' + str(i_axis))
        if angla is not None:
            angla_rad = vec.radians_from_degrees(angla)
            cosine_angla = maths.cos(angla_rad)
            sine_angla = maths.sin(angla_rad)
        else:
            cosine_angla = min(max(vec.dot_product(vector, i_axis), -1.0), 1.0)
            angla_rad = maths.acos(cosine_angla)
            # negate angla if vector is 'clockwise from' i_axis when viewed from above, projected in the xy plane
            # todo: have discussion around angla sign under different ijk handedness (and z inc direction?)
            sine_angla = maths.sin(angla_rad)
            angla = vec.degrees_from_radians(angla_rad)
            if vec.clockwise((0.0, 0.0), i_axis, vector) > 0.0:
                angla = -angla
                angla_rad = -angla_rad  ## as angle_rad before --> typo?
                sine_angla = -sine_angla

        # log.debug('angla: ' + str(angla))

        return angla, sine_angla, cosine_angla

    @staticmethod
    def __get_anglv_for_interval(anglv, entry_xyz, exit_xyz, ee_crs, traj_z_inc_down, grid, grid_crs, cell_kji0,
                                 anglv_ref, angla_plane_ref):
        """Get anglv and related trigonometric transforms for the interval."""

        entry_xyz, exit_xyz = BlockedWell._single_uom_entry_exit_xyz(entry_xyz, exit_xyz, ee_crs)
        vector = vec.unit_vector(np.array(exit_xyz) - np.array(entry_xyz))  # nominal wellbore vector for interval
        if traj_z_inc_down is not None and traj_z_inc_down != grid_crs.z_inc_down:
            vector[2] = -vector[2]
        if grid.crs.xy_units == grid.crs.z_units:
            unit_adjusted_vector = vector
        else:
            unit_adjusted_vector = vector.copy()
            unit_adjusted_vector[2] = wam.convert_lengths(unit_adjusted_vector[2], grid.crs.z_units, grid.crs.xy_units)
        v_ref_vector = BlockedWell.__get_ref_vector(grid, grid_crs, cell_kji0, anglv_ref)
        # log.debug('v ref vector: ' + str(v_ref_vector))
        if angla_plane_ref == anglv_ref:
            a_ref_vector = v_ref_vector
        else:
            a_ref_vector = BlockedWell.__get_ref_vector(grid, grid_crs, cell_kji0, angla_plane_ref)
        # log.debug('a ref vector: ' + str(a_ref_vector))
        if anglv is not None:
            anglv_rad = vec.radians_from_degrees(anglv)
            cosine_anglv = maths.cos(anglv_rad)
            sine_anglv = maths.sin(anglv_rad)
        else:
            cosine_anglv = min(max(vec.dot_product(unit_adjusted_vector, v_ref_vector), -1.0), 1.0)
            anglv_rad = maths.acos(cosine_anglv)
            sine_anglv = maths.sin(anglv_rad)
            anglv = vec.degrees_from_radians(anglv_rad)
        # log.debug('anglv: ' + str(anglv))

        return anglv, sine_anglv, cosine_anglv, vector, a_ref_vector

    @staticmethod
    def __get_ntg_and_directional_perm_for_interval(doing_kh, do_well_inflow, ntg_uuid, grid, tuple_kji0,
                                                    isotropic_perm, preferential_perforation, part_perf_fraction,
                                                    perm_i_uuid, perm_j_uuid, perm_k_uuid):
        """Get the net-to-gross and directional permeability arrays for the interval."""

        ntg_is_one = False
        k_i = k_j = k_k = None
        if doing_kh or do_well_inflow:
            if ntg_uuid is None:
                ntg = 1.0
                ntg_is_one = True
            else:
                ntg = BlockedWell.__prop_array(ntg_uuid, grid)[tuple_kji0]
                ntg_is_one = maths.isclose(ntg, 1.0, rel_tol = 0.001)
            if isotropic_perm and ntg_is_one:
                k_i = k_j = k_k = BlockedWell.__prop_array(perm_i_uuid, grid)[tuple_kji0]
            else:
                if preferential_perforation and not ntg_is_one:
                    if part_perf_fraction <= ntg:
                        ntg = 1.0  # effective ntg when perforated intervals are in pay
                    else:
                        ntg /= part_perf_fraction  # adjusted ntg when some perforations in non-pay
                # todo: check netgross facet type in property perm i & j parts: if set to gross then don't multiply by ntg below
                k_i = BlockedWell.__prop_array(perm_i_uuid, grid)[tuple_kji0] * ntg
                k_j = BlockedWell.__prop_array(perm_j_uuid, grid)[tuple_kji0] * ntg
                k_k = BlockedWell.__prop_array(perm_k_uuid, grid)[tuple_kji0]

        return ntg_is_one, k_i, k_j, k_k

    @staticmethod
    def __get_kh_for_interval(doing_kh, isotropic_perm, ntg_is_one, length, perm_i_uuid, grid, tuple_kji0, k_i, k_j,
                              k_k, anglv, sine_anglv, cosine_anglv, sine_angla, cosine_angla, min_kh, pc, pc_titles,
                              ci):
        """Get the permeability-thickness value for the interval."""

        skip_interval = False
        if doing_kh:
            kh = BlockedWell.__get_kh_if_doing_kh(isotropic_perm = isotropic_perm,
                                                  ntg_is_one = ntg_is_one,
                                                  length = length,
                                                  perm_i_uuid = perm_i_uuid,
                                                  grid = grid,
                                                  tuple_kji0 = tuple_kji0,
                                                  k_i = k_i,
                                                  k_j = k_j,
                                                  k_k = k_k,
                                                  anglv = anglv,
                                                  sine_anglv = sine_anglv,
                                                  cosine_anglv = cosine_anglv,
                                                  sine_angla = sine_angla,
                                                  cosine_angla = cosine_angla)
            if min_kh is not None and kh < min_kh:
                skip_interval = True
        elif 'KH' in pc_titles:
            kh = pc.single_array_ref(citation_title = 'KH')[ci]
        else:
            kh = None
        return skip_interval, kh

    @staticmethod
    def __get_kh_if_doing_kh(isotropic_perm, ntg_is_one, length, perm_i_uuid, grid, tuple_kji0, k_i, k_j, k_k, anglv,
                             sine_anglv, cosine_anglv, sine_angla, cosine_angla):
        # note: this is believed to return required value even when grid crs has mixed xy & z units;
        # angles are true angles accounting for any mixed units
        if isotropic_perm and ntg_is_one:
            kh = length * BlockedWell.__prop_array(perm_i_uuid, grid)[tuple_kji0]
        else:
            if np.isnan(k_i) or np.isnan(k_j):
                kh = 0.0
            elif anglv == 0.0:
                kh = length * maths.sqrt(k_i * k_j)
            elif np.isnan(k_k):
                kh = 0.0
            else:
                k_e = maths.pow(k_i * k_j * k_k, 1.0 / 3.0)
                if k_e == 0.0:
                    kh = 0.0
                else:
                    l_i = length * maths.sqrt(k_e / k_i) * sine_anglv * cosine_angla
                    l_j = length * maths.sqrt(k_e / k_j) * sine_anglv * sine_angla
                    l_k = length * maths.sqrt(k_e / k_k) * cosine_anglv
                    l_p = maths.sqrt(l_i * l_i + l_j * l_j + l_k * l_k)
                    kh = k_e * l_p
        return kh

    @staticmethod
    def __get_pc_arrays_for_interval(pc, pc_titles, ci, length, radw, skin):
        """Get the property collection arrays for the interval."""

        if 'LENGTH' in pc_titles:
            length = pc.single_array_ref(citation_title = 'LENGTH')[ci]
        if 'RADW' in pc_titles:
            radw = pc.single_array_ref(citation_title = 'RADW')[ci]
        assert radw > 0.0  # todo: allow zero for inactive intervals?
        if 'SKIN' in pc_titles:
            skin = pc.single_array_ref(citation_title = 'SKIN')[ci]
        radb = wi = wbc = None
        if 'RADB' in pc_titles:
            radb = pc.single_array_ref(citation_title = 'RADB')[ci]
        if 'WI' in pc_titles:
            wi = pc.single_array_ref(citation_title = 'WI')[ci]
        if 'WBC' in pc_titles:
            wbc = pc.single_array_ref(citation_title = 'WBC')[ci]

        return length, radw, skin, radb, wi, wbc

    @staticmethod
    def __get_well_inflow_parameters_for_interval(do_well_inflow, isotropic_perm, ntg_is_one, k_i, k_j, k_k, sine_anglv,
                                                  cosine_anglv, sine_angla, cosine_angla, grid, cell_kji0, radw, radb,
                                                  wi, wbc, skin, kh, length_uom, column_list):

        if do_well_inflow:
            if not length_uom:
                length_uom = grid.crs.z_units
            k_ei, k_ej, k_ek, radw_e = BlockedWell.__calculate_ke_and_radw_e(isotropic_perm = isotropic_perm,
                                                                             ntg_is_one = ntg_is_one,
                                                                             radw = radw,
                                                                             k_i = k_i,
                                                                             k_j = k_j,
                                                                             k_k = k_k,
                                                                             sine_anglv = sine_anglv,
                                                                             cosine_anglv = cosine_anglv,
                                                                             sine_angla = sine_angla,
                                                                             cosine_angla = cosine_angla)

            cell_axial_vectors = grid.interface_vectors_kji(cell_kji0)
            wam.convert_lengths(cell_axial_vectors[..., :2], grid.crs.xy_units, length_uom)
            wam.convert_lengths(cell_axial_vectors[..., 2], grid.crs.z_units, length_uom)
            d2 = np.empty(3)
            for axis in range(3):
                d2[axis] = np.sum(cell_axial_vectors[axis] * cell_axial_vectors[axis])
            radb_e = BlockedWell.__calculate_radb_e(k_ei = k_ei,
                                                    k_ej = k_ej,
                                                    k_ek = k_ek,
                                                    k_i = k_i,
                                                    k_j = k_j,
                                                    k_k = k_k,
                                                    d2 = d2,
                                                    sine_anglv = sine_anglv,
                                                    cosine_anglv = cosine_anglv,
                                                    sine_angla = sine_angla,
                                                    cosine_angla = cosine_angla)

            if radb is None:
                radb = radw * radb_e / radw_e
            if wi is None:
                wi = 0.0 if radb <= 0.0 else 2.0 * maths.pi / (maths.log(radb / radw) + skin)
            if 'WBC' in column_list and wbc is None:
                assert length_uom == 'm' or length_uom.startswith('ft'),  \
                    'WBC only calculable for length uom of m or ft*'
                conversion_constant = 8.5270171e-5 if length_uom == 'm' else 0.006328286
                wbc = conversion_constant * kh * wi  # note: pperf aleady accounted for in kh

        return radb, wi, wbc

    @staticmethod
    def __calculate_ke_and_radw_e(isotropic_perm, ntg_is_one, radw, k_i, k_j, k_k, sine_anglv, cosine_anglv, sine_angla,
                                  cosine_angla):

        if isotropic_perm and ntg_is_one:
            k_ei = k_ej = k_ek = k_i
            radw_e = radw
        else:
            k_ei = maths.sqrt(k_j * k_k)
            k_ej = maths.sqrt(k_i * k_k)
            k_ek = maths.sqrt(k_i * k_j)
            r_wi = 0.0 if k_ei == 0.0 else 0.5 * radw * (maths.sqrt(k_ei / k_j) + maths.sqrt(k_ei / k_k))
            r_wj = 0.0 if k_ej == 0.0 else 0.5 * radw * (maths.sqrt(k_ej / k_i) + maths.sqrt(k_ej / k_k))
            r_wk = 0.0 if k_ek == 0.0 else 0.5 * radw * (maths.sqrt(k_ek / k_i) + maths.sqrt(k_ek / k_j))
            rwi = r_wi * sine_anglv * cosine_angla
            rwj = r_wj * sine_anglv * sine_angla
            rwk = r_wk * cosine_anglv
            radw_e = maths.sqrt(rwi * rwi + rwj * rwj + rwk * rwk)
            if radw_e == 0.0:
                radw_e = radw  # no permeability in this situation anyway

        return k_ei, k_ej, k_ek, radw_e

    @staticmethod
    def __calculate_radb_e(k_ei, k_ej, k_ek, k_i, k_j, k_k, d2, sine_anglv, cosine_anglv, sine_angla, cosine_angla):

        r_bi = 0.0 if k_ei == 0.0 else 0.14 * maths.sqrt(k_ei * (d2[1] / k_j + d2[0] / k_k))
        r_bj = 0.0 if k_ej == 0.0 else 0.14 * maths.sqrt(k_ej * (d2[2] / k_i + d2[0] / k_k))
        r_bk = 0.0 if k_ek == 0.0 else 0.14 * maths.sqrt(k_ek * (d2[2] / k_i + d2[1] / k_j))
        rbi = r_bi * sine_anglv * cosine_angla
        rbj = r_bj * sine_anglv * sine_angla
        rbk = r_bk * cosine_anglv
        radb_e = maths.sqrt(rbi * rbi + rbj * rbj + rbk * rbk)

        return radb_e

    def __get_xyz_for_interval(self, doing_xyz, length_mode, length_uom, md, traj_crs, depth_inc_down, traj_z_inc_down,
                               entry_xyz, exit_xyz, ee_crs, pc, pc_titles, ci):
        """Get the x, y and z location of the midpoint of the interval."""

        xyz = (np.NaN, np.NaN, np.NaN)
        if doing_xyz:
            xyz = self.__get_xyz_if_doing_xyz(length_mode = length_mode,
                                              md = md,
                                              length_uom = length_uom,
                                              traj_crs = traj_crs,
                                              depth_inc_down = depth_inc_down,
                                              traj_z_inc_down = traj_z_inc_down,
                                              entry_xyz = entry_xyz,
                                              exit_xyz = exit_xyz,
                                              ee_crs = ee_crs)
        xyz = np.array(xyz)
        for i, col_header in enumerate(['X', 'Y', 'DEPTH']):
            if col_header in pc_titles:
                xyz[i] = pc.single_array_ref(citation_title = col_header)[ci]

        return xyz

    def __get_xyz_if_doing_xyz(self, length_mode, md, length_uom, traj_crs, depth_inc_down, traj_z_inc_down, exit_xyz,
                               entry_xyz, ee_crs):

        if length_mode == 'MD' and self.trajectory is not None:
            xyz = self.trajectory.xyz_for_md(md)
            if length_uom is not None and length_uom != self.trajectory.md_uom:
                wam.convert_lengths(xyz, traj_crs.z_units, length_uom)
            if depth_inc_down and traj_z_inc_down is False:
                xyz[2] = -xyz[2]
        else:
            xyz = 0.5 * (np.array(exit_xyz) + np.array(entry_xyz))
            if length_uom is not None and length_uom != ee_crs.z_units:
                xyz[2] = wam.convert_lengths(xyz[2], ee_crs.z_units, length_uom)
            if depth_inc_down != ee_crs.z_inc_down:
                xyz[2] = -xyz[2]

        return xyz

    def __get_md_array_in_correct_units_for_interval(self, md, length_uom, pc, pc_titles, ci):
        """Convert the measured depth to the correct units or get the measured depth from the property collection."""

        if 'MD' in pc_titles:
            md = pc.single_array_ref(citation_title = 'MD')[ci]
        elif length_uom is not None and self.trajectory is not None and length_uom != self.trajectory.md_uom:
            md = wam.convert_lengths(md, self.trajectory.md_uom, length_uom)

        return md

    @staticmethod
    def __append_interval_data_to_dataframe(df, grid_name, radw, skin, angla, anglv, length, kh, xyz, md, stat,
                                            part_perf_fraction, radb, wi, wbc, column_list, one_based, row_dict,
                                            cell_kji0, row_ci_list, ci):
        """Append the row of data corresponding to the interval to the dataframe."""

        column_names = [
            'GRID', 'RADW', 'SKIN', 'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'STAT', 'PPERF', 'RADB',
            'WI', 'WBC'
        ]
        column_values = [
            grid_name, radw, skin, angla, anglv, length, kh, xyz[2], md, xyz[0], xyz[1], stat, part_perf_fraction, radb,
            wi, wbc
        ]
        column_values_dict = dict(zip(column_names, column_values))

        data = df.to_dict()
        data = {k: list(v.values()) for k, v in data.items()}
        for col_index, col in enumerate(column_list):
            if col_index < 3:
                if one_based:
                    row_dict[col] = [cell_kji0[2 - col_index] + 1]
                else:
                    row_dict[col] = [cell_kji0[2 - col_index]]
            else:
                row_dict[col] = [column_values_dict[col]]

        for col, vals in row_dict.items():
            if col in data:
                data[col].extend(vals)
            else:
                data[col] = vals
        df = pd.DataFrame(data)

        row_ci_list.append(ci)

        return df

    def __add_as_properties(self,
                            df,
                            add_as_properties,
                            extra_columns_list,
                            length_uom,
                            time_index = None,
                            time_series_uuid = None):
        """Adds property parts from df with columns listed in add_as_properties or extra_columns_list."""

        if add_as_properties:
            if isinstance(add_as_properties, list):
                for col in add_as_properties:
                    assert col in extra_columns_list
                property_columns = add_as_properties
            else:
                property_columns = extra_columns_list
            self.add_df_properties(df,
                                   property_columns,
                                   length_uom = length_uom,
                                   time_index = time_index,
                                   time_series_uuid = time_series_uuid)

    def add_df_properties(self,
                          df,
                          columns,
                          length_uom = None,
                          time_index = None,
                          time_series_uuid = None,
                          realization = None):
        """Creates a property part for each column in the dataframe, based on the dataframe values.

        arguments:
            df (pd.DataFrame): dataframe containing the columns that will be converted to properties
            columns (List[str]): list of the column names that will be converted to properties
            length_uom (str, optional): the length unit of measure
            time_index (int, optional): if adding a timestamp to the property, this is the timestamp
                index of the TimeSeries timestamps attribute
            time_series_uuid (uuid.UUID, optional): if adding a timestamp to the property, this is
                the uuid of the TimeSeries object
            realization (int, optional): if present, is used as the realization number for all the
                properties

        returns:
            None

        notes:
            the column name is used as the property citation title;
            the blocked well must already exist as a part in the model;
            this method currently only handles single grid situations;
            dataframe rows must be in the same order as the cells in the blocked well
        """
        # todo: enhance to handle multiple grids
        assert len(self.grid_list) == 1
        if columns is None or len(columns) == 0 or len(df) == 0:
            return
        if length_uom is None:
            length_uom = self.trajectory.md_uom
        extra_pc = rqp.PropertyCollection()
        extra_pc.set_support(support = self)
        assert len(df) == self.cell_count

        for column in columns:
            extra = column.upper()
            uom, pk, discrete = self.__set_uom_pk_discrete_for_df_properties(extra = extra, length_uom = length_uom)
            if discrete:
                null_value = -1
                na_value = -1
                dtype = np.int32
            else:
                null_value = None
                na_value = np.NaN
                dtype = float
            # 'SKIN': use defaults for now; todo: create local property kind for skin
            if column == 'STAT':
                col_as_list = list(df[column])
                expanded = np.array([(0 if (str(st).upper() in ['OFF', '0']) else 1) for st in col_as_list],
                                    dtype = int)
            else:
                expanded = df[column].to_numpy(dtype = dtype, copy = True, na_value = na_value)
            extra_pc.add_cached_array_to_imported_list(
                expanded,
                'blocked well dataframe',
                extra,
                discrete = discrete,
                uom = uom,
                property_kind = pk,
                local_property_kind_uuid = None,
                facet_type = None,
                facet = None,
                realization = realization,
                indexable_element = 'cells',
                count = 1,
                time_index = time_index,
                null_value = null_value,
            )
        extra_pc.write_hdf5_for_imported_list()
        extra_pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = time_series_uuid,
                                                                     find_local_property_kinds = True)

    def __set_uom_pk_discrete_for_df_properties(self, extra, length_uom, temperature_uom = None):
        """Set the property kind and unit of measure for all properties in the dataframe."""
        if length_uom not in ['m', 'ft']:
            raise ValueError(f"The length_uom {length_uom} must be either 'm' or 'ft'.")
        if extra == 'TEMP' and (temperature_uom is None or
                                temperature_uom not in wam.valid_uoms('thermodynamic temperature')):
            raise ValueError(f"The temperature_uom must be in {wam.valid_uoms('thermodynamic temperature')}.")

        length_uom_pk_discrete = self.__set_uom_pk_discrete_for_length_based_properties(length_uom = length_uom,
                                                                                        extra = extra)
        uom_pk_discrete_dict = {
            'ANGLA': ('dega', 'azimuth', False),
            'ANGLV': ('dega', 'inclination', False),
            'KH': (f'mD.{length_uom}', 'permeability length', False),
            'PPERF': (f'{length_uom}/{length_uom}', 'perforation fraction', False),
            'STAT': (None, 'well connection status', True),
            'LENGTH': length_uom_pk_discrete,
            'MD': length_uom_pk_discrete,
            'X': length_uom_pk_discrete,
            'Y': length_uom_pk_discrete,
            'DEPTH': (length_uom, 'depth', False),
            'RADW': (length_uom, 'wellbore radius', False),
            'RADB': (length_uom, 'cell equivalent radius', False),
            'RADBP': length_uom_pk_discrete,
            'RADWP': length_uom_pk_discrete,
            'FM': (f'{length_uom}/{length_uom}', 'matrix fraction', False),
            'IRELPM': (None, 'relative permeability index', True),
            'SECT': (None, 'wellbore section index', True),
            'LAYER': (None, 'layer index', True),
            'ANGLE': ('dega', 'plane angle', False),
            'TEMP': (temperature_uom, 'thermodynamic temperature', False),
            'MDCON': length_uom_pk_discrete,
            'K': ('mD', 'permeability rock', False),
            'DZ': (length_uom, 'cell length', False),
            'DTOP': (length_uom, 'depth', False),
            'DBOT': (length_uom, 'depth', False),
            'SKIN': ('Euc', 'skin', False),
            'WI': ('Euc', 'well connection index', False),
        }
        return uom_pk_discrete_dict.get(extra, ('Euc', 'continuous', False))

    def __set_uom_pk_discrete_for_length_based_properties(self, length_uom, extra):
        if length_uom is None or length_uom == 'Euc':
            if extra in ['LENGTH', 'MD', 'MDCON']:
                uom = self.trajectory.md_uom
            elif extra in ['X', 'Y', 'RADW', 'RADB', 'RADBP', 'RADWP']:
                uom = self.grid_list[0].xy_units()
            else:
                uom = self.grid_list[0].z_units()
        else:
            uom = length_uom
        if extra == 'DEPTH':
            pk = 'depth'
        else:
            pk = 'length'
        return uom, pk, False

    def static_kh(self,
                  ntg_uuid = None,
                  perm_i_uuid = None,
                  perm_j_uuid = None,
                  perm_k_uuid = None,
                  satw_uuid = None,
                  sato_uuid = None,
                  satg_uuid = None,
                  region_uuid = None,
                  active_only = False,
                  min_k0 = None,
                  max_k0 = None,
                  k0_list = None,
                  min_length = None,
                  min_kh = None,
                  max_depth = None,
                  max_satw = None,
                  min_sato = None,
                  max_satg = None,
                  perforation_list = None,
                  region_list = None,
                  set_k_face_intervals_vertical = False,
                  anglv_ref = 'gravity',
                  angla_plane_ref = None,
                  length_mode = 'MD',
                  length_uom = None,
                  use_face_centres = False,
                  preferential_perforation = True):
        """Returns the total static K.H (permeability x height).

        notes:
           length units are those of trajectory md_uom unless length_upm is set;
           see doc string for dataframe() method for argument descriptions; perm_i_uuid required
        """

        df = self.dataframe(i_col = 'I',
                            j_col = 'J',
                            k_col = 'K',
                            one_based = False,
                            extra_columns_list = ['KH'],
                            ntg_uuid = ntg_uuid,
                            perm_i_uuid = perm_i_uuid,
                            perm_j_uuid = perm_j_uuid,
                            perm_k_uuid = perm_k_uuid,
                            satw_uuid = satw_uuid,
                            sato_uuid = sato_uuid,
                            satg_uuid = satg_uuid,
                            region_uuid = region_uuid,
                            active_only = active_only,
                            min_k0 = min_k0,
                            max_k0 = max_k0,
                            k0_list = k0_list,
                            min_length = min_length,
                            min_kh = min_kh,
                            max_depth = max_depth,
                            max_satw = max_satw,
                            min_sato = min_sato,
                            max_satg = max_satg,
                            perforation_list = perforation_list,
                            region_list = region_list,
                            set_k_face_intervals_vertical = set_k_face_intervals_vertical,
                            anglv_ref = anglv_ref,
                            angla_plane_ref = angla_plane_ref,
                            length_mode = length_mode,
                            length_uom = length_uom,
                            use_face_centres = use_face_centres,
                            preferential_perforation = preferential_perforation)

        return sum(df['KH'])

    def write_wellspec(self,
                       wellspec_file,
                       well_name = None,
                       mode = 'a',
                       extra_columns_list = [],
                       ntg_uuid = None,
                       perm_i_uuid = None,
                       perm_j_uuid = None,
                       perm_k_uuid = None,
                       satw_uuid = None,
                       sato_uuid = None,
                       satg_uuid = None,
                       region_uuid = None,
                       radw = None,
                       skin = None,
                       stat = None,
                       active_only = False,
                       min_k0 = None,
                       max_k0 = None,
                       k0_list = None,
                       min_length = None,
                       min_kh = None,
                       max_depth = None,
                       max_satw = None,
                       min_sato = None,
                       max_satg = None,
                       perforation_list = None,
                       region_list = None,
                       set_k_face_intervals_vertical = False,
                       depth_inc_down = True,
                       anglv_ref = 'gravity',
                       angla_plane_ref = None,
                       length_mode = 'MD',
                       length_uom = None,
                       preferential_perforation = True,
                       space_instead_of_tab_separator = True,
                       align_columns = True,
                       preceeding_blank_lines = 0,
                       trailing_blank_lines = 0,
                       length_uom_comment = False,
                       write_nexus_units = True,
                       float_format = '5.3',
                       use_properties = False,
                       property_time_index = None):
        """Writes Nexus WELLSPEC keyword to an ascii file.

        returns:
           pandas DataFrame containing data that has been written to the wellspec file

        note:
           see doc string for dataframe() method for most of the argument descriptions;
           align_columns and float_format arguments are deprecated and no longer used
        """

        assert wellspec_file, 'no output file specified to write WELLSPEC to'

        col_width_dict = {
            'IW': 4,
            'JW': 4,
            'L': 4,
            'ANGLA': 8,
            'ANGLV': 8,
            'LENGTH': 8,
            'KH': 10,
            'DEPTH': 10,
            'MD': 10,
            'X': 8,
            'Y': 12,
            'SKIN': 7,
            'RADW': 5,
            'RADB': 8,
            'PPERF': 5
        }

        well_name = self.__get_well_name(well_name = well_name)

        df = self.dataframe(one_based = True,
                            extra_columns_list = extra_columns_list,
                            ntg_uuid = ntg_uuid,
                            perm_i_uuid = perm_i_uuid,
                            perm_j_uuid = perm_j_uuid,
                            perm_k_uuid = perm_k_uuid,
                            satw_uuid = satw_uuid,
                            sato_uuid = sato_uuid,
                            satg_uuid = satg_uuid,
                            region_uuid = region_uuid,
                            radw = radw,
                            skin = skin,
                            stat = stat,
                            active_only = active_only,
                            min_k0 = min_k0,
                            max_k0 = max_k0,
                            k0_list = k0_list,
                            min_length = min_length,
                            min_kh = min_kh,
                            max_depth = max_depth,
                            max_satw = max_satw,
                            min_sato = min_sato,
                            max_satg = max_satg,
                            perforation_list = perforation_list,
                            region_list = region_list,
                            depth_inc_down = depth_inc_down,
                            set_k_face_intervals_vertical = set_k_face_intervals_vertical,
                            anglv_ref = anglv_ref,
                            angla_plane_ref = angla_plane_ref,
                            length_mode = length_mode,
                            length_uom = length_uom,
                            preferential_perforation = preferential_perforation,
                            use_properties = use_properties,
                            property_time_index = property_time_index)

        sep = ' ' if space_instead_of_tab_separator else '\t'

        with open(wellspec_file, mode = mode) as fp:
            for _ in range(preceeding_blank_lines):
                fp.write('\n')

            self.__write_wellspec_file_units_metadata(write_nexus_units = write_nexus_units,
                                                      fp = fp,
                                                      length_uom = length_uom,
                                                      length_uom_comment = length_uom_comment,
                                                      extra_columns_list = extra_columns_list,
                                                      well_name = well_name)

            BlockedWell.__write_wellspec_file_columns(df = df, fp = fp, col_width_dict = col_width_dict, sep = sep)

            fp.write('\n')

            BlockedWell.__write_wellspec_file_rows_from_dataframe(df = df,
                                                                  fp = fp,
                                                                  col_width_dict = col_width_dict,
                                                                  sep = sep)
            for _ in range(trailing_blank_lines):
                fp.write('\n')

        return df

    @staticmethod
    def __tidy_well_name(well_name):
        nexus_friendly = ''
        previous_underscore = False
        for ch in well_name:
            if not 32 <= ord(ch) < 128 or ch in ' ,!*#':
                ch = '_'
            if not (previous_underscore and ch == '_'):
                nexus_friendly += ch
            previous_underscore = (ch == '_')
        if not nexus_friendly:
            well_name = 'WELL_X'
        return nexus_friendly

    @staticmethod
    def __is_float_column(col_name):
        if col_name.upper() in [
                'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'SKIN', 'RADW', 'RADB', 'PPERF'
        ]:
            return True
        return False

    @staticmethod
    def __is_int_column(col_name):
        if col_name.upper() in ['IW', 'JW', 'L']:
            return True
        return False

    def __get_well_name(self, well_name):
        """Get the name of the well whose data is to be written to the Nexus WELLSPEC file."""

        if not well_name:
            if self.well_name:
                well_name = self.well_name
            elif self.root is not None:
                well_name = rqet.citation_title_for_node(self.root)
            elif self.wellbore_interpretation is not None:
                well_name = self.wellbore_interpretation.title
            elif self.trajectory is not None:
                well_name = self.trajectory.title
            if not well_name:
                log.warning('no well name identified for use in WELLSPEC')
                well_name = 'WELLNAME'
        well_name = BlockedWell.__tidy_well_name(well_name)

        return well_name

    def __write_wellspec_file_units_metadata(self, write_nexus_units, fp, length_uom, length_uom_comment,
                                             extra_columns_list, well_name):
        # write the units of measure (uom) and system of measure for length in the WELLSPEC file
        # also write a comment on the length uom if necessary

        if write_nexus_units:
            length_uom_system_list = ['METRIC', 'ENGLISH']
            length_uom_index = ['m', 'ft'].index(length_uom)
            fp.write(f'{length_uom_system_list[length_uom_index]}\n\n')

        if length_uom_comment and self.trajectory is not None and ('LENGTH' in extra_columns_list or 'MD'
                                                                   in extra_columns_list or 'KH' in extra_columns_list):
            fp.write(f'! Length units along wellbore: {self.trajectory.md_uom if length_uom is None else length_uom}\n')
        fp.write('WELLSPEC ' + str(well_name) + '\n')

    @staticmethod
    def __write_wellspec_file_columns(df, fp, col_width_dict, sep):
        """Write the column names to the WELLSPEC file."""
        for col_name in df.columns:
            if col_name in col_width_dict:
                width = col_width_dict[col_name]
            else:
                width = 10
            form = '{0:>' + str(width) + '}'
            fp.write(sep + form.format(col_name))

    @staticmethod
    def __write_wellspec_file_rows_from_dataframe(df, fp, col_width_dict, sep):
        """Writes the non-blank lines of a Nexus WELLSPEC file from a BlockedWell dataframe."""

        for row_info in df.iterrows():
            row = row_info[1]
            for col_name in df.columns:
                try:
                    if col_name in col_width_dict:
                        width = col_width_dict[col_name]
                    else:
                        width = 10
                    if BlockedWell.__is_float_column(col_name):
                        form = '{0:>' + str(width) + '.3f}'
                        value = row[col_name]
                        if col_name == 'ANGLA' and (np.isnan(value) or value is None):
                            value = 0.0
                        fp.write(sep + form.format(float(value)))
                    else:
                        form = '{0:>' + str(width) + '}'
                        if BlockedWell.__is_int_column(col_name):
                            fp.write(sep + form.format(int(row[col_name])))
                        elif col_name == 'STAT':
                            fp.write(sep + form.format('OFF' if str(row['STAT']).upper() in ['0', 'OFF'] else 'ON'))
                        else:
                            fp.write(sep + form.format(str(row[col_name])))
                except Exception:
                    fp.write(sep + str(row[col_name]))
            fp.write('\n')

    def kji0_marker(self, active_only = True):
        """Convenience method returning (k0, j0, i0), grid_uuid of first blocked interval."""

        cells, grids = self.cell_indices_and_grid_list()
        if cells is None or grids is None or len(grids) == 0:
            return None, None, None, None
        return cells[0], grids[0].uuid

    def xyz_marker(self, active_only = True):
        """Convenience method returning (x, y, z), crs_uuid of perforation in first blocked interval.

        notes:
           active_only argument not yet in use;
           returns None, None if no blocked interval found
        """

        cells, grids = self.cell_indices_and_grid_list()
        if cells is None or grids is None or len(grids) == 0:
            return None, None
        node_index = 0
        while node_index < self.node_count - 1 and self.grid_indices[node_index] == -1:
            node_index += 1
        if node_index >= self.node_count - 1:
            return None, None
        md = 0.5 * (self.node_mds[node_index] + self.node_mds[node_index + 1])
        xyz = self.trajectory.xyz_for_md(md)
        return xyz, self.trajectory.crs_uuid

    def create_feature_and_interpretation(self, shared_interpretation = True):
        """Instantiate new empty WellboreFeature and WellboreInterpretation objects.

        note:
            uses the Blocked well citation title or other related object title as the well name
        """
        title = self.well_name
        if not title:
            title = self.title
            if not title and self.trajectory is not None:
                title = rqw.well_name(self.trajectory)
                if not title:
                    title = 'WELL'
        if self.trajectory is not None:
            traj_interp_uuid = self.model.uuid(obj_type = 'WellboreInterpretation', related_uuid = self.trajectory.uuid)
            if traj_interp_uuid is not None:
                if shared_interpretation:
                    self.wellbore_interpretation = rqo.WellboreInterpretation(parent_model = self.model,
                                                                              uuid = traj_interp_uuid)
                traj_feature_uuid = self.model.uuid(obj_type = 'WellboreFeature', related_uuid = traj_interp_uuid)
                if traj_feature_uuid is not None:
                    self.wellbore_feature = rqo.WellboreFeature(parent_model = self.model, uuid = traj_feature_uuid)
        if self.wellbore_feature is None:
            self.wellbore_feature = rqo.WellboreFeature(parent_model = self.model, feature_name = title)
            self.feature_to_be_written = True
        if self.wellbore_interpretation is None:
            title = title if not self.wellbore_feature.title else self.wellbore_feature.title
            self.wellbore_interpretation = rqo.WellboreInterpretation(parent_model = self.model,
                                                                      title = title,
                                                                      wellbore_feature = self.wellbore_feature)
            if self.trajectory.wellbore_interpretation is None and shared_interpretation:
                self.trajectory.wellbore_interpretation = self.wellbore_interpretation
            self.interpretation_to_be_written = True

    def create_md_datum_and_trajectory(self,
                                       grid,
                                       trajectory_mds,
                                       trajectory_points,
                                       length_uom,
                                       well_name,
                                       set_depth_zero = False,
                                       set_tangent_vectors = False,
                                       create_feature_and_interp = True):
        """Creates an Md Datum object and a (simulation) Trajectory object for this blocked well.

        note:
           not usually called directly; used by import methods
        """

        if not well_name:
            well_name = self.title

        # create md datum node for synthetic trajectory, using crs for grid
        datum_location = trajectory_points[0].copy()
        if set_depth_zero:
            datum_location[2] = 0.0
        datum = rqw.MdDatum(self.model,
                            crs_uuid = grid.crs_uuid,
                            location = datum_location,
                            md_reference = 'mean sea level')

        # create synthetic trajectory object, using crs for grid
        trajectory_mds_array = np.array(trajectory_mds)
        trajectory_xyz_array = np.array(trajectory_points)
        trajectory_df = pd.DataFrame({
            'MD': trajectory_mds_array,
            'X': trajectory_xyz_array[..., 0],
            'Y': trajectory_xyz_array[..., 1],
            'Z': trajectory_xyz_array[..., 2]
        })
        self.trajectory = rqw.Trajectory(self.model,
                                         md_datum = datum,
                                         data_frame = trajectory_df,
                                         length_uom = length_uom,
                                         well_name = well_name,
                                         set_tangent_vectors = set_tangent_vectors)
        self.trajectory_to_be_written = True

        if create_feature_and_interp:
            self.create_feature_and_interpretation()

    def create_xml(self,
                   ext_uuid = None,
                   create_for_trajectory_if_needed = True,
                   add_as_part = True,
                   add_relationships = True,
                   title = None,
                   originator = None):
        """Create a blocked wellbore representation node from this BlockedWell object, optionally add as part.

        note:
           trajectory xml node must be in place before calling this function;
           witsml log reference, interval stratigraphic units, and cell fluid phase units not yet supported

        :meta common:
        """

        assert self.trajectory is not None, 'trajectory object missing'

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()

        if title:
            self.title = title
        if not self.title:
            self.title = self.well_name
        title = self.title

        self.__create_wellbore_feature_and_interpretation_xml_if_needed(add_as_part = add_as_part,
                                                                        add_relationships = add_relationships,
                                                                        originator = originator)

        self.__create_trajectory_xml_if_needed(create_for_trajectory_if_needed = create_for_trajectory_if_needed,
                                               add_as_part = add_as_part,
                                               add_relationships = add_relationships,
                                               originator = originator,
                                               ext_uuid = ext_uuid,
                                               title = title)

        assert self.trajectory.root is not None, 'trajectory xml not established'

        bw_node = super().create_xml(title = title, originator = originator, add_as_part = False)

        # wellbore frame elements

        nc_node, mds_node, mds_values_node, cc_node, cis_node, cnull_node, cis_values_node, gis_node, gnull_node,  \
            gis_values_node, fis_node, fnull_node, fis_values_node =  \
            self.__create_bw_node_sub_elements(bw_node = bw_node)

        self.__create_hdf5_dataset_references(ext_uuid = ext_uuid,
                                              mds_values_node = mds_values_node,
                                              cis_values_node = cis_values_node,
                                              gis_values_node = gis_values_node,
                                              fis_values_node = fis_values_node)

        traj_root, grid_root, interp_root = self.__create_trajectory_grid_wellbore_interpretation_reference_nodes(
            bw_node = bw_node)

        self.__add_as_part_and_add_relationships_if_required(add_as_part = add_as_part,
                                                             add_relationships = add_relationships,
                                                             bw_node = bw_node,
                                                             interp_root = interp_root,
                                                             ext_uuid = ext_uuid)

        return bw_node

    def __create_wellbore_feature_and_interpretation_xml_if_needed(self, add_as_part, add_relationships, originator):
        """Create root node for WellboreFeature and WellboreInterpretation objects if necessary."""

        if self.feature_to_be_written:
            if self.wellbore_feature is None:
                self.create_feature_and_interpretation()
            self.wellbore_feature.create_xml(add_as_part = add_as_part, originator = originator)
        if self.interpretation_to_be_written:
            if self.wellbore_interpretation is None:
                self.create_feature_and_interpretation()
            self.wellbore_interpretation.create_xml(add_as_part = add_as_part,
                                                    title_suffix = None,
                                                    add_relationships = add_relationships,
                                                    originator = originator)

    def __create_trajectory_xml_if_needed(self, create_for_trajectory_if_needed, add_as_part, add_relationships,
                                          originator, ext_uuid, title):
        """Create root node for associated Trajectory object if necessary."""
        if create_for_trajectory_if_needed and self.trajectory_to_be_written and self.trajectory.root is None:
            md_datum_root = self.trajectory.md_datum.create_xml(add_as_part = add_as_part,
                                                                add_relationships = add_relationships,
                                                                title = str(self.title),
                                                                originator = originator)
            self.trajectory.create_xml(ext_uuid,
                                       md_datum_root = md_datum_root,
                                       add_as_part = add_as_part,
                                       add_relationships = add_relationships,
                                       title = title,
                                       originator = originator)

    def __create_bw_node_sub_elements(self, bw_node):
        """Append sub-elements to the BlockedWell object's root node."""
        nc_node = rqet.SubElement(bw_node, ns['resqml2'] + 'NodeCount')
        nc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        nc_node.text = str(self.node_count)

        mds_node = rqet.SubElement(bw_node, ns['resqml2'] + 'NodeMd')
        mds_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
        mds_node.text = rqet.null_xml_text

        mds_values_node = rqet.SubElement(mds_node, ns['resqml2'] + 'Values')
        mds_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        mds_values_node.text = rqet.null_xml_text

        cc_node = rqet.SubElement(bw_node, ns['resqml2'] + 'CellCount')
        cc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        cc_node.text = str(self.cell_count)

        cis_node = rqet.SubElement(bw_node, ns['resqml2'] + 'CellIndices')
        cis_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        cis_node.text = rqet.null_xml_text

        if self.cellind_null is None:
            self.cellind_null = -1
        cnull_node = rqet.SubElement(cis_node, ns['resqml2'] + 'NullValue')
        cnull_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        cnull_node.text = str(self.cellind_null)

        cis_values_node = rqet.SubElement(cis_node, ns['resqml2'] + 'Values')
        cis_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        cis_values_node.text = rqet.null_xml_text

        gis_node = rqet.SubElement(bw_node, ns['resqml2'] + 'GridIndices')
        gis_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        gis_node.text = rqet.null_xml_text

        if self.gridind_null is None:
            self.gridind_null = -1
        gnull_node = rqet.SubElement(gis_node, ns['resqml2'] + 'NullValue')
        gnull_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        gnull_node.text = str(self.gridind_null)

        gis_values_node = rqet.SubElement(gis_node, ns['resqml2'] + 'Values')
        gis_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        gis_values_node.text = rqet.null_xml_text

        fis_node = rqet.SubElement(bw_node, ns['resqml2'] + 'LocalFacePairPerCellIndices')
        fis_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        fis_node.text = rqet.null_xml_text

        if self.facepair_null is None:
            self.facepair_null = -1
        fnull_node = rqet.SubElement(fis_node, ns['resqml2'] + 'NullValue')
        fnull_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        fnull_node.text = str(self.facepair_null)

        fis_values_node = rqet.SubElement(fis_node, ns['resqml2'] + 'Values')
        fis_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        fis_values_node.text = rqet.null_xml_text

        return nc_node, mds_node, mds_values_node, cc_node, cis_node, cnull_node, cis_values_node, gis_node, gnull_node, gis_values_node, fis_node, fnull_node, fis_values_node

    def __create_trajectory_grid_wellbore_interpretation_reference_nodes(self, bw_node):
        """Create nodes and add to BlockedWell object root node."""

        traj_root = self.trajectory.root
        self.model.create_ref_node('Trajectory',
                                   rqet.find_nested_tags_text(traj_root, ['Citation', 'Title']),
                                   bu.uuid_from_string(traj_root.attrib['uuid']),
                                   content_type = 'obj_WellboreTrajectoryRepresentation',
                                   root = bw_node)
        for grid in self.grid_list:
            grid_root = grid.root
            self.model.create_ref_node('Grid',
                                       rqet.find_nested_tags_text(grid_root, ['Citation', 'Title']),
                                       bu.uuid_from_string(grid_root.attrib['uuid']),
                                       content_type = 'obj_IjkGridRepresentation',
                                       root = bw_node)

        interp_root = None
        if self.wellbore_interpretation is not None:
            interp_root = self.wellbore_interpretation.root
            self.model.create_ref_node('RepresentedInterpretation',
                                       rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                       bu.uuid_from_string(interp_root.attrib['uuid']),
                                       content_type = 'obj_WellboreInterpretation',
                                       root = bw_node)
        return traj_root, grid_root, interp_root

    def __create_hdf5_dataset_references(self, ext_uuid, mds_values_node, cis_values_node, gis_values_node,
                                         fis_values_node):
        """Create nodes that reference the hdf5 datasets (arrays) and add to the BlockedWell onject's root node."""

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'NodeMd', root = mds_values_node)

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'CellIndices', root = cis_values_node)

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'GridIndices', root = gis_values_node)

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'LocalFacePairPerCellIndices', root = fis_values_node)

    def __add_as_part_and_add_relationships_if_required(self, add_as_part, add_relationships, bw_node, interp_root,
                                                        ext_uuid):
        # add the newly created BlockedWell object's root node as a part in the model and add reciprocal relationships

        if add_as_part:
            self.model.add_part('obj_BlockedWellboreRepresentation', self.uuid, bw_node)
            if add_relationships:
                self.model.create_reciprocal_relationship(bw_node, 'destinationObject', self.trajectory.root,
                                                          'sourceObject')

                for grid in self.grid_list:
                    self.model.create_reciprocal_relationship(bw_node, 'destinationObject', grid.root, 'sourceObject')
                if interp_root is not None:
                    self.model.create_reciprocal_relationship(bw_node, 'destinationObject', interp_root, 'sourceObject')
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(bw_node, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

    def write_hdf5(self, file_name = None, mode = 'a', create_for_trajectory_if_needed = True):
        """Create or append to an hdf5 file, writing datasets for the measured depths, grid, cell & face indices.

        :meta common:
        """

        # NB: array data must all have been set up prior to calling this function

        if self.uuid is None:
            self.uuid = bu.new_uuid()

        h5_reg = rwh5.H5Register(self.model)

        if create_for_trajectory_if_needed and self.trajectory_to_be_written:
            self.trajectory.write_hdf5(file_name, mode = mode)
            mode = 'a'

        h5_reg.register_dataset(self.uuid, 'NodeMd', self.node_mds)
        h5_reg.register_dataset(self.uuid, 'CellIndices', self.cell_indices)  # could use int32?
        h5_reg.register_dataset(self.uuid, 'GridIndices', self.grid_indices)  # could use int32?
        # convert face index pairs from [axis, polarity] back to strange local face numbering
        mask = (self.face_pair_indices.flatten() == -1).reshape((-1, 2))  # 2nd axis is (axis, polarity)
        masked_face_indices = np.where(mask, 0, self.face_pair_indices.reshape((-1, 2)))  # 2nd axis is (axis, polarity)
        # using flat array for raw_face_indices array
        # other resqml writing code might use an array with one int per entry point and one per exit point, with 2nd axis as (entry, exit)
        raw_face_indices = np.where(mask[:, 0], -1, self.face_index_map[masked_face_indices[:, 0],
                                                                        masked_face_indices[:,
                                                                                            1]].flatten()).reshape(-1)

        h5_reg.register_dataset(self.uuid, 'LocalFacePairPerCellIndices', raw_face_indices)  # could use uint8?

        h5_reg.write(file = file_name, mode = mode)

    def add_grid_property_to_blocked_well(self, uuid_list):
        """Add properties to blocked wells from a list of uuids for properties on the supporting grid."""

        part_list = [self.model.part_for_uuid(uuid) for uuid in uuid_list]

        assert len(self.grid_list) == 1, "Only blocked wells with a single grid can be handled currently"
        grid = self.grid_list[0]
        parts = self.model.parts_list_filtered_by_supporting_uuid(part_list,
                                                                  grid.uuid)  # only those properties on the grid
        if len(parts) < len(uuid_list):
            log.warning(f"{len(uuid_list)-len(parts)} uuids ignored as they do not belong to the same grid as the gcs")

        gridpc = grid.extract_property_collection()
        cell_parts = [part for part in parts if gridpc.indexable_for_part(part) == 'cells'
                     ]  # only 'cell' properties are handled
        if len(cell_parts) < len(parts):
            log.warning(f"{len(parts)-len(cell_parts)} uuids ignored as they do not have indexableelement of cells")

        if len(cell_parts) > 0:
            bwpc = rqp.PropertyCollection(support = self)
            if len(gridpc.time_series_uuid_list()) > 0:
                time_dict = {
                }  # Dictionary with keys for time_series uuids and None for static properties. Values for each key are a list of property parts associated with that time_series_uuid, or None
                for part in cell_parts:
                    if gridpc.time_series_uuid_for_part(part) in time_dict.keys():
                        time_dict[gridpc.time_series_uuid_for_part(
                            part)] = time_dict[gridpc.time_series_uuid_for_part(part)] + [part]
                    else:
                        time_dict[gridpc.time_series_uuid_for_part(part)] = [part]
            else:
                time_dict = {None: cell_parts}

            for time_uuid in time_dict.keys():
                parts = time_dict[time_uuid]
                for part in parts:
                    array = gridpc.cached_part_array_ref(part)
                    indices = self.cell_indices_for_grid_uuid(grid.uuid)
                    bwarray = np.empty(shape = (indices.shape[0],))
                    for i, ind in enumerate(indices):
                        bwarray[i] = array[tuple(ind)]
                    bwpc.add_cached_array_to_imported_list(
                        bwarray,
                        source_info = f'property from grid {grid.title}',
                        keyword = gridpc.citation_title_for_part(part),
                        discrete = (not gridpc.continuous_for_part(part)),
                        uom = gridpc.uom_for_part(part),
                        time_index = gridpc.time_index_for_part(part),
                        null_value = gridpc.null_value_for_part(part),
                        property_kind = gridpc.property_kind_for_part(part),
                        local_property_kind_uuid = gridpc.local_property_kind_uuid(part),
                        facet_type = gridpc.facet_type_for_part(part),
                        facet = gridpc.facet_for_part(part),
                        realization = gridpc.realization_for_part(part),
                        indexable_element = 'cells')
                bwpc.write_hdf5_for_imported_list()
                bwpc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = time_uuid)
        else:
            log.debug(
                "No properties added - uuids either not 'cell' properties or blocked well is associated with multiple grids"
            )
