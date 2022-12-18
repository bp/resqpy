"""High level extract box for well function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.grid_surface as rgs
import resqpy.model as rq
import resqpy.olio.box_utilities as bx
import resqpy.olio.xml_et as rqet
import resqpy.well as rqw

import resqpy.derived_model._add_one_grid_property_array as rqdm_aogp
import resqpy.derived_model._extract_box as rqdm_eb


def extract_box_for_well(epc_file = None,
                         source_grid = None,
                         min_k0 = None,
                         max_k0 = None,
                         trajectory_epc = None,
                         trajectory_uuid = None,
                         blocked_well_uuid = None,
                         column_ji0 = None,
                         column_xy = None,
                         well_name = None,
                         radius = None,
                         outer_radius = None,
                         active_cells_shape = 'tube',
                         quad_triangles = True,
                         inherit_properties = False,
                         inherit_realization = None,
                         inherit_all_realizations = False,
                         inherit_well = False,
                         set_parent_window = None,
                         new_grid_title = None,
                         new_epc_file = None):
    """Extends an existing model with a new grid extracted as an IJK box around a well trajectory in the source grid.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       min_k0, max_k0 (integers, optional): layer range to include; default is full vertical range of source grid
       trajectory_epc (string, optional): the source file for the trajectory or blocked well, if different to that for
          the source grid
       trajectory_uuid (uuid.UUID): the uuid of the trajectory object for the well, if working from a trajectory
       blocked_well_uuid (uuid.UUID): the uuid of the blocked well object, an alternative to working from a trajectory;
          must include blocking against source_grid
       column_ji0 (integer pair, optional): an alternative to providing a trajectory: the column indices of a 'vertical' well
       column_xy (float pair, optional): an alternative to column_ji0: the x, y location used to determine the column
       well_name (string, optional): name to use for column well, ignored if trajectory_uuid is not None
       radius (float, optional): the radius around the wellbore to include in the box; units are those of grid xy values;
          radial distances are applied horizontally regardless of well inclination; if not present, only cells penetrated
          by the trajectory are included
       outer_radius (float, optional): an outer radius around the wellbore, beyond which an inactive cell mask for the
          source_grid will be set to True (inactive); units are those of grid xy values
       active_cells_shape (string, default 'tube'): the logical shape of cells marked as active in the extracted box;
          'tube' results in an active shape with circular cross section in IJ planes, that follows the trajectory; 'prism'
          activates all cells in IJ columns where any cell is within the tube; 'box' leaves the entire IJK cuboid active
       quad_triangles (boolean, default True): if True, cell K faces are treated as 4 triangles (with a common face
          centre point) when computing the intersection of the trajectory with layer interfaces (horizons); if False,
          the K faces are treated as 2 triangles
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid, with values taken from the extracted box
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       inherit_well (boolean, default False): if True, the new model will have a copy of the well trajectory, its crs (if
          different from that of the grid), and any related wellbore interpretation and feature
       set_parent_window (boolean, optional): if True, the extracted grid has its parent window attribute set; if False,
          the parent window is not set; if None, the default will be True if new_epc_file is None or False otherwise
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the extracted grid (& crs)

    returns:
       (grid, box) where: grid is the new Grid object with extent as determined by source grid geometry, trajectory and
       radius arguments; and box is a numpy int array of shape (2, 3) with first axis covering min, max and second axis
       covering k,j,i; the box array holds the minimum and maximum indices (zero based) in the source grid that have
       been included in the extraction (nb. maximum indices are included, unlike the usual python protocol)

    notes:
       this function is designed to work fully for vertical and deviated wells; for horizontal wells use blocked well mode;
       the extracted box includes all layers between the specified min and max horizons, even if the trajectory terminates
       above the deeper horizon or does not intersect horizon(s) for other reasons;
       when specifying a column well by providing x,y the IJ column with the centre of the topmost k face closest to the
       given point is selected;
       if an outer_radius is given, a boolean property will be created for the source grid with values set True where the
       centres of the cells are beyond this distance from the well, measured horizontally; if outer_radius and new_epc_file
       are both given, the source grid will be copied to the new epc
    """

    # establish model, source grid, and trajectory model
    new_epc_file, trajectory_epc, model, traj_model, source_grid =  \
        _establish_files_and_models(epc_file, new_epc_file, trajectory_epc, source_grid)

    # ensure geometry exists as points in case of a regular (block) grid
    if source_grid.grid_representation == 'IjkBlockGrid':
        source_grid.make_regular_points_cached()

    # check arguments
    if min_k0 is None:
        min_k0 = 0
    if max_k0 is None:
        max_k0 = source_grid.nk - 1
    assert 0 <= min_k0 <= max_k0 < source_grid.nk
    assert trajectory_uuid is not None or blocked_well_uuid is not None or column_ji0 is not None
    if radius is None:
        radius = 0.0  # cells directly penetrated through a k face will still be included
    assert radius >= 0.0
    if outer_radius is not None:
        assert outer_radius >= radius
    assert active_cells_shape in ['tube', 'prism', 'box']

    # prepare cell centre points for testing inclusion
    centres = source_grid.centre_point(cache_centre_array = True)

    # initialise local variables
    box = None
    trajectory = None
    blocked_well = None
    bw_cells = None
    bw_box = None

    # establish well information, dependent on type of well object
    if trajectory_uuid is not None:
        # prepare a trajectory object
        trajectory_root = traj_model.root(obj_type = 'WellboreTrajectoryRepresentation', uuid = trajectory_uuid)
        assert trajectory_root is not None, 'trajectory object not found for uuid: ' + str(trajectory_uuid)
        trajectory = rqw.Trajectory(traj_model, uuid = trajectory_uuid)
        well_name = rqw.well_name(trajectory)
        traj_crs = rqc.Crs(trajectory.model, uuid = trajectory.crs_uuid)
        grid_crs = rqc.Crs(source_grid.model, uuid = source_grid.crs_uuid)
        # modify in-memory trajectory data to be in the same crs as grid
        traj_crs.convert_array_to(grid_crs,
                                  trajectory.control_points)  # trajectory xyz points converted in situ to grid's crs
        trajectory.crs_uuid = source_grid.crs_uuid  # note: tangent vectors might be messed up, if present
        traj_box = np.empty((2, 3))
        traj_box[0] = np.amin(trajectory.control_points, axis = 0)
        traj_box[1] = np.amax(trajectory.control_points, axis = 0)
        grid_box = source_grid.xyz_box(lazy = False)
        if not bx.boxes_overlap(traj_box, grid_box):
            log.error('no overlap of xyz boxes for trajectory and grid for trajectory uuid: ' + str(trajectory.uuid))
            return None, None
    elif blocked_well_uuid is not None:
        bw_root = traj_model.root(obj_type = 'BlockedWellboreRepresentation', uuid = blocked_well_uuid)
        assert bw_root is not None, 'blocked well object not found for uuid: ' + str(blocked_well_uuid)
        blocked_well = rqw.BlockedWell(traj_model, uuid = blocked_well_uuid)
        well_name = rqw.well_name(blocked_well)
        bw_box = blocked_well.box(grid_uuid = source_grid.uuid)
        assert bw_box is not None, 'blocked well does not include cells in source grid'
        assert bw_box[0, 0] <= max_k0 and bw_box[1, 0] >= min_k0,  \
            'blocked well does not include cells in specified layer range'
        bw_cells = blocked_well.cell_indices_for_grid_uuid(source_grid.uuid)
    else:
        column_ji0, well_name = _check_column(source_grid, column_ji0, column_xy, well_name)

    # create cell mask
    inclusion_mask, outer_inactive_mask = _build_cell_masks(source_grid, centres, min_k0, max_k0, trajectory,
                                                            blocked_well, bw_cells, column_ji0, quad_triangles, radius,
                                                            outer_radius)

    # derive box from inclusion mask
    box = _box_from_inclusion_mask(source_grid, inclusion_mask, min_k0, max_k0)

    # prepare inactive mask to merge in for new grid
    box_inactive = _make_inactive_mask(active_cells_shape, inclusion_mask, box)

    # establish title for the new grid
    new_grid_title = _invent_title(new_grid_title, trajectory, blocked_well, column_ji0)

    # perform the main grid extraction
    grid = rqdm_eb.extract_box(epc_file,
                               source_grid = source_grid,
                               box = box,
                               box_inactive = box_inactive,
                               inherit_properties = inherit_properties,
                               inherit_realization = inherit_realization,
                               inherit_all_realizations = inherit_all_realizations,
                               set_parent_window = set_parent_window,
                               new_grid_title = new_grid_title,
                               new_epc_file = new_epc_file)

    # inherit well if requested
    if inherit_well and new_epc_file:
        _inherit_well(new_epc_file, grid, traj_model, well_name, trajectory, blocked_well, column_ji0, box)

    # add mask property for outer radius, if specified
    if outer_radius is not None:
        _add_outer_mask_property(epc_file, new_epc_file, outer_inactive_mask, source_grid, well_name)

    return grid, box


def _box_from_inclusion_mask(source_grid, inclusion_mask, min_k0, max_k0):
    min_j0 = 0
    while min_j0 < source_grid.nj - 1 and not np.any(inclusion_mask[:, min_j0, :]):
        min_j0 += 1
    max_j0 = source_grid.nj - 1
    while max_j0 > 0 and not np.any(inclusion_mask[:, max_j0, :]):
        max_j0 -= 1
    assert max_j0 >= min_j0
    min_i0 = 0
    while min_i0 < source_grid.ni - 1 and not np.any(inclusion_mask[:, :, min_i0]):
        min_i0 += 1
    max_i0 = source_grid.ni - 1
    while max_i0 > 0 and not np.any(inclusion_mask[:, :, max_i0]):
        max_i0 -= 1
    assert max_i0 >= min_i0
    box = np.array([[min_k0, min_j0, min_i0], [max_k0, max_j0, max_i0]], dtype = int)
    log.info('box for well is: ' + bx.string_iijjkk1_for_box_kji0(box) + ' (simulator protocol)')
    return box


def _build_cell_masks(source_grid, centres, min_k0, max_k0, trajectory, blocked_well, bw_cells, column_ji0,
                      quad_triangles, radius, outer_radius):
    # todo: handle base interfaces above k gaps
    # either work with grid layers of interfaces between layers (grid horizons)
    # intialise masks to False
    radius_sqr = radius * radius
    outer_radius_sqr = None if outer_radius is None else outer_radius * outer_radius
    inclusion_mask = np.zeros(source_grid.extent_kji, dtype = bool)
    outer_inactive_mask = None if outer_radius is None else np.zeros(source_grid.extent_kji, dtype = bool)
    h_or_l = 'layer' if trajectory is None else 'horizon'
    end_k0 = max_k0 + 1 if trajectory is None else max_k0 + 2
    warned = False
    for k in range(min_k0, end_k0):
        cols, intersect_points = _find_intersections(source_grid, k, trajectory, blocked_well, bw_cells, column_ji0,
                                                     centres, quad_triangles)
        if cols is None or len(cols) == 0:
            if not warned:
                log.warning(f"no intersection found between well and {h_or_l}(s) such as: {k}")
                warned = True
            continue
        count = cols.shape[0]
        assert len(intersect_points) == count
        if count > 1:
            log.warning(f"{count} intersections found between well and {h_or_l}: {k}")
        layer_mask = np.zeros((source_grid.nj, source_grid.ni), dtype = bool)  # to be set True within radius
        if outer_radius is not None:
            # to be set False within outer_radius
            outer_layer_mask = np.ones((source_grid.nj, source_grid.ni), dtype = bool)
        for intersect in range(count):
            log.debug(f"well intersects {h_or_l} {k} in column j0,i0: {cols[intersect, 0]}, {cols[intersect, 1]}")
            if radius > 0.0 or outer_radius is not None:
                if k < source_grid.nk:
                    vectors = centres[k] - intersect_points[intersect].reshape((1, 1, 3))
                    distance_sqr = vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1]
                    if radius > 0.0:
                        layer_mask = np.logical_or(layer_mask, np.less_equal(distance_sqr, radius_sqr))
                    if outer_radius is not None:
                        outer_layer_mask = np.logical_and(outer_layer_mask,
                                                          np.greater_equal(distance_sqr, outer_radius_sqr))
                if k > 0 and (not source_grid.k_gaps or k >= source_grid.nk - 1 or
                              not source_grid.k_gap_after_array[k - 1]):
                    vectors = centres[k - 1] - intersect_points[intersect].reshape((1, 1, 3))
                    distance_sqr = vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1]
                    if radius > 0.0:
                        layer_mask = np.logical_or(layer_mask, np.less_equal(distance_sqr, radius_sqr))
                    if outer_radius is not None:
                        outer_layer_mask = np.logical_and(outer_layer_mask,
                                                          np.greater_equal(distance_sqr, outer_radius_sqr))
            layer_mask[cols[intersect, 0], cols[intersect, 1]] = True
        if k <= max_k0:
            inclusion_mask[k] = layer_mask
            if outer_radius is not None:
                outer_inactive_mask[k] = outer_layer_mask
        if k > min_k0:
            inclusion_mask[k - 1] = np.logical_or(inclusion_mask[k - 1], layer_mask)
            if outer_radius is not None:
                outer_inactive_mask[k - 1] = np.logical_and(outer_inactive_mask[k - 1], outer_layer_mask)
        log.debug(f"number of columns found in {h_or_l} {k} within radius around well: {np.count_nonzero(layer_mask)}")
    inc_count = np.count_nonzero(inclusion_mask)
    if inc_count == 0:
        log.error('no cells found within search radius around well')
        return None, None
    log.info('total number of cells found within radius around well: ' + str(inc_count))
    return inclusion_mask, outer_inactive_mask


def _invent_title(new_grid_title, trajectory, blocked_well, column_ji0):
    if not new_grid_title:
        if trajectory is not None:
            new_grid_title = 'local grid extracted for well: ' + str(trajectory.title)
        elif blocked_well is not None:
            new_grid_title = 'local grid extracted for blocked well: ' + str(blocked_well.title)
        elif column_ji0 is not None:
            new_grid_title = 'local grid extracted around column i, j (1 based): ' +  \
                             str(column_ji0[1] + 1) + ', ' + str(column_ji0[0] + 1)
        else:  # should not happen
            new_grid_title = 'local grid extracted for well'
    return new_grid_title


def _add_outer_mask_property(epc_file, new_epc_file, outer_inactive_mask, source_grid, well_name):
    if new_epc_file:
        # TODO: copy source grid and reassign source_grid to new copy
        outer_epc = new_epc_file
    else:
        outer_epc = epc_file
    # todo: make local property kind, or reuse active local property kind?
    rqdm_aogp.add_one_grid_property_array(outer_epc,
                                          outer_inactive_mask,
                                          'discrete',
                                          grid_uuid = source_grid.uuid,
                                          source_info = 'extract box for well outer radius',
                                          title = 'distant mask for well ' + str(well_name),
                                          discrete = True)


def _inherit_well(new_epc_file, grid, traj_model, well_name, trajectory, blocked_well, column_ji0, box):
    newer_model = rq.Model(new_epc_file)
    if trajectory is None and blocked_well is None:
        log.info('creating well objects for column')
        box_column_ji0 = (column_ji0[0] - box[0, 1], column_ji0[1] - box[0, 2])
        bw = rqw.BlockedWell(newer_model,
                             grid = grid,
                             column_ji0 = box_column_ji0,
                             well_name = well_name,
                             use_face_centres = True)
        bw.write_hdf5(create_for_trajectory_if_needed = True)
        bw.create_xml(create_for_trajectory_if_needed = True, title = well_name)
    elif blocked_well is not None:
        log.info('inheriting trajectory for blocked well')  # recursively copies referenced parts
        newer_model.copy_part_from_other_model(traj_model, traj_model.part(uuid = blocked_well.trajectory.uuid))
    else:
        log.info('inheriting well trajectory')  # recursively copies referenced parts
        newer_model.copy_part_from_other_model(traj_model, traj_model.part(uuid = trajectory.uuid))
    newer_model.h5_release()
    newer_model.store_epc()


def _make_inactive_mask(active_cells_shape, inclusion_mask, box):
    if active_cells_shape in ['tube', 'prism']:
        if active_cells_shape == 'prism':
            layer_mask = np.any(inclusion_mask, axis = 0)
            inclusion_mask[:] = layer_mask
        box_inactive = np.logical_not(inclusion_mask[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1,
                                                     box[0, 2]:box[1, 2] + 1])
    else:  # 'box' option: leave all cells active (except where inactive in source grid)
        box_inactive = None
    return box_inactive


def _check_column(source_grid, column_ji0, column_xy, well_name):
    if column_ji0 is None:
        assert len(column_xy) == 2
        column_ji0 = source_grid.find_cell_for_point_xy(column_xy[0], column_xy[1])
        if column_ji0[0] is None or column_ji0[1] is None:
            log.error('no column found for x, y: ' + str(column_xy[0]) + ', ' + str(column_xy[1]))
        return None, None
    assert len(column_ji0) == 2
    assert 0 <= column_ji0[0] < source_grid.nj and 0 <= column_ji0[1] < source_grid.ni
    if not well_name:
        well_name = 'well for global column ' + str(column_ji0[1] + 1) + ', ' + str(column_ji0[0] + 1)
    return column_ji0, well_name


def _establish_files_and_models(epc_file, new_epc_file, trajectory_epc, source_grid):
    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
    if trajectory_epc == epc_file:
        trajectory_epc = None
    if epc_file:
        model = rq.Model(epc_file)
        if source_grid is None:
            source_grid = model.grid()  # requires there to be exactly one grid in model (or one named ROOT)
    else:
        model = source_grid.model
    assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']
    assert model is not None
    if trajectory_epc is None:
        traj_model = model
    else:
        traj_model = rq.Model(trajectory_epc)
    assert traj_model is not None
    return new_epc_file, trajectory_epc, model, traj_model, source_grid


def _find_intersections(source_grid, k, trajectory, blocked_well, bw_cells, column_ji0, centres, quad_triangles):
    if trajectory is None:
        if blocked_well is None:
            cols_ji0 = np.array(column_ji0, dtype = int).reshape((1, 2))
            cols, intersect_points = cols_ji0, centres[k, column_ji0[0], column_ji0[1]].reshape((1, 3))
        else:
            selected_cells = np.where(bw_cells[:, 0] == k)[0]
            cells = bw_cells[selected_cells]
            cols = cells[:, 1:]
            intersect_points = centres[cells[:, 0], cells[:, 1], cells[:, 2]]
    else:
        if k < source_grid.nk:
            cols, intersect_points = rgs.find_intersections_of_trajectory_with_layer_interface(
                trajectory, source_grid, k0 = k, ref_k_faces = 'top', quad_triangles = quad_triangles)
        else:
            cols, intersect_points = rgs.find_intersections_of_trajectory_with_layer_interface(
                trajectory, source_grid, k0 = k - 1, ref_k_faces = 'base', quad_triangles = quad_triangles)
    return cols, intersect_points
