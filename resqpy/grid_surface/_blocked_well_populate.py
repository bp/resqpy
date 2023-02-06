"""Function for populating an empty blocked well from its trajectory and a grid."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.surface as rqs
import resqpy.well as rqw
import resqpy.grid_surface as rqgs
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec


def populate_blocked_well_from_trajectory(blocked_well,
                                          grid,
                                          active_only = False,
                                          quad_triangles = True,
                                          lazy = False,
                                          use_single_layer_tactics = True,
                                          check_for_reentry = True):
    """Populate an empty blocked well object based on the intersection of its trajectory with a grid.

    arguments:
       blocked_well (resqpy.well.BlockedWell object): a largely empty blocked well object to be populated by this
          function; note that the trajectory attribute must be set before calling this function
       grid (resqpy.grid.Grid object): the grid to intersect the well trajectory with
       active_only (boolean, default False): if True, intervals which cover inactive cells will be set as
          unblocked intervals; if False, intervals covering any cell will be set as blocked intervals
       quad_triangles (boolean, default True): if True, each cell face is represented by 4 triangles when
          calculating intersections; if False, only 2 triangles are used
       lazy (boolean, default False): if True, initial penetration must be through a top K face and blocking
          will cease as soon as the trajectory first leaves the gridded volume; if False, initial entry may be
          through any external face or a fault face and re-entry will be handled
       use_single_layer_tactics (boolean, default True): if True and not lazy and the grid does not have k gaps,
          fault planes and grid sidewall are initially treated as if the grid were a single layer, when looking
          for penetrations from outwith the grid
       check_for_reentry (boolean, default True): if True, the trajectory is tracked after leaving the grid through
          the outer skin (eg. base reservoir) in case of re-entry; if False, blocking stops upon the first exit
          of the trajectory through the skin; ignored (treated as False) if lazy is True

    returns:
       the blocked well object (same object as passed in) if successful; None if unsuccessful

    notes:
       the blocked_well trajectory attribute must be set before calling this function;
       grids with k gaps might result in very slow processing;
       the function represents a cell face as 2 or 4 triangles rather than a bilinear patch; setting quad_triangles
       False is not recommended as the 2 triangle formulation gives a non-unique representation of a face (though
       the code is designed to use the same representation for a shared face between neighbouring cells);
       where non-planar faults exist, the triangulation of faces may result in a small misalignment of abutted faces
       between the opposing sides of the fault; this could potentially result in an extra, small, unblocked interval
       being introduced as an artefact between the exit point of one cell and the entry point into the abutted cell
    """

    assert isinstance(blocked_well, rqw.BlockedWell)
    assert isinstance(blocked_well.trajectory, rqw.Trajectory)
    assert grid is not None

    flavour = grr.grid_flavour(grid.root)
    if not flavour.startswith('Ijk'):
        raise NotImplementedError('well blocking only implemented for IjkGridRepresentation')
    is_regular = (flavour == 'IjkBlockGrid') and hasattr(grid, 'is_aligned') and grid.is_aligned
    log.debug(f"well blocking: grid {'is' if is_regular else 'is not'} regular and aligned")

    if grid.k_gaps:
        use_single_layer_tactics = False
        log.debug('skin single layer tactics disabled')

    grid_crs = rqc.Crs(grid.model, uuid = grid.crs_uuid)
    trajectory = __trajectory_init(blocked_well, grid, grid_crs)
    traj_xyz = trajectory.control_points

    if not rqgs.trajectory_grid_overlap(trajectory, grid):
        log.error(f'no overlap of trajectory xyz box with grid for trajectory uuid: {trajectory.uuid}')
        return None
    grid_box = grid.xyz_box(lazy = False, local = True).copy()
    if grid_crs.z_inc_down:
        z_sign = 1.0
        grid_top_z = grid_box[0, 2]
    else:
        z_sign = -1.0
        grid_top_z = -grid_box[1, 2]  # maximum is least negative, ie. shallowest

    assert z_sign * traj_xyz[0, 2] < grid_top_z, 'trajectory does not start above top of grid'  # min z
    if z_sign * traj_xyz[-1, 2] < grid_top_z:
        log.warning('end of trajectory (TD) is above top of grid')

    # scan down wellbore till top of grid reached
    knot = 0
    knot_count = trajectory.knot_count - 1
    while knot < knot_count and z_sign * traj_xyz[knot + 1, 2] < grid_top_z:
        knot += 1
    log.debug(f'skipped trajectory to knot: {knot}')
    if knot == knot_count:
        log.warning('no well blocking due to entire trajectory being shallower than top of grid')
        return None  # entire well is above grid

    if lazy:
        # search for intersection with top of grid: will fail if well comes in from the side (or from below!) or
        # penetrates through fault
        xyz, entry_knot, col_ji0 = rqgs.find_first_intersection_of_trajectory_with_layer_interface(
            trajectory,
            grid,
            k0 = 0,
            ref_k_faces = 'top',
            start = knot,
            heal_faults = False,
            quad_triangles = quad_triangles,
            is_regular = is_regular)
        log.debug(f'lazy top intersection x,y,z: {xyz}; knot: {entry_knot}; col j0,i0: {col_ji0[0]}, {col_ji0[1]}')
        if xyz is None:
            log.error('failed to lazily find intersection of trajectory with top surface of grid')
            return None
        cell_kji0 = np.array((0, col_ji0[0], col_ji0[1]), dtype = int)
        axis = 0
        polarity = 0

    else:  # not lazy
        # note: xyz and entry_fraction might be slightly off when penetrating a skewed fault plane – deemed immaterial
        # for real cases
        skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics, is_regular = is_regular)
        xyz, cell_kji0, axis, polarity, entry_knot = skin.find_first_intersection_of_trajectory(trajectory)
        if xyz is None:
            log.error('failed to find intersection of trajectory with outer skin of grid')
            return None
        else:
            log.debug(f"skin intersection x,y,z: {xyz}; knot: {entry_knot}; cell kji0: {cell_kji0}; face: "
                      f"{'KJI'[axis]}{'-+'[polarity]}")
            cell_kji0 = np.array(cell_kji0, dtype = int)

    previous_kji0 = cell_kji0.copy()
    previous_kji0[axis] += polarity * 2 - 1  # note: previous may legitimately be 'beyond' edge of grid
    entry_fraction = __segment_fraction(traj_xyz, entry_knot, xyz)
    log.debug(f'initial previous kji0: {previous_kji0}')
    next_cell_info = __find_next_cell(grid,
                                      previous_kji0,
                                      axis,
                                      1 - polarity,
                                      trajectory,
                                      entry_knot,
                                      entry_fraction,
                                      xyz,
                                      check_for_reentry,
                                      treat_skin_as_fault = use_single_layer_tactics,
                                      lazy = lazy,
                                      use_single_layer_tactics = use_single_layer_tactics)
    log.debug(f'initial next cell info: {next_cell_info}')
    node_mds_list = [__back_calculated_md(trajectory, entry_knot, entry_fraction)]
    node_count = 1
    grid_indices_list = []
    cell_count = 0
    cell_indices_list = []
    face_pairs_list = []
    kissed = np.zeros(grid.extent_kji, dtype = bool)
    sample_test = np.zeros(grid.extent_kji, dtype = bool)

    while next_cell_info is not None:
        entry_shared, kji0, entry_axis, entry_polarity, entry_knot, entry_fraction, entry_xyz = next_cell_info

        # log.debug('next cell entry x,y,z: ' + str(entry_xyz) + '; knot: ' + str(entry_knot) + '; cell kji0: ' +
        #           str(kji0) + '; face: ' + 'KJI'[entry_axis] + '-+'[entry_polarity])

        if not entry_shared:
            log.debug('adding unblocked interval')
            node_mds_list.append(__back_calculated_md(trajectory, entry_knot, entry_fraction))
            node_count += 1
            grid_indices_list.append(-1)
            kissed[:] = False
            sample_test[:] = False

        exit_xyz, exit_knot, exit_axis, exit_polarity = rqgs.find_first_intersection_of_trajectory_with_cell_surface(
            trajectory, grid, kji0, entry_knot, start_xyz = entry_xyz, nudge = 0.01, quad_triangles = True)

        #  if exit_xyz is None:
        #      log.debug('no exit')
        # else:
        #     log.debug('cell exit x,y,z: ' + str(exit_xyz) + '; knot: ' + str(exit_knot) + '; face: ' +
        #               'KJI'[exit_axis] + '-+'[exit_polarity])

        if exit_xyz is None:
            if rqgs.point_is_within_cell(traj_xyz[-1], grid, kji0):
                # well terminates within cell: add termination blocked interval (or unblocked if inactive cell)
                log.debug(f'adding termination interval for cell {kji0}')
                # todo: check for inactive cell and make an unblocked interval instead, if required
                node_mds_list.append(trajectory.measured_depths[-1])
                node_count += 1
                grid_indices_list.append(0)
                cell_indices_list.append(grid.natural_cell_index(kji0))
                cell_count += 1
                face_pairs_list.append(((entry_axis, entry_polarity), (-1, -1)))
                break
            else:
                next_cell_info = __forward_nudge(axis, entry_axis, entry_knot, entry_polarity, entry_xyz, grid, kissed,
                                                 kji0, previous_kji0, sample_test, traj_xyz, trajectory,
                                                 use_single_layer_tactics)
            if next_cell_info is None:
                log.warning('well blocking got stuck – cells probably omitted at tail of well')
        #      if exit_xyz is not None:  # usual well-behaved case or non-standard due to kiss
        else:
            exit_fraction = __segment_fraction(traj_xyz, exit_knot, exit_xyz)
            log.debug(f'adding blocked interval for cell kji0: {kji0}')
            # todo: check for inactive cell and make an unblocked interval instead, if required
            node_mds_list.append(__back_calculated_md(trajectory, exit_knot, exit_fraction))
            node_count += 1
            grid_indices_list.append(0)
            cell_indices_list.append(grid.natural_cell_index(kji0))
            cell_count += 1
            face_pairs_list.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

            previous_kji0 = kji0
            # log.debug(f'previous kji0 set to {previous_kji0}')
            next_cell_info = __find_next_cell(grid,
                                              kji0,
                                              exit_axis,
                                              exit_polarity,
                                              trajectory,
                                              exit_knot,
                                              exit_fraction,
                                              exit_xyz,
                                              check_for_reentry,
                                              treat_skin_as_fault = use_single_layer_tactics,
                                              lazy = lazy,
                                              use_single_layer_tactics = use_single_layer_tactics)
            kissed[:] = False
            sample_test[:] = False

    if node_count < 2:
        log.warning('no nodes found during attempt to block well')
        return None

    assert node_count > 1
    assert len(node_mds_list) == node_count
    assert len(grid_indices_list) == node_count - 1
    assert cell_count < node_count
    assert len(cell_indices_list) == cell_count
    assert len(face_pairs_list) == cell_count

    blocked_well.node_mds = np.array(node_mds_list, dtype = float)
    blocked_well.node_count = node_count
    blocked_well.grid_indices = np.array(grid_indices_list, dtype = int)
    blocked_well.cell_indices = np.array(cell_indices_list, dtype = int)
    blocked_well.face_pair_indices = np.array(face_pairs_list, dtype = int)
    blocked_well.cell_count = cell_count
    blocked_well.grid_list = [grid]

    assert cell_count == (node_count - np.count_nonzero(blocked_well.grid_indices == -1) - 1)

    log.info(f'{cell_count} cell{__pl(cell_count)} blocked for well trajectory uuid: {trajectory.uuid}')

    return blocked_well


def generate_surface_for_blocked_well_cells(blocked_well,
                                            combined = True,
                                            active_only = False,
                                            min_k0 = 0,
                                            max_k0 = None,
                                            depth_limit = None,
                                            quad_triangles = True):
    """Returns a surface or list of surfaces representing the faces of the cells visited by the well."""

    assert blocked_well is not None and type(blocked_well) is rqw.BlockedWell
    cells_kji0, grid_list = blocked_well.cell_indices_and_grid_list()
    assert len(cells_kji0) == len(grid_list)
    cell_count = len(cells_kji0)
    if cell_count == 0:
        return None

    surface_list = []
    if combined:
        composite_cp = np.zeros((cell_count, 2, 2, 2, 3))

    cell_p = 0
    for cell in range(cell_count):
        grid = grid_list[cell]
        cell_kji0 = cells_kji0[cell]
        if active_only and grid.inactive is not None and grid.inactive[cell_kji0]:
            continue
        if cell_kji0[0] < min_k0 or (max_k0 is not None and cell_kji0[0] > max_k0):
            continue
        if depth_limit is not None and grid.centre_point(cell_kji0) > depth_limit:
            continue
        cp = grid.corner_points(cell_kji0)
        if combined:
            composite_cp[cell_p] = cp
            cell_p += 1
        else:
            cs = rqs.Surface(blocked_well.model)
            cs.set_to_single_cell_faces_from_corner_points(cp, quad_triangles = quad_triangles)
            surface_list.append(cs)

    if cell_p == 0 and len(surface_list) == 0:
        return None

    if combined:
        cs = rqs.Surface(blocked_well.model)
        cs.set_to_multi_cell_faces_from_corner_points(composite_cp[:cell_p])
        return cs

    return surface_list


def __find_next_cell(
    grid,
    previous_kji0,
    axis,
    polarity,
    trajectory,
    segment,
    seg_fraction,
    xyz,
    check_for_reentry,
    treat_skin_as_fault = False,
    lazy = False,
    use_single_layer_tactics = True,
):
    # returns for next cell entry: (shared transit point bool, kji0, axis, polarity, segment, seg_fraction, xyz)
    # or single None if no next cell identified (end of trajectory beyond edge of grid)
    # take care of: edges of model; pinchouts; faults, (k gaps), exact edge or corner crossings
    # note: identified cell may be active or inactive: calling code to handle that
    # note: for convenience, previous_kji0 may lie just outside the extent of the grid
    # note: polarity is relative to previous cell, so its complement applies to the next cell
    #      log.debug('finding next cell with previous kji0: ' + str(previous_kji0) + '; exit: ' + 'kji'[axis] + '-+'[polarity])
    kji0 = np.array(previous_kji0, dtype = int)
    polarity_sign = 2 * polarity - 1
    kji0[axis] += polarity_sign
    # if gone beyond external skin of model, return None
    if np.any(kji0 < 0) or np.any(kji0 >= grid.extent_kji) or (grid.k_gaps and axis == 0 and (
        (polarity == 1 and previous_kji0[0] >= 0 and grid.k_gap_after_array[previous_kji0[0]]) or
        (polarity == 0 and kji0[0] < grid.nk - 1 and grid.k_gap_after_array[kji0[0]]))):
        if check_for_reentry and not lazy:
            skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
            # nudge in following will be problematic if gap has zero (or tiny) thickness at this location
            xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(trajectory,
                                                                                                   start = segment,
                                                                                                   start_xyz = xyz,
                                                                                                   nudge = +0.05)
            if xyz_r is None:
                log.debug('no skin re-entry found after exit through skin')
                return None
            log.debug(f"skin re-entry after skin exit: kji0: {cell_kji0}; face: {'KJI'[axis]}{'-+'[polarity]}")
            seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_r)
            return False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r
        else:
            return None
    # pre-assess split pillar (fault) situation
    faulted = False
    if axis != 0:
        if grid.has_split_coordinate_lines:
            faulted = grid.is_split_column_face(kji0[1], kji0[2], axis, 1 - polarity)
        if not faulted and treat_skin_as_fault:
            faulted = (kji0[axis] == 0 and polarity == 1) or (kji0[axis] == grid.extent_kji[axis] - 1 and polarity == 0)
    # handle the simplest case of a well behaved k neighbour or unsplit j or i neighbour
    if not grid.pinched_out(cell_kji0 = kji0, cache_pinchout_array = True) and not faulted:
        return True, kji0, axis, 1 - polarity, segment, seg_fraction, xyz
    if faulted:
        return __faulted(trajectory, grid, segment, kji0, axis, polarity, xyz, lazy, use_single_layer_tactics,
                         previous_kji0)
    else:
        # skip pinched out cells
        pinchout_skip_sign = -1 if axis == 0 and polarity == 0 else 1
        while grid.pinched_out(cell_kji0 = kji0, cache_pinchout_array = True):
            kji0[0] += pinchout_skip_sign
            if not (np.all(kji0 >= 0) and np.all(kji0 < grid.extent_kji)) or (grid.k_gaps and axis == 0 and (
                (polarity == 0 and grid.k_gap_after_array[kji0[0]]) or
                (polarity == 1 and grid.k_gap_after_array[kji0[0] - 1]))):
                log.debug(
                    f"trajectory reached edge of model {'or k gap ' if grid.k_gaps and axis == 0 else ''}at exit from cell kji0: {previous_kji0}"
                )
                if lazy:
                    return None
                skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
                # nudge in following will be problematic if gap has zero (or tiny) thickness at this location
                xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(trajectory,
                                                                                                       start = segment,
                                                                                                       start_xyz = xyz,
                                                                                                       nudge = +0.01)
                if xyz_r is None:
                    return None  # no re-entry found after exit through skin
                seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_r)
                return False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r
        return True, kji0, axis, 1 - polarity, segment, seg_fraction, xyz


def __faulted(trajectory, grid, segment, kji0, axis, polarity, xyz, lazy, use_single_layer_tactics, previous_kji0):
    # look for intersections with column face
    xyz_f, k0 = rqgs.find_intersection_of_trajectory_interval_with_column_face(trajectory,
                                                                               grid,
                                                                               segment,
                                                                               kji0[1:],
                                                                               axis,
                                                                               1 - polarity,
                                                                               start_xyz = xyz,
                                                                               nudge = -0.1,
                                                                               quad_triangles = True)
    if xyz_f is not None and k0 is not None:
        kji0[0] = k0
        seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_f)
        return vec.isclose(xyz, xyz_f, tolerance = 0.001), kji0, axis, 1 - polarity, segment, seg_fraction, xyz_f
    log.debug('failed to find entry point in column face after crossing fault; checking entire cross section')
    x_sect_surf = rqgs.generate_torn_surface_for_x_section(grid,
                                                           'KJI'[axis],
                                                           ref_slice0 = kji0[axis],
                                                           plus_face = (polarity == 0),
                                                           quad_triangles = True,
                                                           as_single_layer = False)
    xyz_f, segment_f, tri_index_f = rqgs.find_first_intersection_of_trajectory_with_surface(trajectory,
                                                                                            x_sect_surf,
                                                                                            start = segment,
                                                                                            start_xyz = xyz,
                                                                                            nudge = -0.1)
    if xyz_f is not None:
        # back out cell info from triangle index; note 'column_from...' is actually x_section cell face
        k0, j_or_i0 = x_sect_surf.column_from_triangle_index(tri_index_f)
        kji0[0] = k0
        kji0[3 - axis] = j_or_i0
        seg_fraction = __segment_fraction(trajectory.control_points, segment_f, xyz_f)
        return vec.isclose(xyz, xyz_f, tolerance = 0.001), kji0, axis, 1 - polarity, segment_f, seg_fraction, xyz_f
    log.debug("failed to find entry point in cross section after crossing fault"
              f"{'' if lazy else '; checking for skin re-entry'}")
    if lazy:
        return None
    skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
    # following is problematic due to skewed fault planes
    xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(trajectory,
                                                                                           start = segment,
                                                                                           start_xyz = xyz,
                                                                                           nudge = -0.1,
                                                                                           exclude_kji0 = previous_kji0)
    if xyz_r is None:
        log.warning('no skin re-entry found after exit through fault face')
        return None
    log.debug(f"skin re-entry after fault face exit: kji0: {cell_kji0}; face: {'KJI'[axis]}{'-+'[polarity]}")
    seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_r)
    return False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r


def __back_calculated_md(trajectory, segment, fraction):
    base_md = trajectory.measured_depths[segment]
    return base_md + fraction * (trajectory.measured_depths[segment + 1] - base_md)


def __forward_nudge(axis, entry_axis, entry_knot, entry_polarity, entry_xyz, grid, kissed, kji0, previous_kji0,
                    sample_test, traj_xyz, trajectory, use_single_layer_tactics):
    next_cell_info = None
    # trajectory has kissed corner or edge of cell and nudge has gone outside cell
    # nudge forward and look for point inclusion in possible neighbours
    log.debug(f"kiss detected at cell kji0 {kji0} {'KJI'[entry_axis]}{'-+'[entry_polarity]}")
    kissed[tuple(kji0)] = True
    if np.all(previous_kji0 >= 0) and np.all(previous_kji0 < grid.extent_kji):
        kissed[tuple(previous_kji0)] = True  # stops immediate revisit for kiss and tell tactics
    # setup kji offsets of neighbouring cells likely to contain sample point (could check for split
    # column faces)
    axis_a = (entry_axis + 1) % 3
    axis_b = (entry_axis + 2) % 3
    offsets_kji = np.zeros((18, 3), dtype = int)
    offsets_kji[0, axis_a] = -1
    offsets_kji[1, axis_a] = 1
    offsets_kji[2, axis_b] = -1
    offsets_kji[3, axis_b] = 1
    offsets_kji[4, axis_a] = offsets_kji[4, axis_b] = -1
    offsets_kji[5, axis_a] = -1
    offsets_kji[5, axis_b] = 1
    offsets_kji[6, axis_a] = 1
    offsets_kji[6, axis_b] = -1
    offsets_kji[7, axis_a] = offsets_kji[7, axis_b] = 1
    offsets_kji[8:16] = offsets_kji[:8]
    offsets_kji[8:16, axis] = previous_kji0[axis] - kji0[axis]
    pinchout_skip_sign = -1 if entry_axis == 0 and entry_polarity == 1 else 1
    offsets_kji[16, axis] = pinchout_skip_sign
    log.debug(f'kiss pinchout skip sign: {pinchout_skip_sign}')
    next_cell_info = __try_cell_entry(entry_knot, entry_xyz, grid, kissed, kji0, next_cell_info, offsets_kji,
                                      pinchout_skip_sign, traj_xyz, trajectory)
    if next_cell_info is not None:
        return next_cell_info
    else:
        log.debug('kiss and tell failed to find next cell entry; switching to point in cell tactics')
        # take a sample point a little ahead on the trajectory
        remaining = traj_xyz[entry_knot + 1] - entry_xyz
        remaining_length = vec.naive_length(remaining)
        if remaining_length < 0.025:
            log.debug(f'not much remaining of segnemt: {remaining_length}')
        nudge = 0.025
        if remaining_length <= nudge:
            sample_point = entry_xyz + 0.9 * remaining
        else:
            sample_point = entry_xyz + nudge * vec.unit_vector(remaining)
        sample_test[:] = False
        log.debug(f'sample point is {sample_point}')
        for try_index in range(len(offsets_kji)):
            try_kji0 = np.array(kji0, dtype = int) + offsets_kji[try_index]
            while np.all(try_kji0 >= 0) and np.all(try_kji0 < grid.extent_kji) and grid.pinched_out(
                    cell_kji0 = try_kji0):
                try_kji0[0] += pinchout_skip_sign
            if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji):
                continue
            #                 log.debug(f'tentatively trying: {try_kji0}')
            sample_test[tuple(try_kji0)] = True
            if rqgs.point_is_within_cell(sample_point, grid, try_kji0):
                log.debug(f'sample point is in cell {try_kji0}')
                try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = \
                    rqgs.find_first_intersection_of_trajectory_with_cell_surface(trajectory, grid, try_kji0, entry_knot,
                                                                            start_xyz=entry_xyz, nudge=-0.01,
                                                                            quad_triangles=True)
                if try_entry_xyz is None or try_entry_knot is None:
                    log.warning(f'failed to find entry to favoured cell {try_kji0}')
                else:
                    if np.all(try_kji0 == kji0):
                        log.debug(f'staying in cell {kji0} after kiss')
                        # TODO: sort things out to discard kiss completely
                    try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                    return vec.isclose(try_entry_xyz, entry_xyz, tolerance=0.01), try_kji0, try_entry_axis, \
                           try_entry_polarity, try_entry_knot, try_entry_fraction, try_entry_xyz
                break
        log.debug('sample point not found in immediate neighbours, switching to full column search')
        k_offset = 0
        while k_offset <= max(kji0[0], grid.nk - kji0[0] - 1):
            for k_plus_minus in [1, -1]:
                for j_offset in [0, -1, 1]:
                    for i_offset in [0, -1, 1]:
                        try_kji0 = kji0 + np.array((k_offset * k_plus_minus, j_offset, i_offset))
                        if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji) or sample_test[tuple(try_kji0)]:
                            continue
                        sample_test[tuple(try_kji0)] = True
                        if rqgs.point_is_within_cell(sample_point, grid, try_kji0):
                            log.debug(f'sample point is in cell {try_kji0}')
                            try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = \
                                rqgs.find_first_intersection_of_trajectory_with_cell_surface(
                                    trajectory, grid, try_kji0, entry_knot, start_xyz=entry_xyz,
                                    nudge=-0.01, quad_triangles=True)
                            if try_entry_xyz is None or try_entry_knot is None:
                                log.warning(f'failed to find entry to column sampled cell {try_kji0}')
                                return __no_success(entry_knot, entry_xyz, grid, next_cell_info, previous_kji0,
                                                    traj_xyz, trajectory, use_single_layer_tactics)
                            else:
                                try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                                return (vec.isclose(try_entry_xyz, entry_xyz,
                                                    tolerance = 0.01), try_kji0, try_entry_axis, try_entry_polarity,
                                        try_entry_knot, try_entry_fraction, try_entry_xyz)
            k_offset += 1
        return __no_success(entry_knot, entry_xyz, grid, next_cell_info, previous_kji0, traj_xyz, trajectory,
                            use_single_layer_tactics)


def __no_success(entry_knot, entry_xyz, grid, next_cell_info, previous_kji0, traj_xyz, trajectory,
                 use_single_layer_tactics):
    log.debug('no success during full column search')
    if np.any(previous_kji0 == 0) or np.any(previous_kji0 == grid.extent_kji - 1):
        log.debug('looking for skin re-entry after possible kissing exit')
        skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
        try_entry_xyz, try_kji0, try_entry_axis, try_entry_polarity, try_entry_knot = \
            skin.find_first_intersection_of_trajectory(trajectory, start=entry_knot,
                                                       start_xyz=entry_xyz, nudge=+0.1)
        if try_entry_xyz is not None:
            log.debug('skin re-entry found')
            try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
            next_cell_info = vec.isclose(try_entry_xyz, entry_xyz, tolerance=0.01), try_kji0, try_entry_axis, \
                             try_entry_polarity, try_entry_knot, try_entry_fraction, try_entry_xyz
    return next_cell_info


def __try_cell_entry(entry_knot, entry_xyz, grid, kissed, kji0, next_cell_info, offsets_kji, pinchout_skip_sign,
                     traj_xyz, trajectory):
    for try_index in range(len(offsets_kji)):
        try_kji0 = kji0 + offsets_kji[try_index]
        while np.all(try_kji0 >= 0) and np.all(try_kji0 < grid.extent_kji) and grid.pinched_out(cell_kji0 = try_kji0):
            try_kji0[0] += pinchout_skip_sign
        if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji) or kissed[tuple(try_kji0)]:
            continue
        # use tiny negative nudge to look for entry into try cell with current segment; check that entry
        # point is close
        try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = \
            rqgs.find_first_intersection_of_trajectory_with_cell_surface(trajectory,
                                                                    grid,
                                                                    try_kji0,
                                                                    entry_knot,
                                                                    start_xyz=entry_xyz,
                                                                    nudge=-0.01,
                                                                    quad_triangles=True)
        if try_entry_xyz is not None and vec.isclose(try_entry_xyz, entry_xyz, tolerance = 0.02):
            log.debug(f'try accepted for cell: {try_kji0}')
            try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
            # replace the next cell entry info
            next_cell_info = True, try_kji0, try_entry_axis, try_entry_polarity, try_entry_knot, try_entry_fraction, \
                             try_entry_xyz
            break
    return next_cell_info


def __segment_fraction(trajectory_xyz, segment, xyz):
    # returns fraction of way along segment that point xyz lies
    segment_vector = trajectory_xyz[segment + 1] - trajectory_xyz[segment]
    segment_length = vec.naive_length(segment_vector)  # note this is length in straight line, rather than diff in mds
    return vec.naive_length(xyz - trajectory_xyz[segment]) / segment_length


def __trajectory_init(blocked_well, grid, grid_crs):
    if bu.matching_uuids(blocked_well.trajectory.crs_uuid, grid.crs_uuid):
        trajectory = blocked_well.trajectory
        assert grid_crs == rqc.Crs(blocked_well.model, uuid = trajectory.crs_uuid)
    else:
        # create a local trajectory object for crs alignment of control points with grid
        # NB. temporary objects, relationships left in a mess
        # model = rq.Model(create_basics = True)
        log.debug(f'creating temp trajectory in grid crs for well {rqw.well_name(blocked_well.trajectory)}')
        model = blocked_well.trajectory.model
        trajectory = rqw.Trajectory(model, uuid = blocked_well.trajectory.uuid)
        assert trajectory is not None and trajectory.control_points is not None
        traj_crs = rqc.Crs(model, uuid = trajectory.crs_uuid)
        traj_crs.convert_array_to(grid_crs, trajectory.control_points)  # trajectory xyz converted in situ to grid's crs
        trajectory.crs_uuid = grid_crs.uuid
        log.debug(
            f'temp traj xyz box: {np.min(trajectory.control_points, axis = 0)} to {np.max(trajectory.control_points, axis = 0)}'
        )
        # note: any represented interpretation object will not be present in the temporary model
    return trajectory


def __pl(n, use_es = False):
    if n == 1:
        return ''
    return 'es' if use_es else 's'
