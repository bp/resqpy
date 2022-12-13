"""A submodule containing functions relating to xyz grid coordinates"""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.xml_et as rqet
import resqpy.weights_and_measures as bwam
import resqpy.crs as rqc
import resqpy.olio.vector_utilities as vec


def xyz_box(grid, points_root = None, lazy = True, local = False):
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

    if grid.xyz_box_cached is None or (not lazy and not grid.xyz_box_cached_thoroughly):
        grid.xyz_box_cached = np.zeros((2, 3))
        if lazy:
            eight_corners = np.zeros((2, 2, 2, 3))
            for kp in [0, 1]:
                for jp in [0, 1]:
                    for ip in [0, 1]:
                        cell_kji0 = [
                            kp * (grid.extent_kji[0] - 1), jp * (grid.extent_kji[1] - 1), ip * (grid.extent_kji[2] - 1)
                        ]
                        eight_corners[kp, jp, ip] = grid.point(cell_kji0 = cell_kji0,
                                                               corner_index = [kp, jp, ip],
                                                               points_root = points_root,
                                                               cache_array = False)
            grid.xyz_box_cached[0, :] = np.nanmin(eight_corners, axis = (0, 1, 2))
            grid.xyz_box_cached[1, :] = np.nanmax(eight_corners, axis = (0, 1, 2))
        else:
            ps = grid.points_ref()
            if grid.has_split_coordinate_lines:
                grid.xyz_box_cached[0, :] = np.nanmin(ps, axis = (0, 1))
                grid.xyz_box_cached[1, :] = np.nanmax(ps, axis = (0, 1))
            else:
                grid.xyz_box_cached[0, :] = np.nanmin(ps, axis = (0, 1, 2))
                grid.xyz_box_cached[1, :] = np.nanmax(ps, axis = (0, 1, 2))
        grid.xyz_box_cached_thoroughly = not lazy
    if local:
        return grid.xyz_box_cached
    global_xyz_box = grid.xyz_box_cached.copy()
    return grid.local_to_global_crs(global_xyz_box, crs_uuid = grid.crs_uuid)


def xyz_box_centre(grid, points_root = None, lazy = False, local = False):
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

    return np.nanmean(grid.xyz_box(points_root = points_root, lazy = lazy, local = local), axis = 0)


def bounding_box(grid, cell_kji0, points_root = None, cache_cp_array = False):
    """Returns the xyz box which envelopes the specified cell, as a numpy array of shape (2, 3)."""

    result = np.zeros((2, 3))
    cp = grid.corner_points(cell_kji0, points_root = points_root, cache_cp_array = cache_cp_array)
    result[0] = np.min(cp, axis = (0, 1, 2))
    result[1] = np.max(cp, axis = (0, 1, 2))
    return result


def composite_bounding_box(grid, bounding_box_list):
    """Returns the xyz box which envelopes all the boxes in the list, as a numpy array of shape (2, 3)."""

    result = bounding_box_list[0]
    for box in bounding_box_list[1:]:
        result[0] = np.minimum(result[0], box[0])
        result[1] = np.maximum(result[1], box[1])
    return result


def _local_to_global_crs(grid,
                         a,
                         crs_uuid = None,
                         global_xy_units = None,
                         global_z_units = None,
                         global_z_increasing_downward = None):
    """Converts array of points in situ from local coordinate system to global one."""

    if crs_uuid is None:
        crs_uuid = grid.crs_uuid
        if crs_uuid is None:
            return a

    crs = rqc.Crs(grid.model, uuid = crs_uuid)

    flat_a = a.reshape((-1, 3))  # flattened view of array a as vector of (x, y, z) points, in situ

    if crs.rotated:
        flat_a[:] = vec.rotate_array(crs.rotation_matrix, flat_a)

    # note: here negation is made in local crs; if z_offset is not zero, this might not be what is intended
    if global_z_increasing_downward is not None:
        if global_z_increasing_downward != crs.z_inc_down:
            flat_a[:, 2] = np.negative(flat_a[:, 2])

    # This code assumes x, y, z offsets are in local crs units
    flat_a[:, 0] += crs.x_offset
    flat_a[:, 1] += crs.y_offset
    flat_a[:, 2] += crs.z_offset

    if global_xy_units is not None:
        bwam.convert_lengths(flat_a[:, 0], crs.xy_units, global_xy_units)  # x
        bwam.convert_lengths(flat_a[:, 1], crs.xy_units, global_xy_units)  # y
    if global_z_units is not None:
        bwam.convert_lengths(flat_a[:, 2], crs.z_units, global_z_units)  # z

    a[:] = flat_a.reshape(a.shape)  # probably not needed
    return a


def z_inc_down(grid):
    """Return True if z increases downwards in the coordinate reference system used by the grid geometry

    :meta common:
    """

    if grid.crs is None:
        assert grid.crs_uuid is not None
        grid.crs = rqc.Crs(grid.model, uuid = grid.crs_uuid)
    return grid.crs.z_inc_down


def _global_to_local_crs(grid,
                         a,
                         crs_uuid = None,
                         global_xy_units = None,
                         global_z_units = None,
                         global_z_increasing_downward = None):
    """Converts array of points in situ from global coordinate system to established local one."""

    if crs_uuid is None:
        crs_uuid = grid.crs_uuid
        if crs_uuid is None:
            return a

    flat_a = a.reshape((-1, 3))  # flattened view of array a as vector of (x, y, z) points, in situ

    crs = rqc.Crs(grid.model, uuid = crs_uuid)

    if global_xy_units is not None:
        bwam.convert_lengths(flat_a[:, 0], global_xy_units, crs.xy_units)  # x
        bwam.convert_lengths(flat_a[:, 1], global_xy_units, crs.xy_units)  # y
    if global_z_units is not None:
        bwam.convert_lengths(flat_a[:, 2], global_z_units, crs.z_units)  # z

    # This code assumes x, y, z offsets are in local crs units
    flat_a[:, 0] -= crs.x_offset
    flat_a[:, 1] -= crs.y_offset
    flat_a[:, 2] -= crs.z_offset

    # note: here negation is made in local crs; if z_offset is not zero, this might not be what is intended
    if global_z_increasing_downward is not None:
        if global_z_increasing_downward != crs.z_inc_down:
            flat_a[:, 2] = np.negative(flat_a[:, 2])

    if crs.rotated:
        flat_a[:] = vec.rotate_array(crs.reverse_rotation_matrix, flat_a)

    a[:] = flat_a.reshape(a.shape)  # probably not needed
    return a


def check_top_and_base_cell_edge_directions(grid):
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
    grid.point(cache_array = True)
    good = True
    if grid.has_split_coordinate_lines:
        # build top and base I & J cell edge vectors
        grid.create_column_pillar_mapping()  # pillar indices for 4 columns around interior pillars
        top_j_edge_vectors_p = np.zeros((grid.nj, grid.ni, 2, 2))  # third axis is ip
        top_i_edge_vectors_p = np.zeros((grid.nj, 2, grid.ni, 2))  # second axis is jp
        base_j_edge_vectors_p = np.zeros((grid.nj, grid.ni, 2, 2))  # third axis is ip
        base_i_edge_vectors_p = np.zeros((grid.nj, 2, grid.ni, 2))  # second axis is jp
        # todo: rework as numpy operations across nj & ni
        for j in range(grid.nj):
            for i in range(grid.ni):
                for jip in range(2):  # serves as either jp or ip
                    top_j_edge_vectors_p[j, i,
                                         jip, :] = (grid.points_cached[0, grid.pillars_for_column[j, i, 1, jip], :2] -
                                                    grid.points_cached[0, grid.pillars_for_column[j, i, 0, jip], :2])
                    base_j_edge_vectors_p[j, i, jip, :] = (
                        grid.points_cached[grid.nk_plus_k_gaps, grid.pillars_for_column[j, i, 1, jip], :2] -
                        grid.points_cached[grid.nk_plus_k_gaps, grid.pillars_for_column[j, i, 0, jip], :2])
                    top_i_edge_vectors_p[j, jip,
                                         i, :] = (grid.points_cached[0, grid.pillars_for_column[j, i, jip, 1], :2] -
                                                  grid.points_cached[0, grid.pillars_for_column[j, i, jip, 0], :2])
                    base_i_edge_vectors_p[j, jip, i, :] = (
                        grid.points_cached[grid.nk_plus_k_gaps, grid.pillars_for_column[j, i, jip, 1], :2] -
                        grid.points_cached[grid.nk_plus_k_gaps, grid.pillars_for_column[j, i, jip, 0], :2])
        # reshape to allow common checking code with unsplit grid vectors (below)
        top_j_edge_vectors = top_j_edge_vectors_p.reshape((grid.nj, 2 * grid.ni, 2))
        top_i_edge_vectors = top_i_edge_vectors_p.reshape((2 * grid.nj, grid.ni, 2))
        base_j_edge_vectors = base_j_edge_vectors_p.reshape((grid.nj, 2 * grid.ni, 2))
        base_i_edge_vectors = base_i_edge_vectors_p.reshape((2 * grid.nj, grid.ni, 2))
    else:
        top_j_edge_vectors = (grid.points_cached[0, 1:, :, :2] - grid.points_cached[0, :-1, :, :2]).reshape(
            (grid.nj, grid.ni + 1, 2))
        top_i_edge_vectors = (grid.points_cached[0, :, 1:, :2] - grid.points_cached[0, :, :-1, :2]).reshape(
            (grid.nj + 1, grid.ni, 2))
        base_j_edge_vectors = (grid.points_cached[-1, 1:, :, :2] - grid.points_cached[-1, :-1, :, :2]).reshape(
            (grid.nj, grid.ni + 1, 2))
        base_i_edge_vectors = (grid.points_cached[-1, :, 1:, :2] - grid.points_cached[-1, :, :-1, :2]).reshape(
            (grid.nj + 1, grid.ni, 2))
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
        log.warning('top cell edges for neighbouring cells flip direction somewhere: this grid is probably unusable')
        good = False
    dot_jp = np.sum(base_j_edge_vectors[1:, :, :] * base_j_edge_vectors[:-1, :, :], axis = 2)
    dot_ip = np.sum(base_i_edge_vectors[1:, :, :] * base_i_edge_vectors[:-1, :, :], axis = 2)
    if not np.all(dot_jp >= 0.0) and np.all(dot_ip >= 0.0):
        log.warning('base cell edges for neighbouring cells flip direction somewhere: this grid is probably unusable')
        good = False
    return good
