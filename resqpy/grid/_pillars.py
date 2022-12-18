"""A submodule containing functions relating to grid pillars."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.vector_utilities as vec


def create_column_pillar_mapping(grid):
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

    if hasattr(grid, 'pillars_for_column') and grid.pillars_for_column is not None:
        return grid.pillars_for_column

    grid.cache_all_geometry_arrays()

    grid.pillars_for_column = np.empty((grid.nj, grid.ni, 2, 2), dtype = int)
    ni_plus_1 = grid.ni + 1

    for j in range(grid.nj):
        grid.pillars_for_column[j, :, 0, 0] = np.arange(j * ni_plus_1, (j + 1) * ni_plus_1 - 1, dtype = int)
        grid.pillars_for_column[j, :, 0, 1] = np.arange(j * ni_plus_1 + 1, (j + 1) * ni_plus_1, dtype = int)
        grid.pillars_for_column[j, :, 1, 0] = np.arange((j + 1) * ni_plus_1, (j + 2) * ni_plus_1 - 1, dtype = int)
        grid.pillars_for_column[j, :, 1, 1] = np.arange((j + 1) * ni_plus_1 + 1, (j + 2) * ni_plus_1, dtype = int)

    if grid.has_split_coordinate_lines:
        unsplit_pillar_count = (grid.nj + 1) * ni_plus_1
        extras_count = len(grid.split_pillar_indices_cached)
        for extra_index in range(extras_count):
            primary = grid.split_pillar_indices_cached[extra_index]
            primary_ji0 = divmod(primary, grid.ni + 1)
            extra_pillar_index = unsplit_pillar_count + extra_index
            if extra_index == 0:
                start = 0
            else:
                start = grid.cols_for_split_pillars_cl[extra_index - 1]
            for cpscl_index in range(start, grid.cols_for_split_pillars_cl[extra_index]):
                col = grid.cols_for_split_pillars[cpscl_index]
                j, i = divmod(col, grid.ni)
                jp = primary_ji0[0] - j
                ip = primary_ji0[1] - i
                assert (jp == 0 or jp == 1) and (ip == 0 or ip == 1)
                grid.pillars_for_column[j, i, jp, ip] = extra_pillar_index

    return grid.pillars_for_column


def pillar_foursome(grid, ji0, none_if_unsplit = False):
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

    grid.cache_all_geometry_arrays()

    primary = (grid.ni + 1) * j0 + i0
    foursome = np.full((2, 2), primary, dtype = int)  # axes are: jp, ip
    if not grid.has_split_coordinate_lines:
        return None if none_if_unsplit else foursome
    extras = np.where(grid.split_pillar_indices_cached == primary)[0]
    if len(extras) == 0:
        return None if none_if_unsplit else foursome

    primary_count = (grid.nj + 1) * (grid.ni + 1)
    assert len(grid.cols_for_split_pillars) == grid.cols_for_split_pillars_cl[-1]
    for cpscl_index in extras:
        if cpscl_index == 0:
            start_index = 0
        else:
            start_index = grid.cols_for_split_pillars_cl[cpscl_index - 1]
        for csp_index in range(start_index, grid.cols_for_split_pillars_cl[cpscl_index]):
            natural_col = grid.cols_for_split_pillars[csp_index]
            col_j0_e, col_i0_e = divmod(natural_col, grid.ni)
            col_j0_e -= (j0 - 1)
            col_i0_e -= (i0 - 1)
            assert col_j0_e in [0, 1] and col_i0_e in [0, 1]
            foursome[col_j0_e, col_i0_e] = primary_count + cpscl_index

    return foursome


def pillar_distances_sqr(grid, xy, ref_k0 = 0, kp = 0, horizon_points = None):
    """Returns array of the square of the distances of primary pillars in x,y plane to point xy.

    arguments:
       xy (float pair): the xy coordinate to compute the pillar distances to
       ref_k0 (int, default 0): the horizon layer number to use
       horizon_points (numpy array, optional): if present, should be array as returned by
          horizon_points() method; pass for efficiency in case of multiple calls
    """

    # note: currently works with unmasked data and using primary pillars only
    pe_j = grid.extent_kji[1] + 1
    pe_i = grid.extent_kji[2] + 1
    if horizon_points is None:
        horizon_points = grid.horizon_points(ref_k0 = ref_k0, kp = kp)
    pillar_xy = horizon_points[:, :, 0:2]
    dxy = pillar_xy - xy
    dxy2 = dxy * dxy
    return (dxy2[:, :, 0] + dxy2[:, :, 1]).reshape((pe_j, pe_i))


def nearest_pillar(grid, xy, ref_k0 = 0, kp = 0):
    """Returns the (j0, i0) indices of the primary pillar with point closest in x,y plane to point xy."""

    # note: currently works with unmasked data and using primary pillars only
    pe_i = grid.extent_kji[2] + 1
    sum_dxy2 = grid.pillar_distances_sqr(xy, ref_k0 = ref_k0, kp = kp)
    ji = np.nanargmin(sum_dxy2)
    j, i = divmod(ji, pe_i)
    return (j, i)


def nearest_rod(grid, xyz, projection, axis, ref_slice0 = 0, plus_face = False):
    """Returns the (k0, j0) or (k0 ,i0) indices of the closest point(s) to xyz(s); projection is 'xy', 'xz' or 'yz'.

    note:
       currently only for unsplit grids
    """

    x_sect = grid.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)
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
