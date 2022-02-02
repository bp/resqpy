"""Submodule containing the face related grid functions."""

import logging

log = logging.getLogger(__name__)
import numpy as np
import pandas as pd

import resqpy.fault as rqf
import resqpy.olio.vector_utilities as vec


def is_split_column_face(grid, j0, i0, axis, polarity):
    """Returns True if the I or J column face is split; False otherwise."""

    if not grid.has_split_coordinate_lines:
        return False
    assert axis in (1, 2)
    if axis == 1:  # J
        ip = i0
        if polarity:
            if j0 == grid.nj - 1:
                return False
            jp = j0 + 1
        else:
            if j0 == 0:
                return False
            jp = j0 - 1
    else:  # I
        jp = j0
        if polarity:
            if i0 == grid.ni - 1:
                return False
            ip = i0 + 1
        else:
            if i0 == 0:
                return False
            ip = i0 - 1
    cpm = grid.create_column_pillar_mapping()
    if axis == 1:
        return ((cpm[j0, i0, polarity, 0] != cpm[jp, ip, 1 - polarity, 0]) or
                (cpm[j0, i0, polarity, 1] != cpm[jp, ip, 1 - polarity, 1]))
    else:
        return ((cpm[j0, i0, 0, polarity] != cpm[jp, ip, 0, 1 - polarity]) or
                (cpm[j0, i0, 1, polarity] != cpm[jp, ip, 1, 1 - polarity]))


def split_column_faces(grid):
    """Returns a pair of numpy boolean arrays indicating which internal column faces (column edges) are split."""

    if not grid.has_split_coordinate_lines:
        return None, None
    if (hasattr(grid, 'array_j_column_face_split') and grid.array_j_column_face_split is not None and
            hasattr(grid, 'array_i_column_face_split') and grid.array_i_column_face_split is not None):
        return grid.array_j_column_face_split, grid.array_i_column_face_split
    if grid.nj == 1:
        grid.array_j_column_face_split = None
    else:
        grid.array_j_column_face_split = np.zeros((grid.nj - 1, grid.ni),
                                                  dtype = bool)  # NB. internal faces only, index for +ve face
    if grid.ni == 1:
        grid.array_i_column_face_split = None
    else:
        grid.array_i_column_face_split = np.zeros((grid.nj, grid.ni - 1),
                                                  dtype = bool)  # NB. internal faces only, index for +ve face
    grid.create_column_pillar_mapping()
    for spi in grid.split_pillar_indices_cached:
        j_p, i_p = divmod(spi, grid.ni + 1)
        if j_p > 0 and j_p < grid.nj:
            if i_p > 0 and grid.is_split_column_face(j_p, i_p - 1, 1, 0):
                grid.array_j_column_face_split[j_p - 1, i_p - 1] = True
            if i_p < grid.ni - 1 and grid.is_split_column_face(j_p, i_p, 1, 0):
                grid.array_j_column_face_split[j_p - 1, i_p] = True
        if i_p > 0 and i_p < grid.ni:
            if j_p > 0 and grid.is_split_column_face(j_p - 1, i_p, 2, 0):
                grid.array_i_column_face_split[j_p - 1, i_p - 1] = True
            if j_p < grid.nj - 1 and grid.is_split_column_face(j_p, i_p, 2, 0):
                grid.array_i_column_face_split[j_p, i_p - 1] = True
    return grid.array_j_column_face_split, grid.array_i_column_face_split


def clear_face_sets(grid):
    """Discard face sets."""
    # following maps face_set_id to (j faces, i faces, 'K') of array of [j, i] kelp indices for that face_set_id
    # or equivalent for axes 'J' or 'I'
    grid.face_set_dict = {}
    grid.face_set_gcs_list = []


def set_face_set_gcs_list_from_dict(grid, face_set_dict = None, create_organizing_objects_where_needed = False):
    """Creates a grid connection set for each feature in the face set dictionary, based on kelp list pairs."""

    if face_set_dict is None:
        face_set_dict = grid.face_set_dict
    grid.face_set_gcs_list = []
    for feature, kelp_values in face_set_dict.items():
        gcs = rqf.GridConnectionSet(grid.model, grid = grid)
        if len(kelp_values) == 2:
            kelp_j, kelp_i = kelp_values
            axis = 'K'
        elif len(kelp_values) == 3:
            kelp_j, kelp_i, axis = kelp_values
        else:
            raise ValueError('grid face set dictionary item messed up')
        log.debug(f'creating gcs for: {feature} {axis}')
        gcs.set_pairs_from_kelp(kelp_j, kelp_i, feature, create_organizing_objects_where_needed, axis = axis)
        grid.face_set_gcs_list.append(gcs)


# TODO: make separate curtain and K-face versions of following function
def make_face_set_from_dataframe(grid, df):
    """Creates a curtain face set for each named fault in dataframe.

    note:
       this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
       GridConnectionSet class
    """

    # df columns: name, i1, i2, j1, j2, k1, k2, face
    grid.clear_face_sets()
    names = pd.unique(df.name)
    count = 0
    box_kji0 = np.zeros((2, 3), dtype = int)
    k_warning_given = False
    for fs_name in names:
        i_kelp_list = []
        j_kelp_list = []
        # ignore k faces for now
        fs_ds = df[df.name == fs_name]
        for row in range(len(fs_ds)):
            face = fs_ds.iloc[row]['face']
            fl = face[0].upper()
            if fl in 'IJK':
                axis = 'KJI'.index(fl)
            elif fl in 'XYZ':
                axis = 'ZYX'.index(fl)
            else:
                raise ValueError('fault data face not recognized: ' + face)
            if axis == 0:
                continue  # ignore k faces for now
            box_kji0[0, 0] = fs_ds.iloc[row]['k1'] - 1  # k1
            box_kji0[1, 0] = fs_ds.iloc[row]['k2'] - 1  # k2
            box_kji0[0, 1] = fs_ds.iloc[row]['j1'] - 1  # j1
            box_kji0[1, 1] = fs_ds.iloc[row]['j2'] - 1  # j2
            box_kji0[0, 2] = fs_ds.iloc[row]['i1'] - 1  # i1
            box_kji0[1, 2] = fs_ds.iloc[row]['i2'] - 1  # i2
            box_kji0[1, 0] = min(box_kji0[1, 0], grid.extent_kji[0] - 1)
            if not k_warning_given and (box_kji0[0, 0] != 0 or box_kji0[1, 0] != grid.extent_kji[0] - 1):
                log.warning(
                    'one or more entries in face set dataframe does not cover entire layer range: extended to all layers'
                )
                k_warning_given = True
            if len(face) > 1 and face[1] == '-':  # treat negative faces as positive faces of neighbouring cell
                box_kji0[0, axis] = max(box_kji0[0, axis] - 1, 0)
                box_kji0[1, axis] -= 1
            else:
                box_kji0[1, axis] = min(box_kji0[1, axis], grid.extent_kji[axis] - 2)
            if box_kji0[1, axis] < box_kji0[0, axis]:
                continue  # faces are all on edge of grid
            # for now ignore layer range and create curtain of j and i kelp
            for j in range(box_kji0[0, 1], box_kji0[1, 1] + 1):
                for i in range(box_kji0[0, 2], box_kji0[1, 2] + 1):
                    if axis == 1:
                        __add_to_kelp_list(grid.extent_kji, j_kelp_list, True, (j, i))
                    elif axis == 2:
                        __add_to_kelp_list(grid.extent_kji, i_kelp_list, False, (j, i))
        grid.face_set_dict[fs_name] = (j_kelp_list, i_kelp_list, 'K')
        count += 1
    log.info(str(count) + ' face sets extracted from dataframe')


def make_face_sets_from_pillar_lists(grid,
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

    # NB. this code was originally written for axis K and projection xy, working with horizon points
    # it has since been reworked for the cross sectional cases, so variables named 'pillar...' may refer to rods
    # and i_... and j_... may actually represent i,j or i,k or j,k

    assert axis in ['K', 'J', 'I']
    assert projection in ['xy', 'xz', 'yz']

    local_face_set_dict = {}
    full_pillar_list_dict = {}

    if not hasattr(grid, 'face_set_dict'):
        grid.clear_face_sets()

    if axis.upper() == 'K':
        assert projection == 'xy'
        pillar_xy = grid.horizon_points(ref_k0 = ref_slice0, kp = 1 if plus_face else 0)[:, :, 0:2]
        kelp_axes = 'JI'
    else:
        if projection == 'xz':
            pillar_xy = grid.unsplit_x_section_points(axis, ref_slice0 = ref_slice0,
                                                      plus_face = plus_face)[:, :, 0:3:2]  # x,z
        else:  # projection == 'yz'
            pillar_xy = grid.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)[:, :,
                                                                                                            1:]  # y,z
        if axis.upper() == 'J':
            kelp_axes = 'KI'
        else:
            kelp_axes = 'KJ'
    kelp_axes_int = np.empty((2,), dtype = int)
    for a in range(2):
        kelp_axes_int[a] = 'KJI'.index(kelp_axes[a])
    #     grid.clear_face_sets()  # now accumulating sets
    face_set_count = 0
    here = np.zeros(2, dtype = int)
    side_step = np.zeros(2, dtype = int)
    for pillar_list in pillar_list_list:
        if len(pillar_list) < 2:
            continue
        full_pillar_list = [pillar_list[0]]
        face_set_count += 1
        if len(pillar_list_list) > 1:
            id_suffix = '_line_' + str(face_set_count)
        else:
            id_suffix = ''
        # i,j are as stated if axis is 'K'; for axis 'J', i,j are actually i,k; for axis 'I', i,j are actually j,k
        i_kelp_list = []
        j_kelp_list = []
        for p in range(len(pillar_list) - 1):
            ji_0 = pillar_list[p]
            ji_1 = pillar_list[p + 1]
            if np.all(ji_0 == ji_1):
                continue
            # xy might actually be xy, xz or yz depending on projection
            xy_0 = pillar_xy[tuple(ji_0)]
            xy_1 = pillar_xy[tuple(ji_1)]
            if vec.isclose(xy_0, xy_1):
                continue
            dj = ji_1[0] - ji_0[0]
            di = ji_1[1] - ji_0[1]
            abs_dj = abs(dj)
            abs_di = abs(di)
            if dj < 0:
                j_sign = -1
            else:
                j_sign = 1
            if di < 0:
                i_sign = -1
            else:
                i_sign = 1
            here[:] = ji_0
            while np.any(here != ji_1):
                previous = here.copy()  # debug
                if abs_dj >= abs_di:
                    __j_greater_than_i(di, full_pillar_list, grid, here, i_kelp_list, i_sign, j_kelp_list, j_sign, ji_1,
                                       kelp_axes, kelp_axes_int, pillar_xy, side_step, xy_0, xy_1)
                else:
                    __i_greater_than_j(dj, full_pillar_list, grid, here, i_kelp_list, i_sign, j_kelp_list, j_sign, ji_1,
                                       kelp_axes, kelp_axes_int, pillar_xy, side_step, xy_0, xy_1)
                assert np.any(here != previous), 'failed to move'
        grid.face_set_dict[face_set_id + id_suffix] = (j_kelp_list, i_kelp_list, axis)
        local_face_set_dict[face_set_id + id_suffix] = (j_kelp_list, i_kelp_list, axis)
        full_pillar_list_dict[face_set_id + id_suffix] = full_pillar_list.copy()

    return local_face_set_dict, full_pillar_list_dict


def __i_greater_than_j(dj, full_pillar_list, grid, here, i_kelp_list, i_sign, j_kelp_list, j_sign, ji_1, kelp_axes,
                       kelp_axes_int, pillar_xy, side_step, xy_0, xy_1):
    i = here[1]
    if i != ji_1[1]:
        ip = i + i_sign
        __add_to_kelp_list(grid.extent_kji, j_kelp_list, kelp_axes[0], (here[0] - 1, min(i, ip)))
        here[1] = ip
        full_pillar_list.append(tuple(here))
    if dj != 0:
        divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(here)], xy_0, xy_1)
        side_step[:] = here
        side_step[0] += j_sign
        if side_step[0] >= 0 and side_step[0] <= grid.extent_kji[kelp_axes_int[0]]:
            stepped_divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(side_step)], xy_0, xy_1)
            if stepped_divergence < divergence:
                here[:] = side_step
                __add_to_kelp_list(grid.extent_kji, i_kelp_list, kelp_axes[1],
                                   (min(here[0], here[0] - j_sign), here[1] - 1))
                full_pillar_list.append(tuple(here))


def __j_greater_than_i(di, full_pillar_list, grid, here, i_kelp_list, i_sign, j_kelp_list, j_sign, ji_1, kelp_axes,
                       kelp_axes_int, pillar_xy, side_step, xy_0, xy_1):
    j = here[0]
    if j != ji_1[0]:
        jp = j + j_sign
        __add_to_kelp_list(grid.extent_kji, i_kelp_list, kelp_axes[1], (min(j, jp), here[1] - 1))
        here[0] = jp
        full_pillar_list.append(tuple(here))
    if di != 0:
        divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(here)], xy_0, xy_1)
        side_step[:] = here
        side_step[1] += i_sign
        if side_step[1] >= 0 and side_step[1] <= grid.extent_kji[kelp_axes_int[1]]:
            stepped_divergence = vec.point_distance_to_line_2d(pillar_xy[tuple(side_step)], xy_0, xy_1)
            if stepped_divergence < divergence:
                here[:] = side_step
                __add_to_kelp_list(grid.extent_kji, j_kelp_list, kelp_axes[0],
                                   (here[0] - 1, min(here[1], here[1] - i_sign)))
                full_pillar_list.append(tuple(here))


def face_centre(grid,
                cell_kji0,
                axis,
                zero_or_one,
                points_root = None,
                cache_resqml_array = True,
                cache_cp_array = False):
    """Returns xyz location of the centre point of a face of the cell (or all cells)."""

    if axis not in [0, 1, 2]:
        raise ValueError('Axis provided must be either 0, 1, or 2.')

    # todo: optionally compute for all cells and cache
    cp = grid.corner_points(cell_kji0,
                            points_root = points_root,
                            cache_resqml_array = cache_resqml_array,
                            cache_cp_array = cache_cp_array)
    if cell_kji0 is None:
        if axis == 0:
            return 0.25 * np.sum(cp[:, :, :, zero_or_one, :, :], axis = (3, 4))
        elif axis == 1:
            return 0.25 * np.sum(cp[:, :, :, :, zero_or_one, :], axis = (3, 4))
        else:
            return 0.25 * np.sum(cp[:, :, :, :, :, zero_or_one], axis = (3, 4))
    else:
        if axis == 0:
            return 0.25 * np.sum(cp[zero_or_one, :, :], axis = (0, 1))
        elif axis == 1:
            return 0.25 * np.sum(cp[:, zero_or_one, :], axis = (0, 1))
        else:
            return 0.25 * np.sum(cp[:, :, zero_or_one], axis = (0, 1))


def face_centres_kji_01(grid, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
    """Returns an array of shape (3, 2, 3) being (axis, 0 or 1, xyz) of face centre points for cell."""

    assert cell_kji0 is not None
    result = np.zeros((3, 2, 3))
    for axis in range(3):
        for zero_or_one in range(2):
            result[axis, zero_or_one] = grid.face_centre(cell_kji0,
                                                         axis,
                                                         zero_or_one,
                                                         points_root = points_root,
                                                         cache_resqml_array = cache_resqml_array,
                                                         cache_cp_array = cache_cp_array)
    return result


def __add_to_kelp_list(extent_kji, kelp_list, face_axis, ji):
    if isinstance(face_axis, bool):
        face_axis = 'J' if face_axis else 'I'
    # ignore external faces
    if face_axis == 'J':
        if ji[0] < 0 or ji[0] >= extent_kji[1] - 1:
            return
    elif face_axis == 'I':
        if ji[1] < 0 or ji[1] >= extent_kji[2] - 1:
            return
    else:  # ji is actually kj or ki
        assert face_axis == 'K'
        if ji[0] < 0 or ji[0] >= extent_kji[0] - 1:
            return
    pair = ji
    if pair in kelp_list:
        return  # avoid duplication
    kelp_list.append(pair)
