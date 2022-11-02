"""Submodule containing the fault related grid functions."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.vector_utilities as vec


def find_faults(grid, set_face_sets = False, create_organizing_objects_where_needed = False):
    """Searches for column-faces that are faulted and assigns fault ids; creates list of column-faces per fault id.

    note:
       this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
       GridConnectionSet class
    """

    # note:the logic to group kelp into distinct fault ids is simplistic and won't always give the right grouping

    if set_face_sets:
        grid.clear_face_sets()

    if hasattr(grid, 'fault_dict') and grid.fault_dict is not None and len(grid.fault_dict.keys()) > 0:
        if set_face_sets:
            for f, (j_list, i_list) in grid.fault_dict.items():
                fault_name = f'fault_{f}' if isinstance(f, int) else f
                grid.face_set_dict[fault_name] = (j_list, i_list, 'K')
            grid.set_face_set_gcs_list_from_dict(grid.face_set_dict, create_organizing_objects_where_needed)
        return None

    log.info('looking for faults in grid')
    grid.create_column_pillar_mapping()
    if not grid.has_split_coordinate_lines:
        log.info('grid does not have split coordinate lines, ie. is unfaulted')
        grid.fault_dict = None
        return None

    # note: if Ni or Nj is 1, the kelp array has zero size, but that seems to be handled okay
    kelp_j = np.zeros((grid.extent_kji[1] - 1, grid.extent_kji[2]), dtype = 'int')  # fault id between cols j, j+1
    kelp_i = np.zeros((grid.extent_kji[1], grid.extent_kji[2] - 1), dtype = 'int')  # fault id between cols i, i+1

    last_fault_id = 0

    # look for splits affecting j faces
    for j in range(grid.extent_kji[1] - 1):
        for i in range(grid.extent_kji[2]):
            if (grid.pillars_for_column[j, i, 1, 0] != grid.pillars_for_column[j + 1, i, 0, 0] or
                    grid.pillars_for_column[j, i, 1, 1] != grid.pillars_for_column[j + 1, i, 0, 1]):
                if i > 0 and kelp_j[j, i - 1] > 0:
                    kelp_j[j, i] = kelp_j[j, i - 1]
                else:
                    last_fault_id += 1
                    kelp_j[j, i] = last_fault_id

    # look for splits affecting i faces
    for i in range(grid.extent_kji[2] - 1):
        for j in range(grid.extent_kji[1]):
            if (grid.pillars_for_column[j, i, 0, 1] != grid.pillars_for_column[j, i + 1, 0, 0] or
                    grid.pillars_for_column[j, i, 1, 1] != grid.pillars_for_column[j, i + 1, 1, 0]):
                if j > 0 and kelp_i[j - 1, i] > 0:
                    kelp_i[j, i] = kelp_i[j - 1, i]
                else:
                    last_fault_id += 1
                    kelp_i[j, i] = last_fault_id

    # make pass over kelp to reduce distinct ids: combine where pillar has exactly 2 kelps, one in each of i and j
    if kelp_j.size and kelp_i.size:
        for j in range(grid.extent_kji[1] - 1):
            for i in range(grid.extent_kji[2] - 1):
                if (bool(kelp_j[j, i]) != bool(kelp_j[j, i + 1])) and (bool(kelp_i[j, i]) != bool(kelp_i[j + 1, i])):
                    j_id = kelp_j[j, i] + kelp_j[j, i + 1]  # ie. the non-zero value
                    i_id = kelp_i[j, i] + kelp_i[j + 1, i]
                    if j_id == i_id:
                        continue
                    # log.debug('merging fault id {} into {}'.format(i_id, j_id))
                    kelp_i = np.where(kelp_i == i_id, j_id, kelp_i)
                    kelp_j = np.where(kelp_j == i_id, j_id, kelp_j)

    fault_id_list = np.unique(np.concatenate(
        (np.unique(kelp_i.flatten()), np.unique(kelp_j.flatten()))))[1:]  # discard zero from list
    log.info('number of distinct faults: ' + str(fault_id_list.size))
    # for each fault id, make pair of tuples of kelp locations
    grid.fault_dict = {}  # maps fault_id to pair (j faces, i faces) of array of [j, i] kelp indices for that fault_id
    for fault_id in fault_id_list:
        grid.fault_dict[fault_id] = (np.stack(np.where(kelp_j == fault_id),
                                              axis = 1), np.stack(np.where(kelp_i == fault_id), axis = 1))
    grid.fault_id_j = kelp_j.copy()  # fault_id for each internal j kelp, zero is none; extent nj-1, ni
    grid.fault_id_i = kelp_i.copy()  # fault_id for each internal i kelp, zero is none; extent nj, ni-1
    if set_face_sets:
        for f, (j_list, i_list) in grid.fault_dict.items():
            fault_name = f'fault_{f}'
            grid.face_set_dict[fault_name] = (j_list, i_list, 'K')
        grid.set_face_set_gcs_list_from_dict(grid.face_set_dict, create_organizing_objects_where_needed)
    return (grid.fault_id_j, grid.fault_id_i)


def fault_throws(grid):
    """Finds mean throw of each J and I face; adds throw arrays as attributes to this grid and returns them.

    note:
       this method is deprecated, or due for overhaul to make compatible with resqml_fault module and the
       GridConnectionSet class
    """

    if hasattr(grid, 'fault_throw_j') and grid.fault_throw_j is not None and hasattr(
            grid, 'fault_throw_i') and grid.fault_throw_i is not None:
        return (grid.fault_throw_j, grid.fault_throw_i)
    if not grid.has_split_coordinate_lines:
        return None
    if not hasattr(grid, 'fault_id_j') or grid.fault_id_j is None or not hasattr(
            grid, 'fault_id_i') or grid.fault_id_i is None:
        grid.find_faults()
        if not hasattr(grid, 'fault_id_j'):
            return None
    log.debug('computing fault throws (deprecated method)')
    cp = grid.corner_points(cache_cp_array = True)
    grid.fault_throw_j = np.zeros((grid.nk, grid.nj - 1, grid.ni))
    grid.fault_throw_i = np.zeros((grid.nk, grid.nj, grid.ni - 1))
    grid.fault_throw_j = np.where(grid.fault_id_j == 0, 0.0,
                                  0.25 * np.sum(cp[:, 1:, :, :, 0, :, 2] - cp[:, :-1, :, :, 1, :, 2], axis = (3, 4)))
    grid.fault_throw_i = np.where(grid.fault_id_i == 0, 0.0,
                                  0.25 * np.sum(cp[:, :, 1:, :, :, 0, 2] - cp[:, :, -1:, :, :, 1, 2], axis = (3, 4)))
    return (grid.fault_throw_j, grid.fault_throw_i)


def fault_throws_per_edge_per_column(grid, mode = 'maximum', simple_z = False, axis_polarity_mode = True):
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

    assert mode in ['maximum', 'mean', 'minimum']
    if not simple_z:
        assert grid.z_units() == grid.xy_units(), 'differing xy and z units not supported for non-simple-z fault throws'

    log.debug('computing fault throws per edge per column based on corner point geometry')
    if not grid.has_split_coordinate_lines:  # note: no NaNs returned in this situation
        if axis_polarity_mode:
            return np.zeros((grid.nj, grid.ni, 2, 2))
        return np.zeros((grid.nj, grid.ni, 4))
    grid.create_column_pillar_mapping()
    i_pillar_throws = (
        grid.points_cached[:, grid.pillars_for_column[:, :-1, :, 1], 2]
        -  # (nk+1, nj, ni-1, jp) +ve dz I- cell > I+ cell
        grid.points_cached[:, grid.pillars_for_column[:, 1:, :, 0], 2])
    j_pillar_throws = (grid.points_cached[:, grid.pillars_for_column[:-1, :, 1, :], 2] -
                       grid.points_cached[:, grid.pillars_for_column[1:, :, 0, :], 2])
    if not simple_z:
        i_pillar_throws = np.sign(
            i_pillar_throws)  # note: will return zero if displacement is purely horizontal wrt. z axis
        j_pillar_throws = np.sign(j_pillar_throws)
        i_pillar_throws *= vec.naive_lengths(grid.points_cached[:, grid.pillars_for_column[:, :-1, :, 1], :] -
                                             grid.points_cached[:, grid.pillars_for_column[:, 1:, :, 0], :])
        j_pillar_throws *= vec.naive_lengths(grid.points_cached[:, grid.pillars_for_column[:-1, :, 1, :], :] -
                                             grid.points_cached[:, grid.pillars_for_column[1:, :, 0, :], :])

    if mode == 'mean':
        i_edge_throws = np.nanmean(i_pillar_throws, axis = (0, -1))  # (nj, ni-1)
        j_edge_throws = np.nanmean(j_pillar_throws, axis = (0, -1))  # (nj-1, ni)
    else:
        min_i_edge_throws = np.nanmean(np.nanmin(i_pillar_throws, axis = 0), axis = -1)
        max_i_edge_throws = np.nanmean(np.nanmax(i_pillar_throws, axis = 0), axis = -1)
        min_j_edge_throws = np.nanmean(np.nanmin(j_pillar_throws, axis = 0), axis = -1)
        max_j_edge_throws = np.nanmean(np.nanmax(j_pillar_throws, axis = 0), axis = -1)
        i_flip_mask = (np.abs(min_i_edge_throws) > np.abs(max_i_edge_throws))
        j_flip_mask = (np.abs(min_j_edge_throws) > np.abs(max_j_edge_throws))
        if mode == 'maximum':
            i_edge_throws = np.where(i_flip_mask, min_i_edge_throws, max_i_edge_throws)
            j_edge_throws = np.where(j_flip_mask, min_j_edge_throws, max_j_edge_throws)
        elif mode == 'minimum':
            i_edge_throws = np.where(i_flip_mask, max_i_edge_throws, min_i_edge_throws)
            j_edge_throws = np.where(j_flip_mask, max_j_edge_throws, min_j_edge_throws)
        else:
            raise Exception('code failure')

    # positive values indicate column has greater z values, ie. downthrown if z increases with depth
    if axis_polarity_mode:
        throws = np.zeros((grid.nj, grid.ni, 2, 2))  # first 2 is I (0) or J (1); final 2 is -ve or +ve face
        throws[1:, :, 0, 0] = -j_edge_throws  # J-
        throws[:-1, :, 0, 1] = j_edge_throws  # J+
        throws[:, 1:, 1, 0] = -i_edge_throws  # I-
        throws[:, :-1, 1, 1] = i_edge_throws  # I+

    else:  # resqml protocol
        # order I-, J+, I+, J- as required for properties with 'edges per column' indexable element
        throws = np.zeros((grid.nj, grid.ni, 4))
        throws[:, 1:, 0] = -i_edge_throws  # I-
        throws[:-1, :, 1] = j_edge_throws  # J+
        throws[:, :-1, 2] = i_edge_throws  # I+
        throws[1:, :, 3] = -j_edge_throws  # J-

    return throws
