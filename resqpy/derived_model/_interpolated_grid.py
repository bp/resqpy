"""High level interpolated grid function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.grid as grr
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet

import resqpy.derived_model._common as rqdm_c


def interpolated_grid(epc_file,
                      grid_a,
                      grid_b,
                      a_to_b_0_to_1 = 0.5,
                      split_tolerance = 0.01,
                      inherit_properties = False,
                      inherit_realization = None,
                      inherit_all_realizations = False,
                      new_grid_title = None,
                      new_epc_file = None):
    """Extends an existing model with a new grid geometry linearly interpolated between the two source_grids.

    arguments:
       epc_file (string): file name to rewrite the model's xml to
       grid_a, grid_b (grid.Grid objects): a pair of RESQML grid objects representing the end cases, between
          which the new grid will be interpolated
       a_to_b_0_to_1 (float, default 0.5): the interpolation factor in the range zero to one; a value of 0.0 will yield
          a copy of grid a, a value of 1.0 will yield a copy of grid b, intermediate values will yield a grid with all
          points interpolated
       split_tolerance (float, default 0.01): maximum offset of corner points for shared point to be generated; units
          are same as those in grid crs; only relevant if working from corner points, ignored otherwise
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with grid_a
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the interpolated grid (& crs)

    returns:
       new grid object (grid.Grid) with geometry interpolated between grid a and grid b

    notes:
       the hdf5 file used by the grid_a model is appended to, so it is recommended that the grid_a model's epc is specified
       as the first argument (unless a new epc file is required, sharing the hdf5 file)
    """

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    assert grid_a is not None and grid_b is not None, 'at least one source grid is missing'
    assert grid_a.grid_representation == 'IjkGrid' and grid_b.grid_representation == 'IjkGrid'
    assert 0.0 <= a_to_b_0_to_1 <= 1.0, 'interpolation factor outside range 0.0 to 1.0'
    assert tuple(grid_a.extent_kji) == tuple(grid_b.extent_kji), 'source grids have different extents'
    assert grid_a.k_direction_is_down == grid_b.k_direction_is_down, 'source grids have different k directions'
    assert grid_a.grid_is_right_handed == grid_b.grid_is_right_handed, 'source grids have different ijk handedness'
    assert grid_a.pillar_shape == grid_b.pillar_shape, 'source grids have different resqml pillar shapes'

    b_weight = a_to_b_0_to_1
    a_weight = 1.0 - b_weight

    model = grid_a.model

    if not bu.matching_uuids(grid_a.crs_uuid, grid_b.crs_uuid):
        crs_a = rqc.Crs(grid_a.model, uuid = grid_a.crs_uuid)
        crs_b = rqc.Crs(grid_b.model, uuid = grid_b.crs_uuid)
        assert crs_a.is_equivalent(crs_b),  \
            'end point grids for interpolation have different coordinate reference systems'

    log.info('loading geometry for two source grids')
    grid_a.cache_all_geometry_arrays()
    grid_b.cache_all_geometry_arrays()

    assert (grid_a.geometry_defined_for_all_cells() and
            grid_b.geometry_defined_for_all_cells()), 'geometry not defined for all cells'
    # assert grid_a.geometry_defined_for_all_pillars() and grid_b.geometry_defined_for_all_pillars(),  \
    #     'geometry not defined for all pillars'

    work_from_pillars = _determine_work_from_pillars(grid_a, grid_b)

    # create a new, empty grid object
    grid = grr.Grid(model)

    # inherit attributes from source grid
    _inherit_basics(grid, grid_a, grid_b)

    if work_from_pillars:
        _interpolate_points_cached_from_pillars(grid, grid_a, grid_b, a_weight, b_weight)
    else:
        _interpolate_points_cached_from_cp(grid, grid_a, grid_b, a_weight, b_weight, split_tolerance)

    collection = rqdm_c._prepare_simple_inheritance(grid, grid_a, inherit_properties, inherit_realization,
                                                    inherit_all_realizations)

    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'interpolated between two grids with factor: ' + str(a_to_b_0_to_1)

    model.h5_release()
    if new_epc_file:
        rqdm_c._write_grid(new_epc_file,
                           grid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'w')
    else:
        ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(grid_a.root, ['Geometry', 'Points']),
                                                      'Coordinates')
        rqdm_c._write_grid(epc_file,
                           grid,
                           ext_uuid = ext_uuid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'a')

    return grid


def _determine_work_from_pillars(grid_a, grid_b):
    if not grid_a.has_split_coordinate_lines and not grid_b.has_split_coordinate_lines:
        work_from_pillars = True
    elif (grid_a.has_split_coordinate_lines and grid_b.has_split_coordinate_lines and
          grid_a.points_cached.shape == grid_b.points_cached.shape and
          grid_a.split_pillar_indices_cached.shape == grid_b.split_pillar_indices_cached.shape and
          grid_a.cols_for_split_pillars.shape == grid_b.cols_for_split_pillars.shape and
          grid_a.cols_for_split_pillars_cl.shape == grid_b.cols_for_split_pillars_cl.shape and
          np.all(grid_a.split_pillar_indices_cached == grid_b.split_pillar_indices_cached) and
          np.all(grid_a.cols_for_split_pillars == grid_b.cols_for_split_pillars) and
          np.all(grid_a.cols_for_split_pillars_cl == grid_b.cols_for_split_pillars_cl)):
        work_from_pillars = True
    else:
        work_from_pillars = False
    if work_from_pillars:
        log.info('interpolating between compatible pillar grids')
    else:
        log.warning('interpolating between corner points due to pillar incompatibilities')
    return work_from_pillars


def _inherit_basics(grid, grid_a, grid_b):
    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = grid_a.extent_kji.copy()
    grid.nk, grid.nj, grid.ni = grid.extent_kji
    grid.k_direction_is_down = grid_a.k_direction_is_down
    grid.grid_is_right_handed = grid_a.grid_is_right_handed
    grid.pillar_shape = grid_a.pillar_shape
    grid.has_split_coordinate_lines = (grid_a.has_split_coordinate_lines or grid_b.has_split_coordinate_lines)
    # inherit the coordinate reference system used by the grid geometry
    grid.crs_uuid = grid_a.crs_uuid
    if grid_a.model is not grid.model:
        grid.model.duplicate_node(grid_a.model.root_for_uuid(grid_a, grid.crs_uuid), add_as_part = True)
    grid.crs = rqc.Crs(grid.model, uuid = grid.crs_uuid)

    if grid_a.inactive is None or grid_b.inactive is None:
        grid.inactive = None
    else:
        grid.inactive = np.logical_and(grid_a.inactive, grid_b.inactive)
    grid.geometry_defined_for_all_cells_cached = True
    grid.array_cell_geometry_is_defined = np.ones(tuple(grid.extent_kji), dtype = bool)
    grid.geometry_defined_for_all_pillars_cached = True


def _interpolate_points_cached_from_pillars(grid, grid_a, grid_b, a_weight, b_weight):
    grid.points_cached = grid_a.points_cached * a_weight + grid_b.points_cached * b_weight
    grid.has_split_coordinate_lines = grid_a.has_split_coordinate_lines
    if grid.has_split_coordinate_lines:
        grid.split_pillar_indices_cached = grid_a.split_pillar_indices_cached.copy()
        grid.cols_for_split_pillars = grid_a.cols_for_split_pillars.copy()
        grid.cols_for_split_pillars_cl = grid_a.cols_for_split_pillars_cl.copy()
        grid.split_pillars_count = grid_a.split_pillars_count


def _interpolate_points_cached_from_cp(grid, grid_a, grid_b, a_weight, b_weight, split_tolerance):
    grid.pillar_shape = 'curved'  # following fesapi approach of non-parametric pillars even if they are in fact straight
    cp_a = grid_a.corner_points(cache_cp_array = True)
    cp_b = grid_b.corner_points(cache_cp_array = True)
    assert cp_a.shape == cp_b.shape
    grid_cp = cp_a * a_weight + cp_b * b_weight
    _close_vertical_voids(grid, grid_cp, grid_a.z_units())
    # reduce cp array extent in k
    log.debug('reducing k extent of interpolated corner point array (sharing points vertically)')
    k_reduced_cp_array = np.zeros((grid.nk + 1, grid.nj, grid.ni, 2, 2, 3))  # (nk+1, nj, ni, jp, ip, xyz)
    k_reduced_cp_array[0, :, :, :, :, :] = grid_cp[0, :, :, 0, :, :, :]
    k_reduced_cp_array[-1, :, :, :, :, :] = grid_cp[-1, :, :, 1, :, :, :]
    if grid.nk > 1:
        k_reduced_cp_array[1:-1, :, :, :, :, :] = grid_cp[:-1, :, :, 1, :, :, :]
    # create primary pillar reference indices as one of four column corners around pillar, active column preferred
    log.debug('creating primary pillar reference neighbourly indices')
    primary_pillar_jip = np.zeros((grid.nj + 1, grid.ni + 1, 2), dtype = 'int')  # (nj + 1, ni + 1, jp:ip)
    primary_pillar_jip[-1, :, 0] = 1
    primary_pillar_jip[:, -1, 1] = 1
    # build extra pillar references for split pillars
    extras_count = np.zeros((grid.nj + 1, grid.ni + 1), dtype = 'int')  # count (0 to 3) of extras for pillar
    extras_list_index = np.zeros((grid.nj + 1, grid.ni + 1), dtype = 'int')  # index in list of 1st extra for pillar
    extras_list = []  # list of (jp, ip)
    # extras_use: (j, i, jp, ip); -1 means use primary
    extras_use = np.negative(np.ones((grid.nj, grid.ni, 2, 2), dtype = 'int'))
    log.debug('building extra pillar references for split pillars')
    # loop over pillars
    for j in range(grid.nj + 1):
        for i in range(grid.ni + 1):
            primary_jp = primary_pillar_jip[j, i, 0]
            primary_ip = primary_pillar_jip[j, i, 1]
            p_col_j = j - primary_jp
            p_col_i = i - primary_ip
            # loop over 4 columns surrounding this pillar
            for jp in range(2):
                col_j = j - jp
                if col_j < 0 or col_j >= grid.nj:
                    continue  # no column this side of pillar in j
                for ip in range(2):
                    col_i = i - ip
                    if col_i < 0 or col_i >= grid.ni:
                        continue  # no column this side of pillar in i
                    if jp == primary_jp and ip == primary_ip:
                        continue  # this column is the primary for this pillar
                    discrepancy = np.max(
                        np.abs(k_reduced_cp_array[:, col_j, col_i, jp, ip, :] -
                               k_reduced_cp_array[:, p_col_j, p_col_i, primary_jp, primary_ip, :]))
                    if discrepancy <= split_tolerance:
                        continue  # data for this column's corner aligns with primary
                    for e in range(extras_count[j, i]):
                        eli = extras_list_index[j, i] + e
                        pillar_j_extra = j - extras_list[eli][0]
                        pillar_i_extra = i - extras_list[eli][1]
                        discrepancy = np.max(
                            np.abs(k_reduced_cp_array[:, col_j, col_i, jp, ip, :] -
                                   k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, extras_list[eli][0],
                                                      extras_list[eli][1], :]))
                        if discrepancy <= split_tolerance:  # data for this corner aligns with existing extra
                            extras_use[col_j, col_i, jp, ip] = e
                            break
                    if extras_use[col_j, col_i, jp, ip] >= 0:  # reusing an existing extra for this pillar
                        continue
                    # add this corner as an extra
                    if extras_count[j, i] == 0:  # create entry point for this pillar in extras
                        extras_list_index[j, i] = len(extras_list)
                    extras_list.append((jp, ip))
                    extras_use[col_j, col_i, jp, ip] = extras_count[j, i]
                    extras_count[j, i] += 1
    if len(extras_list) == 0:
        grid.has_split_coordinate_lines = False
    log.debug('number of extra pillars: ' + str(len(extras_list)))
    # create points array as used in resqml
    log.debug('creating points array as used in resqml format')
    if grid.has_split_coordinate_lines:
        points_array = np.zeros((grid.nk + 1, (grid.nj + 1) * (grid.ni + 1) + len(extras_list), 3))
        index = 0
        # primary pillars
        for pillar_j in range(grid.nj + 1):
            for pillar_i in range(grid.ni + 1):
                (jp, ip) = primary_pillar_jip[pillar_j, pillar_i]
                points_array[:, index, :] = k_reduced_cp_array[:, pillar_j - jp, pillar_i - ip, jp, ip, :]
                index += 1
        # add extras for split pillars
        for pillar_j in range(grid.nj + 1):
            for pillar_i in range(grid.ni + 1):
                for e in range(extras_count[pillar_j, pillar_i]):
                    eli = extras_list_index[pillar_j, pillar_i] + e
                    (jp, ip) = extras_list[eli]
                    pillar_j_extra = pillar_j - jp
                    pillar_i_extra = pillar_i - ip
                    points_array[:, index, :] = k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, jp, ip, :]
                    index += 1
        assert index == (grid.nj + 1) * (grid.ni + 1) + len(extras_list)
    else:  # unsplit pillars
        points_array = np.zeros((grid.nk + 1, grid.nj + 1, grid.ni + 1, 3))
        for j in range(grid.nj + 1):
            for i in range(grid.ni + 1):
                (jp, ip) = primary_pillar_jip[j, i]
                points_array[:, j, i, :] = k_reduced_cp_array[:, j - jp, i - ip, jp, ip, :]
    grid.points_cached = points_array
    # add split pillar arrays to grid object
    if grid.has_split_coordinate_lines:
        _add_split_pillars_for_extras(grid, extras_count, extras_use, extras_list)


def _add_split_pillars_for_extras(grid, extras_count, extras_use, extras_list):
    log.debug('adding split pillar arrays to grid object')
    split_pillar_indices_list = []
    cumulative_length_list = []
    cols_for_extra_pillar_list = []
    cumulative_length = 0
    for pillar_j in range(grid.nj + 1):
        for pillar_i in range(grid.ni + 1):
            for e in range(extras_count[pillar_j, pillar_i]):
                split_pillar_indices_list.append(pillar_j * (grid.ni + 1) + pillar_i)
                use_count = 0
                for jp in range(2):
                    j = pillar_j - jp
                    if j < 0 or j >= grid.nj:
                        continue
                    for ip in range(2):
                        i = pillar_i - ip
                        if i < 0 or i >= grid.ni:
                            continue
                        if extras_use[j, i, jp, ip] == e:
                            use_count += 1
                            cols_for_extra_pillar_list.append((j * grid.ni) + i)
                assert (use_count > 0)
                cumulative_length += use_count
                cumulative_length_list.append(cumulative_length)
    log.debug('number of extra pillars: ' + str(len(split_pillar_indices_list)))
    assert (len(cumulative_length_list) == len(split_pillar_indices_list))
    grid.split_pillar_indices_cached = np.array(split_pillar_indices_list, dtype = 'int')
    log.debug('number of uses of extra pillars: ' + str(len(cols_for_extra_pillar_list)))
    assert (len(cols_for_extra_pillar_list) == np.count_nonzero(extras_use + 1))
    assert (len(cols_for_extra_pillar_list) == cumulative_length)
    grid.cols_for_split_pillars = np.array(cols_for_extra_pillar_list, dtype = 'int')
    assert (len(cumulative_length_list) == len(extras_list))
    grid.cols_for_split_pillars_cl = np.array(cumulative_length_list, dtype = 'int')
    grid.split_pillars_count = len(extras_list)


def _close_vertical_voids(grid, grid_cp, z_units):
    if grid.nk > 1:
        z_gap = grid_cp[1:, :, :, 0, :, :, :] - grid_cp[:-1, :, :, 1, :, :, :]
        max_gap = np.max(np.abs(z_gap))
        log.info('maximum vertical void distance after corner point interpolation: {0:.3f} {1}'.format(
            max_gap, z_units))
        # close vertical voids (includes shifting x & y)
        if max_gap > 0.0:
            log.debug('closing vertical voids')
            z_gap *= 0.5
            grid_cp[1:, :, :, 0, :, :, :] -= z_gap
            grid_cp[:-1, :, :, 1, :, :, :] += z_gap
