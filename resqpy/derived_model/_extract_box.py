"""High level extract box function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.box_utilities as bx
import resqpy.olio.fine_coarse as fc
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp

import resqpy.derived_model._common as rqdm_c


def extract_box(epc_file = None,
                source_grid = None,
                box = None,
                box_inactive = None,
                inherit_properties = False,
                inherit_realization = None,
                inherit_all_realizations = False,
                set_parent_window = None,
                new_grid_title = None,
                new_epc_file = None):
    """Extends an existing model with a new grid extracted as a logical IJK box from the source grid.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       box (numpy int array of shape (2, 3)): the minimum and maximum kji0 indices in the source grid (zero based) to include
          in the extracted grid; note that cells with index equal to maximum value are included (unlike with python ranges)
       box_inactive (numpy bool array, optional): if present, shape must match box and values will be or'ed in with the
          inactive mask inherited from the source grid; if None, inactive mask will be as inherited from source grid
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid, with values taken from the specified box
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       set_parent_window (boolean, optional): if True, the extracted grid has its parent window attribute set; if False,
          the parent window is not set; if None, the default will be True if new_epc_file is None or False otherwise
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the extracted grid (& crs)

    returns:
       new grid object with extent as implied by the box argument

    note:
       the epc file and associated hdf5 file are appended to (extended) with the new grid, unless a new_epc_file is specified,
       in which case the grid and inherited properties are written there instead
    """

    log.debug('extracting grid for box')

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    if set_parent_window is None:
        set_parent_window = (new_epc_file is None)
    model, source_grid = rqdm_c._establish_model_and_source_grid(epc_file, source_grid)
    assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']
    assert model is not None
    assert box is not None and box.shape == (2, 3)
    assert np.all(box[1, :] >= box[0, :]) and np.all(box[0, :] >= 0) and np.all(box[1, :] < source_grid.extent_kji)

    if source_grid.grid_representation == 'IjkBlockGrid':
        source_grid.make_regular_points_cached()

    box_str = bx.string_iijjkk1_for_box_kji0(box)

    # create a new, empty grid object
    grid = grr.Grid(model)

    # inherit attributes from source grid
    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = box[1, :] - box[0, :] + 1
    if box_inactive is not None:
        assert box_inactive.shape == tuple(grid.extent_kji)
    grid.nk, grid.nj, grid.ni = grid.extent_kji[0], grid.extent_kji[1], grid.extent_kji[2]
    grid.k_direction_is_down = source_grid.k_direction_is_down
    grid.grid_is_right_handed = source_grid.grid_is_right_handed
    grid.pillar_shape = source_grid.pillar_shape
    grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
    # inherit the coordinate reference system used by the grid geometry
    grid.crs_uuid = source_grid.crs_uuid
    if source_grid.model is not model:
        model.duplicate_node(source_grid.model.root_for_uuid(grid.crs_uuid), add_as_part = True)
    grid.crs = rqc.Crs(model, uuid = grid.crs_uuid)

    # inherit k_gaps for selected layer range
    _inherit_k_gaps(source_grid, grid, box)

    # extract inactive cell mask
    _extract_inactive_cell_mask(source_grid, grid, box_inactive, box)

    # extract the grid geometry
    source_grid.cache_all_geometry_arrays()

    # determine cell geometry is defined
    if hasattr(source_grid, 'array_cell_geometry_is_defined'):
        grid.array_cell_geometry_is_defined = _array_box(source_grid.array_cell_geometry_is_defined, box)
        grid.geometry_defined_for_all_cells_cached = np.all(grid.array_cell_geometry_is_defined)
    else:
        grid.geometry_defined_for_all_cells_cached = source_grid.geometry_defined_for_all_cells_cached

    # copy info for pillar geometry is defined
    grid.geometry_defined_for_all_pillars_cached = source_grid.geometry_defined_for_all_pillars_cached
    if hasattr(source_grid, 'array_pillar_geometry_is_defined'):
        grid.array_pillar_geometry_is_defined = _array_box(source_grid.array_pillar_geometry_is_defined, box)
        grid.geometry_defined_for_all_pillars_cached = np.all(grid.array_pillar_geometry_is_defined)

    # get reference to points for source grid geometry
    source_points = source_grid.points_ref()

    pillar_box = box.copy()
    if source_grid.k_gaps:
        pillar_box[:, 0] = source_grid.k_raw_index_array[pillar_box[:, 0]]
    pillar_box[1, :] += 1  # pillar points have extent one greater than cells, in each axis

    if not source_grid.has_split_coordinate_lines:
        log.debug('no split pillars in source grid')
        grid.points_cached = _array_box(source_points, pillar_box)  # should work, ie. preserve xyz axis
    else:
        _process_split_pillars(source_grid, grid, box, pillar_box)

    if set_parent_window:
        fine_coarse = fc.FineCoarse(grid.extent_kji, grid.extent_kji, within_coarse_box = box)
        fine_coarse.set_all_ratios_constant()
        grid.set_parent(source_grid.uuid, True, fine_coarse)

    collection = _inherit_collection(source_grid, grid, inherit_properties, box, inherit_realization,
                                     inherit_all_realizations)

    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'local grid ' + box_str + ' extracted from ' + str(
            rqet.citation_title_for_node(source_grid.root))

    model.h5_release()
    if new_epc_file:
        rqdm_c._write_grid(new_epc_file,
                           grid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'w')
    else:
        ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                      'Coordinates')
        rqdm_c._write_grid(epc_file,
                           grid,
                           ext_uuid = ext_uuid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'a')

    return grid


def _array_box(a, box):
    return a[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1, box[0, 2]:box[1, 2] + 1].copy()


def _local_col_index(extent, box, col):
    # return local equivalent natural column index for global column index, or None if outside box
    j, i = divmod(col, extent[2])
    j -= box[0, 1]
    i -= box[0, 2]
    if j < 0 or i < 0 or j > box[1, 1] - box[0, 1] or i > box[1, 2] - box[0, 2]:
        return None
    return j * (box[1, 2] - box[0, 2] + 1) + i


def _local_pillar_index(extent, box, p):
    # return local equivalent natural pillar index for global pillar index, or None if outside box
    p_j, p_i = divmod(p, extent[2] + 1)
    p_j -= box[0, 1]
    p_i -= box[0, 2]
    if p_j < 0 or p_i < 0 or p_j > box[1, 1] - box[0, 1] + 1 or p_i > box[1, 2] - box[0, 2] + 1:
        return None
    return p_j * (box[1, 2] - box[0, 2] + 2) + p_i


def _cols_for_pillar(extent, p):
    # return 4 naturalized column indices for columns surrounding natural pillar index; -1 where beyond edge of ij space
    cols = np.zeros((4,), dtype = int) - 1
    p_j, p_i = divmod(p, extent[2] + 1)
    if p_j > 0 and p_i > 0:
        cols[0] = (p_j - 1) * extent[2] + p_i - 1
    if p_j > 0 and p_i < extent[2]:
        cols[1] = (p_j - 1) * extent[2] + p_i
    if p_j < extent[1] and p_i > 0:
        cols[2] = p_j * extent[2] + p_i - 1
    if p_j < extent[1] and p_i < extent[2]:
        cols[3] = p_j * extent[2] + p_i
    return cols


def _inherit_k_gaps(source_grid, grid, box):
    if source_grid.k_gaps and box[1, 0] > box[0, 0]:
        k_gaps = np.count_nonzero(source_grid.k_gap_after_array[box[0, 0]:box[1, 0]])
        if k_gaps > 0:
            grid.k_gaps = k_gaps
            grid.k_gap_after_array = source_grid.k_gap_after_array[box[0, 0]:box[1, 0]].copy()
            grid.k_raw_index_array = np.empty(grid.nk, dtype = int)
            k_offset = 0
            for k in range(grid.nk):
                grid.k_raw_index_array[k] = k + k_offset
                if k < grid.nk - 1 and grid.k_gap_after_array[k]:
                    k_offset += 1
            assert k_offset == k_gaps


def _extract_inactive_cell_mask(source_grid, grid, box_inactive, box):
    if source_grid.inactive is None:
        if box_inactive is None:
            log.debug('setting inactive mask to None')
            grid.inactive = None
        else:
            log.debug('setting inactive mask to that passed as argument')
            grid.inactive = box_inactive.copy()
    else:
        if box_inactive is None:
            log.debug('extrating inactive mask')
            grid.inactive = _array_box(source_grid.inactive, box)
        else:
            log.debug('setting inactive mask to merge of source grid extraction and mask passed as argument')
            grid.inactive = np.logical_or(_array_box(source_grid.inactive, box), box_inactive)


def _inherit_collection(source_grid, grid, inherit_properties, box, inherit_realization, inherit_all_realizations):
    collection = None
    if inherit_properties:
        source_collection = source_grid.extract_property_collection()
        if source_collection is not None:
            # do not inherit the inactive property array by this mechanism
            active_collection = rqp.selective_version_of_collection(source_collection, property_kind = 'active')
            source_collection.remove_parts_list_from_dict(active_collection.parts())
            inactive_collection = rqp.selective_version_of_collection(
                source_collection,
                property_kind = 'code',  # for backward compatibility
                facet_type = 'what',
                facet = 'inactive')
            source_collection.remove_parts_list_from_dict(inactive_collection.parts())
            collection = rqp.GridPropertyCollection()
            collection.set_grid(grid)
            collection.extend_imported_list_copying_properties_from_other_grid_collection(
                source_collection,
                box = box,
                realization = inherit_realization,
                copy_all_realizations = inherit_all_realizations)
    return collection


def _process_split_pillars(source_grid, grid, box, pillar_box):
    source_points = source_grid.points_ref()
    source_base_pillar_count = (source_grid.nj + 1) * (source_grid.ni + 1)
    log.debug('number of base pillars in source grid: ' + str(source_base_pillar_count))
    log.debug('number of extra pillars in source grid: ' + str(len(source_grid.split_pillar_indices_cached)))
    base_points = _array_box(
        source_points[:, :source_base_pillar_count, :].reshape(
            (source_grid.nk_plus_k_gaps + 1, source_grid.nj + 1, source_grid.ni + 1, 3)),
        pillar_box).reshape(grid.nk_plus_k_gaps + 1, (grid.nj + 1) * (grid.ni + 1), 3)
    extra_points = np.zeros(
        (pillar_box[1, 0] - pillar_box[0, 0] + 1, source_points.shape[1] - source_base_pillar_count, 3))
    spi_array = np.zeros(len(source_grid.split_pillar_indices_cached), dtype = int)
    local_cols_array = np.zeros(len(source_grid.cols_for_split_pillars), dtype = int)
    local_cols_cl = np.zeros(len(source_grid.split_pillar_indices_cached), dtype = int)
    local_index = 0
    for index in range(len(source_grid.split_pillar_indices_cached)):
        source_pi = source_grid.split_pillar_indices_cached[index]
        local_pi = _local_pillar_index(source_grid.extent_kji, box, source_pi)
        if local_pi is None:
            continue
        cols = _cols_for_pillar(source_grid.extent_kji, source_pi)
        local_cols = _cols_for_pillar(grid.extent_kji, local_pi)
        if index == 0:
            start = 0
        else:
            start = source_grid.cols_for_split_pillars_cl[index - 1]
        finish = source_grid.cols_for_split_pillars_cl[index]
        source_pis = np.zeros((4,), dtype = int)
        for c_i in range(4):
            if local_cols[c_i] < 0:
                source_pis[c_i] = -1
                continue
            if cols[c_i] in source_grid.cols_for_split_pillars[start:finish]:
                source_pis[c_i] = source_base_pillar_count + index
            else:
                source_pis[c_i] = source_pi
        unique_source_pis = np.unique(source_pis)
        unique_count = len(unique_source_pis)
        unique_index = 0
        if unique_source_pis[0] == -1:
            unique_index = 1
            unique_count -= 1
        if unique_count <= 0:
            continue
        base_points[:, local_pi, :] = source_points[pillar_box[0, 0]:pillar_box[1, 0] + 1,
                                                    unique_source_pis[unique_index], :]
        unique_index += 1
        unique_count -= 1
        if unique_count <= 0:
            continue
        while unique_count > 0:
            source_pi = unique_source_pis[unique_index]
            extra_points[:, local_index, :] = source_points[pillar_box[0, 0]:pillar_box[1, 0] + 1, source_pi, :]
            spi_array[local_index] = local_pi
            if local_index == 0:
                lc_index = 0
            else:
                lc_index = local_cols_cl[local_index - 1]
            for c_i in range(4):
                if source_pis[c_i] == source_pi and local_cols[c_i] >= 0:
                    local_cols_array[lc_index] = local_cols[c_i]
                    lc_index += 1
            local_cols_cl[local_index] = lc_index
            local_index += 1
            unique_index += 1
            unique_count -= 1
    if local_index == 0:  # there are no split pillars in the box
        log.debug('box does not inherit any split pillars')
        grid.points_cached = base_points.reshape(grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1, 3)
        grid.has_split_coordinate_lines = False
    else:
        log.debug('number of extra pillars in box: ' + str(local_index))
        grid.points_cached = np.concatenate((base_points, extra_points[:, :local_index, :]), axis = 1)
        grid.split_pillar_indices_cached = spi_array[:local_index].copy()
        grid.cols_for_split_pillars = local_cols_array[:local_cols_cl[local_index - 1]].copy()
        grid.cols_for_split_pillars_cl = local_cols_cl[:local_index].copy()
        grid.split_pillars_count = local_index
