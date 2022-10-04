"""functions working with defined geometry in grids"""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.xml_et as rqet


def cell_geometry_is_defined(grid, cell_kji0 = None, cell_geometry_is_defined_root = None, cache_array = True):
    """Returns True if the geometry of the specified cell is defined.

    Can also be used to cache (load) the boolean array.

    arguments:
       cell_kji0 (triplet of integer, optional): if present, the index of the cell of interest, in kji0 protocol;
          if False, None is returned but the boolean array can still be cached
       cell_geometry_is_defined_root (optional): if present, the root of the 'cell geometry is defined' xml tree for
          this grid; this optional argument is to allow for speed optimisation, to save searching for the node
       cache_array (boolean, default True): if True, the 'cell geometry is defined' array is cached in memory, unless
          the xml tree indicates that geometry is defined for all cells, in which case that is noted

    returns:
       if cell_kji0 is not None, a boolean is returned indicating whether geometry is defined for that cell;
       if cell_kji0 is None, None is returned (but the array caching logic will have been applied)
    """

    if grid.geometry_defined_for_all_cells_cached:
        return True
    if hasattr(grid, 'array_cell_geometry_is_defined') and grid.array_cell_geometry_is_defined is None:
        delattr(grid, 'array_cell_geometry_is_defined')
    if hasattr(grid, 'array_cell_geometry_is_defined'):
        grid.geometry_defined_for_all_cells_cached = np.all(grid.array_cell_geometry_is_defined)
        if grid.geometry_defined_for_all_cells_cached:
            return True
        if cell_kji0 is None:
            return False
        return grid.array_cell_geometry_is_defined[tuple(cell_kji0)]
    is_def_root = grid.resolve_geometry_child('CellGeometryIsDefined', child_node = cell_geometry_is_defined_root)
    if is_def_root is None:
        points = grid.points_ref(masked = False)
        assert points is not None
        grid.geometry_defined_for_all_cells_cached = not np.any(np.isnan(points))
        if grid.geometry_defined_for_all_cells_cached or cell_kji0 is None:
            return grid.geometry_defined_for_all_cells_cached
    is_def_type = rqet.node_type(is_def_root)
    if is_def_type == 'BooleanConstantArray':
        grid.geometry_defined_for_all_cells_cached = (rqet.find_tag_text(is_def_root, 'Value').lower() == 'true')
        return grid.geometry_defined_for_all_cells_cached
    else:
        assert (is_def_type == 'BooleanHdf5Array')
        h5_key_pair = grid.model.h5_uuid_and_path_for_node(is_def_root)
        if h5_key_pair is None:
            return None
        result = grid.model.h5_array_element(h5_key_pair,
                                             index = cell_kji0,
                                             cache_array = cache_array,
                                             object = grid,
                                             array_attribute = 'array_cell_geometry_is_defined',
                                             dtype = 'bool')
        if grid.geometry_defined_for_all_cells_cached is None and cache_array and hasattr(
                grid, 'array_cell_geometry_is_defined'):
            grid.geometry_defined_for_all_cells_cached = (np.count_nonzero(
                grid.array_cell_geometry_is_defined) == grid.array_cell_geometry_is_defined.size)
            if grid.geometry_defined_for_all_cells_cached:
                delattr(grid, 'array_cell_geometry_is_defined')
        return result


def pillar_geometry_is_defined(grid, pillar_ji0 = None, pillar_geometry_is_defined_root = None, cache_array = True):
    """Returns True if the geometry of the specified pillar is defined; False otherwise.

    Can also be used to cache (load) the boolean array.

    arguments:
       pillar_ji0 (pair of integers, optional): if present, the index of the pillar of interest, in ji0 protocol;
          if False, None is returned but the boolean array can still be cached
       pillar_geometry_is_defined_root (optional): if present, the root of the 'pillar geometry is defined' xml tree for
          this grid; this optional argument is to allow for speed optimisation, to save searching for the node
       cache_array (boolean, default True): if True, the 'pillar geometry is defined' array is cached in memory, unless
          the xml tree indicates that geometry is defined for all pillars, in which case that is noted

    returns:
       if pillar_ji0 is not None, a boolean is returned indicating whether geometry is defined for that pillar;
       if pillar_ji0 is None, None is returned unless geometry is defined for all pillars in which case True is returned
    """

    if grid.geometry_defined_for_all_pillars_cached:
        return True
    if hasattr(grid, 'array_pillar_geometry_is_defined') and grid.array_pillar_geometry_is_defined is not None:
        if pillar_ji0 is None:
            return None  # this option allows caching of array without actually referring to any pillar
        return grid.array_pillar_geometry_is_defined[tuple(pillar_ji0)]
    is_def_root = grid.resolve_geometry_child('PillarGeometryIsDefined', child_node = pillar_geometry_is_defined_root)
    if is_def_root is None:
        return True  # maybe default should be False?
    is_def_type = rqet.node_type(is_def_root)
    if is_def_type == 'BooleanConstantArray':
        assert rqet.find_tag(is_def_root, 'Value').text.lower() == 'true'
        grid.geometry_defined_for_all_pillars_cached = True
        return True
    else:
        assert is_def_type == 'BooleanHdf5Array'
        h5_key_pair = grid.model.h5_uuid_and_path_for_node(is_def_root)
        if h5_key_pair is None:
            return None
        result = grid.model.h5_array_element(h5_key_pair,
                                             index = pillar_ji0,
                                             cache_array = cache_array,
                                             object = grid,
                                             array_attribute = 'array_pillar_geometry_is_defined',
                                             dtype = 'bool')
        if grid.geometry_defined_for_all_pillars_cached is None and cache_array and hasattr(
                grid, 'array_pillar_geometry_is_defined'):
            grid.geometry_defined_for_all_pillars_cached = (np.count_nonzero(
                grid.array_pillar_geometry_is_defined) == grid.array_pillar_geometry_is_defined.size)
            if grid.geometry_defined_for_all_pillars_cached:
                del grid.array_pillar_geometry_is_defined  # memory optimisation
        return result


def geometry_defined_for_all_cells(grid, cache_array = True):
    """Returns True if geometry is defined for all cells; False otherwise.

    argument:
       cache_array (boolean, default True): if True, the 'cell geometry is defined' array is cached in memory,
          unless the xml indicates that geometry is defined for all cells, in which case that is noted

    returns:
       boolean: True if geometry is defined for all cells; False otherwise
    """

    if grid.geometry_defined_for_all_cells_cached is not None:
        return grid.geometry_defined_for_all_cells_cached
    if cache_array:
        grid.cell_geometry_is_defined(cache_array = True)
        return grid.geometry_defined_for_all_cells_cached
    # loop over all cells (until a False is encountered) – only executes if cache_array is False
    cell_geom_defined_root = grid.resolve_geometry_child('CellGeometryIsDefined')
    if cell_geom_defined_root is not None:
        for k0 in range(grid.nk):
            for j0 in range(grid.nj):
                for i0 in range(grid.ni):
                    if not grid.cell_geometry_is_defined(cell_kji0 = (k0, j0, i0),
                                                         cell_geometry_is_defined_root = cell_geom_defined_root,
                                                         cache_array = False):
                        grid.geometry_defined_for_all_cells_cached = False
                        return False
    grid.geometry_defined_for_all_cells_cached = True
    return True


def geometry_defined_for_all_pillars(grid, cache_array = True, pillar_geometry_is_defined_root = None):
    """Returns True if geometry is defined for all pillars; False otherwise.

    arguments:
       cache_array (boolean, default True): if True, the 'pillar geometry is defined' array is cached in memory,
          unless the xml indicates that geometry is defined for all pillars, in which case that is noted
       pillar_geometry_is_defined_root (optional): if present, the root of the 'pillar geometry is defined' xml tree for
          this grid; this optional argument is to allow for speed optimisation, to save searching for the node

    returns:
       boolean: True if the geometry is defined for all pillars; False otherwise
    """

    if grid.geometry_defined_for_all_pillars_cached is not None:
        return grid.geometry_defined_for_all_pillars_cached
    if cache_array:
        grid.pillar_geometry_is_defined(cache_array = cache_array)
        return grid.geometry_defined_for_all_pillars_cached
    is_def_root = grid.resolve_geometry_child('PillarGeometryIsDefined', child_node = pillar_geometry_is_defined_root)
    grid.geometry_defined_for_all_pillars_cached = True
    if is_def_root is not None:
        for pillar_j in range(grid.nj):
            for pillar_i in range(grid.ni):
                if not grid.pillar_geometry_is_defined(
                    [pillar_j, pillar_i], pillar_geometry_is_defined_root = is_def_root, cache_array = False):
                    grid.geometry_defined_for_all_pillars_cached = False
                    break
            if not grid.geometry_defined_for_all_pillars_cached:
                break
    return grid.geometry_defined_for_all_pillars_cached


def cell_geometry_is_defined_ref(grid):
    """Returns an in-memory numpy array containing the boolean data indicating which cells have geometry defined.

    returns:
       numpy array of booleans of shape (nk, nj, ni); True value indicates cell has geometry defined; False
       indicates that the cell's geometry (points xyz values) cannot be used

    note:
       if geometry is flagged in the xml as being defined for all cells, then this function returns None;
       geometry_defined_for_all_cells() can be used to test for that situation
    """

    # todo: treat this array like any other property?; handle constant array seamlessly?
    grid.cell_geometry_is_defined(cache_array = True)
    if hasattr(grid, 'array_cell_geometry_is_defined'):
        return grid.array_cell_geometry_is_defined
    return None  # can happen, if geometry is defined for all cells


def pillar_geometry_is_defined_ref(grid):
    """Returns an in-memory numpy array containing the boolean data indicating which pillars have geometry defined.

    returns:
       numpy array of booleans of shape (nj + 1, ni + 1); True value indicates pillar has geometry defined (at
       least for some points); False indicates that the pillar's geometry (points xyz values) cannot be used;
       the resulting array only covers primary pillars; extra pillars for split pillars always have geometry
       defined

    note:
       if geometry is flagged in the xml as being defined for all pillars, then this function returns None
    """

    # todo: double-check behaviour in presence of split pillars
    # todo: treat this array like any other property?; handle constant array seamlessly?
    grid.pillar_geometry_is_defined(cache_array = True)
    if hasattr(grid, 'array_pillar_geometry_is_defined'):
        return grid.array_pillar_geometry_is_defined
    return None  # can happen, if geometry is defined for all pillars


def set_geometry_is_defined(grid,
                            treat_as_nan = None,
                            treat_dots_as_nan = False,
                            complete_partial_pillars = False,
                            nullify_partial_pillars = False,
                            complete_all = False):
    """Set cached flags and/or arrays indicating which primary pillars have any points defined and which cells all points.

    arguments:
       treat_as_nan (float, optional): if present, any point with this value as x, y or z is changed
          to hold NaN values, which is the correct RESQML representation of undefined values
       treat_dots_as_nan (boolean, default False): if True, the points around any inactive cell which has zero length along
          all its I and J edges will be set to NaN (which can intentionally invalidate the geometry of neighbouring cells)
       complete_partial_pillars (boolean, default False): if True, pillars which have some but not all points defined will
          have values generated for the undefined (NaN) points
       nullify_partial_pillars (boolean, default False): if True, pillars which have some undefined (NaN) points will be
          treated as if all the points on the pillar are undefined
       complete_all (boolean, default False): if True, values will be generated for all undefined points (includes
          completion of partial pillars if both partial pillar arguments are False)

    notes:
       this method discards any previous information about which pillars and cells have geometry defined; the new settings
       are based solely on where points data is NaN (or has the value supplied as treat_as_nan etc.);
       the inactive attribute is also updated by this method, though any cells previously flagged as inactive will still be
       inactive;
       if points are generated due to either complete... argument being set True, the inactive mask is set prior to
       generating points, so all cells making use of generated points will be inactive; however, the geometry will show
       as defined where points have been generated;
       at most one of complete_partial_pillars and nullify_partial_pillars may be True;
       although the method modifies the cached (attribute) copies of various arrays, they are not written to hdf5 here
    """

    assert not (complete_partial_pillars and nullify_partial_pillars)
    if complete_all and not nullify_partial_pillars:
        complete_partial_pillars = True

    points = grid.points_ref(masked = False)

    if treat_as_nan is not None:
        nan_mask = np.any(np.logical_or(np.isnan(points), points == treat_as_nan), axis = -1)
    else:
        nan_mask = np.any(np.isnan(points), axis = -1)

    if treat_dots_as_nan:
        areal_dots = grid.point_areally()
        some_areal_dots = np.any(areal_dots)
    else:
        areal_dots = None
        some_areal_dots = False

    grid.geometry_defined_for_all_pillars_cached = None
    if hasattr(grid, 'array_pillar_geometry_is_defined'):
        del grid.array_pillar_geometry_is_defined
    grid.geometry_defined_for_all_cells_cached = None
    if hasattr(grid, 'array_cell_geometry_is_defined'):
        del grid.array_cell_geometry_is_defined

    if not np.any(nan_mask) and not some_areal_dots:
        grid.geometry_defined_for_all_pillars_cached = True
        grid.geometry_defined_for_all_cells_cached = True
        return

    if some_areal_dots:
        nan_mask = __handle_areal_dots(areal_dots, grid, nan_mask)

    assert not np.all(nan_mask), 'grid does not have any geometry defined'

    points[:] = np.where(np.repeat(np.expand_dims(nan_mask, axis = nan_mask.ndim), 3, axis = -1), np.NaN, points)

    surround_z = grid.xyz_box(lazy = False)[1 if grid.z_inc_down() else 0, 2]

    pillar_defined_mask = np.logical_not(np.all(nan_mask, axis = 0)).flatten()
    primary_count = (grid.nj + 1) * (grid.ni + 1)
    if np.all(pillar_defined_mask):
        grid.geometry_defined_for_all_pillars_cached = True
    else:
        grid.geometry_defined_for_all_pillars_cached = False
        grid.array_pillar_geometry_is_defined = pillar_defined_mask[:primary_count].reshape((grid.nj + 1, grid.ni + 1))
    if pillar_defined_mask.size > primary_count and not np.all(pillar_defined_mask[primary_count:]):
        log.warning('at least one split pillar has geometry undefined')

    grid.geometry_defined_for_all_cells_cached = False

    primary_nan_mask = \
        nan_mask.reshape((grid.nk_plus_k_gaps + 1, -1))[:, :primary_count].reshape(
            (grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1))
    column_nan_mask = np.logical_or(np.logical_or(primary_nan_mask[:, :-1, :-1], primary_nan_mask[:, :-1, 1:]),
                                    np.logical_or(primary_nan_mask[:, 1:, :-1], primary_nan_mask[:, 1:, 1:]))
    if grid.k_gaps:
        grid.array_cell_geometry_is_defined = np.logical_not(
            np.logical_or(column_nan_mask[grid.k_raw_index_array], column_nan_mask[grid.k_raw_index_array + 1]))
    else:
        grid.array_cell_geometry_is_defined = np.logical_not(np.logical_or(column_nan_mask[:-1], column_nan_mask[1:]))

    if hasattr(grid, 'inactive') and grid.inactive is not None:
        grid.inactive = np.logical_or(grid.inactive, np.logical_not(grid.array_cell_geometry_is_defined))
    else:
        grid.inactive = np.logical_not(grid.array_cell_geometry_is_defined)
    grid.all_inactive = np.all(grid.inactive)

    if grid.geometry_defined_for_all_cells_cached:
        return

    cells_update_needed = False

    if nullify_partial_pillars:
        partial_pillar_mask = np.logical_and(pillar_defined_mask, np.any(nan_mask, axis = 0).flatten())
        if np.any(partial_pillar_mask):
            points.reshape((grid.nk_plus_k_gaps + 1, -1, 3))[:, partial_pillar_mask, :] = np.NaN
            cells_update_needed = True
    elif complete_partial_pillars:
        partial_pillar_mask = np.logical_and(pillar_defined_mask, np.any(nan_mask, axis = 0).flatten())
        if np.any(partial_pillar_mask):
            log.warning('completing geometry for partially defined pillars')
            for pillar_index in np.where(partial_pillar_mask)[0]:
                __infill_partial_pillar(grid, pillar_index)
            cells_update_needed = True

    if complete_all:
        cells_update_needed = __complete_all_pillars(cells_update_needed, grid, points, surround_z)

    if cells_update_needed:
        # note: each pillar is either fully defined or fully undefined at this point
        if grid.geometry_defined_for_all_pillars_cached:
            grid.geometry_defined_for_all_cells_cached = True
            if hasattr(grid, 'array_cell_geometry_is_defined'):
                del grid.array_cell_geometry_is_defined
        else:
            top_nan_mask = np.isnan(points[0, ..., 0].flatten()[:(grid.nj + 1) * (grid.ni + 1)].reshape(
                (grid.nj + 1, grid.ni + 1)))
            column_nan_mask = np.logical_or(np.logical_or(top_nan_mask[:-1, :-1], top_nan_mask[:-1, 1:]),
                                            np.logical_or(top_nan_mask[1:, :-1], top_nan_mask[1:, 1:]))
            grid.array_cell_geometry_is_defined = np.repeat(np.expand_dims(column_nan_mask, 0), grid.nk, axis = 0)
            grid.geometry_defined_for_all_cells_cached = np.all(grid.array_cell_geometry_is_defined)
            if grid.geometry_defined_for_all_cells_cached:
                del grid.array_cell_geometry_is_defined


def __complete_all_pillars(cells_update_needed, grid, points, surround_z):
    # note: each pillar is either fully defined or fully undefined at this point
    top_nan_mask = np.isnan(points[0, ..., 0].flatten()[:(grid.nj + 1) * (grid.ni + 1)].reshape(
        (grid.nj + 1, grid.ni + 1)))
    surround_mask, coastal_mask = __create_surround_masks(top_nan_mask)
    holes_mask = np.logical_and(top_nan_mask, np.logical_not(surround_mask))
    if np.any(holes_mask):
        __fill_holes(grid, holes_mask)
    if np.any(surround_mask):
        __fill_surround(grid, surround_mask)
    # set z values for coastal and surround to max z for grid
    surround_mask = np.logical_or(surround_mask, coastal_mask).flatten()
    if np.any(surround_mask):
        points.reshape(grid.nk_plus_k_gaps + 1, -1, 3)[:, :(grid.nj + 1) * (grid.ni + 1)][:, surround_mask,
                                                                                          2] = surround_z
    grid.geometry_defined_for_all_pillars_cached = True
    if hasattr(grid, 'array_pillar_geometry_is_defined'):
        del grid.array_pillar_geometry_is_defined
    cells_update_needed = False
    assert not np.any(np.isnan(points))
    grid.geometry_defined_for_all_cells_cached = True
    if hasattr(grid, 'array_cell_geometry_is_defined'):
        del grid.array_cell_geometry_is_defined
    return cells_update_needed


def __handle_areal_dots(areal_dots, grid, nan_mask):
    # inject NaNs into the pillars around any cell that has zero length in I and J
    if grid.k_gaps:
        dot_mask = np.zeros((grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1), dtype = bool)
        dot_mask[grid.k_raw_index_array, :-1, :-1] = areal_dots
        dot_mask[grid.k_raw_index_array + 1, :-1, :-1] = np.logical_or(dot_mask[grid.k_raw_index_array + 1, :-1, :-1],
                                                                       areal_dots)
    else:
        dot_mask = np.zeros((grid.nk + 1, grid.nj + 1, grid.ni + 1), dtype = bool)
        dot_mask[:-1, :-1, :-1] = areal_dots
        dot_mask[1:, :-1, :-1] = np.logical_or(dot_mask[:-1, :-1, :-1], areal_dots)
    dot_mask[:, 1:, :-1] = np.logical_or(dot_mask[:, :-1, :-1], dot_mask[:, 1:, :-1])
    dot_mask[:, :, 1:] = np.logical_or(dot_mask[:, :, :-1], dot_mask[:, :, 1:])
    if grid.has_split_coordinate_lines:
        # only set points in primary pillars to NaN; todo: more thorough to consider split pillars too
        primaries = (grid.nj + 1) * (grid.ni + 1)
        nan_mask[:, :primaries] = np.logical_or(nan_mask[:, :primaries], dot_mask.reshape((-1, primaries)))
    else:
        nan_mask = np.where(dot_mask, np.NaN, nan_mask)
    return nan_mask


def __infill_partial_pillar(grid, pillar_index):
    points = grid.points_ref(masked = False).reshape((grid.nk_plus_k_gaps + 1, -1, 3))
    nan_mask = np.isnan(points[:, pillar_index, 0])
    first_k = 0
    while first_k < grid.nk_plus_k_gaps + 1 and nan_mask[first_k]:
        first_k += 1
    assert first_k < grid.nk_plus_k_gaps + 1
    if first_k > 0:
        points[:first_k, pillar_index] = points[first_k, pillar_index]
    last_k = grid.nk_plus_k_gaps
    while nan_mask[last_k]:
        last_k -= 1
    if last_k < grid.nk_plus_k_gaps:
        points[last_k + 1:, pillar_index] = points[last_k, pillar_index]
    while True:
        while first_k < last_k and not nan_mask[first_k]:
            first_k += 1
        if first_k >= last_k:
            break
        scan_k = first_k + 1
        while nan_mask[scan_k]:
            scan_k += 1
        points[first_k - 1: scan_k, pillar_index] = \
            np.linspace(points[first_k - 1, pillar_index], points[scan_k, pillar_index],
                        num=scan_k - first_k + 1, endpoint=False)
        first_k = scan_k


def __create_surround_masks(top_nan_mask):
    assert top_nan_mask.ndim == 2
    nj1, ni1 = top_nan_mask.shape  # nj + 1, ni + 1
    surround_mask = np.zeros(top_nan_mask.shape, dtype = bool)
    coastal_mask = np.zeros(top_nan_mask.shape, dtype = bool)
    for j in range(nj1):
        i = 0
        while i < ni1 and top_nan_mask[j, i]:
            coastal_mask[j, i] = True
            i += 1
        if i < ni1:
            i = ni1 - 1
            while top_nan_mask[j, i]:
                coastal_mask[j, i] = True
                i -= 1
        else:
            surround_mask[j] = True
            coastal_mask[j] = False
    for i in range(ni1):
        j = 0
        while j < nj1 and top_nan_mask[j, i]:
            coastal_mask[j, i] = True
            j += 1
        if j < nj1:
            j = nj1 - 1
            while top_nan_mask[j, i]:
                coastal_mask[j, i] = True
                j -= 1
        else:
            surround_mask[:, i] = True
            coastal_mask[:, i] = False
    return surround_mask, coastal_mask


def __fill_holes(grid, holes_mask):
    log.debug(f'filling {np.count_nonzero(holes_mask)} pillars for holes')
    points = grid.points_ref(masked = False).reshape(grid.nk_plus_k_gaps + 1, -1, 3)
    ni_plus_1 = grid.ni + 1
    mask_01 = np.empty(holes_mask.shape, dtype = int)
    while np.any(holes_mask):
        flat_holes_mask = holes_mask.flatten()
        mask_01[:] = np.where(holes_mask, 0, 1)
        modified = False
        # fix isolated NaN pillars with 4 neighbours
        neighbours = np.zeros(holes_mask.shape, dtype = int)
        neighbours[:-1, :] += mask_01[1:, :]
        neighbours[1:, :] += mask_01[:-1, :]
        neighbours[:, :-1] += mask_01[:, 1:]
        neighbours[:, 1:] += mask_01[:, :-1]
        foursomes = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 4))[0]
        if len(foursomes) > 0:
            interpolated = 0.25 * (points[:, foursomes - 1, :] + points[:, foursomes + 1, :] +
                                   points[:, foursomes - ni_plus_1, :] + points[:, foursomes + ni_plus_1, :])
            points[:, foursomes, :] = interpolated
            flat_holes_mask[foursomes] = False
            modified = True
        # fix NaN pillars with defined opposing neighbours in -J and +J
        neighbours[:] = 0
        neighbours[:-1, :] += mask_01[1:, :]
        neighbours[1:, :] += mask_01[:-1, :]
        twosomes = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
        if len(twosomes) > 0:
            interpolated = 0.5 * (points[:, twosomes - ni_plus_1, :] + points[:, twosomes + ni_plus_1, :])
            points[:, twosomes, :] = interpolated
            flat_holes_mask[twosomes] = False
            modified = True
        # fix NaN pillars with defined opposing neighbours in -I and +I
        neighbours[:] = 0
        neighbours[:, :-1] += mask_01[:, 1:]
        neighbours[:, 1:] += mask_01[:, :-1]
        twosomes = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
        if len(twosomes) > 0:
            interpolated = 0.5 * (points[:, twosomes - 1, :] + points[:, twosomes + 1, :])
            points[:, twosomes, :] = interpolated
            flat_holes_mask[twosomes] = False
            modified = True
        # fix NaN pillars with defined cornering neighbours in J- and I-
        neighbours[:] = 0
        neighbours[1:, :] += mask_01[:-1, :]
        neighbours[:, 1:] += mask_01[:, :-1]
        corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
        neighbours[1:, 1:] += mask_01[:-1, :-1]
        pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
        if len(corners) > 0:
            interpolated = 0.5 * (points[:, corners - ni_plus_1, :] + points[:, corners - 1, :])
            points[:, corners, :] = interpolated
            pushed = 2.0 * points[:, pushable, :] - points[:, pushable - ni_plus_1 - 1, :]
            points[:, pushable, :] = pushed
            flat_holes_mask[corners] = False
            modified = True
        # fix NaN pillars with defined cornering neighbours in J- and I+
        neighbours[:] = 0
        neighbours[1:, :] += mask_01[:-1, :]
        neighbours[:, :-1] += mask_01[:, 1:]
        corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
        neighbours[1:, :-1] += mask_01[:-1, 1:]
        pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
        if len(corners) > 0:
            interpolated = 0.5 * (points[:, corners - ni_plus_1, :] + points[:, corners + 1, :])
            points[:, corners, :] = interpolated
            pushed = 2.0 * points[:, pushable, :] - points[:, pushable - ni_plus_1 + 1, :]
            points[:, pushable, :] = pushed
            flat_holes_mask[corners] = False
            modified = True
        # fix NaN pillars with defined cornering neighbours in J+ and I-
        neighbours[:] = 0
        neighbours[:-1, :] += mask_01[1:, :]
        neighbours[:, 1:] += mask_01[:, :-1]
        corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
        neighbours[:-1, 1:] += mask_01[1:, :-1]
        pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
        if len(corners) > 0:
            interpolated = 0.5 * (points[:, corners + ni_plus_1, :] + points[:, corners - 1, :])
            points[:, corners, :] = interpolated
            pushed = 2.0 * points[:, pushable, :] - points[:, pushable + ni_plus_1 - 1, :]
            points[:, pushable, :] = pushed
            flat_holes_mask[corners] = False
            modified = True
        # fix NaN pillars with defined cornering neighbours in J+ and I+
        neighbours[:] = 0
        neighbours[:-1, :] += mask_01[1:, :]
        neighbours[:, :-1] += mask_01[:, 1:]
        corners = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 2))[0]
        neighbours[:-1, :-1] += mask_01[1:, 1:]
        pushable = np.where(np.logical_and(flat_holes_mask, neighbours.flatten() == 3))[0]
        if len(corners) > 0:
            interpolated = 0.5 * (points[:, corners + ni_plus_1, :] + points[:, corners + 1, :])
            points[:, corners, :] = interpolated
            pushed = 2.0 * points[:, pushable, :] - points[:, pushable + ni_plus_1 + 1, :]
            points[:, pushable, :] = pushed
            flat_holes_mask[corners] = False
            modified = True
        holes_mask = flat_holes_mask.reshape((grid.nj + 1, ni_plus_1))
        if not modified:
            log.warning('failed to fill all holes in grid geometry')
            break


def __fill_surround(grid, surround_mask):
    # note: only fills x,y; based on bottom layer of points; assumes surround mask is a regularly shaped frame of columns
    log.debug(f'filling {np.count_nonzero(surround_mask)} pillars for surround')
    points = grid.points_ref(masked = False)
    points_view = points[-1, :, :2].reshape((-1, 2))[:(grid.nj + 1) * (grid.ni + 1), :].reshape(
        (grid.nj + 1, grid.ni + 1, 2))
    modified = False
    if grid.nj > 1:
        j_xy_vector = np.nanmean(points_view[1:, :] - points_view[:-1, :])
        j = 0
        while j < grid.nj and np.all(surround_mask[j, :]):
            j += 1
        assert j < grid.nj
        while j > 0:
            points_view[j - 1, :] = points_view[j, :] - j_xy_vector
            modified = True
            j -= 1
        j = grid.nj - 1
        while j >= 0 and np.all(surround_mask[j, :]):
            j -= 1
        assert j >= 0
        while j < grid.nj - 1:
            points_view[j + 1, :] = points_view[j, :] + j_xy_vector
            modified = True
            j += 1
    if grid.ni > 1:
        i_xy_vector = np.nanmean(points_view[:, 1:] - points_view[:, :-1])
        i = 0
        while i < grid.ni and np.all(surround_mask[:, i]):
            i += 1
        assert i < grid.ni
        while i > 0:
            points_view[:, i - 1] = points_view[:, i] - i_xy_vector
            modified = True
            i -= 1
        i = grid.ni - 1
        while i >= 0 and np.all(surround_mask[:, i]):
            i -= 1
        assert i >= 0
        while i < grid.ni - 1:
            points_view[:, i + 1] = points_view[:, i] + i_xy_vector
            modified = True
            i += 1
    if modified:
        points.reshape((grid.nk_plus_k_gaps + 1, -1, 3))[:-1, :(grid.nj + 1) * (grid.ni + 1), :2][:,
        surround_mask.flatten(), :] = \
            points_view[surround_mask, :].reshape((1, -1, 2))
