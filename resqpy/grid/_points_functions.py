"""Submodule containing the functions relating to grid points."""

import logging

log = logging.getLogger(__name__)

import numpy as np
import numpy.ma as ma

import resqpy.grid as grr
import resqpy.olio.point_inclusion as pip
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
import resqpy.weights_and_measures as wam
import resqpy.grid._defined_geometry as grr_dg


def set_cached_points_from_property(grid,
                                    points_property_uuid = None,
                                    property_collection = None,
                                    realization = None,
                                    time_index = None,
                                    set_grid_time_index = True,
                                    set_inactive = True,
                                    active_property_uuid = None,
                                    active_collection = None):
    """Modifies the cached points (geometry), setting the values from a points property.

    arguments:
       points_property_uuid (uuid, optional): the uuid of the points property; if present the
          remaining arguments are ignored except for inactive & active arguments
       property_collection (PropertyCollection, optional): defaults to property collection
          for the grid; should only contain one set of points properties (but may also contain
          other non-points properties)
       realization (int, optional): if present, the property in the collection with this
          realization number is used
       time_index (int, optional): if present, the property in the collection with this
          time index is used
       set_grid_time_index (bool, default True): if True, the grid's time index will be set
          to the time_index argument and the grid's time series uuid will be set to that
          referred to by the points property; if False, the grid's time index will not be
          modified
       set_inactive (bool, default True): if True, the grid's inactive mask will be set
          based on an active cell property
       active_property_uuid (uuid, optional): if present, the uuid of an active cell property
          to base the inactive mask on; ignored if set_inactive is False
       active_collection (uuid, optional): default's to property_collection if present, or
          the grid's property collection otherwise; only used if set_inactive is True and
          active_property_uuid is None

    notes:
       the points property must have indexable element 'nodes' and be the same shape as the
       official points array for the grid;
       note that the shape of the points array is quite different between grids with split
       pillars and those without;
       the points property values must be in the grid's crs;
       the inactive mask of the grid will only be updated if the set_inactive argument is True;
       if points_property_uuid has been provided, and set_inactive is True, the active property
       must be identified with the active_property_uuid argument;
       if set_inactive is True and active_property_uuid is None and points_property_uuid is None and
       realization and/or time_index is in use, the active property collection must contain one
       series of active properties with the same variants (realizations and time indices) as the
       points property series;
       the active cell properties should be discrete and have a local property kind titled 'active';
       various cached data are invalidated and cleared by this method
    """

    if points_property_uuid is None:
        if property_collection is None:
            property_collection = grid.extract_property_collection()
        part = property_collection.singleton(points = True,
                                             indexable = 'nodes',
                                             realization = realization,
                                             time_index = time_index)
        assert part is not None, 'failed to identify points property to use for grid geometry'
        points_property_uuid = property_collection.uuid_for_part(part)
    elif set_inactive:
        assert active_property_uuid is not None, 'active property uuid not provided when setting points from property'

    assert points_property_uuid is not None

    grid.cache_all_geometry_arrays()  # the split pillar information must not vary

    # check for compatibility and overwrite cached points for grid
    points = rprop.Property(grid.model, uuid = points_property_uuid)
    assert points is not None and points.is_points() and points.indexable_element() == 'nodes'
    points_array = points.array_ref(masked = False)
    assert points_array is not None
    assert points_array.shape == grid.points_cached.shape

    # todo: handle optional points crs and convert to grid crs
    grid.points_cached = points_array

    # set grid's time index and series, if requested
    if set_grid_time_index:
        grid.time_index = points.time_index()
        ts_uuid = points.time_series_uuid()
        if ts_uuid is not None and grid.time_series_uuid is not None:
            if not bu.matching_uuids(ts_uuid, grid.time_series_uuid):
                log.warning('change of time series uuid for dynamic grid geometry')
        grid.time_series_uuid = ts_uuid

    # invalidate anything cached that is derived from geometry
    grid.geometry_defined_for_all_pillars_cached = None
    grid.geometry_defined_for_all_cells_cached = None
    for attr in ('array_unsplit_points', 'array_corner_points', 'array_centre_point', 'array_thickness', 'array_volume',
                 'array_half_cell_t', 'array_k_transmissibility', 'array_j_transmissibility',
                 'array_i_transmissibility', 'fgcs', 'array_fgcs_transmissibility', 'pgcs',
                 'array_pgcs_transmissibility', 'kgcs', 'array_kgcs_transmissibility'):
        if hasattr(grid, attr):
            delattr(grid, attr)

    if set_inactive:
        if active_property_uuid is None:
            if active_collection is None:
                active_collection = property_collection
                assert active_collection is not None
            active_part = active_collection.singleton(property_kind = 'active',
                                                      indexable = 'cells',
                                                      continuous = False,
                                                      realization = realization,
                                                      time_index = time_index)
            assert active_part is not None, 'failed to identify active property to use for grid inactive mask'
            active_property_uuid = active_collection.uuid_for_part(active_part)
        active = rprop.Property(grid.model, uuid = active_property_uuid)
        assert active is not None
        active_array = active.array_ref()
        assert active_array.shape == tuple(grid.extent_kji)
        grid.inactive = np.logical_not(active_array)
        grid.all_inactive = np.all(grid.inactive)
        grid.active_property_uuid = active_property_uuid


def point_raw(grid, index = None, points_root = None, cache_array = True):
    """Returns element from points data, indexed as in the hdf5 file; can optionally be used to cache points data.

    arguments:
       index (2 or 3 integers, optional): if not None, the index into the raw points data for the point of interest
       points_root (optional): the xml node holding the points data
       cache_array (boolean, default True): if True, the raw points data is cached in memory as a side effect

    returns:
       (x, y, z) of selected point as a 3 element numpy vector, or None if index is None

    notes:
       this function is typically called either to cache the points data in memory, or to fetch the coordinates of
       a single point; the details of the indexing depend upon whether the grid has split coordinate lines: if not,
       the index should be a triple kji0 with axes ranging over the shared corners nk+k_gaps+1, nj+1, ni+1; if there
       are split pillars, index should be a pair, being the k0 in range nk+k_gaps+1 and a pillar index; note that if
       index is passed, the k0 element must already have been mapped to the raw index value taking into consideration
       any k gaps; if the grid object does not include geometry then None is returned
    """

    # NB: shape of index depends on whether grid has split pillars
    if index is not None and not grr_dg.geometry_defined_for_all_pillars(grid, cache_array = cache_array):
        if len(index) == 3:
            ji = tuple(index[1:])
        else:
            ji = tuple(divmod(index[1], grid.ni))
        if ji[0] < grid.nj and not grr_dg.pillar_geometry_is_defined(grid, ji, cache_array = cache_array):
            return None
    if grid.points_cached is not None:
        if index is None:
            return grid.points_cached
        return grid.points_cached[tuple(index)]
    p_root = grid.resolve_geometry_child('Points', child_node = points_root)
    if p_root is None:
        log.debug('point_raw() returning None as geometry not present')
        return None  # geometry not present
    assert rqet.node_type(p_root) == 'Point3dHdf5Array'
    h5_key_pair = grid.model.h5_uuid_and_path_for_node(p_root, tag = 'Coordinates')
    if h5_key_pair is None:
        return None
    if grid.has_split_coordinate_lines:
        required_shape = None
    else:
        required_shape = (grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1, 3)
    try:
        value = grid.model.h5_array_element(h5_key_pair,
                                            index = index,
                                            cache_array = cache_array,
                                            object = grid,
                                            array_attribute = 'points_cached',
                                            required_shape = required_shape)
    except Exception:
        log.error('hdf5 points failure for index: ' + str(index))
        raise
    if index is None:
        return grid.points_cached
    return value


def point(grid, cell_kji0 = None, corner_index = np.zeros(3, dtype = 'int'), points_root = None, cache_array = True):
    """Return a cell corner point xyz; can optionally be used to cache points data.

    arguments:
       cell_kji0 (3 integers, optional): if not None, the index of the cell for the point of interest, in kji0 protocol
       corner_index (3 integers, default zeros): the kp, jp, ip corner-within-cell indices (each 0 or 1)
       points_root (optional): the xml node holding the points data
       cache_array (boolean, default True): if True, the raw points data is cached in memory as a side effect

    returns:
       (x, y, z) of selected point as a 3 element numpy vector, or None if cell_kji0 is None

    note:
       if cell_kji0 is passed, the k0 value should be the layer index before adjustment for k_gaps, which this
       method will apply
    """

    if cache_array and grid.points_cached is None:
        grid.point_raw(points_root = points_root, cache_array = True)
    if cell_kji0 is None:
        return None
    if grid.k_raw_index_array is None:
        grid.extract_k_gaps()
    if not grr_dg.geometry_defined_for_all_cells(grid):
        if not grr_dg.cell_geometry_is_defined(grid, cell_kji0 = cell_kji0, cache_array = cache_array):
            return None
    p_root = grid.resolve_geometry_child('Points', child_node = points_root)
    #      if p_root is None: return None  # geometry not present
    index = np.zeros(3, dtype = int)
    index[:] = cell_kji0
    index[0] = grid.k_raw_index_array[index[0]]  # adjust for k gaps
    if grid.has_split_coordinate_lines:
        grid.create_column_pillar_mapping()
        pillar_index = grid.pillars_for_column[index[1], index[2], corner_index[1], corner_index[2]]
        return grid.point_raw(index = (index[0] + corner_index[0], pillar_index),
                              points_root = p_root,
                              cache_array = cache_array)
    else:
        index[:] += corner_index
        return grid.point_raw(index = index, points_root = p_root, cache_array = cache_array)


def points_ref(grid, masked = True):
    """Returns an in-memory numpy array containing the xyz data for points used in the grid geometry.

    argument:
       masked (boolean, default True): if True, a masked array is returned with NaN points masked out;
          if False, a simple (unmasked) numpy array is returned

    returns:
       numpy array or masked array of float, of shape (nk + k_gaps + 1, nj + 1, ni + 1, 3) or (nk + k_gaps + 1, np, 3)
       where np is the total number of pillars (primary pillars + extras for split pillars)

    notes:
       this is the usual way to get at the actual grid geometry points data in the native resqml layout;
       the has_split_coordinate_lines boolean attribute can be used to determine which shape to expect;
       the shape is (nk + k_gaps + 1, nj + 1, ni + 1, 3) if there are no split coordinate lines (unfaulted);
       otherwise it is (nk + k_gaps + 1, np, 3), where np > (nj + 1) * (ni + 1), due to extra pillar data for
       the split pillars

    :meta common:
    """

    if grid.points_cached is None:
        grid.point_raw(cache_array = True)
        if grid.points_cached is None:
            return None
    if not masked:
        return grid.points_cached
    return ma.masked_invalid(grid.points_cached)


def uncache_points(grid):
    """Frees up memory by removing the cached copy of the grid's points data.

    note:
       the memory will only actually become free when any other references to it pass out of scope
       or are deleted
    """

    if grid.points_cached is not None:
        del grid.points_cached
        grid.points_cached = None


def unsplit_points_ref(grid, cache_array = False, masked = False):
    """Returns a copy of the points array that has split pillars merged back into an unsplit configuration.

    arguments:
       cache_array (boolean, default False): if True, a copy of the unsplit points array is added as
          attribute array_unsplit_points to this grid object
       masked (boolean, default False): if True, a masked array is returned with NaN points masked out;
          if False, a simple (unmasked) numpy array is returned

    returns:
       numpy array of float of shape (nk + k_gaps + 1, nj + 1, ni + 1, 3)

    note:
       for grids without split pillars, this function simply returns the points array in its native form;
       for grids with split pillars, an unsplit equivalent points array is calculated as the average of
       contributions to each pillar from the surrounding cell columns
    """

    if hasattr(grid, 'array_unsplit_points'):
        return grid.array_unsplit_points
    points = grid.points_ref(masked = masked)
    if not grid.has_split_coordinate_lines:
        if cache_array:
            grid.array_unsplit_points = points.copy()
            return grid.array_unsplit_points
        return points
    # todo: finish version that copies primaries and only modifies split pillars?
    # njkp1 = (grid.nj + 1) * (grid.ni + 1)
    # merged_points = np.empty((grid.nk + 1, njkp1, 3))   # shaped somewhat like split points array
    # merged_points[:, :, :] = points[:, :njkp1, :]       # copy primary data
    result = np.empty((grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1, 3))
    # todo: if not geometry defined for all cells, take nanmean of four points?
    # compute for internal pillars
    grid.create_column_pillar_mapping()  # pillar indices for 4 columns around interior pillars
    pfc_11 = grid.pillars_for_column[:-1, :-1, 1, 1]
    pfc_10 = grid.pillars_for_column[:-1, 1:, 1, 0]
    pfc_01 = grid.pillars_for_column[1:, :-1, 0, 1]
    pfc_00 = grid.pillars_for_column[1:, 1:, 0, 0]
    result[:, 1:-1,
           1:-1, :] = 0.25 * (points[:, pfc_11, :] + points[:, pfc_10, :] + points[:, pfc_01, :] + points[:, pfc_00, :])
    # edges
    # todo: use numpy array operations instead of for loops (see lines above for example code)
    for j in range(1, grid.nj):
        result[:, j, 0, :] = 0.5 * (points[:, grid.pillars_for_column[j - 1, 0, 1, 0], :] +
                                    points[:, grid.pillars_for_column[j, 0, 0, 0], :])
        result[:, j, grid.ni, :] = 0.5 * (points[:, grid.pillars_for_column[j - 1, grid.ni - 1, 1, 1], :] +
                                          points[:, grid.pillars_for_column[j, grid.ni - 1, 0, 1], :])
    for i in range(1, grid.ni):
        result[:, 0, i, :] = 0.5 * (points[:, grid.pillars_for_column[0, i - 1, 0, 1], :] +
                                    points[:, grid.pillars_for_column[0, i, 0, 0], :])
        result[:, grid.nj, i, :] = 0.5 * (points[:, grid.pillars_for_column[grid.nj - 1, i - 1, 1, 1], :] +
                                          points[:, grid.pillars_for_column[grid.nj - 1, i, 1, 0], :])
    # corners (could optimise as these should always be primaries
    result[:, 0, 0, :] = points[:, grid.pillars_for_column[0, 0, 0, 0], :]
    result[:, 0, grid.ni, :] = points[:, grid.pillars_for_column[0, grid.ni - 1, 0, 1], :]
    result[:, grid.nj, 0, :] = points[:, grid.pillars_for_column[grid.nj - 1, 0, 1, 0], :]
    result[:, grid.nj, grid.ni, :] = points[:, grid.pillars_for_column[grid.nj - 1, grid.ni - 1, 1, 1], :]
    if cache_array:
        grid.array_unsplit_points = result
        return grid.array_unsplit_points
    return result


def horizon_points(grid, ref_k0 = 0, heal_faults = False, kp = 0):
    """Returns reference to a points layer array of shape ((nj + 1), (ni + 1), 3) based on primary pillars.

    arguments:
       ref_k0 (integer): the horizon layer number, in the range 0 to nk (or layer number in range 0..nk-1
          in the case of grids with k gaps)
       heal_faults (boolean, default False): if True and the grid has split coordinate lines, an unsplit
          equivalent of the grid points is generated first and the returned points are based on that data;
          otherwise, the primary pillar data is used, which effectively gives a point from one side or
          another of any faults, rather than an averaged point
       kp (integer, default 0): set to 1 to specify the base of layer ref_k0, in case of grids with k gaps

    returns:
       a numpy array of floats of shape ((nj + 1), (ni + 1), 3) being the (shared) cell corner point
       locations for the plane of points, based on the primary pillars or unsplit equivalent pillars

    notes:
       the primary pillars are the 'first' set of points for a pillar; a split pillar will have one to
       three other sets of point data but those are ignored by this function unless heal_faults is True,
       in which case an averaged point will be used for the split pillars;
       to get full unhealed representation of split horizon points, use split_horizon_points() function
       instead;
       for grids without k gaps, ref_k0 can be used alone, in the range 0..nk, to identify the horizon;
       alternatively, or for grids with k gaps, the ref_k0 can specify the layer in the range 0..nk-1,
       with kp being passed the value 0 (default) for the top of the layer, or 1 for the base of the layer
    """

    # note: if heal_faults is False, primary pillars only are used
    pe_j = grid.nj + 1
    pe_i = grid.ni + 1
    if grid.k_gaps:
        ref_k0 = grid.k_raw_index_array[ref_k0]
    ref_k0 += kp
    if grid.has_split_coordinate_lines:
        if heal_faults:
            points = grid.unsplit_points_ref()  # expensive operation: would be better to cache the unsplit points
            return points[ref_k0, :, :, :].reshape((pe_j, pe_i, 3))
        else:
            points = grid.points_ref(masked = False)
            return points[ref_k0, :pe_j * pe_i, :].reshape((pe_j, pe_i, 3))
    # unfaulted grid
    points = grid.points_ref(masked = False)
    return points[ref_k0, :, :, :].reshape((pe_j, pe_i, 3))


def split_horizon_points(grid, ref_k0 = 0, masked = False, kp = 0):
    """Returns reference to a corner points for a horizon, of shape (nj, ni, 2, 2, 3).

    arguments:
       ref_k0 (integer): the horizon layer number, in the range 0 to nk (or layer number in range 0..nk-1
          in the case of grids with k gaps)
       masked (boolean, default False): if True, a masked array is returned with NaN points masked out;
          if False, a simple (unmasked) numpy array is returned
       kp (integer, default 0): set to 1 to specify the base of layer ref_k0, in case of grids with k gaps

    returns:
       numpy array of shape (nj, ni, 2, 2, 3) being corner point x,y,z values for cell corners (j, i, jp, ip)

    notes:
       if split points are needed for a range of horizons, it is more efficient to call split_horizons_points()
       than repeatedly call this function;
       for grids without k gaps, ref_k0 can be used alone, in the range 0..nk, to identify the horizon;
       alternatively, or for grids with k gaps, the ref_k0 can specify the layer in the range 0..nk-1,
       with kp being passed the value 0 (default) for the top of the layer, or 1 for the base of the layer
    """

    if grid.k_gaps:
        ref_k0 = grid.k_raw_index_array[ref_k0]
    ref_k0 += kp
    points = grid.points_ref(masked = masked)
    hp = np.empty((grid.nj, grid.ni, 2, 2, 3))
    if grid.has_split_coordinate_lines:
        assert points.ndim == 3
        # todo: replace for loops with numpy slice indexing
        grid.create_column_pillar_mapping()
        assert grid.pillars_for_column.ndim == 4 and grid.pillars_for_column.shape == (grid.nj, grid.ni, 2, 2)
        for j in range(grid.nj):
            for i in range(grid.ni):
                hp[j, i, 0, 0, :] = points[ref_k0, grid.pillars_for_column[j, i, 0, 0], :]
                hp[j, i, 1, 0, :] = points[ref_k0, grid.pillars_for_column[j, i, 1, 0], :]
                hp[j, i, 0, 1, :] = points[ref_k0, grid.pillars_for_column[j, i, 0, 1], :]
                hp[j, i, 1, 1, :] = points[ref_k0, grid.pillars_for_column[j, i, 1, 1], :]
    else:
        assert points.ndim == 4
        hp[:, :, 0, 0, :] = points[ref_k0, :-1, :-1, :]
        hp[:, :, 1, 0, :] = points[ref_k0, 1:, :-1, :]
        hp[:, :, 0, 1, :] = points[ref_k0, :-1, 1:, :]
        hp[:, :, 1, 1, :] = points[ref_k0, 1:, 1:, :]
    return hp


def split_x_section_points(grid, axis, ref_slice0 = 0, plus_face = False, masked = False):
    """Returns an array of points representing cell corners from an I or J interface slice for a faulted grid.

    arguments:
       axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
       ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
       plus_face (boolean, default False): if False, negative face is used; if True, positive
       masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out

    returns:
       a numpy array of shape (nk + 1, nj, 2, 3) or (nk + 1, ni, 2, 3) being the xyz points of the cell corners
       on the interfacial cross section; 3rd axis is jp or ip; final axis is xyz

    note:
       this function will only work for grids with no k gaps; it is intended for split grids though will also
       function for unsplit grids; use split_gap_x_section_points() if k gaps are present
    """

    log.debug(f'x-sect: axis {axis}; ref_slice0 {ref_slice0}; plus_face {plus_face}; masked {masked}')
    assert axis.upper() in ['I', 'J']
    assert not grid.k_gaps, 'split_x_section_points() method is for grids without k gaps; use split_gap_x_section_points()'

    points = grid.points_ref(masked = masked)
    cpm = grid.create_column_pillar_mapping()

    ij_p = 1 if plus_face else 0

    if axis.upper() == 'I':
        return points[:, cpm[:, ref_slice0, :, ij_p], :]
    else:
        return points[:, cpm[ref_slice0, :, ij_p, :], :]


def split_gap_x_section_points(grid, axis, ref_slice0 = 0, plus_face = False, masked = False):
    """Return array of points representing cell corners from an I or J interface slice for a faulted grid.

    arguments:
       axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
       ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
       plus_face (boolean, default False): if False, negative face is used; if True, positive
       masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out

    returns:
       a numpy array of shape (nk, nj, 2, 2, 3) or (nk, ni, 2, 2, 3) being the xyz points of the cell corners
       on the interfacial cross section; 3rd axis is kp; 4th axis is jp or ip; final axis is xyz

    note:
       this function is intended for split grids with k gaps though will also function for split grids
       without k gaps
    """

    log.debug(f'k gap x-sect: axis {axis}; ref_slice0 {ref_slice0}; plus_face {plus_face}; masked {masked}')
    assert axis.upper() in ['I', 'J']

    if grid.has_split_coordinate_lines:
        points = grid.points_ref(masked = masked)
    else:
        points = grid.points_ref(masked = masked).reshape((grid.nk_plus_k_gaps, (grid.nj + 1) * (grid.ni + 1), 3))
    cpm = grid.create_column_pillar_mapping()

    ij_p = 1 if plus_face else 0

    if grid.k_gaps:
        top_points = points[grid.k_raw_index_array]
        base_points = points[grid.k_raw_index_array + 1]
        if axis.upper() == 'I':
            top = top_points[:, cpm[:, ref_slice0, :, ij_p], :]
            base = base_points[:, cpm[:, ref_slice0, :, ij_p], :]
        else:
            top = top_points[:, cpm[ref_slice0, :, ij_p, :], :]
            base = base_points[:, cpm[ref_slice0, :, ij_p, :], :]
    else:
        p = grid.split_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)
        top = p[:-1]
        base = p[1:]
    return np.stack((top, base), axis = 2)


def unsplit_x_section_points(grid, axis, ref_slice0 = 0, plus_face = False, masked = False):
    """Returns a 2D (+1 for xyz) array of points representing cell corners from an I or J interface slice.

    arguments:
       axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
       ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
       plus_face (boolean, default False): if False, negative face is used; if True, positive
       masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out

    returns:
       a 2+1D numpy array being the xyz points of the cell corners on the interfacial cross section;
       the 2D axes are K,J or K,I - whichever does not involve axis; shape is (nk + 1, nj + 1, 3) or
       (nk + 1, ni + 1, 3)

    note:
       restricted to unsplit grids with no k gaps; use split_x_section_points() for split grids with no k gaps
       or split_gap_x_section_points() for split grids with k gaps or x_section_corner_points() for any grid
    """

    log.debug(f'x-sect: axis {axis}; ref_slice0 {ref_slice0}; plus_face {plus_face}; masked {masked}')
    assert axis.upper() in ['I', 'J']
    assert not grid.has_split_coordinate_lines, 'cross sectional points for unsplit grids require split_x_section_points()'
    assert not grid.k_gaps, 'cross sectional points with k gaps require split_gap_x_section_points()'

    if plus_face:
        ref_slice0 += 1

    points = grid.points_ref(masked = masked)

    if axis.upper() == 'I':
        return points[:, :, ref_slice0, :]
    else:
        return points[:, ref_slice0, :, :]


def x_section_corner_points(grid,
                            axis,
                            ref_slice0 = 0,
                            plus_face = False,
                            masked = False,
                            rotate = False,
                            azimuth = None):
    """Returns a fully expanded array of points representing cell corners from an I or J interface slice.

    arguments:
       axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
       ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
       plus_face (boolean, default False): if False, negative face is used; if True, positive
       masked (boolean, default False): if True, a masked numpy array is returned with NaN values masked out
       rotate (boolean, default False): if True, the cross section points are rotated around the z axis so that
          an azimuthal direction is mapped onto the positive x axis
       aximuth (float, optional): the compass bearing in degrees to map onto the positive x axis if rotating;
          if None, the mean direction of the cross sectional points, along axis, is used; ignored if rotate
          is False

    returns:
       a numpy float array of shape (nk, nj, 2, 2, 3) or (nk, ni, 2, 2, 3) being the xyz points of the cell
       corners on the interfacial cross section; the 3rd index (1st 2) is kp, the 4th index is jp or ip

    note:
       this method will work for unsplit or split grids, with or without k gaps; use rotate argument to yield
       points with predominant variation in xz, suitable for plotting cross sections; if rotate is True then
       the absolute values of x & y will not be very meaningful though the units will still be the grid's xy
       units for relative purposes
    """

    assert axis.upper() in ['I', 'J']
    nj_or_ni = grid.nj if axis.upper() == 'I' else grid.ni

    if grid.k_gaps:
        x_sect = grid.split_gap_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face, masked = masked)
    else:
        if grid.has_split_coordinate_lines:
            no_k_gap_xs = grid.split_x_section_points(axis,
                                                      ref_slice0 = ref_slice0,
                                                      plus_face = plus_face,
                                                      masked = masked)
            x_sect = np.empty((grid.nk, nj_or_ni, 2, 2, 3))
            x_sect[:, :, 0, :, :] = no_k_gap_xs[:-1, :, :, :]
            x_sect[:, :, 1, :, :] = no_k_gap_xs[1:, :, :, :]
        else:
            simple_xs = grid.unsplit_x_section_points(axis,
                                                      ref_slice0 = ref_slice0,
                                                      plus_face = plus_face,
                                                      masked = masked)
            x_sect = np.empty((grid.nk, nj_or_ni, 2, 2, 3))
            x_sect[:, :, 0, 0, :] = simple_xs[:-1, :-1, :]
            x_sect[:, :, 1, 0, :] = simple_xs[1:, :-1, :]
            x_sect[:, :, 0, 1, :] = simple_xs[:-1, 1:, :]
            x_sect[:, :, 1, 1, :] = simple_xs[1:, 1:, :]

    if rotate:
        if azimuth is None:
            direction = vec.points_direction_vector(x_sect, axis = 1)
        else:
            direction = vec.unit_vector_from_azimuth(azimuth)
        x_sect = vec.rotate_xyz_array_around_z_axis(x_sect, direction)
        x_sect[..., 0] -= np.nanmin(x_sect[..., 0])
        x_sect[..., 1] -= np.nanmin(x_sect[..., 1])

    return x_sect


def coordinate_line_end_points(grid):
    """Returns xyz of top and bottom of each primary pillar.

    returns:
       numpy float array of shape (nj + 1, ni + 1, 2, 3)
    """

    points = grid.points_ref(masked = False).reshape((grid.nk_plus_k_gaps + 1, -1, 3))
    primary_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    result = np.empty((grid.nj + 1, grid.ni + 1, 2, 3))
    result[:, :, 0, :] = points[0, :primary_pillar_count, :].reshape((grid.nj + 1, grid.ni + 1, 3))
    result[:, :, 1, :] = points[-1, :primary_pillar_count, :].reshape((grid.nj + 1, grid.ni + 1, 3))
    return result


def z_corner_point_depths(grid, order = 'cellular'):
    """Returns the z (depth) values of each corner of each cell.

    arguments:
       order (string, default 'cellular'): either 'cellular' or 'linear'; if 'cellular' the resulting array has
          shape (nk, nj, ni, 2, 2, 2); if 'linear', the shape is (nk, 2, nj, 2, ni, 2)

    returns:
       numpy array of shape (nk, nj, ni, 2, 2, 2) or (nk, 2, nj, 2, ni, 2); for the cellular ordering, the
       result can be indexed with [k, j, i, kp, jp, ip] (where kp, for example, is 0 for the K- face and 1 for K+);
       for the linear ordering, the equivalent indexing is [k, kp, j, jp, i, ip], as used by some common simulator
       keyword formats
    """

    assert order in ['cellular', 'linear']

    z_cp = np.empty((grid.nk, grid.nj, grid.ni, 2, 2, 2))
    points = grid.points_ref()
    if grid.has_split_coordinate_lines:
        grid.create_column_pillar_mapping()
        # todo: replace j,i for loops with numpy broadcasting
        if grid.k_gaps:
            for j in range(grid.nj):
                for i in range(grid.ni):
                    z_cp[:, j, i, 0, 0, 0] = points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 0, 0], 2]
                    z_cp[:, j, i, 1, 0, 0] = points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 0, 0], 2]
                    z_cp[:, j, i, 0, 1, 0] = points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 1, 0], 2]
                    z_cp[:, j, i, 1, 1, 0] = points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 1, 0], 2]
                    z_cp[:, j, i, 0, 0, 1] = points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 0, 1], 2]
                    z_cp[:, j, i, 1, 0, 1] = points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 0, 1], 2]
                    z_cp[:, j, i, 0, 1, 1] = points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 1, 1], 2]
                    z_cp[:, j, i, 1, 1, 1] = points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 1, 1], 2]
        else:
            for j in range(grid.nj):
                for i in range(grid.ni):
                    z_cp[:, j, i, 0, 0, 0] = points[:-1, grid.pillars_for_column[j, i, 0, 0], 2]
                    z_cp[:, j, i, 1, 0, 0] = points[1:, grid.pillars_for_column[j, i, 0, 0], 2]
                    z_cp[:, j, i, 0, 1, 0] = points[:-1, grid.pillars_for_column[j, i, 1, 0], 2]
                    z_cp[:, j, i, 1, 1, 0] = points[1:, grid.pillars_for_column[j, i, 1, 0], 2]
                    z_cp[:, j, i, 0, 0, 1] = points[:-1, grid.pillars_for_column[j, i, 0, 1], 2]
                    z_cp[:, j, i, 1, 0, 1] = points[1:, grid.pillars_for_column[j, i, 0, 1], 2]
                    z_cp[:, j, i, 0, 1, 1] = points[:-1, grid.pillars_for_column[j, i, 1, 1], 2]
                    z_cp[:, j, i, 1, 1, 1] = points[1:, grid.pillars_for_column[j, i, 1, 1], 2]
    else:
        if grid.k_gaps:
            z_cp[:, :, :, 0, 0, 0] = points[grid.k_raw_index_array, :-1, :-1, 2]
            z_cp[:, :, :, 1, 0, 0] = points[grid.k_raw_index_array + 1, :-1, :-1, 2]
            z_cp[:, :, :, 0, 1, 0] = points[grid.k_raw_index_array, 1:, :-1, 2]
            z_cp[:, :, :, 1, 1, 0] = points[grid.k_raw_index_array + 1, 1:, :-1, 2]
            z_cp[:, :, :, 0, 0, 1] = points[grid.k_raw_index_array, :-1, 1:, 2]
            z_cp[:, :, :, 1, 0, 1] = points[grid.k_raw_index_array + 1, :-1, 1:, 2]
            z_cp[:, :, :, 0, 1, 1] = points[grid.k_raw_index_array, 1:, 1:, 2]
            z_cp[:, :, :, 1, 1, 1] = points[grid.k_raw_index_array + 1, 1:, 1:, 2]
        else:
            z_cp[:, :, :, 0, 0, 0] = points[:-1, :-1, :-1, 2]
            z_cp[:, :, :, 1, 0, 0] = points[1:, :-1, :-1, 2]
            z_cp[:, :, :, 0, 1, 0] = points[:-1, 1:, :-1, 2]
            z_cp[:, :, :, 1, 1, 0] = points[1:, 1:, :-1, 2]
            z_cp[:, :, :, 0, 0, 1] = points[:-1, :-1, 1:, 2]
            z_cp[:, :, :, 1, 0, 1] = points[1:, :-1, 1:, 2]
            z_cp[:, :, :, 0, 1, 1] = points[:-1, 1:, 1:, 2]
            z_cp[:, :, :, 1, 1, 1] = points[1:, 1:, 1:, 2]

    if order == 'linear':
        return np.transpose(z_cp, axes = (0, 3, 1, 4, 2, 5))
    return z_cp


def corner_points(grid, cell_kji0 = None, points_root = None, cache_resqml_array = True, cache_cp_array = False):
    """Returns a numpy array of corner points for a single cell or the whole grid.

    notes:
       if cell_kji0 is not None, a 4D array of shape (2, 2, 2, 3) holding single cell corner points in logical order
       [kp, jp, ip, xyz] is returned; if cell_kji0 is None, a pagoda style 7D array [k, j, i, kp, jp, ip, xyz] is
       cached and returned;
       the ordering of the corner points is in the logical order, which is not the same as that used by Nexus CORP data;
       olio.grid_functions.resequence_nexus_corp() can be used to switch back and forth between this pagoda ordering
       and Nexus corp ordering;
       this is the usual way to access full corner points for cells where working with native resqml data is undesirable

    :meta common:
    """

    # note: this function returns a derived object rather than a native resqml object

    def one_cell_cp(grid, cell_kji0, points_root, cache_array):
        cp = np.full((2, 2, 2, 3), np.NaN)
        if not grr_dg.geometry_defined_for_all_cells(grid):
            if not grr_dg.cell_geometry_is_defined(grid, cell_kji0 = cell_kji0, cache_array = cache_array):
                return cp
        corner_index = np.zeros(3, dtype = 'int')
        for kp in range(2):
            corner_index[0] = kp
            for jp in range(2):
                corner_index[1] = jp
                for ip in range(2):
                    corner_index[2] = ip
                    one_point = grid.point(cell_kji0,
                                           corner_index = corner_index,
                                           points_root = points_root,
                                           cache_array = cache_array)
                    if one_point is not None:
                        cp[kp, jp, ip] = one_point
        return cp

    if cell_kji0 is None:
        cache_cp_array = True
    if hasattr(grid, 'array_corner_points'):
        if cell_kji0 is None:
            return grid.array_corner_points
        return grid.array_corner_points[tuple(cell_kji0)]
    points_root = grid.resolve_geometry_child('Points', child_node = points_root)
    #      if points_root is None: return None  # geometry not present
    if cache_resqml_array:
        grid.point_raw(points_root = points_root, cache_array = True)
    if cache_cp_array:
        grid.array_corner_points = np.zeros((grid.nk, grid.nj, grid.ni, 2, 2, 2, 3))
        points = grid.points_ref()
        if points is None:
            return None  # geometry not present
        if grid.has_split_coordinate_lines:
            grid.create_column_pillar_mapping()
            # todo: replace j,i for loops with numpy broadcasting
            if grid.k_gaps:
                for j in range(grid.nj):
                    for i in range(grid.ni):
                        grid.array_corner_points[:, j, i, 0, 0, 0, :] = points[grid.k_raw_index_array,
                                                                               grid.pillars_for_column[j, i, 0, 0], :]
                        grid.array_corner_points[:, j, i, 1, 0, 0, :] = points[grid.k_raw_index_array + 1,
                                                                               grid.pillars_for_column[j, i, 0, 0], :]
                        grid.array_corner_points[:, j, i, 0, 1, 0, :] = points[grid.k_raw_index_array,
                                                                               grid.pillars_for_column[j, i, 1, 0], :]
                        grid.array_corner_points[:, j, i, 1, 1, 0, :] = points[grid.k_raw_index_array + 1,
                                                                               grid.pillars_for_column[j, i, 1, 0], :]
                        grid.array_corner_points[:, j, i, 0, 0, 1, :] = points[grid.k_raw_index_array,
                                                                               grid.pillars_for_column[j, i, 0, 1], :]
                        grid.array_corner_points[:, j, i, 1, 0, 1, :] = points[grid.k_raw_index_array + 1,
                                                                               grid.pillars_for_column[j, i, 0, 1], :]
                        grid.array_corner_points[:, j, i, 0, 1, 1, :] = points[grid.k_raw_index_array,
                                                                               grid.pillars_for_column[j, i, 1, 1], :]
                        grid.array_corner_points[:, j, i, 1, 1, 1, :] = points[grid.k_raw_index_array + 1,
                                                                               grid.pillars_for_column[j, i, 1, 1], :]
            else:
                for j in range(grid.nj):
                    for i in range(grid.ni):
                        grid.array_corner_points[:, j, i, 0, 0, 0, :] = points[:-1, grid.pillars_for_column[j, i, 0,
                                                                                                            0], :]
                        grid.array_corner_points[:, j, i, 1, 0, 0, :] = points[1:, grid.pillars_for_column[j, i, 0,
                                                                                                           0], :]
                        grid.array_corner_points[:, j, i, 0, 1, 0, :] = points[:-1, grid.pillars_for_column[j, i, 1,
                                                                                                            0], :]
                        grid.array_corner_points[:, j, i, 1, 1, 0, :] = points[1:, grid.pillars_for_column[j, i, 1,
                                                                                                           0], :]
                        grid.array_corner_points[:, j, i, 0, 0, 1, :] = points[:-1, grid.pillars_for_column[j, i, 0,
                                                                                                            1], :]
                        grid.array_corner_points[:, j, i, 1, 0, 1, :] = points[1:, grid.pillars_for_column[j, i, 0,
                                                                                                           1], :]
                        grid.array_corner_points[:, j, i, 0, 1, 1, :] = points[:-1, grid.pillars_for_column[j, i, 1,
                                                                                                            1], :]
                        grid.array_corner_points[:, j, i, 1, 1, 1, :] = points[1:, grid.pillars_for_column[j, i, 1,
                                                                                                           1], :]
        else:
            if grid.k_gaps:
                grid.array_corner_points[:, :, :, 0, 0, 0, :] = points[grid.k_raw_index_array, :-1, :-1, :]
                grid.array_corner_points[:, :, :, 1, 0, 0, :] = points[grid.k_raw_index_array + 1, :-1, :-1, :]
                grid.array_corner_points[:, :, :, 0, 1, 0, :] = points[grid.k_raw_index_array, 1:, :-1, :]
                grid.array_corner_points[:, :, :, 1, 1, 0, :] = points[grid.k_raw_index_array + 1, 1:, :-1, :]
                grid.array_corner_points[:, :, :, 0, 0, 1, :] = points[grid.k_raw_index_array, :-1, 1:, :]
                grid.array_corner_points[:, :, :, 1, 0, 1, :] = points[grid.k_raw_index_array + 1, :-1, 1:, :]
                grid.array_corner_points[:, :, :, 0, 1, 1, :] = points[grid.k_raw_index_array, 1:, 1:, :]
                grid.array_corner_points[:, :, :, 1, 1, 1, :] = points[grid.k_raw_index_array + 1, 1:, 1:, :]
            else:
                grid.array_corner_points[:, :, :, 0, 0, 0, :] = points[:-1, :-1, :-1, :]
                grid.array_corner_points[:, :, :, 1, 0, 0, :] = points[1:, :-1, :-1, :]
                grid.array_corner_points[:, :, :, 0, 1, 0, :] = points[:-1, 1:, :-1, :]
                grid.array_corner_points[:, :, :, 1, 1, 0, :] = points[1:, 1:, :-1, :]
                grid.array_corner_points[:, :, :, 0, 0, 1, :] = points[:-1, :-1, 1:, :]
                grid.array_corner_points[:, :, :, 1, 0, 1, :] = points[1:, :-1, 1:, :]
                grid.array_corner_points[:, :, :, 0, 1, 1, :] = points[:-1, 1:, 1:, :]
                grid.array_corner_points[:, :, :, 1, 1, 1, :] = points[1:, 1:, 1:, :]
    if cell_kji0 is None:
        return grid.array_corner_points
    if not grr_dg.geometry_defined_for_all_cells(grid):
        if not grr_dg.cell_geometry_is_defined(grid, cell_kji0, cache_array = cache_resqml_array):
            return None
    if hasattr(grid, 'array_corner_points'):
        return grid.array_corner_points[tuple(cell_kji0)]
    cp = one_cell_cp(grid, cell_kji0, points_root = points_root, cache_array = cache_resqml_array)
    return cp


def invalidate_corner_points(grid):
    """Deletes cached copy of corner points, if present.

    Use if any pillar geometry changes, or to reclaim memory.
    """
    if hasattr(grid, 'array_corner_points'):
        delattr(grid, 'array_corner_points')


def centre_point(grid, cell_kji0 = None, cache_centre_array = False):
    """Returns centre point of a cell or array of centre points of all cells.

    Optionally cache centre points for all cells.

    arguments:
       cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
          centre point is required; zero based indexing
       cache_centre_array (boolean, default False): If True, or cell_kji0 is None, an array of centre points
          is generated and added as an attribute of the grid, with attribute name array_centre_point

    returns:
       (x, y, z) 3 element numpy array of floats holding centre point of cell;
       or numpy 3+1D array if cell_kji0 is None

    note:
       resulting coordinates are in the same (local) crs as the grid points

    :meta common:
    """

    if cell_kji0 is None:
        cache_centre_array = True

    # note: this function returns a derived object rather than a native resqml object
    if hasattr(grid, 'array_centre_point'):
        if cell_kji0 is None:
            return grid.array_centre_point
        return grid.array_centre_point[tuple(cell_kji0)]  # could check for nan here and return None
    if cache_centre_array:
        # todo: turn off nan warnings
        grid.array_centre_point = np.empty((grid.nk, grid.nj, grid.ni, 3))
        points = grid.points_ref(masked = False)  # todo: think about masking
        if hasattr(grid, 'array_corner_points'):
            grid.array_centre_point = 0.125 * np.sum(grid.array_corner_points,
                                                     axis = (3, 4, 5))  # mean of eight corner points for each cell
        elif grid.has_split_coordinate_lines:
            # todo: replace j,i for loops with numpy broadcasting
            grid.create_column_pillar_mapping()
            if grid.k_gaps:
                for j in range(grid.nj):
                    for i in range(grid.ni):
                        grid.array_centre_point[:, j, i, :] = 0.125 * (
                            points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 0, 0], :] +
                            points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 0, 0], :] +
                            points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 1, 0], :] +
                            points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 1, 0], :] +
                            points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 0, 1], :] +
                            points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 0, 1], :] +
                            points[grid.k_raw_index_array, grid.pillars_for_column[j, i, 1, 1], :] +
                            points[grid.k_raw_index_array + 1, grid.pillars_for_column[j, i, 1, 1], :])
            else:
                for j in range(grid.nj):
                    for i in range(grid.ni):
                        grid.array_centre_point[:, j,
                                                i, :] = 0.125 * (points[:-1, grid.pillars_for_column[j, i, 0, 0], :] +
                                                                 points[1:, grid.pillars_for_column[j, i, 0, 0], :] +
                                                                 points[:-1, grid.pillars_for_column[j, i, 1, 0], :] +
                                                                 points[1:, grid.pillars_for_column[j, i, 1, 0], :] +
                                                                 points[:-1, grid.pillars_for_column[j, i, 0, 1], :] +
                                                                 points[1:, grid.pillars_for_column[j, i, 0, 1], :] +
                                                                 points[:-1, grid.pillars_for_column[j, i, 1, 1], :] +
                                                                 points[1:, grid.pillars_for_column[j, i, 1, 1], :])
        else:
            if grid.k_gaps:
                grid.array_centre_point[:, :, :, :] = 0.125 * (
                    points[grid.k_raw_index_array, :-1, :-1, :] + points[grid.k_raw_index_array, :-1, 1:, :] +
                    points[grid.k_raw_index_array, 1:, :-1, :] + points[grid.k_raw_index_array, 1:, 1:, :] +
                    points[grid.k_raw_index_array + 1, :-1, :-1, :] + points[grid.k_raw_index_array + 1, :-1, 1:, :] +
                    points[grid.k_raw_index_array + 1, 1:, :-1, :] + points[grid.k_raw_index_array + 1, 1:, 1:, :])
            else:
                grid.array_centre_point[:, :, :, :] = 0.125 * (points[:-1, :-1, :-1, :] + points[:-1, :-1, 1:, :] +
                                                               points[:-1, 1:, :-1, :] + points[:-1, 1:, 1:, :] +
                                                               points[1:, :-1, :-1, :] + points[1:, :-1, 1:, :] +
                                                               points[1:, 1:, :-1, :] + points[1:, 1:, 1:, :])
        if cell_kji0 is None:
            return grid.array_centre_point
        return grid.array_centre_point[cell_kji0[0], cell_kji0[1],
                                       cell_kji0[2]]  # could check for nan here and return None
    cp = grid.corner_points(cell_kji0 = cell_kji0, cache_cp_array = False)
    if cp is None:
        return None
    centre = np.zeros(3)
    for axis in range(3):
        centre[axis] = np.mean(cp[:, :, :, axis])
    return centre


def centre_point_list(grid, cell_kji0s):
    """Returns centre points for a list of cells; caches centre points for all cells.

    arguments:
       cell_kji0s (numpy int array of shape (N, 3)): the (k, j, i) indices of the individual cells for which the
          centre points are required; zero based indexing

    returns:
       numpy float array of shape (N, 3) being the (x, y, z) centre points of the cells

    note:
       resulting coordinates are in the same (local) crs as the grid points
    """

    assert cell_kji0s.ndim == 2 and cell_kji0s.shape[1] == 3
    centres_list = np.empty(cell_kji0s.shape)
    for cell in range(len(cell_kji0s)):
        centres_list[cell] = grid.centre_point(cell_kji0 = cell_kji0s[cell], cache_centre_array = True)
    return centres_list


def point_areally(grid, tolerance = 0.001):
    """Returns array indicating which cells are reduced to a point in both I & J axes.

    Returns:
        numopy bool array of shape extent_kji

    Note:
       Any NaN point values will yield True for a cell
    """

    points = grid.points_ref(masked = False)
    # todo: turn off NaN warning for numpy > ?
    if grid.has_split_coordinate_lines:
        pillar_for_col = grid.create_column_pillar_mapping()
        j_pair_vectors = points[:, pillar_for_col[:, :, 1, :], :] - points[:, pillar_for_col[:, :, 0, :], :]
        i_pair_vectors = points[:, pillar_for_col[:, :, :, 1], :] - points[:, pillar_for_col[:, :, :, 0], :]
        j_pair_nans = np.isnan(j_pair_vectors)
        i_pair_nans = np.isnan(i_pair_vectors)
        any_nans = np.any(np.logical_or(j_pair_nans, i_pair_nans), axis = (3, 4))
        j_pair_extant = np.any(np.abs(j_pair_vectors) > tolerance, axis = -1)
        i_pair_extant = np.any(np.abs(i_pair_vectors) > tolerance, axis = -1)
        any_extant = np.any(np.logical_or(j_pair_extant, i_pair_extant), axis = 3)
    else:
        j_vectors = points[:, 1:, :, :] - points[:, :-1, :, :]
        i_vectors = points[:, :, 1:, :] - points[:, :, :-1, :]
        j_nans = np.any(np.isnan(j_vectors), axis = -1)
        i_nans = np.any(np.isnan(i_vectors), axis = -1)
        j_pair_nans = np.logical_or(j_nans[:, :, :-1], j_nans[:, :, 1:])
        i_pair_nans = np.logical_or(i_nans[:, :-1, :], i_nans[:, 1:, :])
        any_nans = np.logical_or(j_pair_nans, i_pair_nans)
        j_extant = np.any(np.abs(j_vectors) > tolerance, axis = -1)
        i_extant = np.any(np.abs(i_vectors) > tolerance, axis = -1)
        j_pair_extant = np.logical_or(j_extant[:, :, :-1], j_extant[:, :, 1:])
        i_pair_extant = np.logical_or(i_extant[:, :-1, :], i_extant[:, 1:, :])
        any_extant = np.logical_or(j_pair_extant, i_pair_extant)
    layered = np.logical_or(any_nans, np.logical_not(any_extant))
    if grid.k_gaps:
        return np.logical_and(layered[grid.k_raw_index_array], layered[grid.k_raw_index_array + 1])
    return np.logical_and(layered[:-1], layered[1:])


def interpolated_point(grid,
                       cell_kji0,
                       interpolation_fraction,
                       points_root = None,
                       cache_resqml_array = True,
                       cache_cp_array = False):
    """Returns xyz point interpolated from corners of cell.

    Depends on 3 interpolation fractions in range 0 to 1.
    """
    # todo: think about best ordering of axes operations given high aspect ratio of cells (for best accuracy)
    fp = np.empty(3)
    fm = np.empty(3)
    for axis in range(3):
        fp[axis] = max(min(interpolation_fraction[axis], 1.0), 0.0)
        fm[axis] = 1.0 - fp[axis]
    cp = grid.corner_points(cell_kji0,
                            points_root = points_root,
                            cache_resqml_array = cache_resqml_array,
                            cache_cp_array = cache_cp_array)
    c00 = (cp[0, 0, 0] * fm[0] + cp[1, 0, 0] * fp[0])
    c01 = (cp[0, 0, 1] * fm[0] + cp[1, 0, 1] * fp[0])
    c10 = (cp[0, 1, 0] * fm[0] + cp[1, 1, 0] * fp[0])
    c11 = (cp[0, 1, 1] * fm[0] + cp[1, 1, 1] * fp[0])
    c0 = c00 * fm[1] + c10 * fp[1]
    c1 = c01 * fm[1] + c11 * fp[1]
    c = c0 * fm[2] + c1 * fp[2]
    return c


def interpolated_points(grid,
                        cell_kji0,
                        interpolation_fractions,
                        points_root = None,
                        cache_resqml_array = True,
                        cache_cp_array = False):
    """Returns xyz points interpolated from corners of cell.

    Depending on 3 interpolation fraction numpy vectors, each value in range 0 to 1.

    arguments:
       cell_kji0 (triple int): indices of individual cell whose corner points are to be interpolated
       interpolation_fractions (list of three numpy vectors of floats): k, j & i interpolation fraction vectors, each element in range 0 to 1
       points_root (xml node, optional): for efficiency when making multiple calls, this can be set to the xml node of the points data
       cache_resqml_array (boolean, default True): if True, the resqml points data will be cached as an attribute of this grid object
       cache_cp_array (boolean, default False): if True a fully expanded 7D corner points array will be established for this grid and
          cached as an attribute (recommended if looping over many or all the cells and if memory space allows)

    returns:
       4D numpy float array of shape (nik, nij, nii, 3) being the interpolated points; nik is the number of elements in the first of the
       interpolation fraction lists (ie. for k); similarly for nij and nii; the final axis covers xyz

    notea:
       this method returns a lattice of trilinear interpolations of the corner point of the host cell; the returned points are in 'shared'
       arrangement (like resqml points data for an IjkGrid without split pillars or k gaps), not a fully expanded 7D array; calling code
       must redistribute to corner points of individual fine cells if that is the intention
    """

    assert len(interpolation_fractions) == 3
    fp = interpolation_fractions
    fm = []
    for axis in range(3):
        fm.append(1.0 - fp[axis])

    cp = grid.corner_points(cell_kji0,
                            points_root = points_root,
                            cache_resqml_array = cache_resqml_array,
                            cache_cp_array = cache_cp_array)

    c00 = (np.outer(fm[2], cp[0, 0, 0]) + np.outer(fp[2], cp[0, 0, 1]))
    c01 = (np.outer(fm[2], cp[0, 1, 0]) + np.outer(fp[2], cp[0, 1, 1]))
    c10 = (np.outer(fm[2], cp[1, 0, 0]) + np.outer(fp[2], cp[1, 0, 1]))
    c11 = (np.outer(fm[2], cp[1, 1, 0]) + np.outer(fp[2], cp[1, 1, 1]))
    c0 = (np.multiply.outer(fm[1], c00) + np.multiply.outer(fp[1], c01))
    c1 = (np.multiply.outer(fm[1], c10) + np.multiply.outer(fp[1], c11))
    c = (np.multiply.outer(fm[0], c0) + np.multiply.outer(fp[0], c1))

    return c


def find_cell_for_point_xy(grid, x, y, k0 = 0, vertical_ref = 'top', local_coords = True):
    """Searches in 2D for a cell containing point x,y in layer k0; return (j0, i0) or (None, None)."""

    # find minimum of manhatten distances from xy to each corner point
    # then check the four cells around that corner point
    if x is None or y is None:
        return (None, None)
    a = np.array([[x, y, 0.0]])  # extra axis needed to keep global_to_local_crs happy
    if not local_coords:
        grid.global_to_local_crs(a, crs_uuid = grid.crs_uuid)
    a[0, 2] = 0.0  # discard z
    kp = 1 if vertical_ref == 'base' else 0
    (pillar_j0, pillar_i0) = grid.nearest_pillar(a[0, :2], ref_k0 = k0, kp = kp)
    if pillar_j0 > 0 and pillar_i0 > 0:
        cell_kji0 = np.array((k0, pillar_j0 - 1, pillar_i0 - 1))
        poly = grid.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
        if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
    if pillar_j0 > 0 and pillar_i0 < grid.ni:
        cell_kji0 = np.array((k0, pillar_j0 - 1, pillar_i0))
        poly = grid.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
        if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
    if pillar_j0 < grid.nj and pillar_i0 > 0:
        cell_kji0 = np.array((k0, pillar_j0, pillar_i0 - 1))
        poly = grid.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
        if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
    if pillar_j0 < grid.nj and pillar_i0 < grid.ni:
        cell_kji0 = np.array((k0, pillar_j0, pillar_i0))
        poly = grid.poly_line_for_cell(cell_kji0, vertical_ref = vertical_ref)
        if poly is not None and pip.pip_cn(a[0, :2], poly):
            return (cell_kji0[1], cell_kji0[2])
    return (None, None)


def find_cell_for_x_sect_xz(x_sect, x, z):
    """Returns the (k0, j0) or (k0, i0) indices of the cell containing point x,z in the cross section.

    arguments:
       x_sect (numpy float array of shape (nk, nj or ni, 2, 2, 2 or 3): the cross section x,z or x,y,z data
       x (float) x-coordinate of point of interest in the cross section space
       z (float): y-coordinate of  point of interest in the cross section space

    note:
       the x_sect data is in the form returned by x_section_corner_points() or split_gap_x_section_points();
       the 2nd of the returned pair is either a J index or I index, whichever was not the axis specified
       when generating the x_sect data; returns (None, None) if point inclusion not detected; if xyz data is
       provided, the y values are ignored; note that the point of interest x,z coordinates are in the space of
       x_sect, so if rotation has occurred, the x value is no longer an easting and is typically picked off a
       cross section plot
    """

    def test_cell(p, x_sect, k0, ji0):
        poly = np.array([
            x_sect[k0, ji0, 0, 0, 0:3:2], x_sect[k0, ji0, 0, 1, 0:3:2], x_sect[k0, ji0, 1, 1, 0:3:2], x_sect[k0, ji0, 1,
                                                                                                             0, 0:3:2]
        ])
        if np.any(np.isnan(poly)):
            return False
        return pip.pip_cn(p, poly)

    assert x_sect.ndim == 5 and x_sect.shape[2] == 2 and x_sect.shape[3] == 2 and 2 <= x_sect.shape[4] <= 3
    n_k = x_sect.shape[0]
    n_j_or_i = x_sect.shape[1]
    tolerance = 1.0e-3

    if x_sect.shape[4] == 3:
        diffs = x_sect[:, :, :, :, 0:3:2].copy()  # x,z points only
    else:
        diffs = x_sect.copy()
    diffs -= np.array((x, z))
    diffs = np.sum(diffs * diffs, axis = -1)  # square of distance of each point from given x,z
    flat_index = np.nanargmin(diffs)
    min_dist_sqr = diffs.flatten()[flat_index]
    cell_flat_k0_ji0, flat_k_ji_p = divmod(flat_index, 4)
    found_k0, found_ji0 = divmod(cell_flat_k0_ji0, n_j_or_i)
    found_kp, found_jip = divmod(flat_k_ji_p, 2)

    found = test_cell((x, z), x_sect, found_k0, found_ji0)
    if found:
        return found_k0, found_ji0
    # check cells below whilst still close to point
    while found_k0 < n_k - 1:
        found_k0 += 1
        if np.nanmin(diffs[found_k0, found_ji0]) > min_dist_sqr + tolerance:
            break
        found = test_cell((x, z), x_sect, found_k0, found_ji0)
        if found:
            return found_k0, found_ji0

    # try neighbouring column (in case of fault or point exactly on face)
    ji_neighbour = 1 if found_jip == 1 else -1
    found_ji0 += ji_neighbour
    if 0 <= found_ji0 < n_j_or_i:
        col_diffs = diffs[:, found_ji0]
        flat_index = np.nanargmin(col_diffs)
        if col_diffs.flatten()[flat_index] <= min_dist_sqr + tolerance:
            found_k0 = flat_index // 4
            found = test_cell((x, z), x_sect, found_k0, found_ji0)
            if found:
                return found_k0, found_ji0
            # check cells below whilst still close to point
            while found_k0 < n_k - 1:
                found_k0 += 1
                if np.nanmin(diffs[found_k0, found_ji0]) > min_dist_sqr + tolerance:
                    break
                found = test_cell((x, z), x_sect, found_k0, found_ji0)
                if found:
                    return found_k0, found_ji0

    return None, None


def split_horizons_points(grid, min_k0 = None, max_k0 = None, masked = False):
    """Returns reference to a corner points layer of shape (nh, nj, ni, 2, 2, 3) where nh is number of horizons.

    arguments:
       min_k0 (integer): the lowest horizon layer number to be included, in the range 0 to nk + k_gaps; defaults to zero
       max_k0 (integer): the highest horizon layer number to be included, in the range 0 to nk + k_gaps; defaults to nk + k_gaps
       masked (boolean, default False): if True, a masked array is returned with NaN points masked out;
          if False, a simple (unmasked) numpy array is returned

    returns:
       numpy array of shape (nh, nj, ni, 2, 2, 3) where nh = max_k0 - min_k0 + 1, being corner point x,y,z values
       for horizon corners (h, j, i, jp, ip) where h is the horizon (layer interface) index in the range
       0 .. max_k0 - min_k0

    notes:
       data for horizon max_k0 is included in the result (unlike with python ranges);
       in the case of a grid with k gaps, the horizons points returned will follow the k indexing of the points data
       and calling code will need to keep track of the min_k0 offset when using k_raw_index_array to select a slice
       of the horizons points array
    """

    if min_k0 is None:
        min_k0 = 0
    else:
        assert 0 <= min_k0 <= grid.nk_plus_k_gaps
    if max_k0 is None:
        max_k0 = grid.nk_plus_k_gaps
    else:
        assert min_k0 <= max_k0 <= grid.nk_plus_k_gaps
    end_k0 = max_k0 + 1
    points = grid.points_ref(masked = False)
    hp = np.empty((end_k0 - min_k0, grid.nj, grid.ni, 2, 2, 3))
    if grid.has_split_coordinate_lines:
        grid.create_column_pillar_mapping()
        for j in range(grid.nj):
            for i in range(grid.ni):
                hp[:, j, i, 0, 0, :] = points[min_k0:end_k0, grid.pillars_for_column[j, i, 0, 0], :]
                hp[:, j, i, 1, 0, :] = points[min_k0:end_k0, grid.pillars_for_column[j, i, 1, 0], :]
                hp[:, j, i, 0, 1, :] = points[min_k0:end_k0, grid.pillars_for_column[j, i, 0, 1], :]
                hp[:, j, i, 1, 1, :] = points[min_k0:end_k0, grid.pillars_for_column[j, i, 1, 1], :]
    else:
        hp[:, :, :, 0, 0, :] = points[min_k0:end_k0, :-1, :-1, :]
        hp[:, :, :, 1, 0, :] = points[min_k0:end_k0, 1:, :-1, :]
        hp[:, :, :, 0, 1, :] = points[min_k0:end_k0, :-1, 1:, :]
        hp[:, :, :, 1, 1, :] = points[min_k0:end_k0, 1:, 1:, :]
    return hp
