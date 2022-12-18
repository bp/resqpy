"""A submodule containing functions relating to grid cell properties."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.vector_utilities as vec
import resqpy.olio.volume as vol
import resqpy.weights_and_measures as wam

import resqpy.grid
import resqpy.grid._defined_geometry as grr_dg


def thickness(grid,
              cell_kji0 = None,
              points_root = None,
              cache_resqml_array = True,
              cache_cp_array = False,
              cache_thickness_array = True,
              property_collection = None):
    """Returns vertical (z) thickness of cell and/or caches thicknesses for all cells.

    arguments:
       cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
                             thickness is required; zero based indexing
       cache_resqml_array (boolean, default True): If True, the raw points array from the hdf5 file
                             is cached in memory, but only if it is needed to generate the thickness
       cache_cp_array (boolean, default True): If True, an array of corner points is generated and
                             added as an attribute of the grid, with attribute name corner_points, but only
                             if it is needed in order to generate the thickness
       cache_thickness_array (boolean, default False): if True, thicknesses are generated for all cells in
                             the grid and added as an attribute named array_thickness
       property_collection (property:GridPropertyCollection, optional): If not None, this collection
                             is probed for a suitable thickness or cell length property which is used
                             preferentially to calculating thickness; if no suitable property is found,
                             the calculation is made as if the collection were None

    returns:
       float, being the thickness of cell identified by cell_kji0; or numpy float array if cell_kji0 is None

    notes:
       the function can be used to find the thickness of a single cell, or cache thickness for all cells, or both;
       if property_collection is not None, a suitable thickness or cell length property will be used if present;
       if calculated, thickness is defined as z difference between centre points of top and base faces (TVT);
       at present, assumes K increases with same polarity as z; if not, negative thickness will be calculated;
       units of result are implicitly those of z coordinates in grid's coordinate reference system, or units of
       measure of property array if the result is based on a suitable property

    :meta common:
    """

    def __load_from_property(collection):
        if collection is None:
            return None
        parts = collection.selective_parts_list(property_kind = 'thickness', facet_type = 'netgross', facet = 'gross')
        if len(parts) == 1:
            return collection.cached_part_array_ref(parts[0])
        parts = collection.selective_parts_list(property_kind = 'cell length', facet_type = 'direction', facet = 'K')
        if len(parts) == 1:
            return collection.cached_part_array_ref(parts[0])
        parts = collection.selective_parts_list(property_kind = 'thickness')
        if len(parts) == 1 and collection.facet_for_part(parts[0]) is None:
            return collection.cached_part_array_ref(parts[0])
        return None

    # note: this function optionally looks for a suitable thickness property, otherwise calculates from geometry
    # note: for some geometries, thickness might need to be defined as length of vector between -k & +k face centres (TST)
    # todo: give more control over source of data through optional args; offer TST or TVT option
    # todo: if cp array is not already cached, compute directly from points without generating cp
    # todo: cache uom
    assert cache_thickness_array or (cell_kji0 is not None)

    if hasattr(grid, 'array_thickness'):
        if cell_kji0 is None:
            return grid.array_thickness
        return grid.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None

    thick = __load_from_property(property_collection)
    if thick is not None:
        log.debug('thickness array loaded from property')
        if cache_thickness_array:
            grid.array_thickness = thick.copy()
        if cell_kji0 is None:
            return grid.array_thickness
        return grid.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None

    points_root = grid.resolve_geometry_child('Points', child_node = points_root)
    if cache_thickness_array:
        if cache_cp_array:
            grid.corner_points(points_root = points_root, cache_cp_array = True)
        if hasattr(grid, 'array_corner_points'):
            grid.array_thickness = np.abs(
                np.mean(grid.array_corner_points[:, :, :, 1, :, :, 2] - grid.array_corner_points[:, :, :, 0, :, :, 2],
                        axis = (3, 4)))
            if cell_kji0 is None:
                return grid.array_thickness
            return grid.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None
        grid.array_thickness = np.empty(tuple(grid.extent_kji))
        points = grid.point_raw(cache_array = True)  # cache points regardless
        if points is None:
            return None  # geometry not present
        if grid.k_gaps:
            pillar_thickness = points[grid.k_raw_index_array + 1, ..., 2] - points[grid.k_raw_index_array, ..., 2]
        else:
            pillar_thickness = points[1:, ..., 2] - points[:-1, ..., 2]
        if grid.has_split_coordinate_lines:
            pillar_for_col = grid.create_column_pillar_mapping()
            grid.array_thickness = np.abs(
                0.25 *
                (pillar_thickness[:, pillar_for_col[:, :, 0, 0]] + pillar_thickness[:, pillar_for_col[:, :, 0, 1]] +
                 pillar_thickness[:, pillar_for_col[:, :, 1, 0]] + pillar_thickness[:, pillar_for_col[:, :, 1, 1]]))
        else:
            grid.array_thickness = np.abs(0.25 * (pillar_thickness[:, :-1, :-1] + pillar_thickness[:, :-1, 1:] +
                                                  pillar_thickness[:, 1:, :-1] + pillar_thickness[:, 1:, 1:]))
        if cell_kji0 is None:
            return grid.array_thickness
        return grid.array_thickness[tuple(cell_kji0)]  # could check for nan here and return None

    cp = grid.corner_points(cell_kji0 = cell_kji0,
                            points_root = points_root,
                            cache_resqml_array = cache_resqml_array,
                            cache_cp_array = cache_cp_array)
    if cp is None:
        return None
    return abs(np.mean(cp[1, :, :, 2]) - np.mean(cp[0, :, :, 2]))


def volume(grid,
           cell_kji0 = None,
           points_root = None,
           cache_resqml_array = True,
           cache_cp_array = False,
           cache_centre_array = False,
           cache_volume_array = True,
           property_collection = None,
           required_uom = None):
    """Returns bulk rock volume of cell or numpy array of bulk rock volumes for all cells.

    arguments:
       cell_kji0 (optional): if present, the (k, j, i) indices of the individual cell for which the
                             volume is required; zero based indexing
       cache_resqml_array (boolean, default True): If True, the raw points array from the hdf5 file
                             is cached in memory, but only if it is needed to generate the volume
       cache_cp_array (boolean, default False): If True, an array of corner points is generated and
                             added as an attribute of the grid, with attribute name corner_points, but only
                             if it is needed in order to generate the volume
       cache_volume_array (boolean, default False): if True, volumes are generated for all cells in
                             the grid and added as an attribute named array_volume
       property_collection (property:GridPropertyCollection, optional): If not None, this collection
                             is probed for a suitable volume property which is used preferentially
                             to calculating volume; if no suitable property is found,
                             the calculation is made as if the collection were None
       required_uom (str, optional): if present, the RESQML unit of measure (for quantity volume) that
                             the volumes will be returned (and cached) in; if None, the grid's CRS
                             z units cubed will be used

    returns:
       float, being the volume of cell identified by cell_kji0;
       or numpy float array of shape (nk, nj, ni) if cell_kji0 is None

    notes:
       the function can be used to find the volume of a single cell, or cache volumes for all cells, or both;
       if property_collection is not None, a suitable volume property will be used if present;
       if calculated, volume is computed using 6 tetras each with a non-planar bilinear base face;

    :meta common:
    """

    # note: this function optionally looks for a suitable volume property, otherwise calculates from geometry
    # todo: give control over source with optional arguments
    # todo: cache uom

    def __load_vol_from_property(collection):  # returns array and uom
        if collection is None:
            return (None, None)
        parts = collection.selective_parts_list(property_kind = 'rock volume', facet_type = 'netgross', facet = 'gross')
        if len(parts) == 1:
            return (collection.cached_part_array_ref(parts[0]), collection.uom_for_part(parts[0]))
        parts = collection.selective_parts_list(property_kind = 'rock volume')
        if len(parts) == 1 and collection.facet_for_part(parts[0]) is None:
            return (collection.cached_part_array_ref(parts[0]), collection.uom_for_part(parts[0]))
        return (None, None)

    assert (cache_volume_array is not None) or (cell_kji0 is not None)

    required_uom = _get_volume_uom(grid, required_uom)

    if hasattr(grid, 'array_volume'):
        if required_uom != grid.array_volume_uom:
            grid.array_volume = wam.convert_volumes(grid.array_volume, grid.array_volume_uom, required_uom)
            grid.array_volume_uom = required_uom
        if cell_kji0 is None:
            return grid.array_volume
        return grid.array_volume[tuple(cell_kji0)]  # could check for nan here and return None

    vol_array, vol_uom = __load_vol_from_property(property_collection)
    if vol_array is not None:
        if (cache_volume_array or cell_kji0 is None) and (vol_uom != required_uom):
            vol_array = vol_array.copy()
            wam.convert_volumes(vol_array, vol_uom, required_uom)
            vol_uom = required_uom
        if cache_volume_array:
            grid.array_volume = vol_array
            grid.array_volume_uom = vol_uom
        if cell_kji0 is None:
            return vol_array
        if vol_uom == required_uom:
            return vol_array[tuple(cell_kji0)]  # could check for nan here and return None
        return wam.convert(vol_array[tuple(cell_kji0)], vol_uom, required_uom)

    cache_cp_array = cache_cp_array or cell_kji0 is None
    cache_volume_array = cache_volume_array or cell_kji0 is None

    off_hand = grid.off_handed()

    conversion_factor = _get_volume_conversion_factor(grid, required_uom)

    points_root = grid.resolve_geometry_child('Points', child_node = points_root)
    if points_root is None:
        return None  # geometry not present
    centre_array = None
    if cache_volume_array:
        grid.corner_points(points_root = points_root, cache_cp_array = True)
        if cache_centre_array:
            grid.centre_point(cache_centre_array = True)
            centre_array = grid.array_centre_point
        vol_array = vol.tetra_volumes(grid.array_corner_points, centres = centre_array, off_hand = off_hand)
        if conversion_factor is not None:
            vol_array *= conversion_factor
        if cache_volume_array:
            grid.array_volume = vol_array
            grid.array_volume_uom = required_uom
        if cell_kji0 is None:
            return vol_array
        return vol_array[tuple(cell_kji0)]  # could check for nan here and return None

    cp = grid.corner_points(cell_kji0 = cell_kji0,
                            points_root = points_root,
                            cache_resqml_array = cache_resqml_array,
                            cache_cp_array = cache_cp_array)
    if cp is None:
        return None
    v = vol.tetra_cell_volume(cp, off_hand = off_hand)
    return v if conversion_factor is None else conversion_factor * v


def _get_volume_uom(grid, required_uom):
    if required_uom:
        assert required_uom in wam.valid_uoms(quantity = 'volume'), f'invalid volume unit of measure: {required_uom}'
    else:
        linear_uom = grid.crs.z_units
        if linear_uom.startswith('ft'):
            linear_uom = 'ft'
        required_uom = linear_uom + '3'
        if required_uom not in wam.valid_uoms(quantity = 'volume'):
            required_uom = 'm3'
    return required_uom


def _get_volume_conversion_factor(grid, required_uom):
    conversion_factor = None
    if grid.crs.xy_units != grid.crs.z_units or required_uom != grid.crs.z_units + '3':
        conversion_factor = wam.convert(1.0, grid.crs.xy_units, 'm')
        conversion_factor *= conversion_factor
        conversion_factor *= wam.convert(1.0, grid.crs.z_units, 'm')
        if required_uom != 'm3':
            conversion_factor *= wam.convert(1.0, 'm3', required_uom)
    return conversion_factor


def pinched_out(grid,
                cell_kji0 = None,
                tolerance = 0.001,
                points_root = None,
                cache_resqml_array = True,
                cache_cp_array = False,
                cache_thickness_array = False,
                cache_pinchout_array = None):
    """Returns boolean or boolean array indicating whether cell is pinched out.

    Pinched out means cell has a thickness less than tolerance.

    :meta common:
    """

    # note: this function returns a derived object rather than a native resqml object
    # note: returns True for cells without geometry
    # todo: check behaviour in response to NaNs and undefined geometry
    if cache_pinchout_array is None:
        cache_pinchout_array = (cell_kji0 is None)

    if grid.pinchout is not None:
        if cell_kji0 is None:
            return grid.pinchout
        return grid.pinchout[tuple(cell_kji0)]

    if points_root is None:
        points_root = grid.resolve_geometry_child('Points', child_node = points_root)
    #         if points_root is None: return None  # geometry not present

    thick = grid.thickness(
        cell_kji0,
        points_root = points_root,
        cache_resqml_array = cache_resqml_array,
        cache_cp_array = cache_cp_array,  # deprecated
        cache_thickness_array = cache_thickness_array or cache_pinchout_array)
    if cache_pinchout_array:
        grid.pinchout = np.where(np.isnan(grid.array_thickness), True, np.logical_not(grid.array_thickness > tolerance))
        if cell_kji0 is None:
            return grid.pinchout
        return grid.pinchout[tuple(cell_kji0)]
    if thick is not None:
        return thick <= tolerance
    return None


def cell_inactive(grid, cell_kji0, pv_array = None, pv_tol = 0.01):
    """Returns True if the cell is inactive."""

    if grid.inactive is not None:
        return grid.inactive[tuple(cell_kji0)]
    grid.extract_inactive_mask()
    if grid.inactive is not None:
        return grid.inactive[tuple(cell_kji0)]
    if pv_array is not None:  # fabricate an inactive mask from pore volume data
        grid.inactive = not (pv_array > pv_tol)  # NaN in pv array will end up inactive
        return grid.inactive[tuple(cell_kji0)]
    return ((not grr_dg.cell_geometry_is_defined(grid, cell_kji0 = cell_kji0)) or
            grid.pinched_out(cell_kji0, cache_pinchout_array = True))


def interface_vector(grid, cell_kji0, axis, points_root = None, cache_resqml_array = True, cache_cp_array = False):
    """Returns an xyz vector between centres of an opposite pair of faces of the cell (or vectors for all cells).

    note:
        units are implicitly those of the grid's crs; differing xy & z units would imply that the direction of the
        vector is not a true direction
    """

    face_0_centre = grid.face_centre(cell_kji0,
                                     axis,
                                     0,
                                     points_root = points_root,
                                     cache_resqml_array = cache_resqml_array,
                                     cache_cp_array = cache_cp_array)
    face_1_centre = grid.face_centre(cell_kji0,
                                     axis,
                                     1,
                                     points_root = points_root,
                                     cache_resqml_array = cache_resqml_array,
                                     cache_cp_array = cache_cp_array)
    return face_1_centre - face_0_centre


def interface_length(grid,
                     cell_kji0,
                     axis,
                     points_root = None,
                     cache_resqml_array = True,
                     cache_cp_array = False,
                     required_uom = None):
    """Returns the length between centres of an opposite pair of faces of the cell.

    note:
       if required_uom is not specified, units of returned length are the grid's crs xy units
    """

    assert cell_kji0 is not None
    v = grid.interface_vector(cell_kji0,
                              axis,
                              points_root = points_root,
                              cache_resqml_array = cache_resqml_array,
                              cache_cp_array = cache_cp_array)
    if grid.crs.xy_units != grid.crs.z_units:
        v[2] = wam.convert_lengths(v[2], grid.crs.z_units, grid.crs.xy_units)
    if required_uom:
        wam.convert_lengths(v, grid.crs.xy_units, required_uom)
    return vec.naive_length(v)


def interface_vectors_kji(grid, cell_kji0, points_root = None, cache_resqml_array = True, cache_cp_array = False):
    """Returns 3 interface centre point difference vectors for axes k, j, i.

    note:
        units are implicitly those of the grid's crs; differing xy & z units would imply that the direction of the
        vectors are not true directions
    """

    result = np.zeros((3, 3))
    for axis in range(3):
        result[axis] = grid.interface_vector(cell_kji0,
                                             axis,
                                             points_root = points_root,
                                             cache_resqml_array = cache_resqml_array,
                                             cache_cp_array = cache_cp_array)
    return result


def interface_lengths_kji(grid,
                          cell_kji0,
                          points_root = None,
                          cache_resqml_array = True,
                          cache_cp_array = False,
                          required_uom = None):
    """Returns 3 interface centre point separation lengths for axes k, j, i.

    note:
       if required_uom is not specified, units of returned lengths are the grid's crs xy units
    """
    result = np.zeros(3)
    for axis in range(3):
        result[axis] = grid.interface_length(cell_kji0,
                                             axis,
                                             points_root = points_root,
                                             cache_resqml_array = cache_resqml_array,
                                             cache_cp_array = cache_cp_array,
                                             required_uom = required_uom)
    return result


def poly_line_for_cell(grid, cell_kji0, vertical_ref = 'top'):
    """Returns a numpy array of shape (4, 3) being the 4 corners.

    Corners are in order J-I-, J-I+, J+I+, J+I-; from the top or base face.
    """
    if vertical_ref == 'top':
        kp = 0
    elif vertical_ref == 'base':
        kp = 1
    else:
        raise ValueError('vertical reference not catered for: ' + vertical_ref)
    poly = np.empty((4, 3))
    cp = grid.corner_points(cell_kji0 = cell_kji0)
    if cp is None:
        return None
    poly[0] = cp[kp, 0, 0]
    poly[1] = cp[kp, 0, 1]
    poly[2] = cp[kp, 1, 1]
    poly[3] = cp[kp, 1, 0]
    return poly
