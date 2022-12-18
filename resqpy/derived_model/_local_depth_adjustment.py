"""High level local depth adjustment function."""

import logging

log = logging.getLogger(__name__)

import os
import math as maths
import numpy as np

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.model as rq
import resqpy.olio.xml_et as rqet

import resqpy.derived_model._common as rqdm_c
import resqpy.derived_model._copy_grid as rqdm_cg


def local_depth_adjustment(epc_file,
                           source_grid,
                           centre_x,
                           centre_y,
                           radius,
                           centre_shift,
                           use_local_coords,
                           decay_shape = 'quadratic',
                           ref_k0 = 0,
                           store_displacement = False,
                           inherit_properties = False,
                           inherit_realization = None,
                           inherit_all_realizations = False,
                           new_grid_title = None,
                           new_epc_file = None):
    """Applies a local depth adjustment to the grid, adding as a new grid part in the model.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): a multi-layer RESQML grid object; if None, the epc_file is loaded
          and it should contain one ijk grid object (or one 'ROOT' grid) which is used as the source grid
       centre_x, centre_y (floats): the centre of the depth adjustment, corresponding to the location of maximum change
          in depth; crs is implicitly that of the grid but see also use_local_coords argument
       radius (float): the radius of adjustment of depths; units are implicitly xy (projected) units of grid crs
       centre_shift (float): the maximum vertical depth adjustment; units are implicily z (vertical) units of grid crs;
          use positive value to increase depth, negative to make shallower
       use_local_coords (boolean): if True, centre_x & centre_y are taken to be in the local coordinates of the grid's
          crs; otherwise the global coordinates
       decay_shape (string): 'linear' yields a cone shaped change in depth values; 'quadratic' (the default) yields a
          bell shaped change
       ref_k0 (integer, default 0): the layer in the grid to use as reference for determining the distance of a pillar
          from the centre of the depth adjustment; the corners of the top face of the reference layer are used
       store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
          displacement of cells' centres brought about by the local depth shift
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the adjusted grid (& crs)

    returns:
       new grid object which is a copy of the source grid with the local depth adjustment applied
    """

    log.info('adjusting depth')
    log.debug('centre x: {0:3.1f}; y: {1:3.1f}'.format(centre_x, centre_y))
    if use_local_coords:
        log.debug('centre x & y interpreted in local crs')
    log.debug('radius of influence: {0:3.1f}'.format(radius))
    log.debug('depth shift at centre: {0:5.3f}'.format(centre_shift))
    log.debug('decay shape: ' + decay_shape)
    log.debug('reference layer (k0 protocol): ' + str(ref_k0))

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    model, source_grid = rqdm_c._establish_model_and_source_grid(epc_file, source_grid)
    assert source_grid.grid_representation == 'IjkGrid'
    assert model is not None

    # take a copy of the grid
    grid = rqdm_cg.copy_grid(source_grid, model, copy_crs = True)

    # if not use_local_coords, convert centre_x & y into local_coords
    if grid.crs is None:
        grid.crs = rqc.Crs(model, uuid = grid.crs_uuid)
    if not use_local_coords:
        rotation = grid.crs.rotation
        if rotation > 0.001:
            log.error('unable to account for rotation in crs: use local coordinates')
            return
        centre_x -= grid.crs.x_offset
        centre_y -= grid.crs.y_offset
    z_inc_down = grid.crs.z_inc_down

    if not z_inc_down:
        centre_shift = -centre_shift

    # cache geometry in memory; needed prior to writing new coherent set of arrays to hdf5
    grid.cache_all_geometry_arrays()
    if grid.has_split_coordinate_lines:
        reshaped_points = grid.points_cached.copy()
    else:
        nkp1, njp1, nip1, xyz = grid.points_cached.shape
        reshaped_points = grid.points_cached.copy().reshape((nkp1, njp1 * nip1, xyz))
    assert reshaped_points.ndim == 3 and reshaped_points.shape[2] == 3
    assert ref_k0 >= 0 and ref_k0 < reshaped_points.shape[0]

    log.debug('reshaped_points.shape: ' + str(reshaped_points.shape))

    log.debug('min z before depth adjustment: ' + str(np.nanmin(reshaped_points[:, :, 2])))

    # for each pillar, find x, y for k = reference_layer_k0
    pillars_adjusted = 0

    # todo: replace with numpy array operations
    radius_sqr = radius * radius
    for pillar in range(reshaped_points.shape[1]):
        x, y, z = tuple(reshaped_points[ref_k0, pillar, :])
        # find distance of this pillar from the centre
        dx = centre_x - x
        dy = centre_y - y
        distance_sqr = (dx * dx) + (dy * dy)
        # if this pillar is beyond radius of influence, no action needed
        if distance_sqr > radius_sqr:
            continue
        distance = maths.sqrt(distance_sqr)
        # compute decayed shift as function of distance
        shift = _decayed_shift(centre_shift, distance, radius, decay_shape)
        # adjust depth values for pillar in cached array
        log.debug('adjusting pillar number {0} at x: {1:3.1f}, y: {2:3.1f}, distance: {3:3.1f} by {4:5.3f}'.format(
            pillar, x, y, distance, shift))
        reshaped_points[:, pillar, 2] += shift
        pillars_adjusted += 1

    # if no pillars adjusted: warn and return
    if pillars_adjusted == 0:
        log.warning('no pillars adjusted')
        return

    log.debug('min z after depth adjustment: ' + str(np.nanmin(reshaped_points[:, :, 2])))
    if grid.has_split_coordinate_lines:
        grid.points_cached[:] = reshaped_points
    else:
        grid.points_cached[:] = reshaped_points.reshape((nkp1, njp1, nip1, xyz))


#   model.copy_part(old_uuid, grid.uuid, change_hdf5_refs = True)   # copies the xml, substituting the new uuid in the root node (and in hdf5 refs)
    log.info(str(pillars_adjusted) + ' pillars adjusted')

    # build cell displacement property array(s)
    if store_displacement:
        log.debug('generating cell displacement property arrays')
        displacement_collection = rqdm_c._displacement_properties(grid, source_grid)
    else:
        displacement_collection = None

    collection = rqdm_c._prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                                    inherit_all_realizations)
    if collection is None:
        collection = displacement_collection
    elif displacement_collection is not None:
        collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'grid derived from {0} with local depth shift of {1:3.1f} applied'.format(
            str(rqet.citation_title_for_node(source_grid.root)), centre_shift)

    # write model
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


def _decayed_shift(centre_shift, distance, radius, decay_shape):
    norm_dist = min(distance / radius, 1.0)  # 0..1
    if decay_shape == 'linear':
        return (1.0 - norm_dist) * centre_shift
    elif decay_shape == 'quadratic':
        if norm_dist >= 0.5:
            x = (1.0 - norm_dist)
            return 2.0 * x * x * centre_shift
        else:
            return centre_shift * (1.0 - 2.0 * norm_dist * norm_dist)
    else:
        raise ValueError('unrecognized decay shape: ' + decay_shape)
