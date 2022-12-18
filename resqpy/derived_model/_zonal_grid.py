"""High level zonal grid function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.derived_model
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp

import resqpy.derived_model._common as rqdm_c
import resqpy.derived_model._zone_layer_ranges_from_array as rqdm_zlr


def zonal_grid(epc_file,
               source_grid = None,
               zone_title = None,
               zone_uuid = None,
               zone_layer_range_list = None,
               k0_min = None,
               k0_max = None,
               use_dominant_zone = False,
               inactive_laissez_faire = True,
               new_grid_title = None,
               new_epc_file = None):
    """Extends an existing model with a new version of the source grid converted to a single, thick, layer per zone.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): a multi-layer RESQML grid object; if None, the epc_file is loaded
          and it should contain one ijk grid object (or one 'ROOT' grid) which is used as the source grid
       zone_title (string): if not None, a discrete property with this as the citation title is used as the zone property
       zone_uuid (string or uuid): if not None, a discrete property with this uuid is used as the zone property (see notes)
       zone_layer_range_list (list of (int, int, int)): each entry being (min_k0, max_k0, zone_index); alternative to
          working from a zone array
       k0_min (int, optional): the minimum layer number in the source grid (zero based) to include in the zonal version;
          default is zero (ie. top layer in source grid)
       k0_max (int, optional): the maximum layer number in the source grid (zero based) to include in the zonal version;
          default is nk - 1 (ie. bottom layer in source grid)
       use_dominant_zone (boolean, default False): if True, the most common zone value in each layer is used for the whole
          layer; if False, then variation of zone values in active cells in a layer will raise an assertion error
       inactive_laissez_faire (boolean, optional): if True, a cell in the zonal grid will be set active if any of the
          corresponding cells in the source grid are active; otherwise all corresponding cells in the source grid
          must be active for the zonal cell to be active; default is True
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the zonal grid (& crs)

    returns:
       new grid object (grid.Grid) with one layer per zone of the source grid

    notes:
       usually one of zone_title or zone_uuid or zone_layer_range_list should be passed, if none are passed then a
       single layer grid is generated; zone_layer_range_list will take precendence if present
    """

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    model, source_grid = rqdm_c._establish_model_and_source_grid(epc_file, source_grid)
    assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']
    if source_grid.grid_representation == 'IjkBlockGrid':
        source_grid.make_regular_points_cached()
    assert model is not None
    single_layer_mode = (not zone_title and not zone_uuid and
                         (zone_layer_range_list is None or len(zone_layer_range_list) == 1))

    k0_min, k0_max = _set_or_check_k_min_max(k0_min, k0_max, source_grid)

    if not single_layer_mode:  # process zone array
        if zone_layer_range_list is None:
            zone_array = _fetch_zone_array(source_grid, zone_title, zone_uuid)
            zone_layer_range_list = rqdm_zlr.zone_layer_ranges_from_array(zone_array,
                                                                          k0_min,
                                                                          k0_max,
                                                                          use_dominant_zone = use_dominant_zone)
        zone_count = len(zone_layer_range_list)
        # above is list of (zone_min_k0, zone_max_k0, zone) sorted by zone_min_k0
        log.info('following layer ranges are based on top layer being numbered 1 (simulator protocol)')
        for (zone_min_k0, zone_max_k0, zone) in zone_layer_range_list:
            log.info('zone id {0:1d} covers layers {1:1d} to {2:1d}'.format(zone, zone_min_k0 + 1, zone_max_k0 + 1))
    else:
        zone_layer_range_list = [(k0_min, k0_max, 0)]
        zone_count = 1
    assert zone_count > 0, 'unexpected lack of zones'

    # create a new, empty grid object
    is_regular = grr.is_regular_grid(source_grid.root) and single_layer_mode
    grid = _empty_grid(model, source_grid, is_regular, k0_min, k0_max, zone_count)

    # aggregate inactive cell mask depending on laissez faire argument
    _set_inactive_cell_mask(source_grid, grid, inactive_laissez_faire, single_layer_mode, k0_min, k0_max,
                            zone_layer_range_list, zone_count)

    if not is_regular:
        _process_geometry(source_grid, grid, single_layer_mode, k0_min, k0_max, zone_layer_range_list, zone_count)

    # establish title for the new grid
    if new_grid_title is None or len(new_grid_title) == 0:
        if single_layer_mode:
            preamble = 'single layer'
        else:
            preamble = 'zonal'
        new_grid_title = preamble + ' version of ' + str(rqet.citation_title_for_node(source_grid.root))

    # write the new grid
    model.h5_release()
    if new_epc_file:
        rqdm_c._write_grid(new_epc_file, grid, grid_title = new_grid_title, mode = 'w')
    else:
        rqdm_c._write_grid(epc_file, grid, ext_uuid = None, grid_title = new_grid_title, mode = 'a')

    return grid


def single_layer_grid(epc_file,
                      source_grid = None,
                      k0_min = None,
                      k0_max = None,
                      inactive_laissez_faire = True,
                      new_grid_title = None,
                      new_epc_file = None):
    """Extends an existing model with a new version of the source grid converted to a single, thick, layer.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): a multi-layer RESQML grid object; if None, the epc_file is loaded
          and it should contain one ijk grid object (or one 'ROOT' grid) which is used as the source grid
       k0_min (int, optional): the minimum layer number in the source grid (zero based) to include in the single layer version;
          default is zero (ie. top layer in source grid)
       k0_max (int, optional): the maximum layer number in the source grid (zero based) to include in the single layer version;
          default is nk - 1 (ie. bottom layer in source grid)
       inactive_laissez_faire (boolean, optional): if True, a cell in the single layer grid will be set active if any of the
          corresponding cells in the source grid are active; otherwise all corresponding cells in the source grid
          must be active for the single layer cell to be active; default is True
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the single layer grid (& crs)

    returns:
       new grid object (grid.Grid) with a single layer representation of the source grid
    """

    return zonal_grid(epc_file,
                      source_grid = source_grid,
                      k0_min = k0_min,
                      k0_max = k0_max,
                      inactive_laissez_faire = inactive_laissez_faire,
                      new_grid_title = new_grid_title,
                      new_epc_file = new_epc_file)


def _fetch_zone_array(grid, zone_title = None, zone_uuid = None, masked = True):
    properties = grid.extract_property_collection()
    assert properties is not None and properties.number_of_parts() > 0, 'no properties found in relation to grid'
    properties = rqp.selective_version_of_collection(properties, continuous = False)
    assert properties is not None and properties.number_of_parts() > 0,  \
       'no discreet properties found in relation to grid'
    if zone_title:
        properties = rqp.selective_version_of_collection(properties,
                                                         citation_title = zone_title)  # could make case insensitive?
        assert properties is not None and properties.number_of_parts() > 0,  \
           'no discreet property found with title ' + zone_title
    if zone_uuid:
        if isinstance(zone_uuid, str):
            zone_uuid = bu.uuid_from_string(zone_uuid)
        zone_uuid_str = str(zone_uuid)
        if zone_title:
            postamble = ' (and title ' + zone_title + ')'
        else:
            postamble = ''
        assert zone_uuid in properties.uuids(), 'no property found with uuid ' + zone_uuid_str + postamble
        part_name = grid.model.part(uuid = zone_uuid)
    else:
        part_name = properties.singleton()
    return properties.cached_part_array_ref(part_name, masked = masked)  # .copy() needed?


def _empty_grid(model, source_grid, is_regular, k0_min, k0_max, zone_count):
    if is_regular:
        assert zone_count == 1
        dxyz_dkji = source_grid.block_dxyz_dkji.copy()
        dxyz_dkji[0] *= k0_max - k0_min + 1
        grid = grr.RegularGrid(model,
                               extent_kji = (1, source_grid.nj, source_grid.ni),
                               dxyz_dkji = dxyz_dkji,
                               origin = source_grid.block_origin,
                               crs_uuid = source_grid.crs_uuid,
                               set_points_cached = False)
    else:
        grid = grr.Grid(model)
        # inherit attributes from source grid
        grid.grid_representation = 'IjkGrid'
        grid.extent_kji = np.array((zone_count, source_grid.nj, source_grid.ni), dtype = 'int')
        grid.nk, grid.nj, grid.ni = zone_count, source_grid.nj, source_grid.ni
        grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
        grid.crs_uuid = source_grid.crs_uuid

    grid.k_direction_is_down = source_grid.k_direction_is_down
    grid.grid_is_right_handed = source_grid.grid_is_right_handed
    grid.pillar_shape = source_grid.pillar_shape
    return grid


def _scan_columns_for_reference_geometry(source_grid, grid, zone_layer_range_list, zone_count):
    source_points = source_grid.points_ref()
    log.debug('scanning columns (split pillars) for reference geometry')
    if not hasattr(source_grid, 'pillars_for_column'):
        source_grid.create_column_pillar_mapping()
    grid.pillars_for_column = source_grid.pillars_for_column.copy()
    no_cgid = (not hasattr(source_grid, 'array_cell_geometry_is_defined') or
               source_grid.array_cell_geometry_is_defined is None)
    for zone_i in range(zone_count):
        zk0_min, zk0_max = zone_layer_range_list[zone_i][0:2]
        for j in range(grid.nj):
            for i in range(grid.ni):
                if grid.inactive[zone_i, j, i]:
                    continue
                if zone_i == 0:
                    for k in range(zk0_min, zk0_max + 1):
                        if no_cgid or source_grid.array_cell_geometry_is_defined[k, j, i]:
                            for jp in range(2):
                                for ip in range(2):
                                    pillar = grid.pillars_for_column[j, i, jp, ip]
                                    grid.points_cached[0, pillar] = source_points[k, pillar]
                            break
                for k in range(zk0_max + 1, zk0_min - 1, -1):
                    if no_cgid or source_grid.array_cell_geometry_is_defined[k, j, i]:
                        for jp in range(2):
                            for ip in range(2):
                                pillar = grid.pillars_for_column[j, i, jp, ip]
                                grid.points_cached[zone_i + 1, pillar] = source_points[k + 1, pillar]
                        grid.array_cell_geometry_is_defined[zone_i, j, i] = True
                        break


def _set_inactive_cell_mask(source_grid, grid, inactive_laissez_faire, single_layer_mode, k0_min, k0_max,
                            zone_layer_range_list, zone_count):
    if source_grid.inactive is None:
        log.debug('setting inactive mask to None')
        grid.inactive = None
    elif single_layer_mode:
        if inactive_laissez_faire:
            log.debug('setting inactive mask using all mode (laissez faire)')
            grid.inactive = np.all(source_grid.inactive[k0_min:k0_max + 1], axis = 0).reshape(grid.extent_kji)
        else:
            log.debug('setting inactive mask using any mode (strict)')
            grid.inactive = np.any(source_grid.inactive[k0_min:k0_max + 1], axis = 0).reshape(grid.extent_kji)
    else:
        grid.inactive = np.zeros(grid.extent_kji, dtype = bool)
        for zone_i in range(zone_count):
            zk0_min, zk0_max, _ = zone_layer_range_list[zone_i]
            if inactive_laissez_faire:
                grid.inactive[zone_i] = np.all(source_grid.inactive[zk0_min:zk0_max + 1], axis = 0)
            else:
                grid.inactive[zone_i] = np.any(source_grid.inactive[zk0_min:zk0_max + 1], axis = 0)


def _process_geometry(source_grid, grid, single_layer_mode, k0_min, k0_max, zone_layer_range_list, zone_count):
    # rework the grid geometry
    source_grid.cache_all_geometry_arrays()
    # determine cell geometry is defined
    if hasattr(source_grid,
               'array_cell_geometry_is_defined') and source_grid.array_cell_geometry_is_defined is not None:
        grid.array_cell_geometry_is_defined = np.empty(grid.extent_kji, dtype = bool)
        if single_layer_mode:
            grid.array_cell_geometry_is_defined[0] = np.logical_and(source_grid.array_cell_geometry_is_defined[k0_min],
                                                                    source_grid.array_cell_geometry_is_defined[k0_max])
        else:
            for zone_i in range(zone_count):
                zk0_min, zk0_max, _ = zone_layer_range_list[zone_i]
                grid.array_cell_geometry_is_defined[zone_i] = np.logical_and(
                    source_grid.array_cell_geometry_is_defined[zk0_min],
                    source_grid.array_cell_geometry_is_defined[zk0_max])
                # could attempt to pick up some corner points from K-neighbouring cells
        grid.geometry_defined_for_all_cells_cached = np.all(grid.array_cell_geometry_is_defined)
    else:
        grid.geometry_defined_for_all_cells_cached = source_grid.geometry_defined_for_all_cells_cached
    # copy info for pillar geometry is defined
    grid.geometry_defined_for_all_pillars_cached = source_grid.geometry_defined_for_all_pillars_cached
    if hasattr(source_grid,
               'array_pillar_geometry_is_defined') and source_grid.array_pillar_geometry_is_defined is not None:
        grid.array_pillar_geometry_is_defined = source_grid.array_pillar_geometry_is_defined.copy()
    # get reference to points for source grid geometry
    source_points = source_grid.points_ref()
    # slice top and base points
    points_shape = list(source_points.shape)
    points_shape[0] = zone_count + 1  # nk + 1
    grid.points_cached = np.zeros(points_shape)
    if grid.geometry_defined_for_all_cells_cached:
        if single_layer_mode:
            grid.points_cached[0] = source_points[k0_min]
            grid.points_cached[1] = source_points[k0_max + 1]  # base face
        else:
            for zone_i in range(zone_count):
                if zone_i == 0:
                    grid.points_cached[0] = source_points[zone_layer_range_list[zone_i][0]]
                grid.points_cached[zone_i + 1] = source_points[zone_layer_range_list[zone_i]
                                                               [1]]  # or could use 0th element of tuple for zone_i+1
    elif not grid.has_split_coordinate_lines:
        log.debug('scanning columns (unsplit pillars) for reference geometry')
        # fill in geometry: todo: replace with array operations if possible
        no_cgid = (not hasattr(source_grid, 'array_cell_geometry_is_defined') or
                   source_grid.array_cell_geometry_is_defined is None)
        for zone_i in range(zone_count):
            zk0_min, zk0_max = zone_layer_range_list[zone_i][0:2]
            for j in range(grid.nj):
                for i in range(grid.ni):
                    if zone_i == 0:
                        for k in range(zk0_min, zk0_max + 1):
                            if no_cgid or source_grid.array_cell_geometry_is_defined[k, j, i]:
                                grid.points_cached[0, j:j + 2, i:i + 2] = source_points[k, j:j + 2, i:i + 2]
                                break
                    for k in range(zk0_max, zk0_min - 1, -1):
                        if no_cgid or source_grid.array_cell_geometry_is_defined[k, j, i]:
                            grid.points_cached[zone_i + 1, j:j + 2, i:i + 2] = source_points[k + 1, j:j + 2, i:i + 2]
                            grid.array_cell_geometry_is_defined[zone_i, j, i] = True
                            break
    else:
        _scan_columns_for_reference_geometry(source_grid, grid, zone_layer_range_list, zone_count)
    if grid.has_split_coordinate_lines:
        grid.split_pillar_indices_cached = source_grid.split_pillar_indices_cached.copy()
        grid.cols_for_split_pillars = source_grid.cols_for_split_pillars.copy()
        grid.cols_for_split_pillars_cl = source_grid.cols_for_split_pillars_cl.copy()
        grid.split_pillars_count = source_grid.split_pillars_count


def _set_or_check_k_min_max(k0_min, k0_max, source_grid):
    if k0_min is None:
        k0_min = 0
    else:
        assert k0_min >= 0 and k0_min < source_grid.nk
    if k0_max is None:
        k0_max = source_grid.nk - 1
    else:
        assert k0_max >= 0 and k0_max < source_grid.nk and k0_max >= k0_min
    return k0_min, k0_max
