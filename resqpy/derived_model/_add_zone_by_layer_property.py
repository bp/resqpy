"""High level add_zone_by_layer_property() function."""

import numpy as np

import resqpy.derived_model
import resqpy.grid as grr
import resqpy.property.property_kind as pk
import resqpy.model as rq

import resqpy.derived_model._add_one_grid_property_array as rqdm_aogp
import resqpy.derived_model._zone_layer_ranges_from_array as rqdm_zlr


def add_zone_by_layer_property(epc_file,
                               grid_uuid = None,
                               zone_by_layer_vector = None,
                               zone_by_cell_property_uuid = None,
                               use_dominant_zone = False,
                               use_local_property_kind = True,
                               null_value = -1,
                               title = 'ZONE',
                               realization = None,
                               extra_metadata = {}):
    """Adds a discrete zone property (and local property kind) with indexable element of layers.

    arguments:
       epc_file (string): file name to load resqml model from and to update with the zonal property
       grid_uuid (uuid.UUID or str, optional): required unless the model has only one grid, or one named ROOT
       zone_by_layer_vector (nk integers, optional): either this or zone_by_cell_property_uuid must be given;
          a 1D numpy array, tuple or list of ints, being the zone number to which each layer belongs
       zone_by_cell_property_uuid (uuid.UUID or str, optional): either this or zone_by_layer_vector must be given;
          the uuid of a discrete property with grid as supporting representation and cells as indexable elements,
          holidng the zone to which the cell belongs
       use_dominant_zone (boolean, default False): if True and more than one zone is represented within the cells
          of a layer, then the whole layer is assigned to the zone with the biggest count of cells in the layer;
          if False, an exception is raised if more than one zone is represented by the cells of a layer; ignored
          if zone_by_cell_property_uuid is None
       use_local_property_kind (boolean, default True): if True, the new zone by layer property is given a
          local property kind titled 'zone'; if False, the property kind will be set to 'discrete'
       null_value (int, default -1): the value to use if a layer does not belong to any zone (rarely used)
       title (str, default 'ZONE'): the citation title of the new zone by layer property
       realization (int, optional): if present the new zone by layer property is marked as belonging to this
          realization
       extra_metadata (dict, optional): any items in this dictionary are added as extra metadata to the new
          property

    returns:
       numpy vector of zone numbers (by layer), uuid of newly created property
    """

    assert zone_by_layer_vector is not None or zone_by_cell_property_uuid is not None
    assert zone_by_layer_vector is None or zone_by_cell_property_uuid is None

    model = rq.Model(epc_file)

    if grid_uuid is None:
        grid = model.grid()
        grid_uuid = grid.uuid
    else:
        grid = model.grid_for_uuid_from_grid_list(grid_uuid)
        if grid is None:
            grid = grr.any_grid(model, uuid = grid_uuid, find_properties = True)
    assert grid is not None, 'failed to establish grid object'

    if zone_by_layer_vector is not None:
        assert len(
            zone_by_layer_vector) == grid.nk, 'length of zone by layer vector does not match number of layers in grid'
        zone_by_layer = np.array(zone_by_layer_vector, dtype = int)
    elif zone_by_cell_property_uuid is not None:
        pc = grid.property_collection
        zone_by_cell_array = pc.single_array_ref(uuid = zone_by_cell_property_uuid)
        assert zone_by_cell_array is not None, 'zone by cell property array not found for uuid: ' + str(
            zone_by_cell_property_uuid)
        zone_range_list = rqdm_zlr.zone_layer_ranges_from_array(zone_by_cell_array,
                                                                use_dominant_zone = use_dominant_zone)
        assert zone_range_list is not None and len(
            zone_range_list) > 0, 'failed to convert zone by cell to zone by layer'
        zone_by_layer = np.full((grid.nk,), null_value, dtype = int)
        for min_k0, max_k0, zone_index in zone_range_list:
            assert 0 <= min_k0 <= max_k0 < grid.nk, 'zonal layer limits out of range for grid (probable bug)'
            zone_by_layer[min_k0:max_k0 + 1] = zone_index
    else:
        raise Exception('code failure')

    assert zone_by_layer.ndim == 1 and zone_by_layer.size == grid.nk

    if use_local_property_kind:
        property_kind = 'zone'
        zone_pk = pk.establish_zone_property_kind(model)
        local_property_kind_uuid = zone_pk.uuid
        model.store_epc(only_if_modified = True)
    else:
        property_kind = 'discrete'
        local_property_kind_uuid = None

    model.h5_release()

    property_uuid = rqdm_aogp.add_one_grid_property_array(epc_file,
                                                          zone_by_layer,
                                                          property_kind,
                                                          grid_uuid = grid.uuid,
                                                          source_info = 'zone defined by layer',
                                                          title = title,
                                                          discrete = True,
                                                          null_value = null_value,
                                                          indexable_element = 'layers',
                                                          realization = realization,
                                                          local_property_kind_uuid = local_property_kind_uuid,
                                                          extra_metadata = extra_metadata)

    return zone_by_layer, property_uuid
