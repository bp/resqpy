"""zone_layer_ranges_from_array() function."""

import logging

log = logging.getLogger(__name__)

import numpy as np


def zone_layer_ranges_from_array(zone_array, min_k0 = 0, max_k0 = None, use_dominant_zone = False):
    """Returns a list of (zone_min_k0, zone_max_k0, zone_index) derived from zone_array.

    arguments:
       zone_array (3D numpy int array or masked array): array holding zone index value per cell
       min_k0 (int, default 0): the minimum layer number (0 based) to be included in the ranges
       max_k0 (int, default None): the maximum layer number (0 based) to be included in the ranges;
          note that this layer is included (unlike in python ranges); if None, the maximum layer
          number in zone_array is used
       use_dominant_zone (boolean, default False): if True, the most common zone value in each layer is used for the whole
          layer; if False, then variation of zone values in active cells in a layer will raise an assertion error

    returns:
       a list of (int, int, int) being (zone_min_k0, zone_max_k0, zone_index) for each zone index value present

    notes:
       the function requires zone indices (for active cells, if zone_array is masked) within a layer
       to be consistent: an assertion error is raised otherwise; the returned list is sorted by layer
       ranges rather than zone index; if use_dominant_zone is True then a side effect of the function is
       to modify the values in zone_array to be consistent across each layer, effectively reassigning some
       cells to a different zone!
    """

    if max_k0 is None:
        max_k0 = zone_array.shape[0] - 1
    if use_dominant_zone:
        _dominant_zone(zone_array)
    zone_list = np.unique(zone_array)
    log.debug('list of zones: ' + str(zone_list))
    assert len(zone_list) > 0, 'no zone values present (all cells inactive?)'
    zone_layer_range_list = []  # list of (zone_min_k0, zone_max_k0, zone)
    for zone in zone_list:
        single_zone_mask = (zone_array == zone)
        zone_min_k0 = None
        zone_max_k0 = None
        for k0 in range(min_k0, max_k0 + 1):
            if np.any(single_zone_mask[k0]):
                zone_min_k0 = k0
                break
        if zone_min_k0 is None:
            log.warning('no active cells for zone ' + str(zone) + ' in layer range ' + str(min_k0 + 1) + ' to ' +
                        str(max_k0 + 1))
            continue
        for k0 in range(max_k0, min_k0 - 1, -1):
            if np.any(single_zone_mask[k0]):
                zone_max_k0 = k0
                break
        # require all active cells in zone layer range to be for this zone
        for k0 in range(zone_min_k0, zone_max_k0 + 1):
            assert np.all(single_zone_mask[k0]), 'unacceptable zone variation with layer ' + str(k0 + 1)
        zone_layer_range_list.append((zone_min_k0, zone_max_k0, zone))
    assert len(zone_layer_range_list) > 0, 'no zone layer ranges derived from zone array'
    zone_layer_range_list.sort()
    for zone_i in range(1, len(zone_layer_range_list)):
        assert zone_layer_range_list[zone_i][0] > zone_layer_range_list[
            zone_i - 1][1], 'overlapping zone layer ranges'  # todo: add more info
        if zone_layer_range_list[zone_i][0] > zone_layer_range_list[zone_i - 1][1] + 1:
            log.warning('gap in zonal layer ranges, missing layer(s) being arbitrarily assigned to zone below'
                       )  # todo: add more info to log
            zone_layer_range_list[zone_i][0] = zone_layer_range_list[zone_i - 1][1] + 1
    return zone_layer_range_list


def _dominant_zone(zone_array):
    # modifies data in zone_array such that each layer has a single (most common) value
    for k in range(zone_array.shape[0]):
        unique_zones = np.unique(zone_array[k])
        if len(unique_zones) <= 1:
            continue
        dominant = unique_zones[0]
        dominant_count = np.count_nonzero(zone_array[k] == dominant)
        for contender in unique_zones[1:]:
            contender_count = np.count_nonzero(zone_array[k] == contender)
            if contender_count > dominant_count:
                dominant = contender
                dominant_count = contender_count
        zone_array[k] = dominant
        log.info('layer ' + str(k + 1) + ' (1 based) zone set to most common value ' + str(dominant))
