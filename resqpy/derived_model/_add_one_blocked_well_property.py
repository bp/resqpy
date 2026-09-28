"""High level add_one_blocked_well_property() function."""

import os

import resqpy.model as rq
import resqpy.property as rqp
import resqpy.well as rqw


def add_one_blocked_well_property(epc_file,
                                  a,
                                  property_kind,
                                  blocked_well_uuid,
                                  source_info = 'imported',
                                  title = None,
                                  discrete = False,
                                  uom = None,
                                  time_index = None,
                                  time_series_uuid = None,
                                  string_lookup_uuid = None,
                                  null_value = None,
                                  indexable_element = 'cells',
                                  facet_type = None,
                                  facet = None,
                                  realization = None,
                                  local_property_kind_uuid = None,
                                  count_per_element = 1,
                                  points = False,
                                  extra_metadata = {},
                                  new_epc_file = None):
    """Adds a blocked well property from a numpy array to an existing resqml dataset.

    arguments:
       epc_file (string): file name to load model resqml model from (and rewrite to if new_epc_file is None)
       a (1D numpy array): the blocked well property array to be added to the model
       property_kind (string): the resqml property kind
       blocked_well_uuid (uuid object or string): the uuid of the blocked well to which the property relates
       source_info (string): typically the name of a file from which the array has been read but can be any
          information regarding the source of the data
       title (string): this will be used as the citation title when a part is generated for the array
       discrete (boolean, default False): if True, the array should contain integer (or boolean) data; if False, float
       uom (string, default None): the resqml units of measure for the data; not relevant to discrete data
       time_index (integer, default None): if not None, the time index to be used when creating a part for the array
       time_series_uuid (uuid object or string, default None): required if time_index is not None
       string_lookup_uuid (uuid object or string, optional): required if the array is to be stored as a categorical
          property; set to None for non-categorical discrete data; only relevant if discrete is True
       null_value (int, default None): if present, this is used in the metadata to indicate that this value
          is to be interpreted as a null value wherever it appears in the data (use for discrete data only)
       indexable_element (string, default 'cells'): the indexable element in the supporting representation (the blocked well);
          valid values are 'cells', 'intervals' (which includes unblocked intervals), or 'nodes'
       facet_type (string): resqml facet type, or None
       facet (string): resqml facet, or None
       realization (int): realization number, or None
       local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
       count_per_element (int, default 1): the number of values per indexable element; if greater than one then this
          must be the fastest cycling axis in the cached array, ie last index; if greater than 1 then a must be a 2D array
       points (bool, default False): if True, this is a points property with an extra dimension of extent 3
       extra_metadata (dict, optional): any items in this dictionary are added as extra metadata to the new
          property
       new_epc_file (string, optional): if None, the source epc_file is extended with the new property object; if present,
          a new epc file (& associated h5 file) is created to contain a copy of the blocked well (and dependencies) and
          the new property

    returns:
       uuid.UUID of newly created property object
    """

    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None

    # open up model and establish grid object
    model = rq.Model(epc_file)

    blocked_well = rqw.BlockedWell(model, uuid = blocked_well_uuid)
    assert blocked_well is not None, f'no blocked well object found with uuid {blocked_well_uuid}'

    if not discrete:
        string_lookup_uuid = None

    # create an empty property collection and add the new array to its 'imported' list
    bwpc = rqp.PropertyCollection()
    bwpc.set_support(support = blocked_well, model = model)
    bwpc.add_cached_array_to_imported_list(a,
                                           source_info,
                                           title,
                                           discrete = discrete,
                                           uom = uom,
                                           time_index = time_index,
                                           null_value = null_value,
                                           property_kind = property_kind,
                                           local_property_kind_uuid = local_property_kind_uuid,
                                           facet_type = facet_type,
                                           facet = facet,
                                           realization = realization,
                                           indexable_element = indexable_element,
                                           count = count_per_element,
                                           points = points)
    bwpc.write_hdf5_for_imported_list()
    uuid_list = bwpc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = time_series_uuid,
                                                                         string_lookup_uuid = string_lookup_uuid,
                                                                         property_kind_uuid = local_property_kind_uuid,
                                                                         extra_metadata = extra_metadata)
    assert len(uuid_list) == 1
    model.store_epc()
    return uuid_list[0]
