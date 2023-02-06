"""High level add_one_grid_property_array() function."""

import os
import numpy as np

import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp

import resqpy.derived_model._common as rqdm_c


def add_one_grid_property_array(epc_file,
                                a,
                                property_kind,
                                grid_uuid = None,
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
                                const_value = None,
                                expand_const_arrays = False,
                                points = False,
                                extra_metadata = {},
                                use_int32 = True,
                                new_epc_file = None):
    """Adds a grid property from a numpy array to an existing resqml dataset.

    arguments:
       epc_file (string): file name to load resqml model from (and rewrite to if new_epc_file is None)
       a (3D numpy array): the property array to be added to the model; for a constant array set this None
          and use the const_value argument, otherwise this array is required
       property_kind (string): the resqml property kind
       grid_uuid (uuid object or string, optional): the uuid of the grid to which the property relates;
          if None, the property is attached to the 'main' grid
       source_info (string): typically the name of a file from which the array has been read but can be any
          information regarding the source of the data
       title (string): this will be used as the citation title when a part is generated for the array; for simulation
          models it is desirable to use the simulation keyword when appropriate
       discrete (boolean, default False): if True, the array should contain integer (or boolean) data; if False, float
       uom (string, default None): the resqml units of measure for the data; not relevant to discrete data
       time_index (integer, default None): if not None, the time index to be used when creating a part for the array
       time_series_uuid (uuid object or string, default None): required if time_index is not None
       string_lookup_uuid (uuid object or string, optional): required if the array is to be stored as a categorical
          property; set to None for non-categorical discrete data; only relevant if discrete is True
       null_value (int, default None): if present, this is used in the metadata to indicate that this value
          is to be interpreted as a null value wherever it appears in the data (use for discrete data only)
       indexable_element (string, default 'cells'): the indexable element in the supporting representation (the grid)
       facet_type (string): resqml facet type, or None
       facet (string): resqml facet, or None
       realization (int): realization number, or None
       local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
       count_per_element (int, default 1): the number of values per indexable element; if greater than one then this
          must be the fastest cycling axis in the cached array, ie last index
       const_value (float or int, optional): if present, a constant array is added 'filled' with this value, in which
          case argument a should be None
       expand_const_arrays (bool, default False): if True and a const_value is provided, a fully expanded array is
          added to the model instead of a const array
       points (bool, default False): if True, this is a points property with an extra dimension of extent 3
       extra_metadata (dict, optional): any items in this dictionary are added as extra metadata to the new
          property
       use_int32 (bool, default True): if True, and the array a has int64 bit elements, they are written as 32 bit data
          to hdf5; if False, 64 bit data is written in that situation
       new_epc_file (string, optional): if None, the source epc_file is extended with the new property object; if present,
          a new epc file (& associated h5 file) is created to contain a copy of the grid and the new property

    returns:
       uuid.UUID of newly created property object
    """

    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None

    # open up model and establish grid object
    model = rq.Model(epc_file)
    if grid_uuid is None:
        grid = model.grid()
        grid_uuid = grid.uuid
    else:
        grid = model.grid_for_uuid_from_grid_list(grid_uuid)
        if grid is None:
            grid = grr.any_grid(model, uuid = grid_uuid, find_properties = False)
    assert grid is not None, 'failed to establish grid object'

    if not discrete:
        string_lookup_uuid = None

    if const_value is not None and expand_const_arrays:
        assert count_per_element == 1 and not points, 'attempt to expand const array for non-standard shape'
        if isinstance(const_value, bool):
            dtype = bool
        elif discrete:
            dtype = int
        else:
            dtype = float
        a = np.full(grid.extent_kji, const_value, dtype = dtype)
        const_value = None

    # create an empty property collection and add the new array to its 'imported' list
    gpc = rqp.GridPropertyCollection()
    gpc.set_grid(grid)
    gpc.add_cached_array_to_imported_list(a,
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
                                          const_value = const_value,
                                          points = points)

    # write or re-write model
    model.h5_release()
    if new_epc_file:
        grid_title = rqet.citation_title_for_node(grid.root)
        uuid_list = rqdm_c._write_grid(new_epc_file,
                                       grid,
                                       property_collection = gpc,
                                       grid_title = grid_title,
                                       mode = 'w',
                                       time_series_uuid = time_series_uuid,
                                       string_lookup_uuid = string_lookup_uuid,
                                       extra_metadata = extra_metadata,
                                       use_int32 = use_int32)
    else:
        # add arrays to hdf5 file holding source grid geometry
        uuid_list = rqdm_c._write_grid(epc_file,
                                       grid,
                                       property_collection = gpc,
                                       mode = 'a',
                                       geometry = False,
                                       time_series_uuid = time_series_uuid,
                                       string_lookup_uuid = string_lookup_uuid,
                                       extra_metadata = extra_metadata,
                                       use_int32 = use_int32)

    if uuid_list is None or len(uuid_list) == 0:
        return None
    return uuid_list[0]
