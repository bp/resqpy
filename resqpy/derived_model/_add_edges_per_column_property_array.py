"""High level add edges per column property array function."""

import resqpy.derived_model
import resqpy.property as rqp

import resqpy.derived_model._add_one_grid_property_array as rqdm_aogp


def add_edges_per_column_property_array(epc_file,
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
                                        facet_type = None,
                                        facet = None,
                                        realization = None,
                                        local_property_kind_uuid = None,
                                        extra_metadata = {},
                                        new_epc_file = None):
    """Adds an edges per column grid property from a numpy array to an existing resqml dataset.

    arguments:
       epc_file (string): file name to load model resqml model from (and rewrite to if new_epc_file is None)
       a (3D numpy array): the property array to be added to the model; expected shape (nj,ni,2,2) or (nj,ni,4)
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
       facet_type (string): resqml facet type, or None
       facet (string): resqml facet, or None
       realization (int): realization number, or None
       local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
       extra_metadata (dict, optional): any items in this dictionary are added as extra metadata to the new
          property
       new_epc_file (string, optional): if None, the source epc_file is extended with the new property object; if present,
          a new epc file (& associated h5 file) is created to contain a copy of the grid and the new property

    returns:
       uuid.UUID - the uuid of the newly created property

    notes:
       the RESQML protocol for saving edges per column properties uses a clockwise ordering of the 4 edges
       of a column; the resqpy protocol uses 2 dimensions of extent 2, being the axis (J, I) and face (-, +);
       this function assumes the array is in RESQML protocol if it has shape (nj, ni, 4) and resqpy protocol
       if it has shape (nj, ni, 2, 2); when reloading the property it will be presented in RESQML protocol;
       calling code can use property module functions reformat_column_edges_from_resqml_format() and
       reformat_column_edges_to_resqml_format() to convert between the protocols if needed
    """

    assert a.ndim in [3, 4]
    if a.ndim == 4:  # resqpy protocol
        assert a.shape[2] == 2 and a.shape[3] == 2, 'Wrong shape! Expected shape (nj, ni, 2, 2)'
        array_rq = rqp.reformat_column_edges_to_resqml_format(a)
    else:  # RESQML protocol
        assert a.shape[2] == 4, 'Wrong shape! Expected shape (nj, ni, 4)'
        array_rq = a

    property_uuid = rqdm_aogp.add_one_grid_property_array(epc_file,
                                                          array_rq,
                                                          property_kind,
                                                          grid_uuid = grid_uuid,
                                                          source_info = source_info,
                                                          title = title,
                                                          discrete = discrete,
                                                          uom = uom,
                                                          time_index = time_index,
                                                          time_series_uuid = time_series_uuid,
                                                          string_lookup_uuid = string_lookup_uuid,
                                                          null_value = null_value,
                                                          indexable_element = 'edges per column',
                                                          facet_type = facet_type,
                                                          facet = facet,
                                                          realization = realization,
                                                          local_property_kind_uuid = local_property_kind_uuid,
                                                          count_per_element = 1,
                                                          extra_metadata = extra_metadata,
                                                          new_epc_file = new_epc_file)
    return property_uuid
