"""_add_ab_properties.py: Module to add binary grid properties to an existing RESQMl grid object."""

import logging

log = logging.getLogger(__name__)

import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property as rp


def add_ab_properties(epc_file, grid_uuid = None, ext_uuid = None, ab_property_list = None):
    """Import a list of pure binary property array files as grid properties.

    arguments:
        epc_file (str): path of existing resqml epc to be added to
        grid_uuid (UUID, optional): the uuid of the grid to receive the properties; required if more than one grid present
        ext_uuid (UUID, optional): the uuid of the hdf5 extension part to use for the arrays; recommended to leave as None
        ab_property_list (list of tuples): each entry contains:
            (file_name, keyword, property_kind, facet_type, facet, uom, time_index, null_value, discrete, realization)

    returns:
        Model, with the new properties added, with hdf5 and epc fully updated
    """

    assert ab_property_list, 'property list is empty or missing'

    model = rq.Model(epc_file = epc_file)
    if grid_uuid is None:
        grid_node = model.root_for_ijk_grid()  # will raise an exception if Model has more than 1 grid
        assert grid_node is not None, 'grid not found in model'
        grid_uuid = rqet.uuid_for_part_root(grid_node)
    grid = grr.any_grid(parent_model = model, uuid = grid_uuid, find_properties = False)

    if ext_uuid is None:
        ext_node = rqet.find_nested_tags(grid.geometry_root, ['Points', 'Coordinates', 'HdfProxy', 'UUID'])
        if ext_node is not None:
            ext_uuid = bu.uuid_from_string(ext_node.text.strip())

    #  ab_property_list: list of (filename, keyword, property_kind, facet_type, facet, uom, time_index, null_value, discrete, realization)
    prop_import_collection = rp.GridPropertyCollection()
    prop_import_collection.set_grid(grid)
    for (p_filename, p_keyword, p_property_kind, p_facet_type, p_facet, p_uom, p_time_index, p_null_value, p_discrete,
         p_realization) in ab_property_list:
        prop_import_collection.import_ab_property_to_cache(p_filename,
                                                           p_keyword,
                                                           grid.extent_kji,
                                                           discrete = p_discrete,
                                                           uom = p_uom,
                                                           time_index = p_time_index,
                                                           null_value = p_null_value,
                                                           property_kind = p_property_kind,
                                                           facet_type = p_facet_type,
                                                           facet = p_facet,
                                                           realization = p_realization)
        # todo: property_kind, facet_type & facet are not currently getting passed through the imported_list tuple in resqml_property

    if prop_import_collection is None:
        log.warning('no pure binary grid properties to import')
    else:
        log.info('number of pure binary grid property arrays: ' + str(prop_import_collection.number_of_imports()))

    # append to hdf5 file using arrays cached in grid property collection above
    hdf5_file = model.h5_file_name()
    log.debug('appending to hdf5 file: ' + hdf5_file)
    grid.write_hdf5_from_caches(hdf5_file,
                                mode = 'a',
                                geometry = False,
                                imported_properties = prop_import_collection,
                                write_active = False)
    # remove cached static property arrays from memory
    if prop_import_collection is not None:
        prop_import_collection.remove_all_cached_arrays()

    # add imported properties parts to model, building property parts list
    if prop_import_collection is not None and prop_import_collection.imported_list is not None:
        prop_import_collection.create_xml_for_imported_list_and_add_parts_to_model(ext_uuid)

    # mark model as modified
    model.set_modified()

    # store new version of model
    log.info('storing model with additional properties in epc file: ' + epc_file)
    model.store_epc(epc_file)

    return model
