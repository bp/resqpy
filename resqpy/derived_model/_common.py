"""Private common functions for derived model package."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp


def _displacement_properties(new_grid, old_grid):
    """Computes cell centre differences in x, y, & z, between old & new grids, and returns a collection of 3 properties."""

    displacement_collection = rqp.GridPropertyCollection()
    displacement_collection.set_grid(new_grid)
    old_grid.centre_point(cache_centre_array = True)
    new_grid.centre_point(cache_centre_array = True)
    displacement = new_grid.array_centre_point - old_grid.array_centre_point
    log.debug('displacement array shape: ' + str(displacement.shape))
    displacement_collection.x_array = displacement[..., 0].copy()
    displacement_collection.y_array = displacement[..., 1].copy()
    displacement_collection.z_array = displacement[..., 2].copy()
    # horizontal_displacement = np.sqrt(x_displacement * x_displacement  +  y_displacement * y_displacement)
    # todo: create prop collection to hold z_displacement and horizontal_displacement; add them to imported list
    xy_units = new_grid.xy_units()
    z_units = new_grid.z_units()
    # todo: could replace 3 displacement properties with a single points property
    displacement_collection.add_cached_array_to_imported_list(displacement_collection.x_array,
                                                              'easterly displacement from tilt',
                                                              'DX_DISPLACEMENT',
                                                              discrete = False,
                                                              uom = xy_units)
    displacement_collection.add_cached_array_to_imported_list(displacement_collection.y_array,
                                                              'northerly displacement from tilt',
                                                              'DY_DISPLACEMENT',
                                                              discrete = False,
                                                              uom = xy_units)
    displacement_collection.add_cached_array_to_imported_list(displacement_collection.z_array,
                                                              'vertical displacement from tilt',
                                                              'DZ_DISPLACEMENT',
                                                              discrete = False,
                                                              uom = z_units)
    return displacement_collection


def _pl(n, use_es = False):
    if n == 1:
        return ''
    elif use_es:
        return 'es'
    else:
        return 's'


def _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization, inherit_all_realizations):
    collection = None
    if inherit_properties:
        source_collection = source_grid.extract_property_collection()
        if source_collection is not None:
            #  do not inherit the inactive property array by this mechanism
            collection = rqp.GridPropertyCollection()
            collection.set_grid(grid)
            collection.extend_imported_list_copying_properties_from_other_grid_collection(
                source_collection, realization = inherit_realization, copy_all_realizations = inherit_all_realizations)
    return collection


def _write_grid(epc_file,
                grid,
                ext_uuid = None,
                property_collection = None,
                grid_title = None,
                mode = 'a',
                geometry = True,
                time_series_uuid = None,
                string_lookup_uuid = None,
                extra_metadata = {},
                use_int32 = None):
    """Append to or create epc and h5 files, with grid and optionally property collection.

    arguments:
       epc_file (string): name of existing epc file (if appending) or new epc file to be created (if writing)
       grid (grid.Grid object): the grid object to be written to the epc & h5 files
       ext_uuid (uuid.UUID object, optional): if present and the mode is 'a', the arrays are appended to
          the hdf5 file that has this uuid; if None or mode is 'w', the hdf5 file is determined automatically
       property_collection (property.GridPropertyCollection object, optional): if present, a collection of
          grid properties to write and relate to the grid
       grid_title (string): used as the citation title for the grid object
       mode (string, default 'a'): 'a' or 'w'; if 'a', epc_file should be an existing file which is extended
          (appended to) with the new grid and properties; if 'w', epc_file is created along with an h5 file
          and will be populated with the grid, crs and properties
       geometry (boolean, default True): if True, the grid object is included in the write; if False, only the
          property collection is written, in which case grid must be fully established with xml in place
       time_series_uuid (uuid.UUID, optional): the uuid of a time series object; required if property_collection
          contains any recurrent properties in its import list
       string_lookup_uuid (optional): if present, the uuid of the string table lookup which any non-continuous
          properties relate to (ie. they are all taken to be categorical); leave as None if discrete property
          objects are required rather than categorical
       extra_metadata (dict, optional): any items in this dictionary are added as extra metadata to any new
          properties
       use_int32 (bool, optional): if None, system default of True is used; if True, int64 property arrays are
          written to hdf5 as int32; if False, int64 data is written

    returns:
       list of uuid.UUID, being the uuids of property parts added from the property_collection, if any

    note:
       this function is not usually called directly by application code
    """

    log.debug('write_grid(): epc_file: ' + str(epc_file) + '; mode: ' + str(mode) + '; grid extent: ' +
              str(grid.extent_kji))

    assert mode in ['a', 'w']
    assert geometry or mode == 'a', 'for now, a copy of the grid must be included with property collection'
    assert geometry or property_collection is not None, 'write_grid called without anything to write'

    if not epc_file.endswith('.epc'):
        epc_file += '.epc'

    is_regular = grr.is_regular_grid(grid.root)

    if not is_regular:
        grid.cache_all_geometry_arrays()
    working_model = grid.model

    if mode == 'a':  # append to existing model
        log.debug('write_grid(): re-using existing model object')
        model = working_model
        if ext_uuid is None:
            ext_uuid = model.h5_uuid()
    else:  # create new model with crs, grid and properties
        log.debug('write_grid(): creating new model object')
        model = rq.new_model(epc_file)
        ext_uuid = model.h5_uuid()
        crs_root = model.duplicate_node(grid.model.root_for_uuid(grid.crs_uuid))
        grid.model = model
        grid.crs_uuid = model.uuid_for_root(crs_root)
        grid.crs = rqc.Crs(model, uuid = model.crs_uuid)
    log.debug('write_grid(): number of starting parts: ' + str(model.number_of_parts()))

    if grid.inactive is not None and geometry:
        inactive_count = np.count_nonzero(grid.inactive)
        if inactive_count == grid.inactive.size:
            log.warning('writing grid with all cells inactive')
        else:
            log.info(f'grid has {grid.inactive.size - inactive_count} active cells out of {grid.inactive.size}')
    collection = property_collection
    if collection is not None:
        log.debug('write_grid(): number of properties in collection: ' + str(collection.number_of_parts()))

    # append to hdf5 file using arrays cached in grid above
    hdf5_file = model.h5_file_name(uuid = ext_uuid, file_must_exist = (mode == 'a'))
    if geometry or collection is not None:
        log.debug('writing grid arrays to hdf5 file')
        grid.write_hdf5_from_caches(hdf5_file,
                                    mode = mode,
                                    geometry = geometry,
                                    imported_properties = collection,
                                    use_int32 = use_int32)
        model.h5_release()
    if ext_uuid is None:
        ext_uuid = model.h5_uuid()

    # build xml for grid geometry
    if geometry:
        log.debug('building xml for grid object')
        ijk_node = grid.create_xml(ext_uuid, add_as_part = True, add_relationships = True, title = grid_title)
        assert ijk_node is not None, 'failed to create IjkGrid node in xml tree'
        if collection is not None:
            collection.set_grid(grid, grid_root = grid.root)
        grid.geometry_root = rqet.find_tag(ijk_node, 'Geometry')

    # add derived inactive array as part
    if collection is not None:
        prop_uuid_list = collection.create_xml_for_imported_list_and_add_parts_to_model(
            ext_uuid,
            support_uuid = grid.uuid,
            time_series_uuid = time_series_uuid,
            string_lookup_uuid = string_lookup_uuid,
            extra_metadata = extra_metadata)
    else:
        prop_uuid_list = []

    # set grid related short cuts in Model object
    if geometry:
        model.grid_list.append(grid)

    log.debug('write_grid(): number of finishing parts: ' + str(model.number_of_parts()))

    # store new version of model
    log.info('writing model xml data in epc file: ' + epc_file)
    model.store_epc(epc_file)

    return prop_uuid_list


def _establish_model_and_source_grid(epc_file, source_grid):
    assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
    if epc_file:
        model = rq.Model(epc_file)
        if source_grid is None:
            source_grid = model.grid()  # requires there to be exactly one grid in model (or one named ROOT)
    else:
        model = source_grid.model
    return model, source_grid
