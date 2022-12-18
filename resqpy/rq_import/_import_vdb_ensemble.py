"""_import_vdb_ensemble.py: Module to import a vdb ensemble into resqml format."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import resqpy.model as rq
import resqpy.olio.vdb as vdb
import resqpy.olio.xml_et as rqet
import resqpy.property as rp
import resqpy.rq_import as rqi
import resqpy.time_series as rts


def import_vdb_ensemble(
        epc_file,
        ensemble_run_dir,
        existing_epc = False,
        keyword_list = None,
        property_kind_list = None,
        vdb_static_properties = True,  # if True, static vdb properties are imported
        vdb_recurrent_properties = True,
        decoarsen = True,
        timestep_selection = 'all',
        create_property_set_per_realization = True,
        create_property_set_per_timestep = True,
        create_complete_property_set = False,
        # remaining arguments only used if existing_epc is False
        extent_ijk = None,  # 3 element numpy vector
        corp_xy_units = 'm',
        corp_z_units = 'm',
        corp_z_inc_down = True,
        ijk_handedness = 'right',
        geometry_defined_everywhere = True,
        treat_as_nan = None,
        resqml_xy_units = 'm',
        resqml_z_units = 'm',
        resqml_z_inc_down = True,
        shift_to_local = True,
        local_origin_place = 'centre',  # 'centre' or 'minimum'
        max_z_void = 0.1,  # import will fail if vertical void greater than this is encountered
        split_pillars = True,
        split_tolerance = 0.01,  # applies to each of x, y, z differences
        progress_fn = None):
    """Adds properties from all vdb's within an ensemble directory tree to a single RESQML dataset.

    Referencing a shared grid.

    args:
       epc_file (string): filename of epc file to be extended with ensemble properties
       ensemble_run_dir (string): path of main ensemble run directory; vdb's within this directory tree are source of import
       existing_epc (boolean, default False): if True, the epc_file must already exist and contain the compatible grid
       keyword_list (list of strings, optional): if present, only properties for keywords within the list are included
       property_kind_list (list of strings, optional): if present, only properties which are mapped to these resqml property
          kinds are included in the import
       vdb_static_properties (boolean, default True): if False, no static properties are included, regardless of keyword and/or
          property kind matches
       vdb_recurrent_properties (boolean, default True): if False, no recurrent properties are included, regardless of keyword
          and/or property kind matches
       decoarsen (boolean, default True): if True and ICOARSE property exists for a grid in a case, the associated property
          data is decoarsened; if False, the property data is as stored in the vdb
       timestep_selection (string, default 'all'): may be 'first', 'last', 'first and last', or 'all', controlling which
          reporting timesteps are included when loading recurrent data
       create_property_set_per_realization (boolean, default True): if True, a property set object is created for each realization
       create_property_set_per_timestep (boolean, default True): if True, a property set object is created for each timestep
          included in the recurrent data import
       create_complete_property_set (boolean, default False): if True, a property set object is created containing all the
          properties imported; only really useful to differentiate from other properties related to the grid
       extent_ijk (triple int, optional): this and remaining arguments are only used if existing_epc is False; the extent
          is only needed in case automatic determination of the extent fails
       corp_xy_units (string, default 'm'): the units of x & y values in the vdb corp data;
          typically 'm' (metres), 'ft' (feet) or 'cm' (centimetres, for lab scale models)
       corp_z_units (string, default 'm'): the units of z values in the vdb corp data;
          typically 'm' (metres), 'ft' (feet) or 'cm' (centimetres, for lab scale models)
       corp_z_inc_down (boolean, default True): set to True if corp z values are depth; False if elevation
       ijk_handedness (string, default 'right'): set to the handedness of the IJK axes in the Nexus model; 'right' or 'left'
       geometry_defined_everywhere (boolean, default True): set to False if inactive cells do not have valid geometry;
          deprecated - use treat_as_nan argument instead
       treat_as_nan (string, optional): if not None, one of 'dots', 'ij_dots', 'inactive'; controls which inactive cells
          have their geometry set to undefined
       resqml_xy_units (string, default 'm'): the units of x & y values to use in the generated resqml grid;
          typically 'm' (metres), 'ft' (feet) or 'cm' (centimetres, for lab scale models)
       resqml_z_units (string, default 'm'): the units of z values to use in the generated resqml grid;
          typically 'm' (metres), 'ft' (feet) or 'cm' (centimetres, for lab scale models)
       resqml_z_inc_down (boolean, default True): set to True if resqml z values are to be depth; False for elevations
       shift_to_local (boolean, default True): if True, the resqml coordinate reference system will use a local origin
       local_origin_place (string, default 'centre'): where to place the local origin; 'centre' or 'minimum'; only
          relevant if shift_to_local is True
       max_z_void (float, default 0.1): the tolerance of voids between layers, in z direction; voids greater than this
          will cause the grid import to fail
       split_pillars (boolean, default True): if False, a grid is generated without split pillars
       split_tolerance (float, default 0.01): the tolerance applied to each of x, y, & z values, beyond which a corner
          point (and hence pillar) will be split
       progress_fn (function(float), optional): if present, this function is called at intervals during processing; it
          must accept one floating point argument which will range from 0.0 to 1.0

    returns:
       resqpy.Model object containing properties for all the realisations; hdf5 and epc files having been updated

    note:
       if existing_epc is True, the epc file must already exist and contain one grid (or one grid named ROOT) which must
       have the correct extent for all realisations within the ensemble; if existing_epc is False, the resqml dataset is
       created afresh with a grid extracted from the first realisation in the ensemble; either way, the single grid is used
       as the representative grid in the ensemble resqml dataset being generated;
       all vdb directories within the directory tree headed by ensemble_run_dir are included in the import; by
       default all properties will be imported; the keyword_list, property_kind_list, vdb_static_properties,
       vdb_recurrent_properties and timestep_selection arguments can be used to filter the required properties;
       if both keyword_list and property_kind_list are provided, a property must match an item in both lists in order
       to be included; if recurrent properties are being included then all vdb's should contain the same number of reporting
       steps in their recurrent data and these should relate to the same set of timestamps; timestamp data is extracted from a
       summary file for the first realisation; no check is made to ensure that reporting timesteps in different realisations
       are actually for the same date.
    """

    assert epc_file.endswith('.epc')
    assert vdb_static_properties or vdb_recurrent_properties, 'no properties selected for ensemble import'

    if progress_fn is not None:
        progress_fn(0.0)

    # fetch a sorted list of the vdb paths found in the run directory tree
    ensemble_list = vdb.ensemble_vdb_list(ensemble_run_dir)
    if len(ensemble_list) == 0:
        log.error("no vdb's found in run directory tree: " + str(ensemble_run_dir))
        return None

    if not existing_epc:
        model = rqi.import_nexus(
            epc_file[:-4],  # output path and file name without .epc or .h5 extension
            extent_ijk = extent_ijk,  # 3 element numpy vector, in case extent is not automatically determined
            vdb_file = ensemble_list[0],  # vdb input file
            corp_xy_units = corp_xy_units,
            corp_z_units = corp_z_units,
            corp_z_inc_down = corp_z_inc_down,
            ijk_handedness = ijk_handedness,
            geometry_defined_everywhere = geometry_defined_everywhere,
            treat_as_nan = treat_as_nan,
            resqml_xy_units = resqml_xy_units,
            resqml_z_units = resqml_z_units,
            resqml_z_inc_down = resqml_z_inc_down,
            shift_to_local = shift_to_local,
            local_origin_place = local_origin_place,  # 'centre' or 'minimum'
            max_z_void = max_z_void,  # import will fail if vertical void greater than this is encountered
            split_pillars = split_pillars,
            split_tolerance = split_tolerance,  # applies to each of x, y, z differences
            vdb_static_properties = False,
            vdb_recurrent_properties = False,
            create_property_set = False)

    model = rq.Model(
        epc_file = epc_file)  # shouldn't be necessary if just created but it feels safer to re-open the model
    assert model is not None, 'failed to instantiate model'
    grid = model.grid()
    assert grid is not None, 'grid not found'
    ext_uuid = model.h5_uuid()
    assert ext_uuid is not None, 'failed to determine uuid for hdf5 file reference'
    hdf5_file = model.h5_file_name(uuid = ext_uuid)

    # create reporting timestep time series for recurrent data, if required, based on the first realisation
    recur_time_series = None
    recur_ts_uuid = None
    timestep_list = None
    if vdb_recurrent_properties:
        summary_file = ensemble_list[0][:-4] + '.sum'  # TODO: check timestep summary file extension, .tssum?
        full_time_series = rts.time_series_from_nexus_summary(summary_file)
        if full_time_series is None:
            log.error('failed to extract info from timestep summary file; disabling recurrent property import')
            vdb_recurrent_properties = False
    if vdb_recurrent_properties:
        vdbase = vdb.VDB(ensemble_list[0])
        timestep_list = vdbase.list_of_timesteps()
        if len(timestep_list) == 0:
            log.warning(
                'no ROOT recurrent data found in vdb for first realisation; disabling recurrent property import')
            vdb_recurrent_properties = False
    if vdb_recurrent_properties:
        if timestep_selection == 'all' or ('first' in timestep_selection):
            fs_index = 0
        else:
            fs_index = -1
        first_stamp = full_time_series.timestamp(timestep_list[fs_index])
        if first_stamp is None:
            log.error('first timestamp number selected for import was not found in summary file: ' +
                      str(timestep_list[fs_index]))
            log.error('disabling recurrent property import')
            vdb_recurrent_properties = False
    if vdb_recurrent_properties:
        recur_time_series = rts.TimeSeries(model, first_timestamp = first_stamp)
        if timestep_selection == 'all':
            remaining_list = timestep_list[1:]
        elif timestep_selection == 'first and last':
            remaining_list = [timestep_list[-1]]
        else:
            remaining_list = []
        for timestep_number in remaining_list:
            stamp = full_time_series.timestamp(timestep_number)
            if stamp is None:
                log.error('timestamp number for which recurrent data exists was not found in summary file: ' +
                          str(timestep_number))
                log.error('disabling recurrent property import')
                vdb_recurrent_properties = False
                recur_time_series = None
                break
            recur_time_series.add_timestamp(stamp)
    if recur_time_series is not None:
        recur_ts_node = recur_time_series.create_xml(title = 'simulator recurrent array timestep series')
        recur_ts_uuid = rqet.uuid_for_part_root(recur_ts_node)
        model.time_series = recur_ts_node  # save as the primary time series for the model

    if create_complete_property_set or create_property_set_per_timestep:
        complete_collection = rp.GridPropertyCollection()
        complete_collection.set_grid(grid)
    else:
        complete_collection = None

    #  main loop over realisations

    for realisation in range(len(ensemble_list)):

        if progress_fn is not None:
            progress_fn(float(1 + realisation) / float(1 + len(ensemble_list)))

        vdb_file = ensemble_list[realisation]
        log.info('processing realisation ' + str(realisation) + ' from: ' + str(vdb_file))
        vdbase = vdb.VDB(vdb_file)
        #      case_list = vdbase.cases()
        #      assert len(case_list) > 0, 'no cases found in vdb: ' + str(vdb_file)
        #      if len(case_list) > 1: log.warning('more than one case found in vdb (using first): ' + str(vdb_file))
        #      vdb_case = case_list[0]
        #      vdbase.set_use_case(vdb_case)
        vdbase.set_extent_kji(grid.extent_kji)

        prop_import_collection = rp.GridPropertyCollection(realization = realisation)
        prop_import_collection.set_grid(grid)

        decoarsen_array = None
        if vdb_static_properties:
            props = vdbase.list_of_static_properties()
            if len(props) > 0:
                for keyword in props:
                    if keyword_list is not None and keyword not in keyword_list:
                        continue
                    prop_kind, facet_type, facet = rp.property_kind_and_facet_from_keyword(keyword)
                    if property_kind_list is not None and prop_kind not in property_kind_list and prop_kind not in [
                            'active', 'region initialization'
                    ]:
                        continue
                    prop_import_collection.import_vdb_static_property_to_cache(vdbase,
                                                                               keyword,
                                                                               realization = realisation,
                                                                               property_kind = prop_kind,
                                                                               facet_type = facet_type,
                                                                               facet = facet)
                if decoarsen:
                    decoarsen_array = prop_import_collection.decoarsen_imported_list()
                    if decoarsen_array is not None:
                        log.debug('static properties decoarsened for realisation ' + str(realisation))
                grid.write_hdf5_from_caches(hdf5_file,
                                            mode = 'a',
                                            geometry = False,
                                            imported_properties = prop_import_collection,
                                            write_active = False)
                prop_import_collection.remove_all_cached_arrays()

        if vdb_recurrent_properties:

            r_timestep_list = vdbase.list_of_timesteps()  # get list of timesteps for which recurrent files exist
            if len(r_timestep_list) < recur_time_series.number_of_timestamps():
                log.error('insufficient number of reporting timesteps; skipping recurrent data for realisation ' +
                          str(realisation))
            else:
                common_recur_prop_set = None
                for tni in range(recur_time_series.number_of_timestamps()):
                    if timestep_selection in ['all', 'first']:
                        timestep_number = timestep_list[tni]
                        r_timestep_number = r_timestep_list[tni]
                    elif timestep_selection == 'last' or tni > 0:
                        timestep_number = timestep_list[-1]
                        r_timestep_number = r_timestep_list[-1]
                    else:
                        timestep_number = timestep_list[0]
                        r_timestep_number = r_timestep_list[0]
                    stamp = full_time_series.timestamp(timestep_number)
                    recur_prop_list = vdbase.list_of_recurrent_properties(r_timestep_number)
                    if common_recur_prop_set is None:
                        common_recur_prop_set = set(recur_prop_list)
                    elif recur_prop_list is not None:
                        common_recur_prop_set = common_recur_prop_set.intersection(set(recur_prop_list))
                    step_import_collection = rp.GridPropertyCollection()
                    step_import_collection.set_grid(grid)
                    # for each property for this timestep, cache array and add to recur prop import collection for this time step
                    if recur_prop_list:
                        for keyword in recur_prop_list:
                            if not keyword or not keyword.isalnum():
                                continue
                            if keyword_list is not None and keyword not in keyword_list:
                                continue
                            prop_kind, facet_type, facet = rp.property_kind_and_facet_from_keyword(keyword)
                            if property_kind_list is not None and prop_kind not in property_kind_list:
                                continue
                            step_import_collection.import_vdb_recurrent_property_to_cache(
                                vdbase,
                                r_timestep_number,
                                keyword,
                                time_index = tni,  # index into recur_time_series
                                realization = realisation,
                                property_kind = prop_kind,
                                facet_type = facet_type,
                                facet = facet)
                    if decoarsen_array is not None:
                        step_import_collection.decoarsen_imported_list(decoarsen_array = decoarsen_array)
                    # extend hdf5 with cached arrays for this timestep
                    #         log.info('number of recurrent grid property arrays for timestep: ' + str(timestep_number) +
                    #                  ' is: ' + str(step_import_collection.number_of_imports()))
                    #         log.info('extending hdf5 file with recurrent properties for timestep: ' + str(timestep_number))
                    grid.write_hdf5_from_caches(hdf5_file,
                                                mode = 'a',
                                                geometry = False,
                                                imported_properties = step_import_collection,
                                                write_active = False)
                    # add imported list for this timestep to full imported list
                    prop_import_collection.inherit_imported_list_from_other_collection(step_import_collection)
                    #         log.debug('total number of property arrays after timestep: ' + str(timestep_number) +
                    #                   ' is: ' + str(prop_import_collection.number_of_imports()))
                    # remove cached copies of arrays
                    step_import_collection.remove_all_cached_arrays()

        if len(prop_import_collection.imported_list) == 0:
            log.warning('no properties imported for realisation ' + str(realisation))
            continue

        prop_import_collection.create_xml_for_imported_list_and_add_parts_to_model(ext_uuid,
                                                                                   time_series_uuid = recur_ts_uuid)

        if create_property_set_per_realization:
            prop_import_collection.create_property_set_xml('property set for realization ' + str(realisation))

        if complete_collection is not None:
            complete_collection.inherit_parts_from_other_collection(prop_import_collection)

    if complete_collection is not None:
        if create_property_set_per_timestep and recur_time_series is not None:
            for tni in range(recur_time_series.number_of_timestamps()):
                ts_collection = rp.selective_version_of_collection(complete_collection, time_index = tni)
                if ts_collection.number_of_parts() > 0:
                    ts_collection.create_property_set_xml('property set for time index ' + str(tni))
        if create_complete_property_set:
            complete_collection.create_property_set_xml('property set for ensemble vdb import')

    # mark model as modified (will already have happened anyway)
    model.set_modified()

    # rewrite epc file
    log.info('storing updated model in epc file ' + epc_file)
    model.store_epc(epc_file)

    if progress_fn is not None:
        progress_fn(1.0)

    # return updated resqml model
    return model
