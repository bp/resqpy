"""_import_nexus.py: Module to import a nexus corp grid & properties, or vdb, or vdb ensemble into resqml format."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import glob
import os
import numpy as np

import resqpy.crs as rqc
import resqpy.model as rq
import resqpy.olio.ab_toolbox as abt
import resqpy.olio.load_data as ld
import resqpy.olio.trademark as tm
import resqpy.olio.vdb as vdb
import resqpy.olio.write_data as wd
import resqpy.olio.xml_et as rqet
import resqpy.property as rp
import resqpy.rq_import as rqi
import resqpy.time_series as rts
import resqpy.weights_and_measures as bwam


def import_nexus(
        resqml_file_root,  # output path and file name without .epc or .h5 extension
        extent_ijk = None,  # 3 element numpy vector
        vdb_file = None,  # vdb input file: either this or corp_file should be not None
        vdb_case = None,  # if None, first case in vdb is used (usually a vdb only holds one case)
        corp_file = None,  # corp ascii input file: nexus corp data without keyword
        corp_bin_file = None,  # corp binary file: nexus corp data in bespoke binary format
        corp_xy_units = 'm',
        corp_z_units = 'm',
        corp_z_inc_down = True,
        ijk_handedness = 'right',
        corp_eight_mode = False,
        geometry_defined_everywhere = True,
        treat_as_nan = None,
        active_mask_file = None,
        use_binary = False,  # this refers to pure binary arrays, not corp bin format
        resqml_xy_units = 'm',
        resqml_z_units = 'm',
        resqml_z_inc_down = True,
        shift_to_local = False,
        local_origin_place = 'centre',  # 'centre' or 'minimum'
        max_z_void = 0.1,  # vertical gaps greater than this will introduce k gaps intp resqml grid
        split_pillars = True,
        split_tolerance = 0.01,  # applies to each of x, y, z differences
        property_array_files = None,  # actually, list of (filename, keyword, uom, time_index, null_value, discrete)
        summary_file = None,  # used to extract timestep dates when loading recurrent data from vdb
        vdb_static_properties = True,
        # if True, static vdb properties are imported (only relevant if vdb_file is not None)
        vdb_recurrent_properties = False,
        timestep_selection = 'all',
        # 'first', 'last', 'first and last', 'all', or list of ints being reporting timestep numbers
        use_compressed_time_series = True,
        decoarsen = True,  # where ICOARSE is present, redistribute data to uncoarse cells
        ab_property_list = None,
        # list of (file_name, keyword, property_kind, facet_type, facet, uom, time_index, null_value, discrete)
        create_property_set = False,
        ensemble_case_dirs_root = None,  # path upto but excluding realisation number
        ensemble_property_dictionary = None,
        # dictionary mapping title (or keyword) to (filename, property_kind, facet_type, facet,
        #                                           uom, time_index, null_value, discrete)
        ensemble_size_limit = None,
        grid_title = 'ROOT',
        mode = 'w',
        progress_fn = None):
    """Read a simulation grid geometry and optionally grid properties.

    Input may be from nexus ascii input files, or nexus vdb output.

    Arguments:
        resqml_file_root (str): output path and file name without .epc or .h5 extension
        extent_ijk (triple float, optional): ijk extents (fortran ordering)
        vdb_file (str, optional): vdb input file, either this or corp_file should be not None. Required if importing from a vdb
        vdb_case (str, optional): required if the vdb contains more than one case. If None, first case in vdb is used
        corp_file (str, optional): required if importing from corp ascii file. corp ascii input file: nexus corp data without keyword
        corp_bin_file (str, optional): required if importing from corp binary file
        corp_xy_units (str, default 'm'): xy length units
        corp_z_units (str, default 'm'): z length units
        corp_z_inc_down (bool, default True): if True z values increase with depth
        ijk_handedness (str, default 'right'): 'right' or 'left'
        corp_eight_mode (bool, default False): if True the ordering of corner point data is in nexus EIGHT mode
        geometry_defined_everywhere (bool, default True): if False then inactive cells are marked as not having geometry
        treat_as_nan (float, default None): if a value is provided corner points with this value will be assigned nan
        active_mask_file (str, default None): ascii property file holding values 0 or 1, with 1 indicating active cells
        use_binary (bool, default False): if True a cached binary version of ascii files will be used (pure binary, not corp bin format)
        resqml_xy_units (str, default 'm'): output xy units for resqml file
        resqml_z_units (str, default 'm'): output z units for resqml file
        resqml_z_inc_down (bool, default True): if True z values increase with depth for output resqml file
        shift_to_local (bool, default False): if True then a local origin will be used in the CRS
        local_origin_place (str, default 'centre'): 'centre' or 'minimum'. If 'centre' the local origin is placed at the centre of the grid; ignored if shift_to_local is False
        max_z_void (float, default 0.1): maximum z gap between vertically neighbouring corner points. Vertical gaps greater than this will introduce k gaps into resqml grid. Units are corp z units
        split_pillars (bool, default True): if False an unfaulted grid will be generated
        split_tolerance (float, default 0.01): maximum distance between neighbouring corner points before a pillar is considered 'split'. Applies to each of x, y, z differences
        property_array_files (list, default None): list of (filename, keyword, uom, time_index, null_value, discrete)
        summary_file (str, default None): nexus output summary file, used to extract timestep dates when loading recurrent data from vdb
        vdb_static_properties (bool, default True): if True, static vdb properties are imported (only relevant if vdb_file is not None)
        vdb_recurrent_properties (bool, default False): # if True, recurrent vdb properties are imported (only relevant if vdb_file is not None)
        timestep_selection (str, default 'all): 'first', 'last', 'first and last', 'all', or list of ints being reporting timestep numbers. Ignored if vdb_recurrent_properties is False
        use_compressed_time_series (bool, default True): generates reduced time series containing timesteps with recurrent properties from vdb, rather than full nexus summary time series
        decoarsen (bool, default True): where ICOARSE is present, redistribute data to uncoarse cells
        ab_property_list (list, default None):  list of (file_name, keyword, property_kind, facet_type, facet, uom, time_index, null_value, discrete)
        create_property_set (bool, default False): if True a resqml PropertySet is created
        ensemble_case_dirs_root (str, default None): path up to but excluding realisation number
        ensemble_property_dictionary (str, default None): dictionary mapping title (or keyword) to (filename, property_kind, facet_type, facet, uom, time_index, null_value, discrete)
        ensemble_size_limit (int, default None): if present processing of ensemble will terminate after this number of cases is reached
        grid_title (str, default 'ROOT'): grid citation title
        mode (str, default 'w'): 'w' or 'a', mode to write or append to hdf5
        progress_fn (function, default None): if present function must have one floating argument with value increasing from 0 to 1, and is called at intervals to indicate progress

    Returns:
        resqml model in memory & written to disc
    """

    if resqml_file_root.endswith('.epc'):
        resqml_file_root = resqml_file_root[:-4]
    assert mode in ['w', 'a']

    if vdb_file:
        using_vdb = True
        corp_file = corp_bin_file = None
        grid_title = grid_title.upper()
        log.info('starting import of Nexus ' + str(grid_title) + ' corp from vdb ' + str(vdb_file))
        tm.log_nexus_tm('info')
        vdbase = vdb.VDB(vdb_file)
        case_list = vdbase.cases()
        assert len(case_list) > 0, 'no cases found in vdb'
        if vdb_case is None:
            vdb_case = case_list[0]
        else:
            assert vdb_case in case_list, 'case ' + vdb_case + ' not found in vdb: ' + vdb_file
            vdbase.set_use_case(vdb_case)
        assert grid_title in vdbase.list_of_grids(), 'grid ' + str(grid_title) + ' not found in vdb'
        if extent_ijk is not None:
            vdbase.set_extent_kji(tuple(reversed(extent_ijk)))
        log.debug('using case ' + vdb_case + ' and grid ' + grid_title + ' from vdb')
        if vdb_recurrent_properties and not summary_file:
            if vdb_file.endswith('.vdb.zip'):
                summary_file = vdb_file[:-8] + '.sum'
            elif vdb_file.endswith('.vdb') or vdb_file.endswith('.zip'):
                summary_file = vdb_file[:-4] + '.sum'
            else:
                sep = vdb_file.rfind(os.sep)
                dot = vdb_file[sep + 1:].find('.')
                if dot > 0:
                    summary_file = vdb_file[:sep + 1 + dot] + ',sum'
                else:
                    summary_file = vdb_file + '.sum'
        cp_array = vdbase.grid_corp(grid_title)
        cp_extent_kji = cp_array.shape[:3]
        if cp_extent_kji[:2] == (1, 1):  # auto determination of extent failed
            assert extent_ijk is not None, 'failed to determine extent of grid from corp data'
            (ni, nj, nk) = extent_ijk
            assert cp_extent_kji[2] == ni * nj * nk, 'number of cells in grid corp does not match extent'
            cp_extent = (nk, nj, ni, 2, 2, 2, 3)  # (nk, nj, ni, kp, jp, ip, xyz)
            cp_array = cp_array.reshape(cp_extent)
        elif extent_ijk is not None:
            for axis in range(3):
                assert cp_extent_kji[axis] == extent_ijk[
                    2 - axis], 'extent of grid corp data from vdb does not match that supplied'

    elif corp_file or corp_bin_file:
        if corp_bin_file:
            corp_file = None
        using_vdb = False
        #     geometry_defined_everywhere = (active_mask_file is None)
        log.info('starting import of Nexus corp file ' + str(corp_file if corp_file else corp_bin_file))
        tm.log_nexus_tm('info')
        if extent_ijk is None:  # auto detect extent
            extent_kji = None
            cp_extent = None
        else:
            (ni, nj, nk) = extent_ijk
            extent_kji = np.array((nk, nj, ni), dtype = 'int')
            cp_extent = (nk, nj, ni, 2, 2, 2, 3)  # (nk, nj, ni, kp, jp, ip, xyz)
        log.debug('reading and resequencing corp data')
        if corp_bin_file:  # bespoke nexus corp bin format, not to be confused with pure binary files used below
            cp_array = ld.load_corp_array_from_file(
                corp_bin_file,
                extent_kji,
                corp_bin = True,
                comment_char = None,  # comment char will be detected automatically
                data_free_of_comments = False,
                use_binary = use_binary)
        else:
            cp_binary_file = abt.cp_binary_filename(
                corp_file, nexus_ordering = False)  # pure binary, not bespoke corp bin used above
            recent_binary_exists = ld.file_exists(cp_binary_file, must_be_more_recent_than_file = corp_file)
            cp_array = None
            if use_binary and (extent_ijk is not None) and recent_binary_exists:
                try:
                    cp_array = ld.load_array_from_file(cp_binary_file, cp_extent, use_binary = True)
                except Exception:
                    cp_array = None
            if cp_array is None:
                cp_array = ld.load_corp_array_from_file(
                    corp_file,
                    extent_kji,
                    corp_bin = False,
                    comment_char = None,  # comment char will be detected automatically
                    data_free_of_comments = False,
                    use_binary = use_binary)
                if use_binary:
                    wd.write_pure_binary_data(cp_binary_file,
                                              cp_array)  # NB: this binary file is resequenced, not in nexus ordering!

    else:
        raise ValueError('vdb_file and corp_file are both None in import_nexus() call')

    if cp_array is None:
        log.error('failed to create corner point array')
        return None

    if extent_ijk is None:
        cp_extent = cp_array.shape
        extent_kji = cp_extent[:3]
        (nk, nj, ni) = extent_kji
        extent_ijk = (ni, nj, nk)
    else:
        ni, nj, nk = extent_ijk

    # convert units
    log.debug('Converting units')
    if corp_xy_units == corp_z_units and resqml_xy_units == resqml_z_units:
        bwam.convert_lengths(cp_array, corp_xy_units, resqml_xy_units)
    else:
        bwam.convert_lengths(cp_array[:, :, :, :, :, :, 0:1], corp_xy_units, resqml_xy_units)
        bwam.convert_lengths(cp_array[:, :, :, :, :, :, 2], corp_z_units, resqml_z_units)

    # invert z if required
    if resqml_z_inc_down != corp_z_inc_down:
        log.debug('Inverting z values')
        inversion = np.negative(cp_array[:, :, :, :, :, :, 2])
        cp_array[:, :, :, :, :, :, 2] = inversion

    # read active cell mask
    log.debug('Setting up active cell mask')
    active_mask = inactive_mask = None
    if vdb_file:
        assert vdbase is not None, 'problem with vdb object'
        inactive_mask = vdbase.grid_kid_inactive_mask(grid_title)  # TODO: check conversion of KID to boolean for LGRs
        if inactive_mask is not None:
            log.debug('using kid array as inactive cell mask')
            active_mask = np.logical_not(inactive_mask)
        else:
            log.warning('kid array not found, using unpack array as active cell indicator')
            unp = vdbase.grid_unpack(grid_title)
            assert unp is not None, 'failed to load active cell indicator mask from vdb kid or unpack arrays'
            active_mask = np.empty((nk, nj, ni), dtype = 'bool')
            active_mask[:] = (unp > 0)
            inactive_mask = np.logical_not(active_mask)
    elif active_mask_file:
        active_mask = ld.load_array_from_file(active_mask_file, extent_kji, data_type = 'bool', use_binary = use_binary)
        if active_mask is None:
            log.error('failed to load active cell indicator array from file: ' + active_mask_file)
        else:
            inactive_mask = np.logical_not(active_mask)  # will crash if active mask load failed

    # shift grid geometry to local crs
    local_origin = np.zeros(3)
    if shift_to_local:
        log.debug('shifting to local origin at ' + local_origin_place)
        if local_origin_place == 'centre':
            local_origin = np.nanmean(cp_array, axis = (0, 1, 2, 3, 4, 5))
        elif local_origin_place == 'minimum':
            local_origin = np.nanmin(cp_array, axis = (0, 1, 2, 3, 4, 5)) - 1.0  # The -1 ensures all coords are >0
        else:
            assert (False)
        cp_array -= local_origin

    # create empty resqml model
    log.debug('creating an empty resqml model')
    if mode == 'w':
        model = rq.Model(resqml_file_root, new_epc = True, create_basics = True, create_hdf5_ext = True)
    else:
        model = rq.Model(resqml_file_root)
    assert model is not None
    ext_uuid = model.h5_uuid()
    assert ext_uuid is not None

    # create coodinate reference system (crs) in model and set references in grid object
    log.debug('creating coordinate reference system')
    crs_uuids = model.uuids(obj_type = 'LocalDepth3dCrs')
    new_crs = rqc.Crs(model,
                      x_offset = local_origin[0],
                      y_offset = local_origin[1],
                      z_offset = local_origin[2],
                      xy_units = resqml_xy_units,
                      z_units = resqml_z_units,
                      z_inc_down = resqml_z_inc_down)
    new_crs.create_xml(reuse = True)
    crs_uuid = new_crs.uuid

    grid = rqi.grid_from_cp(model,
                            cp_array,
                            crs_uuid,
                            active_mask = active_mask,
                            geometry_defined_everywhere = geometry_defined_everywhere,
                            treat_as_nan = treat_as_nan,
                            max_z_void = max_z_void,
                            split_pillars = split_pillars,
                            split_tolerance = split_tolerance,
                            ijk_handedness = ijk_handedness,
                            known_to_be_straight = False)

    # create hdf5 file using arrays cached in grid above
    log.info('writing grid geometry to hdf5 file ' + resqml_file_root + '.h5')
    grid.write_hdf5_from_caches(resqml_file_root + '.h5', mode = mode, write_active = False)

    # build xml for grid geometry
    log.debug('building xml for grid')
    ijk_node = grid.create_xml(ext_uuid = None, title = grid_title, add_as_part = True, add_relationships = True)
    assert ijk_node is not None, 'failed to create IjkGrid node in xml tree'

    # impprt property arrays into a collection
    prop_import_collection = None
    decoarsen_array = None
    ts_node = None
    ts_uuid = None

    if active_mask is None and grid.inactive is not None:
        active_mask = np.logical_not(grid.inactive)

    if using_vdb:
        prop_import_collection = rp.GridPropertyCollection()
        if vdb_static_properties:
            props = vdbase.grid_list_of_static_properties(grid_title)
            if len(props) > 0:
                prop_import_collection = rp.GridPropertyCollection()
                prop_import_collection.set_grid(grid)
                for keyword in props:
                    prop_import_collection.import_vdb_static_property_to_cache(vdbase, keyword, grid_name = grid_title)
    #      if active_mask is not None:
    #         prop_import_collection.add_cached_array_to_imported_list(active_mask, active_mask_file, 'ACTIVE', property_kind = 'active',
    #                                                                  discrete = True, uom = None, time_index = None, null_value = None)

    elif property_array_files is not None and len(property_array_files) > 0:
        prop_import_collection = rp.GridPropertyCollection()
        prop_import_collection.set_grid(grid)
        for (p_filename, p_keyword, p_uom, p_time_index, p_null_value, p_discrete) in property_array_files:
            prop_import_collection.import_nexus_property_to_cache(p_filename,
                                                                  p_keyword,
                                                                  grid.extent_kji,
                                                                  discrete = p_discrete,
                                                                  uom = p_uom,
                                                                  time_index = p_time_index,
                                                                  null_value = p_null_value,
                                                                  use_binary = use_binary)
    #      if active_mask is not None:
    #         prop_import_collection.add_cached_array_to_imported_list(active_mask, active_mask_file, 'ACTIVE', property_kind = 'active',
    #                                                                  discrete = True, uom = None, time_index = None, null_value = None)

    #  ab_property_list: list of (filename, keyword, property_kind, facet_type, facet, uom, time_index, null_value, discrete)
    elif ab_property_list is not None and len(ab_property_list) > 0:
        prop_import_collection = rp.GridPropertyCollection()
        prop_import_collection.set_grid(grid)
        for (p_filename, p_keyword, p_property_kind, p_facet_type, p_facet, p_uom, p_time_index, p_null_value,
             p_discrete) in ab_property_list:
            prop_import_collection.import_ab_property_to_cache(p_filename,
                                                               p_keyword,
                                                               grid.extent_kji,
                                                               discrete = p_discrete,
                                                               property_kind = p_property_kind,
                                                               facet_type = p_facet_type,
                                                               facet = p_facet,
                                                               uom = p_uom,
                                                               time_index = p_time_index,
                                                               null_value = p_null_value)
    #      if active_mask is not None:
    #         prop_import_collection.add_cached_array_to_imported_list(active_mask, active_mask_file, 'ACTIVE', property_kind = 'active',
    #                                                                  discrete = True, uom = None, time_index = None, null_value = None)

    # ensemble_property_dictionary: mapping title (or keyword) to
    #    (filename, property_kind, facet_type, facet, uom, time_index, null_value, discrete)
    elif ensemble_case_dirs_root and ensemble_property_dictionary:
        case_path_list = glob.glob(ensemble_case_dirs_root + '*')
        assert len(case_path_list) > 0, 'no case directories found with path starting: ' + str(ensemble_case_dirs_root)
        case_number_place = len(ensemble_case_dirs_root)
        case_zero_used = False
        case_count = 0
        for case_path in case_path_list:
            if ensemble_size_limit is not None and case_count >= ensemble_size_limit:
                log.warning('stopping after reaching ensemble size limit')
                break
            # NB. import each case individually rather than holding property arrays for whole ensemble in memory at once
            prop_import_collection = rp.GridPropertyCollection()
            prop_import_collection.set_grid(grid)
            tail = case_path[case_number_place:]
            try:
                case_number = int(tail)
                assert case_number >= 0, 'negative case number encountered'
                if case_number == 0:
                    assert not case_zero_used, 'more than one case number evaluated to zero'
                    case_zero_used = True
            except Exception:
                log.error('failed to determine case number for tail: ' + str(tail))
                continue
            for keyword in ensemble_property_dictionary.keys():
                (filename, p_property_kind, p_facet_type, p_facet, p_uom, p_time_index, p_null_value,
                 p_discrete) = ensemble_property_dictionary[keyword]
                p_filename = os.path.join(case_path, filename)
                if not os.path.exists(p_filename):
                    log.error('missing property file: ' + p_filename)
                    continue
                prop_import_collection.import_nexus_property_to_cache(p_filename,
                                                                      keyword,
                                                                      grid.extent_kji,
                                                                      discrete = p_discrete,
                                                                      uom = p_uom,
                                                                      time_index = p_time_index,
                                                                      null_value = p_null_value,
                                                                      property_kind = p_property_kind,
                                                                      facet_type = p_facet_type,
                                                                      facet = p_facet,
                                                                      realization = case_number,
                                                                      use_binary = False)
            if len(prop_import_collection.imported_list) > 0:
                # create hdf5 file using arrays cached in grid above
                log.info('writing properties to hdf5 file ' + str(resqml_file_root) + '.h5 for case: ' +
                         str(case_number))
                grid.write_hdf5_from_caches(resqml_file_root + '.h5',
                                            geometry = False,
                                            imported_properties = prop_import_collection,
                                            write_active = False)
                # add imported properties parts to model, building property parts list
                prop_import_collection.create_xml_for_imported_list_and_add_parts_to_model(ext_uuid,
                                                                                           time_series_uuid = ts_uuid)
                if create_property_set:
                    prop_import_collection.create_property_set_xml('realisation ' + str(case_number))
                case_count += 1
            # remove cached static property arrays from memory

            #         prop_import_collection.remove_all_cached_arrays()
            del prop_import_collection
            prop_import_collection = None
        log.info(f'Nexus ascii ensemble input processed {case_count} cases')
        tm.log_nexus_tm('info')

    # create hdf5 file using arrays cached in grid above
    if prop_import_collection is not None and len(prop_import_collection.imported_list) > 0:
        if decoarsen:
            decoarsen_array = prop_import_collection.decoarsen_imported_list()
            if decoarsen_array is not None:
                log.info('static properties decoarsened')
                prop_import_collection.add_cached_array_to_imported_list(decoarsen_array,
                                                                         'decoarsen',
                                                                         'DECOARSEN',
                                                                         discrete = True,
                                                                         uom = None,
                                                                         time_index = None,
                                                                         null_value = -1,
                                                                         property_kind = 'cell index')
        log.info('writing ' + str(len(prop_import_collection.imported_list)) + ' properties to hdf5 file ' +
                 resqml_file_root + '.h5')
    elif not ensemble_case_dirs_root:
        log.info('no static grid properties to import')
        prop_import_collection = None
    grid.write_hdf5_from_caches(resqml_file_root + '.h5',
                                geometry = False,
                                imported_properties = prop_import_collection,
                                write_active = True)
    # remove cached static property arrays from memory
    if prop_import_collection is not None:
        prop_import_collection.remove_all_cached_arrays()

    ts_selection = None
    if using_vdb and vdb_recurrent_properties and timestep_selection is not None and str(timestep_selection) != 'none':
        if prop_import_collection is None:
            prop_import_collection = rp.GridPropertyCollection()
            prop_import_collection.set_grid(grid)
        # extract timestep dates from summary file (this info might be hidden in the recurrent binary files but I couldn't find it
        # todo: create cut down time series from recurrent files and differentiate between reporting time index and mapped time step number
        full_time_series = rts.time_series_from_nexus_summary(summary_file)
        if full_time_series is None:
            log.error('failed to fetch time series from Nexus summary file; recurrent data excluded')
            tm.log_nexus_tm('error')
        else:
            full_time_series.set_model(model)
            timestep_list = vdbase.grid_list_of_timesteps(
                grid_title)  # get list of timesteps for which recurrent files exist
            recur_time_series = None
            for timestep_number in timestep_list:
                if isinstance(timestep_selection, list):
                    if timestep_number not in timestep_selection:
                        continue
                else:
                    if timestep_selection == 'first':
                        if timestep_number != timestep_list[0]:
                            break
                    elif timestep_selection == 'last':
                        if timestep_number != timestep_list[-1]:
                            continue
                    elif timestep_selection == 'first and last':
                        if timestep_number != timestep_list[0] and timestep_number != timestep_list[-1]:
                            continue
                    # default to importing all timesteps
                stamp = full_time_series.timestamp(timestep_number)
                if stamp is None:
                    log.error('timestamp number for which recurrent data exists was not found in summary file: ' +
                              str(timestep_number))
                    continue
                recur_prop_list = vdbase.grid_list_of_recurrent_properties(grid_title, timestep_number)
                common_recur_prop_set = set()
                if recur_time_series is None:
                    recur_time_series = rts.TimeSeries(model, first_timestamp = stamp)
                    if recur_prop_list is not None:
                        common_recur_prop_set = set(recur_prop_list)
                else:
                    recur_time_series.add_timestamp(stamp)
                    if recur_prop_list is not None:
                        common_recur_prop_set = common_recur_prop_set.intersection(set(recur_prop_list))
                step_import_collection = rp.GridPropertyCollection()
                step_import_collection.set_grid(grid)
                # for each property for this timestep, cache array and add to recur prop import collection for this time step
                if recur_prop_list:
                    for keyword in recur_prop_list:
                        if vdb.bad_keyword(keyword):
                            continue
                        prop_kind, facet_type, facet = rp.property_kind_and_facet_from_keyword(keyword)
                        step_import_collection.import_vdb_recurrent_property_to_cache(
                            vdbase,
                            timestep_number,  # also used as time_index?
                            keyword,
                            grid_name = grid_title,
                            property_kind = prop_kind,
                            facet_type = facet_type,
                            facet = facet)
                # extend hdf5 with cached arrays for this timestep
                log.info('number of recurrent grid property arrays for timestep: ' + str(timestep_number) + ' is: ' +
                         str(step_import_collection.number_of_imports()))
                if decoarsen_array is not None:
                    log.info('decoarsening recurrent properties for timestep: ' + str(timestep_number))
                    step_import_collection.decoarsen_imported_list(decoarsen_array = decoarsen_array)
                log.info('extending hdf5 file with recurrent properties for timestep: ' + str(timestep_number))
                grid.write_hdf5_from_caches(resqml_file_root + '.h5',
                                            mode = 'a',
                                            geometry = False,
                                            imported_properties = step_import_collection,
                                            write_active = False)
                # add imported list for this timestep to full imported list
                prop_import_collection.inherit_imported_list_from_other_collection(step_import_collection)
                log.debug('total number of property arrays after timestep: ' + str(timestep_number) + ' is: ' +
                          str(prop_import_collection.number_of_imports()))
                # remove cached copies of arrays
                step_import_collection.remove_all_cached_arrays()

            ts_node = full_time_series.create_xml(title = 'simulator full timestep series')
            model.time_series = ts_node  # save as the primary time series for the model
            ts_uuid = rqet.uuid_for_part_root(ts_node)
            # create xml for recur_time_series (as well as for full_time_series) and add as part; not needed?
            if recur_time_series is not None:
                rts_node = recur_time_series.create_xml(title = 'simulator recurrent array timestep series')
                if use_compressed_time_series:
                    ts_uuid = rqet.uuid_for_part_root(rts_node)
                    ts_selection = timestep_list

    # add imported properties parts to model, building property parts list
    if prop_import_collection is not None and prop_import_collection.imported_list is not None:
        prop_import_collection.set_grid(grid)  # update to pick up on recently created xml root node for grid
        prop_import_collection.create_xml_for_imported_list_and_add_parts_to_model(
            ext_uuid, time_series_uuid = ts_uuid, selected_time_indices_list = ts_selection)
        if create_property_set:
            prop_import_collection.create_property_set_xml('property set for import for grid ' + str(grid_title))

    # mark model as modified (will already have happened anyway)
    model.set_modified()

    # create epc file
    log.info('storing model in epc file ' + resqml_file_root + '.epc')
    model.store_epc(resqml_file_root + '.epc')

    # return resqml model
    return model
