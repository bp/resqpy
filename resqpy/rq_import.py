"""rq_import.py: Module to import a nexus corp grid & properties, or vdb, or vdb ensemble into resqml format."""

version = '5th August 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('rq_import.py version ' + version)

import os
import numpy as np
import numpy.ma as ma
import glob

import resqpy.olio.load_data as ld
# import resqpy.olio.grid_functions as gf
import resqpy.olio.write_data as wd
import resqpy.olio.ab_toolbox as abt
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.olio.vdb as vdb
import resqpy.olio.vector_utilities as vec
import resqpy.olio.trademark as tm

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.property as rp
import resqpy.time_series as rts
import resqpy.surface as rqs
import resqpy.organize as rqo
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
      vdb_static_properties = True,  # if True, static vdb properties are imported (only relevant if vdb_file is not None)
      vdb_recurrent_properties = False,
      timestep_selection = 'all',  # 'first', 'last', 'first and last', 'all', or list of ints being reporting timestep numbers
      use_compressed_time_series = True,
      decoarsen = True,  # where ICOARSE is present, redistribute data to uncoarse cells
      ab_property_list = None,  # list of (file_name, keyword, property_kind, facet_type, facet, uom, time_index, null_value, discrete)
      create_property_set = False,
      ensemble_case_dirs_root = None,  # path upto but excluding realisation number
      ensemble_property_dictionary = None,  # dictionary mapping title (or keyword) to (filename, property_kind, facet_type, facet,
      #                                           uom, time_index, null_value, discrete)
   ensemble_size_limit = None,
      grid_title = 'ROOT',
      mode = 'w',
      progress_fn = None):
   """Read a simulation grid geometry and optionally grid properties and return a resqml model in memory & written to disc.

      Input may be from nexus ascii input files, or nexus vdb output.
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
         cp_binary_file = abt.cp_binary_filename(corp_file,
                                                 nexus_ordering = False)  # pure binary, not bespoke corp bin used above
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

   grid = grid_from_cp(model,
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
            log.info('writing properties to hdf5 file ' + str(resqml_file_root) + '.h5 for case: ' + str(case_number))
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
                                                                     property_kind = 'discrete')
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
                  if not keyword or not keyword.isalnum():
                     continue
                  step_import_collection.import_vdb_recurrent_property_to_cache(vdbase,
                                                                                timestep_number,
                                                                                keyword,
                                                                                grid_name = grid_title)
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


def import_vdb_all_grids(
   resqml_file_root,  # output path and file name without .epc or .h5 extension
   extent_ijk = None,  # 3 element numpy vector applicable to ROOT
   vdb_file = None,
   vdb_case = None,  # if None, first case in vdb is used (usually a vdb only holds one case)
   corp_xy_units = 'm',
   corp_z_units = 'm',
   corp_z_inc_down = True,
   ijk_handedness = 'right',
   geometry_defined_everywhere = True,
   treat_as_nan = None,
   resqml_xy_units = 'm',
   resqml_z_units = 'm',
   resqml_z_inc_down = True,
   shift_to_local = False,
   local_origin_place = 'centre',  # 'centre' or 'minimum'
   max_z_void = 0.1,  # vertical gaps greater than this will introduce k gaps intp resqml grid
   split_pillars = True,
   split_tolerance = 0.01,  # applies to each of x, y, z differences
   vdb_static_properties = True,  # if True, static vdb properties are imported (only relevant if vdb_file is not None)
   vdb_recurrent_properties = False,
   decoarsen = True,
   timestep_selection = 'all',  # 'first', 'last', 'first and last', 'all', or list of ints being reporting timestep numbers
   create_property_set = False):
   """Creates a RESQML dataset containing grids and grid properties, including LGRs, for a single realisation."""

   vdbase = vdb.VDB(vdb_file)
   case_list = vdbase.cases()
   assert len(case_list) > 0, 'no cases found in vdb'
   if vdb_case is None:
      vdb_case = case_list[0]
   else:
      assert vdb_case in case_list, 'case ' + vdb_case + ' not found in vdb: ' + vdb_file
      vdbase.set_use_case(vdb_case)
   grid_list = vdbase.list_of_grids()
   index = 0
   for grid_name in grid_list:
      if grid_name.upper().startswith('SMALLGRIDS'):
         log.warning('vdb import skipping small grids')
         continue
      log.debug('importing vdb data for grid ' + str(grid_name))
      import_nexus(
         resqml_file_root,
         extent_ijk = extent_ijk if grid_name == 'ROOT' else None,  # 3 element numpy vector applicable to ROOT
         vdb_file = vdb_file,
         vdb_case = vdb_case,  # if None, first case in vdb is used (usually a vdb only holds one case)
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
         max_z_void = max_z_void,  # vertical gaps greater than this will introduce k gaps intp resqml grid
         split_pillars = split_pillars,  # NB: some LGRs may be unsplit even if ROOT is split
         split_tolerance = split_tolerance,  # applies to each of x, y, z differences
         vdb_static_properties = vdb_static_properties,  # if True, static vdb properties are imported
         vdb_recurrent_properties = vdb_recurrent_properties,
         decoarsen = decoarsen,
         timestep_selection = timestep_selection,
         create_property_set = create_property_set,
         grid_title = grid_name,
         mode = 'w' if index == 0 else 'a')
      index += 1


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
      corp_xy_units = 'metres',
      corp_z_units = 'metres',
      corp_z_inc_down = True,
      ijk_handedness = 'right',
      geometry_defined_everywhere = True,
      treat_as_nan = None,
      resqml_xy_units = 'metres',
      resqml_z_units = 'metres',
      resqml_z_inc_down = True,
      shift_to_local = True,
      local_origin_place = 'centre',  # 'centre' or 'minimum'
      max_z_void = 0.1,  # import will fail if vertical void greater than this is encountered
      split_pillars = True,
      split_tolerance = 0.01,  # applies to each of x, y, z differences
      progress_fn = None):
   """Adds properties from all vdb's within an ensemble directory tree to a single RESQML dataset, referencing a shared grid.

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
      corp_xy_units (string, default 'metres'): the units of x & y values in the vdb corp data; should be 'metres' or 'feet'
      corp_z_units (string, default 'metres'): the units of z values in the vdb corp data; should be 'metres' or 'feet'
      corp_z_inc_down (boolean, default True): set to True if corp z values are depth; False if elevation
      ijk_handedness (string, default 'right'): set to the handedness of the IJK axes in the Nexus model; 'right' or 'left'
      geometry_defined_everywhere (boolean, default True): set to False if inactive cells do not have valid geometry;
         deprecated - use treat_as_nan argument instead
      treat_as_nan (string, optional): if not None, one of 'dots', 'ij_dots', 'inactive'; controls which inactive cells
         have their geometry set to undefined
      resqml_xy_units (string, default 'metres'): the units of x & y values to use in the generated resqml grid;
         should be 'metres' or 'feet'
      resqml_z_units (string, default 'metres'): the units of z values to use in the generated resqml grid;
         should be 'metres' or 'feet'
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
      model = import_nexus(
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
         log.warning('no ROOT recurrent data found in vdb for first realisation; disabling recurrent property import')
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
               if property_kind_list is not None:
                  prop_kind, _, _ = rp.property_kind_and_facet_from_keyword(keyword)
                  if prop_kind not in property_kind_list and prop_kind not in ['active', 'region initialization']:
                     continue
               prop_import_collection.import_vdb_static_property_to_cache(vdbase, keyword, realization = realisation)
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
                     if property_kind_list is not None:
                        prop_kind, _, _ = rp.property_kind_and_facet_from_keyword(keyword)
                        if prop_kind not in property_kind_list:
                           continue
                     step_import_collection.import_vdb_recurrent_property_to_cache(
                        vdbase,
                        r_timestep_number,
                        keyword,
                        time_index = tni,  # index into recur_time_series
                        realization = realisation)
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


def add_ab_properties(
   epc_file,  # existing resqml model
   grid_uuid = None,  # optional grid uuid, required if more than one grid in model; todo: handle list of grids?
   ext_uuid = None,  # if None, hdf5 file holding grid geometry will be used
   ab_property_list = None
):  # list of (file_name, keyword, property_kind, facet_type, facet, uom, time_index, null_value,
   #          discrete, realization)
   """Process a list of pure binary property array files, adding as parts of model, related to grid (hdf5 file is appended to)."""

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


def add_surfaces(
   epc_file,  # existing resqml model
   crs_uuid = None,  # optional crs uuid, defaults to crs associated with model (usually main grid crs)
   ext_uuid = None,  # if None, uuid for hdf5 file holding main grid geometry will be used
   surface_file_format = 'zmap',  # zmap, rms (roxar) or GOCAD-Tsurf only formats currently supported
   rq_class = 'surface',  # 'surface' or 'mesh': the class of object to be created
   surface_role = 'map',  # 'map' or 'pick'
   quad_triangles = False,  # if True, 4 triangles per quadrangle will be used for mesh formats, otherwise 2
   surface_file_list = None,  # list of full file names (paths), each holding one surface
   make_horizon_interpretations_and_features = True):  # if True, feature and interpretation objects are created
   """Process a list of surface files, adding each surface as a new part in the resqml model."""

   assert surface_file_list, 'surface file list is empty or missing'
   assert surface_file_format in ['zmap', 'rms', 'roxar',
                                  'GOCAD-Tsurf'], 'unsupported surface file format: ' + str(surface_file_format)
   if 'TriangulatedSet' in rq_class:
      rq_class = 'surface'
   elif 'Grid2d' in rq_class:
      rq_class = 'mesh'
   assert rq_class in ['surface', 'mesh']

   log.info('accessing existing resqml model from: ' + epc_file)
   model = rq.Model(epc_file = epc_file)
   assert model, 'failed to read existing resqml model from file: ' + epc_file

   if crs_uuid is None:
      assert model.crs_root is not None, 'no crs uuid given and no default in model'
      crs_uuid = rqet.uuid_for_part_root(model.crs_root)
      assert crs_uuid is not None
   crs_root = model.root_for_uuid(crs_uuid)

   if ext_uuid is None:
      ext_uuid = model.h5_uuid()
   if ext_uuid is None:  # no pre-existing hdf5 part or references in model
      hdf5_file = epc_file[:-4] + '.h5'
      ext_node = model.create_hdf5_ext(file_name = hdf5_file)
      ext_uuid = rqet.uuid_for_part_root(ext_node)
      h5_mode = 'w'
   else:
      hdf5_file = model.h5_file_name(uuid = ext_uuid)
      h5_mode = 'a'

   assert ext_uuid is not None, 'failed to establish hdf5 uuid'

   # append to hdf5 file using arrays from Surface object's patch(es)
   log.info('will append to hdf5 file: ' + hdf5_file)

   for surf_file in surface_file_list:

      _, short_name = os.path.split(surf_file)
      dot = short_name.rfind('.')
      if dot > 0:
         short_name = short_name[:dot]

      log.info('surface ' + short_name + ' processing file: ' + surf_file + ' using format: ' + surface_file_format)
      if rq_class == 'surface':
         if surface_file_format == 'GOCAD-Tsurf':
            surface = rqs.Surface(model,
                                  tsurf_file = surf_file,
                                  surface_role = surface_role,
                                  quad_triangles = quad_triangles)
         else:
            surface = rqs.Surface(model,
                                  mesh_file = surf_file,
                                  mesh_format = surface_file_format,
                                  surface_role = surface_role,
                                  quad_triangles = quad_triangles)
      elif rq_class == 'mesh':
         if surface_file_format == 'GOCAD-Tsurf':
            log.info(f"Cannot convert a GOCAD-Tsurf to mesh, only to TriangulatedSurface - skipping file {surf_file}")
            break
         else:
            surface = rqs.Mesh(model,
                               mesh_file = surf_file,
                               mesh_format = surface_file_format,
                               mesh_flavour = 'reg&z',
                               surface_role = surface_role,
                               crs_uuid = crs_uuid)
      else:
         log.critical('this is impossible')
      # NB. surface may be either a Surface object or a Mesh object

      log.debug('appending to hdf5 file for surface file: ' + surf_file)
      surface.write_hdf5(hdf5_file, mode = h5_mode)

      if make_horizon_interpretations_and_features:
         feature = rqo.GeneticBoundaryFeature(model, kind = 'horizon', feature_name = short_name)
         feature.create_xml()
         interp = rqo.HorizonInterpretation(model, genetic_boundary_feature = feature, domain = 'depth')
         interp_root = interp.create_xml()
         surface.set_represented_interpretation_root(interp_root)

      surface.create_xml(ext_uuid,
                         add_as_part = True,
                         add_relationships = True,
                         crs_uuid = rqet.uuid_for_part_root(crs_root),
                         title = short_name + ' sourced from ' + surf_file,
                         originator = None)

   # mark model as modified
   model.set_modified()

   # store new version of model
   log.info('storing model with additional parts in epc file: ' + epc_file)
   model.store_epc(epc_file)

   return model


def grid_from_cp(model,
                 cp_array,
                 crs_uuid,
                 active_mask = None,
                 geometry_defined_everywhere = True,
                 treat_as_nan = None,
                 dot_tolerance = 1.0,
                 morse_tolerance = 5.0,
                 max_z_void = 0.1,
                 split_pillars = True,
                 split_tolerance = 0.01,
                 ijk_handedness = 'right',
                 known_to_be_straight = False):
   """Create a resqpy.grid.Grid object from a 7D corner point array.

   notes:
      this function sets up all the geometry arrays in memory but does not write to hdf5 nor create xml: use Grid methods;
      geometry_defined_everywhere is deprecated, use treat_as_nan instead
   """

   if treat_as_nan is None:
      if not geometry_defined_everywhere:
         treat_as_nan = 'morse'
   else:
      assert treat_as_nan in ['none', 'dots', 'ij_dots', 'morse', 'inactive']
      if treat_as_nan == 'none':
         treat_as_nan = None
   geometry_defined_everywhere = (treat_as_nan is None)

   assert cp_array.ndim == 7
   nk, nj, ni = cp_array.shape[:3]
   nk_plus_1 = nk + 1
   nj_plus_1 = nj + 1
   ni_plus_1 = ni + 1

   if active_mask is None:
      active_mask = np.ones((nk, nj, ni), dtype = 'bool')
      inactive_mask = np.zeros((nk, nj, ni), dtype = 'bool')
   else:
      assert active_mask.shape == (nk, nj, ni)
      inactive_mask = np.logical_not(active_mask)
   all_active = np.all(active_mask)

   if all_active and geometry_defined_everywhere:
      cp_nan_mask = None
   else:
      cp_nan_mask = np.any(np.isnan(cp_array), axis = (3, 4, 5, 6))  # ie. if any nan per cell
      if not geometry_defined_everywhere and not all_active:
         if treat_as_nan == 'inactive':
            log.debug('all inactive cell geometry being set to NaN')
            cp_nan_mask = np.logical_or(cp_nan_mask, inactive_mask)
         else:
            if treat_as_nan == 'dots':
               # for speed, only check primary diagonal of cells
               log.debug('geometry for cells with no length to primary cell diagonal being set to NaN')
               dot_mask = np.all(np.abs(cp_array[:, :, :, 1, 1, 1] - cp_array[:, :, :, 0, 0, 0]) < dot_tolerance,
                                 axis = -1)
            elif treat_as_nan in ['ij_dots', 'morse']:
               # check one diagonal of each I & J face
               log.debug(
                  'geometry being set to NaN for inactive cells with no length to primary face diagonal for any I or J face'
               )
               dot_mask = np.zeros((nk, nj, ni), dtype = bool)
               #              k_face_vecs = cp_array[:, :, :, :, 1, 1] - cp_array[:, :, :, :, 0, 0]
               j_face_vecs = cp_array[:, :, :, 1, :, 1] - cp_array[:, :, :, 0, :, 0]
               i_face_vecs = cp_array[:, :, :, 1, 1, :] - cp_array[:, :, :, 0, 0, :]
               dot_mask[:] = np.where(np.all(np.abs(j_face_vecs[:, :, :, 0]) < dot_tolerance, axis = -1), True,
                                      dot_mask)
               dot_mask[:] = np.where(np.all(np.abs(j_face_vecs[:, :, :, 1]) < dot_tolerance, axis = -1), True,
                                      dot_mask)
               dot_mask[:] = np.where(np.all(np.abs(i_face_vecs[:, :, :, 0]) < dot_tolerance, axis = -1), True,
                                      dot_mask)
               dot_mask[:] = np.where(np.all(np.abs(i_face_vecs[:, :, :, 1]) < dot_tolerance, axis = -1), True,
                                      dot_mask)
               log.debug(f'dot mask set for {np.count_nonzero(dot_mask)} cells')
               if treat_as_nan == 'morse':
                  morse_tol_sqr = morse_tolerance * morse_tolerance
                  # compare face vecs lengths in xy against max for active cells: where much greater set to NaN
                  len_j_face_vecs_sqr = np.sum(j_face_vecs[..., :2] * j_face_vecs[..., :2], axis = -1)
                  len_i_face_vecs_sqr = np.sum(j_face_vecs[..., :2] * i_face_vecs[..., :2], axis = -1)
                  dead_mask = inactive_mask.reshape(nk, nj, ni, 1).repeat(2, -1)
                  #                  mean_len_active_j_face_vecs_sqr = np.mean(ma.masked_array(len_j_face_vecs_sqr, mask = dead_mask))
                  #                  mean_len_active_i_face_vecs_sqr = np.mean(ma.masked_array(len_i_face_vecs_sqr, mask = dead_mask))
                  max_len_active_j_face_vecs_sqr = np.max(ma.masked_array(len_j_face_vecs_sqr, mask = dead_mask))
                  max_len_active_i_face_vecs_sqr = np.max(ma.masked_array(len_i_face_vecs_sqr, mask = dead_mask))
                  dot_mask = np.where(
                     np.any(len_j_face_vecs_sqr > morse_tol_sqr * max_len_active_j_face_vecs_sqr, axis = -1), True,
                     dot_mask)
                  dot_mask = np.where(
                     np.any(len_i_face_vecs_sqr > morse_tol_sqr * max_len_active_i_face_vecs_sqr, axis = -1), True,
                     dot_mask)
                  log.debug(f'morse mask set for {np.count_nonzero(dot_mask)} cells')
            else:
               raise Exception('code broken')
            cp_nan_mask = np.logical_or(cp_nan_mask, np.logical_and(inactive_mask, dot_mask))
      geometry_defined_everywhere = not np.any(cp_nan_mask)
      if geometry_defined_everywhere:
         cp_nan_mask = None

   if cp_nan_mask is not None:
      inactive_mask = np.logical_or(inactive_mask, cp_nan_mask)
      active_mask = np.logical_not(inactive_mask)

   # set up masked version of corner point data based on cells with defined geometry
   if geometry_defined_everywhere:
      full_mask = None
      masked_cp_array = ma.masked_array(cp_array, mask = ma.nomask)
      log.info('geometry present for all cells')
   else:
      full_mask = cp_nan_mask.reshape((nk, nj, ni, 1)).repeat(24, axis = 3).reshape((nk, nj, ni, 2, 2, 2, 3))
      masked_cp_array = ma.masked_array(cp_array, mask = full_mask)
      log.info('number of cells without geometry: ' + str(np.count_nonzero(cp_nan_mask)))

   # convert to resqml

   k_gaps = None
   k_gap_after_layer = None
   k_gap_raw_index = None

   if nk > 1:
      # check for (vertical) voids, or un-pillar-like anomalies, which will require k gaps in the resqml ijk grid
      log.debug('checking for voids')
      gap = masked_cp_array[1:, :, :, 0, :, :, :] - masked_cp_array[:-1, :, :, 1, :, :, :]
      max_gap_by_layer_and_xyz = np.max(np.abs(gap), axis = (1, 2, 3, 4))
      max_gap = np.max(max_gap_by_layer_and_xyz)
      log.debug('maximum void distance: {0:.3f}'.format(max_gap))
      if max_gap > max_z_void:
         log.warning('maximum void distance exceeds limit, grid will include k gaps')
         k_gaps = 0
         k_gap_after_layer = np.zeros((nk - 1,), dtype = bool)
         k_gap_raw_index = np.empty((nk,), dtype = int)
         k_gap_raw_index[0] = 0
         for k in range(nk - 1):
            max_layer_gap = np.max(max_gap_by_layer_and_xyz[k])
            if max_layer_gap > max_z_void:
               k_gap_after_layer[k] = True
               k_gaps += 1
            elif max_layer_gap > 0.0:
               # close void (includes shifting x & y)
               log.debug('closing void below layer (0 based): ' + str(k))
               layer_gap = gap[k] * 0.5
               layer_gap_unmasked = np.where(gap[k].mask, 0.0, layer_gap)
               masked_cp_array[k + 1, :, :, 0, :, :, :] -= layer_gap_unmasked
               masked_cp_array[k, :, :, 1, :, :, :] += layer_gap_unmasked
            k_gap_raw_index[k + 1] = k + k_gaps
      elif max_gap > 0.0:
         # close voids (includes shifting x & y)
         log.debug('closing voids')
         gap *= 0.5
         gap_unmasked = np.where(gap.mask, 0.0, gap)
         masked_cp_array[1:, :, :, 0, :, :, :] -= gap_unmasked
         masked_cp_array[:-1, :, :, 1, :, :, :] += gap_unmasked

   if k_gaps:
      nk_plus_1 += k_gaps
   if k_gap_raw_index is None:
      k_gap_raw_index = np.arange(nk, dtype = int)

   # reduce cp array extent in k
   log.debug('reducing k extent of corner point array (sharing points vertically)')
   k_reduced_cp_array = ma.masked_array(np.zeros((nk_plus_1, nj, ni, 2, 2, 3)))  # (nk+1+k_gaps, nj, ni, jp, ip, xyz)
   k_reduced_cp_array[0, :, :, :, :, :] = masked_cp_array[0, :, :, 0, :, :, :]
   k_reduced_cp_array[-1, :, :, :, :, :] = masked_cp_array[-1, :, :, 1, :, :, :]
   if k_gaps:
      raw_k = 1
      for k in range(nk - 1):
         # fill reduced array slice(s) for base of layer k and top of layer k + 1
         if k_gap_after_layer[k]:
            k_reduced_cp_array[raw_k, :, :, :, :, :] = masked_cp_array[k, :, :, 1, :, :, :]
            raw_k += 1
            k_reduced_cp_array[raw_k, :, :, :, :, :] = masked_cp_array[k + 1, :, :, 0, :, :, :]
            raw_k += 1
         else:  # take data from either possible cp slice, whichever is defined
            slice = masked_cp_array[k + 1, :, :, 0, :, :, :]
            k_reduced_cp_array[raw_k, :, :, :, :, :] = np.where(slice.mask, masked_cp_array[k, :, :, 1, :, :, :], slice)
            raw_k += 1
      assert raw_k == nk + k_gaps
   else:
      slice = masked_cp_array[1:, :, :, 0, :, :, :]
      # where cell geometry undefined, if cell above is defined, take data from cell above with kp = 1 and set shared point defined
      k_reduced_cp_array[1:-1, :, :, :, :, :] = np.where(slice.mask, masked_cp_array[:-1, :, :, 1, :, :, :], slice)

   # create 2D array of active columns (columns where at least one cell is active)
   log.debug('creating 2D array of active columns')
   active_mask_2D = np.any(active_mask, axis = 0)

   # create primary pillar reference indices as one of four column corners around pillar, active column preferred
   log.debug('creating primary pillar reference neighbourly indices')
   primary_pillar_jip = np.zeros((nj_plus_1, ni_plus_1, 2), dtype = 'int')  # (nj + 1, ni + 1, jp:ip)
   primary_pillar_jip[-1, :, 0] = 1
   primary_pillar_jip[:, -1, 1] = 1
   for j in range(nj_plus_1):
      for i in range(ni_plus_1):
         if active_mask_2D[j - primary_pillar_jip[j, i, 0], i - primary_pillar_jip[j, i, 1]]:
            continue
         if i > 0 and primary_pillar_jip[j, i, 1] == 0 and active_mask_2D[j - primary_pillar_jip[j, i, 0], i - 1]:
            primary_pillar_jip[j, i, 1] = 1
            continue
         if j > 0 and primary_pillar_jip[j, i, 0] == 0 and active_mask_2D[j - 1, i - primary_pillar_jip[j, i, 1]]:
            primary_pillar_jip[j, i, 0] = 1
            continue
         if i > 0 and j > 0 and primary_pillar_jip[j, i,
                                                   0] == 0 and primary_pillar_jip[j, i,
                                                                                  1] == 0 and active_mask_2D[j - 1,
                                                                                                             i - 1]:
            primary_pillar_jip[j, i, :] = 1

   # build extra pillar references for split pillars
   extras_count = np.zeros((nj_plus_1, ni_plus_1), dtype = 'int')  # count (0 to 3) of extras for pillar
   extras_list_index = np.zeros((nj_plus_1, ni_plus_1), dtype = 'int')  # index in list of 1st extra for pillar
   extras_list = []  # list of (jp, ip)
   extras_use = np.negative(np.ones((nj, ni, 2, 2), dtype = 'int'))  # (j, i, jp, ip); -1 means use primary
   if split_pillars:
      log.debug('building extra pillar references for split pillars')
      # loop over pillars
      for j in range(nj_plus_1):
         for i in range(ni_plus_1):
            primary_jp = primary_pillar_jip[j, i, 0]
            primary_ip = primary_pillar_jip[j, i, 1]
            p_col_j = j - primary_jp
            p_col_i = i - primary_ip
            # loop over 4 columns surrounding this pillar
            for jp in range(2):
               col_j = j - jp
               if col_j < 0 or col_j >= nj:
                  continue  # no column this side of pillar in j
               for ip in range(2):
                  col_i = i - ip
                  if col_i < 0 or col_i >= ni:
                     continue  # no column this side of pillar in i
                  if jp == primary_jp and ip == primary_ip:
                     continue  # this column is the primary for this pillar
                  discrepancy = np.max(
                     np.abs(k_reduced_cp_array[:, col_j, col_i, jp, ip, :] -
                            k_reduced_cp_array[:, p_col_j, p_col_i, primary_jp, primary_ip, :]))
                  if discrepancy <= split_tolerance:
                     continue  # data for this column's corner aligns with primary
                  for e in range(extras_count[j, i]):
                     eli = extras_list_index[j, i] + e
                     pillar_j_extra = j - extras_list[eli][0]
                     pillar_i_extra = i - extras_list[eli][1]
                     discrepancy = np.max(
                        np.abs(k_reduced_cp_array[:, col_j, col_i, jp, ip, :] -
                               k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, extras_list[eli][0],
                                                  extras_list[eli][1], :]))
                     if discrepancy <= split_tolerance:  # data for this corner aligns with existing extra
                        extras_use[col_j, col_i, jp, ip] = e
                        break
                  if extras_use[col_j, col_i, jp, ip] >= 0:  # reusing an existing extra for this pillar
                     continue
                  # add this corner as an extra
                  if extras_count[j, i] == 0:  # create entry point for this pillar in extras
                     extras_list_index[j, i] = len(extras_list)
                  extras_list.append((jp, ip))
                  extras_use[col_j, col_i, jp, ip] = extras_count[j, i]
                  extras_count[j, i] += 1
      if len(extras_list) == 0:
         split_pillars = False
      log.debug('number of extra pillars: ' + str(len(extras_list)))

   # create points array as used in resqml
   log.debug('creating points array as used in resqml format')
   if split_pillars:
      points_array = np.zeros(
         (nk_plus_1, (nj_plus_1 * ni_plus_1) + len(extras_list), 3))  # note: nk_plus_1 might include k_gaps
      index = 0
      # primary pillars
      for pillar_j in range(nj_plus_1):
         for pillar_i in range(ni_plus_1):
            (jp, ip) = primary_pillar_jip[pillar_j, pillar_i]
            slice = k_reduced_cp_array[:, pillar_j - jp, pillar_i - ip, jp, ip, :]
            points_array[:, index, :] = np.where(slice.mask, np.nan, slice)  # NaN indicates undefined/invalid geometry
            index += 1
      # add extras for split pillars
      for pillar_j in range(nj_plus_1):
         for pillar_i in range(ni_plus_1):
            for e in range(extras_count[pillar_j, pillar_i]):
               eli = extras_list_index[pillar_j, pillar_i] + e
               (jp, ip) = extras_list[eli]
               pillar_j_extra = pillar_j - jp
               pillar_i_extra = pillar_i - ip
               slice = k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, jp, ip, :]
               points_array[:, index, :] = np.where(slice.mask, np.nan,
                                                    slice)  # NaN indicates unedefined/invalid geometry
               index += 1
      assert (index == (nj_plus_1 * ni_plus_1) + len(extras_list))
   else:  # unsplit pillars
      points_array = np.zeros((nk_plus_1, nj_plus_1, ni_plus_1, 3))
      for j in range(nj_plus_1):
         for i in range(ni_plus_1):
            (jp, ip) = primary_pillar_jip[j, i]
            slice = k_reduced_cp_array[:, j - jp, i - ip, jp, ip, :]
            points_array[:, j, i, :] = np.where(slice.mask, np.nan, slice)  # NaN indicates undefined/invalid geometry

   # create an empty grid object and fill in some basic info
   log.debug('initialising grid object')
   grid = grr.Grid(model)
   grid.grid_representation = 'IjkGrid'
   grid.extent_kji = np.array((nk, nj, ni), dtype = 'int')
   grid.nk, grid.nj, grid.ni = nk, nj, ni
   grid.k_direction_is_down = True  # assumed direction for corp; todo: determine from geometry and crs z_inc_down flag
   if known_to_be_straight:
      grid.pillar_shape = 'straight'
   else:
      grid.pillar_shape = 'curved'
   grid.has_split_coordinate_lines = split_pillars
   grid.k_gaps = k_gaps
   grid.k_gap_after_array = k_gap_after_layer
   grid.k_raw_index_array = k_gap_raw_index

   grid.crs_uuid = crs_uuid
   grid.crs_root = model.root_for_uuid(crs_uuid)
   crs = rqc.Crs(model, uuid = crs_uuid)

   # add pillar points array to grid object
   log.debug('attaching points array to grid object')
   grid.points_cached = points_array  # NB: reference to points_array, array not copied here

   # add split pillar arrays to grid object
   if split_pillars:
      log.debug('adding split pillar arrays to grid object')
      split_pillar_indices_list = []
      cumulative_length_list = []
      cols_for_extra_pillar_list = []
      cumulative_length = 0
      for pillar_j in range(nj_plus_1):
         for pillar_i in range(ni_plus_1):
            for e in range(extras_count[pillar_j, pillar_i]):
               split_pillar_indices_list.append((pillar_j * ni_plus_1) + pillar_i)
               use_count = 0
               for jp in range(2):
                  j = pillar_j - jp
                  if j < 0 or j >= nj:
                     continue
                  for ip in range(2):
                     i = pillar_i - ip
                     if i < 0 or i >= ni:
                        continue
                     if extras_use[j, i, jp, ip] == e:
                        use_count += 1
                        cols_for_extra_pillar_list.append((j * ni) + i)
               assert (use_count > 0)
               cumulative_length += use_count
               cumulative_length_list.append(cumulative_length)
      log.debug('number of extra pillars: ' + str(len(split_pillar_indices_list)))
      assert (len(cumulative_length_list) == len(split_pillar_indices_list))
      grid.split_pillar_indices_cached = np.array(split_pillar_indices_list, dtype = 'int')
      log.debug('number of uses of extra pillars: ' + str(len(cols_for_extra_pillar_list)))
      assert (len(cols_for_extra_pillar_list) == np.count_nonzero(extras_use + 1))
      assert (len(cols_for_extra_pillar_list) == cumulative_length)
      grid.cols_for_split_pillars = np.array(cols_for_extra_pillar_list, dtype = 'int')
      assert (len(cumulative_length_list) == len(extras_list))
      grid.cols_for_split_pillars_cl = np.array(cumulative_length_list, dtype = 'int')
      grid.split_pillars_count = len(extras_list)

   # following is not part of resqml standard but is used by resqml_grid module for speed optimisation
   log.debug('setting up column to pillars mapping')
   base_pillar_count = nj_plus_1 * ni_plus_1
   grid.pillars_for_column = np.empty((nj, ni, 2, 2), dtype = 'int')
   for j in range(nj):
      for i in range(ni):
         for jp in range(2):
            for ip in range(2):
               if not split_pillars or extras_use[j, i, jp, ip] < 0:  # use primary pillar
                  pillar_index = (j + jp) * ni_plus_1 + i + ip
               else:
                  eli = extras_list_index[j + jp, i + ip] + extras_use[j, i, jp, ip]
                  pillar_index = base_pillar_count + eli
               grid.pillars_for_column[j, i, jp, ip] = pillar_index

   # add inactive cell mask to grid
   log.debug('setting inactive cell mask')
   grid.inactive = inactive_mask.copy()

   # add cell geometry defined array to model (using active cell mask unless geometry_defined_everywhere is True)
   if geometry_defined_everywhere:
      grid.geometry_defined_for_all_cells_cached = True
      grid.array_cell_geometry_is_defined = None
   else:
      log.debug('using active cell mask as indicator of defined cell geometry')
      grid.array_cell_geometry_is_defined = active_mask.copy()  # a bit harsh: disallows reactivation of cells
      grid.geometry_defined_for_all_cells_cached = np.all(active_mask)
   grid.geometry_defined_for_all_pillars_cached = True  # following fesapi convention of defining all pillars regardless
   # note: grid.array_pillar_geometry_is_defined not set, as line above should be sufficient

   # tentatively add corner point array to grid object in case it is needed
   log.debug('noting corner point array in grid')
   grid.array_corner_points = cp_array

   # set handedness of ijk axes
   if ijk_handedness is None or ijk_handedness == 'auto':
      # work out handedness from sample cell / column axes directions and handedness of crs
      sample_kji0 = tuple(np.array(grid.extent_kji) // 2)
      if not geometry_defined_everywhere and not grid.array_cell_geometry_is_defined[sample_kji0]:
         where_defined = np.where(
            np.logical_and(grid.array_cell_geometry_is_defined, np.logical_not(grid.pinched_out())))
         assert len(where_defined) == 3 and len(where_defined[0]) > 0, 'no extant cell geometries'
         sample_kji0 = (where_defined[0][0], where_defined[1][0], where_defined[2][0])
      sample_cp = cp_array[sample_kji0]
      cell_ijk_lefthanded = (vec.clockwise(sample_cp[0, 0, 0], sample_cp[0, 1, 0], sample_cp[0, 0, 1]) >= 0.0)
      if not grid.k_direction_is_down:
         cell_ijk_lefthanded = not cell_ijk_lefthanded
      if crs.is_right_handed_xyz():
         cell_ijk_lefthanded = not cell_ijk_lefthanded
      grid.grid_is_right_handed = not cell_ijk_lefthanded
   else:
      assert ijk_handedness in ['left', 'right']
      grid.grid_is_right_handed = (ijk_handedness == 'right')

   return grid
