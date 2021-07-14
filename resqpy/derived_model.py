"""derived_model.py: Functions creating a derived resqml model from an existing one; mostly grid manipulations."""

version = '14th July 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('derived_model.py version ' + version)

import os
import copy
import math as maths
import numpy as np
from time import time  # debug

import resqpy.olio.xml_et as rqet
import resqpy.olio.vector_utilities as vec
import resqpy.olio.simple_lines as sl
import resqpy.olio.intersection as meet
import resqpy.olio.box_utilities as bx
import resqpy.olio.grid_functions as gf
import resqpy.olio.fine_coarse as fc
import resqpy.olio.uuid as bu

import resqpy.model as rq
import resqpy.crs as rqcrs
import resqpy.grid as grr
import resqpy.grid_surface as rgs
import resqpy.property as rqp
import resqpy.well as rqw
import resqpy.fault as rqf
import resqpy.lines as rql
import resqpy.rq_import as rqi


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

   def dominant_zone(zone_array):
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

   if max_k0 is None:
      max_k0 = zone_array.shape[0] - 1
   if use_dominant_zone:
      dominant_zone(zone_array)
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
      zone_range_list = zone_layer_ranges_from_array(zone_by_cell_array, use_dominant_zone = use_dominant_zone)
      assert zone_range_list is not None and len(zone_range_list) > 0, 'failed to convert zone by cell to zone by layer'
      zone_by_layer = np.full((grid.nk,), null_value, dtype = int)
      for min_k0, max_k0, zone_index in zone_range_list:
         assert 0 <= min_k0 <= max_k0 < grid.nk, 'zonal layer limits out of range for grid (probable bug)'
         zone_by_layer[min_k0:max_k0 + 1] = zone_index
   else:
      raise Exception('code failure')

   assert zone_by_layer.ndim == 1 and zone_by_layer.size == grid.nk

   if use_local_property_kind:
      property_kind = 'zone'
      zone_pk = grr.establish_zone_property_kind(model)
      local_property_kind_uuid = zone_pk.uuid
   else:
      property_kind = 'discrete'
      local_property_kind_uuid = None

   model.h5_release()

   property_uuid = add_one_grid_property_array(epc_file,
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
                                extra_metadata = {},
                                new_epc_file = None):
   """Adds a grid property from a numpy array to an existing resqml dataset.

   arguments:
      epc_file (string): file name to load model resqml model from (and rewrite to if new_epc_file is None)
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
      extra_metadata (dict, optional): any items in this dictionary are added as extra metadata to the new
         property
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
                                         const_value = const_value)

   # write or re-write model
   model.h5_release()
   if new_epc_file:
      grid_title = rqet.citation_title_for_node(grid.root)
      uuid_list = write_grid(new_epc_file,
                             grid,
                             property_collection = gpc,
                             grid_title = grid_title,
                             mode = 'w',
                             time_series_uuid = time_series_uuid,
                             string_lookup_uuid = string_lookup_uuid,
                             extra_metadata = extra_metadata)
   else:
      # add arrays to hdf5 file holding source grid geometry
      uuid_list = write_grid(epc_file,
                             grid,
                             property_collection = gpc,
                             mode = 'a',
                             geometry = False,
                             time_series_uuid = time_series_uuid,
                             string_lookup_uuid = string_lookup_uuid,
                             extra_metadata = extra_metadata)

   if uuid_list is None or len(uuid_list) == 0:
      return None
   return uuid_list[0]


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

   bw_root = model.root(obj_type = '', uuid = blocked_well_uuid)
   assert bw_root is not None, f'no blocked well object found with uuid {blocked_well_uuid}'
   blocked_well = rqw.BlockedWell(model, blocked_well_root = bw_root)

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
                                          count = count_per_element)
   bwpc.write_hdf5_for_imported_list()
   uuid_list = bwpc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = time_series_uuid,
                                                                        string_lookup_uuid = string_lookup_uuid,
                                                                        property_kind_uuid = local_property_kind_uuid,
                                                                        extra_metadata = extra_metadata)
   assert len(uuid_list) == 1
   model.store_epc()
   return uuid_list[0]


def add_wells_from_ascii_file(epc_file,
                              crs_uuid,
                              trajectory_file,
                              comment_character = '#',
                              space_separated_instead_of_csv = False,
                              well_col = 'WELL',
                              md_col = 'MD',
                              x_col = 'X',
                              y_col = 'Y',
                              z_col = 'Z',
                              length_uom = 'm',
                              md_domain = None,
                              drilled = False,
                              z_inc_down = True,
                              new_epc_file = None):
   """Adds new md datum, trajectory, interpretation and feature objects for each well in an ascii file..

   arguments:
      epc_file (string): file name to load model resqml model from (and rewrite to if new_epc_file is None)
      crs_uuid (uuid.UUID): the unique identifier of the coordinate reference system applicable to the x,y,z data;
         if None, a default crs will be created, making use of the length_uom and z_inc_down arguments
      trajectory_file (string): the path of the ascii file holding the well trajectory data to be loaded
      comment_character (string, default '#'): character deemed to introduce a comment in the trajectory file
      space_separated_instead_of_csv (boolean, default False): if True, the columns in the trajectory file are space
         separated; if False, comma separated
      well_col (string, default 'WELL'): the heading for the column containing well names
      md_col (string, default 'MD'): the heading for the column containing measured depths
      x_col (string, default 'X'): the heading for the column containing X (usually easting) data
      y_col (string, default 'Y'): the heading for the column containing Y (usually northing) data
      z_col (string, default 'Z'): the heading for the column containing Z (depth or elevation) data
      length_uom (string, default 'm'): the units of measure for the measured depths; should be 'm' or 'ft'
      md_domain (string, optional): the source of the original deviation data; may be 'logger' or 'driller'
      drilled (boolean, default False): True should be used for wells that have been drilled; False otherwise (planned,
         proposed, or a location being studied)
      z_inc_down (boolean, default True): indicates whether z values increase with depth; only used in the creation
         of a default coordinate reference system; ignored if crs_uuid is not None
      new_epc_file (string, optional): if None, the source epc_file is extended with the new property object; if present,
         a new epc file (& associated h5 file) is created to contain a copy of the grid and the new property

   returns:
      int: the number of wells added

   notes:
      ascii file must be table with first line being column headers, with columns for WELL, MD, X, Y & Z;
      actual column names can be set with optional arguments;
      all the objects are added to the model, with array data being written to the hdf5 file for the trajectories;
      the md_domain and drilled values are stored in the RESQML metadata but are only for human information and do not
      generally affect computations
   """

   assert trajectory_file and os.path.exists(trajectory_file)
   if md_domain:
      assert md_domain in ['driller', 'logger']

   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None

   # open up model
   if new_epc_file:
      model = rq.Model(new_epc_file, copy_from = epc_file)
   else:
      model = rq.Model(epc_file)

   # sort out the coordinate reference system
   if crs_uuid is None:
      crs_uuid = rqet.uuid_for_part_root(model.crs_root)
   if crs_uuid is None:
      if z_inc_down is None:
         z_inc_down = True
      crs_root = rqcrs.Crs(model, xy_units = length_uom, z_units = length_uom, z_inc_down = z_inc_down)
      crs_uuid = rqet.uuid_for_part_root(crs_root)

   # add all the well related objects to the model, based on data in the ascii file
   (feature_list, interpretation_list, trajectory_list, md_datum_list) =  \
      rqw.add_wells_from_ascii_file(model, crs_uuid, trajectory_file, comment_character = comment_character,
                                    space_separated_instead_of_csv = space_separated_instead_of_csv,
                                    well_col = well_col, md_col = md_col, x_col = x_col, y_col = y_col, z_col = z_col,
                                    length_uom = length_uom, md_domain = md_domain, drilled = drilled)

   assert len(feature_list) == len(interpretation_list) == len(trajectory_list) == len(md_datum_list)
   count = len(feature_list)

   log.info('features, interpretations, trajectories and md data added for ' + str(count) + ' well' + _pl(count))

   # write or re-write model
   model.h5_release()
   model.store_epc()

   return count


def zonal_grid(epc_file,
               source_grid = None,
               zone_title = None,
               zone_uuid = None,
               zone_layer_range_list = None,
               k0_min = None,
               k0_max = None,
               use_dominant_zone = False,
               inactive_laissez_faire = True,
               new_grid_title = None,
               new_epc_file = None):
   """Extends an existing model with a new version of the source grid converted to a single, thick, layer per zone.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): a multi-layer RESQML grid object; if None, the epc_file is loaded
         and it should contain one ijk grid object (or one 'ROOT' grid) which is used as the source grid
      zone_title (string): if not None, a discrete property with this as the citation title is used as the zone property
      zone_uuid (string or uuid): if not None, a discrete property with this uuid is used as the zone property (see notes)
      zone_layer_range_list (list of (int, int, int)): each entry being (min_k0, max_k0, zone_index); alternative to
         working from a zone array
      k0_min (int, optional): the minimum layer number in the source grid (zero based) to include in the zonal version;
         default is zero (ie. top layer in source grid)
      k0_max (int, optional): the maximum layer number in the source grid (zero based) to include in the zonal version;
         default is nk - 1 (ie. bottom layer in source grid)
      use_dominant_zone (boolean, default False): if True, the most common zone value in each layer is used for the whole
         layer; if False, then variation of zone values in active cells in a layer will raise an assertion error
      inactive_laissez_faire (boolean, optional): if True, a cell in the zonal grid will be set active if any of the
         corresponding cells in the source grid are active; otherwise all corresponding cells in the source grid
         must be active for the zonal cell to be active; default is True
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the zonal grid (& crs)

   returns:
      new grid object (grid.Grid) with one layer per zone of the source grid

   notes:
      usually one of zone_title or zone_uuid or zone_layer_range_list should be passed, if noe are passed then a
      single layer grid is generated; zone_layer_range_list will take precendence if present
   """

   def fetch_zone_array(grid, zone_title = None, zone_uuid = None, masked = True):
      properties = grid.extract_property_collection()
      assert properties is not None and properties.number_of_parts() > 0, 'no properties found in relation to grid'
      properties = rqp.selective_version_of_collection(properties, continuous = False)
      assert properties is not None and properties.number_of_parts(
      ) > 0, 'no discreet properties found in relation to grid'
      if zone_title:
         properties = rqp.selective_version_of_collection(properties,
                                                          citation_title = zone_title)  # could make case insensitive?
         assert properties is not None and properties.number_of_parts(
         ) > 0, 'no discreet property found with title ' + zone_title
      if zone_uuid:
         zone_uuid_str = str(zone_uuid)
         part_name = None
         for part in properties.parts():
            if zone_uuid_str == str(properties.uuid_for_part(part)):
               part_name = part
               break
         if zone_title:
            postamble = ' (and title ' + zone_title + ')'
         else:
            postamble = ''
         assert part_name, 'no property found with uuid ' + zone_uuid_str + postamble
      else:
         part_name = properties.singleton()
      return properties.cached_part_array_ref(part_name, masked = masked)  # .copy() needed?

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model (or one named ROOT)
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None
   single_layer_mode = (not zone_title and not zone_uuid)

   if k0_min is None:
      k0_min = 0
   else:
      assert k0_min >= 0 and k0_min < source_grid.nk
   if k0_max is None:
      k0_max = source_grid.nk - 1
   else:
      assert k0_max >= 0 and k0_max < source_grid.nk and k0_max >= k0_min

   if not single_layer_mode:  # process zone array
      if zone_layer_range_list is None:
         zone_array = fetch_zone_array(source_grid, zone_title, zone_uuid)
         zone_layer_range_list = zone_layer_ranges_from_array(zone_array,
                                                              k0_min,
                                                              k0_max,
                                                              use_dominant_zone = use_dominant_zone)
      zone_count = len(zone_layer_range_list)
      # above is list of (zone_min_k0, zone_max_k0, zone) sorted by zone_min_k0
      log.info('following layer ranges are based on top layer being numbered 1 (simulator protocol)')
      for (zone_min_k0, zone_max_k0, zone) in zone_layer_range_list:
         log.info('zone id {0:1d} covers layers {1:1d} to {2:1d}'.format(zone, zone_min_k0 + 1, zone_max_k0 + 1))
   else:
      zone_layer_range_list = [(k0_min, k0_max, 0)]
      zone_count = 1
   assert zone_count > 0, 'unexpected lack of zones'

   # create a new, empty grid object
   is_regular = grr.is_regular_grid(source_grid.root) and single_layer_mode
   if is_regular:
      dxyz_dkji = source_grid.block_dxyz_dkji.copy()
      dxyz_dkji[0] *= k0_max - k0_min + 1
      grid = grr.RegularGrid(model,
                             extent_kji = (1, source_grid.nj, source_grid.ni),
                             dxyz_dkji = dxyz_dkji,
                             origin = source_grid.block_origin,
                             crs_uuid = source_grid.crs_uuid,
                             set_points_cached = False)
   else:
      grid = grr.Grid(model)
      # inherit attributes from source grid
      grid.grid_representation = 'IjkGrid'
      grid.extent_kji = np.array((zone_count, source_grid.nj, source_grid.ni), dtype = 'int')
      grid.nk, grid.nj, grid.ni = zone_count, source_grid.nj, source_grid.ni
      grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
      grid.crs_root = source_grid.crs_root
      grid.crs_uuid = source_grid.crs_uuid

   grid.k_direction_is_down = source_grid.k_direction_is_down
   grid.grid_is_right_handed = source_grid.grid_is_right_handed
   grid.pillar_shape = source_grid.pillar_shape

   # aggregate inactive cell mask depending on laissez faire argument
   if source_grid.inactive is None:
      log.debug('setting inactive mask to None')
      grid.inactive = None
   elif single_layer_mode:
      if inactive_laissez_faire:
         log.debug('setting inactive mask using all mode (laissez faire)')
         grid.inactive = np.all(source_grid.inactive[k0_min:k0_max + 1], axis = 0).reshape(grid.extent_kji)
      else:
         log.debug('setting inactive mask using any mode (strict)')
         grid.inactive = np.any(source_grid.inactive[k0_min:k0_max + 1], axis = 0).reshape(grid.extent_kji)
   else:
      grid.inactive = np.zeros(grid.extent_kji, dtype = bool)
      for zone_i in range(zone_count):
         zk0_min, zk0_max, _ = zone_layer_range_list[zone_i]
         if inactive_laissez_faire:
            grid.inactive[zone_i] = np.all(source_grid.inactive[zk0_min:zk0_max + 1], axis = 0)
         else:
            grid.inactive[zone_i] = np.any(source_grid.inactive[zk0_min:zk0_max + 1], axis = 0)

   if not is_regular:

      # rework the grid geometry
      source_grid.cache_all_geometry_arrays()
      # determine cell geometry is defined
      if hasattr(source_grid, 'array_cell_geometry_is_defined'):
         grid.array_cell_geometry_is_defined = np.empty(grid.extent_kji, dtype = bool)
         if single_layer_mode:
            grid.array_cell_geometry_is_defined[0] = np.logical_and(source_grid.array_cell_geometry_is_defined[k0_min],
                                                                    source_grid.array_cell_geometry_is_defined[k0_max])
         else:
            for zone_i in range(zone_count):
               zk0_min, zk0_max, _ = zone_layer_range_list[zone_i]
               grid.array_cell_geometry_is_defined[zone_i] = np.logical_and(
                  source_grid.array_cell_geometry_is_defined[zk0_min],
                  source_grid.array_cell_geometry_is_defined[zk0_max])
               # could attempt to pick up some corner points from K-neighbouring cells
         grid.geometry_defined_for_all_cells_cached = np.all(grid.array_cell_geometry_is_defined)
      else:
         grid.geometry_defined_for_all_cells_cached = source_grid.geometry_defined_for_all_cells_cached
      # copy info for pillar geometry is defined
      grid.geometry_defined_for_all_pillars_cached = source_grid.geometry_defined_for_all_pillars_cached
      if hasattr(source_grid, 'array_pillar_geometry_is_defined'):
         grid.array_pillar_geometry_is_defined = source_grid.array_pillar_geometry_is_defined.copy()
      # get reference to points for source grid geometry
      source_points = source_grid.points_ref()
      # slice top and base points
      points_shape = list(source_points.shape)
      points_shape[0] = zone_count + 1  # nk + 1
      grid.points_cached = np.zeros(points_shape)
      if grid.geometry_defined_for_all_cells_cached:
         if single_layer_mode:
            grid.points_cached[0] = source_points[k0_min]
            grid.points_cached[1] = source_points[k0_max + 1]  # base face
         else:
            for zone_i in range(zone_count):
               if zone_i == 0:
                  grid.points_cached[0] = source_points[zone_layer_range_list[zone_i][0]]
               grid.points_cached[zone_i + 1] = source_points[zone_layer_range_list[zone_i]
                                                              [1]]  # or could use 0th element of tuple for zone_i+1
      elif not grid.has_split_coordinate_lines:
         log.debug('scanning columns (unsplit pillars) for reference geometry')
         # fill in geometry: todo: replace with array operations if possible
         for zone_i in range(zone_count):
            zk0_min, zk0_max = zone_layer_range_list[zone_i][0:2]
            for j in range(grid.nj):
               for i in range(grid.ni):
                  if zone_i == 0:
                     for k in range(zk0_min, zk0_max + 1):
                        if source_grid.array_cell_geometry_is_defined[k, j, i]:
                           grid.points_cached[0, j:j + 2, i:i + 2] = source_points[k, j:j + 2, i:i + 2]
                           break
                  for k in range(zk0_max, zk0_min - 1, -1):
                     if source_grid.array_cell_geometry_is_defined[k, j, i]:
                        grid.points_cached[zone_i + 1, j:j + 2, i:i + 2] = source_points[k + 1, j:j + 2, i:i + 2]
                        grid.array_cell_geometry_is_defined[zone_i, j, i] = True
                        break
      else:
         log.debug('scanning columns (split pillars) for reference geometry')
         if not hasattr(source_grid, 'pillars_for_column'):
            source_grid.create_column_pillar_mapping()
         grid.pillars_for_column = source_grid.pillars_for_column.copy()
         for zone_i in range(zone_count):
            zk0_min, zk0_max = zone_layer_range_list[zone_i][0:2]
            for j in range(grid.nj):
               for i in range(grid.ni):
                  if grid.inactive[zone_i, j, i]:
                     continue
                  if zone_i == 0:
                     for k in range(zk0_min, zk0_max + 1):
                        if source_grid.array_cell_geometry_is_defined[k, j, i]:
                           for jp in range(2):
                              for ip in range(2):
                                 pillar = grid.pillars_for_column[j, i, jp, ip]
                                 grid.points_cached[0, pillar] = source_points[k, pillar]
                           break
                  for k in range(zk0_max + 1, zk0_min - 1, -1):
                     if source_grid.array_cell_geometry_is_defined[k, j, i]:
                        for jp in range(2):
                           for ip in range(2):
                              pillar = grid.pillars_for_column[j, i, jp, ip]
                              grid.points_cached[zone_i + 1, pillar] = source_points[k + 1, pillar]
                        grid.array_cell_geometry_is_defined[zone_i, j, i] = True
                        break
      if grid.has_split_coordinate_lines:
         grid.split_pillar_indices_cached = source_grid.split_pillar_indices_cached.copy()
         grid.cols_for_split_pillars = source_grid.cols_for_split_pillars.copy()
         grid.cols_for_split_pillars_cl = source_grid.cols_for_split_pillars_cl.copy()
         grid.split_pillars_count = source_grid.split_pillars_count

   if new_grid_title is None or len(new_grid_title) == 0:
      if single_layer_mode:
         preamble = 'single layer'
      else:
         preamble = 'zonal'
      new_grid_title = preamble + ' version of ' + str(rqet.citation_title_for_node(source_grid.root))

   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, grid_title = new_grid_title, mode = 'w')
   else:
      #      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']), 'Coordinates')
      write_grid(epc_file, grid, ext_uuid = None, grid_title = new_grid_title, mode = 'a')

   return grid


def single_layer_grid(epc_file,
                      source_grid = None,
                      k0_min = None,
                      k0_max = None,
                      inactive_laissez_faire = True,
                      new_grid_title = None,
                      new_epc_file = None):
   """Extends an existing model with a new version of the source grid converted to a single, thick, layer.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): a multi-layer RESQML grid object; if None, the epc_file is loaded
         and it should contain one ijk grid object (or one 'ROOT' grid) which is used as the source grid
      k0_min (int, optional): the minimum layer number in the source grid (zero based) to include in the single layer version;
         default is zero (ie. top layer in source grid)
      k0_max (int, optional): the maximum layer number in the source grid (zero based) to include in the single layer version;
         default is nk - 1 (ie. bottom layer in source grid)
      inactive_laissez_faire (boolean, optional): if True, a cell in the single layer grid will be set active if any of the
         corresponding cells in the source grid are active; otherwise all corresponding cells in the source grid
         must be active for the single layer cell to be active; default is True
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the single layer grid (& crs)

   returns:
      new grid object (grid.Grid) with a single layer representation of the source grid
   """

   return zonal_grid(epc_file,
                     source_grid = source_grid,
                     k0_min = k0_min,
                     k0_max = k0_max,
                     inactive_laissez_faire = inactive_laissez_faire,
                     new_grid_title = new_grid_title,
                     new_epc_file = new_epc_file)


def interpolated_grid(epc_file,
                      grid_a,
                      grid_b,
                      a_to_b_0_to_1 = 0.5,
                      split_tolerance = 0.01,
                      inherit_properties = False,
                      inherit_realization = None,
                      inherit_all_realizations = False,
                      new_grid_title = None,
                      new_epc_file = None):
   """Extends an existing model with a new grid geometry linearly interpolated between the two source_grids.

   arguments:
      epc_file (string): file name to rewrite the model's xml to
      grid_a, grid_b (grid.Grid objects): a pair of RESQML grid objects representing the end cases, between
         which the new grid will be interpolated
      a_to_b_0_to_1 (float, default 0.5): the interpolation factor in the range zero to one; a value of 0.0 will yield
         a copy of grid a, a value of 1.0 will yield a copy of grid b, intermediate values will yield a grid with all
         points interpolated
      split_tolerance (float, default 0.01): maximum offset of corner points for shared point to be generated; units
         are same as those in grid crs; only relevant if working from corner points, ignored otherwise
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with grid_a
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the interpolated grid (& crs)

   returns:
      new grid object (grid.Grid) with geometry interpolated between grid a and grid b

   notes:
      the hdf5 file used by the grid_a model is appended to, so it is recommended that the grid_a model's epc is specified
      as the first argument (unless a new epc file is required, sharing the hdf5 file)
   """

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert grid_a is not None and grid_b is not None, 'at least one source grid is missing'
   assert grid_a.grid_representation == 'IjkGrid' and grid_b.grid_representation == 'IjkGrid'
   assert 0.0 <= a_to_b_0_to_1 <= 1.0, 'interpolation factor outside range 0.0 to 1.0'
   assert tuple(grid_a.extent_kji) == tuple(grid_b.extent_kji), 'source grids have different extents'
   assert grid_a.k_direction_is_down == grid_b.k_direction_is_down, 'source grids have different k directions'
   assert grid_a.grid_is_right_handed == grid_b.grid_is_right_handed, 'source grids have different ijk handedness'
   assert grid_a.pillar_shape == grid_b.pillar_shape, 'source grids have different resqml pillar shapes'

   b_weight = a_to_b_0_to_1
   a_weight = 1.0 - b_weight

   model = grid_a.model

   if not bu.matching_uuids(grid_a.crs_uuid, grid_b.crs_uuid):
      crs_a = rqcrs.Crs(grid_a.model, uuid = grid_a.crs_uuid)
      crs_b = rqcrs.Crs(grid_b.model, uuid = grid_b.crs_uuid)
      assert crs_a.is_equivalent(crs_b), 'end point grids for interpolation have different coordinate reference systems'

   log.info('loading geometry for two source grids')
   grid_a.cache_all_geometry_arrays()
   grid_b.cache_all_geometry_arrays()

   assert grid_a.geometry_defined_for_all_cells() and grid_b.geometry_defined_for_all_cells(
   ), 'geometry not defined for all cells'
   #  assert grid_a.geometry_defined_for_all_pillars() and grid_b.geometry_defined_for_all_pillars(), 'geometry not defined for all pillars'

   if not grid_a.has_split_coordinate_lines and not grid_b.has_split_coordinate_lines:
      work_from_pillars = True
   elif (grid_a.has_split_coordinate_lines and grid_b.has_split_coordinate_lines and
         grid_a.points_cached.shape == grid_b.points_cached.shape and
         grid_a.split_pillar_indices_cached.shape == grid_b.split_pillar_indices_cached.shape and
         grid_a.cols_for_split_pillars.shape == grid_b.cols_for_split_pillars.shape and
         grid_a.cols_for_split_pillars_cl.shape == grid_b.cols_for_split_pillars_cl.shape and
         np.all(grid_a.split_pillar_indices_cached == grid_b.split_pillar_indices_cached) and
         np.all(grid_a.cols_for_split_pillars == grid_b.cols_for_split_pillars) and
         np.all(grid_a.cols_for_split_pillars_cl == grid_b.cols_for_split_pillars_cl)):
      work_from_pillars = True
   else:
      work_from_pillars = False

   if work_from_pillars:
      log.info('interpolating between compatible pillar grids')
   else:
      log.warning('interpolating between corner points due to pillar incompatibilities')

   # create a new, empty grid object
   grid = grr.Grid(model)

   # inherit attributes from source grid
   grid.grid_representation = 'IjkGrid'
   grid.extent_kji = grid_a.extent_kji.copy()
   grid.nk, grid.nj, grid.ni = grid.extent_kji
   grid.k_direction_is_down = grid_a.k_direction_is_down
   grid.grid_is_right_handed = grid_a.grid_is_right_handed
   grid.pillar_shape = grid_a.pillar_shape
   grid.has_split_coordinate_lines = (grid_a.has_split_coordinate_lines or grid_b.has_split_coordinate_lines)
   # inherit the coordinate reference system used by the grid geometry
   grid.crs_root = grid_a.crs_root
   grid.crs_uuid = grid_a.crs_uuid

   if grid_a.inactive is None or grid_b.inactive is None:
      grid.inactive = None
   else:
      grid.inactive = np.logical_and(grid_a.inactive, grid_b.inactive)

   grid.geometry_defined_for_all_cells_cached = True
   grid.array_cell_geometry_is_defined = np.ones(tuple(grid.extent_kji), dtype = bool)
   grid.geometry_defined_for_all_pillars_cached = True

   if work_from_pillars:
      grid.points_cached = grid_a.points_cached * a_weight + grid_b.points_cached * b_weight
      grid.has_split_coordinate_lines = grid_a.has_split_coordinate_lines
      if grid.has_split_coordinate_lines:
         grid.split_pillar_indices_cached = grid_a.split_pillar_indices_cached.copy()
         grid.cols_for_split_pillars = grid_a.cols_for_split_pillars.copy()
         grid.cols_for_split_pillars_cl = grid_a.cols_for_split_pillars_cl.copy()
         grid.split_pillars_count = grid_a.split_pillars_count
   else:
      grid.pillar_shape = 'curved'  # following fesapi approach of non-parametric pillars even if they are in fact straight
      cp_a = grid_a.corner_points(cache_cp_array = True)
      cp_b = grid_b.corner_points(cache_cp_array = True)
      assert cp_a.shape == cp_b.shape
      grid_cp = cp_a * a_weight + cp_b * b_weight
      if grid.nk > 1:
         z_gap = grid_cp[1:, :, :, 0, :, :, :] - grid_cp[:-1, :, :, 1, :, :, :]
         max_gap = np.max(np.abs(z_gap))
         log.info('maximum vertical void distance after corner point interpolation: {0:.3f} {1}'.format(
            max_gap, grid_a.z_units()))
         # close vertical voids (includes shifting x & y)
         if max_gap > 0.0:
            log.debug('closing vertical voids')
            z_gap *= 0.5
            grid_cp[1:, :, :, 0, :, :, :] -= z_gap
            grid_cp[:-1, :, :, 1, :, :, :] += z_gap
      # reduce cp array extent in k
      log.debug('reducing k extent of interpolated corner point array (sharing points vertically)')
      k_reduced_cp_array = np.zeros((grid.nk + 1, grid.nj, grid.ni, 2, 2, 3))  # (nk+1, nj, ni, jp, ip, xyz)
      k_reduced_cp_array[0, :, :, :, :, :] = grid_cp[0, :, :, 0, :, :, :]
      k_reduced_cp_array[-1, :, :, :, :, :] = grid_cp[-1, :, :, 1, :, :, :]
      if grid.nk > 1:
         k_reduced_cp_array[1:-1, :, :, :, :, :] = grid_cp[:-1, :, :, 1, :, :, :]
      # create primary pillar reference indices as one of four column corners around pillar, active column preferred
      log.debug('creating primary pillar reference neighbourly indices')
      primary_pillar_jip = np.zeros((grid.nj + 1, grid.ni + 1, 2), dtype = 'int')  # (nj + 1, ni + 1, jp:ip)
      primary_pillar_jip[-1, :, 0] = 1
      primary_pillar_jip[:, -1, 1] = 1
      # build extra pillar references for split pillars
      extras_count = np.zeros((grid.nj + 1, grid.ni + 1), dtype = 'int')  # count (0 to 3) of extras for pillar
      extras_list_index = np.zeros((grid.nj + 1, grid.ni + 1), dtype = 'int')  # index in list of 1st extra for pillar
      extras_list = []  # list of (jp, ip)
      extras_use = np.negative(np.ones((grid.nj, grid.ni, 2, 2), dtype = 'int'))  # (j, i, jp, ip); -1 means use primary
      log.debug('building extra pillar references for split pillars')
      # loop over pillars
      for j in range(grid.nj + 1):
         for i in range(grid.ni + 1):
            primary_jp = primary_pillar_jip[j, i, 0]
            primary_ip = primary_pillar_jip[j, i, 1]
            p_col_j = j - primary_jp
            p_col_i = i - primary_ip
            # loop over 4 columns surrounding this pillar
            for jp in range(2):
               col_j = j - jp
               if col_j < 0 or col_j >= grid.nj:
                  continue  # no column this side of pillar in j
               for ip in range(2):
                  col_i = i - ip
                  if col_i < 0 or col_i >= grid.ni:
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
         grid.has_split_coordinate_lines = False
      log.debug('number of extra pillars: ' + str(len(extras_list)))
      # create points array as used in resqml
      log.debug('creating points array as used in resqml format')
      if grid.has_split_coordinate_lines:
         points_array = np.zeros((grid.nk + 1, (grid.nj + 1) * (grid.ni + 1) + len(extras_list), 3))
         index = 0
         # primary pillars
         for pillar_j in range(grid.nj + 1):
            for pillar_i in range(grid.ni + 1):
               (jp, ip) = primary_pillar_jip[pillar_j, pillar_i]
               points_array[:, index, :] = k_reduced_cp_array[:, pillar_j - jp, pillar_i - ip, jp, ip, :]
               index += 1
         # add extras for split pillars
         for pillar_j in range(grid.nj + 1):
            for pillar_i in range(grid.ni + 1):
               for e in range(extras_count[pillar_j, pillar_i]):
                  eli = extras_list_index[pillar_j, pillar_i] + e
                  (jp, ip) = extras_list[eli]
                  pillar_j_extra = pillar_j - jp
                  pillar_i_extra = pillar_i - ip
                  points_array[:, index, :] = k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, jp, ip, :]
                  index += 1
         assert index == (grid.nj + 1) * (grid.ni + 1) + len(extras_list)
      else:  # unsplit pillars
         points_array = np.zeros((grid.nk + 1, grid.nj + 1, grid.ni + 1, 3))
         for j in range(grid.nj + 1):
            for i in range(grid.ni + 1):
               (jp, ip) = primary_pillar_jip[j, i]
               points_array[:, j, i, :] = k_reduced_cp_array[:, j - jp, i - ip, jp, ip, :]
      grid.points_cached = points_array
      # add split pillar arrays to grid object
      if grid.has_split_coordinate_lines:
         log.debug('adding split pillar arrays to grid object')
         split_pillar_indices_list = []
         cumulative_length_list = []
         cols_for_extra_pillar_list = []
         cumulative_length = 0
         for pillar_j in range(grid.nj + 1):
            for pillar_i in range(grid.ni + 1):
               for e in range(extras_count[pillar_j, pillar_i]):
                  split_pillar_indices_list.append(pillar_j * (grid.ni + 1) + pillar_i)
                  use_count = 0
                  for jp in range(2):
                     j = pillar_j - jp
                     if j < 0 or j >= grid.nj:
                        continue
                     for ip in range(2):
                        i = pillar_i - ip
                        if i < 0 or i >= grid.ni:
                           continue
                        if extras_use[j, i, jp, ip] == e:
                           use_count += 1
                           cols_for_extra_pillar_list.append((j * grid.ni) + i)
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

   collection = _prepare_simple_inheritance(grid, grid_a, inherit_properties, inherit_realization,
                                            inherit_all_realizations)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'interpolated between two grids with factor: ' + str(a_to_b_0_to_1)

   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(grid_a.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def extract_box(epc_file = None,
                source_grid = None,
                box = None,
                box_inactive = None,
                inherit_properties = False,
                inherit_realization = None,
                inherit_all_realizations = False,
                set_parent_window = None,
                new_grid_title = None,
                new_epc_file = None):
   """Extends an existing model with a new grid extracted as a logical IJK box from the source grid.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      box (numpy int array of shape (2, 3)): the minimum and maximum kji0 indices in the source grid (zero based) to include
         in the extracted grid; note that cells with index equal to maximum value are included (unlike with python ranges)
      box_inactive (numpy bool array, optional): if present, shape must match box and values will be or'ed in with the
         inactive mask inherited from the source grid; if None, inactive mask will be as inherited from source grid
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid, with values taken from the specified box
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      set_parent_window (boolean, optional): if True, the extracted grid has its parent window attribute set; if False,
         the parent window is not set; if None, the default will be True if new_epc_file is None or False otherwise
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the extracted grid (& crs)

   returns:
      new grid object with extent as implied by the box argument

   note:
      the epc file and associated hdf5 file are appended to (extended) with the new grid, unless a new_epc_file is specified,
      in which case the grid and inherited properties are written there instead
   """

   def array_box(a, box):
      return a[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1, box[0, 2]:box[1, 2] + 1].copy()

   def local_col_index(extent, box, col):
      # return local equivalent natural column index for global column index, or None if outside box
      j, i = divmod(col, extent[2])
      j -= box[0, 1]
      i -= box[0, 2]
      if j < 0 or i < 0 or j > box[1, 1] - box[0, 1] or i > box[1, 2] - box[0, 2]:
         return None
      return j * (box[1, 2] - box[0, 2] + 1) + i

   def local_pillar_index(extent, box, p):
      # return local equivalent natural pillar index for global pillar index, or None if outside box
      p_j, p_i = divmod(p, extent[2] + 1)
      p_j -= box[0, 1]
      p_i -= box[0, 2]
      if p_j < 0 or p_i < 0 or p_j > box[1, 1] - box[0, 1] + 1 or p_i > box[1, 2] - box[0, 2] + 1:
         return None
      return p_j * (box[1, 2] - box[0, 2] + 2) + p_i

   def cols_for_pillar(extent, p):
      # return 4 naturalized column indices for columns surrounding natural pillar index; -1 where beyond edge of ij space
      cols = np.zeros((4,), dtype = int) - 1
      p_j, p_i = divmod(p, extent[2] + 1)
      if p_j > 0 and p_i > 0:
         cols[0] = (p_j - 1) * extent[2] + p_i - 1
      if p_j > 0 and p_i < extent[2]:
         cols[1] = (p_j - 1) * extent[2] + p_i
      if p_j < extent[1] and p_i > 0:
         cols[2] = p_j * extent[2] + p_i - 1
      if p_j < extent[1] and p_i < extent[2]:
         cols[3] = p_j * extent[2] + p_i
      return cols

   log.debug('extracting grid for box')

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if set_parent_window is None:
      set_parent_window = (new_epc_file is None)
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model (or one named 'ROOT')
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None
   assert box is not None and box.shape == (2, 3)
   assert np.all(box[1, :] >= box[0, :]) and np.all(box[0, :] >= 0) and np.all(box[1, :] < source_grid.extent_kji)

   box_str = bx.string_iijjkk1_for_box_kji0(box)

   if not source_grid.k_gaps and source_grid.nk_plus_k_gaps is None:
      source_grid.nk_plus_k_gaps = source_grid.nk

   # create a new, empty grid object
   grid = grr.Grid(model)

   # inherit attributes from source grid
   grid.grid_representation = 'IjkGrid'
   grid.extent_kji = box[1, :] - box[0, :] + 1
   if box_inactive is not None:
      assert box_inactive.shape == tuple(grid.extent_kji)
   grid.nk, grid.nj, grid.ni = grid.extent_kji[0], grid.extent_kji[1], grid.extent_kji[2]
   grid.k_direction_is_down = source_grid.k_direction_is_down
   grid.grid_is_right_handed = source_grid.grid_is_right_handed
   grid.pillar_shape = source_grid.pillar_shape
   grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
   # inherit the coordinate reference system used by the grid geometry
   grid.crs_root = source_grid.crs_root
   grid.crs_uuid = source_grid.crs_uuid

   # inherit k_gaps for selected layer range
   grid.nk_plus_k_gaps = grid.nk
   if source_grid.k_gaps and box[1, 0] > box[0, 0]:
      k_gaps = np.count_nonzero(source_grid.k_gap_after_array[box[0, 0]:box[1, 0]])
      if k_gaps > 0:
         grid.k_gaps = k_gaps
         grid.nk_plus_k_gaps = grid.nk + k_gaps
         grid.k_gap_after_array = source_grid.k_gap_after_array[box[0, 0]:box[1, 0]].copy()
         grid.k_raw_index_array = np.empty(grid.nk, dtype = int)
         k_offset = 0
         for k in range(grid.nk):
            grid.k_raw_index_array[k] = k + k_offset
            if k < grid.nk - 1 and grid.k_gap_after_array[k]:
               k_offset += 1
         assert k_offset == k_gaps

   # extract inactive cell mask
   if source_grid.inactive is None:
      if box_inactive is None:
         log.debug('setting inactive mask to None')
         grid.inactive = None
      else:
         log.debug('setting inactive mask to that passed as argument')
         grid.inactive = box_inactive.copy()
   else:
      if box_inactive is None:
         log.debug('extrating inactive mask')
         grid.inactive = array_box(source_grid.inactive, box)
      else:
         log.debug('setting inactive mask to merge of source grid extraction and mask passed as argument')
         grid.inactive = np.logical_or(array_box(source_grid.inactive, box), box_inactive)

   # extract the grid geometry
   source_grid.cache_all_geometry_arrays()

   # determine cell geometry is defined
   if hasattr(source_grid, 'array_cell_geometry_is_defined'):
      grid.array_cell_geometry_is_defined = array_box(source_grid.array_cell_geometry_is_defined, box)
      grid.geometry_defined_for_all_cells_cached = np.all(grid.array_cell_geometry_is_defined)
   else:
      grid.geometry_defined_for_all_cells_cached = source_grid.geometry_defined_for_all_cells_cached

   # copy info for pillar geometry is defined
   grid.geometry_defined_for_all_pillars_cached = source_grid.geometry_defined_for_all_pillars_cached
   if hasattr(source_grid, 'array_pillar_geometry_is_defined'):
      grid.array_pillar_geometry_is_defined = array_box(source_grid.array_pillar_geometry_is_defined, box)
      grid.geometry_defined_for_all_pillars_cached = np.all(grid.array_pillar_geometry_is_defined)

   # get reference to points for source grid geometry
   source_points = source_grid.points_ref()

   pillar_box = box.copy()
   if source_grid.k_gaps:
      pillar_box[:, 0] = source_grid.k_raw_index_array[pillar_box[:, 0]]
   pillar_box[1, :] += 1  # pillar points have extent one greater than cells, in each axis

   if not source_grid.has_split_coordinate_lines:
      log.debug('no split pillars in source grid')
      grid.points_cached = array_box(source_points, pillar_box)  # should work, ie. preserve xyz axis
   else:
      source_base_pillar_count = (source_grid.nj + 1) * (source_grid.ni + 1)
      log.debug('number of base pillars in source grid: ' + str(source_base_pillar_count))
      log.debug('number of extra pillars in source grid: ' + str(len(source_grid.split_pillar_indices_cached)))
      base_points = array_box(
         source_points[:, :source_base_pillar_count, :].reshape(
            (source_grid.nk_plus_k_gaps + 1, source_grid.nj + 1, source_grid.ni + 1, 3)),
         pillar_box).reshape(grid.nk_plus_k_gaps + 1, (grid.nj + 1) * (grid.ni + 1), 3)
      extra_points = np.zeros(
         (pillar_box[1, 0] - pillar_box[0, 0] + 1, source_points.shape[1] - source_base_pillar_count, 3))
      spi_array = np.zeros(len(source_grid.split_pillar_indices_cached), dtype = int)
      local_cols_array = np.zeros(len(source_grid.cols_for_split_pillars), dtype = int)
      local_cols_cl = np.zeros(len(source_grid.split_pillar_indices_cached), dtype = int)
      local_index = 0
      for index in range(len(source_grid.split_pillar_indices_cached)):
         source_pi = source_grid.split_pillar_indices_cached[index]
         local_pi = local_pillar_index(source_grid.extent_kji, box, source_pi)
         if local_pi is None:
            continue
         cols = cols_for_pillar(source_grid.extent_kji, source_pi)
         local_cols = cols_for_pillar(grid.extent_kji, local_pi)
         if index == 0:
            start = 0
         else:
            start = source_grid.cols_for_split_pillars_cl[index - 1]
         finish = source_grid.cols_for_split_pillars_cl[index]
         source_pis = np.zeros((4,), dtype = int)
         for c_i in range(4):
            if local_cols[c_i] < 0:
               source_pis[c_i] = -1
               continue
            if cols[c_i] in source_grid.cols_for_split_pillars[start:finish]:
               source_pis[c_i] = source_base_pillar_count + index
            else:
               source_pis[c_i] = source_pi
         unique_source_pis = np.unique(source_pis)
         unique_count = len(unique_source_pis)
         unique_index = 0
         if unique_source_pis[0] == -1:
            unique_index = 1
            unique_count -= 1
         if unique_count <= 0:
            continue
         base_points[:, local_pi, :] = source_points[pillar_box[0, 0]:pillar_box[1, 0] + 1,
                                                     unique_source_pis[unique_index], :]
         unique_index += 1
         unique_count -= 1
         if unique_count <= 0:
            continue
         while unique_count > 0:
            source_pi = unique_source_pis[unique_index]
            extra_points[:, local_index, :] = source_points[pillar_box[0, 0]:pillar_box[1, 0] + 1, source_pi, :]
            spi_array[local_index] = local_pi
            if local_index == 0:
               lc_index = 0
            else:
               lc_index = local_cols_cl[local_index - 1]
            for c_i in range(4):
               if source_pis[c_i] == source_pi and local_cols[c_i] >= 0:
                  local_cols_array[lc_index] = local_cols[c_i]
                  lc_index += 1
            local_cols_cl[local_index] = lc_index
            local_index += 1
            unique_index += 1
            unique_count -= 1
      if local_index == 0:  # there are no split pillars in the box
         log.debug('box does not inherit any split pillars')
         grid.points_cached = base_points.reshape(grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1, 3)
         grid.has_split_coordinate_lines = False
      else:
         log.debug('number of extra pillars in box: ' + str(local_index))
         grid.points_cached = np.concatenate((base_points, extra_points[:, :local_index, :]), axis = 1)
         grid.split_pillar_indices_cached = spi_array[:local_index].copy()
         grid.cols_for_split_pillars = local_cols_array[:local_cols_cl[local_index - 1]].copy()
         grid.cols_for_split_pillars_cl = local_cols_cl[:local_index].copy()
         grid.split_pillars_count = local_index

   if set_parent_window:
      fine_coarse = fc.FineCoarse(grid.extent_kji, grid.extent_kji, within_coarse_box = box)
      fine_coarse.set_all_ratios_constant()
      grid.set_parent(source_grid.uuid, True, fine_coarse)

   collection = None
   if inherit_properties:
      source_collection = source_grid.extract_property_collection()
      if source_collection is not None:
         # do not inherit the inactive property array by this mechanism
         active_collection = rqp.selective_version_of_collection(source_collection, property_kind = 'active')
         source_collection.remove_parts_list_from_dict(active_collection.parts())
         inactive_collection = rqp.selective_version_of_collection(
            source_collection,
            property_kind = 'code',  # for backward compatibility
            facet_type = 'what',
            facet = 'inactive')
         source_collection.remove_parts_list_from_dict(inactive_collection.parts())
         collection = rqp.GridPropertyCollection()
         collection.set_grid(grid)
         collection.extend_imported_list_copying_properties_from_other_grid_collection(
            source_collection,
            box = box,
            realization = inherit_realization,
            copy_all_realizations = inherit_all_realizations)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'local grid ' + box_str + ' extracted from ' + str(rqet.citation_title_for_node(
         source_grid.root))

   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def extract_box_for_well(epc_file = None,
                         source_grid = None,
                         min_k0 = None,
                         max_k0 = None,
                         trajectory_epc = None,
                         trajectory_uuid = None,
                         blocked_well_uuid = None,
                         column_ji0 = None,
                         column_xy = None,
                         well_name = None,
                         radius = None,
                         outer_radius = None,
                         active_cells_shape = 'tube',
                         quad_triangles = True,
                         inherit_properties = False,
                         inherit_realization = None,
                         inherit_all_realizations = False,
                         inherit_well = False,
                         set_parent_window = None,
                         new_grid_title = None,
                         new_epc_file = None):
   """Extends an existing model with a new grid extracted as an IJK box around a well trajectory in the source grid.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      min_k0, max_k0 (integers, optional): layer range to include; default is full vertical range of source grid
      trajectory_epc (string, optional): the source file for the trajectory or blocked well, if different to that for
         the source grid
      trajectory_uuid (uuid.UUID): the uuid of the trajectory object for the well, if working from a trajectory
      blocked_well_uuid (uuid.UUID): the uuid of the blocked well object, an alternative to working from a trajectory;
         must include blocking against source_grid
      column_ji0 (integer pair, optional): an alternative to providing a trajectory: the column indices of a 'vertical' well
      column_xy (float pair, optional): an alternative to column_ji0: the x, y location used to determine the column
      well_name (string, optional): name to use for column well, ignored if trajectory_uuid is not None
      radius (float, optional): the radius around the wellbore to include in the box; units are those of grid xy values;
         radial distances are applied horizontally regardless of well inclination; if not present, only cells penetrated
         by the trajectory are included
      outer_radius (float, optional): an outer radius around the wellbore, beyond which an inactive cell mask for the
         source_grid will be set to True (inactive); units are those of grid xy values
      active_cells_shape (string, default 'tube'): the logical shape of cells marked as active in the extracted box;
         'tube' results in an active shape with circular cross section in IJ planes, that follows the trajectory; 'prism'
         activates all cells in IJ columns where any cell is within the tube; 'box' leaves the entire IJK cuboid active
      quad_triangles (boolean, default True): if True, cell K faces are treated as 4 triangles (with a common face
         centre point) when computing the intersection of the trajectory with layer interfaces (horizons); if False,
         the K faces are treated as 2 triangles
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid, with values taken from the extracted box
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      inherit_well (boolean, default False): if True, the new model will have a copy of the well trajectory, its crs (if
         different from that of the grid), and any related wellbore interpretation and feature
      set_parent_window (boolean, optional): if True, the extracted grid has its parent window attribute set; if False,
         the parent window is not set; if None, the default will be True if new_epc_file is None or False otherwise
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the extracted grid (& crs)

   returns:
      (grid, box) where: grid is the new Grid object with extent as determined by source grid geometry, trajectory and
      radius arguments; and box is a numpy int array of shape (2, 3) with first axis covering min, max and second axis
      covering k,j,i; the box array holds the minimum and maximum indices (zero based) in the source grid that have
      been included in the extraction (nb. maximum indices are included, unlike the usual python protocol)

   notes:
      this function is designed to work fully for vertical and deviated wells; for horizontal wells use blocked well mode;
      the extracted box includes all layers between the specified min and max horizons, even if the trajectory terminates
      above the deeper horizon or does not intersect horizon(s) for other reasons;
      when specifying a column well by providing x,y the IJ column with the centre of the topmost k face closest to the
      given point is selected;
      if an outer_radius is given, a boolean property will be created for the source grid with values set True where the
      centres of the cells are beyond this distance from the well, measured horizontally; if outer_radius and new_epc_file
      are both given, the source grid will be copied to the new epc
   """

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if trajectory_epc == epc_file:
      trajectory_epc = None
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model (or one named 'ROOT')
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None
   if trajectory_epc is None:
      traj_model = model
   else:
      traj_model = rq.Model(trajectory_epc)
   assert traj_model is not None

   if min_k0 is None:
      min_k0 = 0
   if max_k0 is None:
      max_k0 = source_grid.nk - 1
   assert 0 <= min_k0 <= max_k0 < source_grid.nk
   assert trajectory_uuid is not None or blocked_well_uuid is not None or column_ji0 is not None
   if radius is None:
      radius = 0.0  # cells directly penetrated through a k face will still be included
   assert radius >= 0.0
   if outer_radius is not None:
      assert outer_radius >= radius
   assert active_cells_shape in ['tube', 'prism', 'box']
   warned = False
   box = None
   # prepare cell centre points for testing inclusion
   centres = source_grid.centre_point(cache_centre_array = True)
   inclusion_mask = np.zeros(source_grid.extent_kji, dtype = bool)  # intialised to False
   radius_sqr = radius * radius
   if outer_radius is not None:
      outer_radius_sqr = outer_radius * outer_radius
      outer_inactive_mask = np.zeros(source_grid.extent_kji, dtype = bool)  # intialised to False
   trajectory = blocked_well = None
   bw_box = None
   if trajectory_uuid is not None:
      # prepare a trajectory object
      trajectory_root = traj_model.root(obj_type = 'WellboreTrajectoryRepresentation', uuid = trajectory_uuid)
      assert trajectory_root is not None, 'trajectory object not found for uuid: ' + str(trajectory_uuid)
      trajectory = rqw.Trajectory(traj_model, trajectory_root = trajectory_root)
      well_name = rqw.well_name(trajectory)
      traj_crs = rqcrs.Crs(trajectory.model, uuid = trajectory.crs_uuid)
      grid_crs = rqcrs.Crs(source_grid.model, uuid = source_grid.crs_uuid)
      # modify in-memory trajectory data to be in the same crs as grid
      traj_crs.convert_array_to(grid_crs,
                                trajectory.control_points)  # trajectory xyz points converted in situ to grid's crs
      trajectory.crs_uuid = source_grid.crs_uuid  # note: tangent vectors might be messed up, if present
      traj_box = np.empty((2, 3))
      traj_box[0] = np.amin(trajectory.control_points, axis = 0)
      traj_box[1] = np.amax(trajectory.control_points, axis = 0)
      grid_box = source_grid.xyz_box(lazy = False)
      if not bx.boxes_overlap(traj_box, grid_box):
         log.error('no overlap of xyz boxes for trajectory and grid for trajectory uuid: ' + str(trajectory.uuid))
         return None, None
   elif blocked_well_uuid is not None:
      bw_root = traj_model.root(obj_type = 'BlockedWellboreRepresentation', uuid = blocked_well_uuid)
      assert bw_root is not None, 'blocked well object not found for uuid: ' + str(blocked_well_uuid)
      blocked_well = rqw.BlockedWell(traj_model, blocked_well_root = bw_root)
      bw_box = blocked_well.box(grid_uuid = source_grid.uuid)
      assert bw_box is not None, 'blocked well does not include cells in source grid'
      assert bw_box[0,
                    0] <= max_k0 and bw_box[1,
                                            0] >= min_k0, 'blocked well does not include cells in specified layer range'
      bw_cells = blocked_well.cell_indices_for_grid_uuid(source_grid.uuid)
   else:
      if column_ji0 is None:
         assert len(column_xy) == 2
         column_ji0 = source_grid.find_cell_for_point_xy(column_xy[0], column_xy[1])
         if column_ji0[0] is None or column_ji0[1] is None:
            log.error('no column found for x, y: ' + str(column_xy[0]) + ', ' + str(column_xy[1]))
         return None, None
      assert len(column_ji0) == 2
      assert 0 <= column_ji0[0] < source_grid.nj and 0 <= column_ji0[1] < source_grid.ni
      cols_ji0 = np.array(column_ji0, dtype = int).reshape((1, 2))
      if not well_name:
         well_name = 'well for global column ' + str(column_ji0[1] + 1) + ', ' + str(column_ji0[0] + 1)
   # build up mask info
   # todo: handle base interfaces above k gaps
   h_or_l = 'layer' if trajectory is None else 'horizon'
   end_k0 = max_k0 + 1 if trajectory is None else max_k0 + 2
   for k in range(min_k0, end_k0):
      if trajectory is None:
         if blocked_well is None:
            cols, intersect_points = cols_ji0, centres[k, column_ji0[0], column_ji0[1]].reshape((1, 3))
         else:
            selected_cells = np.where(bw_cells[:, 0] == k)[0]
            cells = bw_cells[selected_cells]
            cols = cells[:, 1:]
            intersect_points = centres[cells[:, 0], cells[:, 1], cells[:, 2]]
      else:
         if k < source_grid.nk:
            cols, intersect_points = rgs.find_intersections_of_trajectory_with_layer_interface(
               trajectory, source_grid, k0 = k, ref_k_faces = 'top', quad_triangles = quad_triangles)
         else:
            cols, intersect_points = rgs.find_intersections_of_trajectory_with_layer_interface(
               trajectory, source_grid, k0 = k - 1, ref_k_faces = 'base', quad_triangles = quad_triangles)
      if cols is None or len(cols) == 0:
         if not warned:
            log.warning(f"no intersection found between well and {h_or_l}(s) such as: {k}")
            warned = True
         continue
      count = cols.shape[0]
      assert len(intersect_points) == count
      if count > 1:
         log.warning(f"{count} intersections found between well and {h_or_l}: {k}")
      layer_mask = np.zeros((source_grid.nj, source_grid.ni), dtype = bool)  # to be set True within radius
      if outer_radius is not None:
         outer_layer_mask = np.ones((source_grid.nj, source_grid.ni),
                                    dtype = bool)  # to be set False within outer_radius
      for intersect in range(count):
         log.debug(f"well intersects {h_or_l} {k} in column j0,i0: {cols[intersect, 0]}, {cols[intersect, 1]}")
         if radius > 0.0 or outer_radius is not None:
            if k < source_grid.nk:
               vectors = centres[k] - intersect_points[intersect].reshape((1, 1, 3))
               distance_sqr = vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1]
               if radius > 0.0:
                  layer_mask = np.logical_or(layer_mask, np.less_equal(distance_sqr, radius_sqr))
               if outer_radius is not None:
                  outer_layer_mask = np.logical_and(outer_layer_mask, np.greater_equal(distance_sqr, outer_radius_sqr))
            if k > 0 and (not source_grid.k_gaps or k >= source_grid.nk - 1 or
                          not source_grid.k_gap_after_array[k - 1]):
               vectors = centres[k - 1] - intersect_points[intersect].reshape((1, 1, 3))
               distance_sqr = vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1]
               if radius > 0.0:
                  layer_mask = np.logical_or(layer_mask, np.less_equal(distance_sqr, radius_sqr))
               if outer_radius is not None:
                  outer_layer_mask = np.logical_and(outer_layer_mask, np.greater_equal(distance_sqr, outer_radius_sqr))
         layer_mask[cols[intersect, 0], cols[intersect, 1]] = True
      if k <= max_k0:
         inclusion_mask[k] = layer_mask
         if outer_radius is not None:
            outer_inactive_mask[k] = outer_layer_mask
      if k > min_k0:
         inclusion_mask[k - 1] = np.logical_or(inclusion_mask[k - 1], layer_mask)
         if outer_radius is not None:
            outer_inactive_mask[k - 1] = np.logical_and(outer_inactive_mask[k - 1], outer_layer_mask)
      log.debug(f"number of columns found in {h_or_l} {k} within radius around well: {np.count_nonzero(layer_mask)}")
   inc_count = np.count_nonzero(inclusion_mask)
   if inc_count == 0:
      log.error('no cells found within search radius around well')
      return None, None
   log.info('total number of cells found within radius around well: ' + str(inc_count))
   # derive box and inactive mask from inclusion mask
   min_j0 = 0
   while min_j0 < source_grid.nj - 1 and not np.any(inclusion_mask[:, min_j0, :]):
      min_j0 += 1
   max_j0 = source_grid.nj - 1
   while max_j0 > 0 and not np.any(inclusion_mask[:, max_j0, :]):
      max_j0 -= 1
   assert max_j0 >= min_j0
   min_i0 = 0
   while min_i0 < source_grid.ni - 1 and not np.any(inclusion_mask[:, :, min_i0]):
      min_i0 += 1
   max_i0 = source_grid.ni - 1
   while max_i0 > 0 and not np.any(inclusion_mask[:, :, max_i0]):
      max_i0 -= 1
   assert max_i0 >= min_i0
   box = np.array([[min_k0, min_j0, min_i0], [max_k0, max_j0, max_i0]], dtype = int)
   log.info('box for well is: ' + bx.string_iijjkk1_for_box_kji0(box) + ' (simulator protocol)')
   # prepare inactive mask to merge in for new grid
   if active_cells_shape in ['tube', 'prism']:
      if active_cells_shape == 'prism':
         layer_mask = np.any(inclusion_mask, axis = 0)
         inclusion_mask[:] = layer_mask
      box_inactive = np.logical_not(inclusion_mask[min_k0:max_k0 + 1, min_j0:max_j0 + 1, min_i0:max_i0 + 1])
   else:  # 'box' option: leave all cells active (except where inactive in source grid)
      box_inactive = None

   if not new_grid_title:
      if trajectory is not None:
         new_grid_title = 'local grid extracted for well: ' + rqet.citation_title_for_node(trajectory_root)
      elif blocked_well is not None:
         new_grid_title = 'local grid extracted for blocked well: ' + rqet.citation_title_for_node(bw_root)
      elif column_ji0 is not None:
         new_grid_title = 'local grid extracted around column i, j (1 based): ' +  \
                          str(column_ji0[1] + 1) + ', ' + str(column_ji0[0] + 1)
      else:  # should not happen
         new_grid_title = 'local grid extracted for well'

   grid = extract_box(epc_file,
                      source_grid = source_grid,
                      box = box,
                      box_inactive = box_inactive,
                      inherit_properties = inherit_properties,
                      inherit_realization = inherit_realization,
                      inherit_all_realizations = inherit_all_realizations,
                      set_parent_window = set_parent_window,
                      new_grid_title = new_grid_title,
                      new_epc_file = new_epc_file)

   if inherit_well and new_epc_file:
      newer_model = rq.Model(new_epc_file)
      if trajectory is None and blocked_well is None:
         log.info('creating well objects for column')
         box_column_ji0 = (column_ji0[0] - box[0, 1], column_ji0[1] - box[0, 2])
         bw = rqw.BlockedWell(newer_model,
                              grid = grid,
                              column_ji0 = box_column_ji0,
                              well_name = well_name,
                              use_face_centres = True)
         bw.write_hdf5(create_for_trajectory_if_needed = True)
         bw.create_xml(create_for_trajectory_if_needed = True, title = well_name)
      elif blocked_well is not None:
         log.info('inheriting trajectory for blocked well')
         newer_model.copy_part_from_other_model(traj_model,
                                                rqet.part_name_for_part_root(blocked_well.trajectory.root_node))
      else:
         log.info('inheriting well trajectory')
         newer_model.copy_part_from_other_model(
            traj_model, rqet.part_name_for_part_root(trajectory_root))  # recursively copies referenced parts
      newer_model.h5_release()
      newer_model.store_epc()

   if outer_radius is not None:
      if new_epc_file:
         # TODO: copy source grid and reassign source_grid to new copy
         outer_epc = new_epc_file
      else:
         outer_epc = epc_file
      # todo: make local property kind, or reuse active local property kind?
      add_one_grid_property_array(outer_epc,
                                  outer_inactive_mask,
                                  'discrete',
                                  grid_uuid = source_grid.uuid,
                                  source_info = 'extract box for well outer radius',
                                  title = 'distant mask for well ' + str(well_name),
                                  discrete = True,
                                  uom = source_grid.xy_units())

   return grid, box


def refined_grid(epc_file,
                 source_grid,
                 fine_coarse,
                 inherit_properties = False,
                 inherit_realization = None,
                 inherit_all_realizations = False,
                 source_grid_uuid = None,
                 set_parent_window = None,
                 infill_missing_geometry = True,
                 new_grid_title = None,
                 new_epc_file = None):
   """Generates a refined version of the source grid, optionally inheriting properties.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid unless source_grid_uuid is specified to identify the grid
      fine_coarse (resqpy.olio.fine_coarse.FineCoarse object): the mapping between cells in the fine (output) and
         coarse (source) grids
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid, with values resampled in the simplest way onto the finer grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      source_grid_uuid (uuid.UUID, optional): the uuid of the source grid – an alternative to the source_grid argument
         as a way of identifying the grid
      set_parent_window (boolean or str, optional): if True or 'parent', the refined grid has its parent window attribute
         set; if False, the parent window is not set; if None, the default will be True if new_epc_file is None or False
         otherwise; if 'grandparent' then an intervening parent window with no refinement or coarsening will be skipped
         and its box used in the parent window for the new grid, relating directly to the original grid
      infill_missing_geometry (boolean, default True): if True, an attempt is made to generate grid geometry in the
         source grid wherever it is undefined; if False, any undefined geometry will result in an assertion failure
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the refined grid (& crs)

   returns:
      new grid object being the refined grid; the epc and hdf5 files are written to as an intentional side effect

   notes:
      this function refines an entire grid; to refine a local area of a grid, first use the extract_box function
      and then use this function on the extracted grid; in such a case, using a value of 'grandparent' for the
      set_parent_window argument will relate the refined grid back to the original;
      if geometry infilling takes place, cached geometry and mask arrays within the source grid object will be
      modified as a side-effect of the function (but not written to hdf5 or changed in xml)
   """

   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if not epc_file:
      epc_file = source_grid.model.epc_file
      assert epc_file, 'unable to ascertain epc filename from grid object'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   if set_parent_window is None:
      set_parent_window = (new_epc_file is None)
   model = None
   if new_epc_file:
      log.debug('creating fresh model for refined grid')
      model = rq.Model(epc_file = new_epc_file, new_epc = True, create_basics = True, create_hdf5_ext = True)
   if epc_file:
      model_in = rq.Model(epc_file)
      if source_grid is None:
         if source_grid_uuid is None:
            log.debug('using default source grid from existing epc')
            source_grid = model_in.grid()
         else:
            log.debug('selecting source grid from existing epc based on uuid')
            source_grid = grr.Grid(model_in, uuid = source_grid_uuid)
      else:
         if source_grid_uuid is not None:
            assert bu.matching_uuids(source_grid_uuid, source_grid.uuid)
         grid_uuid = source_grid.uuid
         log.debug('reloading source grid from existing epc file')
         source_grid = grr.Grid(model_in, uuid = grid_uuid)
      if model is None:
         model = model_in
   else:
      model_in = source_grid.model
   assert model_in is not None
   assert model is not None
   assert source_grid is not None
   assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']
   assert fine_coarse is not None and isinstance(fine_coarse, fc.FineCoarse)

   if infill_missing_geometry and (not source_grid.geometry_defined_for_all_cells() or
                                   not source_grid.geometry_defined_for_all_pillars()):
      log.debug('attempting infill of geometry missing in source grid')
      source_grid.set_geometry_is_defined(treat_as_nan = None,
                                          treat_dots_as_nan = True,
                                          complete_partial_pillars = True,
                                          nullify_partial_pillars = False,
                                          complete_all = True)
   assert source_grid.geometry_defined_for_all_pillars(), 'refinement requires geometry to be defined for all pillars'
   assert source_grid.geometry_defined_for_all_cells(), 'refinement requires geometry to be defined for all cells'

   assert tuple(fine_coarse.coarse_extent_kji) == tuple(source_grid.extent_kji),  \
          'fine_coarse mapping coarse extent does not match that of source grid'
   fine_coarse.assert_valid()

   source_grid.cache_all_geometry_arrays()
   if source_grid.has_split_coordinate_lines:
      source_grid.create_column_pillar_mapping()
   source_points = source_grid.points_ref()
   assert source_points is not None

   if model is not model_in:
      crs_part = model_in.part_for_uuid(source_grid.crs_uuid)
      assert crs_part is not None
      model.copy_part_from_other_model(model_in, crs_part)

   # todo: set nan-abled numpy operations?

   if source_grid.has_split_coordinate_lines:

      source_grid.corner_points(cache_cp_array = True)
      fnk, fnj, fni = fine_coarse.fine_extent_kji
      fine_cp = np.empty((fnk, fnj, fni, 2, 2, 2, 3))
      for ck0 in range(source_grid.nk):
         fine_k_base = fine_coarse.fine_base_for_coarse_axial(0, ck0)
         k_ratio = fine_coarse.ratio(0, ck0)
         k_interp = np.ones((k_ratio + 1,))
         k_interp[:-1] = fine_coarse.interpolation(0, ck0)
         for cj0 in range(source_grid.nj):
            fine_j_base = fine_coarse.fine_base_for_coarse_axial(1, cj0)
            j_ratio = fine_coarse.ratio(1, cj0)
            j_interp = np.ones((j_ratio + 1,))
            j_interp[:-1] = fine_coarse.interpolation(1, cj0)
            for ci0 in range(source_grid.ni):
               fine_i_base = fine_coarse.fine_base_for_coarse_axial(2, ci0)
               i_ratio = fine_coarse.ratio(2, ci0)
               i_interpolation = fine_coarse.interpolation(2, ci0)
               i_interp = np.ones((i_ratio + 1,))
               i_interp[:-1] = fine_coarse.interpolation(2, ci0)

               shared_fine_points = source_grid.interpolated_points((ck0, cj0, ci0), (k_interp, j_interp, i_interp))

               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 0, 0] =  \
                  shared_fine_points[:-1, :-1, :-1]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 0, 1] =  \
                  shared_fine_points[:-1, :-1, 1:]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 1, 0] =  \
                  shared_fine_points[:-1, 1:, :-1]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 1, 1] =  \
                  shared_fine_points[:-1, 1:, 1:]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 0, 0] =  \
                  shared_fine_points[1:, :-1, :-1]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 0, 1] =  \
                  shared_fine_points[1:, :-1, 1:]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 1, 0] =  \
                  shared_fine_points[1:, 1:, :-1]
               fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 1, 1] =  \
                  shared_fine_points[1:, 1:, 1:]

      grid = rqi.grid_from_cp(model,
                              fine_cp,
                              source_grid.crs_uuid,
                              ijk_handedness = 'right' if source_grid.grid_is_right_handed else 'left')

   else:

      # create a new, empty grid object
      grid = grr.Grid(model)

      # inherit attributes from source grid
      grid.grid_representation = 'IjkGrid'
      grid.extent_kji = fine_coarse.fine_extent_kji
      grid.nk, grid.nj, grid.ni = grid.extent_kji[0], grid.extent_kji[1], grid.extent_kji[2]
      grid.k_direction_is_down = source_grid.k_direction_is_down
      grid.grid_is_right_handed = source_grid.grid_is_right_handed
      grid.pillar_shape = source_grid.pillar_shape
      grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
      grid.split_pillars_count = source_grid.split_pillars_count
      grid.k_gaps = source_grid.k_gaps
      if grid.k_gaps:
         grid.nk_plus_k_gaps = grid.nk + grid.k_gaps
         grid.k_gap_after_array = np.zeros((grid.nk - 1,), dtype = bool)
         grid.k_raw_index_array = np.zeros((grid.nk,), dtype = int)
         # k gap arrays populated below
      else:
         grid.nk_plus_k_gaps = grid.nk
      # inherit the coordinate reference system used by the grid geometry
      grid.crs_root = source_grid.crs_root
      grid.crs_uuid = source_grid.crs_uuid

      refined_points = np.empty((grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1, 3))

      #      log.debug(f'source grid: {source_grid.extent_kji}; k gaps: {source_grid.k_gaps}')
      #      log.debug(f'refined grid: {grid.extent_kji}; k gaps: {grid.k_gaps}')
      fk0 = 0
      gaps_so_far = 0
      for ck0 in range(fine_coarse.coarse_extent_kji[0] + 1):
         end_k = (ck0 == fine_coarse.coarse_extent_kji[0])
         if end_k:
            k_ratio = 1
            k_interpolation = [0.0]
         else:
            k_ratio = fine_coarse.ratio(0, ck0)
            k_interpolation = fine_coarse.interpolation(0, ck0)
         one_if_gap = 1 if source_grid.k_gaps and ck0 < fine_coarse.coarse_extent_kji[
            0] - 1 and source_grid.k_gap_after_array[ck0] else 0
         for flk0 in range(k_ratio + one_if_gap):
            #            log.debug(f'ck0: {ck0}; fk0: {fk0}; flk0: {flk0}; k_ratio: {k_ratio}; one_if_gap: {one_if_gap}; gaps so far: {gaps_so_far}')
            if flk0 < k_ratio:
               k_fraction = k_interpolation[flk0]
            else:
               k_fraction = 1.0
            if grid.k_gaps:
               if end_k:
                  k_plane = source_points[source_grid.k_raw_index_array[ck0 - 1] + 1, :, :, :]
               else:
                  k_plane = (k_fraction * source_points[source_grid.k_raw_index_array[ck0] + 1, :, :, :] +
                             (1.0 - k_fraction) * source_points[source_grid.k_raw_index_array[ck0], :, :, :])
               if flk0 == k_ratio:
                  grid.k_gap_after_array[fk0 - 1] = True
               elif fk0 < grid.nk:
                  grid.k_raw_index_array[fk0] = fk0 + gaps_so_far
            else:
               if end_k:
                  k_plane = source_points[ck0, :, :, :]
               else:
                  k_plane = k_fraction * source_points[ck0 + 1, :, :, :] + (1.0 -
                                                                            k_fraction) * source_points[ck0, :, :, :]
            fj0 = 0
            for cj0 in range(fine_coarse.coarse_extent_kji[1] + 1):
               end_j = (cj0 == fine_coarse.coarse_extent_kji[1])
               if end_j:
                  j_ratio = 1
                  j_interpolation = [0.0]
               else:
                  j_ratio = fine_coarse.ratio(1, cj0)
                  j_interpolation = fine_coarse.interpolation(1, cj0)
               for flj0 in range(j_ratio):
                  j_fraction = j_interpolation[flj0]
                  # note: shape of j_line will be different if there are split pillars in play
                  if end_j:
                     j_line = k_plane[cj0, :, :]
                  else:
                     j_line = j_fraction * k_plane[cj0 + 1, :, :] + (1.0 - j_fraction) * k_plane[cj0, :, :]

                  fi0 = 0
                  for ci0 in range(fine_coarse.coarse_extent_kji[2] + 1):
                     end_i = (ci0 == fine_coarse.coarse_extent_kji[2])
                     if end_i:
                        i_ratio = 1
                        i_interpolation = [0.0]
                     else:
                        i_ratio = fine_coarse.ratio(2, ci0)
                        i_interpolation = fine_coarse.interpolation(2, ci0)
                     for fli0 in range(i_ratio):
                        i_fraction = i_interpolation[fli0]
                        if end_i:
                           p = j_line[ci0, :]
                        else:
                           p = i_fraction * j_line[ci0 + 1, :] + (1.0 - i_fraction) * j_line[ci0, :]

                        refined_points[fk0 + gaps_so_far, fj0, fi0] = p

                        fi0 += 1

                  assert fi0 == fine_coarse.fine_extent_kji[2] + 1

                  fj0 += 1

            assert fj0 == fine_coarse.fine_extent_kji[1] + 1

            if flk0 == k_ratio:
               gaps_so_far += 1
            else:
               fk0 += 1

      assert fk0 == fine_coarse.fine_extent_kji[0] + 1
      assert grid.nk + gaps_so_far == grid.nk_plus_k_gaps

      grid.points_cached = refined_points

      grid.geometry_defined_for_all_pillars_cached = True
      grid.geometry_defined_for_all_cells_cached = True
      grid.array_cell_geometry_is_defined = np.full(tuple(grid.extent_kji), True, dtype = bool)

   # todo: option of re-draping interpolated pillars to surface

   collection = None
   if inherit_properties:
      source_collection = source_grid.extract_property_collection()
      if source_collection is not None:
         #  do not inherit the inactive property array by this mechanism
         collection = rqp.GridPropertyCollection()
         collection.set_grid(grid)
         collection.extend_imported_list_copying_properties_from_other_grid_collection(
            source_collection,
            refinement = fine_coarse,
            realization = inherit_realization,
            copy_all_realizations = inherit_all_realizations)

   if set_parent_window:
      pw_grid_uuid = source_grid.uuid
      if isinstance(set_parent_window, str):
         if set_parent_window == 'grandparent':
            assert fine_coarse.within_coarse_box is None or (np.all(fine_coarse.within_coarse_box[0] == 0) and
                                                             np.all(fine_coarse.within_coarse_box[1]) == source_grid.extent_kji - 1),  \
               'attempt to set grandparent window for grid when parent window is present'
            source_fine_coarse = source_grid.parent_window
            if source_fine_coarse is not None and (source_fine_coarse.within_fine_box is not None or
                                                   source_fine_coarse.within_coarse_box is not None):
               assert source_fine_coarse.fine_extent_kji == source_fine_coarse.coarse_extent_kji, 'parentage involves refinement or coarsening'
               if source_fine_coarse.within_coarse_box is not None:
                  fine_coarse.within_coarse_box = source_fine_coarse.within_coarse_box
               else:
                  fine_coarse.within_coarse_box = source_fine_coarse.within_fine_box
               pw_grid_uuid = bu.uuid_from_string(
                  rqet.find_nested_tags_text(source_grid.root, ['ParentWindow', 'ParentGrid', 'UUID']))
         else:
            assert set_parent_window == 'parent', 'set_parent_window value not recognized: ' + set_parent_window
      grid.set_parent(pw_grid_uuid, True, fine_coarse)

   # write grid
   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'grid refined from ' + str(rqet.citation_title_for_node(source_grid.root))

   model.h5_release()
   if model is not model_in:
      model_in.h5_release()

   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def coarsened_grid(epc_file,
                   source_grid,
                   fine_coarse,
                   inherit_properties = False,
                   inherit_realization = None,
                   inherit_all_realizations = False,
                   set_parent_window = None,
                   infill_missing_geometry = True,
                   new_grid_title = None,
                   new_epc_file = None):
   """Generates a coarsened version of an unsplit source grid, todo: optionally inheriting properties.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      fine_coarse (resqpy.olio.fine_coarse.FineCoarse object): the mapping between cells in the fine (source) and
         coarse (output) grids
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid, with values upscaled or sampled
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      set_parent_window (boolean or str, optional): if True or 'parent', the coarsened grid has its parent window attribute
         set; if False, the parent window is not set; if None, the default will be True if new_epc_file is None or False
         otherwise; if 'grandparent' then an intervening parent window with no refinement or coarsening will be skipped
         and its box used in the parent window for the new grid, relating directly to the original grid
      infill_missing_geometry (boolean, default True): if True, an attempt is made to generate grid geometry in the
         source grid wherever it is undefined; if False, any undefined geometry will result in an assertion failure
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the refined grid (& crs)

   returns:
      new grid object being the coarsened grid; the epc and hdf5 files are written to as an intentional side effect

   note:
      this function coarsens an entire grid; to coarsen a local area of a grid, first use the extract_box function
      and then use this function on the extracted grid; in such a case, using a value of 'grandparent' for the
      set_parent_window argument will relate the coarsened grid back to the original
   """

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if set_parent_window is None:
      set_parent_window = (new_epc_file is None)
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model (or one named 'ROOT')
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None
   assert fine_coarse is not None and isinstance(fine_coarse, fc.FineCoarse)

   assert not source_grid.has_split_coordinate_lines, 'coarsening only available for unsplit grids: use other functions to heal faults first'

   if infill_missing_geometry and (not source_grid.geometry_defined_for_all_cells() or
                                   not source_grid.geometry_defined_for_all_pillars()):
      log.debug('attempting infill of geometry missing in source grid')
      source_grid.set_geometry_is_defined(treat_as_nan = None,
                                          treat_dots_as_nan = True,
                                          complete_partial_pillars = True,
                                          nullify_partial_pillars = False,
                                          complete_all = True)

   assert source_grid.geometry_defined_for_all_pillars(), 'coarsening requires geometry to be defined for all pillars'
   assert source_grid.geometry_defined_for_all_cells(), 'coarsening requires geometry to be defined for all cells'
   assert not source_grid.k_gaps, 'coarsening of grids with k gaps not currently supported'

   assert tuple(fine_coarse.fine_extent_kji) == tuple(source_grid.extent_kji),  \
          'fine_coarse mapping fine extent does not match that of source grid'
   fine_coarse.assert_valid()

   source_grid.cache_all_geometry_arrays()
   source_points = source_grid.points_ref().reshape((source_grid.nk + 1), (source_grid.nj + 1) * (source_grid.ni + 1),
                                                    3)

   # create a new, empty grid object
   grid = grr.Grid(model)

   # inherit attributes from source grid
   grid.grid_representation = 'IjkGrid'
   grid.extent_kji = fine_coarse.coarse_extent_kji
   grid.nk, grid.nj, grid.ni = grid.extent_kji[0], grid.extent_kji[1], grid.extent_kji[2]
   grid.nk_plus_k_gaps = source_grid.nk_plus_k_gaps  # k_gaps not currently supported anyway
   grid.k_direction_is_down = source_grid.k_direction_is_down
   grid.grid_is_right_handed = source_grid.grid_is_right_handed
   grid.pillar_shape = source_grid.pillar_shape
   grid.has_split_coordinate_lines = False
   grid.split_pillars_count = None
   # inherit the coordinate reference system used by the grid geometry
   grid.crs_root = source_grid.crs_root
   grid.crs_uuid = source_grid.crs_uuid

   coarsened_points = np.empty(
      (grid.nk + 1, (grid.nj + 1) * (grid.ni + 1), 3))  # note: gets reshaped after being populated

   k_ratio_constant = fine_coarse.constant_ratios[0]
   if k_ratio_constant:
      k_indices = None
   else:
      k_indices = np.empty(grid.nk + 1, dtype = int)
      k_indices[0] = 0
      for k in range(grid.nk):
         k_indices[k + 1] = k_indices[k] + fine_coarse.vector_ratios[0][k]
      assert k_indices[-1] == source_grid.nk

   for cjp in range(grid.nj + 1):
      for cji in range(grid.ni + 1):
         natural_coarse_pillar = cjp * (grid.ni + 1) + cji
         natural_fine_pillar = fine_coarse.fine_for_coarse_natural_pillar_index(natural_coarse_pillar)
         if k_ratio_constant:
            coarsened_points[:, natural_coarse_pillar, :] = source_points[0:source_grid.nk + 1:k_ratio_constant,
                                                                          natural_fine_pillar, :]
         else:
            coarsened_points[:, natural_coarse_pillar, :] = source_points[k_indices, natural_fine_pillar, :]

   grid.points_cached = coarsened_points.reshape(((grid.nk + 1), (grid.nj + 1), (grid.ni + 1), 3))

   grid.geometry_defined_for_all_pillars_cached = True
   grid.geometry_defined_for_all_cells_cached = True
   grid.array_cell_geometry_is_defined = np.full(tuple(grid.extent_kji), True, dtype = bool)

   collection = None
   if inherit_properties:
      source_collection = source_grid.extract_property_collection()
      if source_collection is not None:
         collection = rqp.GridPropertyCollection()
         collection.set_grid(grid)
         collection.extend_imported_list_copying_properties_from_other_grid_collection(
            source_collection,
            coarsening = fine_coarse,
            realization = inherit_realization,
            copy_all_realizations = inherit_all_realizations)

   if set_parent_window:
      pw_grid_uuid = source_grid.uuid
      if isinstance(set_parent_window, str):
         if set_parent_window == 'grandparent':
            assert fine_coarse.within_fine_box is None or (np.all(fine_coarse.within_fine_box[0] == 0) and
                                                           np.all(fine_coarse.within_fine_box[1]) == source_grid.extent_kji - 1),  \
               'attempt to set grandparent window for grid when parent window is present'
            source_fine_coarse = source_grid.parent_window
            if source_fine_coarse is not None and (source_fine_coarse.within_fine_box is not None or
                                                   source_fine_coarse.within_coarse_box is not None):
               assert source_fine_coarse.fine_extent_kji == source_fine_coarse.coarse_extent_kji, 'parentage involves refinement or coarsening'
               if source_fine_coarse.within_fine_box is not None:
                  fine_coarse.within_fine_box = source_fine_coarse.within_fine_box
               else:
                  fine_coarse.within_fine_box = source_fine_coarse.within_coarse_box
               pw_grid_uuid = bu.uuid_from_string(
                  rqet.find_nested_tags_text(source_grid.root, ['ParentWindow', 'ParentGrid', 'UUID']))
         else:
            assert set_parent_window == 'parent', 'set_parent_window value not recognized: ' + set_parent_window
      grid.set_parent(pw_grid_uuid, False, fine_coarse)

   # write grid
   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'grid coarsened from ' + str(rqet.citation_title_for_node(source_grid.root))

   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def local_depth_adjustment(epc_file,
                           source_grid,
                           centre_x,
                           centre_y,
                           radius,
                           centre_shift,
                           use_local_coords,
                           decay_shape = 'quadratic',
                           ref_k0 = 0,
                           store_displacement = False,
                           inherit_properties = False,
                           inherit_realization = None,
                           inherit_all_realizations = False,
                           new_grid_title = None,
                           new_epc_file = None):
   """Applies a local depth adjustment to the grid, adding as a new grid part in the model.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): a multi-layer RESQML grid object; if None, the epc_file is loaded
         and it should contain one ijk grid object (or one 'ROOT' grid) which is used as the source grid
      centre_x, centre_y (floats): the centre of the depth adjustment, corresponding to the location of maximum change
         in depth; crs is implicitly that of the grid but see also use_local_coords argument
      radius (float): the radius of adjustment of depths; units are implicitly xy (projected) units of grid crs
      centre_shift (float): the maximum vertical depth adjustment; units are implicily z (vertical) units of grid crs;
         use positive value to increase depth, negative to make shallower
      use_local_coords (boolean): if True, centre_x & centre_y are taken to be in the local coordinates of the grid's
         crs; otherwise the global coordinates
      decay_shape (string): 'linear' yields a cone shaped change in depth values; 'quadratic' (the default) yields a
         bell shaped change
      ref_k0 (integer, default 0): the layer in the grid to use as reference for determining the distance of a pillar
         from the centre of the depth adjustment; the corners of the top face of the reference layer are used
      store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
         displacement of cells' centres brought about by the local depth shift
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the adjusted grid (& crs)

   returns:
      new grid object which is a copy of the source grid with the local depth adjustment applied
   """

   def decayed_shift(centre_shift, distance, radius, decay_shape):
      norm_dist = min(distance / radius, 1.0)  # 0..1
      if decay_shape == 'linear':
         return (1.0 - norm_dist) * centre_shift
      elif decay_shape == 'quadratic':
         if norm_dist >= 0.5:
            x = (1.0 - norm_dist)
            return 2.0 * x * x * centre_shift
         else:
            return centre_shift * (1.0 - 2.0 * norm_dist * norm_dist)
      else:
         raise ValueError('unrecognized decay shape: ' + decay_shape)

   log.info('adjusting depth')
   log.debug('centre x: {0:3.1f}; y: {1:3.1f}'.format(centre_x, centre_y))
   if use_local_coords:
      log.debug('centre x & y interpreted in local crs')
   log.debug('radius of influence: {0:3.1f}'.format(radius))
   log.debug('depth shift at centre: {0:5.3f}'.format(centre_shift))
   log.debug('decay shape: ' + decay_shape)
   log.debug('reference layer (k0 protocol): ' + str(ref_k0))

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None

   # take a copy of the grid
   grid = copy_grid(source_grid, model)

   # if not use_local_coords, convert centre_x & y into local_coords
   crs_root = grid.extract_crs_root()
   assert (crs_root is not None)
   if not use_local_coords:
      rotation = abs(float(rqet.node_text(rqet.find_tag(crs_root, 'ArealRotation'))))
      if rotation > 0.001:
         log.error('unable to account for rotation in crs: use local coordinates')
         return
      centre_x -= float(rqet.node_text(rqet.find_tag(crs_root, 'XOffset')))
      centre_y -= float(rqet.node_text(rqet.find_tag(crs_root, 'YOffset')))
   z_inc_down = rqet.bool_from_text(rqet.node_text(rqet.find_tag(crs_root, 'ZIncreasingDownward')))

   if not z_inc_down:
      centre_shift = -centre_shift

   # cache geometry in memory; needed prior to writing new coherent set of arrays to hdf5
   grid.cache_all_geometry_arrays()
   if grid.has_split_coordinate_lines:
      reshaped_points = grid.points_cached.copy()
   else:
      nkp1, njp1, nip1, xyz = grid.points_cached.shape
      reshaped_points = grid.points_cached.copy().reshape((nkp1, njp1 * nip1, xyz))
   assert reshaped_points.ndim == 3 and reshaped_points.shape[2] == 3
   assert ref_k0 >= 0 and ref_k0 < reshaped_points.shape[0]

   log.debug('reshaped_points.shape: ' + str(reshaped_points.shape))

   log.debug('min z before depth adjustment: ' + str(np.nanmin(reshaped_points[:, :, 2])))

   # for each pillar, find x, y for k = reference_layer_k0
   pillars_adjusted = 0

   # todo: replace with numpy array operations
   radius_sqr = radius * radius
   for pillar in range(reshaped_points.shape[1]):
      x, y, z = tuple(reshaped_points[ref_k0, pillar, :])
      # find distance of this pillar from the centre
      dx = centre_x - x
      dy = centre_y - y
      distance_sqr = (dx * dx) + (dy * dy)
      # if this pillar is beyond radius of influence, no action needed
      if distance_sqr > radius_sqr:
         continue
      distance = maths.sqrt(distance_sqr)
      # compute decayed shift as function of distance
      shift = decayed_shift(centre_shift, distance, radius, decay_shape)
      # adjust depth values for pillar in cached array
      log.debug('adjusting pillar number {0} at x: {1:3.1f}, y: {2:3.1f}, distance: {3:3.1f} by {4:5.3f}'.format(
         pillar, x, y, distance, shift))
      reshaped_points[:, pillar, 2] += shift
      pillars_adjusted += 1

   # if no pillars adjusted: warn and return
   if pillars_adjusted == 0:
      log.warning('no pillars adjusted')
      return

   log.debug('min z after depth adjustment: ' + str(np.nanmin(reshaped_points[:, :, 2])))
   if grid.has_split_coordinate_lines:
      grid.points_cached[:] = reshaped_points
   else:
      grid.points_cached[:] = reshaped_points.reshape((nkp1, njp1, nip1, xyz))


#   model.copy_part(old_uuid, grid.uuid, change_hdf5_refs = True)   # copies the xml, substituting the new uuid in the root node (and in hdf5 refs)
   log.info(str(pillars_adjusted) + ' pillars adjusted')

   # build cell displacement property array(s)
   if store_displacement:
      log.debug('generating cell displacement property arrays')
      displacement_collection = displacement_properties(grid, source_grid)
   else:
      displacement_collection = None

   collection = _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                            inherit_all_realizations)
   if collection is None:
      collection = displacement_collection
   elif displacement_collection is not None:
      collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'grid derived from {0} with local depth shift of {1:3.1f} applied'.format(
         str(rqet.citation_title_for_node(source_grid.root)), centre_shift)

   # write model
   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def tilted_grid(epc_file,
                source_grid = None,
                pivot_xyz = None,
                azimuth = None,
                dip = None,
                store_displacement = False,
                inherit_properties = False,
                inherit_realization = None,
                inherit_all_realizations = False,
                new_grid_title = None,
                new_epc_file = None):
   """Extends epc file with a new grid which is a version of the source grid tilted.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      pivot_xyz (triple float): a point in 3D space on the pivot axis, which is horizontal and orthogonal to azimuth
      azimuth: the direction of tilt (orthogonal to tilt axis), as a compass bearing in degrees
      dip: the angle to tilt the grid by, in degrees; a positive value tilts points in direction azimuth downwards (needs checking!)
      store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
         displacement of cells' centres brought about by the tilting
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the tilted grid (& crs)

   returns:
      a new grid (grid.Grid object) which is a copy of the source grid tilted in 3D space
   """

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None

   # take a copy of the grid
   grid = copy_grid(source_grid, model)

   if grid.inactive is not None:
      log.debug('copied grid inactive shape: ' + str(grid.inactive.shape))

   # tilt the grid
   grid.cache_all_geometry_arrays()  # probably already cached anyway
   vec.tilt_points(pivot_xyz, azimuth, dip, grid.points_cached)

   # build cell displacement property array(s)
   if store_displacement:
      displacement_collection = displacement_properties(grid, source_grid)
   else:
      displacement_collection = None

   collection = _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                            inherit_all_realizations)
   if collection is None:
      collection = displacement_collection
   elif displacement_collection is not None:
      collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'tilted version ({0:4.2f} degree dip) of '.format(abs(dip)) + str(
         rqet.citation_title_for_node(source_grid.root))

   # write model
   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def unsplit_grid(epc_file,
                 source_grid = None,
                 inherit_properties = False,
                 inherit_realization = None,
                 inherit_all_realizations = False,
                 new_grid_title = None,
                 new_epc_file = None):
   """Extends epc file with a new grid which is a version of the source grid with all faults healed.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the unsplit grid (& crs)

   returns:
      a new grid (grid.Grid object) which is an unfaulted copy of the source grid

   notes:
      the faults are healed by shifting the thrown sides up and down to the midpoint, only along the line of the fault;
      to smooth the adjustments away from the line of the fault, use the global_fault_throw_scaling() function first
   """

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None

   assert source_grid.has_split_coordinate_lines, 'source grid is unfaulted'

   # take a copy of the grid
   grid = copy_grid(source_grid, model)

   if grid.inactive is not None:
      log.debug('copied grid inactive shape: ' + str(grid.inactive.shape))

   # heal faults in the grid
   grid.cache_all_geometry_arrays()  # probably already cached anyway
   unsplit = source_grid.unsplit_points_ref()
   grid.points_cached = unsplit.copy()
   assert grid.points_cached.shape == (grid.nk + 1, grid.nj + 1, grid.ni + 1, 3), 'unsplit points have incorrect shape'

   grid.has_split_coordinate_lines = False
   delattr(grid, 'split_pillar_indices_cached')
   delattr(grid, 'cols_for_split_pillars')
   delattr(grid, 'cols_for_split_pillars_cl')
   if hasattr(grid, 'pillars_for_column'):
      delattr(grid, 'pillars_for_column')

   collection = _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                            inherit_all_realizations)
   # todo: recompute depth properties (and volumes, cell lengths etc. if being strict)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'unfaulted version of ' + str(rqet.citation_title_for_node(source_grid.root))

   # write model
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def add_faults(epc_file,
               source_grid,
               polylines = None,
               lines_file_list = None,
               lines_crs_uuid = None,
               full_pillar_list_dict = None,
               left_right_throw_dict = None,
               create_gcs = True,
               inherit_properties = False,
               inherit_realization = None,
               inherit_all_realizations = False,
               new_grid_title = None,
               new_epc_file = None):
   """Extends epc file with a new grid which is a version of the source grid with new curtain fault(s) added.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      polylines (lines.PolylineSet or list of lines.Polyline, optional): list of poly lines for which curtain faults
         are to be added; either this or lines_file_list or full_pillar_list_dict must be present
      lines_file_list (list of str, optional): a list of file paths, each containing one or more poly lines in simple
         ascii format§; see notes; either this or polylines or full_pillar_list_dicr must be present
      lines_crs_uuid (uuid, optional): if present, the uuid of a coordinate reference system with which to interpret
         the contents of the lines files; if None, the crs used by the grid will be assumed
      full_pillar_list_dict (dict mapping str to list of pairs of ints, optional): dictionary mapping from a fault name
         to a list of pairs of ints being the ordered neigbouring primary pillar (j0, i0) defining the curtain fault;
         either this or polylines or lines_file_list must be present
      left_right_throw_dict (dict mapping str to pair of floats, optional): dictionary mapping from a fault name to a
         pair of floats being the semi-throw adjustment on the left and the right of the fault (see notes); semi-throw
         values default to (+0.5, -0.5)
      create_gcs (boolean, default True): if True, and faults are being defined by lines, a grid connection set is
         created with one feature per new fault and associated organisational objects are also created; ignored if
         lines_file_list is None
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the unsplit grid (& crs)

   returns:
      a new grid (grid.Grid object) which is a copy of the source grid with the structure modified to incorporate
      the new faults

   notes:
      full_pillar_list_dict is typically generated by Grid.make_face_sets_from_pillar_lists();
      pillars will be split as needed to model the new faults, though existing splits will be used as appropriate, so
      this function may also be used to add a constant to the throw of existing faults;
      the left_right_throw_dict contains a pair of floats for each fault name (as found in keys of full_pillar_list_dict);
      these throw values are lengths in the uom of the crs used by the grid (which must have the same xy units as z units);

      this function does not add a GridConnectionSet to the model – calling code may wish to do that
   """

   def make_face_sets_for_new_lines(new_lines, face_set_id, grid, full_pillar_list_dict, composite_face_set_dict):
      """Adds entries to full_pillar_list_dict & composite_face_set_dict for new lines."""
      pillar_list_list = sl.nearest_pillars(new_lines, grid)
      face_set_dict, full_pll_dict = grid.make_face_sets_from_pillar_lists(pillar_list_list, face_set_id)
      for key, pll in full_pll_dict.items():
         full_pillar_list_dict[key] = pll
      for key, fs_info in face_set_dict.items():
         composite_face_set_dict[key] = fs_info

   def fault_from_pillar_list(grid, full_pillar_list, delta_throw_left, delta_throw_right):
      """Creates and/or adjusts throw on a single fault defined by a full pillar list, in memory.

      arguments:
         grid (grid.Grid): the grid object to be adjusted in memory (should have originally been copied
            without the hdf5 arrays having been written yet, nor xml created)
         full_pillar_list (list of pairs of ints (j0, i0)): the full list of primary pillars defining
            the fault; neighbouring pairs must differ by exactly one in either j0 or i0 but not both
         delta_throw_left (float): the amount to add to the 'depth' of points to the left of the line
            when viewed from above, looking along the line in the direction of the pillar list entries;
            units are implicitly the length units of the crs used by the grid; see notes about 'depth'
         delta_throw_right (float): as for delta_throw_left but applied to points to the right of the
            line
      """

      def pillar_vector(grid, p_index):
         # return a unit vector for direction of pillar, in direction of increasing k
         if np.all(np.isnan(grid.points_cached[:, p_index])):
            return None
         k_top = 0
         while np.any(np.isnan(grid.points_cached[k_top, p_index])):
            k_top += 1
         k_bot = grid.nk_plus_k_gaps - 1
         while np.any(np.isnan(grid.points_cached[k_bot, p_index])):
            k_bot -= 1
         if k_bot == k_top:  # following coded to treat None directions as downwards
            if grid.k_direction_is_down is False:
               if grid.z_inc_down() is False:
                  return (0.0, 0.0, 1.0)
               else:
                  return (0.0, 0.0, -1.0)
            else:
               if grid.z_inc_down() is False:
                  return (0.0, 0.0, -1.0)
               else:
                  return (0.0, 0.0, 1.0)
         else:
            return vec.unit_vector(grid.points_cached[k_bot, p_index] - grid.points_cached[k_top, p_index])

      def extend_points_cached(grid, exist_p):
         s = grid.points_cached.shape
         e = np.empty((s[0], s[1] + 1, s[2]), dtype = float)
         e[:, :-1, :] = grid.points_cached
         e[:, -1, :] = grid.points_cached[:, exist_p, :]
         grid.points_cached = e

      def np_int_extended(a, i):
         e = np.empty(a.size + 1, dtype = int)
         e[:-1] = a
         e[-1] = i
         return e

      if full_pillar_list is None or len(full_pillar_list) < 3:
         return
      assert grid.z_units() == grid.xy_units()
      grid.cache_all_geometry_arrays()
      assert hasattr(grid, 'points_cached')
      if not grid.has_split_coordinate_lines:
         grid.points_cached = grid.points_cached.reshape((grid.nk_plus_k_gaps + 1, (grid.nj + 1) * (grid.ni + 1), 3))
         grid.split_pillar_indices_cached = np.array([], dtype = int)
         grid.cols_for_split_pillars = np.array([], dtype = int)
         grid.cols_for_split_pillars_cl = np.array([], dtype = int)
         grid.has_split_coordinate_lines = True
      assert grid.points_cached.ndim == 3
      if len(grid.cols_for_split_pillars_cl) == 0:
         cl = 0
      else:
         cl = grid.cols_for_split_pillars_cl[-1]
      original_p = np.zeros((grid.nk_plus_k_gaps + 1, 3), dtype = float)
      n_primaries = (grid.nj + 1) * (grid.ni + 1)
      for p_index in range(1, len(full_pillar_list) - 1):
         primary_ji0 = full_pillar_list[p_index]
         primary = primary_ji0[0] * (grid.ni + 1) + primary_ji0[1]
         p_vector = np.array(pillar_vector(grid, primary), dtype = float)
         if p_vector is None:
            continue
         throw_left_vector = np.expand_dims(delta_throw_left * p_vector, axis = 0)
         throw_right_vector = np.expand_dims(delta_throw_right * p_vector, axis = 0)
         #         log.debug(f'T: p ji0: {primary_ji0}; p vec: {p_vector}; left v: {throw_left_vector}; right v: {throw_right_vector}')
         existing_foursome = grid.pillar_foursome(primary_ji0, none_if_unsplit = False)
         lr_foursome = gf.left_right_foursome(full_pillar_list, p_index)
         p_j, p_i = primary_ji0
         #         log.debug(f'P: p ji0: {primary_ji0}; e foursome:\n{existing_foursome}; lr foursome:\n{lr_foursome}')
         for exist_p in np.unique(existing_foursome):
            exist_lr = None
            new_p_made = False
            for jp in range(2):
               if (p_j == 0 and jp == 0) or (p_j == grid.nj and jp == 1):
                  continue
               for ip in range(2):
                  if (p_i == 0 and ip == 0) or (p_i == grid.ni and ip == 1):
                     continue
                  if existing_foursome[jp, ip] != exist_p:
                     continue
                  if exist_lr is None:
                     original_p[:] = grid.points_cached[:, exist_p, :]
                     exist_lr = lr_foursome[jp, ip]
                     #                     log.debug(f'A: p ji0: {primary_ji0}; exist_p: {exist_p}; jp,ip: {(jp,ip)}; exist_lr: {exist_lr}')
                     grid.points_cached[:, exist_p, :] += throw_right_vector if exist_lr else throw_left_vector
                     continue
                  if lr_foursome[jp, ip] == exist_lr:
                     continue
                  natural_col = (p_j + jp - 1) * grid.ni + p_i + ip - 1
                  if exist_p != primary:  # remove one of the columns currently assigned to exist_p
                     extra_p = exist_p - n_primaries
                     #                     log.debug(f're-split: primary: {primary}; exist: {exist_p}; col: {natural_col}; extra: {extra_p}')
                     #                     log.debug(f'pre re-split: cols: {grid.cols_for_split_pillars}')
                     #                     log.debug(f'pre re-split: ccl:  {grid.cols_for_split_pillars_cl}')
                     assert grid.split_pillar_indices_cached[extra_p] == primary
                     if extra_p == 0:
                        start = 0
                     else:
                        start = grid.cols_for_split_pillars_cl[extra_p - 1]
                     found = False
                     for cols_index in range(start, start + grid.cols_for_split_pillars_cl[extra_p]):
                        if grid.cols_for_split_pillars[cols_index] == natural_col:
                           grid.cols_for_split_pillars = np.concatenate(
                              (grid.cols_for_split_pillars[:cols_index], grid.cols_for_split_pillars[cols_index + 1:]))
                           found = True
                           break
                     assert found
                     grid.cols_for_split_pillars_cl[extra_p:] -= 1
                     cl -= 1
                     assert grid.cols_for_split_pillars_cl[extra_p] > 0
#                     log.debug(f'post re-split: cols: {grid.cols_for_split_pillars}')
#                     log.debug(f'post re-split: ccl:  {grid.cols_for_split_pillars_cl}')
                  if not new_p_made:  # create a new split of pillar
                     extend_points_cached(grid, exist_p)
                     #                     log.debug(f'B: p ji0: {primary_ji0}; exist_p: {exist_p}; jp,ip: {(jp,ip)}; lr: {lr_foursome[jp, ip]}; c ji0: {natural_col}')
                     grid.points_cached[:, -1, :] = original_p + (throw_right_vector
                                                                  if lr_foursome[jp, ip] else throw_left_vector)
                     grid.split_pillar_indices_cached = np_int_extended(grid.split_pillar_indices_cached, primary)
                     if grid.split_pillars_count is None:
                        grid.split_pillars_count = 0
                     grid.split_pillars_count += 1
                     grid.cols_for_split_pillars = np_int_extended(grid.cols_for_split_pillars, natural_col)
                     cl += 1
                     grid.cols_for_split_pillars_cl = np_int_extended(grid.cols_for_split_pillars_cl, cl)
                     new_p_made = True
                  else:  # include this column in newly split version of pillar
                     #                     log.debug(f'C: p ji0: {primary_ji0}; exist_p: {exist_p}; jp,ip: {(jp,ip)}; lr: {lr_foursome[jp, ip]}; c ji0: {natural_col}')
                     grid.cols_for_split_pillars = np_int_extended(grid.cols_for_split_pillars, natural_col)
                     cl += 1
                     grid.cols_for_split_pillars_cl[-1] = cl

   log.info('adding faults')

   assert epc_file or new_epc_file, 'epc file name not specified'
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'  # RegularGrid not catered for
   assert model is not None
   assert len([arg for arg in (polylines, lines_file_list, full_pillar_list_dict) if arg is not None]) == 1

   # take a copy of the resqpy grid object, without writing to hdf5 or creating xml
   grid = copy_grid(source_grid, model)
   grid_crs = rqcrs.Crs(model, uuid = grid.crs_uuid)
   assert grid_crs is not None

   if isinstance(polylines, rql.PolylineSet):
      polylines = polylines.convert_to_polylines()

   if lines_crs_uuid is None:
      lines_crs = None
   else:
      lines_crs = rqcrs.Crs(model, uuid = lines_crs_uuid)

   if full_pillar_list_dict is None:
      full_pillar_list_dict = {}
      composite_face_set_dict = {}
      if polylines:
         for i, polyline in enumerate(polylines):
            new_line = polyline.coordinates.copy()
            if polyline.crs_uuid is not None and polyline.crs_uuid != lines_crs_uuid:
               lines_crs_uuid = polyline.crs_uuid
               lines_crs = rqcrs.Crs(model, uuid = lines_crs_uuid)
            if lines_crs:
               lines_crs.convert_array_to(grid_crs, new_line)
            title = polyline.title if polyline.title else 'fault_' + str(i)
            make_face_sets_for_new_lines([new_line], title, grid, full_pillar_list_dict, composite_face_set_dict)
      else:
         for filename in lines_file_list:
            new_lines = sl.read_lines(filename)
            if lines_crs is not None:
               for a in new_lines:
                  lines_crs.convert_array_to(grid_crs, a)
            _, f_name = os.path.split(filename)
            if f_name.lower().endswith('.dat'):
               face_set_id = f_name[:-4]
            else:
               face_set_id = f_name
            make_face_sets_for_new_lines(new_lines, face_set_id, grid, full_pillar_list_dict, composite_face_set_dict)


#   log.debug(f'full_pillar_list_dict:\n{full_pillar_list_dict}')

   for fault_key in full_pillar_list_dict:

      full_pillar_list = full_pillar_list_dict[fault_key]
      left_right_throw = None
      if left_right_throw_dict is not None:
         left_right_throw = left_right_throw_dict.get(fault_key)
      if left_right_throw is None:
         left_right_throw = (+0.5, -0.5)

      log.debug(
         f'generating fault {fault_key} pillar count {len(full_pillar_list)}; left, right throw {left_right_throw}')

      fault_from_pillar_list(grid, full_pillar_list, left_right_throw[0], left_right_throw[1])

   collection = _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                            inherit_all_realizations)
   # todo: recompute depth properties (and volumes, cell lengths etc. if being strict)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'copy of ' + str(rqet.citation_title_for_node(source_grid.root)) + ' with added faults'

   # write model
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   if create_gcs and (polylines is not None or lines_file_list is not None):
      if new_epc_file is not None:
         grid_uuid = grid.uuid
         model = rq.Model(new_epc_file)
         grid = grr.Grid(model, root = model.root(uuid = grid_uuid), find_properties = False)
      grid.set_face_set_gcs_list_from_dict(composite_face_set_dict, create_organizing_objects_where_needed = True)
      combined_gcs = grid.face_set_gcs_list[0]
      for gcs in grid.face_set_gcs_list[1:]:
         combined_gcs.append(gcs)
      combined_gcs.write_hdf5()
      combined_gcs.create_xml(title = 'faults added from lines')
      grid.clear_face_sets()
      grid.model.store_epc()

   return grid


def fault_throw_scaling(epc_file,
                        source_grid = None,
                        scaling_factor = None,
                        connection_set = None,
                        scaling_dict = None,
                        ref_k0 = 0,
                        ref_k_faces = 'top',
                        cell_range = 0,
                        offset_decay = 0.5,
                        store_displacement = False,
                        inherit_properties = False,
                        inherit_realization = None,
                        inherit_all_realizations = False,
                        inherit_gcs = True,
                        new_grid_title = None,
                        new_epc_file = None):
   """Extends epc with a new grid with fault throws multiplied by scaling factors.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      scaling_factor (float, optional): if present, the default scaling factor to apply to split pillars which do not
         appear in any of the faults in the scaling dictionary; if None, such pillars are left unchanged
      connection_set (fault.GridConnectionSet object): the connection set with associated fault feature list, used to
         identify which faces (and hence pillars) belong to which named fault
      scaling_dict (dictionary mapping string to float): the scaling factor to apply to each named fault; any faults not
         included in the dictionary will be left unadjusted (unless a default scaling factor is given as scaling_factor arg)
      ref_k0 (integer, default 0): the reference layer (zero based) to use when determining the pre-existing throws
      ref_k_faces (string, default 'top'): 'top' or 'base' identifying which bounding interface to use as the reference
      cell_range (integer, default 0): the number of cells away from faults which will have depths adjusted to spatially
         smooth the effect of the throw scaling (ie. reduce sudden changes in gradient due to the scaling)
      offset_decay (float, default 0.5): DEPRECATED; ignored
      store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
         displacement of cells' centres brought about by the fault throw scaling
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      inherit_gcs (boolean, default True): if True, any grid connection set objects related to the source grid will be
         inherited by the modified grid
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the derived grid (& crs)

   returns:
      new grid (grid.Grid object), with fault throws scaled according to values in the scaling dictionary

   notes:
      grid points are moved along pillar lines;
      stretch is towards or away from mid-point of throw;
      same shift is applied to all layers along pillar;
      pillar lines assumed to be straight;
      the offset decay argument might be changed in a future version to give improved smoothing;
      if a large fault is represented by a series of parallel minor faults 'stepping' down, each minor fault will have the
      scaling factor applied independently, leading to some unrealistic results
   """

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None

   assert source_grid.has_split_coordinate_lines, 'cannot scale fault throws in unfaulted grid'
   assert scaling_factor is not None or (connection_set is not None and scaling_dict is not None)

   if ref_k_faces == 'base':
      ref_k0 += 1
   assert ref_k0 >= 0 and ref_k0 <= source_grid.nk, 'reference layer out of range'

   # take a copy of the grid
   log.debug('copying grid')
   grid = copy_grid(source_grid, model)
   grid.cache_all_geometry_arrays()  # probably already cached anyway

   # todo: handle pillars with no geometry defined, and cells without geometry defined
   assert grid.geometry_defined_for_all_pillars(), 'not all pillars have defined geometry'
   all_good = grid.geometry_defined_for_all_cells()
   if not all_good:
      log.warning('not all cells have defined geometry')

   primaries = (grid.nj + 1) * (grid.ni + 1)
   offsets = np.zeros(grid.points_cached.shape[1:])

   if scaling_factor is not None:  # apply global scaling to throws
      # fetch unsplit equivalent of grid points for reference layer interface
      log.debug('fetching unsplit equivalent grid points')
      unsplit_points = grid.unsplit_points_ref().reshape(grid.nk + 1, -1, 3)
      # determine existing throws on split pillars
      semi_throws = np.zeros(grid.points_cached.shape[1:])  # same throw applied to all layers
      unique_spi = np.unique(grid.split_pillar_indices_cached)
      semi_throws[unique_spi, :] = (grid.points_cached[ref_k0, unique_spi, :] - unsplit_points[ref_k0, unique_spi, :])
      semi_throws[primaries:, :] = (grid.points_cached[ref_k0, primaries:, :] -
                                    unsplit_points[ref_k0, grid.split_pillar_indices_cached, :]
                                   )  # unsplit points are mid points
      # ensure no adjustment in pillar where geometry is not defined in reference layer
      if not all_good:
         semi_throws[:, :] = np.where(np.isnan(semi_throws), 0.0, semi_throws)
      # apply global scaling to throws
      offsets[:] = semi_throws * (scaling_factor - 1.0)

   if connection_set is not None and scaling_dict is not None:  # overwrite any global offsets with named fault throw adjustments
      connection_set.cache_arrays()
      for fault_index in range(len(connection_set.feature_list)):
         fault_name = connection_set.fault_name_for_feature_index(fault_index)
         if fault_name not in scaling_dict:
            continue  # no scaling for this fault
         fault_scaling = scaling_dict[fault_name]
         if fault_scaling == 1.0:
            continue
         log.info('scaling throw on fault ' + fault_name + ' by factor of: {0:.4f}'.format(fault_scaling))
         kelp_j, kelp_i = connection_set.simplified_sets_of_kelp_for_feature_index(fault_index)
         p_list = []  # list of adjusted pillars
         for kelp in kelp_j:
            for ip in [0, 1]:
               p_a = grid.pillars_for_column[kelp[0], kelp[1], 1, ip]
               p_b = grid.pillars_for_column[kelp[0] + 1, kelp[1], 0, ip]  # other side of fault
               mid_point = 0.5 * (grid.points_cached[ref_k0, p_a] + grid.points_cached[ref_k0, p_b])
               if np.any(np.isnan(mid_point)):
                  continue
               if p_a not in p_list:
                  offsets[p_a] = (grid.points_cached[ref_k0, p_a] - mid_point) * (fault_scaling - 1.0)
                  p_list.append(p_a)
               if p_b not in p_list:
                  offsets[p_b] = (grid.points_cached[ref_k0, p_b] - mid_point) * (fault_scaling - 1.0)
                  p_list.append(p_b)
         for kelp in kelp_i:
            for jp in [0, 1]:
               p_a = grid.pillars_for_column[kelp[0], kelp[1], jp, 1]
               p_b = grid.pillars_for_column[kelp[0], kelp[1] + 1, jp, 0]  # other side of fault
               mid_point = 0.5 * (grid.points_cached[ref_k0, p_a] + grid.points_cached[ref_k0, p_b])
               if np.any(np.isnan(mid_point)):
                  continue
               if p_a not in p_list:
                  offsets[p_a] = (grid.points_cached[ref_k0, p_a] - mid_point) * (fault_scaling - 1.0)
                  p_list.append(p_a)
               if p_b not in p_list:
                  offsets[p_b] = (grid.points_cached[ref_k0, p_b] - mid_point) * (fault_scaling - 1.0)
                  p_list.append(p_b)

   # initialise flag array for adjustments
   adjusted = np.zeros((primaries,), dtype = bool)

   # insert adjusted throws to all layers of split pillars
   grid.points_cached[:, grid.split_pillar_indices_cached, :] += offsets[grid.split_pillar_indices_cached, :].reshape(
      1, -1, 3)
   adjusted[grid.split_pillar_indices_cached] = True
   grid.points_cached[:, primaries:, :] += offsets[primaries:, :].reshape(1, -1, 3)

   # iteratively look for pillars neighbouring adjusted pillars, adjusting by a decayed amount
   adjusted = adjusted.reshape((grid.nj + 1, grid.ni + 1))
   while cell_range > 0:
      offset_decay = (maths.pow(2.0, cell_range) - 1.0) / (maths.pow(2.0, cell_range + 1) - 1.0)
      newly_adjusted = np.zeros((grid.nj + 1, grid.ni + 1), dtype = bool)
      for j in range(grid.nj + 1):
         for i in range(grid.ni + 1):
            if adjusted[j, i]:
               continue
            p = j * (grid.ni + 1) + i
            if p in grid.split_pillar_indices_cached:
               continue
            contributions = 0
            accum = 0.0
            if (i > 0) and adjusted[j, i - 1]:
               if j > 0:
                  accum += offsets[grid.pillars_for_column[j - 1, i - 1, 1, 0], 2]
                  contributions += 1
               if j < grid.nj:
                  accum += offsets[grid.pillars_for_column[j, i - 1, 0, 0], 2]
                  contributions += 1
            if (j > 0) and adjusted[j - 1, i]:
               if i > 0:
                  accum += offsets[grid.pillars_for_column[j - 1, i - 1, 0, 1], 2]
                  contributions += 1
               if i < grid.ni:
                  accum += offsets[grid.pillars_for_column[j - 1, i, 0, 0], 2]
                  contributions += 1
            if (i < grid.ni) and adjusted[j, i + 1]:
               if j > 0:
                  accum += offsets[grid.pillars_for_column[j - 1, i, 1, 1], 2]
                  contributions += 1
               if j < grid.nj:
                  accum += offsets[grid.pillars_for_column[j, i, 0, 1], 2]
                  contributions += 1
            if (j < grid.nj) and adjusted[j + 1, i]:
               if i > 0:
                  accum += offsets[grid.pillars_for_column[j, i - 1, 1, 1], 2]
                  contributions += 1
               if i < grid.ni:
                  accum += offsets[grid.pillars_for_column[j, i, 1, 0], 2]
                  contributions += 1
            if contributions == 0:
               continue
            dxy_dz = ((grid.points_cached[grid.nk, p, :2] - grid.points_cached[0, p, :2]) /
                      (grid.points_cached[grid.nk, p, 2] - grid.points_cached[0, p, 2]))
            offsets[p, 2] = offset_decay * accum / float(contributions)
            offsets[p, :2] = offsets[p, 2] * dxy_dz
            grid.points_cached[:, p, :] += offsets[p, :].reshape((1, 3))
            newly_adjusted[j, i] = True
      adjusted = np.logical_or(adjusted, newly_adjusted)
      cell_range -= 1

   # check cell edge relative directions (in x,y) to ensure geometry is still coherent
   log.debug('checking grid geometry coherence')
   grid.check_top_and_base_cell_edge_directions()

   # build cell displacement property array(s)
   if store_displacement:
      log.debug('generating cell displacement property arrays')
      displacement_collection = displacement_properties(grid, source_grid)
   else:
      displacement_collection = None

   collection = _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                            inherit_all_realizations)
   if collection is None:
      collection = displacement_collection
   elif displacement_collection is not None:
      collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'grid with fault throws scaled by ' + str(scaling_factor) + ' from ' +  \
                       str(rqet.citation_title_for_node(source_grid.root))

   gcs_list = []
   if inherit_gcs:
      gcs_uuids = model.uuids(obj_type = 'GridConnectionSetRepresentation', related_uuid = source_grid.uuid)
      for gcs_uuid in gcs_uuids:
         gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
         gcs.cache_arrays()
         gcs_list.append((gcs, gcs.title))
      log.debug(f'{len(gcs_list)} grid connection sets to be inherited')

   # write model
   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
      epc_file = new_epc_file
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   if len(gcs_list):
      log.debug(f'inheriting grid connection sets related to source grid: {source_grid.uuid}')
      gcs_inheritance_model = rq.Model(epc_file)
      for gcs, gcs_title in gcs_list:
         #         log.debug(f'inheriting gcs: {gcs_title}; old gcs uuid: {gcs.uuid}')
         gcs.uuid = bu.new_uuid()
         grid_list_modifications = []
         for gi, g in enumerate(gcs.grid_list):
            #            log.debug(f'gcs uses grid: {g.title}; grid uuid: {g.uuid}')
            if bu.matching_uuids(g.uuid, source_grid.uuid):
               grid_list_modifications.append(gi)
         assert len(grid_list_modifications)
         for gi in grid_list_modifications:
            gcs.grid_list[gi] = grid
         gcs.model = gcs_inheritance_model
         gcs.write_hdf5()
         gcs.create_xml(title = gcs_title)
      gcs_inheritance_model.store_epc()
      gcs_inheritance_model.h5_release()

   return grid


def global_fault_throw_scaling(epc_file,
                               source_grid = None,
                               scaling_factor = None,
                               ref_k0 = 0,
                               ref_k_faces = 'top',
                               cell_range = 0,
                               offset_decay = 0.5,
                               store_displacement = False,
                               inherit_properties = False,
                               inherit_realization = None,
                               inherit_all_realizations = False,
                               inherit_gcs = True,
                               new_grid_title = None,
                               new_epc_file = None):
   """Rewrites epc with a new grid with all the fault throws multiplied by the same scaling factor.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      scaling_factor (float): the scaling factor to apply to the throw across all split pillars
      ref_k0 (integer, default 0): the reference layer (zero based) to use when determining the pre-existing throws
      ref_k_faces (string, default 'top'): 'top' or 'base' identifying which bounding interface to use as the reference
      cell_range (integer, default 0): the number of cells away from faults which will have depths adjusted to spatially
         smooth the effect of the throw scaling (ie. reduce sudden changes in gradient due to the scaling)
      offset_decay (float, default 0.5): DEPRECATED; ignored
      store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
         displacement of cells' centres brought about by the fault throw scaling
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      inherit_gcs (boolean, default True): if True, any grid connection set objects related to the source grid will be
         inherited by the modified grid
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the derived grid (& crs)

   returns:
      new grid (grid.Grid object), with all fault throws scaled by the scaling factor

   notes:
      a scaling factor of 1 implies no change;
      calls fault_throw_scaling(), see also documentation for that function
   """

   return fault_throw_scaling(epc_file,
                              source_grid = source_grid,
                              scaling_factor = scaling_factor,
                              connection_set = None,
                              scaling_dict = None,
                              ref_k0 = ref_k0,
                              ref_k_faces = ref_k_faces,
                              cell_range = cell_range,
                              store_displacement = store_displacement,
                              inherit_properties = inherit_properties,
                              inherit_realization = inherit_realization,
                              inherit_all_realizations = inherit_all_realizations,
                              inherit_gcs = inherit_gcs,
                              new_grid_title = new_grid_title,
                              new_epc_file = new_epc_file)


def drape_to_surface(epc_file,
                     source_grid = None,
                     surface = None,
                     scaling_factor = None,
                     ref_k0 = 0,
                     ref_k_faces = 'top',
                     quad_triangles = True,
                     border = None,
                     store_displacement = False,
                     inherit_properties = False,
                     inherit_realization = None,
                     inherit_all_realizations = False,
                     new_grid_title = None,
                     new_epc_file = None):
   """Extends a resqml model with a new grid where the reference layer boundary of the source grid has been re-draped to a surface.

   arguments:
      epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
      source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
         (or one 'ROOT' grid) which is used as the source grid
      surface (surface.Surface object, optional): the surface to drape the grid to; if None, a surface is generated from
         the reference layer boundary (which can then be scaled with the scaling_factor)
      scaling_factor (float, optional): if not None, prior to draping, the surface is stretched vertically by this factor,
         away from a horizontal plane located at the surface's shallowest depth
      ref_k0 (integer, default 0): the reference layer (zero based) to drape to the surface
      ref_k_faces (string, default 'top'): 'top' or 'base' identifying which bounding interface to use as the reference
      quad_triangles (boolean, default True): if True and surface is None, each cell face in the reference boundary layer
         is represented by 4 triangles (with a common vertex at the face centre) in the generated surface; if False,
         only 2 trianges are used for each cell face (which gives a non-unique solution)
      cell_range (integer, default 0): the number of cells away from faults which will have depths adjusted to spatially
         smooth the effect of the throw scaling (ie. reduce sudden changes in gradient due to the scaling)
      offset_decay (float, default 0.5): the factor to reduce depth shifts by with each cell step away from faults (used
         in conjunction with cell_range)
      store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
         displacement of cells' centres brought about by the local depth shift
      inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
         with the source grid
      inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
         inherit_properties is False
      inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
         realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
         inherit_properties is False or inherit_realization is not None
      new_grid_title (string): used as the citation title text for the new grid object
      new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
         a new epc file (& associated h5 file) is created to contain the draped grid (& crs)

   returns:
      new grid (grid.Grid object), with geometry draped to surface

   notes:
      at least one of a surface or a scaling factor must be given;
      if no surface is given, one is created from the fault-healed grid points for the reference layer interface;
      if a scaling factor other than 1.0 is given, the surface is flexed vertically, relative to its shallowest point;
      layer thicknesses measured along pillars are maintained; cell volumes may change;
      the coordinate reference systems for the surface and the grid are assumed to be the same;
      this function currently uses an exhaustive, computationally and memory intensive algorithm;
      setting quad_triangles argument to False should give a factor of 2 speed up and reduction in memory requirement;
      the epc file and associated hdf5 file are appended to (extended) with the new grid, as a side effect of this function
   """

   log.info('draping grid to surface')

   assert epc_file or new_epc_file, 'epc file name not specified'
   if new_epc_file and epc_file and (
      (new_epc_file == epc_file) or
      (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
      new_epc_file = None
   assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
   if source_grid is None:
      model = rq.Model(epc_file)
      source_grid = model.grid()  # requires there to be exactly one grid in model
   else:
      model = source_grid.model
   assert source_grid.grid_representation == 'IjkGrid'
   assert model is not None

   assert ref_k0 >= 0 and ref_k0 < source_grid.nk
   assert ref_k_faces in ['top', 'base']
   assert surface is not None or (scaling_factor is not None and scaling_factor != 1.0)

   if surface is None:
      surface = rgs.generate_untorn_surface_for_layer_interface(source_grid,
                                                                k0 = ref_k0,
                                                                ref_k_faces = ref_k_faces,
                                                                quad_triangles = quad_triangles,
                                                                border = border)

   if scaling_factor is not None and scaling_factor != 1.0:
      scaled_surf = copy.deepcopy(surface)
      scaled_surf.vertical_rescale_points(scaling_factor = scaling_factor)
      surface = scaled_surf

   # todo: check that surface and grid use same crs; if not, convert to same

   # take a copy of the grid
   log.debug('copying grid')
   grid = copy_grid(source_grid, model)
   grid.cache_all_geometry_arrays()  # probably already cached anyway

   # todo: handle pillars with no geometry defined, and cells without geometry defined
   assert grid.geometry_defined_for_all_pillars(), 'not all pillars have defined geometry'
   assert grid.geometry_defined_for_all_cells(), 'not all cells have defined geometry'

   # fetch unsplit equivalent of grid points
   log.debug('fetching unsplit equivalent grid points')
   unsplit_points = grid.unsplit_points_ref(cache_array = True)

   # assume pillars to be straight lines based on top and base points
   log.debug('setting up pillar sample points and directional vectors')
   line_p = unsplit_points[0, :, :, :].reshape((-1, 3))
   line_v = unsplit_points[-1, :, :, :].reshape((-1, 3)) - line_p
   if ref_k_faces == 'base':
      ref_k0 += 1

   # access triangulated surface as triangle node indices into array of points
   log.debug('fetching surface points and triangle corner indices')
   t, p = surface.triangles_and_points()

   # compute intersections of all pillars with all triangles (sparse array returned with NaN for no intersection)
   log.debug('computing intersections of all pillars with all triangles')
   intersects = meet.line_set_triangles_intersects(line_p, line_v, p[t])

   # reduce to a single intersection point per pillar; todo: flag multiple intersections with a warning
   log.debug('selecting last intersection for each pillar (there should be only one intersection anyway)')
   picks = meet.last_intersects(intersects)

   # count the number of pillars with no intersection at surface (indicated by triple nan)
   log.debug('counting number of pillars which fail to intersect with surface')
   failures = np.count_nonzero(np.isnan(picks)) // 3
   log.info('number of pillars which do not intersect with surface: ' + str(failures))
   assert failures == 0, 'cannot proceed as some pillars do not intersect with surface'

   # compute a translation vector per pillar
   log.debug('computing translation vectors for pillars')
   translate = picks - unsplit_points[ref_k0, :, :, :].reshape((-1, 3))

   # shift all points by translation vectors
   log.debug('shifting entire grid along pillars')
   if grid.has_split_coordinate_lines:
      jip1 = (grid.nj + 1) * (grid.ni + 1)
      # adjust primary pillars
      grid.points_cached[:, :jip1, :] += translate.reshape((1, jip1, 3))  # will be broadcast over k axis
      # adjust split pillars
      for p in range(grid.split_pillars_count):
         primary = grid.split_pillar_indices_cached[p]
         grid.points_cached[:, jip1 + p, :] += translate.reshape(
            (1, jip1, 3))[0, primary, :]  # will be broadcast over k axis
   else:
      grid.points_cached[:, :, :, :] +=  \
         translate.reshape((1, grid.points_cached.shape[1], grid.points_cached.shape[2], 3))    # will be broadcast over k axis

   # check cell edge relative directions (in x,y) to ensure geometry is still coherent
   log.debug('checking grid geometry coherence')
   grid.check_top_and_base_cell_edge_directions()

   # build cell displacement property array(s)
   if store_displacement:
      log.debug('generating cell displacement property arrays')
      displacement_collection = displacement_properties(grid, source_grid)
   else:
      displacement_collection = None

   collection = _prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                            inherit_all_realizations)
   if collection is None:
      collection = displacement_collection
   elif displacement_collection is not None:
      collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

   if new_grid_title is None or len(new_grid_title) == 0:
      new_grid_title = 'grid flexed from ' + str(rqet.citation_title_for_node(source_grid.root))

   # write model
   model.h5_release()
   if new_epc_file:
      write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
   else:
      ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                    'Coordinates')
      write_grid(epc_file,
                 grid,
                 ext_uuid = ext_uuid,
                 property_collection = collection,
                 grid_title = new_grid_title,
                 mode = 'a')

   return grid


def add_single_cell_grid(points, new_grid_title = None, new_epc_file = None):
   """Creates a model with a single cell IJK Grid, with a cuboid cell aligned with x,y,z axes, enclosing the range of points."""

   # determine range of points
   min_xyz = np.nanmin(points.reshape((-1, 3)), axis = 0)
   max_xyz = np.nanmax(points.reshape((-1, 3)), axis = 0)
   assert not np.any(np.isnan(min_xyz)) and not np.any(np.isnan(max_xyz))

   # create corner point array in pagoda protocol
   cp = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]], [max_xyz[0], min_xyz[1], min_xyz[2]],
                  [min_xyz[0], max_xyz[1], min_xyz[2]], [max_xyz[0], max_xyz[1], min_xyz[2]],
                  [min_xyz[0], min_xyz[1], max_xyz[2]], [max_xyz[0], min_xyz[1], max_xyz[2]],
                  [min_xyz[0], max_xyz[1], max_xyz[2]], [max_xyz[0], max_xyz[1], max_xyz[2]]]).reshape(
                     (1, 1, 1, 2, 2, 2, 3))

   # switch to nexus ordering
   gf.resequence_nexus_corp(cp)

   # write cp to temp pure binary file
   temp_file = new_epc_file[:-4] + '.temp.db'
   with open(temp_file, 'wb') as fp:
      fp.write(cp.data)

   # use_rq_import to create a new model
   one_cell_model = rqi.import_nexus(new_epc_file[:-4],
                                     extent_ijk = (1, 1, 1),
                                     corp_file = temp_file,
                                     ijk_handedness = 'left',
                                     use_binary = True,
                                     split_pillars = False,
                                     grid_title = new_grid_title)
   grid = one_cell_model.grid()

   os.remove(temp_file)

   return grid


# functions below primarily for 'private' use but can be exposed


def copy_grid(source_grid, target_model = None, copy_crs = True):
   """Creates a copy of the grid object in the target model (usually prior to modifying points in situ).

   note:
      this function is not usually called directly by application code; it does not write to the hdf5
      file nor create xml for the copied grid
   """

   model = source_grid.model
   if target_model is None:
      target_model = model
   if target_model is model:
      copy_crs = False

   # create empty grid object (with new uuid)
   grid = grr.Grid(target_model)

   # inherit attributes from source grid
   grid.grid_representation = 'IjkGrid'
   grid.extent_kji = np.array(source_grid.extent_kji, dtype = 'int')
   grid.nk, grid.nj, grid.ni = source_grid.nk, source_grid.nj, source_grid.ni
   grid.nk_plus_k_gaps = source_grid.nk_plus_k_gaps
   grid.k_direction_is_down = source_grid.k_direction_is_down
   grid.grid_is_right_handed = source_grid.grid_is_right_handed
   grid.pillar_shape = source_grid.pillar_shape
   grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
   grid.k_gaps = source_grid.k_gaps
   if grid.k_gaps:
      grid.k_gap_after_array = source_grid.k_gap_after_array.copy()
      grid.k_raw_index_array = source_grid.k_raw_index_array.copy()

   # inherit a copy of the coordinate reference system used by the grid geometry
   if copy_crs:
      grid.crs_root = model.duplicate_node(source_grid.crs_root)
   else:
      grid.crs_root = source_grid.crs_root  # pointer to a source model xml tree
   grid.crs_uuid = rqet.uuid_for_part_root(grid.crs_root)

   # inherit a copy of the inactive cell mask
   if source_grid.inactive is None:
      grid.inactive = None
   else:
      grid.inactive = source_grid.inactive.copy()
   grid.active_property_uuid = source_grid.active_property_uuid

   # take a copy of the grid geometry
   source_grid.cache_all_geometry_arrays()
   grid.geometry_defined_for_all_pillars_cached = source_grid.geometry_defined_for_all_pillars_cached
   if hasattr(source_grid, 'array_pillar_geometry_is_defined'):
      grid.array_pillar_geometry_is_defined = source_grid.array_pillar_geometry_is_defined.copy()
   if hasattr(source_grid, 'array_cell_geometry_is_defined'):
      grid.array_cell_geometry_is_defined = source_grid.array_cell_geometry_is_defined.copy()
   grid.geometry_defined_for_all_cells_cached = source_grid.geometry_defined_for_all_cells_cached
   grid.points_cached = source_grid.points_cached.copy()
   if grid.has_split_coordinate_lines:
      source_grid.create_column_pillar_mapping()
      grid.split_pillar_indices_cached = source_grid.split_pillar_indices_cached.copy()
      grid.cols_for_split_pillars = source_grid.cols_for_split_pillars.copy()
      grid.cols_for_split_pillars_cl = source_grid.cols_for_split_pillars_cl.copy()
      grid.split_pillars_count = source_grid.split_pillars_count
      grid.pillars_for_column = source_grid.pillars_for_column.copy()

   return grid


def displacement_properties(new_grid, old_grid):
   """Computes cell centre differences in x, y, & z, between old & new grids, and returns a collection of 3 properties.

   note:
      this function is not usually called directly by application code
   """

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
   xy_units = rqet.find_tag(new_grid.crs_root, 'ProjectedUom').text.lower()
   z_units = rqet.find_tag(new_grid.crs_root, 'VerticalUom').text.lower()
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


def write_grid(epc_file,
               grid,
               ext_uuid = None,
               property_collection = None,
               grid_title = None,
               mode = 'a',
               geometry = True,
               time_series_uuid = None,
               string_lookup_uuid = None,
               extra_metadata = {}):
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
      crs_root = model.duplicate_node(grid.crs_root)
      grid.model = model
      grid.crs_root = crs_root
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
      grid.write_hdf5_from_caches(hdf5_file, mode = mode, geometry = geometry, imported_properties = collection)
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
      a (3D numpy array): the property array to be added to the model (expected shape nj,ni,2,2)
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
   """

   assert a.ndim == 4 and a.shape[2] == 2 and a.shape[3] == 2, 'Wrong shape! Expected shape (nj, ni, 2, 2)'
   array_rq = rqp.reformat_column_edges_to_resqml_format(a)
   property_uuid = add_one_grid_property_array(epc_file,
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


def gather_ensemble(case_epc_list,
                    new_epc_file,
                    consolidate = True,
                    shared_grids = True,
                    shared_time_series = True,
                    create_epc_lookup = True):
   """Creates a composite resqml dataset by merging all parts from all models in list, assigning realization numbers.

   arguments:
      case_epc_list (list of strings): paths of individual realization epc files
      new_epc_file (string): path of new composite epc to be created (with paired hdf5 file)
      consolidate (boolean, default True): if True, simple parts are tested for equivalence and where similar enough
         a single shared object is established in the composite dataset
      shared_grids (boolean, default True): if True and consolidate is True, then grids are also consolidated
         with equivalence based on extent of grids (and citation titles if grid extents within the first case
         are not distinct); ignored if consolidate is False
      shared_time_series (boolean, default False): if True and consolidate is True, then time series are consolidated
         with equivalence based on title, without checking that timestamp lists are the same
      create_epc_lookup (boolean, default True): if True, a StringLookupTable is created to map from realization
         number to case epc path

   notes:
      property objects will have an integer realization number assigned, which matches the corresponding index into
      the case_epc_list;
      if consolidating with shared grids, then only properties will be gathered from realisations after the first and
      an exception will be raised if the grids are not matched between realisations
   """

   if not consolidate:
      shared_grids = False

   composite_model = rq.Model(new_epc_file, new_epc = True, create_basics = True, create_hdf5_ext = True)

   epc_lookup_dict = {}

   for r, case_epc in enumerate(case_epc_list):
      t_r_start = time()  # debug
      log.info(f'gathering realszation {r}: {case_epc}')
      epc_lookup_dict[r] = case_epc
      case_model = rq.Model(case_epc)
      if r == 0:  # first case
         log.info('first case')  # debug
         composite_model.copy_all_parts_from_other_model(case_model, realization = 0, consolidate = consolidate)
         if shared_time_series:
            host_ts_uuids = case_model.uuids(obj_type = 'TimeSeries')
            host_ts_titles = []
            for ts_uuid in host_ts_uuids:
               host_ts_titles.append(case_model.title(uuid = ts_uuid))
         if shared_grids:
            host_grid_uuids = case_model.uuids(obj_type = 'IjkGridRepresentation')
            host_grid_shapes = []
            host_grid_titles = []
            title_match_required = False
            for grid_uuid in host_grid_uuids:
               grid_root = case_model.root(uuid = grid_uuid)
               host_grid_shapes.append(grr.extent_kji_from_root(grid_root))
               host_grid_titles.append(rqet.citation_title_for_node(grid_root))
            if len(set(host_grid_shapes)) < len(host_grid_shapes):
               log.warning(
                  'shapes of representative grids are not distinct, grid titles must match during ensemble gathering')
               title_match_required = True
      else:  # subsequent cases
         log.info('subsequent case')  # debug
         composite_model.consolidation = None  # discard any previous mappings to limit dictionary growth
         if shared_time_series:
            for ts_uuid in case_model.uuids(obj_type = 'TimeSeries'):
               ts_title = case_model.title(uuid = ts_uuid)
               ts_index = host_ts_titles.index(ts_title)
               host_ts_uuid = host_ts_uuids[ts_index]
               composite_model.force_consolidation_uuid_equivalence(ts_uuid, host_ts_uuid)
         if shared_grids:
            log.info('shared grids')  # debug
            for grid_uuid in case_model.uuids(obj_type = 'IjkGridRepresentation'):
               grid_root = case_model.root(uuid = grid_uuid)
               grid_extent = grr.extent_kji_from_root(grid_root)
               host_index = None
               if grid_extent in host_grid_shapes:
                  if title_match_required:
                     case_grid_title = rqet.citation_title_for_node(grid_root)
                     for host_grid_index in len(host_grid_uuids):
                        if grid_extent == host_grid_shapes[host_grid_index] and case_grid_title == host_grid_titles[
                              host_grid_index]:
                           host_index = host_grid_index
                           break
                  else:
                     host_index = host_grid_shapes.index(grid_extent)
               assert host_index is not None, 'failed to match grids when gathering ensemble'
               composite_model.force_consolidation_uuid_equivalence(grid_uuid, host_grid_uuids[host_index])
               grid_relatives = case_model.parts(related_uuid = grid_uuid)
               t_props = 0.0
               composite_h5_file_name = composite_model.h5_file_name()
               composite_h5_uuid = composite_model.h5_uuid()
               case_h5_file_name = case_model.h5_file_name()
               for part in grid_relatives:
                  if 'Property' in part:
                     t_p_start = time()
                     composite_model.copy_part_from_other_model(case_model,
                                                                part,
                                                                realization = r,
                                                                consolidate = True,
                                                                force = shared_time_series,
                                                                self_h5_file_name = composite_h5_file_name,
                                                                h5_uuid = composite_h5_uuid,
                                                                other_h5_file_name = case_h5_file_name)
                     t_props += time() - t_p_start
               log.info(f'time props: {t_props:.3f} sec')  # debug
         else:
            log.info('non shared grids')  # debug
            composite_model.copy_all_parts_from_other_model(case_model, realization = r, consolidate = consolidate)
      log.info(f'case time: {time() - t_r_start:.2f} secs')  # debug

   if create_epc_lookup and len(epc_lookup_dict):
      epc_lookup = rqp.StringLookup(composite_model, int_to_str_dict = epc_lookup_dict, title = 'ensemble epc table')
      epc_lookup.create_xml()

   composite_model.store_epc()

   log.info(f'{len(epc_lookup_dict)} realizations merged into ensemble {new_epc_file}')
