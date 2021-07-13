"""rq_print.py: simple print functionality for summarizing resqml objects."""

version = '1st July 2021'

import logging

log = logging.getLogger(__name__)
log.debug('rq_print.py version ' + version)

import numpy as np
import h5py

import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
import resqpy.olio.class_dict as rcd

# todo: redistribute functions to main object class modules and remove circular imports
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.time_series as rts
import resqpy.property as rprop
import resqpy.fault as rqf
import resqpy.well as rqw


def pl(n, use_e = False):
   if n == 1:
      return ''
   if use_e:
      return 'es'
   return 's'


def required(value):
   if value is None:
      return '(missing)'
   text = str(value)
   if len(text):
      return text
   return '(empty)'


def optional(value):
   if value is None:
      return '(not present)'
   text = str(value)
   if len(text):
      return text
   return '(empty)'


def format_xyz(xyz):
   x, y, z = xyz
   if x is None or y is None or z is None:
      return '(invalid)'
   return '(x: {0:5.3f}, y: {1:5.3f}, z: {2:5.3f})'.format(x, y, z)


def point_3d(node, flavour):
   p3d_node = rqet.find_tag(node, flavour)
   if p3d_node is None:
      return '(missing)'
   x = rqet.find_tag_float(p3d_node, 'Coordinate1')
   y = rqet.find_tag_float(p3d_node, 'Coordinate2')
   z = rqet.find_tag_float(p3d_node, 'Coordinate3')
   return (x, y, z)


def print_citation(node):
   """Prints information from a citation xml tree."""

   if node is None:
      return
   c_node = rqet.find_tag(node, 'Citation')
   if c_node is None:
      return
   print('citation:')
   print('   title:   ' + str(rqet.node_text(rqet.find_tag(c_node, 'Title'))))
   print('   user:    ' + str(rqet.node_text(rqet.find_tag(c_node, 'Originator'))))
   print('   created: ' + str(rqet.node_text(rqet.find_tag(c_node, 'Creation'))))
   print('   format:  ' + str(rqet.node_text(rqet.find_tag(c_node, 'Format'))))
   description = rqet.find_tag_text(c_node, 'Description')
   if description:
      print('   description: ' + str(description))


def print_LocalDepth3dCrs(model, node, z_is_time = False):
   """Prints information from a coordinate reference system xml tree."""

   print('units xy: ' + str(rqet.length_units_from_node(rqet.find_tag(node, 'ProjectedUom'))))
   print('units z:  ' + str(rqet.length_units_from_node(rqet.find_tag(node, 'VerticalUom'))))  # applicable to time?
   xy_axes = rqet.node_text(rqet.find_tag(node, 'ProjectedAxisOrder'))
   print('x y axes: ' + str(xy_axes))
   z_inc_down = rqet.bool_from_text(rqet.node_text(rqet.find_tag(
      node, 'ZIncreasingDownward')))  # todo: check applicability to Time3dCrs?
   if z_inc_down is None:
      z_dir = 'unknown'
   elif z_inc_down:
      z_dir = 'downwards'
   else:
      z_dir = 'upwards'
   print('z increases: ' + str(z_dir))
   print('(xyz handedness: ' + str(rqet.xyz_handedness(xy_axes, z_inc_down) + ')'))
   print('x offset: ' + str(rqet.node_text(rqet.find_tag(node, 'XOffset'))))
   print('y offset: ' + str(rqet.node_text(rqet.find_tag(node, 'YOffset'))))
   print('z offset: ' + str(rqet.node_text(rqet.find_tag(node, 'ZOffset'))))  # todo: check applicability to Time3dCrs?
   print('rotation: ' + str(rqet.node_text(rqet.find_tag(node, 'ArealRotation'))))
   parent_xy_crs_node = rqet.find_tag(node, 'ProjectedCrs')
   parent_xy_crs_node_type = rqet.node_type(parent_xy_crs_node)
   if parent_xy_crs_node_type == 'ProjectedCrsEpsgCode':
      print('parent xy crs: EPSG code ' + str(rqet.node_text(rqet.find_tag(parent_xy_crs_node, 'EpsgCode'))))
   elif parent_xy_crs_node_type == 'ProjectedUnknownCrs':
      print('parent xy crs: unknown')
   parent_z_crs_node = rqet.find_tag(node, 'VerticalCrs')  # todo: check applicability to Time3dCrs?
   parent_z_crs_node_type = rqet.node_type(parent_z_crs_node)
   if parent_z_crs_node_type == 'VerticalCrsEpsgCode':
      print('parent z crs: EPSG code ' + str(rqet.node_text(rqet.find_tag(parent_z_crs_node, 'EpsgCode'))))
   elif parent_z_crs_node_type == 'VerticalUnknownCrs':
      print('parent z crs: unknown')
   # todo: something with parent xy Crs & z Crs other than EPSG or Unknown


def print_LocalTime3dCrs(model, node):
   """Prints information from a time based (seismic) coordinate reference system xml tree."""

   # very similar to obj_LocalDepth3dCrs, with added TimeUom
   print_LocalDepth3dCrs(model, node, z_is_time = True)
   print('time units: ' + str(rqet.time_units_from_node(rqet.find_tag(node, 'TimeUom'))))


def print_IjkGridRepresentation(model, node, detail_level = 0, grid = None):
   """Prints information about an IJK Grid object, mostly from the xml tree."""

   print('ni: ' + str(rqet.node_text(rqet.find_tag(node, 'Ni'))))
   print('nj: ' + str(rqet.node_text(rqet.find_tag(node, 'Nj'))))
   print('nk: ' + str(rqet.node_text(rqet.find_tag(node, 'Nk'))))
   k_gaps_node = rqet.find_tag(node, 'KGaps')
   if k_gaps_node is None:
      print('grid does not have k gaps')
   else:
      k_gaps_count = rqet.find_tag_int(k_gaps_node, 'Count')
      print('number of k gaps: ' + str(k_gaps_count))
   pw_node = rqet.find_tag(node, 'ParentWindow')
   if pw_node is not None:
      parent_uuid = rqet.find_nested_tags_text(pw_node, ['ParentGrid', 'UUID'])
      print('parent window present (grid is a local grid); parent uuid: ' + str(parent_uuid))
   geom_node = rqet.find_tag(node, 'Geometry')
   if geom_node is None:
      if pw_node is not None:
         print('geometry is not present (to be derived from parent grid)')
      else:
         print('geometry is missing!')
      return
   print('k increases: ' + str(rqet.find_tag_text(geom_node, 'KDirection')))
   print('ijk handedness: ' + str(rqet.ijk_handedness(geom_node)))
   print('pillar shape: ' + str(rqet.find_tag_text(geom_node, 'PillarShape')))
   if not detail_level:
      return
   # create a Grid object and interrogate for extra details
   if grid is None:
      grid = grr.Grid(model, grid_root = node)
   cell_count = grid.cell_count()
   print('(number of cells: ' + str(cell_count) + ')')
   if grid.has_split_coordinate_lines:
      print('grid has split pillars (faults)')
      split_root = grid.resolve_geometry_child('SplitCoordinateLines')
      assert split_root is not None
      print('number of split pillars: ' + str(rqet.node_text(rqet.find_tag(split_root, 'Count'))))
   else:
      print('grid does not have split pillars (grid is unfaulted)')
   pillar_is_def_root = grid.resolve_geometry_child('PillarGeometryIsDefined')
   cells_all_defined = True
   if pillar_is_def_root is None:
      print('grid does not include boolean array indicating which pillar geometries are defined!')
   else:
      pillar_is_def_type = rqet.node_type(pillar_is_def_root)
      if pillar_is_def_type == 'BooleanConstantArray':
         pillars_all_defined = grid.pillar_geometry_is_defined()
         assert pillars_all_defined is not None
         if pillars_all_defined:
            print('geometry is defined for all pillars')
         else:
            print('geometry is not defined for any pillar!')
         cells_all_defined = pillars_all_defined
      else:
         assert pillar_is_def_type == 'BooleanHdf5Array'
         if detail_level > 1:
            if grid.geometry_defined_for_all_pillars():
               print('geometry is defined for all pillars with explicit boolean array')
            else:
               print('geometry is not defined for all pillars')
               cells_all_defined = False
   cell_is_def_root = grid.resolve_geometry_child('CellGeometryIsDefined')
   cells_all_defined = False
   if cell_is_def_root is None:
      print('grid does not include boolean array indicating which cell geometries are defined!')
   else:
      cell_is_def_type = rqet.node_type(cell_is_def_root)
      if cell_is_def_type == 'BooleanConstantArray':
         cells_all_defined = grid.cell_geometry_is_defined()
         assert cells_all_defined is not None
         if cells_all_defined:
            print('geometry is defined for all cells')
         else:
            print('geometry is not defined for any cells!')
      else:
         assert cell_is_def_type == 'BooleanHdf5Array'
         if detail_level > 1:
            cell_is_def_array = grid.cell_geometry_is_defined_ref()
            if grid.geometry_defined_for_all_cells_cached is None or not grid.geometry_defined_for_all_cells_cached:
               assert cell_is_def_array is not None
               cells_defined = np.count_nonzero(cell_is_def_array)
               print('geometry is defined for {0:1d} cells ({1:4.2f}%)'.format(
                  cells_defined, 100.0 * float(cells_defined) / float(cell_count)))
               cells_all_defined = (cells_defined == cell_count)
            else:
               cells_all_defined = True
            if cells_all_defined:
               print('geometry is defined for all cells with explicit boolean array')
   points_root = grid.resolve_geometry_child('Points')
   if points_root is None:
      print('grid does not include (corner) points data!')


#   elif cells_all_defined:
   else:
      laziness = (detail_level < 2)
      for local in [True, False]:
         xyz_box = grid.xyz_box(points_root = points_root, lazy = laziness, local = local)
         if local:
            local_str = 'local'
         else:
            local_str = 'global'
         if laziness:
            laziness_str = 'lazy'
         else:
            laziness_str = 'thorough'
         print(local_str + ' bounding values of geometry min, max (' + laziness_str + '):')
         for xyz in range(3):
            print(' {0}: {1:12.3f} {2:12.3f}'.format('xyz'[xyz], xyz_box[0, xyz], xyz_box[1, xyz]))
   # todo: check whether all HDF5 refs are to same file

hdf5_dataset_count = 0


def print_hdf5_line(name, group_or_dataset):
   """Prints information about one dataset (array) in an hdf5 file.

      :meta private:
   """

   global hdf5_dataset_count
   if isinstance(group_or_dataset, h5py.Group):
      return None
   print(name + '  ' + str(group_or_dataset.shape) + '  ' + str(group_or_dataset.dtype))
   hdf5_dataset_count += 1
   return None


def print_EpcExternalPartReference(model, node, detail_level = 0):
   """Prints information about an hdf5 external part."""

   global hdf5_dataset_count
   print('mime type: ' + str(rqet.node_text(rqet.find_tag(node, 'MimeType'))))
   if not detail_level or 'uuid' not in node.attrib.keys():
      return
   uuid = bu.uuid_from_string(node.attrib['uuid'])
   file_name = model.h5_file_name(uuid)
   print('hdf5 file: ' + str(file_name))
   hdf5 = model.h5_access(uuid)
   hdf5_dataset_count = 0
   if hdf5 is not None:
      for (key, group) in hdf5.items():
         print('*** ' + str(key) + ': ' + str(group))
         group.visititems(print_hdf5_line)
   print('number of data sets (arrays): ' + str(hdf5_dataset_count))


def print_TimeSeries(model, node, detail_level = 0):
   """Prints information form a time series xml tree."""

   if node is None:
      return
   print('number of timestamps: ' + str(rqet.count_tag(node, 'Time')))
   if not detail_level:
      return
   time_series = rts.TimeSeries(model, uuid = node.attrib['uuid'])
   for index in range(time_series.number_of_timestamps()):
      if index:
         print('{0:>5d}  {1} {2:>5d}'.format(index, rts.simplified_timestamp(time_series.timestamp(index)),
                                             time_series.step_days(index)))
      else:
         print('{0:>5d}  {1}  step (days)'.format(0, rts.simplified_timestamp(time_series.timestamp(0))))


def print_StringTableLookup(model, node, detail_level = 0):
   """Prints information from a string lookup table xml tree."""

   entries_node_list = rqet.list_of_tag(node, 'Value')
   if entries_node_list is None or len(entries_node_list) == 0:
      print('no entries in lookup table')
      return
   print(str(len(entries_node_list)) + ' entries in lookup table:')
   if not detail_level:
      return
   entries_pair_list = []
   for entry_node in entries_node_list:
      key_node = rqet.find_tag(entry_node, 'Key')
      if key_node is None:
         continue
      key_type = rqet.node_type(key_node)
      if key_type == 'integer':
         key = int(key_node.text)
      else:
         key = key_node.text
      value = rqet.node_text(rqet.find_tag(entry_node, 'Value'))
      entries_pair_list.append((key, value))
   entries_pair_list.sort()
   for (key, value) in entries_pair_list:
      if isinstance(key, int):
         print('{0:>4d}: {1}'.format(key, value))
      else:
         print(str(key) + ': ' + str(value))


def print_reference_node_and_return_referenced_part(node, heading):
   """Prints some information about a part being referenced in an xml node."""

   if node is None:
      print(str(heading) + ': (missing)')
      return None
   print(str(heading) + ':')
   print('   title: ' + str(rqet.node_text(rqet.find_tag(node, 'Title'))))
   content_type = rqet.content_type(rqet.node_text(rqet.find_tag(node, 'ContentType')))
   if content_type is not None:
      print('   type:  ' + str(rcd.readable_class(content_type)))
   uuid_str = rqet.node_text(rqet.find_tag(node, 'UUID'))
   if uuid_str is not None:
      print('   uuid:  ' + str(uuid_str))
   if content_type is None or content_type[:4] != 'obj_' or uuid_str is None:
      return None
   if uuid_str[0] == '_':
      uuid_str = uuid_str[1:]  # tolerate fesapi quirk
   return content_type + '_' + uuid_str + '.xml'


def print_Property(model, node, detail_level = 0):
   """Prints information about a grid property, mostly from the xml tree."""

   property_kind = None
   property_kind_node = rqet.find_tag(node, 'PropertyKind')
   if property_kind_node is None:
      print('property kind is missing!')
   else:
      kind_node = rqet.find_tag(property_kind_node, 'Kind')
      if kind_node is not None:
         property_kind = rqet.node_text(kind_node)
         print('kind: ' + str(property_kind))
      else:
         local_kind_node = rqet.find_tag(property_kind_node, 'LocalPropertyKind')
         if local_kind_node is not None:
            print('kind: ' + str(rqet.find_tag_text(local_kind_node, 'Title')))
         else:
            print('property kind xml not recognised!')
   facet_type = None
   facet = None
   facet_node_list = rqet.list_of_tag(node, 'Facet')
   if facet_node_list is not None:
      for facet_node in facet_node_list:
         print('facet: ' + str(rqet.node_text(rqet.find_tag(facet_node, 'Facet'))) + ': ' +
               str(rqet.find_tag_text(facet_node, 'Value')))
         if not facet_type:  # arbitrarily pick up first facet in list to use in helping to guess uom, if needed
            facet_type = rqet.node_text(rqet.find_tag(facet_node, 'Facet'))
            facet = rqet.node_text(rqet.find_tag(facet_node, 'Value'))
   citation_title = rqet.node_text(rqet.find_tag(rqet.find_tag(node, 'Citation'), 'Title'))
   if citation_title is not None:
      (derived_property_kind, derived_facet_type,
       derived_facet) = rprop.property_kind_and_facet_from_keyword(citation_title)
      if derived_property_kind is not None:
         if property_kind is None:
            property_kind = derived_property_kind
         if derived_facet_type is None:
            print('(derived kind: ' + derived_property_kind + ')')
         else:
            print('(derived kind: ' + str(derived_property_kind) + '; facet: ' + str(derived_facet_type) + ': ' +
                  str(derived_facet) + ')')
   print('realization: ' + optional(rqet.find_tag_int(node, 'RealizationIndex')))
   grid = None
   time_series = time_series_uuid = time_series_part = None
   time_index_node = rqet.find_tag(node, 'TimeIndex')
   if time_index_node is None:
      print('static property (no time index)')
   else:
      index_node = rqet.find_tag(time_index_node, 'Index')
      print('time index: ' + str(rqet.node_text(index_node)))
      if index_node is None:
         time_index = None
      else:
         time_index = int(index_node.text)
      time_series_ref_node = rqet.find_tag(time_index_node, 'TimeSeries')
      time_series_part = print_reference_node_and_return_referenced_part(time_series_ref_node, 'time series')
      if detail_level > 0:
         time_series_uuid = model.uuid_for_part(time_series_part)
         if time_series_uuid is None:
            print('   time series part not found!')
            log.warning('missing time series part: ' + str(time_series_part))
         else:
            time_series = rts.TimeSeries(model, uuid = time_series_uuid)
            print('   number of timestamps in series: ' + str(time_series.number_of_timestamps()))
         if time_index is not None and time_series is not None:
            print('timestamp for this part: ' + str(rts.simplified_timestamp(time_series.timestamp(time_index))))
   minimum = rqet.node_text(rqet.find_tag(node, 'MinimumValue'))
   print('minimum (xml): ' + str(minimum))
   maximum = rqet.node_text(rqet.find_tag(node, 'MaximumValue'))
   print('maximum (xml): ' + str(maximum))
   units = rqet.find_tag_text(node, 'UOM')
   if units is not None:
      print('units: ' + units)
      if units in ['unknown', 'Euc'] and property_kind is not None and detail_level > 0:
         guessed_uom = rprop.guess_uom(property_kind, minimum, maximum, grid, facet_type = facet_type, facet = facet)
         if guessed_uom:
            print('(guessed units: ' + str(guessed_uom) + ')')
   print('count: ' + str(rqet.find_tag_text(node, 'Count')))
   print('indexable element: ' + str(rqet.find_tag_text(node, 'IndexableElement')))
   values_node = rqet.find_tag(node, 'PatchOfValues')
   if values_node is None:
      print('patch of values is missing!')
      return
   print('patch of values is present in xml')
   if detail_level < 2 or grid is None:
      return
   single_prop_collection = rprop.GridPropertyCollection()
   single_prop_collection.set_grid(grid)
   part_name = rqet.part_name_for_part_root(node)
   single_prop_collection.add_part_to_dict(part_name,
                                           trust_uom = units is not None and units not in ['', 'unknown', 'Euc'])
   cached_array = single_prop_collection.cached_part_array_ref(part_name)
   if cached_array is None:
      print('failed to cache array for part ' + str(part_name))
      return None
   print('type of array elements: ' + str(cached_array.dtype))
   print('shape of array (nk, nj, ni): ' + str(cached_array.shape))
   print('number of elements in array: ' + str(cached_array.size))
   non_zero_count = np.count_nonzero(cached_array)
   print('number of non-zero elements: ' + str(non_zero_count))
   print('minimum (data, all cells): ' + str(np.nanmin(cached_array)))
   print('maximum (data, all cells): ' + str(np.nanmax(cached_array)))
   if cached_array.dtype == 'float':
      average = np.mean(cached_array)  # todo: handle NaN and Inf in a better way
      print('mean value (all cells): ' + str(average))
      if property_kind in ['rock volume', 'pore volume']:
         print('sum of values (all cells): ' + str(np.sum(cached_array)))
   if grid and grid.inactive is not None:
      masked_array = single_prop_collection.cached_part_array_ref(part_name, masked = True)
      inactive_count = np.count_nonzero(masked_array.mask)
      print('active cell count: ' + str(cached_array.size - inactive_count))
      print('inactive cell count: ' + str(inactive_count))
      print('minimum (data, active cells): ' + str(np.nanmin(masked_array)))
      print('maximum (data, active cells): ' + str(np.nanmax(masked_array)))
      if cached_array.dtype == 'float':
         average = np.mean(masked_array)  # todo: handle NaN and Inf in a better way
         print('mean value (active cells): ' + str(average))
         if property_kind in ['rock volume', 'pore volume']:
            print('sum of values (active cells): ' + str(np.nansum(masked_array)))
   single_prop_collection.uncache_part_array(part_name)


def print_CategoricalProperty(model, node, detail_level = 0):
   """Prints information about a categorical grid property, mostly from the xml tree."""

   print_Property(model, node, detail_level = detail_level)
   lookup_node = rqet.find_tag(node, 'Lookup')
   if lookup_node is None:
      print('no lookup table referenced')
   else:
      print_reference_node_and_return_referenced_part(lookup_node, 'lookup table reference')


def print_ContinuousProperty(model, node, detail_level = 0):
   """Prints information about a continuous (float) grid property, mostly from the xml tree."""

   print_Property(model, node, detail_level = detail_level)


def print_DiscreteProperty(model, node, detail_level = 0):
   """Prints information about a discrete (integer) grid property, mostly from the xml tree."""

   print_Property(model, node, detail_level = detail_level)


def print_TriangulatedSetRepresentation(model, node, detail_level = 0):
   """Prints information about a triangulated set representation (surface) from the xml tree."""

   print('surface role: ' + required(rqet.find_tag_text(node, 'SurfaceRole')))
   patch_count = rqet.count_tag(node, 'TrianglePatch')
   print('patch count:  ' + str(patch_count))
   if detail_level > 0:
      for patch in rqet.list_of_tag(node, 'TrianglePatch'):
         print('   patch index:     ' + required(rqet.find_tag_int(patch, 'PatchIndex')))
         print('   triangle count:  ' + required(rqet.find_tag_int(patch, 'Count')))
         print('   node count:      ' + required(rqet.find_tag_int(patch, 'NodeCount')))
         if detail_level > 1:
            print('   node null value: ' + optional(rqet.find_tag_int(patch, 'NullValue')))


def print_PointSetRepresentation(model, node, detail_level = 0):
   """Prints information about a point set representation from the xml tree."""

   patch_count = rqet.count_tag(node, 'NodePatch')
   print('patch count:  ' + str(patch_count))
   if detail_level > 0:
      total_count = 0
      for patch in rqet.list_of_tag(node, 'NodePatch'):
         print('   patch index: ' + required(rqet.find_tag_int(patch, 'PatchIndex')))
         point_count = rqet.find_tag_int(patch, 'Count')
         print('   point count: ' + required(point_count))
         if point_count:
            total_count += point_count
      print('(total point count: {})'.format(total_count))


def print_GridConnectionSetRepresentation(model, node, detail_level = 0):
   """Prints information about a grid connection set representation (fault cell face set), mostly from the xml tree."""

   print('face count: ' + required(rqet.find_tag_int(node, 'Count')))
   ci_node = rqet.find_tag(node, 'ConnectionInterpretations')
   if ci_node is None:
      print('(connection interpretations missing)')
      return
   feature_count = rqet.count_tag(ci_node, 'FeatureInterpretation')
   print('feature (fault or fracture) count: ' + str(feature_count))
   if detail_level > 0:
      if detail_level > 1:
         gcs = rqf.GridConnectionSet(model, connection_set_root = node)
      feature_index = 0
      for feature in rqet.list_of_tag(ci_node, 'FeatureInterpretation'):
         print_reference_node_and_return_referenced_part(feature, 'feature interpretation reference')
         if detail_level > 1:
            face_count_per_feature = len(gcs.raw_list_of_cell_face_pairs_for_feature_index(feature_index)[0])
            print('   face pair count: ' + str(face_count_per_feature))
         feature_index += 1


def print_MdDatum(model, node, detail_level = 0):
   """Prints information about a measured depth datum, from the xml tree."""

   print('reference: ' + required(rqet.find_tag_text(node, 'MdReference')))
   location_xyz = point_3d(node, 'Location')
   print('location:  ' + format_xyz(location_xyz))
   if detail_level > 0:
      crs_ref = rqet.find_tag(node, 'LocalCrs')
      crs_part = print_reference_node_and_return_referenced_part(crs_ref, 'local coordinate reference system reference')
      if detail_level > 1:
         crs = rqc.Crs(model, uuid = rqet.uuid_in_part_name(crs_part))
         global_xyz = crs.local_to_global(location_xyz)
         print('global location: ' + format_xyz(global_xyz))


def print_WellboreFrameRepresentation(model, node, detail_level = 0):
   """Prints information about a wellbore frame representation, from the xml tree."""

   print('md mode count: ' + required(rqet.find_tag_int(node, 'NodeCount')))
   trajectory_ref = rqet.find_tag(node, 'Trajectory')
   print_reference_node_and_return_referenced_part(trajectory_ref, 'wellbore trajectory reference')


def print_WellboreMarkerFrameRepresentation(model, node, detail_level = 0):
   """Prints information about a wellbore marker frame representation, from the xml tree."""

   print_WellboreFrameRepresentation(model, node, detail_level = detail_level)
   # todo: add details
   marker_count = len(rqet.list_of_tag(node, 'WellboreMarker'))
   print('marker count: ' + str(marker_count))
   print('(other details not yet available for this type of object)')


def print_BlockedWellboreRepresentation(model, node, detail_level = 0):
   """Prints information about a blocked wellbore representation, from the xml tree."""

   print_WellboreFrameRepresentation(model, node, detail_level = detail_level)
   print('cell count: ' + required(rqet.find_tag_int(node, 'CellCount')))
   print('grid count: ' + required(rqet.count_tag(node, 'Grid')))
   if detail_level > 0:
      for grid_ref_node in rqet.list_of_tag(node, 'Grid'):
         print_reference_node_and_return_referenced_part(grid_ref_node, 'grid reference')


def print_WellboreTrajectoryRepresentation(model, node, detail_level = 0):
   """Prints information about a wellbore trajectory representation, from the xml tree."""

   lk_dict = {
      -1: 'null - no line',
      0: 'vertical)',
      1: 'linear spline',
      2: 'natural cubic spline',
      3: 'cubic spline',
      4: 'z linear cubic spline',
      5: 'minimum-curvature spline'
   }
   unit_str = rqet.find_tag_text(node, 'MdUom')
   if unit_str is None:
      unit_str = '(missing units)'
   print('start md:  ' + required(rqet.find_tag_float(node, 'StartMd')) + ' ' + unit_str)
   print('finish md: ' + required(rqet.find_tag_float(node, 'FinishMd')) + ' ' + unit_str)
   print('md units:  ' + unit_str)
   md_ref_node = rqet.find_tag(node, 'MdDatum')
   if md_ref_node is None:
      print('(missing measured depth datum reference)')
   else:
      print_reference_node_and_return_referenced_part(md_ref_node, 'measured depth datum reference')
   geom_node = rqet.find_tag(node, 'Geometry')
   if geom_node is None:
      print('(presumed vertical - no geometry)')
      return
   line_kind = rqet.find_tag_int(geom_node, 'LineKindIndex')
   if line_kind is None:
      lk_str = '(missing)'
   elif line_kind >= -1 and line_kind <= 5:
      lk_str = str(line_kind) + ' (' + lk_dict[line_kind] + ')'
   else:
      lk_str = str(line_kind) + ' (invalid value)'
   print('line kind:  ' + lk_str)
   print('knot count: ' + required(rqet.find_tag_int(geom_node, 'KnotCount')))
   tv_node = rqet.find_tag(geom_node, 'TangentVectors')
   if tv_node:
      print('(tangent vectors are present)')
   else:
      print('(optional tangent vectors are not present)')
   print('domain:     ' + optional(rqet.find_tag_text(node, 'MdDomain')))
   ds_ref_node = rqet.find_tag(node, 'DeviationSurvey')
   if detail_level > 0:
      crs_ref = rqet.find_tag(geom_node, 'LocalCrs')
      print_reference_node_and_return_referenced_part(crs_ref, 'local coordinate reference system reference')
   if ds_ref_node is not None:
      print_reference_node_and_return_referenced_part(ds_ref_node, 'deviation survey reference')
   if detail_level > 1:
      try:
         trajectory = rqw.Trajectory(model, trajectory_root = node)
         print('min xyz: ' + str(np.amin(trajectory.control_points, axis = 0)))
         print('max xyz: ' + str(np.amax(trajectory.control_points, axis = 0)))
         print('    x        y       z')
         for (x, y, z) in trajectory.control_points:
            print(f' {x:8.1f} {y:8.1f} {z:7.1f}')
      except Exception:
         print('(problem displaying control points)')
         log.exception('trajectory issue')


def print_TectonicBoundaryFeature(model, node, detail_level = 0):
   """Prints information about a tectonic boundary feature (fault or fracture), from the xml tree."""

   print('tectonic boundary kind: ' + required(rqet.find_tag_text(node, 'TectonicBoundaryKind')))


def print_GeneticBoundaryFeature(model, node, detail_level = 0):
   """Prints information about a genetic boundary feature (horizon or geobody boundary), from the xml tree."""

   print('genetic boundary kind: ' + required(rqet.find_tag_text(node, 'GeneticBoundaryKind')))
   # todo: optional absolute age


def print_WellboreFeature(model, node, detail_level = 0):
   """Prints information about a wellbore feature, from the xml tree."""

   pass  # nothing but citation block in this feature class


def print_OrganizationFeature(model, node, detail_level = 0):
   """Prints information about an organization feature, from the xml tree."""

   pass  # nothing but citation block in this feature class


def print_FaultInterpretation(model, node, detail_level = 0):
   """Prints information about a fault interpretation, from the xml tree."""

   print('domain:        ' + required(rqet.find_tag_text(node, 'Domain')))
   print('is listric:    ' + optional(rqet.find_tag_text(node, 'IsListric')))
   print('maximum throw: ' + optional(rqet.find_tag_float(node, 'MaximumThrow')))
   print('mean azimuth:  ' + optional(rqet.find_tag_float(node, 'MeanAzimuth')))
   print('mean dip:      ' + optional(rqet.find_tag_float(node, 'MeanDip')))
   throw_understanding = rqet.find_tag_text(node, 'ThrowInterpretation')
   if not throw_understanding:
      throw_understanding = '(presumed normal)'
   print('throw interpretation: ' + throw_understanding)


def print_HorizonInterpretation(model, node, detail_level = 0):
   """Prints information about a horizon interpretation, from the xml tree."""

   print('domain: ' + required(rqet.find_tag_text(node, 'Domain')))
   br_node_list = rqet.list_of_tag(node, 'BoundaryRelation')
   if len(br_node_list):
      for br_node in br_node_list:
         print('boundary relation: ' + required(br_node.text))
   else:
      print('(no boundary relation(s) specified)')
   sss_node_list = rqet.list_of_tag(node, 'SequenceStratigraphySurface')
   if len(sss_node_list):
      for sss_node in sss_node_list:
         print('sequence stratigraphy surface: ' + required(sss_node.text))
   else:
      print('(no sequence stratigraphy surface(s) specified)')


def print_WellboreInterpretation(model, node, detail_level = 0):
   """Prints information about a wellbore interpretation, from the xml tree."""

   print('is drilled: ' + required(rqet.find_tag_bool(node, 'IsDrilled')))


def print_EarthModelInterpretation(model, node, detail_level = 0):
   """Prints information about an earth model interpretation, from the xml tree."""

   print('domain: ' + required(rqet.find_tag_text(node, 'Domain')))


def print_node(model, node, detail_level = 0):
   """Prints information about an object from an xml node."""

   if node is None:
      return
   title = rqet.citation_title_for_node(node)
   if title:
      print('title: ' + str(title))
   obj_type = rqet.node_type(node)
   if obj_type is None:
      return
   print('object type: ' + str(rcd.readable_class(obj_type)))
   if 'uuid' in node.attrib.keys():
      print('uuid: ' + str(node.attrib['uuid']))
   if obj_type.endswith('Property'):
      support_node = rqet.find_tag(node, 'SupportingRepresentation')
      if support_node is None:
         print('no supporting representation referenced')
      else:
         print_reference_node_and_return_referenced_part(support_node, 'supporting representation reference')
   elif obj_type.endswith('Representation'):
      interp_node = rqet.find_tag(node, 'RepresentedInterpretation')
      if interp_node is None:
         print('no represented interpretation referenced')
      else:
         print_reference_node_and_return_referenced_part(interp_node, 'represented interpretation reference')
   elif obj_type.endswith('Interpretation'):
      feature_node = rqet.find_tag(node, 'InterpretedFeature')
      if feature_node is None:
         print('no interpreted feature referenced')
      else:
         print_reference_node_and_return_referenced_part(feature_node, 'interpreted feature reference')
   if obj_type == 'obj_LocalDepth3dCrs':
      print_LocalDepth3dCrs(model, node)
   elif obj_type == 'obj_LocalTime3dCrs':
      print_LocalTime3dCrs(model, node)
   elif obj_type == 'obj_IjkGridRepresentation':
      print_IjkGridRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_EpcExternalPartReference':
      print_EpcExternalPartReference(model, node, detail_level = detail_level)
   elif obj_type == 'obj_TimeSeries':
      print_TimeSeries(model, node, detail_level = detail_level)
   elif obj_type == 'obj_StringTableLookup':
      print_StringTableLookup(model, node, detail_level = detail_level)
   elif obj_type == 'obj_ContinuousProperty':
      print_ContinuousProperty(model, node, detail_level = detail_level)
   elif obj_type == 'obj_DiscreteProperty':
      print_DiscreteProperty(model, node, detail_level = detail_level)
   elif obj_type == 'obj_CategoricalProperty':
      print_CategoricalProperty(model, node, detail_level = detail_level)
   elif obj_type == 'obj_TriangulatedSetRepresentation':
      print_TriangulatedSetRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_PointSetRepresentation':
      print_PointSetRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_GridConnectionSetRepresentation':
      print_GridConnectionSetRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_MdDatum':
      print_MdDatum(model, node, detail_level = detail_level)
   elif obj_type == 'obj_WellboreFrameRepresentation':
      print_WellboreFrameRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_WellboreMarkerFrameRepresentation':
      print_WellboreMarkerFrameRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_BlockedWellboreRepresentation':
      print_BlockedWellboreRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_WellboreTrajectoryRepresentation':
      print_WellboreTrajectoryRepresentation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_TectonicBoundaryFeature':
      print_TectonicBoundaryFeature(model, node, detail_level = detail_level)
   elif obj_type == 'obj_GeneticBoundaryFeature':
      print_GeneticBoundaryFeature(model, node, detail_level = detail_level)
   elif obj_type == 'obj_WellboreFeature':
      print_WellboreFeature(model, node, detail_level = detail_level)
   elif obj_type == 'obj_FaultInterpretation':
      print_FaultInterpretation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_HorizonInterpretation':
      print_HorizonInterpretation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_WellboreInterpretation':
      print_WellboreInterpretation(model, node, detail_level = detail_level)
   elif obj_type == 'obj_OrganizationFeature':
      print_OrganizationFeature(model, node, detail_level = detail_level)
   elif obj_type == 'obj_EarthModelInterpretation':
      print_EarthModelInterpretation(model, node, detail_level = detail_level)
      # todo: other object types
   else:
      print('(details unavailable for this type of object)')
   print_citation(node)
