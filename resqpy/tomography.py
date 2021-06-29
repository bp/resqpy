"""tomography.py: high level functions to support 2D mapping and generation of cross sectional images."""

# todo: cross section functions

version = '29th April 2021'

import logging
log = logging.getLogger(__name__)
log.debug('tomography.py version ' + version)

import os
import numpy as np
from PIL import Image, ImageDraw

import resqpy.surface as rqs
import resqpy.property as rqp
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu


def make_pixel_map(grid, width, height, origin = None, dx = None, dy = None, border = 0.0,
                   k0 = 0, vertical_ref = 'top', trim = True):
   """Generate a regular mesh and related pixel map property, writing to hdf5 and adding as parts.

   arguments:
      grid (resqpy.grid.Grid object): the (usually irregular) source grid which is to be mapped
      width (int): the width of the pixel rectangle (number of pixels)
      height (int): the height of the pixel rectangle (number of pixels)
      origin (float pair, optional): x, y of south west corner of area covered by pixel rectangle,
         in local crs of grid; if None then the minimum point of the grid's box is used and border
         applied
      dx (float, optional): the size (west to east) of a pixel, in locel crs of grid; if dx and dy
         are both None, they are computed from the grid's box, border, width and height
      dy (float, optional): the size (south to north) of a pixel, in locel crs; if only one of dx
         or dy is given, the other is set the same; if both None they are computed (see dx)
      border (float, default 0.0): a border width, in local crs of grid, applied around bounding box
         of grid when computing origin and/or dx & dy; ignored if those values are all supplied
      k0 (int, default 0): the layer to create a 2D pixel map for
      vertical_ref (string, default 'top'): 'top' or 'base'
      trim (boolean, default True): if True, the width or height may be reduced so that the resulting
         pixel map has an equal border on all 4 sides

   returns:
      resqpy.surface.Mesh, resqpy.property.GridPropertyCollection

   note:
      this function does not re-write the epc but it does write new pixel map data to hdf5 and create xml
   """

   log.debug(f'making pixel map with width, height: {width}, {height}')

   assert grid is not None
   assert width > 0 and height > 0
   assert border >= 0.0

   model = grid.model
   assert model is not None

   if origin is None or (dx is None and dy is None):
      grid_xyz_box = grid.xyz_box(lazy = False, local = True)
      if origin is None: origin = tuple(grid_xyz_box[0, :2] - border)
      if dx is None and dy is None:
         max_x, max_y = grid_xyz_box[1, :2] + border
         dx = (max_x - origin[0]) / float(width)
         dy = (max_y - origin[1]) / float(height)
         if dx > dy:
            dy = dx
            if trim: height = int((max_y - origin[1]) / dy)
         else:
            dx = dy
            if trim: width = int((max_x - origin[0]) / dx)
      elif dx is None:
         dx = dy
      elif dy is None:
         dy = dx
   assert len(origin) == 2
   origin_xyz = tuple(list(origin) + [0.0])
   assert dx > 0.0 and dy > 0.0

   p_map = grid.pixel_maps(origin, width, height, dx, dy = dy, k0 = k0, vertical_ref = vertical_ref)

   source_str = 'pixel map for grid ' + rqet.citation_title_for_node(grid.root)

   p_mesh = rqs.Mesh(parent_model = model, nj = height + 1, ni = width + 1,
                     origin = origin_xyz, dxyz_dij = np.array([[dx, 0.0, 0.0], [0.0, dy, 0.0]]),
                     crs_uuid = grid.crs_uuid)

   p_mesh.create_xml(title = 'pixel map frame')

   property_collection = rqp.PropertyCollection()
   property_collection.set_support(support = p_mesh)

   p_uuid = property_collection.add_cached_array_to_imported_list(
      p_map,
      source_str,
      keyword = 'pixel map',
      discrete = True, null_value = -1,
      property_kind = 'index', facet_type = 'what', facet = 'pixel map',
      indexable_element = 'cells', count = 2)

   property_collection.write_hdf5_for_imported_list()
   property_collection.create_xml_for_imported_list_and_add_parts_to_model()

   # todo: add extra metadata to p_map property holding uuid of grid

   p_node = model.root_for_uuid(p_uuid)

   model.create_reciprocal_relationship(p_node, 'destinationObject', grid.root, 'sourceObject')

   log.debug('pixel map made')

   return p_mesh, property_collection


def find_pixel_maps(grid, min_width = None, max_width = None, min_height = None, max_height = None):
   """Returns a property collection of existing pixel maps in model for mapping grid, optionally filtered by resolution.

   arguments:
      grid (resqpy.grid.Grid object): the (usually irregular) source grid which is to be mapped
      min_width (int, optional): if present, only pixel maps of at least this width are included
      max_width (int, optional): if present, only pixel maps of at most this width are included
      min_height (int, optional): if present, only pixel maps of at least this height are included
      max_height (int, optional): if present, only pixel maps of at most this height are included

   returns:
      resqpy.property.PropertyCollection containing the existing pixel maps for mapping the grid object
   """

   log.debug('finding pixel maps')

   model = grid.model

   filtering = (min_width or max_width or min_height or max_height)

   # pixel maps have a secondary relationship with the grid for mapping
   property_parts_list = model.parts_list_of_type(type_of_interest = 'obj_DiscreteProperty')
   log.debug('number of discrete properties: ' + str(len(property_parts_list)))
   property_parts_list = model.parts_list_filtered_by_related_uuid(property_parts_list, grid.uuid)
   log.debug('number of discrete properties related to grid: ' + str(len(property_parts_list)))
   # property parts list will contain pixel maps and native discrete properties for grid

   # following property collections may contain parts with various supporting representations
   pm_collection = rqp.PropertyCollection()
   pm_collection.model = model
   pm_collection.add_parts_list_to_dict(property_parts_list)
   log.debug('initial property collection size: ' + str(pm_collection.number_of_parts()))

   pm_collection = rqp.selective_version_of_collection(pm_collection,
                                                       count = 2,
                                                       property_kind = 'index',
                                                       facet_type = 'what',
                                                       facet = 'pixel map',
                                                       indexable = 'cells')
   log.debug('selective property collection size: ' + str(pm_collection.number_of_parts()))

   if filtering:
      log.debug('filtering')
      pixel_map_mesh_uuid = None
      filtered_parts_list = []
      for part in pm_collection.parts():
         support_uuid = model.supporting_representation_for_part(part)
         if support_uuid is None: continue
         if pixel_map_mesh_uuid is None or not bu.matching_uuids(support_uuid, pixel_map_mesh_uuid):
            pixel_map_mesh_root = model.root_for_uuid(support_uuid)
            pixel_map_extent_ji = np.ones(2, dtype = 'int')
            pixel_map_extent_ji[0] = rqet.find_nested_tags_int(pixel_map_mesh_root, ['Grid2dPatch', 'SlowestAxisCount']) - 1
            pixel_map_extent_ji[1] = rqet.find_nested_tags_int(pixel_map_mesh_root, ['Grid2dPatch', 'FastestAxisCount']) - 1
            pixel_map_mesh_uuid = support_uuid
         if min_width is not None and (pixel_map_extent_ji[1] < min_width): continue
         if max_width is not None and (pixel_map_extent_ji[1] > max_width): continue
         if min_height is not None and (pixel_map_extent_ji[0] < min_height): continue
         if max_height is not None and (pixel_map_extent_ji[0] > max_height): continue
         filtered_parts_list.append(part)
      pm_collection = rqp.PropertyCollection()
      pm_collection.model = model
      pm_collection.add_parts_list_to_dict(filtered_parts_list)

   if pm_collection.number_of_parts() == 0: pm_collection = None

   if pm_collection:
      log.debug('found ' + str(pm_collection.number_of_parts()) + ' pixel maps')
   else:
      log.debug('no pixel maps found')

   return pm_collection


def find_or_make_pixel_map(grid, width, height, exact_resolution = False,
                           min_width = None, max_width = None, min_height = None, max_height = None,
                           origin = None, dx = None, dy = None, border = 0.0,
                           k0 = 0, vertical_ref = 'top', trim = True):
   """Looks for pixel map(s) for grid and returns if found; otherwise makes one.

   arguments:
      width (int): the width of the pixel rectangle (number of pixels)
      height (int): the height of the pixel rectangle (number of pixels)
      exact_resolution (bool, default False): if True, equivalent to setting the following min & max
         arguments equal to width and to height, forcing an exact match to the given resolution
      min_width (int, optional): if present, only pixel maps of at least this width are eligible
      max_width (int, optional): if present, only pixel maps of at most this width are eligible
      min_height (int, optional): if present, only pixel maps of at least this height are eligible
      max_height (int, optional): if present, only pixel maps of at most this height are eligible
      origin (float pair, optional): x, y of south west corner of area covered by pixel rectangle,
         in local crs of grid; if None then the minimum point of the grid's box is used and border
         applied
      dx (float, optional): the size (west to east) of a pixel, in locel crs of grid; if dx and dy
         are both None, they are computed from the grid's box, border, width and height
      dy (float, optional): the size (south to north) of a pixel, in locel crs; if only one of dx
         or dy is given, the other is set the same; if both None they are computed (see dx)
      border (float, default 0.0): a border width, in local crs of grid, applied around bounding box
         of grid when computing origin and/or dx & dy; ignored if those values are all supplied
      k0 (int, default 0): the layer to create a 2D pixel map for
      vertical_ref (string, default 'top'): 'top' or 'base'

   returns:
      resqpy.property.PropertyCollection holding at least one pixel map property

   notes:
      if exact_resolution is True, the individual min_ and max_ arguments are ignored;
      the origin, dx, dy, border, k0 & vertical_ref arguments are only used if a new pixel map is created;
      the epc is not rewritten by this function, even if new parts have been introduced and the hdf5 written to;
      calling code should use model.store_epc(only_if_modifed = True)
   """

   log.debug(f'finding or making pixel map with width, height: {width}, {height}')
   log.debug(f'width range: {min_width} to {max_width}; height range: {min_height} to {max_height}')

   if exact_resolution:
      min_width = max_width = width
      min_height = max_height = height

   pm_collection = find_pixel_maps(grid, min_width = min_width, max_width = max_width,
                                   min_height = min_height, max_height = max_height)

   if pm_collection is not None and pm_collection.number_of_parts() > 0: return pm_collection

   _, pm_collection = make_pixel_map(grid, width, height, origin = origin, dx = dx, dy = dy, border = border,
                                     k0 = k0, vertical_ref = vertical_ref, trim = trim)

   return pm_collection


def get_pixel_map(grid, width, height, exact_resolution = False,
                  min_width = None, max_width = None, min_height = None, max_height = None,
                  origin = None, dx = None, dy = None, border = 0.0,
                  k0 = None, vertical_ref = 'top', trim = True):
   """Finds or makes a suitable pixel map and returns numpy int array holding map.

   arguments:
      width (int): the width of the pixel rectangle (number of pixels)
      height (int): the height of the pixel rectangle (number of pixels)
      exact_resolution (bool, default False): if True, equivalent to setting the following min & max
         arguments equal to width and to height, forcing an exact match to the given resolution
      min_width (int, optional): if present, only pixel maps of at least this width are eligible
      max_width (int, optional): if present, only pixel maps of at most this width are eligible
      min_height (int, optional): if present, only pixel maps of at least this height are eligible
      max_height (int, optional): if present, only pixel maps of at most this height are eligible
      origin (float pair, optional): x, y of south west corner of area covered by pixel rectangle,
         in local crs of grid; if None then the minimum point of the grid's box is used and border
         applied
      dx (float, optional): the size (west to east) of a pixel, in locel crs of grid; if dx and dy
         are both None, they are computed from the grid's box, border, width and height
      dy (float, optional): the size (south to north) of a pixel, in locel crs; if only one of dx
         or dy is given, the other is set the same; if both None they are computed (see dx)
      border (float, default 0.0): a border width, in local crs of grid, applied around bounding box
         of grid when computing origin and/or dx & dy; ignored if those values are all supplied
      k0 (int, default None): the layer to create a 2D pixel map for
      vertical_ref (string, default 'top'): 'top' or 'base'
      trim (boolean, default True): if True, width or height may be trimmed to leave equal border

   returns:
      resqpy.property.PropertyCollection holding at least one pixel map property

   notes:
      if exact_resolution is True, the individual min_ and max_ arguments are ignored;
      the origin, dx, dy, border, k0 & vertical_ref arguments are only used if a new pixel map is created;
      the epc is not rewritten by this function, even if new parts have been introduced and the hdf5 written to;
      calling code should use model.store_epc(only_if_modifed = True);
      if trim is True, the resulting map may be smaller than min_height or min_width
   """

   log.debug(f'getting pixel map with width, height: {width}, {height}')

   pm_collection = find_or_make_pixel_map(grid, width, height, exact_resolution = exact_resolution,
                                          min_width = min_width, max_width = max_width,
                                          min_height = min_height, max_height = max_height,
                                          origin = origin, dx = dx, dy = dy, border = border,
                                          k0 = k0, vertical_ref = vertical_ref, trim = trim)

   pm_count = pm_collection.number_of_parts()
   assert pm_collection is not None and pm_count > 0
   pm_parts = pm_collection.parts()

   part = pm_parts[0]  # arbitrarily pick the first qualifying pixel map
   pixel_map = pm_collection.cached_part_array_ref(part, masked = False)
   assert pixel_map is not None

   return pixel_map


def width_and_height_of_pixel_map(pixel_map):
   """Returns the resolution of the pixel map as two integers being width, height."""

   assert pixel_map.ndim == 3 and pixel_map.shape[2] == 2
   height, width = pixel_map.shape[:2]
   return width, height


def colour_array(norm_prop, colour_map):
   """For a normalized float array, returns a numpy uint8 array with extent of extra last axis being 3 (RGB).

   arguments:
      norm_prop (numpy float array): array of data normalised to values between 0.0 and 1.0
      colour_map (matplotlib.cm colormap): the colour map to use to represent the normalised data

   returns:
      numpy uint8 array with same shape as norm_prop except with an extra axis appended with extent 3;
      the returned array holds RGB values for each of the input normalised elements
   """

   return (255.0 * (colour_map(norm_prop)[..., :3] + 0.00392)).astype(np.uint8)


def map_image(pixel_map, c_array, k0 = 0, image_file = None, background = 'lightgrey', file_only = False,
              title = None, title_location = (10, 10), title_colour = 'black'):
   """Produce a map image, written to file in PNG format and returned in a form to be assigned to an Image widget.

   arguments:
      pixel_map (numpy int array): pixel mapping array as returned by get_pixel_map()
      c_array (numpy uint8 array): RGB data for each cell in the source grid property, as returned by colour_array()
      k0 (int, default 0): the layer number to use from c_array, if for 3D property data
      image_file (string): the filename to write the PNG format image to; if None, a temporary file is written
         and removed immediately
      background (string, default 'lightgrey'): the matplotlib color to use for the background (unmapped pixels)
      file_only (boolean, default False): if True, the PNG file is written but None is returned
      title (string, optional): if present, text to be added to the map image
      title_location (pair of ints, default (10, 10)): pixel location to place origin of title, as (right, down)
         from top left corner of image; ignored if no title
      title_colour (string, default 'black'): the colour to stroke the title with; ignored if no title

   returns:
      image data suitable for assigning to the value attribute of an Image widget with a format of 'png',
      or None if file_only is True
   """

   assert pixel_map.ndim == 3 and pixel_map.shape[-1] == 2
   assert image_file or not file_only
   assert c_array.ndim == 4 or (c_array.ndim == 3 and k0 == 0)
   assert c_array.shape[-1] == 3   # todo: could allow for alpha channel?

   if c_array.ndim == 3: c_array = c_array.reshape(tuple([1] + list(c_array.shape)))
   assert 0 <= k0 < c_array.shape[0]

   remove_file = False
   if not image_file:
      image_file = '/tmp/map_' + str(bu.new_uuid()) + '.png'
      remove_file = True

   height, width = pixel_map.shape[:2]

   # create a new image using the Python Image Library package
   im = Image.new('RGB', (width, height), background)

   # get reference to actual pixels for image
   pixels = im.load()

   # each pixel has to be set individually!
   for row in range(height):
      pix_row = height - row - 1   # image starts at top left, whilst our pixel has origin bottom left
      for col in range(width):
          cell = pixel_map[row, col]
          if cell[0] < 0: continue  # pixel does not map to any IJ column, leave as background
          # colour-in pixel: RGB values must be presented as a tuple of 3 ints
          pixels[col, pix_row] = tuple(c_array[k0, cell[0], cell[1], :])

   if title:
      _add_text(im, title_location, title, title_colour)

   # save image to disc in PNG format
   im.save(image_file)

   png_image = None

   if not file_only:
      with open(image_file, "rb") as fh:
         png_image = fh.read()

   if remove_file: os.remove(image_file)

   return png_image


def _add_text(im, location, message, colour):
    """Add message text to image."""

    dim = ImageDraw.Draw(im)
    dim.text(location, message, fill = colour)
