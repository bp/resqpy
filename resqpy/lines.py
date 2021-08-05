"""polylines.py: Resqml polylines module."""

version = '14th July 2021'

import logging

log = logging.getLogger(__name__)
log.debug('polylines.py version ' + version)

from resqpy.olio.base import BaseResqpy
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vu
import resqpy.olio.intersection as meet
import resqpy.olio.point_inclusion as pip
from resqpy.olio.xml_namespaces import curly_namespace as ns
import resqpy.olio.write_hdf5 as rwh5
import resqpy.crs as rcrs
import resqpy.olio.simple_lines as rsl
import resqpy.organize as rqo

import math as maths
import numpy as np
import os


class _BasePolyline(BaseResqpy):
   """Base class to implement shared methods for other classes in this module"""

   def create_interpretation_and_feature(self,
                                         kind = 'horizon',
                                         name = None,
                                         interp_title_suffix = None,
                                         is_normal = True):
      """Creates xml and objects for a represented interpretaion and interpreted feature, if not already present."""

      assert kind in ['horizon', 'fault', 'fracture', 'geobody boundary']
      assert name or self.title, 'title missing'
      if not name:
         name = self.title

      if self.rep_int_root is not None:
         log.debug(f'represented interpretation already exisrs for surface {self.title}')
         return
      if kind in ['horizon', 'geobody boundary']:
         feature = rqo.GeneticBoundaryFeature(self.model, kind = kind, feature_name = name)
         feature.create_xml()
         if kind == 'horizon':
            interp = rqo.HorizonInterpretation(self.model, genetic_boundary_feature = feature, domain = 'depth')
         else:
            interp = rqo.GeobodyBoundaryInterpretation(self.model, genetic_boundary_feature = feature, domain = 'depth')
      elif kind in ['fault', 'fracture']:
         feature = rqo.TectonicBoundaryFeature(self.model, kind = kind, feature_name = name)
         feature.create_xml()
         interp = rqo.FaultInterpretation(self.model,
                                          is_normal = is_normal,
                                          tectonic_boundary_feature = feature,
                                          domain = 'depth')  # might need more arguments
      else:
         log.critical('code failure')
      interp_root = interp.create_xml(title_suffix = interp_title_suffix)
      self.rep_int_root = interp_root


def load_hdf5_array(object, node, array_attribute, tag = 'Values'):
   """Loads the property array data as an attribute of object, from the hdf5 referenced in xml node.

   :meta private:
   """

   assert (rqet.node_type(node) in ['DoubleHdf5Array', 'IntegerHdf5Array', 'Point3dHdf5Array'])
   # ignore null value
   h5_key_pair = object.model.h5_uuid_and_path_for_node(node, tag = tag)
   if h5_key_pair is None:
      return None
   return object.model.h5_array_element(h5_key_pair,
                                        index = None,
                                        cache_array = True,
                                        object = object,
                                        array_attribute = array_attribute)


class Polyline(_BasePolyline):
   """Class for RESQML polyline representation."""

   resqml_type = 'PolylineRepresentation'

   def __init__(self,
                parent_model,
                poly_root = None,
                uuid = None,
                set_bool = None,
                set_coord = None,
                set_crs = None,
                set_crsroot = None,
                title = None,
                rep_int_root = None,
                originator = None,
                extra_metadata = None):
      """Initialises a new PolylineRepresentation object.

        arguments:
            parent_model (model.Model object): the model which the new PolylineRepresentation belongs to
            poly_root (DEPRECATED): use uuid instead;
                the root node of the xml tree representing the PolylineRepresentation;
                if not None, the new PolylineRepresentation object is initialised based on data in tree;
                if None, wait for further improvements
            uuid (uuid.UUID, optional): the uuid of an existing RESQML PolylineRepresentation from which
                to initialise the resqpy Polyline
            set_bool (boolean, optional): if True, a new polyline created from coordinates is flagged as
                a closed polyline (polygon); ignored if uuid or poly_root is not None
            set_coord (numpy array of shape (..., 3), optional): an ordered set of xyz values used to define
                a new polyline; ignored if uuid or poly_root is not None
            set_crs (uuid.UUID, optional): the uuid of a crs to be used when initialising from coordinates;
                ignored if uuid or poly_root is not None
            set_crsroot (DEPRECATED): the xml root node for the crs; ignored
            title (str, optional): the citation title to use for a new polyline;
                ignored if uuid or poly_root is not None
            rep_int_root
            originator (str, optional): the name of the person creating the polyline, defaults to login id;
                ignored if uuid or poly_root is not None
            extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the polyline;
                ignored if uuid or poly_root is not None

        returns:
            the newly instantiated Polyline object

        :meta common:
        """

      self.model = parent_model
      self.isclosed = set_bool
      self.nodepatch = None
      self.crs_uuid = set_crs
      self.coordinates = None
      self.centre = None
      self.rep_int_root = rep_int_root  # Optional
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       root_node = poly_root,
                       extra_metadata = extra_metadata)

      if self.root is None and all(i is not None for i in [set_bool, set_coord, set_crs, title]):
         # Using data from a polyline set
         assert set_coord.ndim > 1 and 2 <= set_coord.shape[-1] <= 3
         if rep_int_root is not None:
            self.rep_int_root = rep_int_root
         else:
            self.rep_int_root = None
         # allow for x,y or x,y,z incoming coordinates but use x,y,z internally
         coord_shape = list(set_coord.shape)
         coord_shape[-1] = 3
         self.coordinates = np.zeros(tuple(coord_shape))
         self.coordinates[..., :set_coord.shape[-1]] = set_coord
         if set_coord.ndim > 2:
            self.coordinates = self.coordinates.reshape((-1, 3))
         self.nodepatch = (0, len(self.coordinates))
         assert not any(map(lambda x: x is None, self.nodepatch))  # Required fields - assert neither are None

      # TODO: Add SeismicCoordinates later - optional field
      # TODO: Add LineRole later - optional field

   def _load_from_xml(self):

      assert self.root is not None  # polyline xml node is specified
      poly_root = self.root

      self.title = rqet.citation_title_for_node(poly_root)

      self.extra_metadata = rqet.load_metadata_from_xml(self.root)

      self.isclosed = rqet.bool_from_text(rqet.node_text(rqet.find_tag(poly_root, 'IsClosed')))
      assert self.isclosed is not None  # Required field

      patch_node = rqet.find_tag(poly_root, 'NodePatch')
      assert patch_node is not None  # Required field

      geometry_node = rqet.find_tag(patch_node, 'Geometry')
      assert geometry_node is not None  # Required field

      self.crs_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(geometry_node, ['LocalCrs', 'UUID']))
      assert self.crs_uuid is not None  # Required field

      points_node = rqet.find_tag(geometry_node, 'Points')
      assert points_node is not None  # Required field
      load_hdf5_array(self, points_node, 'coordinates', tag = 'Coordinates')

      self.nodepatch = (rqet.find_tag_int(patch_node, 'PatchIndex'), rqet.find_tag_int(patch_node, 'Count'))
      assert not any(map(lambda x: x is None, self.nodepatch))  # Required fields - assert neither are None

      self.rep_int_root = self.model.referenced_node(rqet.find_tag(poly_root, 'RepresentedInterpretation'))

   @property
   def crs_root(self):
      """XML node corresponding to self.crs_uuid"""

      return self.model.root_for_uuid(self.crs_uuid)

   @property
   def rep_int_uuid(self):
      # TODO: Track uuid only, not root
      return rqet.uuid_for_part_root(self.rep_int_root)

   def is_convex(self, trust_metadata = True):
      """Returns True if the polyline is closed and convex in the xy plane, otherwise False."""

      if not self.isclosed:
         return False
      if trust_metadata and self.extra_metadata is not None and 'is_convex' in self.extra_metadata.keys():
         return self.extra_metadata['is_convex']
      nodes = len(self.coordinates)
      extras = {}
      if nodes < 3:
         result = False
      else:
         cw_bool = None
         result = True
         for node in range(3, nodes):
            cw = vu.clockwise(self.coordinates[node - 2], self.coordinates[node - 1], self.coordinates[node])
            if cw == 0.0:
               continue  # striaght line section
            if cw_bool is None:
               cw_bool = (cw > 0.0)
            elif cw_bool != (cw > 0.0):
               result = False
               break
         if cw_bool is None:
            result = False  # whole polyline lies in a straight line
         if result:
            extras['is_clockwise'] = cw_bool
      extras['is_convex'] = result
      self.append_extra_metadata(extras)
      return result

   def point_is_inside_xy(self, p, mode = 'crossing'):
      """Returns True if point p is inside closed polygon, in xy plane, otherwise False.

        :meta common:
        """

      assert mode in ['crossing', 'winding'], 'unrecognised point inclusion mode: ' + str(mode)
      assert self.isclosed, 'point inclusion is not applicable to unclosed polylines'

      if mode == 'crossing':
         return pip.pip_cn(p, self.coordinates)
      return pip.pip_wn(p, self.coordinates)

   def segment_length(self, segment_index, in_xy = False):
      """Returns the naive length (ie. assuming x,y & z units are the same) of an individual segment of the polyline."""

      successor = segment_index + 1
      if self.isclosed:
         assert 0 <= segment_index < len(self.coordinates)
         if segment_index == len(self.coordinates) - 1:
            successor = 0
      else:
         assert 0 <= segment_index < len(self.coordinates) - 1

      d = 2 if in_xy else 3
      return vu.naive_length(self.coordinates[successor, :d] - self.coordinates[segment_index, :d])

   def full_length(self, in_xy = False):
      """Returns the naive length of the entire polyline.

        :meta common:
        """

      length = 0.0
      end_index = len(self.coordinates) - 1
      if self.isclosed:
         end_index += 1
      for i in range(end_index):
         length += self.segment_length(i, in_xy = in_xy)
      return length

   def interpolated_point(self, fraction, in_xy = False):
      """Returns x,y,z point on the polyline at fractional distance along entire polyline.

        :meta common:
        """

      assert 0.0 <= fraction <= 1.0
      target = fraction * self.full_length(in_xy = in_xy)
      seg_index = 0
      while True:
         next_seg_length = self.segment_length(seg_index, in_xy = in_xy)
         if target > next_seg_length or next_seg_length == 0.0:
            target -= next_seg_length
            seg_index += 1
            continue
         seg_fraction = target / next_seg_length
         successor = (seg_index + 1) % len(self.coordinates)
         return seg_fraction * self.coordinates[successor] + (1.0 - seg_fraction) * self.coordinates[seg_index]

   def equidistant_points(self, n, in_xy = False):
      """Returns array of shape (n, 3) being points equally distributed along entire polyline."""

      assert n > 1
      sample_points = np.empty((n, 3))
      if self.isclosed:
         step_fraction = 1.0 / float(n)
      else:
         step_fraction = 1.0 / float(n - 1)
      f = 0.0
      for step in range(n):
         sample_points[step] = self.interpolated_point(f, in_xy = in_xy)
         f += step_fraction
         if f > 1.0:
            f = 1.0  # to avoid possible assertion error due to rounding
      return sample_points

   def balanced_centre(self, mode = 'weighted', n = 20, cache = True, in_xy = False):
      """Returns a mean x,y,z based on sampling polyline at regular intervals."""

      if cache and self.centre is not None:
         return self.centre
      assert mode in ['weighted', 'sampled']
      if mode == 'sampled':  # this mode is deprecated as it simply approximates the weighted mode
         sample_points = self.equidistant_points(n, in_xy = in_xy)
         centre = np.mean(sample_points, axis = 0)
      else:  # 'weighted'
         sum = np.zeros(3)
         seg_count = len(self.coordinates) - 1
         if self.isclosed:
            seg_count += 1
         d = 2 if in_xy else 3
         for seg_index in range(seg_count):
            successor = (seg_index + 1) % len(self.coordinates)
            p1, p2 = self.coordinates[seg_index, :d], self.coordinates[successor, :d]
            sum += (p1 + p2) * vu.naive_length(p2 - p1)
         centre = sum / (2.0 * self.full_length(in_xy = in_xy))
      if cache:
         self.centre = centre
      return centre

   def first_line_intersection(self, x1, y1, x2, y2, half_segment = False):
      """Returns segment number & x, y of first intersection of (half) bounded line x,y 1 to 2 with polyline, or None, None, None."""

      seg_count = len(self.coordinates) - 1
      if self.isclosed:
         seg_count += 1

      for segment in range(seg_count):
         successor = (segment + 1) % len(self.coordinates)
         x, y = meet.line_line_intersect(x1,
                                         y1,
                                         x2,
                                         y2,
                                         self.coordinates[segment][0],
                                         self.coordinates[segment][1],
                                         self.coordinates[successor][0],
                                         self.coordinates[successor][1],
                                         line_segment = True,
                                         half_segment = half_segment)
         if x is not None:
            return segment, x, y
      return None, None, None

   def normalised_xy(self, x, y, mode = 'square'):
      """Returns a normalised x',y' pair (in range 0..1) being point x,y under mapping from convex polygon.

        arguments:
            x, y (floats): location of a point inside the polyline, which must be closed and project to a
                convex polygon in the xy plane
            mode (string): which mapping algorithm to use, one of: 'square', 'circle', or 'perimeter'

        returns:
            x', y' (floats, each in range 0..1) being the normalised representation of point x,y

        notes:
            this method is the inverse of denormalised_xy(), as long as a consistent mode is selected;
            for more details of the mapping used by the 3 modes, see documentation for denormalised_xy()
        """

      assert mode in ['square', 'perimeter', 'circle']
      assert self.is_convex(), 'attempt to find normalised x,y within a polyline that is not a closed convex polygon'
      centre_xy = self.balanced_centre()[:2]
      if tuple(centre_xy) == (x, y):
         if mode == 'square':
            return 0.5, 0.5
         return 0.0, 0.0
      segment, px, py = self.first_line_intersection(centre_xy[0], centre_xy[1], x, y, half_segment = True)
      assert px is not None
      norm_x, norm_y = None, None
      if mode == 'square':
         if px == centre_xy[0]:
            norm_x = 0.5
         else:
            norm_x = 0.5 + 0.5 * (x - centre_xy[0]) / abs(px - centre_xy[0])
         if py == centre_xy[1]:
            norm_y = 0.5
         else:
            norm_y = 0.5 + 0.5 * (y - centre_xy[1]) / abs(py - centre_xy[1])
      elif mode == 'circle':
         d_xy = np.array((x, y)) - centre_xy
         dp_xy = np.array((px, py)) - centre_xy
         if abs(dp_xy[0]) > abs(dp_xy[1]):
            rf = d_xy[0] / dp_xy[0]
            theta = maths.atan(d_xy[1] / d_xy[0])
            if d_xy[0] < 0.0:
               theta += maths.pi
         else:
            rf = d_xy[1] / dp_xy[1]
            theta = maths.pi / 2.0 - maths.atan(d_xy[0] / d_xy[1])
            if d_xy[1] < 0.0:
               theta += maths.pi
         norm_x = rf * rf
         theta /= 2.0 * maths.pi
         if theta < 0.0:
            theta += 1.0
         elif theta > 1.0:
            theta -= 1.0
         norm_y = theta
      elif mode == 'perimeter':
         d_xy = np.array((x, y)) - centre_xy
         dp_xy = np.array((px, py)) - centre_xy
         if abs(dp_xy[0]) > abs(dp_xy[1]):
            rf = d_xy[0] / dp_xy[0]
         else:
            rf = d_xy[1] / dp_xy[1]
         norm_x = rf * rf
         seg_xy = np.array((px, py)) - self.coordinates[segment, :2]
         length = 0.0
         for seg in range(segment):
            length += self.segment_length(seg, in_xy = True)
         length += vu.naive_length(seg_xy)
         norm_y = length / self.full_length(in_xy = True)
      return norm_x, norm_y

   def denormalised_xy(self, norm_x, norm_y, mode = 'perimeter'):
      """Returns a denormalised x,y pair being point norm_x,norm_y (in range 0..1) under mapping onto convex polygon.

        arguments:
            norm_x, norm_y (floats): normalised values, each in range 0..1, identifying a location in a unit shape
            mode (string): 'square', 'circle', or 'perimeter'; if square, norm_x and norm_y are coordinates within a
                unit square; if circle or perimeter, norm_x is the square of a radial fraction between the centre
                of the polygon and the perimeter; if mode is circle, norm_y is simply the polar coordinate bearing
                of the point relative to the polygon centre, with values 0..1 corresponding to 0..2pi radians;
                if mode is perimeter, norm_y is a fractional distance along the length of the closed polyline
                projected onto the xy plane

        returns:
            x, y (floats): the location of a point within the area outlined by the polyline, which must be closed
                and project to a convex polygon in the xy plane

        notes:
            this method is the inverse of normalised_xy(), as long as a consistent mode is selected;
            each mode gives a reversible mapping but there are some variations in density which could introduce
            slightly different biases when working with stochastic locations; density distortion with square and
            circle modes tends to increase as the polygon shape becomes less square (and aligned) or circular
            respectively; perimeter mode is believed to yield least density distortion generally;
            for circle and perimeter modes, the normalised x value is treated as the square of the fractional
            distance to the boundary in order for a rectangular distribution of normalised values to map to
            an even density of denormalised locations
        """

      assert mode in ['square', 'perimeter', 'circle']
      assert 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0
      assert self.is_convex(), 'attempt to find denormalised x,y within a polyline that is not a closed convex polygon'
      centre_xy = self.balanced_centre()[:2]
      x, y = None, None
      if mode == 'square':
         if (norm_x, norm_y) == (0.5, 0.5):
            return tuple(centre_xy)
         n_x = (norm_x - 0.5) + centre_xy[0]
         n_y = (norm_y - 0.5) + centre_xy[1]
         _, px, py = self.first_line_intersection(centre_xy[0], centre_xy[1], n_x, n_y, half_segment = True)
         assert px is not None
         fx = abs(norm_x - 0.5)
         fy = abs(norm_y - 0.5)
         f = 2.0 * max(fx, fy)
         log.debug(f'intersect x,y: {px}, {py}')
         if px == centre_xy[0]:
            x = centre_xy[0]
         else:
            x = centre_xy[0] + f * (px - centre_xy[0])
         if py == centre_xy[1]:
            y = centre_xy[1]
         else:
            y = centre_xy[1] + f * (py - centre_xy[1])
         log.debug(f'denormal x,y: {x}, {y}')
      elif mode == 'perimeter':
         if norm_y == 1.0:
            norm_y = 0.0
         p_xy = np.empty(2)
         p_xy[:] = self.interpolated_point(norm_y)[:2]
         f = maths.sqrt(norm_x)
         x, y = centre_xy + f * (p_xy - centre_xy)
      elif mode == 'circle':
         theta = 2 * maths.pi * norm_y
         n_x = centre_xy[0] + maths.cos(theta)
         n_y = centre_xy[1] + maths.sin(theta)
         p_xy = np.empty(2)
         p_xy[:] = self.first_line_intersection(centre_xy[0], centre_xy[1], n_x, n_y, half_segment = True)[1:]
         f = maths.sqrt(norm_x)
         x, y = centre_xy + f * (p_xy - centre_xy)
      return x, y

   def tangent_vectors(self):
      """Returns a numpy array of unit length tangent vectors, one for each coordinate in the line."""

      return tangents(self.coordinates)

   def splined(self,
               tangent_weight = 'square',
               min_subdivisions = 1,
               max_segment_length = None,
               max_degrees_per_knot = 5.0,
               title = None,
               rep_int_root = None):
      """Retrurns a new Polyline being a cubic spline of this polyline.

        :meta common:
        """

      spline_coords = spline(self.coordinates,
                             tangent_weight = tangent_weight,
                             min_subdivisions = min_subdivisions,
                             max_segment_length = max_segment_length,
                             max_degrees_per_knot = max_degrees_per_knot,
                             closed = self.isclosed)

      if not title:
         title = self.title
      if rep_int_root is None:
         rep_int_root = self.rep_int_root  # todo: check whether it is legal to have 2 representations for 1 interpretation

      return Polyline(self.model,
                      set_bool = self.isclosed,
                      set_coord = spline_coords,
                      set_crs = self.crs_uuid,
                      title = title,
                      rep_int_root = rep_int_root)

   def create_xml(self,
                  ext_uuid = None,
                  add_as_part = True,
                  add_relationships = True,
                  root = None,
                  title = None,
                  originator = None):
      """Create xml from polyline.

        args:
            ext_uuid: the uuid of the hdf5 external part

        :meta common:
        """

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      if title is not None:
         self.title = title
      if self.title is None:
         self.title = 'polyline'

      polyline = super().create_xml(add_as_part = False, originator = originator)

      if self.rep_int_root is not None:
         rep_int = self.rep_int_root
         if "FaultInterpretation" in str(rqet.content_type(rep_int)):
            content_type = 'obj_FaultInterpretation'
         else:
            content_type = 'obj_HorizonInterpretation'
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.citation_title_for_node(rep_int),
                                    rep_int.attrib['uuid'],
                                    content_type = content_type,
                                    root = polyline)

      isclosed = rqet.SubElement(polyline, ns['resqml2'] + 'IsClosed')
      isclosed.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
      isclosed.text = str(self.isclosed).lower()

      nodepatch = rqet.SubElement(polyline, ns['resqml2'] + 'NodePatch')
      nodepatch.set(ns['xsi'] + 'type', ns['resqml2'] + 'NodePatch')
      nodepatch.text = '\n'

      patchindex = rqet.SubElement(nodepatch, ns['resqml2'] + 'PatchIndex')
      patchindex.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
      patchindex.text = str(self.nodepatch[0])

      count = rqet.SubElement(nodepatch, ns['resqml2'] + 'Count')
      count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      count.text = str(self.nodepatch[1])

      geom = rqet.SubElement(nodepatch, ns['resqml2'] + 'Geometry')
      geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'PointGeometry')
      geom.text = '\n'

      self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

      points = rqet.SubElement(geom, ns['resqml2'] + 'Points')
      points.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
      points.text = '\n'

      coords = rqet.SubElement(points, ns['resqml2'] + 'Coordinates')
      coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      coords.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points_patch0', root = coords)

      if root is not None:
         root.append(polyline)
      if add_as_part:
         self.model.add_part('obj_PolylineRepresentation', self.uuid, polyline)
         if add_relationships:
            self.model.create_reciprocal_relationship(polyline, 'destinationObject', self.crs_root, 'sourceObject')
            if self.rep_int_root is not None:  # Optional
               self.model.create_reciprocal_relationship(polyline, 'destinationObject', self.rep_int_root,
                                                         'sourceObject')

            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(polyline, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

      return polyline

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Create or append the coordinates hdf5 array to hdf5 file.

        :meta common:
        """

      if self.uuid is None:
         self.uuid = bu.new_uuid()
      h5_reg = rwh5.H5Register(self.model)
      h5_reg.register_dataset(self.uuid, 'points_patch0', self.coordinates)
      h5_reg.write(file_name, mode = mode)

   def append_extra_metadata(self, meta_dict):
      """Append a given dictionary of metadata to the existing metadata."""

      if self.extra_metadata is None:
         self.extra_metadata = {}
      for key in meta_dict:
         self.extra_metadata[key] = meta_dict[key]


class PolylineSet(_BasePolyline):
   """Class for RESQML polyline set representation."""

   resqml_type = 'PolylineSetRepresentation'

   def __init__(self,
                parent_model,
                set_root = None,
                uuid = None,
                polylines = None,
                irap_file = None,
                charisma_file = None,
                title = None,
                originator = None,
                extra_metadata = None):
      """Initialises a new PolylineSet object.

        arguments:
            parent_model (model.Model object): the model which the new PolylineSetRepresentation belongs to
            set_root (DEPRECATED): use uuid instead;
                the root node of the xml tree representing the PolylineSetRepresentation;
                if not None, the new PolylineSetRepresentation object is initialised based on data in tree;
                if None, expectes a list of polyline objects
            uuid (uuid.UUID, optional): the uuid of an existing RESQML PolylineSetRepresentation object from
                which to initialise this resqpy PolylineSet
            polylines (optional): list of polyline objects from which to build the polylineset
            irap_file (str, optional): the name of a file in irap format from which to import the polyline set
            charisma_file (str, optional): the name of a file in charisma format from which to import the polyline set
            title (str, optional): the citation title to use for a new polyline set;
                ignored if uuid or set_root is not None
            originator (str, optional): the name of the person creating the polyline set, defaults to login id;
                ignored if uuid or set_root is not None
            extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the polyline set;
                ignored if uuid or set_root is not None

        returns:
            the newly instantiated PolylineSet object

        :meta common:
        """

      self.model = parent_model
      self.coordinates = None
      self.count_perpol = None
      self.polys = []
      self.rep_int_root = None
      self.save_polys = False
      self.boolnotconstant = None
      self.boolvalue = None
      self.crs_uuid = None

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       root_node = set_root,
                       extra_metadata = extra_metadata)

      if self.root is not None:
         return

      if polylines is not None:  # Create from list of polylines
         crs_list = []
         for poly in polylines:
            crs_list.append(poly.crs_uuid)
         crs_set = set(crs_list)
         assert len(crs_set) == 1, 'More than one CRS found in input polylines for polyline set'
         for crs_uuid in crs_set:
            self.crs_uuid = crs_uuid
            if self.crs_root is not None:
               break
         self.polys = polylines
         # Setting the title of the first polyline given as the PolylineSet title
         if len(polylines) > 1:
            self.title = f"{polylines[0].title} + {len(polylines)-1} polylines"
         else:
            self.title = polylines[0].title

      elif irap_file is not None:  # Create from an input IRAP file
         inpoints = rsl.read_lines(irap_file)
         self.count_perpol = []
         closed_array = []
         self.title = os.path.basename(irap_file).split(".")[0]
         for i, poly in enumerate(inpoints):
            if len(poly) > 1:  # Polylines must have at least 2 points
               self.count_perpol.append(len(poly))
               if vu.isclose(poly[0], poly[-1]):
                  closed_array.append(True)
               else:
                  closed_array.append(False)
               if i == 0:
                  self.coordinates = poly
               else:
                  self.coordinates = np.concatenate((self.coordinates, poly))
         self.count_perpol = np.array(self.count_perpol)
         if self.crs_root is None:  # If no crs_uuid is provided, assume the main model crs is valid
            self.crs_uuid = self.model.crs_uuid
         self.polys = self.convert_to_polylines(closed_array, self.count_perpol, self.coordinates, self.crs_uuid,
                                                self.crs_root, self.rep_int_root)

      elif charisma_file is not None:
         with open(charisma_file) as f:
            inpoints = f.readlines()
         self.count_perpol = []
         closed_array = []
         self.title = os.path.basename(charisma_file).split(".")[0]
         for i, line in enumerate(inpoints):
            line = line.split()
            if i == 0:
               self.coordinates = (np.array([[float(line[3]), float(line[4]), float(line[5])]]))
               stick = line[7]
               count = 1
            else:
               self.coordinates = np.concatenate(
                  (self.coordinates, np.array(([[float(line[3]), float(line[4]),
                                                 float(line[5])]]))))
               count += 1
               if stick != line[7] or i == len(inpoints) - 1:
                  if count <= 2:  # Line has fewer than 2 points
                     log.info(f"Polylines must contain at least 2 points - ignoring point {self.coordinates[-2]}")
                     self.coordinates = np.delete(self.coordinates, -2, 0)  # Remove the second to last entry
                  else:
                     self.count_perpol.append(count - 1)
                     closed_array.append(False)
                  count = 1
                  stick = line[7]
         self.count_perpol = np.array(self.count_perpol)
         if self.crs_root is None:  # If no crs_uuid is provided, assume the main model crs is valid
            self.crs_uuid = self.model.crs_uuid
         self.polys = self.convert_to_polylines(closed_array, self.count_perpol, self.coordinates, self.crs_uuid,
                                                self.crs_root, self.rep_int_root)

   def _load_from_xml(self):

      assert self.root is not None  # polyline set xml node specified
      root = self.root

      self.rep_int_root = self.model.referenced_node(rqet.find_tag(root, 'RepresentedInterpretation'))

      for patch_node in rqet.list_of_tag(root, 'LinePatch'):  # Loop over all LinePatches - likely just the one
         assert patch_node is not None  # Required field

         geometry_node = rqet.find_tag(patch_node, 'Geometry')
         assert geometry_node is not None  # Required field

         crs_root = self.model.referenced_node(rqet.find_tag(geometry_node, 'LocalCrs'))
         assert crs_root is not None  # Required field
         self.crs_uuid = rqet.uuid_for_part_root(crs_root)
         assert self.crs_uuid is not None  # Required field

         closed_node = rqet.find_tag(patch_node, 'ClosedPolylines')
         assert closed_node is not None  # Required field
         # The ClosedPolylines could be a BooleanConstantArray, or a BooleanArrayFromIndexArray
         closed_array = self.get_bool_array(closed_node)

         count_node = rqet.find_tag(patch_node, 'NodeCountPerPolyline')
         load_hdf5_array(self, count_node, 'count_perpol', tag = 'Values')

         points_node = rqet.find_tag(geometry_node, 'Points')
         assert points_node is not None  # Required field
         load_hdf5_array(self, points_node, 'coordinates', tag = 'Coordinates')

         # Check that the number of bools aligns with the number of count_perpoly
         # Check that the total of the count_perpoly aligns with the number of coordinates
         assert len(self.count_perpol) == len(closed_array)
         assert np.sum(self.count_perpol) == len(self.coordinates)

         subpolys = self.convert_to_polylines(closed_array, self.count_perpol, self.coordinates, self.crs_uuid,
                                              self.crs_root, self.rep_int_root)
         # Check we have the right number of polygons
         assert len(subpolys) == len(self.count_perpol)

         # Remove duplicate coordinates and count arrays (exist in polylines now)
         # delattr(self,'coordinates')
         # delattr(self,'count_perpol')

         self.polys.extend(subpolys)

   @property
   def crs_root(self):
      """XML node corresponding to self.crs_uuid"""

      return self.model.root_for_uuid(self.crs_uuid)

   def poly_index_containing_point_in_xy(self, p, mode = 'crossing'):
      """Returns the index of the first (closed) polyline containing point p in the xy plane, or None.

        :meta common:
        """

      assert mode in ['crossing', 'winding'], 'unrecognised mode when looking for polygon containing point'

      for i, poly in enumerate(self.polys):
         if not poly.isclosed:
            continue
         if poly.point_is_inside_xy(p, mode = mode):
            return i
      return None

   def create_xml(self,
                  ext_uuid = None,
                  add_as_part = True,
                  add_relationships = True,
                  root = None,
                  title = None,
                  originator = None,
                  save_polylines = False):
      """Create xml from polylineset

        args:
            save_polylines: If true, polylines are also saved individually

        :meta common:
        """

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      self.save_polys = save_polylines

      if title:
         self.title = title
      if not self.title:
         self.title = 'polyline set'

      if self.save_polys:
         for poly in self.polys:
            poly.create_xml(ext_uuid, add_relationships = add_relationships, originator = originator)

      polyset = super().create_xml(add_as_part = False, originator = originator)

      if self.rep_int_root is not None:
         rep_int = self.rep_int_root
         if "FaultInterpretation" in str(rqet.content_type(rep_int)):
            content_type = 'obj_FaultInterpretation'
         else:
            content_type = 'obj_HorizonInterpretation'
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.citation_title_for_node(rep_int),
                                    rep_int.attrib['uuid'],
                                    content_type = content_type,
                                    root = polyset)

      # We convert all Polylines to the CRS of the first Polyline in the set, so set this as crs_uuid

      patch = rqet.SubElement(polyset, ns['resqml2'] + 'LinePatch')
      patch.set(ns['xsi'] + 'type', ns['resqml2'] + 'PolylineSetPatch')
      patch.text = '\n'

      pindex = rqet.SubElement(patch, ns['resqml2'] + 'PatchIndex')
      pindex.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
      pindex.text = '0'

      if self.boolnotconstant:
         # We have mixed data - use a BooleanArrayFromIndexArray
         closed = rqet.SubElement(patch, ns['resqml2'] + 'ClosedPolylines')
         closed.set(ns['xsi'] + 'type', ns['xsd'] + 'BooleanArrayFromIndexArray')
         closed.text = '\n'

         bool_val = rqet.SubElement(closed, ns['resqml2'] + 'Value')
         bool_val.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
         bool_val.text = str(self.boolvalue).lower()

         ind_val = rqet.SubElement(closed, ns['resqml2'] + 'Indices')
         ind_val.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         ind_val.text = '\n'

         count = rqet.SubElement(closed, ns['resqml2'] + 'Count')
         count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         count.text = str(len(self.count_perpol))

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'indices_patch0', root = ind_val)
      else:
         # All bools are the same - use a BooleanConstantArray
         closed = rqet.SubElement(patch, ns['resqml2'] + 'ClosedPolylines')
         closed.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanConstantArray')
         closed.text = '\n'
         bool_val = rqet.SubElement(closed, ns['resqml2'] + 'Value')
         bool_val.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
         bool_val.text = str(self.boolvalue).lower()
         count = rqet.SubElement(closed, ns['resqml2'] + 'Count')
         count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         count.text = str(len(self.count_perpol))

      count_pp = rqet.SubElement(patch, ns['resqml2'] + 'NodeCountPerPolyline')
      count_pp.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
      count_pp.text = '\n'

      null = rqet.SubElement(count_pp, ns['resqml2'] + 'NullValue')
      null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
      null.text = '0'

      count_val = rqet.SubElement(count_pp, ns['resqml2'] + 'Values')
      count_val.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      count_val.text = '\n'

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'NodeCountPerPolyline_patch0', root = count_val)

      geom = rqet.SubElement(patch, ns['resqml2'] + 'Geometry')
      geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'PointGeometry')
      geom.text = '\n'

      self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

      points = rqet.SubElement(geom, ns['resqml2'] + 'Points')
      points.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
      points.text = '\n'

      coords = rqet.SubElement(points, ns['resqml2'] + 'Coordinates')
      coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      coords.text = '\n'

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points_patch0', root = coords)

      if root is not None:
         root.append(polyset)
      if add_as_part:
         self.model.add_part('obj_PolylineSetRepresentation', self.uuid, polyset)
         if add_relationships:
            self.model.create_reciprocal_relationship(polyset, 'destinationObject', self.crs_root, 'sourceObject')
            if self.rep_int_root is not None:  # Optional
               self.model.create_reciprocal_relationship(polyset, 'destinationObject', self.rep_int_root,
                                                         'sourceObject')
            if self.save_polys:
               for poly in self.polys:
                  self.model.create_reciprocal_relationship(polyset, 'destinationObject', poly.root_node,
                                                            'sourceObject')

            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(polyset, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

      return polyset

   def write_hdf5(self, file_name = None, mode = 'a', save_polylines = False):
      """Create or append the coordinates, counts and indices hdf5 arrays to hdf5 file.

        :meta common:
        """

      if self.uuid is None:
         self.uuid = bu.new_uuid()
      self.combine_polylines(self.polys)
      self.bool_array_format(self.closed_array)
      self.save_polys = save_polylines
      if self.save_polys:
         for poly in self.polys:
            poly.write_hdf5(file_name)

      h5_reg = rwh5.H5Register(self.model)
      h5_reg.register_dataset(self.uuid, 'points_patch0', self.coordinates)
      h5_reg.register_dataset(self.uuid, 'NodeCountPerPolyline_patch0', self.count_perpol.astype(np.int32))
      if self.boolnotconstant:
         h5_reg.register_dataset(self.uuid, 'indices_patch0', self.indices)
      h5_reg.write(file_name, mode = mode)

   def get_bool_array(self, closed_node):
      # TODO: Check if also defined boolean arrays
      """ Returns a boolean array using details in the node location.

        If type of boolean array is BooleanConstantArray, uses the array value and count to generate the array. If type of boolean array is BooleanArrayFromIndexArray, find the "other" value bool and indices of the "other" values, and insert these into an array opposite to the main bool.

        args:
            closed_node: the node under which the boolean array information sits"""
      if rqet.node_type(closed_node) == 'BooleanConstantArray':
         count = rqet.find_tag_int(closed_node, 'Count')
         value = rqet.bool_from_text(rqet.node_text(rqet.find_tag(closed_node, 'Value')))
         return np.full((count), value)
      elif rqet.node_type(closed_node) == 'BooleanArrayFromIndexArray':
         count = rqet.find_tag_int(closed_node, 'Count')
         indices_arr = load_hdf5_array(self, closed_node, 'indices_arr', tag = 'Indices')
         istrue = rqet.bool_from_text(rqet.node_text(rqet.find_tag(closed_node, 'IndexIsTrue')))
         out = np.full((count), not istrue)
         out[indices_arr] = istrue
         return out

   def convert_to_polylines(
         self,
         closed_array = None,
         count_perpol = None,
         coordinates = None,
         crs_uuid = None,
         crs_root = None,  # deprecated
         rep_int_root = None):
      """Returns a list of Polylines objects from a PolylineSet

        note:
            all arguments are optional and by default the data will be taken from self

        args:
            closed_array: array containing a bool for each polygon in if it is open (False) or closed (True)
            count_perpol: array containing a list of polygon "lengths" for each polygon
            coordinates: array containing coordinates for all the polygons
            crs_uuid: crs_uuid for polylineset
            crs_root: DEPRECATED; ignored
            rep_int_root: represented interpretation root (optional)

        returns:
            list of polyline objects

        :meta common:
        """

      if count_perpol is None:
         count_perpol = self.count_perpol
      if closed_array is None:
         closed_array = np.zeros(count_perpol, dtype = bool)
         closed_node = rqet.find_nested_tag(self.root, ['LinePatch', 'ClosedPolylines'])
         if closed_node is not None:
            closed_array[:] = self.get_bool_array(closed_node)
      if coordinates is None:
         coordinates = self.coordinates
      if crs_uuid is None:
         crs_uuid = self.crs_uuid
      if rep_int_root is None:
         rep_int_root = self.rep_int_root
      polys = []
      count = 0
      for i in range(len(count_perpol)):
         if i != len(count_perpol) - 1:
            subset = coordinates[count:int(count_perpol[i]) + count].copy()
         else:
            subset = coordinates[count:int(count_perpol[i]) + count + 1].copy()
         if vu.isclose(subset[0], subset[-1]):
            isclosed = True
         else:
            isclosed = closed_array[i]
         count += int(count_perpol[i])
         subtitle = f"{self.title} {i+1}"
         polys.append(
            Polyline(self.model,
                     poly_root = None,
                     set_bool = isclosed,
                     set_coord = subset,
                     set_crs = crs_uuid,
                     title = subtitle,
                     rep_int_root = rep_int_root))

      return polys

   def combine_polylines(self, polylines):
      """Combines the isclosed boolean array, coordinates and count data for a list of polyline objects

        args:
            polylines: list of polyline objects
        """

      self.count_perpol = []
      self.closed_array = []

      for poly in polylines:
         if poly == polylines[0]:
            master_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
            self.crs_uuid = poly.crs_uuid
            self.coordinates = poly.coordinates.copy()
         else:
            curr_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
            if not curr_crs.is_equivalent(master_crs):
               shifted = curr_crs.convert_array_to(master_crs, poly.coordinates)
               self.coordinates = np.concatenate((self.coordinates, shifted))
            else:
               self.coordinates = np.concatenate((self.coordinates, poly.coordinates))

         self.closed_array.append(poly.isclosed)
         self.count_perpol.append(int(len(poly.coordinates)))

      self.count_perpol = np.array(self.count_perpol)

      assert len(self.closed_array) == len(self.count_perpol)
      assert np.sum(self.count_perpol) == len(self.coordinates)

   def bool_array_format(self, closed_array):
      """Determines an appropriate output boolean array format from an input array of bools

        self.boolnotconstant - set to True if all are not open or all closed
        self.boolvalue - value of isclosed for all polylines, or for the majority of polylines if mixed
        self.indices - array of indices where the values are not self.boolvalue, if the polylines are mixed
        """

      self.indices = []
      self.boolnotconstant = False
      if all(closed_array):
         self.boolvalue = True
      elif not all(closed_array) and not any(closed_array):
         self.boolvalue = False
      else:
         if np.count_nonzero(closed_array) > (len(closed_array) / 2):
            self.boolvalue = True
            for i, val in enumerate(closed_array):
               if not val:
                  self.indices.append(i)
         else:
            self.boolvalue = False
            for i, val in enumerate(closed_array):
               if val:
                  self.indices.append(i)
      if len(self.indices) > 0:
         self.boolnotconstant = True

   def set_interpretation_root(self, rep_int_root, recursive = True):
      """Updates the rep_int_root for the polylineset

        args:
            rep_int_root: new rep_int_root
            recursive: boolean, if true will update individual polys with same root
        """

      self.rep_int_root = rep_int_root

      if recursive:
         for poly in self.polys:
            poly.rep_int_root = rep_int_root

   def convert_to_irap(self, file_name):
      """Output an irap file from a polyline set.

        If file_name exists, it will be overwritten.

        args:
            file_name: output file name for polyline set representation
        """

      end_of_line = np.array([[999.0, 999.0, 999.0]])
      for poly in self.polys:
         if poly == self.polys[0]:
            out_coords = poly.coordinates
         else:
            out_coords = np.concatenate((out_coords, poly.coordinates))
         out_coords = np.concatenate((out_coords, end_of_line))
      np.savetxt(file_name, out_coords, delimiter = ' ')

   def convert_to_charisma(self, file_name):
      """Output to Charisma fault sticks from a polyline set.

        If file_name exists, it will be overwritten.

        args:
            file_name: output file name for polyline set representation
        """

      faultname = self.title.replace(" ", "_")
      lines = []
      for i, poly in enumerate(self.polys):
         for point in poly.coordinates:
            lines.append(f"INLINE-\t0\t0\t{point[0]}\t{point[1]}\t{point[2]}\t{faultname}\t{i+1}\n")
      with open(file_name, 'w') as f:
         for item in lines:
            f.write(item)

   def append_extra_metadata(self, meta_dict):
      """Append a given dictionary of metadata to the existing metadata."""

      if self.extra_metadata is None:
         self.extra_metadata = {}
      for key in meta_dict:
         self.extra_metadata[key] = meta_dict[key]


def shift_polyline(parent_model, poly_root, xyz_shift = (0, 0, 0), title = ''):
   """Returns a new polyline object, shifted by given coordinates."""

   poly = Polyline(parent_model = parent_model, poly_root = poly_root)
   if title != '':
      poly.title = title
   else:
      poly.title = poly.title + f" shifted by xyz({xyz_shift})"
   poly.uuid = bu.new_uuid()
   poly.coordinates = np.array(xyz_shift) + poly.coordinates
   return poly


def flatten_polyline(parent_model, poly_root, axis = "z", value = 0.0, title = ''):
   """Returns a new polyline object, flattened (projected) on a chosen axis to a given value."""

   axis = axis.lower()
   value = float(value)
   assert axis in ["x", "y", "z"], 'Axis must be x, y or z'
   poly = Polyline(parent_model = parent_model, poly_root = poly_root)
   if title != '':
      poly.title = title
   else:
      poly.title = poly.title + f" flattened in {axis} to value {value:.3f}"
   poly.uuid = bu.new_uuid()
   index = "xyz".index(axis)
   poly.coordinates[..., index] = value
   return poly


def tangents(points, weight = 'linear', closed = False):
   """Returns a numpy array of tangent unit vectors for an ordered list of points.

    arguments:
        points (numpy float array of shape (N, 3)): points defining a line
        weight (string, default 'linear'): one of 'linear', 'square' or 'cube', giving
            increased weight to relatively shorter of 2 line segments at each knot
        closed (boolean, default False): if True, the points are treated as a closed
            polyline with regard to end point tangents, otherwise as an open line

    returns:
        numpy float array of the same shape as points, containing a unit length tangent
        vector for each knot (point)

    note:
        if two neighbouring points are identical, a divide by zero will occur
    """

   def one_tangent(points, k1, k2, k3, weight):
      v1 = points[k2] - points[k1]
      v2 = points[k3] - points[k2]
      l1 = vu.naive_length(v1)
      l2 = vu.naive_length(v2)
      if weight == 'square':
         return vu.unit_vector(v1 / (l1 * l1) + v2 / (l2 * l2))
      elif weight == 'cube':
         return vu.unit_vector(v1 / (l1 * l1 * l1) + v2 / (l2 * l2 * l2))
      else:  # linear weight mode
         return vu.unit_vector(v1 / l1 + v2 / l2)

   assert points.ndim == 2 and points.shape[1] == 3
   assert weight in ['linear', 'square', 'cube']

   knot_count = len(points)
   assert knot_count > 1
   tangent_vectors = np.empty((knot_count, 3))

   for knot in range(1, knot_count - 1):
      tangent_vectors[knot] = one_tangent(points, knot - 1, knot, knot + 1, weight)
   if closed:
      assert knot_count > 2, 'closed poly line must contain at least 3 knots for tangent generation'
      tangent_vectors[0] = one_tangent(points, -1, 0, 1, weight)
      tangent_vectors[-1] = one_tangent(points, -2, -1, 0, weight)
   else:
      tangent_vectors[0] = vu.unit_vector(points[1] - points[0])
      tangent_vectors[-1] = vu.unit_vector(points[-1] - points[-2])

   return tangent_vectors


def spline(points,
           tangent_vectors = None,
           tangent_weight = 'square',
           min_subdivisions = 1,
           max_segment_length = None,
           max_degrees_per_knot = 5.0,
           closed = False):
   """Returns a numpy array containing resampled cubic spline of line defined by points.

    arguments:
        points (numpy float array of shape (N, 3)): points defining a line
        tangent_vectors (numpy float array of shape (N, 3), optional) if present, tangent
            vectors to use in the construction of the cubic spline; if None, tangents are
            calculated
        tangent_weight (string, default 'linear'): one of 'linear', 'square' or 'cube', giving
            increased weight to relatively shorter of 2 line segments at each knot when computing
            tangent vectors; ignored if tangent_vectors is not None
        min_subdivisions (int, default 1): the resulting line will have at least this number
            of segments per original line segment
        max_segment_length (float, optional): if present, resulting line segments will not
            exceed this length by much (see notes)
        max_degrees_per_knot (float, default 5.0): the change in direction at each resulting
            knot will not usually exceed this value (see notes)
        closed (boolean, default False): if True, the points are treated as a closed
            polyline with regard to end point tangents, otherwise as an open line

    returns:
        numpy float array of shape (>=N, 3) being knots on a cubic spline defined by points;
        original points are a subset of returned knots

    notes:
        the max_segment_length argument, if present, is compared with the length of each original
        segment to give a lower bound on the number of derived line segments; as the splined line
        may have extra curvature, the length of individual segments in the returned line can exceed
        the argument value, though usually not by much
        similarly, the max_degrees_per_knot is compared to the original deviations to provide another
        lower bound on the number of derived line segments for each original segment; as the spline
        may divide the segment unequally and also sometimes add loops, the resulting deviations can
        exceed the argument value
    """

   assert points.ndim == 2 and points.shape[1] == 3
   knot_count = len(points)
   assert knot_count > (2 if closed else 1)
   if tangent_vectors is None:
      assert tangent_weight in ['linear', 'square', 'cube']
      tangent_vectors = tangents(points, weight = tangent_weight, closed = closed)
   assert tangent_vectors.shape == points.shape
   assert min_subdivisions >= 1
   assert max_segment_length is None or max_segment_length > 0.0
   assert max_degrees_per_knot is None or max_degrees_per_knot > 0.0

   seg_count = knot_count if closed else knot_count - 1
   seg_lengths = np.empty(seg_count)
   for seg in range(knot_count - 1):
      seg_lengths[seg] = vu.naive_length(points[seg + 1] - points[seg])
   if closed:
      seg_lengths[-1] = vu.naive_length(points[0] - points[-1])

   knot_insertions = np.full(seg_count, min_subdivisions - 1, dtype = int)
   if max_segment_length is not None:
      for seg in range(seg_count):
         sub_count = maths.ceil(seg_lengths[seg] / max_segment_length)
         if sub_count > knot_insertions[seg] + 1:
            knot_insertions[seg] = sub_count - 1
   if max_degrees_per_knot is not None:
      for seg in range(seg_count):
         next = seg + 1 if seg < knot_count - 1 else 0
         segment = points[next] - points[seg]
         turn = (vu.degrees_difference(tangent_vectors[seg], segment) +
                 vu.degrees_difference(segment, tangent_vectors[next]))
         log.debug(f'spline segment {seg}; turn: {turn:4.2f}')
         sub_count = maths.ceil(turn / max_degrees_per_knot)
         if sub_count > knot_insertions[seg] + 1:
            knot_insertions[seg] = sub_count - 1
   insertion_count = np.sum(knot_insertions)
   log.debug(f'{insertion_count} knot insertions for spline')
   spline_knot_count = knot_count + insertion_count

   spline_points = np.empty((spline_knot_count, 3))
   sk = 0
   for knot in range(seg_count):
      next = knot + 1 if knot < knot_count - 1 else 0
      spline_points[sk] = points[knot]
      sk += 1
      for i in range(knot_insertions[knot]):
         t = float(i + 1) / float(knot_insertions[knot] + 1)
         t2 = t * t
         t3 = t * t2
         p = np.zeros(3)
         p += (2.0 * t3 - 3.0 * t2 + 1.0) * points[knot]
         p += (t3 - 2.0 * t2 + t) * tangent_vectors[knot] * seg_lengths[knot]
         p += (-2.0 * t3 + 3.0 * t2) * points[next]
         p += (t3 - t2) * tangent_vectors[next] * seg_lengths[knot]
         spline_points[sk] = p
         sk += 1
   if not closed:
      spline_points[sk] = points[-1]
      sk += 1
   assert sk == spline_knot_count

   return spline_points
