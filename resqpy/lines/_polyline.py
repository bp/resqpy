"""_polyline.py: Resqml polyline module."""

version = '23rd November 2021'

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.olio.intersection as meet
from resqpy.lines._common import _BasePolyline, load_hdf5_array, tangents, spline
import resqpy.olio.point_inclusion as pip
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns


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
        self.rep_int_root = rep_int_root  # Optional represented interpretation xml root node
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         root_node = poly_root,
                         extra_metadata = extra_metadata)

        if self.root is None and all(i is not None for i in [set_bool, set_coord, set_crs, title]):
            # Using data from a polyline set
            assert set_coord.ndim > 1 and 2 <= set_coord.shape[-1] <= 3
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
        """XML node corresponding to self.crs_uuid."""

        return self.model.root_for_uuid(self.crs_uuid)

    @property
    def rep_int_uuid(self):
        """Returns the uuid of the represented interpretation."""

        # TODO: Track uuid only, not root
        return rqet.uuid_for_part_root(self.rep_int_root)

    @classmethod
    def from_scaled_polyline(cls, original, scaling, title = None, originator = None, extra_metadata = None):
        """Returns a scaled version of the original polyline.

        arguments:
           original (Polyline): the polyline from which the new polyline will be sporned
           scaling (float): the factor by which the original will be scaled
           title (str, optional): the citation title for the new polyline; inherited from
              original if None
           originator (str, optional): the name of the person creating the polyline; inherited
              from original if None
           extra_metadata (dict, optional): extra metadata for the new polyline; inherited from
              original if None

        returns:
           a new Polyline

        notes:
           the scaling factor is applied to vectors radiating from the balanced centre of the
           original polyline to its coordinates; a scaling of 1.0 will result in a copy of the original;
           if extra_metadata is not None, no extra metadata is inherited from original
        """

        if extra_metadata is None:
            extra_metadata = original.extra_metadata

        polyline = cls(original.model,
                       set_crs = original.crs_uuid,
                       set_bool = original.isclosed,
                       title = title if title else original.title,
                       originator = originator if originator else original.originator,
                       extra_metadata = extra_metadata)

        o_centre = original.balanced_centre()
        polyline.coordinates = scaling * (original.coordinates - o_centre) + o_centre
        polyline.nodepatch = (0, len(polyline.coordinates))

        return polyline

    def is_convex(self, trust_metadata = True):
        """Returns True if the polyline is closed and convex in the xy plane, otherwise False."""

        if not self.isclosed:
            return False
        if trust_metadata and self.extra_metadata is not None and 'is_convex' in self.extra_metadata.keys():
            return self.extra_metadata['is_convex'].lower() == 'true'
        nodes = len(self.coordinates)
        extras = {}
        if nodes < 3:
            result = False
        else:
            cw_bool = None
            result = True
            for node in range(nodes):
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
                extras['is_clockwise'] = str(cw_bool).lower()
        extras['is_convex'] = str(result).lower()
        self.append_extra_metadata(extras)
        return result

    def is_clockwise(self, trust_metadata = True):
        """Returns True if first non-straight triplet of nodes is clockwise in the xy plane; False if anti-clockwise.

        note:
           this method currently assumes that the xy axes are left-handed
        """

        if trust_metadata and self.extra_metadata is not None and 'is_clockwise' in self.extra_metadata.keys():
            return str(self.extra_metadata['is_clockwise']).lower() == 'true'
        result = None
        for node in range(3, len(self.coordinates)):
            cw = vu.clockwise(self.coordinates[node - 2], self.coordinates[node - 1], self.coordinates[node])
            if cw == 0.0:
                continue  # striaght line section
            result = (cw > 0.0)
            break
        if result is not None:
            self.append_extra_metadata({'is_clockwise': str(result).lower()})
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
        """Returns the naive length (ie.

        assuming x,y & z units are the same) of an individual segment of the polyline.
        """

        successor = self._successor(segment_index)
        d = 2 if in_xy else 3
        return vu.naive_length(self.coordinates[successor, :d] - self.coordinates[segment_index, :d])

    def segment_midpoint(self, segment_index):
        """Returns the midpoint of an individual segment of the polyline."""

        successor = self._successor(segment_index)
        return 0.5 * (self.coordinates[segment_index] + self.coordinates[successor])

    def segment_normal(self, segment_index):
        """For a closed polyline return a unit vector giving the 2D (xy) direction of an outward facing normal to a segment."""

        successor = self._successor(segment_index)
        segment_vector = self.coordinates[successor, :2] - self.coordinates[segment_index, :2]
        segment_vector = vu.unit_vector(segment_vector)
        normal_vector = np.zeros(3)
        normal_vector[0] = -segment_vector[1]
        normal_vector[1] = segment_vector[0]
        cw = self.is_clockwise()
        assert cw is not None, 'polyline is straight'
        if not cw:
            normal_vector = -normal_vector
        return normal_vector

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
            p1 = np.zeros(3)
            p2 = np.zeros(3)
            for seg_index in range(seg_count):
                successor = (seg_index + 1) % len(self.coordinates)
                p1[:d], p2[:d] = self.coordinates[seg_index, :d], self.coordinates[successor, :d]
                sum += (p1 + p2) * vu.naive_length(p2 - p1)
            centre = sum / (2.0 * self.full_length(in_xy = in_xy))
        if cache:
            self.centre = centre
        return centre

    def first_line_intersection(self, x1, y1, x2, y2, half_segment = False):
        """Finds the first intersection of (half) bounded line x,y 1 to 2 with polyline.

        returns:
           segment number & x, y of first intersection of (half) bounded line x,y 1 to 2 with polyline,
           or None, None, None if no intersection found
        """
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
        if mode == 'square':  # todo: check square mode – looks wrong
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
        assert self.is_convex(
        ), 'attempt to find denormalised x,y within a polyline that is not a closed convex polygon'
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

    def area(self):
        """Returns the area in the xy plane of a closed convex polygon."""
        assert self.isclosed
        assert self.is_convex()
        centre = np.mean(self.coordinates, axis = 0)
        a = 0.0
        for node in range(len(self.coordinates)):
            a += vu.area_of_triangle(centre, self.coordinates[node - 1], self.coordinates[node])
        return a

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

    def _successor(self, segment_index):
        """Returns segment_index + 1; or zero if segment_index is the last segment in a closed polyline."""

        successor = segment_index + 1
        if self.isclosed:
            assert 0 <= segment_index < len(self.coordinates)
            if segment_index == len(self.coordinates) - 1:
                successor = 0
        else:
            assert 0 <= segment_index < len(self.coordinates) - 1
        return successor
