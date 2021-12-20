"""_common.py: Resqml polyline common functions module."""

version = '23rd November 2021'

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
from resqpy.olio.base import BaseResqpy


class _BasePolyline(BaseResqpy):
    """Base class to implement shared methods for other classes in this module."""

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
                interp = rqo.GeobodyBoundaryInterpretation(self.model,
                                                           genetic_boundary_feature = feature,
                                                           domain = 'depth')
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


def shift_polyline(parent_model, poly_root, xyz_shift = (0, 0, 0), title = ''):
    """Returns a new polyline object, shifted by given coordinates."""

    from resqpy.lines._polyline import Polyline

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

    from resqpy.lines._polyline import Polyline

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

    assert points.ndim == 2 and points.shape[1] == 3
    assert weight in ['linear', 'square', 'cube']

    knot_count = len(points)
    assert knot_count > 1
    tangent_vectors = np.empty((knot_count, 3))

    for knot in range(1, knot_count - 1):
        tangent_vectors[knot] = _one_tangent(points, knot - 1, knot, knot + 1, weight)
    if closed:
        assert knot_count > 2, 'closed poly line must contain at least 3 knots for tangent generation'
        tangent_vectors[0] = _one_tangent(points, -1, 0, 1, weight)
        tangent_vectors[-1] = _one_tangent(points, -2, -1, 0, weight)
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
    _prepare_knot_insertions(knot_insertions, max_segment_length, seg_count, seg_lengths, max_degrees_per_knot,
                             knot_count, tangent_vectors, points)
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


def _one_tangent(points, k1, k2, k3, weight):
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


def _prepare_knot_insertions(knot_insertions, max_segment_length, seg_count, seg_lengths, max_degrees_per_knot,
                             knot_count, tangent_vectors, points):
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
