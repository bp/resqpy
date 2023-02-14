"""Utilities for working with 3D vectors in cartesian space.

note: some of these functions are redundant as they are provided by built-in numpy operations.

a vector is a one dimensional numpy array with 3 elements: x, y, z.
some functions accept a tuple or list of 3 elements as an alternative to a numpy array.
"""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np
import numba  # type: ignore
from numba import njit  # type: ignore
from typing import Tuple, Optional


def radians_from_degrees(deg):
    """Converts angle from degrees to radians."""
    return np.radians(deg)


def degrees_from_radians(rad):
    """Converts angle from radians to degrees."""
    return np.degrees(rad)


def zero_vector():
    """Returns a zero vector [0.0, 0.0, 0.0]."""
    return np.zeros(3)


def v_3d(v):
    """Returns a 3D vector for a 2D or 3D vector."""
    assert 2 <= len(v) <= 3
    if len(v) == 3:
        return v
    v3 = np.zeros(3)
    v3[:2] = v
    return v3


def add(a, b):  # note: could just use numpy a + b facility
    """Returns vector sum a+b."""
    a = np.array(a)
    b = np.array(b)
    assert a.size == b.size
    return a + b


def subtract(a, b):  # note: could just use numpy a - b facility
    """Returns vector difference a-b."""
    a = np.array(a)
    b = np.array(b)
    assert a.size == b.size
    return a - b


def elemental_multiply(a, b):  # note: could just use numpy a * b facility
    """Returns vector with products of corresponding elements of a and b."""
    a = np.array(a)
    b = np.array(b)
    assert a.size == b.size
    return a * b


def amplify(v, scaling):  # note: could just use numpy a * scalar facility
    """Returns vector with direction of v, amplified by scaling."""
    v = np.array(v)
    return scaling * v


def unit_vector(v):
    """Returns vector with same direction as v but with unit length."""
    assert 2 <= len(v) <= 3
    v = np.array(v, dtype = float)
    if np.all(v == 0.0):
        return v
    return v / maths.sqrt(np.sum(v * v))


def unit_vectors(v):
    """Returns vectors with same direction as those in v but with unit length."""
    scaling = np.sqrt(np.sum(v * v, axis = -1))
    zero_mask = np.zeros(v.shape, dtype = bool)
    zero_mask[np.where(scaling == 0.0), :] = True
    restore = np.seterr(all = 'ignore')
    result = np.where(zero_mask, 0.0, v / np.expand_dims(scaling, -1))
    np.seterr(**restore)
    return result


def nan_unit_vectors(v):
    """Returns vectors with same direction as those in v but with unit length, allowing NaNs."""
    assert v.shape[-1] == 3
    vf = v.reshape((-1, 3))
    nan_mask = np.isnan(vf)
    restore = np.seterr(all = 'ignore')
    scaling = np.sqrt(np.sum(vf * vf, axis = -1))
    zero_mask = np.zeros(vf.shape, dtype = bool)
    zero_mask[np.where(scaling == 0.0), :] = True
    result = np.where(zero_mask, 0.0, vf / np.expand_dims(scaling, -1))
    result = np.where(nan_mask, np.nan, result)
    np.seterr(**restore)
    return result.reshape(v.shape)


def unit_vector_from_azimuth(azimuth):
    """Returns horizontal unit vector in compass bearing given by azimuth (x = East, y = North)."""
    azimuth = azimuth % 360.0
    azimuth_radians = radians_from_degrees(azimuth)
    result = zero_vector()
    result[0] = maths.sin(azimuth_radians)  # x (increasing to east)
    result[1] = maths.cos(azimuth_radians)  # y (increasing to north)
    return result  # leave z as zero


def azimuth(v):  # 'azimuth' is synonymous with 'compass bearing'
    """Returns the compass bearing in degrees of the direction of v (x = East, y = North), ignoring z."""
    assert 2 <= v.size <= 3
    z_zero_v = np.zeros(3)
    z_zero_v[:2] = v[:2]
    unit_v = unit_vector(z_zero_v)  # also checks that z_zero_v is not zero vector
    x = unit_v[0]
    y = unit_v[1]  # ignore z component
    if x == 0.0 and y == 0.0:
        return 0.0  # arbitrary azimuth of a vertical vector
    if abs(x) >= abs(y):
        radians = maths.pi / 2.0 - maths.atan(y / x)
        if x < 0.0:
            radians += maths.pi
    else:
        radians = maths.atan(x / y)
        if y < 0.0:
            radians += maths.pi
    if radians < 0.0:
        radians += 2.0 * maths.pi
    return degrees_from_radians(radians)


def azimuths(va):  # 'azimuth' is synonymous with 'compass bearing'
    """Returns the compass bearings in degrees of the direction of each vector in va (x = East, y = North), ignoring z."""
    assert va.ndim > 1 and 2 <= va.shape[-1] <= 3
    shape = tuple(list(va.shape[:-1]) + [3])
    z_zero_v = np.zeros(shape)
    z_zero_v[..., :2] = va[..., :2]
    unit_v = unit_vectors(z_zero_v)  # also checks that z_zero_v is not zero vector
    x = unit_v[..., 0]
    y = unit_v[..., 1]  # ignore z component
    # todo: handle cases where x == y == 0
    restore = np.seterr(all = 'ignore')
    radians = np.where(
        np.abs(x) >= np.abs(y),
        np.where(x < 0.0, maths.pi * 3.0 / 2.0 - np.arctan(y / x), maths.pi / 2.0 - np.arctan(y / x)),
        np.where(y < 0.0, maths.pi + np.arctan(x / y), np.arctan(x / y)))
    radians = radians % (2.0 * maths.pi)
    np.seterr(**restore)
    return np.degrees(radians)


def inclination(v):
    """Returns the inclination in degrees of v (angle relative to +ve z axis)."""
    assert len(v) == 3
    unit_v = unit_vector(v)
    radians = maths.acos(unit_v[2])
    return degrees_from_radians(radians)


def inclinations(a):
    """Returns the inclination in degrees of each vector in a (angle relative to +ve z axis)."""
    assert a.ndim > 1 and a.shape[-1] == 3
    unit_vs = unit_vectors(a)
    radians = np.arccos(unit_vs[..., 2])
    return degrees_from_radians(radians)


def nan_inclinations(a, already_unit_vectors = False):
    """Returns the inclination in degrees of each vector in a (angle relative to +ve z axis), allowing NaNs."""
    assert a.ndim > 1 and a.shape[-1] == 3
    unit_vs = a if already_unit_vectors else nan_unit_vectors(a)
    restore = np.seterr(all = 'ignore')
    radians = np.where(np.isnan(unit_vs[..., 2]), np.nan, np.arccos(unit_vs[..., 2]))
    result = np.where(np.isnan(radians), np.nan, degrees_from_radians(radians))
    np.seterr(**restore)
    return result


def points_direction_vector(a, axis):
    """Returns an average direction vector based on first and last non-NaN points or slices in given axis."""

    assert a.ndim > 1 and 0 <= axis < a.ndim - 1 and a.shape[-1] > 1 and a.shape[axis] > 1
    if np.all(np.isnan(a)):
        return None
    start = 0
    start_slicing = [slice(None)] * a.ndim
    while True:
        start_slicing[axis] = slice(start)
        if not np.all(np.isnan(a[tuple(start_slicing)])):
            break
        start += 1
    finish = a.shape[axis] - 1
    finish_slicing = [slice(None)] * a.ndim
    while True:
        finish_slicing[axis] = slice(finish)
        if not np.all(np.isnan(a[tuple(finish_slicing)])):
            break
        finish += 1
    if start >= finish:
        return None
    if a.ndim > 2:
        mean_axes = tuple(range(a.ndim - 1))
        start_p = np.nanmean(a[tuple(start_slicing)], axis = mean_axes)
        finish_p = np.nanmean(a[tuple(finish_slicing)], axis = mean_axes)
    else:
        start_p = a[start]
        finish_p = a[finish]

    return finish_p - start_p


def dot_product(a, b):
    """Returns the dot product (scalar product) of the two vectors."""
    return np.dot(a, b)


def dot_products(a, b):
    """Returns the dot products of pairs of vectors; last axis covers element of a vector."""
    return np.sum(a * b, axis = -1)


def cross_product(a, b):
    """Returns the cross product (vector product) of the two vectors."""
    return np.cross(a, b)


def naive_length(v):
    """Returns the length of the vector assuming consistent units."""
    return maths.sqrt(dot_product(v, v))


def naive_lengths(v):
    """Returns the lengths of the vectors assuming consistent units."""
    return np.sqrt(np.sum(v * v, axis = -1))


def naive_2d_length(v):
    """Returns the length of the vector projected onto xy plane, assuming consistent units."""
    return maths.sqrt(dot_product(v[0:2], v[0:2]))


def naive_2d_lengths(v):
    """Returns the lengths of the vectors projected onto xy plane, assuming consistent units."""
    v2d = v[..., :2]
    return np.sqrt(np.sum(v2d * v2d, axis = -1))


def unit_corrected_length(v, unit_conversion):
    """Returns the length of the vector v after applying the unit_conversion factors."""
    # unit_conversion might be [1.0, 1.0, 0.3048] to convert z from feet to metres, for example
    # or [3.28084, 3.28084, 1.0] to convert x and y from metres to feet
    converted = elemental_multiply(v, unit_conversion)
    return naive_length(converted)


def manhatten_distance(p1, p2):
    """Returns the Manhattan distance between two points."""
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + abs(p2[2] - p1[2])


def manhattan_distance(p1, p2):  #  alternative spelling to above
    """Returns the Manhattan distance between two points."""
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + abs(p2[2] - p1[2])


def radians_difference(a, b):
    """Returns the angle between two vectors, in radians."""

    return maths.acos(min(1.0, max(-1.0, dot_product(unit_vector(a), unit_vector(b)))))


def degrees_difference(a, b):
    """Returns the angle between two vectors, in degrees."""

    return degrees_from_radians(radians_difference(a, b))


def rotation_matrix_3d_axial(axis, angle):
    """Returns a rotation matrix which will rotate points about axis (0: x, 1: y, or 2: z) by angle in degrees.

    note:
       this function follows the mathematical convention: a positive angle results in anti-clockwise rotation
       when viewed in direction of positive axis
    """

    axis_a = (axis + 1) % 3
    axis_b = (axis_a + 1) % 3
    matrix = np.zeros((3, 3))
    matrix[axis, axis] = 1.0
    radians = radians_from_degrees(angle)
    cosine = maths.cos(radians)
    sine = maths.sin(radians)
    matrix[axis_a, axis_a] = cosine
    matrix[axis_b, axis_b] = cosine
    matrix[axis_a, axis_b] = -sine  # left handed coordinate system, eg. UTM & depth
    matrix[axis_b, axis_a] = sine
    return matrix


def no_rotation_matrix():
    """Returns a rotation matrix which will not move points."""
    matrix = np.zeros((3, 3))
    for axis in range(3):
        matrix[axis, axis] = 1.0
    return matrix


def rotation_3d_matrix(xzy_axis_angles):
    """Returns a rotation matrix which will rotate points about 3 axes by angles in degrees."""

    # matrix = np.zeros((3, 3))
    # for axis in range(3):
    #     matrix[axis, axis] = 1.0
    # for axis in range(3):
    #     matrix = np.dot(matrix, rotation_matrix_3d_axial(axis, xzy_axis_angles[axis]))
    matrix = rotation_matrix_3d_axial(1, xzy_axis_angles[2])  # about y axis
    matrix = np.dot(matrix, rotation_matrix_3d_axial(2, xzy_axis_angles[1]))  # about z axis
    matrix = np.dot(matrix, rotation_matrix_3d_axial(0, xzy_axis_angles[0]))  # about x axis
    return matrix


def reverse_rotation_3d_matrix(xzy_axis_angles):
    """Returns a rotation matrix which will rotate back points about 3 axes by angles in degrees."""

    return rotation_3d_matrix(xzy_axis_angles).T


def rotate_vector(rotation_matrix, vector):
    """Returns the rotated vector."""

    return np.dot(rotation_matrix, vector)


def rotate_array(rotation_matrix, a):
    """Returns a copy of array a with each vector rotated by the rotation matrix."""

    s = a.shape
    return np.matmul(rotation_matrix, a.reshape(-1, 3).T).T.reshape(s)


def rotate_xyz_array_around_z_axis(a, target_xy_vector):
    """Returns a copy of array a suitable for presenting a cross-section using the resulting x,z values.

    arguments:
       a (numpy float array of shape (..., 3)): the xyz points to be rotated
       target_xy_vector (2 (or 3) floats): a vector indicating which direction in source xy space will end up
          being mapped to the positive x axis in the returned data

    returns:
       numpy float array of same shape as a

    notes:
       if the input points of a lie in a vertical plane parallel to the target xy vector, then the resulting
       points will have constant y values; in general, a full rotation of the points is applied, so resulting
       y values will indicate distance 'into the page' for non-planar or unaligned data
    """

    target_v = np.zeros(3)
    target_v[:2] = target_xy_vector[:2]
    rotation_angle = azimuth(target_v) - 90.0
    rotation_matrix = rotation_matrix_3d_axial(2, rotation_angle)  # todo: check sign of rotation angle
    return rotate_array(rotation_matrix, a)


def unit_vector_from_azimuth_and_inclination(azimuth, inclination):
    """Returns unit vector with compass bearing of azimuth and inclination off +z axis.

    note:
       assumes a left handed coordinate system with y axis north and x axis east
    """

    matrix = rotation_3d_matrix((inclination, 180.0 - azimuth, 0.0))
    return rotate_vector(matrix, np.array((0.0, 0.0, 1.0)))


def tilt_3d_matrix(azimuth, dip):
    """Returns a 3D rotation matrix for applying a dip in a certain azimuth.

    note:
       if azimuth is compass bearing in degrees, and dip is in degrees, the resulting matrix can be used
       to rotate xyz points where x values are eastings, y values are northings and z increases downwards
    """

    matrix = rotation_matrix_3d_axial(2, -azimuth)  # will yield rotation around z axis so azimuth goes north
    matrix = np.dot(matrix, rotation_matrix_3d_axial(0, dip))  # adjust for dip
    matrix = np.dot(matrix, rotation_matrix_3d_axial(2, azimuth))  # rotate azimuth back to original
    return matrix


def rotation_matrix_3d_vector(v):
    """Returns a rotation matrix which will rotate points by inclination and azimuth of vector.

    note:
       the returned matrix will map a positive z axis vector onto v
    """

    m = tilt_3d_matrix(azimuth(v), inclination(v))
    m[:2, :] = -m[:2, :]  # todo: should this change be in the tilt matrix?
    return m


def tilt_points(pivot_xyz, azimuth, dip, points):
    """Modifies array of xyz points in situ to apply dip in direction of azimuth, about pivot point."""

    matrix = tilt_3d_matrix(azimuth, dip)
    points_shape = points.shape
    points[:] -= pivot_xyz
    points[:] = np.matmul(matrix, points.reshape((-1, 3)).transpose()).transpose().reshape(points_shape)
    points[:] += pivot_xyz


def project_points_onto_plane(plane_xyz, normal_vector, points):
    """Modifies array of xyz points in situ to project onto a plane defined by a point and normal vector.

    note:
       implicit xy & z units must be the same
    """

    az = azimuth(normal_vector)
    incl = inclination(normal_vector)
    tilt_points(plane_xyz, az, -incl, points)
    points[..., 2] = plane_xyz[2]
    tilt_points(plane_xyz, az, incl, points)


def perspective_vector(xyz_box, view_axis, vanishing_distance, vector):
    """Returns a version of vector with a perspective applied."""

    mid_points = np.zeros(3)
    xyz_ranges = np.zeros(3)
    result = np.zeros(3)
    for axis in range(3):
        mid_points[axis] = 0.5 * (xyz_box[0, axis] + xyz_box[1, axis])
        xyz_ranges[axis] = xyz_box[1, axis] - xyz_box[0, axis]
    factor = 1.0 - (vector[view_axis] - xyz_box[0, view_axis]) / (vanishing_distance * (xyz_ranges[view_axis]))
    result[view_axis] = vector[view_axis]
    for axis in range(3):
        if axis == view_axis:
            continue
        result[axis] = mid_points[axis] + factor * (vector[axis] - mid_points[axis])
    return result


def determinant(a, b, c):
    """Returns the determinant of the 3 x 3 matrix comprised of the 3 vectors."""

    return (a[0] * b[1] * c[2] + a[1] * b[2] * c[0] + a[2] * b[0] * c[1] - a[2] * b[1] * c[0] - a[1] * b[0] * c[2] -
            a[0] * b[2] * c[1])


def determinant_3x3(a):
    """Returns the determinant of the 3 x 3 matrix."""

    return determinant(a[0], a[1], a[2])


def clockwise(a, b, c):
    """Returns a +ve value if 2D points a,b,c are in clockwise order, 0.0 if in line, -ve for ccw.

    note:
       assumes positive y-axis is anticlockwise from positive x-axis
    """

    return (c[0] - a[0]) * (b[1] - a[1]) - ((c[1] - a[1]) * (b[0] - a[0]))


def clockwise_triangles(p, t, projection = 'xy'):
    """Returns a numpy array of +ve values where triangle points are in clockwise order, 0.0 if in line, -ve for ccw.

    arguments:
       p (numpy float array of shape (N, 2 or 3)): points in use as vertices of triangles
       t (numpy int array of shape (M, 3)): indices into first axis of p defining the triangles
       projection (string, default 'xy'): one of 'xy', 'xz' or 'yz' being the direction of projection,
          ie. which elements of the second axis of p to use; must be 'xy' if p has shape (N, 2)

    returns:
       numpy float array of shape (M,) being the +ve or -ve values indicating clockwise or anti-clockwise
       ordering of each triangle's vertices when projected onto the specified plane and viewed in the
       direction negative to positive of the omitted axis

    note:
       assumes xyz axes are left-handed (reverse the result for a right handed system)
    """

    a0, a1 = _projected_xyz_axes(projection)

    return (((p[t[:, 2], a0] - p[t[:, 0], a0]) * (p[t[:, 1], a1] - p[t[:, 0], a1])) -
            ((p[t[:, 2], a1] - p[t[:, 0], a1]) * (p[t[:, 1], a0] - p[t[:, 0], a0])))


def in_triangle(a, b, c, d):
    """Returns True if point d lies wholly within the triangle pf ccw points a, b, c, projected onto xy plane.

    note:
       a, b & c must be sorted into anti-clockwise order before calling this function
    """

    return clockwise(a, b, d) < 0.0 and clockwise(b, c, d) < 0.0 and clockwise(c, a, d) < 0.0


def in_triangle_edged(a, b, c, d):
    """Returns True if d lies within or on the boudnary of triangle of ccw points a,b,c projected onto xy plane.

    note:
       a, b & c must be sorted into anti-clockwise order before calling this function
    """

    return clockwise(a, b, d) <= 0.0 and clockwise(b, c, d) <= 0.0 and clockwise(c, a, d) <= 0.0


def points_in_triangles(p, t, da, projection = 'xy', edged = False):
    """Returns 2D numpy bool array indicating which of points da are within which triangles.

    arguments:
       p (numpy float array of shape (N, 2 or 3)): points in use as vertices of triangles
       t (numpy int array of shape (M, 3)): indices into first axis of p defining the triangles
       da (numpy float array of shape (D, 2 or 3)): points to test for
       projection (string, default 'xy'): one of 'xy', 'xz' or 'yz' being the direction of projection,
          ie. which elements of the second axis of p  and da to use; must be 'xy' if p and da have shape (N, 2)
       edged (bool, default False): if True, points lying exactly on the edge of a triangle are included
          as being in the triangle, otherwise they are excluded

    returns:
       numpy bool array of shape (M, D) indicating which points are within which triangles

    note:
       the triangles do not need to be in a consistent clockwise or anti-clockwise order
    """

    assert p.ndim == 2 and t.ndim == 2 and da.ndim == 2 and da.shape[1] == p.shape[1]
    cwt = clockwise_triangles(p, t, projection = projection)
    a0, a1 = _projected_xyz_axes(projection)
    d_count = len(da)
    d_base = len(p)
    t_count = len(t)
    pp = np.concatenate((p, da), axis = 0)
    # build little triangles using da points and two of triangle vertices
    tp = np.empty((t_count, d_count, 3, 3), dtype = int)
    tp[:, :, :, 2] = np.arange(d_count).reshape((1, -1, 1)) + d_base
    tp[:, :, 0, 0] = np.where(cwt > 0.0, t[:, 1], t[:, 0]).reshape((-1, 1))
    tp[:, :, 0, 1] = np.where(cwt > 0.0, t[:, 0], t[:, 1]).reshape((-1, 1))
    tp[:, :, 1, 0] = np.where(cwt > 0.0, t[:, 2], t[:, 1]).reshape((-1, 1))
    tp[:, :, 1, 1] = np.where(cwt > 0.0, t[:, 1], t[:, 2]).reshape((-1, 1))
    tp[:, :, 2, 0] = np.where(cwt > 0.0, t[:, 0], t[:, 2]).reshape((-1, 1))
    tp[:, :, 2, 1] = np.where(cwt > 0.0, t[:, 2], t[:, 0]).reshape((-1, 1))
    cwtd = clockwise_triangles(pp, tp.reshape((-1, 3)), projection = projection).reshape((t_count, d_count, 3))
    if edged:
        return np.all(cwtd <= 0.0, axis = 2)
    else:
        return np.all(cwtd < 0.0, axis = 2)


@njit
def point_in_polygon(x, y, polygon):
    """Calculates if a point in within a polygon in 2D.
    
    arguments:
        x (float): the point's x-coordinate.
        y (float): the point's y-coordinate.
        polygon (np.ndarray): array of the polygon's vertices in 2D.
    
    returns:
        inside (bool): True if point is within the polygon, False otherwise.

    note:
        the polygon is assumed closed, the closing point should not be repeated
    """
    polygon_vertices = len(polygon)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in numba.prange(polygon_vertices + 1):
        p2x, p2y = polygon[i % polygon_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@njit
def point_in_triangle(x, y, triangle):
    """Calculates if a point in within a triangle in 2D.

    arguments:
        x (float): the point's x-coordinate.
        y (float): the point's y-coordinate.
        triangle (np.ndarray): array of the triangles's vertices in 2D, of shape (3, 2)

    returns:
        inside (bool): True if point is within the polygon, False otherwise.
    """
    p0x = triangle[0, 0]
    p1x = triangle[1, 0]
    p2x = triangle[2, 0]
    min_p01x = min(p0x, p1x)
    if x < min(min_p01x, p2x):
        return False
    max_p01x = max(p0x, p1x)
    if x > max(max_p01x, p2x):
        return False
    p0y = triangle[0, 1]
    p1y = triangle[1, 1]
    p2y = triangle[2, 1]
    min_p01y = min(p0y, p1y)
    if y < min(min_p01y, p2y):
        return False
    max_p01y = max(p0y, p1y)
    if y > max(max_p01y, p2y):
        return False
    inside = False
    xints = 0.0
    if y > min_p01y:
        if y <= max_p01y:
            if x <= max_p01x:
                if p0x == p1x:
                    inside = True
                else:
                    if p0y == p1y:
                        inside = True
                    else:
                        xints = (y - p0y) * (p1x - p0x) / (p1y - p0y) + p0x
                        if x <= xints:
                            inside = True
    if y > min(p1y, p2y):
        if y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1x == p2x:
                    inside = not inside
                else:
                    if p1y == p2y:
                        inside = not inside
                    else:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if x <= xints:
                            inside = not inside
    if y > min(p2y, p0y):
        if y <= max(p2y, p0y):
            if x <= max(p2x, p0x):
                if p2x == p0x:
                    inside = not inside
                else:
                    if p2y == p0y:
                        inside = not inside
                    else:
                        xints = (y - p2y) * (p0x - p2x) / (p0y - p2y) + p2x
                        if x <= xints:
                            inside = not inside
    return inside


@njit
def points_in_polygon(points: np.ndarray, polygon: np.ndarray, points_xlen: int, polygon_num: int = 0) -> np.ndarray:
    """Calculates which points are within a polygon in 2D.

    arguments:
        points (np.ndarray): array of shape (N, 2 or 3), of the points in 2D (xy, any z values are ignored)
        polygon (np.ndarray): list-like array of the polygon's vertices in 2D
        points_xlen (int): the original I extent of the now flattened points, use 1 if not applicable
        polygon_num (int): the polygon number, default is 0, for copying to output

    returns:
        polygon_points (np.ndarray): list-like 2D array containing only the points within the polygon,
            with each row being the polygon number (as input), points J index, and points I index

    note:
        the polygon is assumed closed, the closing point should not be repeated
    """
    polygon_points = np.empty((0, 3), dtype = numba.int32)
    for point_num in numba.prange(len(points)):
        p = point_in_polygon(points[point_num, 0], points[point_num, 1], polygon)
        if p is True:
            j, i = divmod(point_num, points_xlen)
            polygon_points = np.append(polygon_points, np.array([[polygon_num, j, i]], dtype = numba.int32), axis = 0)

    return polygon_points


@njit
def points_in_triangle(points: np.ndarray, triangle: np.ndarray, points_xlen: int, triangle_num: int = 0) -> np.ndarray:
    """Calculates which points are within a triangle in 2D.

    arguments:
        points (np.ndarray): array of the points in 2D.
        triangle (np.ndarray): array of the triangle's vertices in 2D, shape (3, 2).
        points_xlen (int): the number of unique x coordinates.
        triangle_num (int): the triangle number, default is 0.

    returns:
        triangle_points (np.ndarray): 2D array containing only the points within the triangle,
            with each row being the triangle number, points y index, and points x index.
    """
    triangle_points = np.empty((0, 3), dtype = numba.int32)
    for point_num in numba.prange(len(points)):
        p = point_in_triangle(points[point_num, 0], points[point_num, 1], triangle)
        if p is True:
            yi, xi = divmod(point_num, points_xlen)
            triangle_points = np.append(triangle_points,
                                        np.array([[triangle_num, yi, xi]], dtype = numba.int32),
                                        axis = 0)

    return triangle_points


@njit
def mesh_points_in_triangle(triangle: np.ndarray,
                            points_xlen: int,
                            points_ylen: int,
                            triangle_num: int = 0) -> np.ndarray:
    """Calculates which implicit mesh points are within a triangle in 2D for normalised triangle.

    arguments:
        triangle (np.ndarray): array of the triangle's vertices in 2D, shape (3, 2).
        points_xlen (int): the number of unique x coordinates, starting at 0.0, spacing 1.0.
        points_ylen (int): the number of unique y coordinates, starting at 0.0, spacing 1.0.
        triangle_num (int): the triangle number, default is 0.

    returns:
        triangle_points (np.ndarray): 2D array containing only the points within the triangle,
            with each row being the triangle number, points y index, and points x index.
    """
    triangle_points = np.empty((0, 3), dtype = numba.int32)
    y = 0.0
    for yi in numba.prange(points_ylen):
        x = 0.0
        for xi in numba.prange(points_xlen):
            p = point_in_triangle(x, y, triangle)
            if p is True:
                triangle_points = np.append(triangle_points,
                                            np.array([[triangle_num, yi, xi]], dtype = numba.int32),
                                            axis = 0)
            x += 1.0
        y += 1.0
    return triangle_points


@njit
def points_in_polygons(points: np.ndarray, polygons: np.ndarray, points_xlen: int) -> np.ndarray:
    """Calculates which points are within which polygons in 2D.

    arguments:
        points (np.ndarray): array of the points in 2D.
        polygons (np.ndarray): array of each polygons' vertices in 2D.
        points_xlen (int): the number of unique x coordinates.

    returns:
        polygons_points (np.ndarray): 2D array (list-like) containing only the points within each polygon,
            with each row being the polygon number, points y index, and points x index.
    """
    polygons_points = np.empty((0, 3), dtype = numba.int32)
    for polygon_num in numba.prange(len(polygons)):
        polygon_points = points_in_polygon(points, polygons[polygon_num], points_xlen, polygon_num)
        polygons_points = np.append(polygons_points, polygon_points, axis = 0)

    return polygons_points


@njit
def points_in_triangles_njit(points: np.ndarray, triangles: np.ndarray, points_xlen: int) -> np.ndarray:
    """Calculates which points are within which triangles in 2D.

    arguments:
        points (np.ndarray): array of the points in 2D.
        triangles (np.ndarray): array of each triangles' vertices in 2D, shape (N, 3, 2).
        points_xlen (int): the number of unique x coordinates.

    returns:
        triangles_points (np.ndarray): 2D array (list-like) containing only the points within each triangle,
            with each row being the triangle number, points y index, and points x index.
    """
    triangles_points = np.empty((0, 3), dtype = numba.int32)
    for triangle_num in numba.prange(len(triangles)):
        triangle_points = points_in_triangle(points, triangles[triangle_num], points_xlen, triangle_num)
        triangles_points = np.append(triangles_points, triangle_points, axis = 0)

    return triangles_points


@njit
def meshgrid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns coordinate matrices from coordinate vectors x and y.

    arguments:
        x (np.ndarray): 1d array of x coordinates.
        y (np.ndarray): 1d array of y coordinates.

    returns:
        Tuple containing:

            - xx (np.ndarray): the elements of x repeated to fill the matrix along the first dimension.
            - yy (np.ndarray): the elements of y repeated to fill the matrix along the second dimension.
    """
    xx = np.empty(shape = (y.size, x.size), dtype = x.dtype)
    yy = np.empty(shape = (y.size, x.size), dtype = y.dtype)
    for i in range(y.size):
        for j in range(x.size):
            xx[i, j] = x[j]
            yy[i, j] = y[i]

    return xx, yy


def points_in_triangles_aligned(nx: int, ny: int, dx: float, dy: float, triangles: np.ndarray) -> np.ndarray:
    """Calculates which points are within which triangles in 2D for a regular mesh of aligned points.

    arguments:
        nx (int): number of points in x axis
        ny (int): number of points in y axis
        dx (float): spacing of points in x axis (first point is at half dx)
        dy (float): spacing of points in y axis (first point is at half dy)
        triangles (np.ndarray): float array of each triangles' vertices in 2D, shape (N, 3, 2).
        points_xlen (int): the number of unique x coordinates.

    returns:
        triangles_points (np.ndarray): 2D array (list-like) containing only the points within each triangle,
            with each row being the triangle number, points y index, and points x index.
    """
    triangles_points = np.empty((0, 3), dtype = np.int32)
    dx_dy = np.expand_dims(np.array([dx, dy], dtype = np.float32), axis = 0)
    # for triangle_num in numba.prange(len(triangles)):
    for triangle_num in range(len(triangles)):
        tp = (triangles[triangle_num] / dx_dy) - 0.5
        min_tpx = max(maths.ceil(min(tp[0, 0], tp[1, 0], tp[2, 0])), 0)
        max_tpx = min(maths.floor(max(tp[0, 0], tp[1, 0], tp[2, 0])), nx - 1)
        if max_tpx < min_tpx:
            continue
        min_tpy = max(maths.ceil(min(tp[0, 1], tp[1, 1], tp[2, 1])), 0)
        max_tpy = min(maths.floor(max(tp[0, 1], tp[1, 1], tp[2, 1])), ny - 1)
        if max_tpy < min_tpy:
            continue
        ntpx = max_tpx - min_tpx + 1
        ntpy = max_tpy - min_tpy + 1
        # x = np.linspace(float(min_tpx), float(max_tpx), ntpx)
        # y = np.linspace(float(min_tpy), float(max_tpy), ntpy)
        # p = np.stack(meshgrid(x, y), axis = -1).reshape((-1, 2))
        # triangle_points = points_in_triangle(p, tp, ntpx, triangle_num)
        tp[:, 0] -= float(min_tpx)
        tp[:, 1] -= float(min_tpy)
        triangle_points = mesh_points_in_triangle(tp, ntpx, ntpy, triangle_num)
        triangle_points[:, 1] += min_tpy
        triangle_points[:, 2] += min_tpx
        triangles_points = np.append(triangles_points, triangle_points, axis = 0)

    return triangles_points


@njit
def triangle_box(triangle: np.ndarray) -> Tuple[float, float, float, float]:
    """Finds the minimum and maximum x and y values of a single traingle.

    arguments:
        triangle (np.ndarray): array of the traingle's vertices' x and y coordinates.

    returns:
        Tuple containing:

            - (float): minimum x value.
            - (float): maximum x value.
            - (float): minimum y value.
            - (float): maximum y value.
    """
    x_values = triangle[:, 0]
    y_values = triangle[:, 1]
    return min(x_values), max(x_values), min(y_values), max(y_values)


@njit
def vertical_intercept(x: float, x_values: np.ndarray, y_values: np.ndarray) -> Optional[float]:
    """Finds the y value of a straight line between two points at a given x.
    
    If the x value given is not within the x values of the points, returns None.

    arguments:
        x (float): x value at which to determine the y value.
        x_values (np.ndarray): the x coordinates of point 1 and point 2.
        y_values (np.ndarray): the y coordinates of point 1 and point 2.

    returns:
        y (Optional[float]): y value of the straight line between point 1 and point 2,
            evaluated at x. If x is outside the x_values range, y is None.
    """
    y = None
    if x >= min(x_values) and x <= max(x_values):
        if x_values[0] == x_values[1]:
            y = y_values[0]
        else:
            m = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
            c = y_values[1] - m * x_values[1]
            y = m * x + c
    return y


@njit
def points_in_triangles_aligned_optimised(nx: int, ny: int, dx: float, dy: float, triangles: np.ndarray) -> np.ndarray:
    """Calculates which points are within which triangles in 2D for a regular mesh of aligned points.

    arguments:
        nx (int): number of points in x axis
        ny (int): number of points in y axis
        dx (float): spacing of points in x axis (first point is at half dx)
        dy (float): spacing of points in y axis (first point is at half dy)
        triangles (np.ndarray): float array of each triangles' vertices in 2D, shape (N, 3, 2)

    returns:
        triangles_points (np.ndarray): 2D array (list-like) containing only the points within each triangle,
            with each row being the triangle number, points y index, and points x index
    """
    grid_x = dx * (0.5 + np.arange(nx).astype(np.float64))
    grid_y = dy * (0.5 + np.arange(ny).astype(np.float64))
    triangles_points_list = []
    for triangle_num in range(len(triangles)):
        triangle = triangles[triangle_num]
        min_x, max_x, min_y, max_y = triangle_box(triangle)
        x_values = grid_x[np.logical_and(grid_x >= min_x, grid_x < max_x)]
        y_values = grid_y[np.logical_and(grid_y >= min_y, grid_y < max_y)]
        for x in x_values:
            ys = []
            ys.append(vertical_intercept(x, triangle[1:, 0], triangle[1:, 1]))
            ys.append(vertical_intercept(x, triangle[:2, 0], triangle[:2, 1]))
            ys.append(vertical_intercept(x, triangle[::2, 0], triangle[::2, 1]))
            ys = [u for u in ys if u is not None]
            valid_y = y_values[np.logical_and(y_values >= min(ys), y_values <= max(ys))]
            x_idx = int(x / dx)
            triangles_points_list.extend([[triangle_num, int(y / dy), x_idx] for y in valid_y])

    if len(triangles_points_list) == 0:
        triangles_points = np.empty((0, 3), dtype = np.int32)
        return triangles_points

    triangles_points = np.array(triangles_points_list, dtype = np.int32)
    return triangles_points


def triangle_normal_vector(p3):
    """For a triangle in 3D space, defined by 3 vertex points, returns a unit vector normal to the plane of the triangle.

    note:
        resulting vector implicitly assumes that xy & z units are the same; if this is not the case, adjust vector
        afterwards as required
    """

    # todo: handle degenerate triangles
    return unit_vector(cross_product(p3[0] - p3[1], p3[0] - p3[2]))


@njit
def triangle_normal_vector_numba(points):
    """For a triangle in 3D space, defined by 3 vertex points, returns a unit vector normal to the plane of the triangle.

    note:
        resulting vector implicitly assumes that xy & z units are the same; if this is not the case, adjust vector
        afterwards as required
    """
    v = np.cross(points[0] - points[1], points[0] - points[2])
    return v / np.linalg.norm(v)


def in_circumcircle(a, b, c, d):
    """Returns True if point d lies within the circumcircle pf ccw points a, b, c, projected onto xy plane.

    note:
       a, b & c must be sorted into anti-clockwise order before calling this function
    """

    m = np.empty((3, 3))
    m[0, :2] = a[:2] - d[:2]
    m[1, :2] = b[:2] - d[:2]
    m[2, :2] = c[:2] - d[:2]
    m[:, 2] = (m[:, 0] * m[:, 0]) + (m[:, 1] * m[:, 1])
    return determinant_3x3(m) > 0.0


def point_distance_to_line_2d(p, l1, l2):
    """Ignoring any z values, returns the xy distance of point p from line passing through l1 and l2."""

    if np.all(l2[:2] == l1[:2]):
        return naive_2d_length(p[:2] - l1[:2])
    return (abs(p[0] * (l1[1] - l2[1]) + l1[0] * (l2[1] - p[1]) + l2[0] * (p[1] - l1[1])) /
            naive_2d_length(l2[:2] - l1[:2]))


def point_distance_to_line_segment_2d(p, l1, l2):
    """Ignoring any z values, returns the xy distance of point p from line segment between l1 and l2."""

    if is_obtuse_2d(l1, p, l2):
        return naive_2d_length(p[:2] - l1[:2])
    elif is_obtuse_2d(l2, p, l1):
        return naive_2d_length(p[:2] - l2[:2])
    else:
        return point_distance_to_line_2d(p, l1, l2)


def is_obtuse_2d(p, p1, p2):
    """Returns True if the angle at point p subtended by points p1 and p2, in xy plane, is greater than 90 degrees; else False."""

    return np.dot((p1[:2] - p[:2]), (p2[:2] - p[:2])) < 0.0


def isclose(a, b, tolerance = 1.0e-6):
    """Returns True if the two points are extremely close to one another (ie.

    the same point).
    """

    #   return np.all(np.isclose(a, b, atol = tolerance))
    # cheap and cheerful alternative to thorough numpy version commented out above
    return np.max(np.abs(a - b)) <= tolerance


def is_close(a, b, tolerance = 1.0e-6):
    """Returns True if the two points are extremely close to one another (ie.

    the same point).
    """

    return isclose(a, b, tolerance = tolerance)


def point_distance_sqr_to_points_projected(p, points, projection):
    """Returns an array of projected distances squared between p and points; projection is 'xy', 'xz' or 'yz'."""

    if projection == 'xy':
        d = points[..., :2] - p[:2]
    elif projection == 'xz':
        d = points[..., 0:3:2] - p[0:3:2]
    elif projection == 'yz':
        d = points[..., 1:] - p[1:]
    else:
        raise ValueError("projection must be 'xy', 'xz' or 'yz'")
    return np.sum(d * d, axis = -1)


def nearest_point_projected(p, points, projection):
    """Returns the index into points array closest to point p; projection is 'xy', 'xz' or 'yz'."""

    # note: in the case of equidistant points, the index of the 'first' point is returned

    d2 = point_distance_sqr_to_points_projected(p, points, projection)
    return np.unravel_index(np.nanargmin(d2), d2.shape)


def area_of_triangle(a, b, c):
    """Returns the area of the triangle defined by three vertices."""

    # uses Heron's formula
    la = naive_length(a - b)
    lb = naive_length(b - c)
    lc = naive_length(c - a)
    s = 0.5 * (la + lb + lc)
    return maths.sqrt(s * (s - la) * (s - lb) * (s - lc))


def area_of_triangles(p, t, xy_projection = False):
    """Returns numpy array of areas of triangles, optionally when projected onto xy plane."""

    # uses Heron's formula
    pt = p[t]
    if xy_projection:
        la = naive_2d_lengths(pt[:, 0, :] - pt[:, 1, :])
        lb = naive_2d_lengths(pt[:, 1, :] - pt[:, 2, :])
        lc = naive_2d_lengths(pt[:, 2, :] - pt[:, 0, :])
    else:
        la = naive_lengths(pt[:, 0, :] - pt[:, 1, :])
        lb = naive_lengths(pt[:, 1, :] - pt[:, 2, :])
        lc = naive_lengths(pt[:, 2, :] - pt[:, 0, :])
    s = 0.5 * (la + lb + lc)
    return np.sqrt(s * (s - la) * (s - lb) * (s - lc))


def clockwise_sorted_indices(p, b):
    """Returns a clockwise sorted numpy list of indices b into the points p.

    note:
       this function is designed for preparing a list of points defining a convex polygon when projected in
       the xy plane, starting from a subset of the unsorted points; more specifically, it assumes that the
       mean of p (over axis 0) lies within the polygon and the clockwise ordering is relative to that mean point
    """

    # note: this function currently assumes that the mean of points bp lies within the hull of bp
    # and that the points form a convex polygon from the perspective of the mean point
    assert p.ndim == 2 and len(p) >= 3
    centre = np.mean(p, axis = 0)
    hull_list = []  # list of azimuths and indices into p (axis 0)
    for i in b:
        azi = azimuth(p[i] - centre)
        hull_list.append((azi, i))
    return np.array([i for (_, i) in sorted(hull_list)], dtype = int)


def xy_sorted(p, axis = None):
    """Returns copy of points p sorted according to x or y (whichever has greater range).

    arguments:
        p (numpy float array of shape (..., 2) or (..., 3)): points to be sorted
        axis (int, optional): 0 for x sort; 1 for y sort; None for whichever has greater range

    returns:
        p', axis where p' is a list-like (2D) version of p, sorted by either x or y and axis is 0
        if the sort was by x, 1 if it were by y

    note:
       returned array is always 2D, ie. list of points
    """
    assert p.ndim >= 2 and p.shape[-1] >= 2
    p = p.reshape((-1, p.shape[-1]))
    xy_range = np.nanmax(p, axis = 0) - np.nanmin(p, axis = 0)
    if axis is None:
        axis = (1 if xy_range[1] > xy_range[0] else 0)
    spi = np.argsort(p[:, axis])
    return p[spi], axis


def _projected_xyz_axes(projection):
    assert projection in ['xy', 'xz', 'yz'], f'invalid projection {projection}'
    a0 = 'xyz'.index(projection[0])
    a1 = 'xyz'.index(projection[1])
    return a0, a1


# end of vector_utilities module
