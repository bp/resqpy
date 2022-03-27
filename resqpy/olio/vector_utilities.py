""" Utilities for working with 3D vectors in cartesian space.

note: some of these functions are redundant as they are provided by built-in numpy operations.

a vector is a one dimensional numpy array with 3 elements: x, y, z.
some functions accept a tuple or list of 3 elements as an alternative to a numpy array.
"""

version = '8th March 2022'

import logging

log = logging.getLogger(__name__)

import math as maths

import numpy as np


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
    nan_mask = np.isnan(v)
    restore = np.seterr(all = 'ignore')
    scaling = np.sqrt(np.sum(v * v, axis = -1))
    zero_mask = np.zeros(v.shape, dtype = bool)
    zero_mask[np.where(scaling == 0.0), :] = True
    result = np.where(zero_mask, 0.0, v / np.expand_dims(scaling, -1))
    result = np.where(nan_mask, np.nan, result)
    np.seterr(**restore)
    return result


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


def nan_inclinations(a):
    """Returns the inclination in degrees of each vector in a (angle relative to +ve z axis), allowing NaNs."""
    assert a.ndim > 1 and a.shape[-1] == 3
    unit_vs = nan_unit_vectors(a)
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
    """Returns a rotation matrix which will rotate points about axis (0, 1, or 2) by angle in degrees."""

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


def rotation_3d_matrix(xzy_axis_angles):
    """Returns a rotation matrix which will rotate points about 3 axes by angles in degrees."""

    matrix = np.zeros((3, 3))
    for axis in range(3):
        matrix[axis, axis] = 1.0
    for axis in range(3):
        matrix = np.dot(matrix, rotation_matrix_3d_axial(axis, xzy_axis_angles[axis]))
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
    """Returns unit vector with compass bearing of azimuth and inclination off +z axis."""

    matrix = rotation_3d_matrix((inclination, azimuth, 0.0))
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


def tilt_points(pivot_xyz, azimuth, dip, points):
    """Modifies array of xyz points in situ to apply dip in direction of azimuth, about pivot point."""

    matrix = tilt_3d_matrix(azimuth, dip)
    points_shape = points.shape
    points[:] -= pivot_xyz
    points[:] = np.matmul(matrix, points.reshape((-1, 3)).transpose()).transpose().reshape(points_shape)
    points[:] += pivot_xyz


def project_points_onto_plane(plane_xyz, normal_vector, points):
    """Modifies array of xyz points in situ to project onto a plane defined by a point and normal vector."""

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


def triangle_normal_vector(p3):
    """For a triangle in 3D space, defined by 3 vertex points, returns a unit vector normal to the plane of the triangle."""

    # todo: handle degenerate triangles
    return unit_vector(cross_product(p3[0] - p3[1], p3[0] - p3[2]))


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


def _projected_xyz_axes(projection):
    assert projection in ['xy', 'xz', 'yz'], f'invalid projection {projection}'
    a0 = 'xyz'.index(projection[0])
    a1 = 'xyz'.index(projection[1])
    return a0, a1


# end of vector_utilities module
