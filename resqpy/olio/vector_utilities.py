# vector_utilities module
# note: many of these functions are redundant as they are provided by built-in numpy operations

version = '1st September 2021'

import logging

log = logging.getLogger(__name__)
log.debug('vector_utilities.py version %s', version)

# works with 3D vectors in a cartesian space
# a vector is a one dimensional numpy array with 3 elements: x, y, z

# functions defined here:
#    def radians_from_degrees(deg):
#    def degrees_from_radians(rad):
#    def zero_vector():
#    def add(a, b):                            # note: could just use numpy a + b facility
#    def subtract(a, b):                       # note: could just use numpy a - b facility
#    def elemental_multiply(a, b):             # note: could just use numpy a * b facility
#    def amplify(v, scaling):                  # note: could just use numpy a * scalar facility
#    def unit_vector(v):
#    def unit_vector_from_azimuth(azimuth):
#    def azimuth(v):                           # 'azimuth' is synonymous with 'compass bearing'
#    def dot_product(a, b):
#    def cross_product(a, b):
#    def naive_length(v):
#    def unit_corrected_length(v, unit_conversion):
#    def manhattan_distance(p1, p2):
#    def rotation_matrix_3d_axial(axis, angle):
#    def rotation_3d_matrix(xzy_axis_angles):
#    def rotate_vector(rotation_matrix, vector):
#    def perspective_vector(xyz_box, view_axis, vanishing_distance, vector):
#    def determinant(a, b, c):
#    def determinant_3x3(a):

import math as maths
import numpy as np


def radians_from_degrees(deg):
   """Converts angle from degrees to radians."""
   return maths.radians(deg)


def degrees_from_radians(rad):
   """Converts angle from radians to degrees."""
   return maths.degrees(rad)


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
   assert (a.size == b.size == 3)
   result = zero_vector()
   for i in range(3):
      result[i] = a[i] + b[i]
   return result


def subtract(a, b):  # note: could just use numpy a - b facility
   """Returns vector difference a-b."""
   assert (a.size == b.size == 3)
   result = zero_vector()
   for i in range(3):
      result[i] = a[i] - b[i]
   return result


def elemental_multiply(a, b):  # note: could just use numpy a * b facility
   """Returns vector with products of corresponding elements of a and b."""
   assert (a.size == b.size == 3)
   result = zero_vector()
   for i in range(3):
      result[i] = a[i] * b[i]
   return result


def amplify(v, scaling):  # note: could just use numpy a * scalar facility
   """Returns vector with direction of v, amplified by scaling."""
   assert (v.size == 3)
   result = zero_vector()
   for i in range(3):
      result[i] = scaling * v[i]
   return result


def unit_vector(v):
   """Returns vector with same direction as v but with unit length."""
   assert v.size == 3
   result = zero_vector()
   if np.all(v == result):
      return result  # NB: v is zero vector: could raise an exception
   scaling = 0.0
   for i in range(3):
      scaling += v[i] * v[i]
   scaling = maths.sqrt(scaling)
   for i in range(3):
      result[i] = v[i] / scaling
   return result


def unit_vectors(v):
   """Returns vectors with same direction as those in v but with unit length."""
   scaling = np.sqrt(np.sum(v * v, axis = -1))
   zero_mask = np.zeros(v.shape, dtype = bool)
   zero_mask[np.where(scaling == 0.0), :] = True
   np.seterr(divide = 'ignore')
   result = np.where(zero_mask, 0.0, v / np.expand_dims(scaling, -1))
   np.seterr(divide = 'warn')
   return result


def unit_vector_from_azimuth(azimuth):
   """Returns horizontal unit vector in compass bearing given by azimuth (x = East, y = North)."""
   assert (0.0 <= azimuth <= 360.0)
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


def inclination(v):
   """Returns the inclination in degrees of v (angle relative to +ve z axis)."""
   assert 2 <= v.size <= 3
   unit_v = unit_vector(v)
   radians = maths.acos(dot_product(unit_v, np.array((0.0, 0.0, 1.0))))
   return degrees_from_radians(radians)


def points_direction_vector(a, axis):
   """Returns an average direction vector based on first and last points or slices in given axis."""

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
#   log.debug(f'axis: {axis}; start: {start}; finish: {finish}')
   if start >= finish:
      return None
   if a.ndim > 2:
      mean_axes = tuple(range(a.ndim - 1))
      start_p = np.nanmean(a[tuple(start_slicing)], axis = mean_axes)
      finish_p = np.nanmean(a[tuple(finish_slicing)], axis = mean_axes)
   else:
      start_p = a[start]
      finish_p = a[finish]


#   log.debug(f'start_p: {start_p}')
#   log.debug(f'finish_p: {finish_p}')
   return finish_p - start_p


def dot_product(a, b):
   """Returns the dot product (scalar product) of the two vectors."""
   return np.dot(a, b)


#   assert(a.size == b.size)
#   result = 0.0
#   for i in range(a.size):
#      result += a[i] * b[i]
#   return result


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


def manhattan_distance(p1, p2):
   """Returns the Manhattan distance between two points."""
   return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + abs(p2[2] - p1[2])


def radians_difference(a, b):
   """Returns the angle between two vectors, in radians."""

   return maths.acos(min(1.0, max(-1.0, dot_product(unit_vector(a), unit_vector(b)))))


def degrees_difference(a, b):
   """Returns the angle between two vectors, in degrees."""

   return degrees_from_radians(radians_difference(a, b))


def rotation_matrix_3d_axial(axis, angle):
   """Retuns a rotation matrix which will rotate points about axis (0, 1, or 2) by angle in degrees."""

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
   matrix = np.zeros((3, 3))
   for axis in range(3):
      matrix[axis, axis] = 1.0
   for axis in range(3):
      matrix = np.dot(matrix, rotation_matrix_3d_axial(axis, xzy_axis_angles[axis]))
   return matrix


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

   #   log.debug('pivot xyz: ' + str(pivot_xyz))
   matrix = tilt_3d_matrix(azimuth, dip)
   points_shape = points.shape
   points[:] -= pivot_xyz
   #   log.debug('points shape: ' + str(points.shape))
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

   return (abs(p[0] * (l1[1] - l2[1]) + l1[0] * (l2[1] - p[1]) + l2[0] * (p[1] - l1[1])) / naive_2d_length(l2 - l1))


def isclose(a, b, tolerance = 1.0e-6):
   """Returns True if the two points are extremely close to one another (ie. the same point)."""

   #   return np.all(np.isclose(a, b, atol = tolerance))
   # cheap and cheerful alternative to thorough numpy version commented out above
   return np.max(np.abs(a - b)) <= tolerance


def is_close(a, b, tolerance = 1.0e-6):
   """Returns True if the two points are extremely close to one another (ie. the same point)."""

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


# end of vector_utilities module
