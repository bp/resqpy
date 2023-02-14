"""intersection.py: functions to test whether lines intersect with planes."""

import logging

log = logging.getLogger(__name__)

import numpy as np
from numba import njit  # type: ignore
from typing import Union

import resqpy.olio.vector_utilities as vec


def line_plane_intersect(line_p, line_v, triangle):
    """Find the intersection of a line with a plane defined by a triangle.

    arguments:
       line_p (3 element numpy vector): a point on the line
       line_v (3 element numpy vector): vector being the direction of the line
       triangle ((3, 3) numpy array): three points on the plane (second index is xyz)

    returns:
       point (3 element numpy vector) of intersection of the line with the plane,
       or None if line is parallel to plane
    """

    p01 = triangle[1] - triangle[0]
    p02 = triangle[2] - triangle[0]

    norm = np.cross(p01, p02)  # normal to plane
    denom = np.dot(np.negative(line_v), norm)
    if denom == 0.0:
        return None  # line is parallel to plane
    t = np.dot(norm, line_p - triangle[0]) / denom
    return line_p + t * line_v


def line_triangle_intersect(line_p, line_v, triangle, line_segment = False, l_tol = 0.0, t_tol = 0.0):
    """Find the intersection of a line within a triangle in 3D space.

    arguments:
       line_p (3 element numpy vector): a point on the line
       line_v (3 element numpy vector): vector being the direction of the line
       triangle ((3, 3) numpy array): three corners of the triangle (second index is xyz)
       line_segment (boolean): if True, returns None if intersection is outwith (line_p .. line_p + line_v)
       l_tol (float, default 0.0): a fraction of the line length to allow for an intersection to be found
           just outside the segment
       t_tol (float, default 0.0): a fraction of the triangle size to allow for an intersection to be found
           just outside the triangle

    returns:
       point (3 element numpy vector) of intersection of the line within the triangle,
       or None if line is parallel to plane of triangle or intersection with the plane is
       outside the triangle
    """

    p01 = triangle[1] - triangle[0]
    p02 = triangle[2] - triangle[0]

    norm = np.cross(p01, p02)  # normal to plane
    line_rv = np.negative(line_v)
    denom = np.dot(line_rv, norm)
    if denom == 0.0:
        return None  # line is parallel to plane
    lp_t0 = line_p - triangle[0]
    t = np.dot(norm, lp_t0) / denom
    if line_segment and (t < 0.0 - l_tol or t > 1.0 + l_tol):
        return None
    u = np.dot(np.cross(p02, line_rv), lp_t0) / denom
    if u < 0.0 - t_tol or u > 1.0 + t_tol:
        return None
    v = np.dot(np.cross(line_rv, p01), lp_t0) / denom
    if v < 0.0 - t_tol or u + v > 1.0 + t_tol:
        return None

    return line_p + t * line_v


@njit
def line_triangle_intersect_numba(
    line_p: np.ndarray,
    line_v: np.ndarray,
    triangle: np.ndarray,
    line_segment: bool = False,
    l_tol: float = 0.0,
    t_tol: float = 0.0,
) -> Union[None, np.ndarray]:
    """Find the intersection of a line within a triangle in 3D space.

    arguments:
        line_p (np.ndarray): a point on the line.
        line_v (np.ndarray): vector being the direction of the line.
        triangle (np.ndarray): shape (3, 3); three corners of the triangle (second index is xyz).
        line_segment (bool): if True, returns None if intersection is outwith (line_p .. line_p + line_v).
        l_tol (float, default 0.0): a fraction of the line length to allow for an intersection to be found
            just outside the segment.
        t_tol (float, default 0.0): a fraction of the triangle size to allow for an intersection to be found
            just outside the triangle.

    returns:
        point (np.ndarray) of intersection of the line within the triangle,
        or None if line is parallel to plane of triangle or intersection with the plane is
        outside the triangle.
    """

    p01 = triangle[1] - triangle[0]
    p02 = triangle[2] - triangle[0]

    norm = np.cross(p01, p02)  # normal to plane
    line_rv = np.negative(line_v)
    denom = np.dot(line_rv, norm)
    if denom == 0.0:
        return None  # line is parallel to plane
    lp_t0 = line_p - triangle[0]
    t = np.dot(norm, lp_t0) / denom
    if line_segment and (t < 0.0 - l_tol or t > 1.0 + l_tol):
        return None
    u = np.dot(np.cross(p02, line_rv), lp_t0) / denom
    if u < 0.0 - t_tol or u > 1.0 + t_tol:
        return None
    v = np.dot(np.cross(line_rv, p01), lp_t0) / denom
    if v < 0.0 - t_tol or u + v > 1.0 + t_tol:
        return None

    return line_p + t * line_v


def line_triangles_intersects(line_p, line_v, triangles, line_segment = False):
    """Find the intersections of a line within each of a set of triangles in 3D space.

    arguments:
       line_p (3 element numpy vector): a point on the line
       line_v (3 element numpy vector): vector being the direction of the line
       triangles ((n, 3, 3) numpy array): three corners of each of the n triangles (final index is xyz)
       line_segment (boolean, default False): if True, the line is treated as a finite segment between
          p and p + v, and only intersections within the segment are included

    returns:
       points ((n, 3) numpy array) of intersection points of the line within the triangles,
       (nan, nan, nan) where line is parallel to plane of triangle or intersection with the plane is
       outside the triangle (or beyond the ends of the segment if applicable)
    """

    n = triangles.shape[0]  # number of triangles

    p01s = triangles[:, 1, :] - triangles[:, 0, :]  # p01s has shape (n, 3)
    p02s = triangles[:, 2, :] - triangles[:, 0, :]  # p02s has shape (n, 3)

    norms = np.cross(p01s, p02s)  # normals to planes, shape (n, 3)
    line_rv = np.negative(line_v)
    denoms = np.dot(norms, line_rv)  # shape (n)
    #  if denom == 0.0: return None  # line is parallel to plane
    lp_t0s = line_p - triangles[:, 0]  # shape (n, 3)
    ts = np.empty(n)
    ts.fill(np.nan)
    us = ts.copy()
    vs = ts.copy()
    nz = (denoms != 0.0)
    #  np.divide(np.dot(norms, lp_t0s), denoms, out = ts, where = nz)
    np.seterr(divide = 'ignore')
    np.divide(np.sum(norms * lp_t0s, axis = 1), denoms, out = ts, where = nz)
    np.seterr(invalid = 'ignore')
    if line_segment:
        ts[:] = np.where(np.logical_or(ts < 0.0, ts > 1.0), np.nan, ts)
    np.divide(np.sum(np.cross(p02s, line_rv) * lp_t0s, axis = 1), denoms, out = us, where = nz)
    np.divide(np.sum(np.cross(line_rv, p01s) * lp_t0s, axis = 1), denoms, out = vs, where = nz)
    ts[:] = np.where(np.logical_or(np.logical_or(us < 0.0, us > 1.0), np.logical_or(vs < 0.0, us + vs > 1.0)), np.nan,
                     ts)

    intersects = np.empty((n, 3))
    intersects[:] = line_v * np.repeat(ts, 3).reshape((n, 3)) + line_p
    np.seterr(all = 'warn')
    return intersects


def intersects_indices(single_line_intersects):
    """Returns a list of the (triangle) indices where a valid intersection has been found for a single line."""

    return list(np.where(np.logical_not(np.isnan(single_line_intersects[..., 0])))[0])


def line_set_triangles_intersects(line_ps, line_vs, triangles, line_segment = False):
    """Find the intersections of each of set of lines within each of a set of triangles in 3D space.

    arguments:
       line_ps ((c, 3) numpy array): a point on each of c lines
       line_vs ((c, 3) numpy array): vectors being the direction of each of the c lines (or 1 common vector)
       triangles ((n, 3, 3) numpy array): three corners of each of the n triangles (final index is xyz)
       line_segment (boolean, default False): if True, each line is treated as a finite segment between
          p and p + v, and only intersections within the segment are included

    returns:
       points ((c, n, 3) numpy array) of intersections of the lines within the triangles,
       (nan, nan, nan) where a line is parallel to plane of triangle or intersection with the plane is
       outside the triangle

    note:
       this function is computationally and memory intensive; it could benefit from parallelisation
    """

    c = line_ps.shape[0]  # number of lines
    n = triangles.shape[0]  # number of triangles

    if line_vs.ndim == 1:  # single common direction vector; for now simply replicate
        line_vs = np.repeat(np.expand_dims(line_vs, axis = 0), c, axis = 0)

    p01s = triangles[:, 1, :] - triangles[:, 0, :]  # p01s has shape (n, 3)
    p02s = triangles[:, 2, :] - triangles[:, 0, :]  # p02s has shape (n, 3)

    norms = np.cross(p01s, p02s)  # normals to planes, shape (n, 3)
    line_rvs = np.negative(line_vs)  # shape (c, 3)
    denoms = np.dot(line_rvs, norms.T)  # shape (c, n)  where denoms == 0.0 line is parallel to plane of triangle
    lps_t0s = line_ps.reshape((c, 1, 3)) - triangles[:, 0, :].reshape((1, n, 3))  # shape (c, n, 3)
    ts = np.empty((c, n))  # shape (c, n)
    ts.fill(np.nan)
    us = ts.copy()  # shape (c, n)
    vs = ts.copy()  # shape (c, n)
    nz = (denoms != 0.0)  # shape (c, n)
    # the np.sum() clause implememts a dot product over the xyz axis
    np.seterr(divide = 'ignore')
    np.divide(np.sum(norms.reshape((1, n, 3)) * lps_t0s, axis = 2), denoms, out = ts, where = nz)  # shape (c, n)
    np.seterr(invalid = 'ignore')
    if line_segment:
        ts[:] = np.where(np.logical_or(ts < 0.0, ts > 1.0), np.nan, ts)
    cp02s = p02s.reshape((1, n, 3)).repeat(c, axis = 0)  # shape (c, n, 3)
    cl_rvs = line_rvs.reshape((c, 1, 3)).repeat(n, axis = 1)  # shape (c, n, 3)
    np.divide(np.sum(np.cross(cp02s, cl_rvs) * lps_t0s, axis = 2), denoms, out = us, where = nz)  # shape (c, n)
    cp01s = p01s.reshape((1, n, 3)).repeat(c, axis = 0)  # shape (c, n, 3)
    np.divide(np.sum(np.cross(cl_rvs, cp01s) * lps_t0s, axis = 2), denoms, out = vs, where = nz)
    ts[:] = np.where(np.logical_or(np.logical_or(us < 0.0, us > 1.0), np.logical_or(vs < 0.0, us + vs > 1.0)), np.nan,
                     ts)

    intersects = np.empty((c, n, 3))
    intersects[:] = line_vs.reshape((c, 1, 3)) * np.repeat(ts, 3).reshape((c, n, 3)) + line_ps.reshape((c, 1, 3))
    np.seterr(all = 'warn')
    return intersects


def poly_line_triangles_intersects(line_ps, triangles):
    """Find the intersections of each segment of an open poly-line with each of a set of triangles in 3D space.

    arguments:
       line_ps ((c, 3) numpy array): ordered points on each of c-1 segments of a poly-line
       triangles ((n, 3, 3) numpy array): three corners of each of the n triangles (final index is xyz)

    returns:
       points ((c-1, n, 3) numpy array) of intersections of the line segments within the triangles,
       (nan, nan, nan) where a line segment is parallel to plane of triangle or intersection of
       the segment with the plane is outside the triangle or beyond the ends of the segment

    note:
       this function is computationally and memory intensive; it could benefit from parallelisation
    """

    return line_set_triangles_intersects(line_ps[:-1], line_ps[1:] - line_ps[:-1], triangles, line_segment = True)


def distilled_intersects(line_set_intersections):
    """Returns lists of line and triangle indices, and corresponding intersection points.

    argument:
       line_set_intersections (numpy float array of shape (nl, nt, 3)): where nl is the number of lines,
          nt is the number of triangles and the final axis is x,y,z; nan values indicate no intersection;
          this array is as returned by the line_set_triangles_intersects() function or the
          poly_line_triangles_intersects() function

    returns:
       (numpy int array of shape (N,), numpy int array of shape (N,), numpy float array of shape (N, 3)):
          for N intersections, first array is list of line indices, second is list of triangle indices, the
          third array contains the (x, y, z) coordinates of the intersection points
    """

    lines, triangles = np.where(np.logical_not(np.isnan(line_set_intersections[:, :, 0])))
    return lines, triangles, line_set_intersections[lines, triangles, :]


def last_intersects(line_set_intersections):
    """From the result of line_set_triangles_intersects(), returns a vector of intersection points, one per line.

    argument:
       line_set_intersections (numpy float array of shape (nl, nt, 3)): where nl is the number of lines,
          nt is the number of triangles and the final axis is x,y,z; nan values indicate no intersection;
          this array is as returned by the line_set_triangles_intersects() function or the
          poly_line_triangles_intersects() function

    returns:
       numpy float array (nl, 3): intersection points, where nl is number of lines

    notes:
       Use this function to force at most one intersection point per line (with a triangulated surface).
       Applicable where lines are expected to be very roughly orthogonal to a gently verying untorn
       surface, eg. pillar lines intersecting an unfaulted horizon.
       If more than one triangle is intersected by a line, the returned point is for the 'last'
       triangle intersected by the line (when checking triangles in the order they appear in the
       list of triangles).
       If no triangles are intersected by a line, the resulting point will be (nan, nan, nan).
    """

    assert line_set_intersections.ndim == 3 and line_set_intersections.shape[2] == 3
    c = line_set_intersections.shape[0]

    pick = np.empty((c, 3))
    pick.fill(np.nan)
    pair_l, pair_t = np.where(np.logical_not(np.isnan(line_set_intersections[:, :, 0])))
    pick[pair_l, :] = line_set_intersections[pair_l, pair_t, :]

    return pick


def triangles_for_line(line_set_intersections, line_index):
    """From the result of line_set_triangles_intersects(), returns a list of intersected triangles for a line.

    arguments:
       line_set_intersections (numpy float array of shape (nl, nt, 3)): where nl is the number of lines,
          nt is the number of triangles and the final axis is x,y,z; nan values indicate no intersection;
          this array is as returned by the line_set_triangles_intersects() function or the
          poly_line_triangles_intersects() function
       line_index (integar): the index of the line for which the intersected triangle list is required

    returns:
       (numpy 1D int array of size N, numpy 2D array of shape (N, 3)): the first of the pair of arrays
          returned is a list of the triangle indices which the given line intersects; the second array is
          the list of corresponding intersection points (each x,y,z)

    notes:
       if the line does not intersect any triangles, both the resulting arrays will have size zero
    """

    triangles = np.where(np.logical_not(np.isnan(line_set_intersections[line_index, :, 0])))[0]
    return triangles, line_set_intersections[line_index, triangles, :]


def lines_for_triangle(line_set_intersections, triangle_index):
    """From the result of line_set_triangles_intersects(), returns a list of lines intersecing given triangle.

    arguments:
       line_set_intersections (numpy float array of shape (nl, nt, 3)): where nl is the number of lines,
          nt is the number of triangles and the final axis is x,y,z; nan values indicate no intersection;
          this array is as returned by the line_set_triangles_intersects() function or the
          poly_line_triangles_intersects() function
       triangle_index (integar): the index of the triangle for which the intersecting line list is required

    returns:
       (numpy 1D int array of size N, numpy 2D array of shape (N, 3)): the first of the pair of arrays
          returned is a list of the indices of lines which intersect with the given triangle; the second
          array is the list of corresponding intersection points (each x,y,z)

    notes:
       if no lines intersect the triangle, both the resulting arrays will have size zero
    """

    lines = np.where(np.logical_not(np.isnan(line_set_intersections[:, triangle_index, 0])))[0]
    return lines, line_set_intersections[lines, triangle_index, :]


def poly_line_triangles_first_intersect(line_ps, triangles, start = 0):
    """Finds the first intersection of a segment of an open poly-line with any of a set of triangles in 3D space.

    arguments:
       line_ps ((c, 3) numpy array): ordered points on each of c-1 segments of a poly-line
       triangles ((n, 3, 3) numpy array): three corners of each of the n triangles (final index is xyz)
       start (int, default 0): the index of the point in line_ps to start searching from

    returns:
       (int, numpy float array of shape (3,)) where the int is the line segment index where one or more
       intersections were found and the floats are the xyz of one intersection of that line segment within
       the triangles; if no line segments intersect any of the trianges, (None, None) is returned

    note:
       this function is computationally and memory intensive; it could benefit from parallelisation
    """

    for c in range(start, len(line_ps) - 1):
        points = line_triangles_intersects(line_ps[c], line_ps[c + 1] - line_ps[c], triangles, line_segment = True)
        if np.all(np.isnan(points)):
            continue  # no intersection found
        xyz = last_intersects(points.reshape((1, -1, 3))).reshape((3,))
        return c, xyz
    return (None, None)


def line_line_intersect(x1, y1, x2, y2, x3, y3, x4, y4, line_segment = False, half_segment = False):
    """Returns the intersection x',y' of two lines x,y 1 to 2 and x,y 3 to 4.

    arguments:
       x1, y1, x2, y2: coordinates of two points defining first line
       x3, y3, x4, y4: coordinates of two points defining second line
       line_segment (bool, default False): if False, both lines are treated as unbounded; if True second line
          is treated as segment bounded by both end points and first line bounding depends on half_segment arg
       half_segment (bool, default False): if True and line_segment is True, first line is bounded only at
          end point x1, y1, whilst second line is fully bounded; if False and line_segment is True, first line
          is also treated as fully bounded; ignored if line_segment is False (ie. both lines unbounded)

    returns:
       pair of floats being x, y of intersection point; or None, None if no qualifying intersection

    note:
       in the case of bounded line segments, both end points are 'included' in the segment
    """

    divisor = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if divisor == 0.0:
        return None, None  # parallel or coincident lines

    if line_segment:
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / divisor
        if t < 0.0 or (not half_segment and t > 1.0):
            return None, None
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / divisor
        if not (0.0 <= u <= 1.0):
            return None, None
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

    else:
        a = x1 * y2 - y1 * x2
        b = x3 * y4 - y3 * x4
        x = (a * (x3 - x4) - (x1 - x2) * b) / divisor
        y = (a * (y3 - y4) - (y1 - y2) * b) / divisor

    return x, y


def point_projected_to_line_2d(p, l1, l2):
    """Return the point on the unbounded line passing through l1 & l2 which is closest to point p, in xy plane."""

    # note: result should be at closest point even in the presence of mixed xy & z units (?)
    # create normal vector to l1, l2
    v = l2 - l1
    n = np.array((-v[1], v[0]))
    # find intersection of p, p->n with l1, l2
    pn = np.array(p[:2]) + n
    return line_line_intersect(l1[0], l1[1], l2[0], l2[1], p[0], p[1], pn[0], pn[1], line_segment = False)


def point_snapped_to_line_segment_2d(p, l1, l2):
    """Returns the point on the bounded line segment l1, l2 which is closest to point p, in xy plane."""

    if vec.is_obtuse_2d(l1, p, l2):
        return l1[:2]
    elif vec.is_obtuse_2d(l2, p, l1):
        return l2[:2]
    else:
        return point_projected_to_line_2d(p, l1, l2)
