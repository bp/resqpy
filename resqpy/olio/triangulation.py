"""triangulation.py: functions for finding Delaunay triangulation and Voronoi graph from a set of points."""

import logging

log = logging.getLogger(__name__)

from typing import Tuple
import math as maths
import numpy as np
from scipy.spatial import Delaunay  # type: ignore

import resqpy.crs as rqc
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.olio.intersection as meet
import resqpy.olio.vector_utilities as vec

# _ccw_t() no longer needed: triangle vertices maintained in anti-clockwise order throughout
# def _ccw_t(p, t):   # puts triangle vertex indices into anti-clockwise order, in situ
#    if vec.clockwise(p[t[0]], p[t[1]], p[t[2]]) > 0.0:
#       t[1], t[2] = t[2], t[1]


def _dt_scipy(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the Delaunay triangulation for an array of points and the convex hull indices.
    
    arguments:
        points (np.ndarray): coordinates of the points to triangulate; array has shape
            (npoints, ndim)

    returns:
        (tuple): tuple containing:

            simplices (np.ndarray): indices of the points forming the triangulation simplices; array
                has shape (nsimplex, ndim+1)
            convex_hull_indices (np.ndarray): indices of the points forming the convex hull; array
                has shape (nhull,)

    note:
        the triangulation is carried out on the points as projected onto the xy plane
    """
    delaunay = Delaunay(points[..., :2])
    simplices = delaunay.simplices
    convex_hull_indices = np.unique(delaunay.convex_hull)
    return simplices, convex_hull_indices


def _dt_simple(po, plot_fn = None, progress_fn = None, container_size_factor = None):
    # returns Delauney triangulation of po and list of hull point indices, using a simple algorithm

    def flip(ei):
        nonlocal fm, e, t, te, p, nt, p_i, ne
        if fm[ei]:
            return  # this edge has already been flipped since last point insertion
        t0, te0 = e[ei, 0]
        t1, te1 = e[ei, 1]
        if t0 < 0 or t1 < 0:
            return  # no triangle on other side
        t0n = te0 - 1
        if t0n < 0:
            t0n = 2
        t1n = te1 - 1
        if t1n < 0:
            t1n = 2
        if (not vec.in_circumcircle(p[t[t0, 0]], p[t[t0, 1]], p[t[t0, 2]], p[t[t1, t1n]]) and
                not vec.in_circumcircle(p[t[t1, 0]], p[t[t1, 1]], p[t[t1, 2]], p[t[t0, t0n]])):
            return  # not sure both are needed
        # flip needed
        ft0 = (t[t0, te0], t[t1, t1n], t[t0, t0n])
        ft1 = (t[t1, te1], t[t0, t0n], t[t1, t1n])
        e[ei, :, 1] = 1
        fte0 = (te[t1, 3 - (te1 + t1n)], ei, te[t0, t0n])
        fte1 = (te[t0, 3 - (te0 + t0n)], ei, te[t1, t1n])
        if e[fte0[0], 0, 0] == t1:
            e[fte0[0], 0] = (t0, 0)
        elif e[fte0[0], 1, 0] == t1:
            e[fte0[0], 1] = (t0, 0)
        else:
            raise Exception('edge breakdown')
        if e[fte1[0], 0, 0] == t0:
            e[fte1[0], 0] = (t1, 0)
        elif e[fte1[0], 1, 0] == t0:
            e[fte1[0], 1] = (t1, 0)
        else:
            raise Exception('edge breakdown')
        if e[fte0[2], 0, 0] == t0:
            e[fte0[2], 0, 1] = 2
        elif e[fte0[2], 1, 0] == t0:
            e[fte0[2], 1, 1] = 2
        else:
            raise Exception('edge breakdown')
        if e[fte1[2], 0, 0] == t1:
            e[fte1[2], 0, 1] = 2
        elif e[fte1[2], 1, 0] == t1:
            e[fte1[2], 1, 1] = 2
        else:
            raise Exception('edge breakdown')
        t[t0] = ft0
        t[t1] = ft1
        te[t0] = fte0
        te[t1] = fte1
        fm[ei] = True
        # debug plot here
        if plot_fn is not None:
            plot_fn(p, t[:nt])
        # recursively flip, not sure all these are needed
        flip(fte0[0])
        flip(fte0[2])
        flip(fte1[0])
        flip(fte1[2])

    n_p = len(po)
    if n_p < 3:
        return None, None  # not enough points
    elif n_p == 3:
        return np.array([0, 1, 2], dtype = int).reshape((1, 3)), np.array([0, 1, 2], dtype = int)

    if progress_fn is not None:
        progress_fn(0.0)

    min_xy = np.min(po[:, :2], axis = 0)
    max_xy = np.max(po[:, :2], axis = 0)
    dxy = max_xy - min_xy
    assert dxy[0] > 0.0 and dxy[1] > 0.0, 'points lie in straight line or are conincident'
    p = np.empty((n_p + 3, 2))
    p[:-3] = po[:, :2]
    # add 3 points sure of containing all po
    csf = 100.0 if container_size_factor is None else max(1.0, container_size_factor)
    p[-3] = (min_xy[0] - 0.8 * csf * dxy[0], min_xy[1] - 0.1 * csf * dxy[1])
    p[-2] = (max_xy[0] + 0.8 * csf * dxy[0], min_xy[1] - 0.1 * csf * dxy[1])
    p[-1] = (min_xy[0] + 0.5 * csf * dxy[0], max_xy[1] + 0.8 * csf * dxy[1])

    # triangle vertex indices
    t = np.empty((2 * n_p + 2, 3), dtype = int)  # empty space for triangle vertex indices
    t[0] = (n_p, n_p + 1, n_p + 2)  # initial set of one containing triangle
    nt = 1  # number of triangles so far populated

    # edges: list of indices of triangles and edge within triangle; -1 indicates no triangle using edge
    e = np.full((3 * n_p + 6, 2, 2), fill_value = -1, dtype = int)
    e[:3, 0, 0] = 0
    for edge in range(3):
        e[edge, 0, 1] = edge
    ne = 3  # number of edges so far in use

    # edge indices (in e) for each triangle, first axis indexed in sync with t
    te = np.empty((2 * n_p + 2, 3), dtype = int)  # empty space for triangle edge indices
    te[0] = (0, 1, 2)

    # mask tracking which edges have been flipped
    fm = np.zeros((3 * n_p + 6), dtype = bool)

    # debug plot here
    #   if plot_fn: plot_fn(p, t[:nt])

    progress_period = max(n_p // 100, 1)
    progress_count = progress_period

    for p_i in range(n_p):  # index of next point to consider

        if progress_fn is not None and progress_count <= 0:
            progress_fn(float(p_i) / float(n_p))
            progress_count = progress_period

        # find triangle that contains this point
        f_t = None
        for ti in range(nt):
            if vec.in_triangle_edged(p[t[ti, 0]], p[t[ti, 1]], p[t[ti, 2]], p[p_i]):
                f_t = ti
                break
        assert f_t is not None, 'failed to find triangle containing point'

        # take copy of edge indices for containing triangle
        e0, e1, e2 = te[f_t]

        # split containing triangle into 3, using new point as common vertex
        t[nt] = (t[f_t, 1], t[f_t, 2], p_i)
        t[nt + 1] = (t[f_t, 2], t[f_t, 0], p_i)
        t[f_t, 2] = p_i
        # add 3 new edges and update te
        e[ne, 0] = (f_t, 2)
        e[ne, 1] = (nt + 1, 1)
        e[ne + 1, 0] = (f_t, 1)
        e[ne + 1, 1] = (nt, 2)
        e[ne + 2, 0] = (nt, 1)
        e[ne + 2, 1] = (nt + 1, 2)
        if e[e1, 0, 0] == f_t:
            e[e1, 0] = (nt, 0)
        elif e[e1, 1, 0] == f_t:
            e[e1, 1] = (nt, 0)
        else:
            raise Exception('edge breakdown')
        if e[e2, 0, 0] == f_t:
            e[e2, 0] = (nt + 1, 0)
        elif e[e2, 1, 0] == f_t:
            e[e2, 1] = (nt + 1, 0)
        else:
            raise Exception('edge breakdown')
        te[nt] = (e1, ne + 2, ne + 1)
        te[nt + 1] = (e2, ne, ne + 2)
        te[f_t, 1] = ne + 1
        te[f_t, 2] = ne

        nt += 2
        ne += 3

        # now recursively try flipping sides
        fm[:] = False

        # debug plot here, with new point, before flipping
        if plot_fn is not None:
            plot_fn(p, t[:nt])

        flip(e0)
        flip(e1)
        flip(e2)

        progress_count -= 1

    # remove any triangles using invented container vertices
    tri_set = t[np.where(np.all(t[:nt] < n_p, axis = 1))]
    if plot_fn is not None:
        plot_fn(p, tri_set)

    external_t = t[np.where(np.any(t[:nt] >= n_p, axis = 1))]
    external_pi = np.unique(external_t)[:-3]  # discard 3 invented container vertices

    if progress_fn is not None:
        progress_fn(1.0)

    return tri_set, external_pi


def dt(p, algorithm = "scipy", plot_fn = None, progress_fn = None, container_size_factor = 100.0, return_hull = False):
    """Returns the Delauney Triangulation of 2D point set p.

    arguments:
       p (numpy float array of shape (N, 2): the x,y coordinates of the points
       algorithm (string, optional): selects which algorithm to use; current options: ['simple', 'scipy'];
          if None, the current best algorithm is selected
       plot_fn (function of form f(p, t), optional): if present, this function is called each time the
          algorithm feels it is worth refreshing a plot of the progress; p is a copy of the point set,
          depending on the algorithm with 3 extra points added to form an enveloping triangle
       progress_fn (function of form f(x), optional): if present, this function is called at regulat
          intervals by the algorithm, passing increasing values in the range 0.0 to 1.0 as x
       container_size_factor (float, default 100.0): the larger this number, the more likely the
          resulting triangulation is to be convex; reduce to 1.0 to allow slight concavities
       return_hull (boolean, default False): if True, a pair is returned with the second item being
          a clockwise ordered list of indices into p identifying the points on the boundary of the
          returned triangulation

    returns:
       numpy int array of shape (M, 3) - being the indices into the first axis of p of the 3 points
          per triangle in the Delauney Triangulation - and if return_hull is True, another int array
          of shape (B,) - being indices into p of the clockwise ordered points on the boundary of
          the triangulation

    notes:
       the plot_fn, progress_fn and container_size_factor arguments are only used by the 'simple' algorithm;
       if points p are 3D, the projection onto the xy plane is used for the triangulation
    """
    assert p.ndim == 2 and p.shape[1] >= 2, 'bad points shape for 2D Delauney Triangulation'

    if not algorithm:
        algorithm = 'scipy'

    if algorithm == 'scipy':
        tri, boundary = _dt_scipy(p)
    elif algorithm == 'simple':
        tri, boundary = _dt_simple(p,
                                   plot_fn = plot_fn,
                                   progress_fn = progress_fn,
                                   container_size_factor = container_size_factor)
    else:
        raise Exception(f'unrecognised Delauney Triangulation algorithm name: {algorithm}')

    assert tri.ndim == 2 and tri.shape[1] == 3

    if return_hull:
        return tri, vec.clockwise_sorted_indices(p, boundary)
    else:
        return tri


def ccc(p1, p2, p3):
    """Returns the centre of the circumcircle of the three points in the xy plane."""

    # two edges as vectors
    v12 = p2 - p1
    v13 = p3 - p1
    # midpoints of the two edges
    m12 = 0.5 * (p1 + p2)
    m13 = 0.5 * (p1 + p3)
    # pairs of points defining two edge normals passing through the midpoints
    o12 = m12.copy()
    o12[0] += v12[1]
    o12[1] -= v12[0]
    o13 = m13.copy()
    o13[0] += v13[1]
    o13[1] -= v13[0]
    return meet.line_line_intersect(m12[0], m12[1], o12[0], o12[1], m13[0], m13[1], o13[0], o13[1])


def voronoi(p, t, b, aoi: rql.Polyline):
    """Returns dual Voronoi diagram for a Delauney triangulation.

    arguments:
       p (numpy float array of shape (N, 2)): seed points used in the Delauney triangulation
       t (numpy int array of shape (M, 3)): the Delauney triangulation of p as returned by dt()
       b (numpy int array of shape (B,)): clockwise sorted list of indices into p of the boundary
          points of the triangulation t
       aoi (lines.Polyline): area of interest; a closed clockwise polyline that must strictly contain
          all p (no points exactly on or outside the polyline)

    returns:
       c, v where: c is a numpy float array of shape (M+E, 2) being the circumcircle centres of
       the M triangles and E boundary points from the aoi polygon line; and v is a list of
       N Voronoi cell lists of clockwise ints, each int being an index into c

    notes:
       the aoi polyline forms the outer boundary for the Voronoi polygons for points on the
       outer edge of the triangulation; all points p must lie strictly within the aoi, which
       must be convex; the triangulation t, of points p, must also have a convex hull; note
       that the dt() function can produce a triangulation with slight concavities on the hull,
       especially for smaller values of its container_size_factor argument
    """

    # this code assumes that the Voronoi polygon for a seed point visits the circumcentres of
    # all the triangles that make use of the point – currently understood to be always the case
    # for a Delauney triangulation

    def __aoi_intervening_nodes(aoi_count, c_count, seg_a, seg_c):
        nodes = []
        seg = seg_a
        while seg != seg_c:
            seg = (seg + 1) % aoi_count
            nodes.append(c_count + seg)
        return nodes

    def __shorter_sides_p_i(p3):
        max_length = -1.0
        max_i = None
        for i in range(3):
            opp_length = vec.naive_length(p3[i - 1] - p3[i - 2])
            if opp_length > max_length:
                max_length = opp_length
                max_i = i
        return max_i

    def __azi_between(a, c, t):
        if c < a:
            c += 360.0
        return (a <= t <= c) or (a <= t + 360.0 <= c)

    def __seg_for_ci(ci):  # returns hull segment for a boundary index
        nonlocal ca_count, cah_count, caho_count, cahon_count, wing_hull_segments
        if ci < ca_count:
            return None
        if ci < cah_count:  # hull edge intersection
            return ci - ca_count
        if ci < caho_count:  # wings
            oi, wing = divmod(ci - cah_count, 2)
            return wing_hull_segments[oi, wing]
        if ci < cahon_count:  # virtual centre for hull edge
            return ci - caho_count
        # else virtual centre for hull point; arbitrarily pick clockwise segment
        return ci - cahon_count

    def __intervening_aoi_indices(aoi_count, aoi_intersect_segments, c_count, ca_count, cah_count, ci_for_p,
                                  out_pair_intersect_segments):
        # build list of intervening aoi boundary point indices and append to list
        aoi_nodes = []
        r = [0] if len(ci_for_p) == 2 else range(len(ci_for_p))
        just_done_pair = False
        for cii in r:
            cip = ci_for_p[cii]
            ci = ci_for_p[(cii + 1) % len(ci_for_p)]
            if cip >= c_count and ci >= c_count and not just_done_pair:
                # identify aoi segments
                if cip < cah_count:
                    aoi_seg_a = aoi_intersect_segments[cip - ca_count]
                else:
                    aoi_seg_a = out_pair_intersect_segments[divmod(cip - cah_count, 2)]
                if ci < cah_count:
                    aoi_seg_c = aoi_intersect_segments[ci - ca_count]
                else:
                    aoi_seg_c = out_pair_intersect_segments[divmod(ci - cah_count, 2)]
                aoi_nodes += __aoi_intervening_nodes(aoi_count, c_count, aoi_seg_a, aoi_seg_c)
                just_done_pair = True
            else:
                just_done_pair = False
        return aoi_nodes

    def __ci_non_hull(b_i, c, c_count, ci_for_p, p, p_i):
        trimmed_ci = []
        cii = 0
        finish_at = len(ci_for_p)
        while cii < finish_at:
            if ci_for_p[cii] < c_count:
                trimmed_ci.append(ci_for_p[cii])
                cii += 1
                continue
            start_cii = cii
            cii_seg = __seg_for_ci(ci_for_p[cii])
            while cii == 0 and ci_for_p[start_cii - 1] >= c_count and __seg_for_ci(ci_for_p[start_cii - 1]) == cii_seg:
                start_cii -= 1
            start_cii = start_cii % len(ci_for_p)
            end_cii = cii + 1
            while end_cii < len(ci_for_p) and ci_for_p[end_cii] >= c_count and __seg_for_ci(
                    ci_for_p[end_cii]) == cii_seg:
                end_cii += 1
            end_cii -= 1  # unpythonesque: end element included in scan
            if end_cii == start_cii:
                trimmed_ci.append(ci_for_p[cii])
                cii += 1
                continue
            if end_cii < start_cii:
                finish_at = start_cii
            if end_cii == (start_cii + 1) % len(ci_for_p):
                trimmed_ci.append(ci_for_p[start_cii])
                trimmed_ci.append(ci_for_p[end_cii])
                cii = end_cii + 1
                continue
            start_azi = vec.azimuth(c[ci_for_p[start_cii]] - p[p_i, :2])
            end_azi = vec.azimuth(c[ci_for_p[end_cii]] - p[p_i, :2])
            scan_cii = start_cii
            while True:
                if scan_cii == start_cii:
                    trimmed_ci.append(ci_for_p[scan_cii])
                elif scan_cii == end_cii:
                    trimmed_ci.append(ci_for_p[scan_cii])
                    break
                else:
                    # if point is around a hull corner from previous, then include (?)
                    next_cii = (scan_cii + 1) % len(ci_for_p)
                    if b_i is None and __seg_for_ci(ci_for_p[scan_cii]) != __seg_for_ci(ci_for_p[next_cii]):
                        trimmed_ci.append(ci_for_p[scan_cii])
                        trimmed_ci.append(ci_for_p[next_cii])
                        scan_cii = next_cii
                    elif b_i is None and __seg_for_ci(trimmed_ci[-1]) != __seg_for_ci(ci_for_p[scan_cii]):
                        trimmed_ci.append(ci_for_p[scan_cii])
                    else:
                        azi = vec.azimuth(c[ci_for_p[scan_cii]] - p[p_i, :2])
                        if __azi_between(start_azi, end_azi, azi):
                            trimmed_ci.append(ci_for_p[scan_cii])
                if scan_cii == end_cii:
                    break
                scan_cii = (scan_cii + 1) % len(ci_for_p)
            cii = end_cii + 1
        return trimmed_ci

    def __closest_to_seed(c, c_count, ci_for_p, hull_node_azi, p, p_i):
        best_a_i = None
        best_c_i = None
        best_a_azi = -181.0
        best_c_azi = 181.0
        for cii, val in enumerate(ci_for_p):
            if val < c_count:
                continue
            azi = vec.azimuth(c[val] - p[p_i, :2]) - hull_node_azi
            if azi > 180.0:
                azi -= 360.0
            elif azi < -180.0:
                azi += 360.0
            if 0.0 > azi > best_a_azi:
                best_a_azi = azi
                best_a_i = cii
            elif 0.0 <= azi < best_c_azi:
                best_c_azi = azi
                best_c_i = cii
        assert best_a_i is not None and best_c_i is not None
        trimmed_ci = []
        for cii, val in enumerate(ci_for_p):
            if val < c_count or cii == best_a_i or cii == best_c_i:
                trimmed_ci.append(val)
        return trimmed_ci

    def __ci_replace(c_count, ca_count, cah_count, caho_count, cahon_count, ci_for_p, p, p_i, t, tc_outwith_aoi):
        # where circumcirle (or virtual) centre is outwith aoi, replace with a point on aoi boundary
        # virtual centres related to hull points (not hull edges) can be discarded
        trimmed_ci = []
        for ci in ci_for_p:
            if ci < c_count:  # genuine triangle
                if ci in tc_outwith_aoi:  # replace with one or two wing normal intersection points
                    oi = tc_outwith_aoi.index(ci)
                    wing_i = cah_count + 2 * oi
                    shorter_t_i = __shorter_sides_p_i(p[t[ci]])
                    if t[ci, shorter_t_i] == p_i:
                        trimmed_ci += [wing_i, wing_i + 1]
                    elif t[ci, shorter_t_i - 1] == p_i:
                        trimmed_ci.append(wing_i)
                    else:
                        trimmed_ci.append(wing_i + 1)
                else:
                    trimmed_ci.append(ci)
            elif ci < cahon_count:
                # extended virtual centre for a hull edge (discard hull point virtual centres)
                # replace with index for intersection point on aoi boundary
                trimmed_ci.append(ca_count + ci - caho_count)
        return trimmed_ci

    def __veroni_cells(aoi_count, aoi_intersect_segments, b, c, c_count, ca_count, cah_count, caho_count, cahon_count,
                       hull_count, out_pair_intersect_segments, p, t, tc_outwith_aoi):
        # list of voronoi cells (each a numpy list of node indices into c extended with aoi points etc)
        v = []
        # for each seed point build the voronoi cell
        for p_i in range(len(p)):

            # find triangles making use of that point
            ci_for_p = list(np.where(t == p_i)[0])

            # if seed point is on hull boundary, introduce three extended virtual centres
            b_i = None
            if p_i in b:
                b_i = np.where(b == p_i)[0][0]  # index into hull coordinates
                p_b_i = (b_i - 1) % hull_count  # predecessor, ie. anti-clockwise boundary point
                ci_for_p += [caho_count + p_b_i, cahon_count + b_i, caho_count + b_i]

            # find azimuths of vectors from seed point to circumcircle centres (and virtual centres)
            azi = [vec.azimuth(centre - p[p_i, :2]) for centre in c[ci_for_p, :2]]
            # if this is a hull seed point, make a note of azimuth to virtual centre
            hull_node_azi = None if b_i is None else azi[-2]
            # sort triangle indices for seed point into clockwise order of circumcircle (and virtual) centres
            ci_for_p = [ti for (_, ti) in sorted(zip(azi, ci_for_p))]

            ci_for_p = __ci_replace(c_count, ca_count, cah_count, caho_count, cahon_count, ci_for_p, p, p_i, t,
                                    tc_outwith_aoi)

            # if this is a hull seed point, classify aoi boundary points into anti-clockwise or clockwise, and find
            # closest to seed
            if b_i is not None:
                ci_for_p = __closest_to_seed(c, c_count, ci_for_p, hull_node_azi, p, p_i)

            # for sequences on aoi boundary, just keep those between the first and last (?)

            #      elif any([ci < c_count for ci in ci_for_p]):
            else:
                ci_for_p = __ci_non_hull(b_i, c, c_count, ci_for_p, p, p_i)

            # reverse points if needed for pair of aoi points only
            assert len(ci_for_p) >= 2
            if len(ci_for_p) == 2:
                seg_0 = __seg_for_ci(ci_for_p[0])
                seg_1 = __seg_for_ci(ci_for_p[1])
                if seg_0 is not None and seg_1 is not None and seg_0 == (seg_1 + 1) % hull_count:
                    ci_for_p.reverse()

            ci_for_p += __intervening_aoi_indices(aoi_count, aoi_intersect_segments, c_count, ca_count, cah_count,
                                                  ci_for_p, out_pair_intersect_segments)

            #  remove circumcircle centres that are outwith area of interest
            ci_for_p = np.array([ti for ti in ci_for_p if ti >= c_count or ti not in tc_outwith_aoi], dtype = int)

            # find azimuths of vectors from seed point to circumcircle centres and aoi boundary points
            azi = [vec.azimuth(centre - p[p_i, :2]) for centre in c[ci_for_p, :2]]

            # re-sort triangle indices for seed point into clockwise order of circumcircle centres and boundary points
            ordered_ci = [ti for (_, ti) in sorted(zip(azi, ci_for_p))]

            v.append(ordered_ci)
        return v

    # log.debug(f'\n\nVoronoi: nt: {len(p)}; nt: {len(t)}; hull: {len(b)}; aoi: {len(aoi.coordinates)}')
    # todo: allow aoi to be None in which case create an aoi as hull with border

    assert p.ndim == 2 and p.shape[0] > 2 and p.shape[1] >= 2
    assert t.ndim == 2 and t.shape[1] == 3
    assert b.ndim == 1 and b.shape[0] > 2
    assert len(aoi.coordinates) >= 3
    assert aoi.isclosed
    assert aoi.is_clockwise()

    # create temporary polyline for hull of triangulation
    hull = rql.Polyline(aoi.model,
                        is_closed = True,
                        set_coord = p[b],
                        set_crs = aoi.crs_uuid,
                        title = 'triangulation hull')
    hull_count = len(b)

    # check for concavities in hull
    if not hull.is_convex():
        log.warning('Delauney triangulation is not convex; Voronoi diagram construction might fail')

    # compute circumcircle centres
    c = np.zeros((t.shape[0], 2))
    for ti in range(len(t)):
        c[ti] = ccc(p[t[ti, 0]], p[t[ti, 1]], p[t[ti, 2]])
    c_count = len(c)

    # make list of triangle indices whose circumcircle centres are outwith the area of interest
    tc_outwith_aoi = [ti for ti in range(c_count) if not aoi.point_is_inside_xy(c[ti])]
    o_count = len(tc_outwith_aoi)

    # make space for combined points data needed for all voronoi cell nodes:
    # 1. circumcircle centres for triangles in delauney triangulation
    # 2. nodes defining area of interest polygon
    # 3. intersection of normals to triangulation hull edges with aoi polygon
    # 4. extra intersections for normals to other two (non-hull) triangle edges, with aoi,
    #    where circumcircle centre is outside the area of interest
    # 5. extended ccc for hull edge normals
    # 6. extended ccc for hull points
    # (5 & 6 only used during construction)
    c = np.concatenate((c, aoi.coordinates[:, :2], np.zeros(
        (hull_count, 2), dtype = float), np.zeros(
            (2 * o_count, 2), dtype = float), np.zeros((hull_count, 2),
                                                       dtype = float), np.zeros((hull_count, 2), dtype = float)))
    aoi_count = len(aoi.coordinates)
    ca_count = c_count + aoi_count
    cah_count = ca_count + hull_count
    caho_count = cah_count + 2 * o_count
    cahon_count = caho_count + hull_count
    assert cahon_count + hull_count == len(c)

    #  compute intersection points between hull edge normals and aoi polyline
    # also extended virtual centres for hull edges
    extension_scaling = 1000.0 * np.sum((np.max(aoi.coordinates, axis = 0) - np.min(aoi.coordinates, axis = 0))[:2])
    aoi_intersect_segments = np.empty((hull_count,), dtype = int)
    for ei in range(hull_count):
        # use segment midpoint and normal methods of hull to project out
        m = hull.segment_midpoint(ei)[:2]  # midpoint
        norm_vec = hull.segment_normal(ei)[:2]
        n = m + norm_vec  # point on normal
        # use first intersection method of aoi to intersect projected normal from triangulation hull
        aoi_seg, aoi_x, aoi_y = aoi.first_line_intersection(m[0], m[1], n[0], n[1], half_segment = True)
        assert aoi_seg is not None
        # inject intersection points to extension area of c and take note of aoi segment of intersection
        c[ca_count + ei] = (aoi_x, aoi_y)
        aoi_intersect_segments[ei] = aoi_seg
        # inject extended virtual circle centres for hull edges, a long way out
        c[caho_count + ei] = c[ca_count + ei] + extension_scaling * norm_vec

    # compute extended virtual centres for hull nodes
    for ei in range(hull_count):
        pei = (ei - 1) % hull_count
        vector = vec.unit_vector(hull.segment_normal(pei)[:2] + hull.segment_normal(ei)[:2])
        c[cahon_count + ei] = hull.coordinates[ei, :2] + extension_scaling * vector

    # where cicrumcircle centres are outwith aoi, compute intersections of normals of wing edges with aoi
    out_pair_intersect_segments = np.empty((o_count, 2), dtype = int)
    wing_hull_segments = np.empty((o_count, 2), dtype = int)
    for oi, ti in enumerate(tc_outwith_aoi):
        tpi = __shorter_sides_p_i(p[t[ti]])
        for wing in range(2):
            # note: triangle nodes are anticlockwise
            m = 0.5 * (p[t[ti, tpi - 1]] + p[t[ti, tpi]])[:2]  # triangle edge midpoint
            edge_v = p[t[ti, tpi]] - p[t[ti, tpi - 1]]
            n = m + np.array((-edge_v[1], edge_v[0]))  # point on perpendicular bisector of triangle edge
            o_seg, o_x, o_y = aoi.first_line_intersection(m[0], m[1], n[0], n[1], half_segment = True)
            c[cah_count + 2 * oi + wing] = (o_x, o_y)
            out_pair_intersect_segments[oi, wing] = o_seg
            wing_hull_segments[oi, wing], _, _ = hull.first_line_intersection(m[0],
                                                                              m[1],
                                                                              n[0],
                                                                              n[1],
                                                                              half_segment = True)
            tpi = (tpi + 1) % 3

    v = __veroni_cells(aoi_count, aoi_intersect_segments, b, c, c_count, ca_count, cah_count, caho_count, cahon_count,
                       hull_count, out_pair_intersect_segments, p, t, tc_outwith_aoi)

    return c[:caho_count], v


def triangulated_polygons(p, v, centres = None):
    """Returns triangulation of polygons using centres as extra points.

    arguments:
       p (2D numpy float array): points used as vertices of polygons
       v (list of list of ints): ordered indices into p for each polygon
       centres (2D numpy float array, optional): the points to use as the centre for each polygon

    returns:
       points, triangles where: points is a copy of p extended with the centre points of polygons;
       and triangles is a numpy int array of shape (N, 3) being the triangulation of points, where N is
       equal to the overall length of v

    notes:
       if no centres are provided, balanced centre points are computed for the polygons;
       the polygons must be convex (at least from the perspective of the centre points);
       the clockwise/anti-clockwise order of the triangle edges will match that of the polygon;
       the centre point is the first point in each triangle;
       the order of triangles will match the order of vertices in a flattened view of list v;
       p and centres may have a shape of 2 or 3 in the second dimension (xy or xyz data);
       p & v could be the values (c, v) returned by the voronoi() function, in which case the
       original seed points p passed into voronoi() can be passed as centres here
    """

    assert p.ndim == 2 and p.shape[1] in [2, 3]
    assert len(v) > 0
    if centres is not None:
        assert centres.ndim == 2
        assert len(centres) == len(v)
        assert centres.shape[1] == p.shape[1]

    model = rq.Model(create_basics = True)  # temporary object for handling Polylines
    crs = rqc.Crs(model)

    points = np.zeros((len(p) + len(v), p.shape[1]))  # polygon nodes, to be extended with centre points
    points[:len(p)] = p
    if centres is not None:
        points[len(p):] = centres

    t_count = sum([len(x) for x in v])
    triangles = np.empty((t_count, 3), dtype = int)
    t_index = 0

    for cell, poly_vertices in enumerate(v):
        # add polygon centre points to points array, if not already provided
        centre_i = len(p) + cell
        if centres is None:
            polygon = rql.Polyline(model,
                                   set_coord = p[np.array(poly_vertices, dtype = int)],
                                   is_closed = True,
                                   set_crs = crs.uuid,
                                   title = 'v cell')
            poly_centre = polygon.balanced_centre()
            points[centre_i] = poly_centre[:p.shape[1]]
        # and populate triangles for this polygon
        for ti in range(len(poly_vertices)):
            triangles[t_index] = (centre_i, poly_vertices[ti], poly_vertices[(ti + 1) % len(poly_vertices)])
            t_index += 1
    assert t_index == t_count

    return points, triangles


def reorient(points, rough = True, max_dip = None, use_linalg = False):
    """Returns a reoriented copy of a set of points, such that z axis is approximate normal to average plane of points.

    arguments:
       points (numpy float array of shape (..., 3)): the points to be reoriented
       rough (bool, default True): if True, the resulting orientation will be within around 10 degrees of the optimum;
          if False, that reduces to around 2.5 degrees of the optimum; iugnored if use_linalg is True
       max_dip (float, optional): if present, the reorientation of perspective off vertical is
          limited to this angle in degrees
       use_linalg (bool, default False): if True, the numpy linear algebra svd function is used and rough is ignored

    returns:
       numpy float array of the same shape as points, numpy xyz vector, numpy 3x3 matrix;
       the array being a copy of points rotated in 3D space to minimise the z range;
       the vector is a normal vector to the original points;
       the matrix is rotation matrix used to transform the original points to the reoriented points

    notes:
       the original points array is not modified by this function;
       implicit xy & z units for points are assumed to be the same;
       the function may typically be called prior to the Delauney triangulation, which uses an xy projection to
       determine the triangulation;
       the numpy linear algebra option seems to be memory intensive, not recommended
    """

    def z_range(p):
        return np.nanmax(p[..., 2]) - np.nanmin(p[..., 2])

    def best_angles(points, mid_x, mid_y, steps, d_theta):
        best_range = None
        best_x_rotation = None
        best_y_rotation = None
        half_steps = float(steps - 1) / 2.0
        for xi in range(steps):
            x_degrees = mid_x + (float(xi) - half_steps) * d_theta
            for yi in range(steps):
                y_degrees = mid_y + (float(yi) - half_steps) * d_theta
                rotation_m = vec.rotation_3d_matrix((x_degrees, 0.0, y_degrees))
                p = points.copy()
                rotated_p = vec.rotate_array(rotation_m, p)
                z_r = z_range(rotated_p)
                if best_range is None or z_r < best_range:
                    best_range = z_r
                    best_x_rotation = x_degrees
                    best_y_rotation = y_degrees
        return (best_x_rotation, best_y_rotation)

    def linalg_normal_vector(p):
        assert p.ndim >= 2 and p.shape[-1] == 3
        p = p.reshape((-1, 3))
        centre = p.sum(axis = 0) / p.shape[0]
        u, s, vh = np.linalg.svd(p - centre)
        # unit normal vector
        return vh[2, :]

    assert points.ndim >= 2 and points.shape[-1] == 3

    if use_linalg:

        normal_vector = linalg_normal_vector(points)
        incl = vec.inclination(normal_vector)
        if incl == 0.0:
            rotation_m = vec.no_rotation_matrix()
        else:
            azi = vec.azimuth(normal_vector)
            rotation_m = vec.tilt_3d_matrix(azi, incl)

    else:

        # coarse iteration trying a few different angles
        best_x_rotation, best_y_rotation = best_angles(points, 0.0, 0.0, 7, 30.0)

        # finer iteration searching around the best coarse rotation
        best_x_rotation, best_y_rotation = best_angles(points, best_x_rotation, best_y_rotation, 5, 10.0)

        if not rough:
            # finer iteration searching around the best coarse rotation
            best_x_rotation, best_y_rotation = best_angles(points, best_x_rotation, best_y_rotation, 7, 2.5)

        rotation_m = vec.rotation_3d_matrix((best_x_rotation, 0.0, best_y_rotation))

        normal_vector = vec.rotate_vector(rotation_m.T, np.array((0.0, 0.0, 1.0)))

    if max_dip is not None:
        v = vec.rotate_vector(rotation_m.T, np.array((0.0, 0.0, 1.0)))
        incl = vec.inclination(v)
        if incl > max_dip:
            azi = vec.azimuth(v)
            rotation_m = vec.tilt_3d_matrix(azi, max_dip)  # TODO: check whether any reverse direction errors here
        normal_vector = vec.rotate_vector(rotation_m.T, np.array((0.0, 0.0, 1.0)))

    p = points.copy()

    return vec.rotate_array(rotation_m, p), normal_vector, rotation_m


def make_all_clockwise_xy(t, p):
    """Modifies t in situ such that each triangle is clockwise in xy plane (viewed from -ve z axis).

    note:
       assumes xyz axes are left handed; all will be made anti-clockwise in the case of right handed xyz axes
    """

    cw = (vec.clockwise_triangles(p, t, projection = 'xy') > 0.0)
    t_flip = t.copy()
    t_flip[:, 0] = t[:, 1]
    t_flip[:, 1] = t[:, 0]
    t[:] = np.where(np.expand_dims(cw, axis = 1), t, t_flip)
    return t


def surrounding_xy_ring(p, count = 12, radial_factor = 10.0, radial_distance = None):
    """Creates a set of points surrounding the point set p, in the xy plane.

    arguments:
       p (numpy float array of shape (..., 3)): xyz set of points to be surrounded
       count (int): the number of points to generate in the surrounding ring
       radial_factor (float): a distance factor roughly determining the radius of the ring relative to
          the 'radius' of the outermost points in p
       radial_distance (float): if present, the radius of the ring of points, unless radial_factor
          results in a greater distance in which case that is used

    returns:
       numpy float array of shape (count, 3) being xyz points in surrounding ring; z is set constant to
       mean value of z in p
    """

    assert p.shape[-1] == 3
    assert radial_factor >= 1.0
    centre = np.nanmean(p.reshape((-1, 3)), axis = 0)
    p_radius_v = np.nanmax(np.abs(p.reshape((-1, 3)) - np.expand_dims(centre, axis = 0)), axis = 0)[:2]
    p_radius = maths.sqrt(np.sum(p_radius_v * p_radius_v))
    radius = p_radius * radial_factor
    if radial_distance is not None and radial_distance > radius:
        radius = radial_distance
    delta_theta = 2.0 * maths.pi / float(count)
    ring = np.zeros((count, 3))
    for i in range(count):
        theta = float(i) * delta_theta
        ring[i] = centre + radius * np.array([maths.cos(theta), maths.sin(theta), 0.0])
    return ring
