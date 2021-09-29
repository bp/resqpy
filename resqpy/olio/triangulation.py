"""triangulation.py: functions for finding Delaunay triangulation and Voronoi graph from a set of points."""

version = '29th September 2021'

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.lines as rql
import resqpy.olio.vector_utilities as vec
import resqpy.olio.intersection as meet

# _ccw_t() no longer needed: triangle vertices maintained in anti-clockwise order throughout
# def _ccw_t(p, t):   # puts triangle vertex indices into anti-clockwise order, in situ
#    if vec.clockwise(p[t[0]], p[t[1]], p[t[2]]) > 0.0:
#       t[1], t[2] = t[2], t[1]


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
      return (np.array([0, 1, 2], dtype = int).reshape((1, 3)), np.array([0, 1, 2], dtype = int))

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

   log.debug(f'dt p: {p}')
   log.debug(f'dt t: {t[:nt]}')
   log.debug(f'dt b: {external_pi}')

   return tri_set, external_pi


def dt(p, algorithm = None, plot_fn = None, progress_fn = None, container_size_factor = 100.0, return_hull = False):
   """Returns the Delauney Triangulation of 2D point set p.

   arguments:
      p (numpy float array of shape (N, 2): the x,y coordinates of the points
      algorithm (string, optional): selects which algorithm to use; current options: ['simple'];
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
   """
   assert p.ndim == 2 and p.shape[1] >= 2, 'bad points shape for 2D Delauney Triangulation'

   if not algorithm:
      algorithm = 'simple'

   if algorithm == 'simple':
      t, boundary = _dt_simple(p,
                               plot_fn = plot_fn,
                               progress_fn = progress_fn,
                               container_size_factor = container_size_factor)
      if return_hull:
         return t, vec.clockwise_sorted_indices(p, boundary)
      else:
         return t
   else:
      raise Exception('unrecognised Delauney Triangulation algorithm name')


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


def voronoi(p, t, b, aoi):
   """Returns dual Voronoi diagram for a Delauney triangulation.

   arguments:
      p (numpy float array of shape (N, 2)): seed points used in the Delauney triangulation
      t (numpy int array of shape (M, 3)): the Delauney triangulation of p as returned by dt()
      b (numpy int array of shape (B,)): clockwise sorted list of indices into p of the boundary
         points of the triangulation t
      aoi (lines.Polyline): area of interest; a closed polyline that must strictly contain
         all p (no points exactly on or outside the polyline)

   returns:
      c, v where: c is a numpy float array of shape (M+E, 2) being the circumcircle centres of
      the M triangles and E boundary points from the aoi polygon line; and v is a list of
      N Voronoi cell lists of clockwise ints, each int being an index into c

   notes:
      the aoi polyline forms the outer boundary for the Voronoi polygons for points on the
      outer edge of the triangulation; all points p must lie strictly within the aoi
   """

   # this code assumes that the Voronoi polygon for a seed point visits the circumcentres of
   # all the triangles that make use of the point – currently understood to be always the case
   # for a Delauney triangulation

   def aoi_intervening_nodes(aoi_count, c_count, seg_a, seg_c):
      nodes = []
      seg = seg_a
      while seg != seg_c:
         seg = (seg + 1) % aoi_count
         nodes.append(c_count + seg)
      return nodes

   def shorter_sides_p_i(p3):
      max_length = -1.0
      max_i = None
      for i in range(3):
         opp_length = vec.naive_length(p3[i - 1] - p3[i - 2])
         if opp_length > max_length:
            max_length = opp_length
            max_i = i
      return max_i

   def azi_between(a, c, t):
      if c < a:
         c += 360.0
      return (a <= t <= c) or (a <= t + 360.0 <= c)

   log.debug(f'p: {p}')
   log.debug(f't: {t}')
   log.debug(f'b: {b}')
   # todo: allow aoi to be None in which case create an aoi as hull with border
   assert p.ndim == 2 and p.shape[0] > 2 and p.shape[1] >= 2
   assert t.ndim == 2 and t.shape[1] == 3
   assert b.ndim == 1 and b.shape[0] > 2
   assert aoi.isclosed

   # create temporary polyline for hull of triangulation
   hull = rql.Polyline(
      aoi.model,
      set_bool = True,  # polyline is closed
      set_coord = p[b],
      set_crs = aoi.crs_uuid,
      title = 'triangulation hull')
   hull_count = len(b)

   # compute circumcircle centres
   c = np.empty((t.shape[0], 2))
   for ti in range(len(t)):
      c[ti] = ccc(p[t[ti, 0]], p[t[ti, 1]], p[t[ti, 2]])
   c_count = len(c)

   # make list of triangle indices whose circumcircle centres are outwith the area of interest
   tc_outwith_aoi = [ti for ti in range(c_count) if not aoi.point_is_inside_xy(c[ti])]
   o_count = len(tc_outwith_aoi)

   # make space for combined points data needed for all voronoi cell nodes:
   # 1. circumcircle centres for traingles in delauney triangulation
   # 2. nodes defining area of interest polygon
   # 3. intersection of normals to traingulation hull edges with aoi polygon
   # 4. extra intersections for normals to other two (non-hull) triangle edges, with aoi,
   #    where circumcircle centre is outside the area of interest
   c = np.concatenate((c, aoi.coordinates[:, :2], np.zeros((hull_count, 2),
                                                           dtype = float), np.zeros((2 * o_count, 2), dtype = float)))
   aoi_count = len(aoi.coordinates)
   ca_count = c_count + aoi_count
   cah_count = ca_count + hull_count
   assert cah_count + 2 * o_count == len(c)

   # compute intersection points between hull normals and aoi polyline
   aoi_intersect_segments = np.empty((hull_count,), dtype = int)
   for ei in range(len(b)):
      # use segment midpoint and normal methods of hull to project out
      m = hull.segment_midpoint(ei)  # midpoint
      n = m + hull.segment_normal(ei)  # point on normal
      # use first intersection method of aoi to intersect projected normal from triangulation hull
      aoi_seg, aoi_x, aoi_y = aoi.first_line_intersection(m[0], m[1], n[0], n[1], half_segment = True)
      assert aoi_seg is not None
      # inject intersection points to extension area of c and take note of aoi segment of intersection
      c[ca_count + ei] = (aoi_x, aoi_y)
      aoi_intersect_segments[ei] = aoi_seg

   # where cicrumcircle centres are outwith aoi, compute intersections of normals of wing edges with aoi
   out_pair_intersect_segments = np.empty((o_count, 2), dtype = int)
   for oi, ti in enumerate(tc_outwith_aoi):
      non_hull_p_count = 0
      non_hull_p_in_t = None
      for tpi in range(3):
         if t[ti, tpi] not in b:
            non_hull_p_count += 1
            non_hull_p_in_t = tpi
      if non_hull_p_count != 1:
         non_hull_p_in_t = shorter_sides_p_i(p[t[ti]])
      tpi = non_hull_p_in_t
      for wing in range(2):
         # note: triangle nodes are anticlockwise
         m = 0.5 * (p[t[ti, tpi - 1]] + p[t[ti, tpi]])[:2]  # triangle edge midpoint
         edge_v = p[t[ti, tpi]] - p[t[ti, tpi - 1]]
         n = m + np.array((-edge_v[1], edge_v[0]))  # point on perpendicular bisector of triangle edge
         o_seg, o_x, o_y = aoi.first_line_intersection(m[0], m[1], n[0], n[1], half_segment = True)
         c[cah_count + 2 * oi + wing] = (o_x, o_y)
         out_pair_intersect_segments[oi, wing] = o_seg
         tpi = (tpi + 1) % 3

   # list of voronoi cells (each a numpy list of node indices into c extended with aoi points then aoi intersections)
   v = []
   # for each seed point build the voronoi cell
   for p_i in range(len(p)):
      log.debug(f'p_i: {p_i}')
      # find triangles making use of that point
      ci_for_p = np.where(t == p_i)[0]
      log.debug(f'ci_for_p: {ci_for_p}')
      # if this point is on boundary, identify the neighbouring boundary points
      e_a_list = []
      e_c_list = []
      aoi_seg_a_list = []
      aoi_seg_c_list = []
      if (p_i in b):
         log.debug('boundary')
         b_i = np.where(b == p_i)[0][0]
         p_b_i_a = (b_i - 1) % hull_count  # predecessor, ie. anti-clockwise boundary point
         hull_t_a = [ti for ti in ci_for_p if b[p_b_i_a] in t[ti]]
         assert len(hull_t_a) == 1
         hull_t_a = hull_t_a[0]
         p_b_i_c = (b_i + 1) % hull_count  # successor, ie. clockwise boundary point
         hull_t_c = [ti for ti in ci_for_p if b[p_b_i_c] in t[ti]]
         assert len(hull_t_c) == 1
         hull_t_c = hull_t_c[0]
         # identify (out, in) intersection info (fill in with aoi points after)
         # if hull triangles have circumcircle centre outwith aoi, use wing normal intersection with aoi
         #    else use hull edge normal intersection with aoi
         # one intersection point (anticlockwise wrt. p_i)
         if hull_t_a in tc_outwith_aoi:
            oi = tc_outwith_aoi.index(hull_t_a)
            e_a = cah_count + 2 * oi + 1
            aoi_seg_a = out_pair_intersect_segments[oi, 1]
         else:
            e_a = ca_count + p_b_i_a
            aoi_seg_a = aoi_intersect_segments[p_b_i_a]
         # the other intersection point (clockwise wrt. p_i)
         if hull_t_c in tc_outwith_aoi:
            oi = tc_outwith_aoi.index(hull_t_c)
            e_c = cah_count + 2 * oi
            aoi_seg_c = out_pair_intersect_segments[oi, 0]
         else:
            e_c = ca_count + b_i
            aoi_seg_c = aoi_intersect_segments[b_i]
         e_a_list = [e_a]
         e_c_list = [e_c]
         aoi_seg_a_list = [aoi_seg_a]
         aoi_seg_c_list = [aoi_seg_c]
      # check for (non-hull) triangles with a circumcircle centre outwith aoi
      for t_i in [ti for ti in ci_for_p if ti in tc_outwith_aoi]:
         t_hull_count = 0
         for i in range(3):
            if t[t_i, i] in b:
               t_hull_count += 1
         is_hull_triangle = (t_hull_count >= 2)
         if is_hull_triangle and (p_i in b):
            continue  # already handled above
         oi = tc_outwith_aoi.index(t_i)
         e_a_t = cah_count + 2 * oi
         e_c_t = e_a_t + 1
         a_wing = 0
         if not is_hull_triangle:
            e_a_t, e_c_t = e_c_t, e_a_t
            a_wing = 1
         e_a_list.append(e_a_t)
         aoi_seg_a_list.append(out_pair_intersect_segments[oi, a_wing])
         e_c_list.append(e_c_t)
         aoi_seg_c_list.append(out_pair_intersect_segments[oi, 1 - a_wing])

      # check for interference between aoi boundary sections and distill
      if len(e_a_list) > 1:
         assert len(e_a_list) <= 3, f'cannot handle {len(e_a_list)} aoi boundary sections for a single point'
         azi_a_list = []
         azi_c_list = []
         for i in range(len(e_a_list)):
            azi_a_list.append(vec.azimuth(p[p_i][:2] - c[e_a_list[i]]))
            azi_c_list.append(vec.azimuth(p[p_i][:2] - c[e_c_list[i]]))
         distilled_a_indices = []  # indices into existing lists
         distilled_c_indices = []
         if len(e_a_list) == 2:
            pairs = ((0, 1),)
         else:
            pairs = ((0, 1), (0, 2), (1, 2))
         for pair in pairs:
            a0, a1 = azi_a_list[pair[0]], azi_a_list[pair[1]]
            c0, c1 = azi_c_list[pair[0]], azi_c_list[pair[1]]
            if azi_between(a0, c0, a1) and azi_between(a0, c0, c1):
               distilled_a_indices.append(pair[1])
               distilled_c_indices.append(pair[1])
            elif azi_between(a1, c1, a0) and azi_between(a1, c1, c0):
               distilled_a_indices.append(pair[0])
               distilled_c_indices.append(pair[0])
            elif azi_between(a0, c0, a1):
               distilled_a_indices.append(pair[1])
               distilled_c_indices.append(pair[0])
            elif azi_between(a1, c1, a0):
               distilled_a_indices.append(pair[0])
               distilled_c_indices.append(pair[1])
            else:
               distilled_a_indices += list(pair)
               distilled_c_indices += list(pair)
         distilled_e_a_list = []
         distilled_e_c_list = []
         distilled_aoi_seg_a_list = []
         distilled_aoi_seg_c_list = []
         for a_i in distilled_a_indices:
            distilled_e_a_list.append(e_a_list[a_i])
            distilled_aoi_seg_a_list.append(aoi_seg_a_list[a_i])
         for c_i in distilled_c_indices:
            distilled_e_c_list.append(e_c_list[c_i])
            distilled_aoi_seg_c_list.append(aoi_seg_c_list[c_i])
         e_a_list = distilled_e_a_list
         e_c_list = distilled_e_c_list
         aoi_seg_a_list = distilled_aoi_seg_a_list
         aoi_seg_c_list = distilled_aoi_seg_c_list

      # build list of aoi boundary point indices ready for insertion into cell node indices
      aoi_nodes = []
      for e_a, e_c, aoi_seg_a, aoi_seg_c in zip(e_a_list, e_c_list, aoi_seg_a_list, aoi_seg_c_list):
         aoi_nodes = [e_a]
         aoi_nodes += aoi_intervening_nodes(aoi_count, c_count, aoi_seg_a, aoi_seg_c)
         aoi_nodes.append(e_c)

      # append any aoi boundary point c indices to ci_for_p
      if len(aoi_nodes):
         log.debug(f'aoi_nodes: {aoi_nodes}')
         ci_for_p = np.concatenate((ci_for_p, np.array(aoi_nodes, dtype = int)))
         log.debug(f'expanded ci_for_p: {ci_for_p}')

      # remove circumcircle centres that are outwith area of interest
      ci_for_p = np.array([ti for ti in ci_for_p if ti >= c_count or ti not in tc_outwith_aoi], dtype = int)

      # find azimuths of vectors from seed point to circumcircle centres and aoi boundary points
      azi = [vec.azimuth(centre - p[p_i]) for centre in c[ci_for_p]]
      # sort triangle indices for seed point into clockwise order of circumcircle centres and boundary points
      ordered_ci = [ti for (_, ti) in sorted(zip(azi, ci_for_p))]

      v.append(ordered_ci)

   return c, v
