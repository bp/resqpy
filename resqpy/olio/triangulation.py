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
   c = np.zeros((t.shape[0], 2))
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

   # compute intersection points between hull edge normals and aoi polyline
   # also extended virtual centres for hull edges
   extension_scaling = 1000.0 * np.sum((np.max(aoi.coordinates, axis = 0) - np.min(aoi.coordinates, axis = 0))[:2])
   aoi_intersect_segments = np.empty((hull_count,), dtype = int)
   for ei in range(len(b)):
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
   for ei in range(len(b)):
      pei = (ei - 1) % len(b)
      vector = vec.unit_vector(hull.segment_normal(pei)[:2] + hull.segment_normal(ei)[:2])
      c[cahon_count + ei] = hull.coordinates[ei, :2] + extension_scaling * vector

   # where cicrumcircle centres are outwith aoi, compute intersections of normals of wing edges with aoi
   out_pair_intersect_segments = np.empty((o_count, 2), dtype = int)
   for oi, ti in enumerate(tc_outwith_aoi):
      tpi = shorter_sides_p_i(p[t[ti]])
      for wing in range(2):
         # note: triangle nodes are anticlockwise
         m = 0.5 * (p[t[ti, tpi - 1]] + p[t[ti, tpi]])[:2]  # triangle edge midpoint
         edge_v = p[t[ti, tpi]] - p[t[ti, tpi - 1]]
         n = m + np.array((-edge_v[1], edge_v[0]))  # point on perpendicular bisector of triangle edge
         o_seg, o_x, o_y = aoi.first_line_intersection(m[0], m[1], n[0], n[1], half_segment = True)
         c[cah_count + 2 * oi + wing] = (o_x, o_y)
         out_pair_intersect_segments[oi, wing] = o_seg
         tpi = (tpi + 1) % 3
         log.debug(f"wings: ti: {ti}; c pair {'c' if wing else 'a'}: {(o_x, o_y)}")

   # list of voronoi cells (each a numpy list of node indices into c extended with aoi points etc)
   v = []

   # for each seed point build the voronoi cell
   for p_i in range(len(p)):

      if p_i in [0, 10]:
         log.debug(f'****** p_i: {p_i}')
      # find triangles making use of that point
      ci_for_p = list(np.where(t == p_i)[0])
      if p_i in [0, 10]:
         log.debug(f'ci_for_p: {ci_for_p}')

      # if seed point is on hull boundary, introduce three extended virtual centres
      b_i = None
      if (p_i in b):
         b_i = np.where(b == p_i)[0][0]  # index into hull coordinates
         p_b_i = (b_i - 1) % hull_count  # predecessor, ie. anti-clockwise boundary point
         ci_for_p += [caho_count + p_b_i, cahon_count + b_i, caho_count + b_i]
         if p_i in [0, 10]:
            log.debug(f'boundary ci_for_p: {ci_for_p}')

      # find azimuths of vectors from seed point to circumcircle centres (and virtual centres)
      azi = [vec.azimuth(centre - p[p_i]) for centre in c[ci_for_p]]
      # if this is a hull seed point, make a note of azimuth to virtual centre
      hull_node_azi = None if b_i is None else azi[-2]
      # sort triangle indices for seed point into clockwise order of circumcircle (and virtual) centres
      ci_for_p = [ti for (_, ti) in sorted(zip(azi, ci_for_p))]

      if p_i in [0, 10]:
         log.debug(f'sorted ci_for_p: {ci_for_p}')

      # where circumcirle (or virtual) centre is outwith aoi, replace with a point on aoi boundary
      # virtual centres related to hull points (not hull edges) can be discarded
      trimmed_ci = []
      for ci in ci_for_p:
         if ci < c_count:  # genuine triangle
            if ci in tc_outwith_aoi:  # replace with one or two wing normal intersection points
               oi = tc_outwith_aoi.index(ci)
               wing_i = cah_count + 2 * oi
               shorter_t_i = shorter_sides_p_i(p[t[ci]])
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
      ci_for_p = trimmed_ci
      if p_i in [0, 10]:
         log.debug(f'trimmed to aoi ci_for_p: {ci_for_p}')

      # if this is a hull seed point, classify aoi boundary points into anti- or clockwise, and find closest to seed
      if b_i is not None:
         best_a_i = None
         best_c_i = None
         best_a_azi = -181.0
         best_c_azi = 181.0
         for cii in range(len(ci_for_p)):
            if ci_for_p[cii] < c_count:
               continue
            azi = vec.azimuth(c[ci_for_p[cii]] - p[p_i, :2]) - hull_node_azi
            if azi > 180.0:
               azi -= 360.0
            elif azi < -180.0:
               azi += 360.0
            if azi < 0.0:
               if azi > best_a_azi:
                  best_a_azi = azi
                  best_a_i = cii
            else:
               if azi < best_c_azi:
                  best_c_azi = azi
                  best_c_i = cii
         assert best_a_i is not None and best_c_i is not None
         trimmed_ci = []
         for cii in range(len(ci_for_p)):
            if ci_for_p[cii] < c_count or cii == best_a_i or cii == best_c_i:
               trimmed_ci.append(ci_for_p[cii])
         ci_for_p = trimmed_ci

      # for sequences on aoi boundary, just keep those between the first and last (?)
      elif any([ci < c_count for ci in ci_for_p]):
         trimmed_ci = []
         cii = 0
         finish_at = len(ci_for_p)
         while True:
            if cii >= finish_at:
               break
            if ci_for_p[cii] < c_count:
               trimmed_ci.append(ci_for_p[cii])
               cii += 1
               continue
            start_cii = cii
            while ci_for_p[start_cii - 1] >= c_count:
               start_cii -= 1
            start_cii = start_cii % len(ci_for_p)
            end_cii = (cii + 1) % len(ci_for_p)
            while ci_for_p[end_cii] >= c_count:
               end_cii = (end_cii + 1) % len(ci_for_p)
            end_cii = (end_cii - 1) % len(ci_for_p)  # unpythonesque: end element included in scan
            if end_cii == start_cii:
               trimmed_ci.append(ci_for_p[cii])
               cii += 1
               continue
            elif end_cii == (start_cii + 1) % len(ci_for_p):
               trimmed_ci.append(ci_for_p[start_cii])
               trimmed_ci.append(ci_for_p[end_cii])
               cii += 2
               continue
            if end_cii < start_cii:
               finish_at = start_cii
            start_azi = vec.azimuth(c[ci_for_p[start_cii]] - p[p_i, :2])
            end_azi = vec.azimuth(c[ci_for_p[end_cii]] - p[p_i, :2])
            scan_cii = start_cii
            while True:
               if scan_cii == start_cii:
                  trimmed_ci.append(ci_for_p[scan_cii])
               if scan_cii == end_cii:
                  trimmed_ci.append(ci_for_p[scan_cii])
                  break
               else:
                  # TODO: if b_i is None and (aoi_segs are different, or hull segs are different), append
                  azi = vec.azimuth(c[ci_for_p[scan_cii]] - p[p_i, :2])
                  if azi_between(start_azi, end_azi, azi):
                     trimmed_ci.append(ci_for_p[scan_cii])
               scan_cii = (scan_cii + 1) % len(ci_for_p)
            cii = end_cii + 1
         ci_for_p = trimmed_ci
      if p_i in [0, 10]:
         log.debug(f'first & last aoi ci_for_p: {ci_for_p}')

      assert len(ci_for_p) >= 2

      # build list of intervening aoi boundary point indices and append to list
      aoi_nodes = []
      r = [1] if len(ci_for_p) == 2 else range(len(ci_for_p))
      for cii in r:
         cip = ci_for_p[cii - 1]
         ci = ci_for_p[cii]
         if cip >= c_count and ci >= c_count:
            # identify aoi segments
            if cip < cah_count:
               aoi_seg_a = aoi_intersect_segments[cip - ca_count]
            else:
               aoi_seg_a = out_pair_intersect_segments[divmod(cip - cah_count, 2)]
            if ci < cah_count:
               aoi_seg_c = aoi_intersect_segments[ci - ca_count]
            else:
               aoi_seg_c = out_pair_intersect_segments[divmod(ci - cah_count, 2)]
            aoi_nodes += aoi_intervening_nodes(aoi_count, c_count, aoi_seg_a, aoi_seg_c)
      ci_for_p += aoi_nodes
      if p_i in [0, 10]:
         log.debug(f'aoi node infill ci_for_p: {ci_for_p}')

      # remove circumcircle centres that are outwith area of interest
      ci_for_p = np.array([ti for ti in ci_for_p if ti >= c_count or ti not in tc_outwith_aoi], dtype = int)
      if p_i in [0, 10]:
         log.debug(f'outwith removed ci_for_p: {ci_for_p}')

      # find azimuths of vectors from seed point to circumcircle centres and aoi boundary points
      azi = [vec.azimuth(centre - p[p_i]) for centre in c[ci_for_p]]

      # re-sort triangle indices for seed point into clockwise order of circumcircle centres and boundary points
      ordered_ci = [ti for (_, ti) in sorted(zip(azi, ci_for_p))]
      if p_i in [0, 10]:
         log.debug(f'ordered ci: {ci_for_p}')

      v.append(ordered_ci)

   return c[:caho_count], v
