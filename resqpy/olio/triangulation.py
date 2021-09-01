"""triangulation.py: functions for finding Delaunay triangulation and Voronoi graph from a set of points."""

version = '1st September 2021'

import numpy as np

import resqpy.olio.vector_utilities as vec

# _ccw_t() no longer needed: triangle vertices maintained in anti-clockwise order throughout
# def _ccw_t(p, t):   # puts triangle vertex indices into anti-clockwise order, in situ
#    if vec.clockwise(p[t[0]], p[t[1]], p[t[2]]) > 0.0:
#       t[1], t[2] = t[2], t[1]


def _dt_simple(po, plot_fn = None, progress_fn = None):

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
      return None  # not enough points
   elif n_p == 3:
      return np.array([0, 1, 2], dtype = int).reshape((1, 3))

   if progress_fn is not None:
      progress_fn(0.0)

   min_xy = np.min(po[:, :2], axis = 0)
   max_xy = np.max(po[:, :2], axis = 0)
   dxy = max_xy - min_xy
   assert dxy[0] > 0.0 and dxy[1] > 0.0, 'points lie in straight line or are conincident'
   p = np.empty((n_p + 3, 2))
   p[:-3] = po[:, :2]
   # add 3 points sure of containing all po
   p[-3] = (min_xy[0] - 0.8 * dxy[0], min_xy[1] - 0.1 * dxy[1])
   p[-2] = (max_xy[0] + 0.8 * dxy[0], min_xy[1] - 0.1 * dxy[1])
   p[-1] = (min_xy[0] + 0.5 * dxy[0], max_xy[1] + 0.8 * dxy[1])

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
   if progress_fn is not None:
      progress_fn(1.0)

   return tri_set


def dt(p, algorithm = None, plot_fn = None, progress_fn = None):
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

   returns:
      numpy int array of shape (M, 3) being the indices into the first axis of p of the 3 points
         per triangle in the Delauney Triangulation
   """
   assert p.ndim == 2 and p.shape[1] >= 2, 'bad points shape for 2D Delauney Triangulation'

   if not algorithm:
      algorithm = 'simple'

   if algorithm == 'simple':
      return _dt_simple(p, plot_fn = plot_fn, progress_fn = progress_fn)
   else:
      raise Exception('unrecognised Delauney Triangulation algorithm name')
