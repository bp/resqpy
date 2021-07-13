# grid functions module

version = '29th April 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('grid_functions.py version %s', version)

# defs:
# def infill_block_geometry(extent, depth, thickness, x, y,
#    k_increase_direction = 'down', depth_zero_tolerance = 0.01,
#    x_y_zero_tolerance = 0.01,
#    vertical_cell_overlap_tolerance = 0.01,
#    snap_to_top_and_base = True, nudge = True):
# def resequence_nexus_corp(corner_points, eight_mode = False, undo = False):
# def random_cell(corner_points, border = 0.25, max_tries = 20, tolerance = 0.003):
# def determine_corp_ijk_handedness(corner_points, xyz_is_left_handed = True):
# def determine_corp_extent(corner_points, tolerance = 0.003):
# def translate_corp(corner_points, x_shift = None, y_shift = None, min_xy = 0.0, shift_rounding_digits = None):

import math as maths
import random
import numpy as np

import resqpy.olio.vector_utilities as vec
import resqpy.olio.factors as factors

##########################################################################################
# infill_block_geometry():
# scans each logically vertical column of cells,
# assigning depth and thickness values for inactive cells sandwiched between active cells
# extent is a 3 element vector: nk,nj,ni
# depth is a 3D numpy float array of size matching extent
# depth values assumed more positive with increasing depth
# zero values in depth input array indicate inactive cells
# thickness is a 3D numpy float array of size matching extent
# x and y are each a 3D numpy float array of size matching extent
# k_increase_direction is either 'up' or 'down'
# depth_zero_tolerance is maximum value for which depth is considered zero
# vertical_cell_overlap_tolerance is the maximum acceptable overlap of cells on input
# snap_to_top_and_base, when True, causes cells above topmost active and below deepest active
# to be populated with pinched out cells at the top and bottom faces respectively
# nudge causes the depth of cells with greater k to be moved to clean up overlap over pinchouts


def infill_block_geometry(extent,
                          depth,
                          thickness,
                          x,
                          y,
                          k_increase_direction = 'down',
                          depth_zero_tolerance = 0.01,
                          x_y_zero_tolerance = 0.01,
                          vertical_cell_overlap_tolerance = 0.01,
                          snap_to_top_and_base = True,
                          nudge = True):
   """Scans logically vertical columns of cells setting depth (& thickness) of inactive cells."""

   if k_increase_direction == 'down':
      k_dir_sign = 1.0
   elif k_increase_direction == 'up':
      k_dir_sign = -1.0
   else:
      assert (False)

   for j in range(extent[1]):
      for i in range(extent[2]):
         k_top = 0  # NB: 'top' & 'bottom' are misleading if k_increase_direction == 'up'
         while k_top < extent[0] and abs(depth[k_top, j, i]) <= depth_zero_tolerance:
            depth[k_top, j, i] = 0.0  # clean up tiny values
            thickness[k_top, j, i] = 0.0
            if abs(x[k_top, j, i]) <= x_y_zero_tolerance:
               x[k_top, j, i] = 0.0
            if abs(y[k_top, j, i]) <= x_y_zero_tolerance:
               y[k_top, j, i] = 0.0
            k_top += 1  # skip topmost inactive batch
         if k_top >= extent[0]:
            continue  # whole column is inactive
         if snap_to_top_and_base:
            snap_depth = depth[k_top, j, i] - k_dir_sign * thickness[k_top, j, i] / 2.0
            snap_x = x[k_top, j, i]
            snap_y = y[k_top, j, i]
            for k_snap in range(k_top):
               depth[k_snap, j, i] = snap_depth
               x[k_snap, j, i] = snap_x
               y[k_snap, j, i] = snap_y
         while True:
            while k_top < extent[0] and abs(depth[k_top, j, i]) > depth_zero_tolerance:  # skip active layers
               k_top += 1
            k_base = k_top + 1
            while k_base < extent[0] and abs(depth[k_base, j, i]) <= depth_zero_tolerance:
               depth[k_base, j, i] = 0.0  # clean up tiny depth values
               thickness[k_base, j, i] = 0.0
               if abs(x[k_base, j, i]) <= x_y_zero_tolerance:
                  x[k_base, j, i] = 0.0
               if abs(y[k_base, j, i]) <= x_y_zero_tolerance:
                  y[k_base, j, i] = 0.0
               k_base += 1  # look for deeper active layer
            if k_base >= extent[0]:  # no deeper active cells found
               if snap_to_top_and_base:
                  snap_depth = depth[k_top - 1, j, i] + k_dir_sign * thickness[k_top - 1, j, i] / 2.0
                  snap_x = x[k_top - 1, j, i]
                  snap_y = y[k_top - 1, j, i]
                  for k_snap in range(extent[0] - k_top):
                     depth[k_top + k_snap, j, i] = snap_depth
                     x[k_top + k_snap, j, i] = snap_x
                     y[k_top + k_snap, j, i] = snap_y
               break
            void_cell_count = k_base - k_top
            assert (void_cell_count > 0)
            void_top_depth = depth[k_top - 1, j, i] + (thickness[k_top - 1, j, i] / 2.0) * k_dir_sign
            void_bottom_depth = depth[k_base, j, i] - (thickness[k_base, j, i] / 2.0) * k_dir_sign
            void_top_x = x[k_top - 1, j, i]
            void_top_y = y[k_top - 1, j, i]
            void_interval = k_dir_sign * (void_bottom_depth - void_top_depth)
            void_x_interval = x[k_base, j, i] - void_top_x
            void_y_interval = y[k_base, j, i] - void_top_y
            infill_cell_thickness = void_interval / void_cell_count
            if void_interval < 0.0:  # overlapping cells
               if -void_interval < vertical_cell_overlap_tolerance:
                  if nudge:
                     nudge_count = 0  # debug
                     for k_nudge in range(extent[0] - k_base):
                        if depth[k_base + k_nudge, j, i] > depth_zero_tolerance:
                           depth[k_base + k_nudge, j, i] += -void_interval * k_dir_sign
                           nudge_count += 1  # debug
                     log.debug('%1d cells nudged in [ i j ] column [%1d, %1d]', nudge_count, i + 1, j + 1)
                     void_bottom_depth += -void_interval
                  void_interval = 0.0
                  infill_cell_thickness = 0.0
               else:
                  log.warn('Cells [%1d, %1d, %1d] and [%1d, %1d, %1d] overlap ...', i + 1, j + 1, k_top, i + 1, j + 1,
                           k_base + 1)
                  log.warn('   check k_increase_direction and tolerances')
                  log.warn('Skipping rest of i,j column')  # todo: could abort here
                  break
            assert (infill_cell_thickness >= 0.0)
            for void_k in range(void_cell_count):
               depth[k_top + void_k, j, i] = void_top_depth + (0.5 + void_k) * infill_cell_thickness * k_dir_sign
               thickness[k_top + void_k, j, i] = infill_cell_thickness
               x[k_top + void_k, j, i] = void_top_x + (0.5 + void_k) * void_x_interval / void_cell_count
               y[k_top + void_k, j, i] = void_top_y + (0.5 + void_k) * void_y_interval / void_cell_count
            k_top = k_base


# end of def infill_block_geometry()
##########################################################################################

##########################################################################################
# def resequence_nexus_corp():


def resequence_nexus_corp(corner_points, eight_mode = False, undo = False):
   """Reorders corner point data in situ, to handle bizarre nexus orderings"""

   # undo False for corp to internal; undo True for internal to corp; only relevant in eight_mode
   assert (corner_points.ndim == 7)
   extent = np.array(corner_points.shape, dtype = 'int')
   if eight_mode:
      for k in range(extent[0]):
         for j in range(extent[1]):
            for i in range(extent[2]):
               if undo:
                  xyz = np.zeros((3, 8))
                  c = 0
                  for kp in range(2):
                     for jp in range(2):
                        for ip in range(2):
                           xyz[:, c] = corner_points[k, j, i, kp, jp, ip, :]
                           c += 1
                  corner_points[k, j, i] = xyz.reshape((2, 2, 2, 3))
               else:
                  xyz = corner_points[k, j, i].reshape((3, 8)).copy()
                  c = 0
                  for kp in range(2):
                     for jp in range(2):
                        for ip in range(2):
                           corner_points[k, j, i, kp, jp, ip, :] = xyz[:, c]
                           c += 1
   else:  # reversible, so not dependent on undo argument
      jp_slice = corner_points[:, :, :, :, 1, 0, :].copy()
      corner_points[:, :, :, :, 1, 0, :] = corner_points[:, :, :, :, 1, 1, :]
      corner_points[:, :, :, :, 1, 1, :] = jp_slice


# end of def resequence_nexus_corp()
##########################################################################################

##########################################################################################
# def random_cell():


def random_cell(corner_points, border = 0.25, max_tries = 20, tolerance = 0.003):
   """Returns a random cell's (k,j,i) tuple for a cell with non-zero lengths on all 3 primary edges."""

   assert (corner_points.ndim == 7)
   assert (border >= 0.0 and border < 0.5)
   assert (max_tries > 0)

   extent = np.array(corner_points.shape, dtype = 'int')
   kji_extent = extent[:3]
   kji_border = np.zeros(3, dtype = 'int')
   kji_upper = np.zeros(3, dtype = 'int')
   for axis in range(3):
      kji_border[axis] = int(float(kji_extent[axis]) * border)
      kji_upper[axis] = kji_extent[axis] - kji_border[axis] - 1
      if kji_upper[axis] < kji_border[axis]:
         kji_upper[axis] = kji_border[axis]

   kji_cell = np.empty(3, dtype = 'int')
   attempt = 0
   while attempt < max_tries:
      attempt += 1
      for axis in range(3):
         if kji_extent[axis] == 1:
            kji_cell[axis] = 0
         else:
            kji_cell[axis] = random.randint(kji_border[axis], kji_upper[axis])
      cell_cp = corner_points[tuple(kji_cell)]
      assert (cell_cp.shape == (2, 2, 2, 3))
      if vec.manhatten_distance(cell_cp[0, 0, 0], cell_cp[1, 0, 0]) < tolerance:
         continue
      if vec.manhatten_distance(cell_cp[0, 0, 0], cell_cp[0, 1, 0]) < tolerance:
         continue
      if vec.manhatten_distance(cell_cp[0, 0, 0], cell_cp[0, 0, 1]) < tolerance:
         continue
      return tuple(kji_cell)

   log.warning('failed to find random voluminous cell')
   return None


# end of def random_cell()
##########################################################################################

##########################################################################################
# def determine_corp_ijk_handedness():


def determine_corp_ijk_handedness(corner_points, xyz_is_left_handed = True):
   """Determine true ijk handedness from corner point data in pagoda style 7D array; returns 'right' or 'left'."""

   assert (corner_points.ndim == 7)
   cell_kji = random_cell(corner_points)
   assert (cell_kji is not None)
   log.debug('using cell ijk0 [{}, {}, {}] to determine ijk handedness'.format(cell_kji[2], cell_kji[1], cell_kji[0]))
   cell_cp = corner_points[cell_kji]
   origin = cell_cp[0, 0, 0]
   det = vec.determinant(cell_cp[0, 0, 1] - origin, cell_cp[0, 1, 0] - origin,
                         cell_cp[1, 0, 0] - origin)  # NB. IJK ordering
   if det == 0.0:
      log.warning('indeterminate handedness in cell ijk0 [{}, {}, {}]'.format(cell_kji[2], cell_kji[1], cell_kji[0]))
      return None
   if det > 0.0:
      ijk_is_left_handed = xyz_is_left_handed
   else:
      ijk_is_left_handed = not xyz_is_left_handed
   if ijk_is_left_handed:
      return 'left'
   return 'right'


# end of def determine_corp_ijk_handedness()
##########################################################################################

##########################################################################################
# def determine_corp_extent():


def determine_corp_extent(corner_points, tolerance = 0.003):
   """Returns extent of grid derived from 7D corner points with all cells temporarily in I."""

   def neighbours(corner_points, sextuple_cell_a_p1, sextuple_cell_a_p2, sextuple_cell_b_p1, sextuple_cell_b_p2,
                  tolerance):
      # allows for reversal of points (or not) in neighbouring cell
      if (
         (vec.manhatten_distance(corner_points[sextuple_cell_a_p1], corner_points[sextuple_cell_b_p1]) <= tolerance) and
         (vec.manhatten_distance(corner_points[sextuple_cell_a_p2], corner_points[sextuple_cell_b_p2]) <= tolerance)):
         return True
      if (
         (vec.manhatten_distance(corner_points[sextuple_cell_a_p1], corner_points[sextuple_cell_b_p2]) <= tolerance) and
         (vec.manhatten_distance(corner_points[sextuple_cell_a_p2], corner_points[sextuple_cell_b_p1]) <= tolerance)):
         return True
      return False

   assert (corner_points.ndim == 7 and corner_points.shape[:2] == (1, 1))

   confirmation = 3  # number of identical results needed for each of NI and NJ
   max_failures = 100  # maximum number of failed random cells for each of NI and NJ
   min_cell_length = 10.0 * tolerance

   cell_count = corner_points.shape[2]
   prime_factorization = factors.factorize(cell_count)
   log.debug('cell count is ' + str(cell_count) + '; prime factorization: ' + str(prime_factorization))
   possible_extents = factors.all_factors_from_primes(prime_factorization)
   log.debug('possible extents are: ' + str(possible_extents))

   ni = None
   redundancy = confirmation
   remaining_attempts = max_failures
   while redundancy:
      kji_cell = random_cell(corner_points, tolerance = min_cell_length)
      found = False
      for e in possible_extents:
         candidate = kji_cell[2] + e
         if candidate >= cell_count:
            continue
         if neighbours(corner_points, (0, 0, kji_cell[2], 0, 1, 0), (0, 0, kji_cell[2], 0, 1, 1),
                       (0, 0, candidate, 0, 0, 0), (0, 0, candidate, 0, 0, 1), tolerance):
            if ni is not None and ni != e:
               log.error('inconsistent NI values of {} and {} determined from corner points'.format(ni, e))
               return None
            found = True
            ni = e
            redundancy -= 1
            break
      if not found:
         remaining_attempts -= 1
         if remaining_attempts <= 0:
            log.error('failed to determine NI from corner points (out of tries)')  # could assume NJ = 1 here
            return None

   log.info('NI determined from corner points to be ' + str(ni))

   if ni > 1:
      ni_prime_factors = factors.factorize(ni)
      factors.remove_subset(prime_factorization, ni_prime_factors)
      log.debug('remaining prime factors after accounting for NI are: ' + str(prime_factorization))
      possible_extents = factors.all_factors_from_primes(prime_factorization)
      log.debug('possible extents for NJ & NK are: ' + str(possible_extents))

   nj = None
   redundancy = confirmation
   remaining_attempts = max_failures
   while redundancy:
      kji_cell = random_cell(corner_points)
      found = False
      for e in possible_extents:
         candidate = kji_cell[2] + (e * ni)
         if candidate >= cell_count:
            continue
         if vec.manhatten_distance(corner_points[0, 0, kji_cell[2], 1, 0, 0], corner_points[0, 0, candidate, 0, 0,
                                                                                            0]) <= tolerance:
            if nj is not None and nj != e:
               log.error('inconsistent NJ values of {} and {} determined from corner points'.format(nj, e))
               return None
            found = True
            nj = e
            redundancy -= 1
            break
      if not found:
         remaining_attempts -= 1
         if remaining_attempts <= 0:
            log.error(
               'failed to determine NJ from corner points (out of tries)')  # could assume or check if NK = 1 here
            return None

   log.info('NJ determined from corner points to be ' + str(nj))

   nk, remainder = divmod(cell_count, ni * nj)
   assert (remainder == 0)
   log.info('NK determined from corner points to be ' + str(nk))
   assert (nk in possible_extents)

   return [nk, nj, ni]


# end def determine_corp_extent():
##########################################################################################

##########################################################################################
# def translate_corp():


def translate_corp(corner_points, x_shift = None, y_shift = None, min_xy = None, preserve_digits = None):
   """Adjusts x and y values of corner points by a constant offset."""

   assert (corner_points.ndim == 7)
   if min_xy is None:
      minimum_xy = 0.0
   else:
      minimum_xy = min_xy
   if x_shift is None:
      x_sub = np.min(corner_points[:, :, :, :, :, :, 0]) - minimum_xy
   else:
      x_sub = -x_shift
   if y_shift is None:
      y_sub = np.min(corner_points[:, :, :, :, :, :, 1]) - minimum_xy
   else:
      y_sub = -y_shift
   if preserve_digits is not None:
      divisor = maths.pow(10.0, preserve_digits)
      x_sub = divisor * maths.floor(x_sub / divisor)
      y_sub = divisor * maths.floor(y_sub / divisor)

   log.info('translating corner points by %3.1f in x and %3.1f in y', -x_sub, -y_sub)
   corner_points[:, :, :, :, :, :, 0] -= x_sub
   corner_points[:, :, :, :, :, :, 1] -= y_sub


# end of def translate_corp()
##########################################################################################


def triangles_for_cell_faces(cp):
   """Returns numpy array of shape (3, 2, 4, 3, 3) with axes being kji, -+, triangle within face, triangle corner, xyz.

   args:
      cp (numpy float array of shape (2, 2, 2, 3)): single cell corner point array in pagoda protocol

   returns:
      numpy float array of shape (3, 2, 4, 3, 3) holding triangle corner coordinates for cell faces represented with
      quad triangles

   note:
      resqpy.surface also contains methods for working with cell faces as triangulated sets
   """

   tri = np.empty((3, 2, 4, 3, 3))

   # create face centre points and assign as one vertex in each of 4 trangles for face
   tri[0, :, :, 0] = np.mean(cp, axis = (1, 2)).reshape((2, 1, 3)).repeat(4, axis = 1).reshape(
      (2, 4, 3))  # k face centres
   tri[1, :, :, 0] = np.mean(cp, axis = (0, 2)).reshape((2, 1, 3)).repeat(4, axis = 1).reshape(
      (2, 4, 3))  # j face centres
   tri[2, :, :, 0] = np.mean(cp, axis = (0, 1)).reshape((2, 1, 3)).repeat(4, axis = 1).reshape(
      (2, 4, 3))  # i face centres

   # k faces
   tri[0, :, 0, 1] = cp[:, 0, 0]
   tri[0, :, 0, 2] = cp[:, 0, 1]
   tri[0, :, 1, 1] = cp[:, 0, 1]
   tri[0, :, 1, 2] = cp[:, 1, 1]
   tri[0, :, 2, 1] = cp[:, 1, 1]
   tri[0, :, 2, 2] = cp[:, 1, 0]
   tri[0, :, 3, 1] = cp[:, 1, 0]
   tri[0, :, 3, 2] = cp[:, 0, 0]

   # j faces
   tri[1, :, 0, 1] = cp[0, :, 0]
   tri[1, :, 0, 2] = cp[0, :, 1]
   tri[1, :, 1, 1] = cp[0, :, 1]
   tri[1, :, 1, 2] = cp[1, :, 1]
   tri[1, :, 2, 1] = cp[1, :, 1]
   tri[1, :, 2, 2] = cp[1, :, 0]
   tri[1, :, 3, 1] = cp[1, :, 0]
   tri[1, :, 3, 2] = cp[0, :, 0]

   # i faces
   tri[2, :, 0, 1] = cp[0, 0, :]
   tri[2, :, 0, 2] = cp[0, 1, :]
   tri[2, :, 1, 1] = cp[0, 1, :]
   tri[2, :, 1, 2] = cp[1, 1, :]
   tri[2, :, 2, 1] = cp[1, 1, :]
   tri[2, :, 2, 2] = cp[1, 0, :]
   tri[2, :, 3, 1] = cp[1, 0, :]
   tri[2, :, 3, 2] = cp[0, 0, :]

   return tri


# end of grid_functions module
##########################################################################################


def actual_pillar_shape(pillar_points, tolerance = 0.001):
   """Returns 'curved', 'straight' or 'vertical' for shape of fully defined points array of shape (nk + k_gaps + 1, ..., 3)."""

   assert pillar_points.ndim >= 3 and pillar_points.shape[-1] == 3

   pp = pillar_points.reshape((pillar_points.shape[0], -1, 3))

   from_top = pp - pp[0]
   xy_drift = np.abs(from_top[:, :, 0]) + np.abs(from_top[:, :,
                                                          1])  # use Manhattan distance as cheap proxy for true distance
   if np.max(xy_drift) <= tolerance:
      return 'vertical'
   if np.max(xy_drift[-1]) <= tolerance:
      return 'curved'  # top & bottom are vertically aligned, so pillar must be curved

   # where z variation is tiny (null pillar), don't interpolate, just treat these pillars as straight
   # elsewhere find drift from vector defined by from_top[-1]
   null_pillar_mask = (abs(from_top[-1, :, 2]) <= tolerance)
   from_top[-1, :, 2] = np.where(null_pillar_mask, tolerance, from_top[-1, :, 2])  # avoid divide by zero issues
   z_fraction = from_top[:, :, 2] / from_top[-1, :, 2]
   xy_drift = from_top[:, :, :2] - z_fraction.reshape((pp.shape[0], pp.shape[1], 1)) * from_top[-1, :, :2].reshape(
      (1, pp.shape[1], 2))
   straight = (np.max(np.sum(np.abs(xy_drift), axis = -1), axis = 0) <= tolerance)
   masked_straight = np.where(null_pillar_mask, True, straight)
   if np.all(masked_straight):
      return 'straight'
   return 'curved'


##########################################################################################


def columns_to_nearest_split_face(grid):
   """Returns a numpy integer array of shape (NJ, NI) being number of cells to nearest split edge (Manhattan distance)."""

   if not grid.has_split_coordinate_lines:
      return None

   j_col_faces_split, i_col_faces_split = grid.split_column_faces()
   abutting = np.zeros((grid.nj, grid.ni), dtype = bool)
   abutting[:-1, :] = j_col_faces_split
   abutting[1:, :] = np.logical_or(abutting[1:, :], j_col_faces_split)
   abutting[:, :-1] = np.logical_or(abutting[:, :-1], i_col_faces_split)
   abutting[:, 1:] = np.logical_or(abutting[:, 1:], i_col_faces_split)
   framed = np.full((grid.nj + 2, grid.ni + 2), grid.nj + grid.ni, dtype = int)
   framed[1:-1, 1:-1] = np.where(abutting, 0, grid.nj + grid.ni)

   while True:
      plus_one = framed + 1
      updated = np.minimum(framed[1:-1, 1:-1], plus_one[:-2, 1:-1])
      updated[:] = np.minimum(updated, plus_one[2:, 1:-1])
      updated[:] = np.minimum(updated, plus_one[1:-1, :-2])
      updated[:] = np.minimum(updated, plus_one[1:-1, 2:])
      if np.all(updated == framed[1:-1, 1:-1]):
         break
      framed[1:-1, 1:-1] = updated

   return framed[1:-1, 1:-1]


##########################################################################################


def left_right_foursome(full_pillar_list, p_index):
   """Returns (2, 2) bool numpy array indicating which columns around a primary pillar are to the right of a line."""

   assert 0 < p_index < len(full_pillar_list) - 1
   here = np.array(full_pillar_list[p_index], dtype = int)
   previous = np.array(full_pillar_list[p_index - 1], dtype = int)
   next = np.array(full_pillar_list[p_index + 1], dtype = int)
   entry = tuple(here - previous)
   exit = tuple(next - here)
   if entry == (0, 1):
      if exit == (-1, 0):
         return np.array([[False, True], [True, True]], dtype = bool)
      elif exit == (0, 1):
         return np.array([[False, False], [True, True]], dtype = bool)
      elif exit == (1, 0):
         return np.array([[False, False], [True, False]], dtype = bool)
      else:
         raise Exception('code failure whilst taking exit sides from dubious full pillar list')
   elif entry == (0, -1):
      if exit == (-1, 0):
         return np.array([[False, True], [False, False]], dtype = bool)
      elif exit == (0, -1):
         return np.array([[True, True], [False, False]], dtype = bool)
      elif exit == (1, 0):
         return np.array([[True, True], [True, False]], dtype = bool)
      else:
         raise Exception('code failure whilst taking exit sides from dubious full pillar list')
   elif entry == (1, 0):
      if exit == (0, -1):
         return np.array([[True, False], [False, False]], dtype = bool)
      elif exit == (1, 0):
         return np.array([[True, False], [True, False]], dtype = bool)
      elif exit == (0, 1):
         return np.array([[True, False], [True, True]], dtype = bool)
      else:
         raise Exception('code failure whilst taking exit sides from dubious full pillar list')
   elif entry == (-1, 0):
      if exit == (0, -1):
         return np.array([[True, True], [False, True]], dtype = bool)
      elif exit == (-1, 0):
         return np.array([[False, True], [False, True]], dtype = bool)
      elif exit == (0, 1):
         return np.array([[False, False], [False, True]], dtype = bool)
      else:
         raise Exception('code failure whilst taking exit sides from dubious full pillar list')
   else:
      log.debug(f'entry pair: {entry}')
      raise Exception('code failure whilst taking entry sides from dubious full pillar list')


##########################################################################################
