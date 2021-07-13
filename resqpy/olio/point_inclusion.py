"""point_inclusion.py: functions to test whether a point is within a polygon; also line intersection with planes."""

version = '29th April 2021'

import logging

log = logging.getLogger(__name__)
log.debug('point_inclusion.py version ' + version)

import os
import math as maths
import numpy as np

import resqpy.olio.simple_lines as sl


def pip_cn(p, poly):
   """2D point inclusion: returns True if point is inside polygon (uses crossing number algorithm)"""

   # p should be a tuple-like object with first two elements x, y
   # poly should be a numpy array with first axis being the different vertices
   #    and the second starting x, y

   crossings = 0
   vertex_count = poly.shape[0]

   for edge in range(vertex_count):
      v1 = poly[edge]
      if edge == vertex_count - 1:
         v2 = poly[0]
      else:
         v2 = poly[edge + 1]
      if ((v1[0] > p[0] or v2[0] > p[0]) and ((v1[1] <= p[1] and v2[1] > p[1]) or (v1[1] > p[1] and v2[1] <= p[1]))):
         if v1[0] > p[0] and v2[0] > p[0]:
            crossings += 1
         elif p[0] < v1[0] + (v2[0] - v1[0]) * (p[1] - v1[1]) / (v2[1] - v1[1]):
            crossings += 1

   return bool(crossings & 1)


def pip_wn(p, poly):
   """2D point inclusion: returns True if point is inside polygon (uses winding number algorithm)"""

   winding = 0
   vertex_count = poly.shape[0]

   for edge in range(vertex_count):
      v1 = poly[edge]
      if edge == vertex_count - 1:
         v2 = poly[0]
      else:
         v2 = poly[edge + 1]
      if v1[0] <= p[0] and v2[0] <= p[0]:
         continue
      if v1[1] <= p[1] and v2[1] > p[1]:
         if v1[0] > p[0] and v2[0] > p[0]:
            winding += 1
         elif p[0] < v1[0] + (v2[0] - v1[0]) * (p[1] - v1[1]) / (v2[1] - v1[1]):
            winding += 1
      elif v1[1] > p[1] and v2[1] <= p[1]:
         if v1[0] > p[0] and v2[0] > p[0]:
            winding -= 1
         elif p[0] < v1[0] + (v2[0] - v1[0]) * (p[1] - v1[1]) / (v2[1] - v1[1]):
            winding -= 1

   return bool(winding)


def pip_array_cn(p_a, poly):
   """array of 2D points inclusion: returns bool array True where point is inside polygon (uses crossing number algorithm)"""

   # p_array should be a numpy array of 2 or more axes; the final axis has extent at least 2, being x, y, ...
   # returned boolean array has shape of p_array less the final axis

   log.debug('type(poly): ' + str(type(poly)))
   elements = np.prod(list(p_a.shape)[:-1], dtype = int)
   log.debug('elements: ' + str(elements))
   p = p_a.reshape((elements, -1))
   log.debug('p.shape: ' + str(p.shape))
   crossings = np.zeros((elements,), dtype = int)
   vertex_count = poly.shape[0]
   log.debug('vertex_count: ' + str(vertex_count))

   np_err_dict = np.geterr()
   np.seterr(divide = 'ignore', invalid = 'ignore')
   for edge in range(vertex_count):
      v1 = poly[edge]
      if edge == vertex_count - 1:
         v2 = poly[0]
      else:
         v2 = poly[edge + 1]
      crossings += np.where(
         np.logical_and(
            np.logical_and(
               np.logical_or(v1[0] > p[:, 0], v2[0] > p[:, 0]),
               np.logical_or(np.logical_and(v1[1] <= p[:, 1], v2[1] > p[:, 1]),
                             np.logical_and(v1[1] > p[:, 1], v2[1] <= p[:, 1]))),
            np.logical_or(np.logical_and(v1[0] > p[:, 0], v2[0] > p[:, 0]),
                          (p[:, 0] < (v1[0] + (v2[0] - v1[0]) * (p[:, 1] - v1[1]) / (v2[1] - v1[1]))))), 1, 0)
   if 'divide' in np_err_dict:
      np.seterr(divide = np_err_dict['divide'])
   if 'invalid' in np_err_dict:
      np.seterr(invalid = np_err_dict['invalid'])

   return np.array(np.bitwise_and(crossings, 1), dtype = bool).reshape(list(p_a.shape)[:-1])


def points_in_polygon(x, y, polygon_file, poly_unit_multiplier = None):
   """Takes a pair of numpy arrays x, y defining points to be tested against polygon specified in polygon_file."""

   assert x.shape == y.shape, 'x and y arrays have differing shapes or are not numpy arrays'
   assert polygon_file and os.path.exists(polygon_file), 'polygon file is missing'
   if poly_unit_multiplier is not None:
      assert poly_unit_multiplier != 0.0, 'zero multiplier for polygon units not allowed'

   try:
      polygon_list = sl.read_lines(polygon_file)
      assert len(polygon_list) > 0, 'unable to read polygon from file ' + polygon_file
      assert len(polygon_list) == 1, 'more than one polygon in file ' + polygon_file
      polygon = polygon_list[0]
      if poly_unit_multiplier is not None:
         polygon[:] *= poly_unit_multiplier
      points = np.stack((x, y), axis = -1)
      return pip_array_cn(points, polygon)
   except Exception:
      log.exception('failed to determine points in polygon')

   return None


def scan(origin, ncol, nrow, dx, dy, poly):
   """Scans lines of pixels returning 2D boolean array of points within convex polygon."""

   vertex_count = poly.shape[0]
   inside = np.zeros((nrow, ncol), dtype = bool)
   ix = np.empty((2))

   for row in range(nrow):
      ix[:] = 0.0
      y = origin[1] + dy * row
      ic = 0
      for edge in range(vertex_count):
         v1 = poly[edge]
         if edge == vertex_count - 1:
            v2 = poly[0]
         else:
            v2 = poly[edge + 1]
         if (v1[1] > y and v2[1] > y) or (v1[1] <= y and v2[1] <= y):
            continue
         ix[ic] = v1[0] + (v2[0] - v1[0]) * (y - v1[1]) / (v2[1] - v1[1])
         ic += 1
         if ic == 2:
            break
      if ic < 2:
         continue
      if ix[1] < ix[0]:
         sx, ex = ix[1], ix[0]
      else:
         sx, ex = ix[0], ix[1]
      scol = maths.ceil((sx - origin[0]) / dx)
      ecol = int((ex - origin[0]) / dx)
      inside[row, scol:ecol + 1] = True

   return inside
