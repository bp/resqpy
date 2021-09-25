import pytest
import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.vector_utilities as vec
import resqpy.olio.triangulation as tri


def test_ccc():
   # 3 points in orthogonal pattern
   p1 = np.array((0.0, 0.0, 0.0))
   p2 = np.array((20.0, 0.0, 0.0))
   p3 = np.array((0.0, 10.0, 5.0))
   assert_array_almost_equal(tri.ccc(p1, p2, p3), (10.0, 5.0))
   # equilateral triangle
   s = 23.57
   p1 = np.array((10.0, 20.0, 30.0))
   p2 = np.array((10.0, 20.0 + s, -45.0))
   p3 = np.array((10.0 + s * maths.cos(maths.radians(30.0)), 20.0 + 0.5 * s, 12.13))
   assert_array_almost_equal(tri.ccc(p1, p2, p3), np.mean(np.stack((p1, p2, p3)), axis = 0)[:2])
   # asymmetric triangle
   p1 = np.array((25.3, 12.1, 0.0))
   p2 = np.array((23.6, 2.9, -1.0))
   p3 = np.array((22.1, 87.3, 1.5))
   c = np.array(tri.ccc(p1, p2, p3))
   v = np.stack((c - p1[:2], c - p2[:2], c - p3[:2]))
   r = vec.naive_2d_lengths(v)
   assert_array_almost_equal(r[1:], r[0])
