# test vector utilities

import math as maths
import numpy as np
# from numpy.testing import assert_array_almost_equal

import resqpy.olio.vector_utilities as vec


def test_clockwise():
   p1 = (4.4, 4.4)
   p2 = (6.8, 6.8)
   p3 = (-21.0, -21.0)
   assert maths.isclose(vec.clockwise(p1, p2, p3), 0.0)
   p1 = [1.0, 1.0, -45.0]
   p2 = [2.0, 8.0, 23.1]
   p3 = [5.0, 3.0, -11.0]
   assert vec.clockwise(p1, p2, p3) > 0.0
   p1, p2 = np.array(p2), np.array(p1)
   p3 = np.array(p3)
   assert vec.clockwise(p1, p2, p3) < 0.0
