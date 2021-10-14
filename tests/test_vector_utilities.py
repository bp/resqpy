# test vector utilities

import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.vector_utilities as vec


def test_simple_functions():
   assert_array_almost_equal(vec.zero_vector(), (0.0, 0.0, 0.0))
   assert_array_almost_equal(vec.add((1.2, 3.4), (5.6, 7.8)), np.array([6.8, 11.2]))
   assert_array_almost_equal(vec.subtract((5.6, 7.8, 10.0), (1.2, 3.4, -1.0)), np.array([4.4, 4.4, 11.0]))
   assert_array_almost_equal(vec.elemental_multiply((1.2, 3.4), (5.6, 7.8)), (6.72, 26.52))
   assert_array_almost_equal(vec.amplify((1.0, 2.0, 4.5), 2.5), (2.5, 5.0, 11.25))
   assert_array_almost_equal(vec.v_3d((23.4, -98.7)), (23.4, -98.7, 0.0))
   assert_array_almost_equal(vec.v_3d((23.4, 45.4, -98.7)), np.array((23.4, 45.4, -98.7)))


def test_unit_vectors():
   v_set = np.array([(3.0, 4.0, 0.0), (3.7, -3.7, 3.7), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])
   one_over_root_three = 1.0 / maths.sqrt(3.0)
   expected = np.array([(3.0 / 5.0, 4.0 / 5.0, 0.0), (one_over_root_three, -one_over_root_three, one_over_root_three),
                        (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])
   for v, e in zip(v_set, expected):
      assert_array_almost_equal(vec.unit_vector(v), e)
   assert_array_almost_equal(vec.unit_vectors(v_set), expected)
   azi = [0.0, -90.0, 120.0, 180.0, 270.0, 360.0]
   expected = np.array([(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (maths.cos(maths.pi / 6), -0.5, 0.0), (0.0, -1.0, 0.0),
                        (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
   for a, e in zip(azi, expected):
      assert_array_almost_equal(vec.unit_vector_from_azimuth(a), e)


def test_angles():
   pi_by_2 = maths.pi / 2.0
   assert maths.isclose(vec.radians_from_degrees(90.0), pi_by_2)
   assert_array_almost_equal(vec.radians_from_degrees((0.0, -90.0, 120.0, 270.0)),
                             (0.0, -pi_by_2, maths.pi * 2 / 3, pi_by_2 * 3))
   assert maths.isclose(vec.radians_from_degrees(vec.degrees_from_radians(1.2647)), 1.2647)
   assert_array_almost_equal(vec.degrees_from_radians((0.0, -pi_by_2, maths.pi * 2 / 3, pi_by_2 * 3)),
                             (0.0, -90.0, 120.0, 270.0))
   assert maths.isclose(vec.degrees_from_radians(maths.pi), 180.0)


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
