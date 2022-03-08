# test intersection functions

import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.intersection as meet


def test_line_triangle_intersect():
    line_p = np.array((2.0, 0.0, 3.0))
    line_v = np.array((0.0, 10.0, 0.0))
    tp = np.array([(2.0, 5.0, 2.0), (2.0, 6.0, 4.0), (1.0, 5.7, 3.2)])

    xyz = meet.line_triangle_intersect(line_p, line_v, tp, line_segment = True)
    assert xyz is not None
    expected = np.array((2.0, 5.5, 3.0))
    assert_array_almost_equal(xyz, expected)

    line_p[0] += 0.01
    xyz = meet.line_triangle_intersect(line_p, line_v, tp, line_segment = True)
    assert xyz is None

    xyz = meet.line_triangle_intersect(line_p, line_v, tp, line_segment = True, t_tol = 0.1)
    assert xyz is not None


def test_point_projected_to_line_2d():
    l1, l2 = np.array((-1.0, -1.0)), np.array((9.9, 9.9))
    for p in [(21431.98, -3145.3), (12.56, 12.56), (5.0, 6.0), (0.0, 2347856.0)]:
        x, y = meet.point_projected_to_line_2d(np.array(p), l1, l2)
        assert maths.isclose(x, y)
    p = (20.0, 0.0)
    assert_array_almost_equal(meet.point_projected_to_line_2d(np.array(p), l1, l2), (10.0, 10.0))


def test_point_snapped_to_line_segment_2d():
    l1, l2 = np.array((-1.0, -1.0)), np.array((9.9, 9.9))
    p = (-1.0, -1.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (-1.0, -1.0))
    p = (-2.0, -2.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (-1.0, -1.0))
    p = (-3425423.0, -1.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (-1.0, -1.0))
    p = (-2.0, -5.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (-1.0, -1.0))
    p = (10.0, 10.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (9.9, 9.9))
    p = (20.0, 0.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (9.9, 9.9))
    p = (10.0, 0.0)
    assert_array_almost_equal(meet.point_snapped_to_line_segment_2d(np.array(p), l1, l2), (5.0, 5.0))
