# test intersection functions

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
