import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.triangulation as tri


def test_ccc():
   p1 = np.array((0.0, 0.0, 0.0))
   p2 = np.array((20.0, 0.0, 0.0))
   p3 = np.array((0.0, 10.0, 5.0))
   assert_array_almost_equal(tri.ccc(p1, p2, p3), (10.0, 5.0))
