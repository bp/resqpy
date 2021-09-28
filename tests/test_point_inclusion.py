# test point in polygon functions

# import math as maths
import numpy as np
# from numpy.testing import assert_array_almost_equal

import resqpy.olio.point_inclusion as pip


def test_pip_cn_and_wn():
   # unit square polygon
   poly = np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
   p_in = np.array([(0.00001, 0.00001), (0.00001, 0.99999), (0.99999, 0.00001), (0.99999, 0.99999)])
   p_out = np.array([(1.1, 0.1), (-0.1, 0.2), (0.5, 1.00001), (0.4, -0.0001), (1.00001, 1.00001), (1.00001, -0.00001)])
   for pip_fn in [pip.pip_cn, pip.pip_wn]:
      assert pip_fn((0.5, 0.5), poly)
      for p in p_in:
         assert pip_fn(p, poly)
      for p in p_out:
         assert not pip_fn(p, poly)
   assert np.all(pip.pip_array_cn(p_in, poly))
   assert not np.any(pip.pip_array_cn(p_out, poly))
