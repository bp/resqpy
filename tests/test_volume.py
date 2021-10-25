# test hexahedral volume calculations

import math as maths

import numpy as np
from numpy.random import random
from numpy.testing import assert_array_almost_equal

import resqpy.olio.volume as vol


def unit_cube():
    cp = np.zeros((2, 2, 2, 3))  # pagoda style corner point array for a single cell
    # set to unit cube with xyz aligned with ijk respectively
    cp[:, :, 1, 0] = 1.0
    cp[:, 1, :, 1] = 1.0
    cp[1, :, :, 2] = 1.0
    return cp


def test_one_cell_volume():
    cp = unit_cube().copy()
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False), 1.0), 'unit cube cell volume calculation fails'
    # stretch in x (I)
    cp[:, :, 1, 0] *= 2.5
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False), 2.5), 'square prism cell volume calculation fails'
    # stretch in y & z (J & K)
    cp[:, 1, :, 1] *= 2.0
    cp[1, :, :, 2] *= 3.0
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False),
                         15.0), 'regular cuboid cell volume calculation fails'
    # apply a shear in x & y (should not change volume)
    cp[1, :, :, :2] += 0.5
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False),
                         15.0), 'parallelepiped cell volume calculation fails'
    # translate into negative coordinate realm
    cp[:] -= 3.827
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False), 15.0), 'translated cell volume calculation fails '
    # invert one axis
    cp[:, :, :, 1] *= -1.0
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = True),
                         15.0), 'off hand True cell volume calculation fails'
    # collapse to plane
    cp[:, :, :, 2] = 0.0
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False), 0.0), 'planar cell volume calculation fails'
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = True),
                         0.0), 'off hand True planar cell volume calculation fails'
    # test wedge
    cp = unit_cube().copy()
    cp[1, 1, :, 2] = 0.0
    assert maths.isclose(vol.tetra_cell_volume(cp, off_hand = False), 0.5), 'triangular prism volume calculation fails'


def test_array_volume():
    # note: the multiple cells generated here are all overlapping, which would not usually be the case
    cp = np.zeros((5, 10, 15, 2, 2, 2, 3))  # pagoda style corner point array
    cp[:] = unit_cube().reshape((1, 1, 1, 2, 2, 2, 3)) * 2.0
    cp[:] += random(cp.size).reshape(cp.shape) - 0.5
    # move whole cells by random translations
    translate = (random(cp.size // 8) * 123.2356).reshape((cp.shape[0], cp.shape[1], cp.shape[2], 1, 1, 1, 3))
    cp[:] += translate
    # compute volumes using two different routines in the volume module and make sure they give the same answer
    v1 = vol.tetra_volumes_slow(cp, off_hand = False)
    v2 = vol.tetra_volumes(cp, off_hand = False)
    # following is a stringent test requiring that exactly the same mathematical operations have been performed
    assert_array_almost_equal(v1, v2)

    # the random changes to corner points should have left the volumes within a certain range
    assert np.all(v1 >= 0.0), 'negative volume(s) returned by array function'
    assert np.all(v1 <= 27.0), 'exaggerated volume(s) returned by array function'
    # translate again and check that volumes are unchanged
    translate = (random(cp.size // 8) * 3.23478).reshape((cp.shape[0], cp.shape[1], cp.shape[2], 1, 1, 1, 3))
    cp[:] -= translate
    v2 = vol.tetra_volumes(cp, off_hand = False)
    assert_array_almost_equal(v1, v2)

    # test handedness inversion
    cp[:, :, :, :, :, :, 0] *= -1.0
    v1 = vol.tetra_volumes_slow(cp, off_hand = True)
    v2 = vol.tetra_volumes(cp, off_hand = True)
    assert_array_almost_equal(v1, v2)
    assert np.all(v1 >= 0.0), 'negative volume(s) returned by array function'
    assert np.all(v1 <= 27.0), 'exaggerated volume(s) returned by array function'


def test_pyramid_volume():
    apex = (1.0, 1.0, -3.0)
    a = (0.0, 0.0, 0.0)
    b = (0.0, 2.0, 0.0)
    c = (2.0, 2.0, 0.0)
    d = (2.0, 0.0, 0.0)
    assert maths.isclose(vol.pyramid_volume(apex, a, b, c, d), 4.0)
    assert maths.isclose(vol.pyramid_volume(apex, a, d, c, b, crs_is_right_handed = True), 4.0)


def test_tetrahedron_volume():
    one_over_root_two = 1.0 / maths.sqrt(2.0)
    a = (-1.0, 0.0, -one_over_root_two)
    b = (1.0, 0.0, -one_over_root_two)
    c = (0.0, -1.0, one_over_root_two)
    d = (0.0, 1.0, one_over_root_two)
    assert maths.isclose(vol.tetrahedron_volume(a, b, c, d), 4.0 / (3.0 * maths.sqrt(2.0)))
