# test point in polygon functions

import os

# import math as maths
import numpy as np

import resqpy.olio.point_inclusion as pip

# from numpy.testing import assert_array_almost_equal


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


def test_figure_of_eight():
    fig_8 = np.array([(-100.0, -200.0), (100.0, 200.0), (-100.0, 200.0), (100.0, -200.0)])
    p_in = np.array([(-99.0, -199.0), (0.0, -1.0), (99.0, 199.0), (0.0, 1.0), (49.9, -100.0)])
    p_out = np.array([(1000.0, -23.0), (1.0, 0.0), (-0.001, 0.0), (-50.1, 100.0)])
    for pip_fn in (pip.pip_cn, pip.pip_wn):
        for p in p_in:
            assert pip_fn(p, fig_8)
        for p in p_out:
            assert not pip_fn(p, fig_8)
    assert np.all(pip.pip_array_cn(p_in, fig_8))
    assert not np.any(pip.pip_array_cn(p_out, fig_8))


def test_points_in_polygon(tmp_path):
    # create an ascii file holding vertices of a polygon
    poly_file = os.path.join(tmp_path, 'diamond.txt')
    diamond = np.array([(0.0, 3.0, 0.0), (3.0, 6.0, -1.3), (6.0, 3.0, 12.5), (3.0, 0.0, 0.0)])
    with open(poly_file, 'w') as fp:
        for xyz in diamond:
            fp.write(f'{xyz[0]} {xyz[1]} {xyz[2]}\n')
        fp.write('999.0 999.0 999.0\n')
    # test some points with no multiplier applied to polygon geometry
    p_in = np.array([(3.0, 3.0), (0.1, 3.0), (1.55, 1.55), (4.49, 4.49), (5.99, 3.0), (3.1, 0.11)])
    p_out = np.array([(-3.0, -3.0), (2.0, 0.99), (4.51, 4.51), (6.01, 3.0)])
    assert np.all(pip.points_in_polygon(p_in[:, 0], p_in[:, 1], poly_file))
    assert not np.any(pip.points_in_polygon(p_out[:, 0], p_out[:, 1], poly_file))
    # test some points with multiplier applied to polygon geometry
    multiplier = 2.0 / 3.0
    p_in *= multiplier
    p_out *= multiplier
    assert np.all(pip.points_in_polygon(p_in[:, 0], p_in[:, 1], poly_file, poly_unit_multiplier = multiplier))
    assert not np.any(pip.points_in_polygon(p_out[:, 0], p_out[:, 1], poly_file, poly_unit_multiplier = multiplier))


def test_scan():
    kite = np.array([(0.1, 3.0, 0.0), (3.0, 4.9, 0.0), (5.9, 3.0, 0.0), (3.0, 0.1, 0.0)])
    origin = (-1.0, 0.0)
    ncol = 8
    nrow = 6
    expected = np.array([(0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1, 0, 0),
                         (0, 0, 1, 1, 1, 1, 1, 0), (0, 0, 0, 1, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0)],
                        dtype = bool)
    assert np.all(pip.scan(origin, ncol, nrow, 1.0, 1.0, kite) == expected)
