# test module for resqpy.olio.transmission.py

import math as maths
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.transmission as rqtr
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec


def test_regular_grid_half_cell_transmission(tmp_path):

    def try_one_half_t_regular(model,
                               extent_kji = (2, 2, 2),
                               dxyz = (1.0, 1.0, 1.0),
                               perm_kji = (1.0, 1.0, 1.0),
                               ntg = 1.0,
                               darcy_constant = 1.0,
                               rotate = None,
                               dip = None):
        ones = np.ones(extent_kji)
        grid = grr.RegularGrid(model, extent_kji = extent_kji, dxyz = dxyz)
        if dip is not None:  # dip positive x axis downwards
            r_matrix = vec.rotation_matrix_3d_axial(1, dip)
            p = grid.points_ref(masked = False)
            p[:] = vec.rotate_array(r_matrix, p)
        if rotate is not None:  # rotate anticlockwise in xy plane (viewed from above)
            r_matrix = vec.rotation_matrix_3d_axial(2, rotate)
            p = grid.points_ref(masked = False)
            p[:] = vec.rotate_array(r_matrix, p)
        half_t = rqtr.half_cell_t(grid,
                                  perm_k = perm_kji[0] * ones,
                                  perm_j = perm_kji[1] * ones,
                                  perm_i = perm_kji[2] * ones,
                                  ntg = ntg * ones,
                                  darcy_constant = darcy_constant)
        expected = 2.0 * darcy_constant * np.array(
            (perm_kji[0] * dxyz[0] * dxyz[1] / dxyz[2], ntg * perm_kji[1] * dxyz[0] * dxyz[2] / dxyz[1],
             ntg * perm_kji[2] * dxyz[1] * dxyz[2] / dxyz[0]))
        assert np.all(np.isclose(half_t, expected.reshape(1, 1, 1, 3)))

    temp_epc = str(os.path.join(tmp_path, f"{bu.new_uuid()}.epc"))
    model = rq.Model(temp_epc, new_epc = True, create_basics = True)

    try_one_half_t_regular(model)
    try_one_half_t_regular(model, extent_kji = (3, 4, 5))
    try_one_half_t_regular(model, dxyz = (127.53, 21.05, 12.6452))
    try_one_half_t_regular(model, perm_kji = (123.23, 512.4, 314.7))
    try_one_half_t_regular(model, ntg = 0.7)
    try_one_half_t_regular(model, darcy_constant = 0.001127)
    try_one_half_t_regular(model,
                           extent_kji = (5, 4, 3),
                           dxyz = (84.23, 77.31, 15.823),
                           perm_kji = (0.6732, 298.14, 384.2),
                           ntg = 0.32,
                           darcy_constant = 0.008527)
    try_one_half_t_regular(model, rotate = 67.8)

    # should not have written anything, but try clean-up just in case
    try:
        os.remove(temp_epc)
    except Exception:
        pass
    try:
        os.remove(temp_epc[-4] + '.h5')
    except Exception:
        pass


def test_half_cell_t_2d_triangular_precursor_equilateral():
    side = 123.45
    height = side * maths.sin(vec.radians_from_degrees(60.0))
    # create a pair of equilateral triangles
    p = np.array([(0.0, 0.0), (side, 0.0), (side / 2.0, height), (side * 3.0 / 2.0, height)])
    t = np.array([(0, 1, 2), (1, 2, 3)], dtype = int)
    a, b = rqtr.half_cell_t_2d_triangular_precursor(p, t)
    a_over_l = a / b
    expected = side / (height / 3)
    assert a_over_l.shape == (2, 3)
    assert_array_almost_equal(a_over_l, expected)
