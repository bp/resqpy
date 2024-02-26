import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.surface as rqs
import resqpy.surface._tri_mesh_stencil as tms


def test_gaussian_pattern():
    pattern = tms._gaussian_pattern(3, 2.0)
    assert_array_almost_equal(pattern, (1.0, 0.60653066, 0.13533528))
    pattern = tms._gaussian_pattern(2, 1.1774100225)
    assert_array_almost_equal(pattern, (1.0, 0.5))
    pattern = tms._gaussian_pattern(2, 2.1459660263)
    assert_array_almost_equal(pattern, (1.0, 0.1))
    pattern = tms._gaussian_pattern(10, 2.1459660263)
    assert_array_almost_equal(
        pattern,
        (1.0, 0.97197327, 0.89251862, 0.77426368, 0.6345548, 0.49131274, 0.35938137, 0.24834861, 0.16213491, 0.1))
    pattern = tms._gaussian_pattern(15, 5.0)
    assert_array_almost_equal(
        pattern,
        (1.0, 9.38215596e-1, 7.74837429e-1, 5.63279351e-1, 3.60447789e-1, 2.03032796e-1, 1.00668900e-1, 4.39369336e-2,
         1.68798841e-2, 5.70840102e-3, 1.69927937e-3, 4.45266876e-4, 1.02702565e-4, 2.08519889e-5, 3.72665317e-6))


def test_tri_mesh_stencil_init_normalize_none():
    stencil = rqs.TriMeshStencil((3.0, 5.0, 1.0), normalize = None)
    assert stencil is not None
    assert isinstance(stencil, rqs.TriMeshStencil)
    assert stencil.n == 3
    assert_array_almost_equal(stencil.pattern, (3.0, 5.0, 1.0))
    assert np.all(stencil.start_ip == (-2, -2, -1))
    assert np.all(stencil.row_length == (5, 4, 3))
    assert_array_almost_equal(
        stencil.half_hex,
        np.array([(1.0, 5.0, 3.0, 5.0, 1.0), (1.0, 5.0, 5.0, 1.0, np.nan), (1.0, 1.0, 1.0, np.nan, np.nan)],
                 dtype = float))


def test_tri_mesh_stencil_init_normalize_flat():
    stencil = rqs.TriMeshStencil((3.0, 5.0, 1.0), normalize = 100.0, normalize_mode_flat = True)
    assert stencil.n == 3
    assert_array_almost_equal(stencil.pattern, np.array((60.0, 100.0, 20.0), dtype = float) / 9.0)
    assert np.all(stencil.start_ip == (-2, -2, -1))
    assert np.all(stencil.row_length == (5, 4, 3))
    assert_array_almost_equal(
        stencil.half_hex,
        np.array([(20.0, 100.0, 60.0, 100.0, 20.0), (20.0, 100.0, 100.0, 20.0, np.nan),
                  (20.0, 20.0, 20.0, np.nan, np.nan)],
                 dtype = float) / 9.0)


def test_tri_mesh_stencil_init_normalize_non_flat():
    stencil = rqs.TriMeshStencil((3.0, 5.0, 1.0), normalize = 100.0, normalize_mode_flat = False)
    assert stencil.n == 3
    assert_array_almost_equal(stencil.pattern, np.array((900.0, 250.0, 25.0), dtype = float) / 27.0)
    assert np.all(stencil.start_ip == (-2, -2, -1))
    assert np.all(stencil.row_length == (5, 4, 3))
    assert_array_almost_equal(
        stencil.half_hex,
        np.array([(25.0, 250.0, 900.0, 250.0, 25.0), (25.0, 250.0, 250.0, 25.0, np.nan),
                  (25.0, 25.0, 25.0, np.nan, np.nan)],
                 dtype = float) / 27.0)
