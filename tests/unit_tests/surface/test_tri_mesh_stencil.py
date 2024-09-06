import pytest

import os
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.surface as rqs
import resqpy.surface._tri_mesh_stencil as tms


def make_wavy_tri_mesh(model,
                       crs_uuid,
                       t_side = 5.0,
                       nj = 11,
                       ni = 10,
                       j_waves = 3.0,
                       i_waves = 2.0,
                       add_dip = True,
                       add_random = False,
                       random_amplitude = 9.0,
                       make_hole = False):
    z_in = np.zeros((nj, ni))
    if add_random:
        z_in += np.random.random((nj, ni)) * random_amplitude
    z_in[:] += 20.0 * np.expand_dims(
        np.sin(np.arange(nj, dtype = int).astype(float) * j_waves * 2.0 * np.pi / float(nj)), axis = 1)
    z_in[:] += 10.0 * np.expand_dims(
        np.sin(np.arange(ni, dtype = int).astype(float) * i_waves * 2.0 * np.pi / float(ni)), axis = 0)
    if add_dip:
        z_in[:] += np.expand_dims(np.arange(ni, dtype = int).astype(float), axis = 0)
    if make_hole:
        z_in[nj // 2:2 * nj // 3, ni // 3:ni // 2] = np.nan
    tm_in = rqs.TriMesh(model,
                        crs_uuid = crs_uuid,
                        t_side = t_side,
                        nj = nj,
                        ni = ni,
                        z_uom = 'm',
                        z_values = z_in,
                        title = 'jaggedy')
    tm_in.write_hdf5()
    tm_in.create_xml()
    return tm_in


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


def test_stencil_for_constant_unnormalized():
    stencil = rqs.TriMeshStencil.for_constant_unnormalized(4, 7.0)
    assert stencil.n == 4
    assert stencil.pattern.shape == (4,) and np.all(np.isclose(stencil.pattern, 7.0))
    assert np.all(stencil.start_ip == (-3, -3, -2, -2))
    assert np.all(stencil.row_length == (7, 6, 5, 4))
    assert np.isclose(np.nanmin(stencil.half_hex), 7.0)
    assert np.isclose(np.nanmax(stencil.half_hex), 7.0)


def test_stencil_for_constant_normalized_flat():
    stencil = rqs.TriMeshStencil.for_constant_normalized(5, normalize_mode_flat = True)
    assert stencil.n == 5
    assert np.all(np.isclose(stencil.pattern, 1.0 / 61))
    assert np.all(stencil.start_ip == (-4, -4, -3, -3, -2))
    assert np.all(stencil.row_length == (9, 8, 7, 6, 5))
    assert np.count_nonzero(np.isnan(stencil.half_hex)) == 10
    assert np.isclose(np.nanmin(stencil.half_hex), 1.0 / 61)
    assert np.isclose(np.nanmax(stencil.half_hex), 1.0 / 61)


def test_stencil_for_constant_normalized_non_flat():
    stencil = rqs.TriMeshStencil.for_constant_normalized(5, normalize_mode_flat = False)
    assert stencil.n == 5
    assert_array_almost_equal(stencil.pattern, (1.0 / 5, 1.0 / 30.0, 1.0 / 60, 1.0 / 90, 1.0 / 120))
    assert np.all(stencil.start_ip == (-4, -4, -3, -3, -2))
    assert np.all(stencil.row_length == (9, 8, 7, 6, 5))
    assert np.count_nonzero(np.isnan(stencil.half_hex)) == 10
    assert np.isclose(np.nanmin(stencil.half_hex), 1.0 / 120)
    assert np.isclose(np.nanmax(stencil.half_hex), 1.0 / 5)


def test_stencil_for_linear_unnormalized():
    stencil = rqs.TriMeshStencil.for_linear_unnormalized(3, 5.0, 1.0)
    assert stencil.n == 3
    assert_array_almost_equal(stencil.pattern, (5.0, 3.0, 1.0))
    assert np.all(stencil.start_ip == (-2, -2, -1))
    assert np.all(stencil.row_length == (5, 4, 3))
    assert_array_almost_equal(
        stencil.half_hex, np.array([(1, 3, 5, 3, 1), (1, 3, 3, 1, np.nan), (1, 1, 1, np.nan, np.nan)], dtype = float))


def test_stencil_for_linear_normalized_flat():
    stencil = rqs.TriMeshStencil.for_linear_normalized(4, normalize_mode_flat = True)
    assert stencil.n == 4
    assert_array_almost_equal(stencil.pattern, np.array((4, 3, 2, 1), dtype = float) / 64.0)
    assert np.all(stencil.start_ip == (-3, -3, -2, -2))
    assert np.all(stencil.row_length == (7, 6, 5, 4))
    assert_array_almost_equal(
        stencil.half_hex, 3.0 / (16.0 * np.array([(12, 6, 4, 3, 4, 6, 12), (12, 6, 4, 4, 6, 12, np.nan),
                                                  (12, 6, 6, 6, 12, np.nan, np.nan),
                                                  (12, 12, 12, 12, np.nan, np.nan, np.nan)],
                                                 dtype = float)))


def test_stencil_for_linear_normalized_non_flat():
    stencil = rqs.TriMeshStencil.for_linear_normalized(4, normalize_mode_flat = False)
    assert stencil.n == 4
    assert_array_almost_equal(stencil.pattern, 2.0 / np.array((5, 40, 120, 360), dtype = float))
    assert np.all(stencil.start_ip == (-3, -3, -2, -2))
    assert np.all(stencil.row_length == (7, 6, 5, 4))
    assert_array_almost_equal(
        stencil.half_hex, 2.0 / np.array([(360, 120, 40, 5, 40, 120, 360), (360, 120, 40, 40, 120, 360, np.nan),
                                          (360, 120, 120, 120, 360, np.nan, np.nan),
                                          (360, 360, 360, 360, np.nan, np.nan, np.nan)],
                                         dtype = float))


def test_stancil_for_gaussian_unnormalized():
    stencil = rqs.TriMeshStencil.for_gaussian_unnormalized(5, 10.0, sigma = 3.0)
    assert stencil.n == 5
    assert np.all(stencil.start_ip == (-4, -4, -3, -3, -2))
    assert np.all(stencil.row_length == (9, 8, 7, 6, 5))
    assert_array_almost_equal(stencil.pattern,
                              np.array((10.0, 7.54839602, 3.24652467, 0.79559509, 0.11108997), dtype = float))
    assert_array_almost_equal(
        stencil.half_hex,
        np.array(
            [(0.11108997, 0.79559509, 3.24652467, 7.54839602, 10.0, 7.54839602, 3.24652467, 0.79559509, 0.11108997),
             (0.11108997, 0.79559509, 3.24652467, 7.54839602, 7.54839602, 3.24652467, 0.79559509, 0.11108997, np.nan),
             (0.11108997, 0.79559509, 3.24652467, 3.24652467, 3.24652467, 0.79559509, 0.11108997, np.nan, np.nan),
             (0.11108997, 0.79559509, 0.79559509, 0.79559509, 0.79559509, 0.11108997, np.nan, np.nan, np.nan),
             (0.11108997, 0.11108997, 0.11108997, 0.11108997, 0.11108997, np.nan, np.nan, np.nan, np.nan)],
            dtype = float))


def test_stancil_for_gaussian_normalized_flat():
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(4, sigma = 2.3, normalize_mode_flat = True)
    assert stencil.n == 4
    assert np.all(stencil.start_ip == (-3, -3, -2, -2))
    assert np.all(stencil.row_length == (7, 6, 5, 4))
    assert_array_almost_equal(stencil.pattern, np.array((0.09565697, 0.07129881, 0.02952428, 0.00679216),
                                                        dtype = float))
    assert_array_almost_equal(
        stencil.half_hex,
        np.array([(0.00679216, 0.02952428, 0.07129881, 0.09565697, 0.07129881, 0.02952428, 0.00679216),
                  (0.00679216, 0.02952428, 0.07129881, 0.07129881, 0.02952428, 0.00679216, np.nan),
                  (0.00679216, 0.02952428, 0.02952428, 0.02952428, 0.00679216, np.nan, np.nan),
                  (0.00679216, 0.00679216, 0.00679216, 0.00679216, np.nan, np.nan, np.nan)],
                 dtype = float))


def test_stancil_for_gaussian_normalized_non_flat():
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(4, sigma = 2.3, normalize_mode_flat = False)
    assert stencil.n == 4
    assert np.all(stencil.start_ip == (-3, -3, -2, -2))
    assert np.all(stencil.row_length == (7, 6, 5, 4))
    assert_array_almost_equal(stencil.pattern, np.array((0.47058555, 0.05845922, 0.01210375, 0.00185634),
                                                        dtype = float))
    assert_array_almost_equal(
        stencil.half_hex,
        np.array([(0.00185634, 0.01210375, 0.05845922, 0.47058555, 0.05845922, 0.01210375, 0.00185634),
                  (0.00185634, 0.01210375, 0.05845922, 0.05845922, 0.01210375, 0.00185634, np.nan),
                  (0.00185634, 0.01210375, 0.01210375, 0.01210375, 0.00185634, np.nan, np.nan),
                  (0.00185634, 0.00185634, 0.00185634, 0.00185634, np.nan, np.nan, np.nan)],
                 dtype = float))


def test_apply_identity(tmp_path):
    epc = os.path.join(tmp_path, 'identity.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    tm = make_wavy_tri_mesh(model, crs.uuid, add_random = True)
    stencil = rqs.TriMeshStencil((1, 0), normalize = None)
    tm_id = stencil.apply(tm)
    assert_array_almost_equal(tm_id.full_array_ref(), tm.full_array_ref())
    tm_id.write_hdf5()
    tm_id.create_xml()
    tm_id_reloaded = rqs.TriMesh(model, uuid = tm_id.uuid)
    assert_array_almost_equal(tm_id_reloaded.full_array_ref(), tm.full_array_ref())


def test_apply_gaussian_normalized_flat(tmp_path):
    epc = os.path.join(tmp_path, 'wavy.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    tm = make_wavy_tri_mesh(model, crs.uuid)
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(5, sigma = 2.5, normalize_mode_flat = True)
    tm_smooth = stencil.apply(tm)

    expected_z = np.array([[
        7.82352342e+00, 8.34976138e+00, 7.50538978e+00, 6.13493846e+00, 5.99923454e+00, 8.16198592e+00, 1.09600586e+01,
        1.18483191e+01, 1.05464316e+01, 8.78589554e+00
    ],
                           [
                               7.61820155e+00, 7.35259960e+00, 6.08276494e+00, 4.97596690e+00, 6.14140646e+00,
                               9.13676536e+00, 1.11828620e+01, 1.06052549e+01, 8.43149981e+00, 6.81052215e+00
                           ],
                           [
                               5.21004433e+00, 5.48148386e+00, 4.70261040e+00, 3.08384811e+00, 2.68951228e+00,
                               4.89275089e+00, 7.90644035e+00, 8.91585385e+00, 7.46818470e+00, 4.92780975e+00
                           ],
                           [
                               4.66752512e+00, 4.27218165e+00, 2.67694191e+00, 1.35878768e+00, 2.32576950e+00,
                               5.29212437e+00, 7.46997746e+00, 7.08659250e+00, 5.12423350e+00, 2.43338336e+00
                           ],
                           [
                               5.27707304e+00, 5.13703689e+00, 4.34424700e+00, 2.83935491e+00, 2.51787134e+00,
                               4.66383434e+00, 7.56513465e+00, 8.50449187e+00, 6.94112326e+00, 4.54107765e+00
                           ],
                           [
                               5.53145243e+00, 4.93640111e+00, 3.63362254e+00, 2.48880541e+00, 3.54695230e+00,
                               6.48559413e+00, 8.58914842e+00, 8.10846523e+00, 5.85927047e+00, 3.85091887e+00
                           ],
                           [
                               3.97063151e+00, 4.37588382e+00, 3.31846277e+00, 1.73285126e+00, 1.36844287e+00,
                               3.51440587e+00, 6.48574695e+00, 7.52937047e+00, 6.22494748e+00, 3.69414545e+00
                           ],
                           [
                               4.66664289e+00, 4.22462125e+00, 2.66816093e+00, 1.38661502e+00, 2.37461631e+00,
                               5.34097118e+00, 7.50402266e+00, 7.08392997e+00, 5.06417333e+00, 2.50970806e+00
                           ],
                           [
                               5.33050480e+00, 5.07920225e+00, 4.43339459e+00, 2.89104026e+00, 2.54977627e+00,
                               4.75301489e+00, 7.69431251e+00, 8.57165032e+00, 6.86013832e+00, 4.47382245e+00
                           ],
                           [
                               4.02394827e+00, 3.56274374e+00, 2.16208128e+00, 9.96598887e-01, 2.12704305e+00,
                               5.12240195e+00, 7.16185824e+00, 6.59491477e+00, 4.49592996e+00, 2.46335413e+00
                           ],
                           [
                               -4.70205540e-03, 6.80382692e-01, -3.04368270e-01, -1.78866132e+00, -2.00564186e+00,
                               1.57109521e-01, 3.00833765e+00, 3.99346006e+00, 2.83312093e+00, 5.68032776e-01
                           ]],
                          dtype = float)

    assert_array_almost_equal(tm_smooth.full_array_ref()[:, :, 2], expected_z)


def test_apply_gaussian_normalized_non_flat(tmp_path):
    epc = os.path.join(tmp_path, 'wavy.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    tm = make_wavy_tri_mesh(model, crs.uuid)
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(5, sigma = 2.5, normalize_mode_flat = False)
    tm_smooth = stencil.apply(tm)

    expected_z = np.array([[
        3.81409041, 10.61535224, 9.23872682, 2.77984472, 0.89493332, 7.41012706, 14.7016452, 14.07603046, 7.52007865,
        4.35971993
    ],
                           [
                               13.41839327, 17.75146492, 14.99594634, 9.04228905, 8.95650708, 16.13308335, 22.05384002,
                               19.96463947, 13.88864801, 13.25292809
                           ],
                           [
                               -0.83259185, 4.92932055, 3.9976712, -1.8804199, -3.93665731, 1.85969368, 8.86120658,
                               8.76823954, 2.78317593, -1.00216843
                           ],
                           [
                               -5.2700118, -0.29111998, -2.76968706, -8.44648715, -8.52366379, -1.59876757, 4.12188895,
                               2.00845158, -4.60645788, -9.33905124
                           ],
                           [
                               8.17016526, 12.04311096, 10.69731412, 4.84770467, 2.82534187, 8.53207919, 15.43130797,
                               15.3775775, 9.60058226, 6.42378396
                           ],
                           [
                               10.08103451, 13.91876589, 11.24132598, 5.56812892, 5.50904758, 12.41222224, 18.12891235,
                               16.16686671, 10.26085202, 9.11710887
                           ],
                           [
                               -6.21019883, 0.45081572, -0.36705007, -6.10709397, -8.11895955, -2.41222224, 4.49800096,
                               4.42771389, -1.4980896, -5.54324429
                           ],
                           [
                               -1.90719658, 2.81684959, 0.28512786, -5.3959816, -5.46997218, 1.45492404, 7.17844358,
                               5.09540181, -1.35966588, -5.29434674
                           ],
                           [
                               12.30141275, 15.44749425, 13.98980573, 8.00415879, 5.93848324, 11.73483424, 18.7258953,
                               18.64719056, 12.80616009, 9.85538792
                           ],
                           [
                               5.42765571, 9.91685237, 7.19374376, 1.24130993, 1.15048337, 8.32705964, 14.23725345,
                               12.09421174, 5.78317251, 3.72602968
                           ],
                           [
                               -12.34207491, -4.70857955, -5.8511041, -12.27053957, -14.15910041, -7.64390668,
                               -0.3478976, -0.98774533, -7.66301496, -11.49117229
                           ]],
                          dtype = float)

    assert_array_almost_equal(tm_smooth.full_array_ref()[:, :, 2], expected_z)


def test_apply_gaussian_preserve_nan(tmp_path):
    epc = os.path.join(tmp_path, 'wavy.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    tm = make_wavy_tri_mesh(model, crs.uuid, make_hole = True)
    nan_in = np.isnan(tm.full_array_ref()[:, :, 2])
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(5, sigma = 2.5, normalize_mode_flat = False)
    tm_smooth = stencil.apply(tm, preserve_nan = True)
    nan_out = np.isnan(tm_smooth.full_array_ref()[:, :, 2])
    assert np.all(nan_out == nan_in)

    expected_z = np.array([[
        3.81409041, 10.61535224, 9.23872682, 2.77984472, 0.89493332, 7.41012706, 14.7016452, 14.07603046, 7.52007865,
        4.35971993
    ],
                           [
                               13.41839327, 17.75601416, 15.00246827, 9.03930237, 8.95338649, 16.14136398, 22.06374477,
                               19.96463947, 13.88864801, 13.25292809
                           ],
                           [
                               -0.83259185, 4.94198442, 3.99605923, -1.95547427, -4.03090826, 1.82003578, 8.87794422,
                               8.7676083, 2.78317593, -1.00216843
                           ],
                           [
                               -5.27423781, -0.26207427, -2.92548272, -8.97768048, -9.05753049, -1.7419426, 4.11295031,
                               2.00276228, -4.60645788, -9.33905124
                           ],
                           [
                               8.1928308, 12.19694759, 11.26786958, 5.13736561, 2.57806622, 9.00624525, 15.7020556,
                               15.42908733, 9.60057932, 6.42378396
                           ],
                           [
                               10.2293434, 14.63823461, 13.66179199, np.nan, np.nan, 13.26689746, 18.46566432,
                               16.2231374, 10.2613752, 9.11710887
                           ],
                           [
                               -6.15077532, 0.79725905, 0.80350108, np.nan, np.nan, -1.99984539, 4.87517323, 4.51615648,
                               -1.49194411, -5.54324429
                           ],
                           [
                               -1.82844774, 3.21028828, 1.51330689, -4.19247103, -4.88642352, 1.71181098, 7.29226265,
                               5.11130123, -1.35966588, -5.29434674
                           ],
                           [
                               12.33292302, 15.62200561, 14.66163121, 8.86921587, 6.71761155, 12.38012944, 18.95035786,
                               18.68369182, 12.80616009, 9.85538792
                           ],
                           [
                               5.44957471, 10.06135456, 7.41153773, 1.3981885, 1.30632945, 8.46486333, 14.26870119,
                               12.09421174, 5.78317251, 3.72602968
                           ],
                           [
                               -12.34207491, -4.69513915, -5.82523629, -12.25717071, -14.14933713, -7.62179224,
                               -0.32867695, -0.98774533, -7.66301496, -11.49117229
                           ]],
                          dtype = float)

    assert_array_almost_equal(tm_smooth.full_array_ref()[:, :, 2], expected_z)


def test_apply_gaussian_handle_nan(tmp_path):
    epc = os.path.join(tmp_path, 'wavy.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    tm = make_wavy_tri_mesh(model, crs.uuid, make_hole = True)
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(5, sigma = 2.5, normalize_mode_flat = True)
    tm_smooth = stencil.apply(tm)
    assert not np.any(np.isnan(tm_smooth.full_array_ref()[:, :, 2]))

    expected_z = np.array([[
        7.82352342e+00, 8.34976138e+00, 7.50538978e+00, 6.13493846e+00, 5.99923454e+00, 8.16198592e+00, 1.09600586e+01,
        1.18483191e+01, 1.05464316e+01, 8.78589554e+00
    ],
                           [
                               7.61820155e+00, 7.33365603e+00, 6.04833555e+00, 4.93467834e+00, 6.10832710e+00,
                               9.12441769e+00, 1.11883624e+01, 1.06052549e+01, 8.43149981e+00, 6.81052215e+00
                           ],
                           [
                               5.21004433e+00, 5.54380485e+00, 4.74670640e+00, 3.03146238e+00, 2.62556165e+00,
                               4.89614081e+00, 7.96037833e+00, 8.91371641e+00, 7.46818470e+00, 4.92780975e+00
                           ],
                           [
                               4.73027835e+00, 4.52288758e+00, 2.82053147e+00, 1.24299894e+00, 2.29674248e+00,
                               5.45189391e+00, 7.51384517e+00, 7.07896265e+00, 5.12423350e+00, 2.43338336e+00
                           ],
                           [
                               5.35936889e+00, 5.44080112e+00, 5.08571104e+00, 3.55227051e+00, 2.99091680e+00,
                               5.24536311e+00, 7.85392939e+00, 8.56650880e+00, 6.93266105e+00, 4.54107765e+00
                           ],
                           [
                               5.90528255e+00, 5.78940697e+00, 5.14418837e+00, 4.16649938e+00, 4.87175539e+00,
                               7.32162360e+00, 8.93902293e+00, 8.16522454e+00, 5.84617735e+00, 3.85091887e+00
                           ],
                           [
                               4.49837898e+00, 5.47802694e+00, 5.13338463e+00, 3.82734646e+00, 3.20057402e+00,
                               5.10349786e+00, 7.48088130e+00, 7.89591928e+00, 6.29976052e+00, 3.69414545e+00
                           ],
                           [
                               5.11746821e+00, 5.22375705e+00, 4.35838808e+00, 3.37027971e+00, 3.92481609e+00,
                               6.24621557e+00, 7.86882495e+00, 7.16049273e+00, 5.06417333e+00, 2.50970806e+00
                           ],
                           [
                               5.45500423e+00, 5.49188005e+00, 5.43745664e+00, 4.22147303e+00, 3.83478205e+00,
                               5.73824636e+00, 8.08277566e+00, 8.65934919e+00, 6.86013832e+00, 4.47382245e+00
                           ],
                           [
                               4.13243740e+00, 3.95853466e+00, 2.71481469e+00, 1.49469181e+00, 2.66296807e+00,
                               5.51868821e+00, 7.25048764e+00, 6.59491477e+00, 4.49592996e+00, 2.46335413e+00
                           ],
                           [
                               -4.70205540e-03, 7.83953677e-01, -1.22489116e-01, -1.62853331e+00, -1.84892921e+00,
                               3.33409507e-01, 3.11535987e+00, 3.99346006e+00, 2.83312093e+00, 5.68032776e-01
                           ]],
                          dtype = float)

    assert_array_almost_equal(tm_smooth.full_array_ref()[:, :, 2], expected_z)


def test_apply_gaussian_do_not_handle_nan(tmp_path):
    epc = os.path.join(tmp_path, 'wavy.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    tm = make_wavy_tri_mesh(model, crs.uuid)
    stencil = rqs.TriMeshStencil.for_gaussian_normalized(3, sigma = 1.3, normalize_mode_flat = True)
    tm_smooth = stencil.apply(tm, handle_nan = False, title = 'smooth')
    assert np.count_nonzero(np.logical_not(np.isnan(tm_smooth.full_array_ref()[:, :, 2]))) == 42
