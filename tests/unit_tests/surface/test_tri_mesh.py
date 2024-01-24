import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.surface as rqs


def test_tri_mesh_create_save_reload(example_model_and_crs):

    def check(trim):
        assert trim is not None
        assert trim.nj == 3 and trim.ni == 4
        assert trim.flavour == 'explicit'
        assert maths.isclose(trim.t_side, 10.0)
        assert trim.z_uom is None
        xyz = trim.full_array_ref()
        assert xyz is not None
        assert xyz.shape == (3, 4, 3)
        assert np.all(np.isclose(xyz[..., 2], 0.0))
        assert_array_almost_equal(xyz[0, :, 0], (0.0, 10.0, 20.0, 30.0))
        assert_array_almost_equal(xyz[1, :, 0], (5.0, 15.0, 25.0, 35.0))
        assert_array_almost_equal(xyz[0, :, 0], xyz[2, :, 0])
        for i in range(4):
            assert_array_almost_equal(xyz[:, i, 1], np.array((0.0, 5.0 * maths.sqrt(3.0), 10.0 * maths.sqrt(3.0))))

    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 10.0, nj = 3, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    trim.write_hdf5()
    trim.create_xml()
    check(trim)
    model.store_epc()
    epc = model.epc_file
    trim_uuid = trim.uuid
    xyz = trim.full_array_ref()
    del model, trim
    model = rq.Model(epc)
    trim_reloaded = rqs.TriMesh(model, uuid = trim_uuid)
    assert trim_reloaded is not None
    check(trim_reloaded)
    assert_array_almost_equal(trim_reloaded.full_array_ref(), xyz)


def test_tri_mesh_create_save_reload_with_z(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.array([(200.0, 250.0, -123.4, 300.0), (400.0, 450.0, 0.0, 500.0), (300.0, 350.0, 1234.5, 400.0)],
                        dtype = float)
    trim = rqs.TriMesh(model,
                       t_side = 10.0,
                       nj = 3,
                       ni = 4,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    trim.write_hdf5()
    trim.create_xml()
    model.store_epc()
    epc = model.epc_file
    trim_uuid = trim.uuid
    del model, trim
    model = rq.Model(epc)
    trim_reloaded = rqs.TriMesh(model, uuid = trim_uuid)
    assert trim_reloaded is not None
    p = trim_reloaded.full_array_ref()
    assert_array_almost_equal(p[..., 2], z_values)


def test_tri_mesh_tji_for_xy(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 10.0, nj = 4, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    assert trim.tji_for_xy((0.0, 5.0)) is None
    assert trim.tji_for_xy((5.0, 5.0)) == (0, 0)
    assert trim.tji_for_xy((10.0, 5.0)) == (0, 1)
    assert trim.tji_for_xy((15.0, 5.0)) == (0, 2)
    assert trim.tji_for_xy((25.0, 5.0)) == (0, 4)
    assert trim.tji_for_xy((30.0, 5.0)) == (0, 5)
    assert trim.tji_for_xy((35.0, 5.0)) is None
    assert trim.tji_for_xy((5.0, 9.0)) == (1, 0)
    assert trim.tji_for_xy((6.0, 9.0)) == (1, 1)
    assert trim.tji_for_xy((30.0, 17.0)) == (1, 5)
    assert trim.tji_for_xy((29.0, 17.0)) == (1, 4)
    assert trim.tji_for_xy((5.0, 25.5)) == (2, 0)
    assert trim.tji_for_xy((30.0, 25.5)) == (2, 5)
    assert trim.tji_for_xy((30.0, 26.0)) is None


def test_tri_mesh_tji_for_xy_array(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 10.0, nj = 4, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    xy = np.array([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0), (15.0, 5.0), (25.0, 5.0), (30.0, 5.0), (35.0, 5.0), (5.0, 9.0),
                   (6.0, 9.0), (30.0, 17.0), (29.0, 17.0), (5.0, 25.5), (30.0, 25.5), (30.0, 26.0)],
                  dtype = float)
    etji = np.array([(-1, -1), (0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (-1, -1), (1, 0), (1, 1), (1, 5), (1, 4), (2, 0),
                     (2, 5), (-1, -1)],
                    dtype = int)
    tji = trim.tji_for_xy_array(xy)
    assert tji.shape == (14, 2)
    assert np.all(tji == etji)


def test_tri_mesh_tri_nodes_for_tji(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 37.1, nj = 5, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    assert np.all(trim.tri_nodes_for_tji((0, 0)) == [(0, 0), (0, 1), (1, 0)])
    assert np.all(trim.tri_nodes_for_tji((0, 1)) == [(1, 0), (1, 1), (0, 1)])
    assert np.all(trim.tri_nodes_for_tji((0, 2)) == [(0, 1), (0, 2), (1, 1)])
    assert np.all(trim.tri_nodes_for_tji((0, 5)) == [(1, 2), (1, 3), (0, 3)])
    assert np.all(trim.tri_nodes_for_tji((3, 0)) == [(4, 0), (4, 1), (3, 0)])
    assert np.all(trim.tri_nodes_for_tji((3, 1)) == [(3, 0), (3, 1), (4, 1)])
    assert np.all(trim.tri_nodes_for_tji((3, 5)) == [(3, 2), (3, 3), (4, 3)])


def test_tri_mesh_tri_nodes_for_tji_array(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 37.1, nj = 5, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    tji = np.array([(0, 0), (0, 1), (0, 2), (0, 5), (3, 0), (3, 1), (3, 5)], dtype = int)
    # yapf: disable
    e_tn = np.array([[(0, 0), (0, 1), (1, 0)],
                     [(1, 0), (1, 1), (0, 1)],
                     [(0, 1), (0, 2), (1, 1)],
                     [(1, 2), (1, 3), (0, 3)],
                     [(4, 0), (4, 1), (3, 0)],
                     [(3, 0), (3, 1), (4, 1)],
                     [(3, 2), (3, 3), (4, 3)]], dtype = int)
    # yapf: enable
    tn = trim.tri_nodes_for_tji_array(tji)
    assert tn.shape == (7, 3, 2)
    assert np.all(tn == e_tn)


def test_tri_mesh_all_tri_nodes(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 0.7, nj = 3, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    all_nodes = trim.all_tri_nodes()
    assert all_nodes.shape == ((2, 6, 3, 2))
    for j in range(2):
        for i in range(6):
            assert np.all(all_nodes[j, i] == trim.tri_nodes_for_tji((j, i)))


def test_tri_mesh_triangles_and_points(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 1200.0, nj = 4, ni = 3, crs_uuid = crs.uuid, title = 'test tri mesh')
    t, p = trim.triangles_and_points()
    assert t.shape == (12, 3)
    assert p.shape == (12, 3)
    assert p[t].shape == (12, 3, 3)
    assert trim.tji_for_triangle_index(0) == (0, 0)
    assert trim.tji_for_triangle_index(1) == (0, 1)
    assert trim.tji_for_triangle_index(3) == (0, 3)
    assert trim.tji_for_triangle_index(4) == (1, 0)
    assert trim.tji_for_triangle_index(11) == (2, 3)
    for ti in range(12):
        assert trim.triangle_index_for_tji(trim.tji_for_triangle_index(ti)) == ti


def test_surface_from_tri_mesh(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 1000.0, nj = 3, ni = 3, crs_uuid = crs.uuid, title = 'test tri mesh surface')
    surf = rqs.Surface.from_tri_mesh(trim)
    assert surf is not None
    st, sp = surf.triangles_and_points()
    tt, tp = trim.triangles_and_points()
    assert np.all(st == tt)
    assert_array_almost_equal(sp, tp)


def test_tri_mesh_trilinear_coordinates(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 100.0, nj = 3, ni = 3, crs_uuid = crs.uuid, title = 'test tri mesh')
    third = 1.0 / 3.0
    tji, tc = trim.tji_tc_for_xy((50.0, 100.0 * maths.sqrt(3.0) / 6.0))
    assert tji == (0, 0)
    assert_array_almost_equal(tc, (third, third, third))
    tji, tc = trim.tji_tc_for_xy((100.0, 100.0 * maths.sqrt(3.0) * 2.0 / 3.0))
    assert tji == (1, 1)
    assert_array_almost_equal(tc, (third, third, third))
    tji, tc = trim.tji_tc_for_xy((150.0, 100.0 * maths.sqrt(3.0) - 1.0e-10))
    assert tji == (1, 2)
    assert_array_almost_equal(tc, (0.5, 0.5, 0.0))
    tji, tc = trim.tji_tc_for_xy((175.0 - 1.0e-10, 100.0 * maths.sqrt(3.0) / 4.0 - 1.0e-10))
    assert tji == (0, 2)
    assert_array_almost_equal(tc, (0.0, 0.5, 0.5))
    tji, tc = trim.tji_tc_for_xy((50.0, 100.0 * maths.sqrt(3.0) / 2.0 - 1.0e-10))
    assert tji == (0, 0)
    assert_array_almost_equal(tc, (0.0, 0.0, 1.0))
    tji, tc = trim.tji_tc_for_xy((150.0, 100.0 * maths.sqrt(3.0) / 2.0 + 1.0e-10))
    assert tji == (1, 2)
    assert_array_almost_equal(tc, (0.0, 0.0, 1.0))
    tji, tc = trim.tji_tc_for_xy((25.0, 50.0))
    assert tji is None and tc is None


def test_tri_mesh_ji_and_weights(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 100.0, nj = 3, ni = 3, crs_uuid = crs.uuid, title = 'test tri mesh')
    third = 1.0 / 3.0
    ji, tc = trim.ji_and_weights_for_xy((50.0, 100.0 * maths.sqrt(3.0) / 6.0))
    assert np.all(ji == [(0, 0), (0, 1), (1, 0)])
    assert_array_almost_equal(tc, (third, third, third))
    ji, tc = trim.ji_and_weights_for_xy((100.0, 100.0 * maths.sqrt(3.0) * 2.0 / 3.0))
    assert np.all(ji == [(1, 0), (1, 1), (2, 1)])
    assert_array_almost_equal(tc, (third, third, third))
    ji, tc = trim.ji_and_weights_for_xy((150.0, 100.0 * maths.sqrt(3.0) - 1.0e-10))
    assert np.all(ji == [(2, 1), (2, 2), (1, 1)])
    assert_array_almost_equal(tc, (0.5, 0.5, 0.0))
    ji, tc = trim.ji_and_weights_for_xy((175.0 - 1.0e-10, 100.0 * maths.sqrt(3.0) / 4.0 - 1.0e-10))
    assert np.all(ji == [(0, 1), (0, 2), (1, 1)])
    assert_array_almost_equal(tc, (0.0, 0.5, 0.5))
    ji, tc = trim.ji_and_weights_for_xy((50.0, 100.0 * maths.sqrt(3.0) / 2.0 - 1.0e-10))
    assert np.all(ji == [(0, 0), (0, 1), (1, 0)])
    assert_array_almost_equal(tc, (0.0, 0.0, 1.0))
    ji, tc = trim.ji_and_weights_for_xy((150.0, 100.0 * maths.sqrt(3.0) / 2.0 + 1.0e-10))
    assert np.all(ji == [(2, 1), (2, 2), (1, 1)])
    assert_array_almost_equal(tc, (0.0, 0.0, 1.0))


def test_tri_mesh_z_interpolation(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.array([(200.0, 250.0, 300.0), (400.0, 450.0, 500.0), (300.0, 350.0, 400.0)], dtype = float)
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 3,
                       ni = 3,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    z = trim.interpolated_z((50.0, 100.0 * maths.sqrt(3.0) / 6.0))
    assert maths.isclose(z, 850.0 / 3.0)
    z = trim.interpolated_z((100.0, 100.0 * maths.sqrt(3.0) * 2.0 / 3.0))
    assert maths.isclose(z, 400.0)
    z = trim.interpolated_z((150.0, 100.0 * maths.sqrt(3.0) - 1.0e-10))
    assert maths.isclose(z, 375.0)
    z = trim.interpolated_z((175.0, 100.0 * maths.sqrt(3.0) / 4.0))
    assert maths.isclose(z, 375.0)
    z = trim.interpolated_z((50.0, 100.0 * maths.sqrt(3.0) / 2.0 - 1.0e-10))
    assert maths.isclose(z, 400.0)
    z = trim.interpolated_z((150.0, 100.0 * maths.sqrt(3.0) / 2.0 + 1.0e-10))
    assert maths.isclose(z, 450.0)
    z = trim.interpolated_z((25.0, 50.0))
    assert z is None


def test_tri_mesh_z_interpolation_array(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.array([(200.0, 250.0, 300.0), (400.0, 450.0, 500.0), (300.0, 350.0, 400.0)], dtype = float)
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 3,
                       ni = 3,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    xy = np.array([(50.0, 100.0 * maths.sqrt(3.0) / 6.0), (100.0, 100.0 * maths.sqrt(3.0) * 2.0 / 3.0),
                   (150.0, 100.0 * maths.sqrt(3.0) - 1.0e-10), (175.0, 100.0 * maths.sqrt(3.0) / 4.0),
                   (50.0, 100.0 * maths.sqrt(3.0) / 2.0 - 1.0e-10), (150.0, 100.0 * maths.sqrt(3.0) / 2.0 + 1.0e-10),
                   (25.0, 50.0)],
                  dtype = float)
    z = trim.interpolated_z_array(xy)
    ez = np.array([850.0 / 3.0, 400.0, 375.0, 375.0, 400.0, 450.0, np.NaN], dtype = float)
    assert_array_almost_equal(z, ez)


def test_tri_mesh_nodes_in_triangles(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 10.0, nj = 5, ni = 5, crs_uuid = crs.uuid, title = 'test tri mesh')
    other_t = np.array([[(7.0, 5.0), (7.0, 25.0), (37.0, 5.0)], [(37.0, 25.0), (7.0, 25.0), (37.0, 5.0)]],
                       dtype = float)
    nit = trim.tri_nodes_in_triangles(other_t)
    assert nit.ndim == 2 and nit.shape[1] == 3
    assert len(nit) == 6
    assert np.all(np.logical_or(nit[:, 0] == 0, nit[:, 0] == 1))  # other triangle number
    assert np.all(np.logical_and(nit[:, 1] > 0, nit[:, 1] < 3))  # node j index
    assert np.all(np.logical_and(nit[:, 2] > 0, nit[:, 2] < 4))  # node i index
    assert np.count_nonzero(nit[:, 0] == 0) == 3
    assert np.count_nonzero(nit[:, 0] == 1) == 3
    for i in range(6):
        ji = tuple(nit[i, 1:])
        if nit[i, 0] == 0:
            assert ji in [(1, 1), (1, 2), (2, 1)]
        else:
            assert ji in [(1, 3), (2, 2), (2, 3)]


def test_tri_mesh_nodes_in_triangles_with_origin(example_model_and_crs):
    model, crs = example_model_and_crs
    origin = (127.34, -1523.19, 0.0)
    trim = rqs.TriMesh(model,
                       t_side = 10.0,
                       nj = 5,
                       ni = 6,
                       origin = origin,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    other_t = np.array([[(7.0, 5.0), (7.0, 25.0),
                         (37.0, 5.0)], [(37.0, 25.0), (7.0, 25.0),
                                        (37.0, 5.0)], [(42.0, 15.0), (55.0, 15.0),
                                                       (51.0, 3.0)], [(43.0, 25.0), (47.0, 25.0), (45.0, 28.0)]],
                       dtype = float)
    other_t += np.expand_dims(np.expand_dims(np.array(origin[:2], dtype = float), axis = 0), axis = 0)
    nit = trim.tri_nodes_in_triangles(other_t)
    assert nit.ndim == 2 and nit.shape[1] == 3
    assert len(nit) == 7
    assert np.all(np.logical_or(np.logical_or(nit[:, 0] == 0, nit[:, 0] == 1), nit[:, 0] == 3))  # other triangle number
    assert np.all(np.logical_and(nit[:-1, 1] > 0, nit[:-1, 1] < 3))  # node j index
    assert np.all(np.logical_and(nit[:-1, 2] > 0, nit[:-1, 2] < 4))  # node i index
    assert np.all(nit[-1] == (3, 3, 4))  # other triangle number, node j, node i; here assumed to be last in return list
    assert np.count_nonzero(nit[:, 0] == 0) == 3
    assert np.count_nonzero(nit[:, 0] == 1) == 3
    assert np.count_nonzero(nit[:, 0] == 2) == 0
    assert np.count_nonzero(nit[:, 0] == 3) == 1
    for i in range(6):
        ji = tuple(nit[i, 1:])
        if nit[i, 0] == 0:
            assert ji in [(1, 1), (1, 2), (2, 1)]
        else:
            assert ji in [(1, 3), (2, 2), (2, 3)]


def test_tri_mesh_nodes_in_triangles_with_origin_nan_z(example_model_and_crs):
    model, crs = example_model_and_crs
    origin = (127.34, -1523.19, np.nan)
    trim = rqs.TriMesh(model,
                       t_side = 10.0,
                       nj = 5,
                       ni = 6,
                       origin = origin,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    other_t = np.array([[(7.0, 5.0), (7.0, 25.0),
                         (37.0, 5.0)], [(37.0, 25.0), (7.0, 25.0),
                                        (37.0, 5.0)], [(42.0, 15.0), (55.0, 15.0),
                                                       (51.0, 3.0)], [(43.0, 25.0), (47.0, 25.0), (45.0, 28.0)]],
                       dtype = float)
    other_t += np.expand_dims(np.expand_dims(np.array(origin[:2], dtype = float), axis = 0), axis = 0)
    nit = trim.tri_nodes_in_triangles(other_t)
    assert nit.ndim == 2 and nit.shape[1] == 3
    assert len(nit) == 7
    assert np.all(np.logical_or(np.logical_or(nit[:, 0] == 0, nit[:, 0] == 1), nit[:, 0] == 3))  # other triangle number
    assert np.all(np.logical_and(nit[:-1, 1] > 0, nit[:-1, 1] < 3))  # node j index
    assert np.all(np.logical_and(nit[:-1, 2] > 0, nit[:-1, 2] < 4))  # node i index
    assert np.all(nit[-1] == (3, 3, 4))  # other triangle number, node j, node i; here assumed to be last in return list
    assert np.count_nonzero(nit[:, 0] == 0) == 3
    assert np.count_nonzero(nit[:, 0] == 1) == 3
    assert np.count_nonzero(nit[:, 0] == 2) == 0
    assert np.count_nonzero(nit[:, 0] == 3) == 1
    for i in range(6):
        ji = tuple(nit[i, 1:])
        if nit[i, 0] == 0:
            assert ji in [(1, 1), (1, 2), (2, 1)]
        else:
            assert ji in [(1, 3), (2, 2), (2, 3)]


def test_tri_mesh_axial_edge_crossing(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.array([(-100.0, -100.0, -100.0, -100.0), (100.0, 100.0, 100.0, 100.0), (500.0, 500.0, 500.0, 500.0)],
                        dtype = float)
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 3,
                       ni = 4,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    # z axis
    z_cross = trim.axial_edge_crossings(2)
    assert z_cross.shape == (7, 3)
    xi = np.argsort(z_cross[:, 0])
    assert_array_almost_equal(z_cross[:, 2], 0.0)
    assert_array_almost_equal(z_cross[:, 1], 50.0 * maths.sqrt(3.0) / 2.0)
    assert_array_almost_equal(z_cross[xi[1:], 0] - z_cross[xi[:-1], 0], 50.0)
    assert maths.isclose(z_cross[xi[0], 0], 25.0)
    z_cross = trim.axial_edge_crossings(2, value = 300.0)
    assert z_cross.shape == (7, 3)
    xi = np.argsort(z_cross[:, 0])
    assert_array_almost_equal(z_cross[:, 2], 300.0)
    assert_array_almost_equal(z_cross[:, 1], 150.0 * maths.sqrt(3.0) / 2.0)
    assert_array_almost_equal(z_cross[xi[1:], 0] - z_cross[xi[:-1], 0], 50.0)
    assert maths.isclose(z_cross[xi[0], 0], 25.0)
    # y axis
    z_cross = trim.axial_edge_crossings(1, value = 150.0 * maths.sqrt(3.0) / 2.0)
    assert z_cross.shape == (7, 3)
    xi = np.argsort(z_cross[:, 0])
    assert_array_almost_equal(z_cross[:, 2], 300.0)
    assert_array_almost_equal(z_cross[:, 1], 150.0 * maths.sqrt(3.0) / 2.0)
    assert_array_almost_equal(z_cross[xi[1:], 0] - z_cross[xi[:-1], 0], 50.0)
    assert maths.isclose(z_cross[xi[0], 0], 25.0)
    # x axis
    z_cross = trim.axial_edge_crossings(0, value = 125.0)
    assert z_cross.shape == (5, 3)
    yi = np.argsort(z_cross[:, 1])
    assert_array_almost_equal(z_cross[yi, 2], (-100.0, 0.0, 100.0, 300.0, 500.0))
    assert_array_almost_equal(z_cross[yi[1:], 1] - z_cross[yi[:-1], 1], 50.0 * maths.sqrt(3.0) / 2.0)
    maths.isclose(z_cross[yi[0], 1], 0.0)
    assert_array_almost_equal(z_cross[:, 0], 125.0)
