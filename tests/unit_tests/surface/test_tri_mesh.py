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


def test_tri_mesh_tji_for_xy(example_model_and_crs):
    model, crs = example_model_and_crs
    trim = rqs.TriMesh(model, t_side = 10.0, nj = 4, ni = 4, crs_uuid = crs.uuid, title = 'test tri mesh')
    assert trim.tji_for_xy((0.0, 5.0)) == (None, None)
    assert trim.tji_for_xy((5.0, 5.0)) == (0, 0)
    assert trim.tji_for_xy((10.0, 5.0)) == (0, 1)
    assert trim.tji_for_xy((15.0, 5.0)) == (0, 2)
    assert trim.tji_for_xy((25.0, 5.0)) == (0, 4)
    assert trim.tji_for_xy((30.0, 5.0)) == (0, 5)
    assert trim.tji_for_xy((35.0, 5.0)) == (None, None)
    assert trim.tji_for_xy((5.0, 9.0)) == (1, 0)
    assert trim.tji_for_xy((6.0, 9.0)) == (1, 1)
    assert trim.tji_for_xy((30.0, 17.0)) == (1, 5)
    assert trim.tji_for_xy((29.0, 17.0)) == (1, 4)
    assert trim.tji_for_xy((5.0, 25.5)) == (2, 0)
    assert trim.tji_for_xy((30.0, 25.5)) == (2, 5)
    assert trim.tji_for_xy((30.0, 26.0)) == (None, None)


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
