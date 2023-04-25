import math as maths
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.crs as rqc
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.olio.triangulation as tri
import resqpy.olio.vector_utilities as vec
import resqpy.surface as rqs
from resqpy.olio.random_seed import seed


def test_ccc():

    def check_equidistant(c, p_list):
        v = np.stack(tuple([c - p[:2] for p in p_list]))
        r = vec.naive_2d_lengths(v)
        assert_array_almost_equal(r[1:], r[0])

    # 3 points in orthogonal pattern
    p1 = np.array((0.0, 0.0, 0.0))
    p2 = np.array((20.0, 0.0, 0.0))
    p3 = np.array((0.0, 10.0, 5.0))
    assert_array_almost_equal(tri.ccc(p1, p2, p3), (10.0, 5.0))
    # equilateral triangle
    s = 23.57
    p1 = np.array((10.0, 20.0, 30.0))
    p2 = np.array((10.0, 20.0 + s, -45.0))
    p3 = np.array((10.0 + s * maths.cos(maths.radians(30.0)), 20.0 + 0.5 * s, 12.13))
    assert_array_almost_equal(tri.ccc(p1, p2, p3), np.mean(np.stack((p1, p2, p3)), axis = 0)[:2])
    # asymmetric triangle
    p1 = np.array((25.3, 12.1, 0.0))
    p2 = np.array((23.6, 2.9, -1.0))
    p3 = np.array((22.1, 87.3, 1.5))
    c = np.array(tri.ccc(p1, p2, p3))
    check_equidistant(c, (p1, p2, p3))
    # highly asymmetric triangle
    p1 = np.array((0.0, 0.0, 0.0))
    p2 = np.array((100.0, 0.0, 0.0))
    p3 = np.array((200.0, 1.0, 0.0))
    c = np.array(tri.ccc(p1, p2, p3))
    check_equidistant(c, (p1, p2, p3))


def test_voronoi():
    seed_value = 3567
    n_list = range(5, 50, 3)
    model = rq.Model(create_basics = True)
    crs = rqc.Crs(model)
    crs.create_xml()
    # setup unit square area of interest
    aoi_xyz = np.zeros((4, 3))
    aoi_xyz[1, 1] = 1.0
    aoi_xyz[2, :2] = 1.0
    aoi_xyz[3, 0] = 1.0
    aoi = rql.Polyline(model, set_coord = aoi_xyz, is_closed = True, set_crs = crs.uuid, title = 'aoi')
    # and an alternative area of interest
    aoi_heptagon_xyz = np.zeros((7, 3))
    aoi_heptagon_xyz[:, :2] = ((-0.5, -1.0), (-1.5, 0.5), (-1.0, 1.7), (0.5, 2.0), (2.0, 1.7), (2.5, 0.5), (1.5, -1.0))
    aoi_heptagon = rql.Polyline(model,
                                set_coord = aoi_heptagon_xyz,
                                is_closed = True,
                                set_crs = crs.uuid,
                                title = 'heptagon')
    aoi_heptagon_area = aoi_heptagon.area()

    for n in n_list:  # number of seed points (number of Voronoi cells)
        seed(seed_value)
        x = np.random.random(n)
        y = np.random.random(n)
        p = np.stack((x, y), axis = -1)
        # compute the Delauney triangulation
        t, b = tri.dt(p, plot_fn = None, progress_fn = None, return_hull = True)
        # dt function can return triangulation with a slightly concave hull, which voronoi function cannot handle
        hull = rql.Polyline(model, set_coord = p[b], is_closed = True, set_crs = crs.uuid, title = 'v cell')
        if not hull.is_convex():
            continue
        # test the Voronoi diagram with the unit square area of interest
        c, v = tri.voronoi(p, t, b, aoi)
        assert len(v) == n
        # check that the areas of the Voronoi cells sum to the area of interest
        area = 0.0
        for nodes in v:
            v_cell = rql.Polyline(model, set_coord = c[nodes], is_closed = True, set_crs = crs.uuid, title = 'v cell')
            area += v_cell.area()
        assert maths.isclose(area, 1.0, rel_tol = 0.001)
        # test the Voronoi diagram with the heptagonal area of interest
        c_hept, v_hept = tri.voronoi(p, t, b, aoi_heptagon)
        assert len(v_hept) == n
        area = 0.0
        for nodes in v_hept:
            v_cell = rql.Polyline(model,
                                  set_coord = c_hept[nodes],
                                  is_closed = True,
                                  set_crs = crs.uuid,
                                  title = 'v cell')
            area += v_cell.area()
        assert maths.isclose(area, aoi_heptagon_area, rel_tol = 0.001)
        # test re-triangulation of a Voronoi diagram, passing centre points
        points, triangles = tri.triangulated_polygons(c_hept, v_hept, centres = p)
        assert len(points) == len(c_hept) + n
        assert len(triangles) == sum(len(vh) for vh in v_hept)
        # test re-triangulation of a Voronoi diagram, computing centre points
        points_2, triangles_2 = tri.triangulated_polygons(c_hept, v_hept)
        assert len(points_2) == len(c_hept) + n
        assert len(triangles_2) == sum(len(vh) for vh in v_hept)
        assert np.all(triangles_2 == triangles)
        for cell in range(n):
            v_cell = rql.Polyline(model,
                                  set_coord = c_hept[v_hept[cell]],
                                  is_closed = True,
                                  set_crs = crs.uuid,
                                  title = 'v cell')
            assert v_cell.point_is_inside_xy(points_2[len(c_hept) + cell])


def test_delaunay_triagulation():
    # Arrange
    points = np.array([
        [0.84500347, 0.84401839],
        [0.86625247, 0.05204284],
        [0.1220099, 0.56185864],
        [0.35034472, 0.02159957],
        [0.23992621, 0.23115569],
        [0.08040452, 0.75776318],
        [0.28902254, 0.86261331],
        [0.85916324, 0.63093401],
        [0.87953801, 0.27585486],
        [0.10630805, 0.46939924],
    ])

    def sort_array(array):
        return np.sort(array)[np.lexsort((np.sort(array)[:, 2], np.sort(array)[:, 1], np.sort(array)[:, 0]))]

    # Act
    tri_simple, hull_indices_simple = tri._dt_simple(points)
    tri_scipy, hull_indices_scipy = tri._dt_scipy(points)

    # Assert
    np.testing.assert_array_equal(sort_array(tri_simple), sort_array(tri_scipy))
    np.testing.assert_array_equal(hull_indices_simple, hull_indices_scipy)


def test_reorient():
    points = np.array([
        [0.84500347, 0.84401839, 0.0],
        [0.86625247, 0.05204284, 0.0],
        [0.1220099, 0.56185864, 0.0],
        [0.35034472, 0.02159957, 0.0],
        [0.23992621, 0.23115569, 0.0],
        [0.08040452, 0.75776318, 0.0],
        [0.28902254, 0.86261331, 0.0],
        [0.85916324, 0.63093401, 0.0],
        [0.87953801, 0.27585486, 0.0],
        [0.10630805, 0.46939924, 0.0],
    ])
    points[:, 2] = points[:, 0] + 0.5 * points[:, 1] + 0.1 * np.random.random(len(points))

    la_p, la_nv, la_m = tri.reorient(points, rough = False, use_linalg = True)
    old_p, old_nv, old_m = tri.reorient(points, rough = False, use_linalg = False)

    assert all([x is not None for x in [la_p, la_nv, la_m, old_p, old_nv, old_m]])
    assert la_p.shape == points.shape and old_p.shape == points.shape
    assert la_nv.shape == (3,) and old_nv.shape == (3,)
    assert la_m.shape == (3, 3) and old_m.shape == (3, 3)
    if la_nv[2] * old_nv[2] < 0.0:
        old_nv = -old_nv
    la_incl = vec.inclination(la_nv)
    la_azi = vec.azimuth(la_nv)
    old_incl = vec.inclination(old_nv)
    old_azi = vec.azimuth(old_nv)
    assert abs(la_incl - old_incl) < 5.0
    d_azi = abs(la_azi - old_azi)
    if d_azi > 180.0:
        d_azi = 360.0 - d_azi
    assert d_azi < 5.0


def test_reorient_vertical_pointset():
    points = np.array([(0.0, 0.0, 1000.0), (100.0, 100.0, 1000.0), (-100.0, -100.0, 1000.0), (0.0, 0.0, 1100.0),
                       (100.0, 100.0, 1100.0), (-100.0, -100.0, 1100.0)],
                      dtype = float)
    p, nv, m = tri.reorient(points, rough = False)
    if nv[0] < 0.0:
        nv = -nv
    assert_array_almost_equal(nv, (1.0 / maths.sqrt(2.0), -1.0 / maths.sqrt(2.0), 0.0), decimal = 2)
    np.max(p[:, 2]) - np.min(p[:, 2]) < 5.0


def test_reorient_vertical_pointset_2():
    points = np.array([[941.42146238, 234.31468119, 120.47005384], [234.31468119, 941.42146238, 120.47005384],
                       [941.42146238, 234.31468119, 195.47005384], [234.31468119, 941.42146238, 195.47005384],
                       [941.42146238, 234.31468119, 270.47005384], [234.31468119, 941.42146238, 270.47005384],
                       [587.86807178, 587.86807178, 157.97005384], [587.86807178, 587.86807178, 232.97005384]],
                      dtype = float)
    p, nv, m = tri.reorient(points, rough = False, use_linalg = True)
    assert_array_almost_equal(nv, (-1.0 / maths.sqrt(2.0), -1.0 / maths.sqrt(2.0), 0.0), decimal = 2)
    np.max(p[:, 2]) - np.min(p[:, 2]) < 5.0


def test_edges_and_rims():
    p = np.zeros((5, 6, 3), dtype = float)
    p[:, :, 0] = np.expand_dims(100.0 * np.arange(6, dtype = int).astype(float), axis = 0)
    p[:, :, 1] = np.expand_dims(100.0 * np.arange(5, dtype = int).astype(float), axis = 1)
    p = p.reshape((-1, 3))
    t = np.empty((4, 5, 2, 3), dtype = int)
    for j in range(4):
        for i in range(5):
            t[j, i, 0, 0] = 6 * j + i
            t[j, i, 0, 1] = 6 * j + i + 1
            t[j, i, 0, 2] = 6 * (j + 1) + i
            t[j, i, 1, 0] = 6 * (j + 1) + i + 1
            t[j, i, 1, 1] = 6 * j + i + 1
            t[j, i, 1, 2] = 6 * (j + 1) + i
    assert np.all(t >= 0) and np.all(t < len(p))
    t = t.reshape((-1, 2, 3))
    mask = np.ones((4, 5), dtype = bool)
    mask[1, 1] = False
    mask[2, 1] = False
    mask[1, 3] = False
    t = t[mask.flatten()].reshape((-1, 3))
    assert t.shape == (34, 3)

    e, c = tri.edges(t)
    assert e.shape == (6 * 4 + 5 * 5 + 20 - 4, 2)
    assert c.ndim == 1 and len(c) == len(e)
    assert np.all((c - 1) * (c - 2) == 0)  # each edge is used once or twice
    assert np.count_nonzero(c == 1) == 4 * 2 + 5 * 2 + 6 + 4

    re = tri.rim_edges(e, c)
    assert re.shape == (4 * 2 + 5 * 2 + 6 + 4, 2)
    ie = tri.internal_edges(e, c)
    assert ie.shape == (len(e) - re.shape[0], 2)

    rims_edge_list, rims_points_list = tri.rims(re)
    assert len(rims_edge_list) == 3
    assert len(rims_points_list) == 3
    for rer, rp in zip(rims_edge_list, rims_points_list):
        assert rer.ndim == 1 and rp.ndim == 1
        assert len(rer) == len(rp)
        assert not np.any(rp == 21) and not np.any(rp == 22)
        count = len(rp)
        assert count in (4, 6, 18)
        if count == 4:
            assert set(rp) == set((9, 10, 15, 16))
            trp = tuple(rp)
            if rp[0] == 9:
                assert trp == (9, 10, 16, 15) or trp == (9, 15, 16, 10)
            elif rp[0] == 10:
                assert trp == (10, 16, 15, 9) or trp == (10, 9, 15, 16)
            elif rp[0] == 15:
                assert trp == (15, 9, 10, 16) or trp == (15, 16, 10, 9)
            else:
                assert trp == (16, 15, 9, 10) or trp == (16, 10, 9, 15)
        elif count == 6:
            assert set(rp) == set((7, 8, 13, 14, 19, 20))
        else:
            assert set(rp) == (set(np.arange(6, dtype = int)) | set(24 + np.arange(6, dtype = int)) | set(
                (6, 12, 18, 11, 17, 23)))


def test_internal_edges():
    t = np.array([(0, 3, 4), (0, 1, 4), (1, 2, 4)], dtype = int)
    e, c = tri.edges(t)
    assert len(e) == 7
    ie = tri.internal_edges(e, c)
    assert ie.shape == (2, 2)
    assert np.all(ie == np.array([(0, 4), (1, 4)], dtype = int))


def test_triangles_using_point():
    p = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0)], dtype = float)
    t = np.array([(0, 3, 4), (0, 1, 4), (1, 2, 4)], dtype = int)
    assert tuple(tri.triangles_using_point(t, 1)) == (1, 2)
    assert tuple(tri.triangles_using_point(t, 2)) == (2,)
    assert np.all(tri.triangles_using_point(t, 4) == [0, 1, 2])
    assert tuple(tri.triangles_using_point(t, 3)) == (0,)
    assert len(tri.triangles_using_point(t, 5)) == 0


def test_triangles_using_edge():
    p = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0)], dtype = float)
    t = np.array([(0, 3, 4), (0, 1, 4), (1, 2, 4)], dtype = int)
    assert set(tri.triangles_using_edge(t, 1, 4)) == set((1, 2))
    assert set(tri.triangles_using_edge(t, 4, 1)) == set((1, 2))
    assert set(tri.triangles_using_edge(t, 4, 0)) == set((0, 1))
    assert set(tri.triangles_using_edge(t, 0, 4)) == set((0, 1))
    assert set(tri.triangles_using_edge(t, 3, 4)) == set((0,))
    assert set(tri.triangles_using_edge(t, 4, 3)) == set((0,))
    assert set(tri.triangles_using_edge(t, 4, 2)) == set((2,))
    assert len(tri.triangles_using_edge(t, 1, 3)) == 0


def test_triangles_using_edges():
    p = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0)], dtype = float)
    t = np.array([(0, 3, 4), (0, 1, 4), (1, 2, 4)], dtype = int)
    edges = np.array([(1, 4), (4, 1), (4, 0), (0, 4), (3, 4), (4, 3), (4, 2), (1, 3)], dtype = int)
    e = np.array([(1, 2), (1, 2), (0, 1), (0, 1), (0, -1), (0, -1), (2, -1), (-1, -1)], dtype = int)
    tue = tri.triangles_using_edges(t, edges)
    assert np.all(tue == e)


def test_first_false():
    a = np.array((0, 0, 1, 0, 1, 1), dtype = bool)
    assert tri._first_false(a) == 0
    a = np.logical_not(a)
    assert tri._first_false(a) == 2
    a = np.zeros(5, dtype = bool)
    assert tri._first_false(a) == 0
    a = np.logical_not(a)
    assert tri._first_false(a) == 5  # indication of no False values present


def test_first_match():
    a = np.array((0, 0, 1, 0, 1, 1), dtype = bool)
    assert tri._first_match(a, False) == 0
    assert tri._first_match(a, True) == 2
    a = np.arange(7, dtype = int)
    assert tri._first_match(a, 5) == 5
    assert tri._first_match(a, -1) == 7
    assert tri._first_match(a, 0) == 0
    assert tri._first_match(a, 13487) == 7


def test_first_unused():
    ap = np.array([(2, 5), (5, 3), (7, 2), (3, 5), (5, 4), (3, 6), (5, 8)], dtype = int)
    used = np.array((True, True, False, True, False, False, False))
    i, v = tri._find_unused(ap, used, 0)
    assert i == 7 and v == -1  # not found
    i, v = tri._find_unused(ap, used, 2)
    assert i == 2 and v == 7  # not found
    i, v = tri._find_unused(ap, used, 7)
    assert i == 2 and v == 2
    i, v = tri._find_unused(ap, used, 8)
    assert i == 6 and v == 5
    i, v = tri._find_unused(ap, used, 5)
    assert i == 4 and v == 4


def test_surrounding_xy_ring():
    p = np.array([(-23.0, -23.0, 457.0), (-23.0, 23.0, 457.0), (23.0, 23.0, 457.0), (23.0, -23.0, 457.0)],
                 dtype = float)
    P_radius = maths.sqrt(2 * 23.0 * 23.0)
    for n in (5, 17):
        ring = tri.surrounding_xy_ring(p, count = n, radial_factor = 10.0, inner_ring = False)
        assert ring.shape == (n, 3)
        assert_array_almost_equal(ring[:, 2], 457.0)
        r = np.sqrt(ring[:, 0] * ring[:, 0] + ring[:, 1] * ring[:, 1])
        assert_array_almost_equal(r, 10.0 * P_radius)
    for n in (6, 11):
        ring = tri.surrounding_xy_ring(p, count = n, radial_factor = 2.0, radial_distance = 234.0, inner_ring = True)
        assert ring.shape == (3 * n, 3)
        assert_array_almost_equal(ring[:, 2], 457.0)
        r = np.sqrt(ring[:, 0] * ring[:, 0] + ring[:, 1] * ring[:, 1])
        assert_array_almost_equal(r[:2 * n], 1.1 * P_radius)  # inner ring points
        assert_array_almost_equal(r[2 * n:], 234.0)
