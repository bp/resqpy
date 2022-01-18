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
    aoi = rql.Polyline(model, set_coord = aoi_xyz, set_bool = True, set_crs = crs.uuid, title = 'aoi')
    # and an alternative area of interest
    aoi_heptagon_xyz = np.zeros((7, 3))
    aoi_heptagon_xyz[:, :2] = ((-0.5, -1.0), (-1.5, 0.5), (-1.0, 1.7), (0.5, 2.0), (2.0, 1.7), (2.5, 0.5), (1.5, -1.0))
    aoi_heptagon = rql.Polyline(model,
                                set_coord = aoi_heptagon_xyz,
                                set_bool = True,
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
        hull = rql.Polyline(model, set_coord = p[b], set_bool = True, set_crs = crs.uuid, title = 'v cell')
        if not hull.is_convex():
            continue
        # test the Voronoi diagram with the unit square area of interest
        c, v = tri.voronoi(p, t, b, aoi)
        assert len(v) == n
        # check that the areas of the Voronoi cells sum to the area of interest
        area = 0.0
        for nodes in v:
            v_cell = rql.Polyline(model, set_coord = c[nodes], set_bool = True, set_crs = crs.uuid, title = 'v cell')
            area += v_cell.area()
        assert maths.isclose(area, 1.0, rel_tol = 0.001)
        # test the Voronoi diagram with the heptagonal area of interest
        c_hept, v_hept = tri.voronoi(p, t, b, aoi_heptagon)
        assert len(v_hept) == n
        area = 0.0
        for nodes in v_hept:
            v_cell = rql.Polyline(model,
                                  set_coord = c_hept[nodes],
                                  set_bool = True,
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
                                  set_bool = True,
                                  set_crs = crs.uuid,
                                  title = 'v cell')
            assert v_cell.point_is_inside_xy(points_2[len(c_hept) + cell])
