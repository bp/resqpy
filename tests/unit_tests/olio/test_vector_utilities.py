# test vector utilities

import math as maths

import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.vector_utilities as vec


def test_simple_functions():
    assert_array_almost_equal(vec.zero_vector(), (0.0, 0.0, 0.0))
    assert_array_almost_equal(vec.add((1.2, 3.4), (5.6, 7.8)), np.array([6.8, 11.2]))
    assert_array_almost_equal(vec.subtract((5.6, 7.8, 10.0), (1.2, 3.4, -1.0)), np.array([4.4, 4.4, 11.0]))
    assert_array_almost_equal(vec.elemental_multiply((1.2, 3.4), (5.6, 7.8)), (6.72, 26.52))
    assert_array_almost_equal(vec.amplify((1.0, 2.0, 4.5), 2.5), (2.5, 5.0, 11.25))
    assert_array_almost_equal(vec.v_3d((23.4, -98.7)), (23.4, -98.7, 0.0))
    assert_array_almost_equal(vec.v_3d((23.4, 45.4, -98.7)), np.array((23.4, 45.4, -98.7)))


def test_unit_vectors():
    v_set = np.array([(3.0, 4.0, 0.0), (3.7, -3.7, 3.7), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])
    one_over_root_three = 1.0 / maths.sqrt(3.0)
    expected = np.array([(3.0 / 5.0, 4.0 / 5.0, 0.0), (one_over_root_three, -one_over_root_three, one_over_root_three),
                         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])
    for v, e in zip(v_set, expected):
        assert_array_almost_equal(vec.unit_vector(v), e)
    assert_array_almost_equal(vec.unit_vectors(v_set), expected)
    azi = [0.0, -90.0, 120.0, 180.0, 270.0, 360.0]
    expected = np.array([(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (maths.cos(maths.pi / 6), -0.5, 0.0), (0.0, -1.0, 0.0),
                         (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    for a, e in zip(azi, expected):
        assert_array_almost_equal(vec.unit_vector_from_azimuth(a), e)


def test_nan_unit_vectors():
    v_set = np.array([(3.0, 4.0, 0.0), (3.7, -3.7, np.NaN), (0.0, 0.0, 1.0), (0.0, np.NaN, 0.0)])
    one_over_root_three = 1.0 / maths.sqrt(3.0)
    expected = np.array([(3.0 / 5.0, 4.0 / 5.0, 0.0), (np.NaN, np.NaN, np.NaN), (0.0, 0.0, 1.0),
                         (np.NaN, np.NaN, np.NaN)])
    for v, e in zip(v_set, expected):
        assert_array_almost_equal(vec.unit_vector(v), e)


def test_angles():
    pi_by_2 = maths.pi / 2.0
    assert maths.isclose(vec.radians_from_degrees(90.0), pi_by_2)
    assert_array_almost_equal(vec.radians_from_degrees((0.0, -90.0, 120.0, 270.0)),
                              (0.0, -pi_by_2, maths.pi * 2 / 3, pi_by_2 * 3))
    assert maths.isclose(vec.radians_from_degrees(vec.degrees_from_radians(1.2647)), 1.2647)
    assert_array_almost_equal(vec.degrees_from_radians((0.0, -pi_by_2, maths.pi * 2 / 3, pi_by_2 * 3)),
                              (0.0, -90.0, 120.0, 270.0))
    assert maths.isclose(vec.degrees_from_radians(maths.pi), 180.0)


def test_azimuth():
    vectors = np.array([
        (0.0, 1.0, 0.0),  # north
        (27.3, 0.0, 2341.0),  # east
        (0.0, -4.5e6, -12000.0),  # south
        (-0.001, 0.0, 23.2),  # west
        (0.0, 0.0, 100.0),  # no bearing (expected to return 0.0 for azimuth(), NaN for azimuths())
        (123.45, -123.45, 0.03),  # south east
        (maths.cos(maths.radians(30.0)), 0.5, 0.0)  # bearing of 60 degrees
    ])
    expected = np.array((0.0, 90.0, 180.0, 270.0, 0.0, 135.0, 60.0))
    for v, e in zip(vectors, expected):
        assert maths.isclose(vec.azimuth(v), e)
        assert maths.isclose(vec.azimuth(v[:2]), e)
    expected[4] = np.NaN
    assert_array_almost_equal(vec.azimuths(vectors), expected)
    assert_array_almost_equal(vec.azimuths(vectors[:, :2]), expected)


def test_inclination():
    vectors = np.array([(0.0, 0.0, 1.0), (0.0, 0.0, -458.21), (0.0, 233.67, 233.67), (910.0, 0.0, 910.0),
                        (0.0, 0.0, 0.0)])
    expected = (0.0, 180.0, 45.0, 45.0, 90.0)  # last is an arbitrary value returned by the function for zero vector
    for v, e in zip(vectors, expected):
        assert maths.isclose(vec.inclination(v), e)
    assert maths.isclose(vec.inclination((456.7, -456.7, 456.7)),
                         90.0 - vec.degrees_from_radians(maths.acos(maths.sqrt(2) / maths.sqrt(3))))


def test_clockwise():
    p1 = (4.4, 4.4)
    p2 = (6.8, 6.8)
    p3 = (-21.0, -21.0)
    assert maths.isclose(vec.clockwise(p1, p2, p3), 0.0)
    p1 = [1.0, 1.0, -45.0]
    p2 = [2.0, 8.0, 23.1]
    p3 = [5.0, 3.0, -11.0]
    assert vec.clockwise(p1, p2, p3) > 0.0
    p1, p2 = np.array(p2), np.array(p1)
    p3 = np.array(p3)
    assert vec.clockwise(p1, p2, p3) < 0.0


def test_clockwise_triangles():
    p = np.array([(1.0, 1.0, 1.0), (2.0, 2.0, -1.0), (1.0, 2.0, 0.0), (2.0, 1.0, 0.0)])
    t = np.array([(0, 1, 2), (0, 1, 3)], dtype = int)
    e_xy = np.array([-1.0, 1.0])
    assert np.all(vec.clockwise_triangles(p, t) * e_xy > 0.0)
    e_xz = np.array([1.0, -1.0])
    assert np.all(vec.clockwise_triangles(p, t, projection = 'xz') * e_xz > 0.0)
    e_yz = np.array([-1.0, 1.0])
    assert np.all(vec.clockwise_triangles(p, t, projection = 'yz') * e_yz > 0.0)


def test_points_in_triangles():
    p = np.array([(1.0, 1.0, 1.0), (2.0, 2.0, -1.0), (1.0, 2.0, 0.0), (2.0, 1.0, 0.0)])
    t = np.array([(0, 1, 2), (0, 1, 3)], dtype = int)
    d = np.array([(0.5, 1.5, 0.0), (1.25, 1.75, 0.0), (1.75, 1.25, -3.0), (1.5, 1.5, 1.5), (1.0, 1.0, 3.0),
                  (1.5, 0.5, 1.0), (1.5, 2.5, 0.0), (0.5, 0.5, 0.5), (2.5, 1.5, -1.0)])
    e_no_edge = np.zeros((len(t), len(d)), dtype = bool)
    e_no_edge[0, 1] = True
    e_no_edge[1, 2] = True
    e_edge = e_no_edge.copy()
    e_edge[:, 3] = True
    e_edge[:, 4] = True
    r_no_edge = vec.points_in_triangles(p, t, d, projection = 'xy', edged = False)
    r_edge = vec.points_in_triangles(p, t, d, projection = 'xy', edged = True)
    assert np.all(r_no_edge == e_no_edge)
    assert np.all(r_edge == e_edge)


def test_point_in_triangle():
    p = np.array([[1.0, 2.0], [1.0, 6.0], [5.0, 6.0], [5.0, 2.0], [3.0, 4.0]])
    t = np.empty((3, 2), dtype = float)
    e = 1.0e-6
    t[0] = p[0]
    t[1] = p[1]
    t[2] = p[4]
    assert vec.point_in_triangle(p[0, 0] + e, p[0, 1] + 2.0 * e,
                                 t)  # actually right on a vertex, so might fail due to precision
    assert vec.point_in_triangle(p[1, 0] + e, p[1, 1] - 2.0 * e,
                                 t)  # actually right on a vertex, so might fail due to precision
    assert vec.point_in_triangle(p[4, 0] - e, p[4, 1], t)  # actually right on a vertex, so might fail due to precision
    for pi in [2, 3]:
        assert not vec.point_in_triangle(p[pi, 0], p[pi, 1], t)
    assert vec.point_in_triangle(2.0, 3.5, t)
    assert not vec.point_in_triangle(0.0, 0.0, t)
    assert not vec.point_in_triangle(0.0, 4.0, t)
    assert not vec.point_in_triangle(7.0, 4.0, t)
    assert not vec.point_in_triangle(2.0, 0.0, t)
    assert not vec.point_in_triangle(2.0, 7.0, t)
    assert not vec.point_in_triangle(2.0, 2.5, t)
    assert not vec.point_in_triangle(2.0, 5.5, t)
    t[0] = p[2]
    t[1] = p[1]
    t[2] = p[4]
    assert vec.point_in_triangle(p[2, 0] - 2.0 * e, p[2, 1] - e,
                                 t)  # actually right on a vertex, so might fail due to precision
    assert vec.point_in_triangle(p[1, 0] + 2.0 * e, p[1, 1] - e,
                                 t)  # actually right on a vertex, so might fail due to precision
    assert vec.point_in_triangle(p[4, 0], p[4, 1] + e, t)  # actually right on a vertex, so might fail due to precision
    for pi in [0, 3]:
        assert not vec.point_in_triangle(p[pi, 0], p[pi, 1], t)
    assert vec.point_in_triangle(3.0, 5.5, t)
    assert not vec.point_in_triangle(0.0, 0.0, t)
    assert not vec.point_in_triangle(0.0, 4.0, t)
    assert not vec.point_in_triangle(7.0, 4.0, t)
    assert not vec.point_in_triangle(2.0, 0.0, t)
    assert not vec.point_in_triangle(2.0, 7.0, t)
    assert not vec.point_in_triangle(2.0, 4.5, t)
    assert not vec.point_in_triangle(4.0, 4.5, t)


def test_is_obtuse_2d():
    p = np.array((0.0, 0.0))
    p1, p2 = np.array((5.0, 0.01)), np.array((0.01, 5.0))
    assert not vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = np.array((10.0, 9.7)), np.array((9.9, 10.0))
    assert not vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = -np.array((10.0, 9.7)), -np.array((9.9, 10.0))
    assert not vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = np.array((10.0, 9.7)), np.array((-1.0, 9.0))
    assert not vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = np.array((10.0, 9.7)), np.array((1.0, -9.0))
    assert vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = np.array((5.0, 0.01)), np.array((-2.0, 5.0))
    assert vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = np.array((-5.0, 0.01)), np.array((5.0, 1.0))
    assert vec.is_obtuse_2d(p, p1, p2)
    p1, p2 = np.array((5.0, -0.01)), np.array((0.0, 10.0))
    assert vec.is_obtuse_2d(p, p1, p2)


def test_point_distance_to_line_segment_2d():
    l1, l2 = np.array((3.0, 3.0)), np.array((7.0, 7.0))
    p = np.array((3.0, 3.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), 0.0)
    p = np.array((5.0, 5.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), 0.0)
    p = np.array((7.0, 7.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), 0.0)
    p = np.array((8.0, 8.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), maths.sqrt(2.0))
    p = np.array((0.0, -1.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), 5.0)
    p = np.array((15.0, 1.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), 10.0)
    p = np.array((9.0, 5.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), 2.0 * maths.sqrt(2.0))
    p = np.array((3.0, 5.0))
    assert maths.isclose(vec.point_distance_to_line_segment_2d(p, l1, l2), maths.sqrt(2.0))


def test_rotation():
    x = 47.3
    y = -32.9
    p = np.array((1234.2, 106.7, 742.5))
    m = vec.rotation_3d_matrix((x, 0.0, y))
    rm = vec.reverse_rotation_3d_matrix((x, 0.0, y))
    pp = vec.rotate_vector(m, p)
    ppp = vec.rotate_vector(rm, pp)
    assert_array_almost_equal(p, ppp)


def test_vector_rotation():
    v = vec.unit_vector((3.0, 4.0, 5.0))
    m = vec.rotation_matrix_3d_vector(v)
    assert_array_almost_equal(v, vec.rotate_vector(m, (0.0, 0.0, 1.0)))


def test_points_in_triangles_aligned_optimised():
    # Arrange
    nx = ny = 20
    dx = dy = 0.5
    triangles = np.array([
        [[4.271, 4.992], [1.295, 8.921], [7.201, 9.822]],
        [[8.182, 0.832], [7.384, 5.939], [2.302, 2.039]],
    ])

    def sort_array(array):
        array = array[array[:, 2].argsort()]
        array = array[array[:, 1].argsort(kind = 'mergesort')]
        array = array[array[:, 0].argsort(kind = 'mergesort')]
        return array

    # Act
    triangles_points = vec.points_in_triangles_aligned(nx, ny, dx, dy, triangles)
    triangles_points_optimised = vec.points_in_triangles_aligned_optimised(nx, ny, dx, dy, triangles)

    # Assert
    np.testing.assert_array_almost_equal(sort_array(triangles_points), sort_array(triangles_points_optimised))


def test_points_in_triangles_aligned_optimised_point_on_triangle_side():
    # Arrange
    nx = ny = 10
    dx = dy = 1
    triangles = np.array([
        [[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
    ])

    # Act
    triangles_points = vec.points_in_triangles_aligned_optimised(nx, ny, dx, dy, triangles)

    # Assert
    np.testing.assert_array_almost_equal(triangles_points, np.array([[0, 0, 1], [1, 1, 1]]))


def test_points_in_triangles_aligned_optimised_surface_outside_grid():
    # Arrange
    nx = ny = 10
    dx = dy = 0.01
    triangles = np.array([
        [[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
    ])

    # Act
    triangles_points = vec.points_in_triangles_aligned_optimised(nx, ny, dx, dy, triangles)

    # Assert
    np.testing.assert_array_almost_equal(triangles_points, np.empty((0, 3)))


def test_unit_vector_from_azimuth_and_inclination():
    one_over_root_two = 1.0 / maths.sqrt(2.0)
    root_three_over_two = maths.sqrt(3.0) / 2.0
    v = vec.unit_vector_from_azimuth_and_inclination(0.0, 0.0)
    assert_array_almost_equal(v, (0.0, 0.0, 1.0))
    v = vec.unit_vector_from_azimuth_and_inclination(220.0, 0.0)
    assert_array_almost_equal(v, (0.0, 0.0, 1.0))
    v = vec.unit_vector_from_azimuth_and_inclination(0.0, 45.0)
    assert_array_almost_equal(v, (0.0, one_over_root_two, one_over_root_two))
    v = vec.unit_vector_from_azimuth_and_inclination(90.0, 45.0)
    assert_array_almost_equal(v, (one_over_root_two, 0.0, one_over_root_two))
    v = vec.unit_vector_from_azimuth_and_inclination(180.0, 45.0)
    assert_array_almost_equal(v, (0.0, -one_over_root_two, one_over_root_two))
    v = vec.unit_vector_from_azimuth_and_inclination(270.0, 60.0)
    assert_array_almost_equal(v, (-root_three_over_two, 0.0, 0.5))
    v = vec.unit_vector_from_azimuth_and_inclination(45.0, 90.0)
    assert_array_almost_equal(v, (one_over_root_two, one_over_root_two, 0.0))


def test_xy_sort():
    py = np.array([(5.0, 4.0, 1.0), (3.0, 7.0, 2.0), (1.0, 0.0, 3.0)], dtype = float)
    spy, ay = vec.xy_sorted(py)
    espy = np.array([(1.0, 0.0, 3.0), (5.0, 4.0, 1.0), (3.0, 7.0, 2.0)], dtype = float)
    px = py.copy()
    px[0, 0] = 10.0
    espx = np.array([(1.0, 0.0, 3.0), (3.0, 7.0, 2.0), (10.0, 4.0, 1.0)], dtype = float)
    spx, ax = vec.xy_sorted(px)
    espxy = np.array([(1.0, 0.0, 3.0), (10.0, 4.0, 1.0), (3.0, 7.0, 2.0)], dtype = float)
    spxy, axy = vec.xy_sorted(px, axis = 1)
    assert ax == 0
    assert ay == 1
    assert axy == 1
    assert_array_almost_equal(spx, espx)
    assert_array_almost_equal(spy, espy)
    assert_array_almost_equal(spxy, espxy)


def test_inclinations():
    v_set = np.array([(3.0, 4.0, 0.0), (0.0, -3.7, 3.7), (0.0, 0.0, 1.0), (100.0 * maths.sqrt(3.0) / 2.0, 0.0, -50.0)],
                     dtype = float)
    e = np.array([90.0, 45.0, 0.0, 120.0], dtype = float)
    inclines = vec.inclinations(v_set)
    assert_array_almost_equal(inclines, e)


def test_nan_inclinations():
    v_set = np.array([(3.0, np.NaN, 0.0), (0.0, -3.7, 3.7), (np.NaN, 0.0, 1.0),
                      (100.0 * maths.sqrt(3.0) / 2.0, 0.0, -50.0)],
                     dtype = float)
    e = np.array([np.NaN, 45.0, np.NaN, 120.0], dtype = float)
    inclines = vec.nan_inclinations(v_set)
    assert_array_almost_equal(inclines, e)


def test_points_direction_vector():
    p = np.zeros((3, 5, 3), dtype = float)
    p[0, :, 0] = 100.0 * np.arange(5).astype(float)
    p[1, :, 0] = p[0, :, 0] + 100.0
    p[2, :, 0] = p[0, :, 0] - 100.0
    p[:, :, 1] = np.expand_dims(50.0 * np.arange(3).astype(float), axis = -1)
    p[:, :, 2] = p[:, :, 0] + p[:, :, 1]
    e_0 = np.mean(p[2], axis = 0) - np.mean(p[0], axis = 0)
    e_1 = np.mean(p[:, 4], axis = 0) - np.mean(p[:, 0], axis = 0)
    pdv_0 = vec.points_direction_vector(p, axis = 0)
    pdv_1 = vec.points_direction_vector(p, axis = 1)
    assert_array_almost_equal(pdv_0, e_0)
    assert_array_almost_equal(pdv_1, e_1)


def test_points_direction_vector_some_nan():
    p = np.zeros((3, 5, 3), dtype = float)
    p[0, :, 0] = 100.0 * np.arange(5).astype(float)
    p[1, :, 0] = p[0, :, 0] + 100.0
    p[2, :, 0] = p[0, :, 0] - 100.0
    p[:, :, 1] = np.expand_dims(50.0 * np.arange(3).astype(float), axis = -1)
    p[:, :, 2] = p[:, :, 0] + p[:, :, 1]
    p[0, 2, :] = np.NaN
    p[1, 4, :] = np.NaN
    e_0 = np.nanmean(p[2], axis = 0) - np.nanmean(p[0], axis = 0)
    e_1 = np.nanmean(p[:, 4], axis = 0) - np.nanmean(p[:, 0], axis = 0)
    pdv_0 = vec.points_direction_vector(p, axis = 0)
    pdv_1 = vec.points_direction_vector(p, axis = 1)
    assert_array_almost_equal(pdv_0, e_0)
    assert_array_almost_equal(pdv_1, e_1)


def test_points_direction_vector_nan_sloces():
    p = np.zeros((3, 5, 3), dtype = float)
    p[0, :, 0] = 100.0 * np.arange(5).astype(float)
    p[1, :, 0] = p[0, :, 0] + 100.0
    p[2, :, 0] = p[0, :, 0] - 100.0
    p[:, :, 1] = np.expand_dims(50.0 * np.arange(3).astype(float), axis = -1)
    p[:, :, 2] = p[:, :, 0] + p[:, :, 1]
    p[0, :, :] = np.NaN
    p[:, 4, :] = np.NaN
    e_0 = np.nanmean(p[2], axis = 0) - np.nanmean(p[1], axis = 0)
    e_1 = np.nanmean(p[:, 3], axis = 0) - np.nanmean(p[:, 0], axis = 0)
    pdv_0 = vec.points_direction_vector(p, axis = 0)
    pdv_1 = vec.points_direction_vector(p, axis = 1)
    assert_array_almost_equal(pdv_0, e_0)
    assert_array_almost_equal(pdv_1, e_1)
