import numpy as np
import pytest

import resqpy.grid._points_functions as pf


def test_uncache_points(basic_regular_grid):
    # Act & Assert
    assert basic_regular_grid.points_cached is not None

    pf.uncache_points(basic_regular_grid)

    assert basic_regular_grid.points_cached is None


def test_horizon_points_default(basic_regular_grid):
    # Arrange
    expected_horizon_points = np.array([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
                                        [[0.0, 50.0, 0.0], [100.0, 50.0, 0.0], [200.0, 50.0, 0.0]],
                                        [[0.0, 100.0, 0.0], [100.0, 100.0, 0.0], [200.0, 100.0, 0.0]]])

    # Act
    horizon_points = pf.horizon_points(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(horizon_points, expected_horizon_points)


@pytest.mark.parametrize("ref_k0, expected_horizon_points",
                         [(1,
                           np.array([[[0.0, 0.0, 20.0], [100.0, 0.0, 20.0], [200.0, 0.0, 20.0]],
                                     [[0.0, 50.0, 20.0], [100.0, 50.0, 20.0], [200.0, 50.0, 20.0]],
                                     [[0.0, 100.0, 20.0], [100.0, 100.0, 20.0], [200.0, 100.0, 20.0]]])),
                          (2,
                           np.array([[[0.0, 0.0, 40.0], [100.0, 0.0, 40.0], [200.0, 0.0, 40.0]],
                                     [[0.0, 50.0, 40.0], [100.0, 50.0, 40.0], [200.0, 50.0, 40.0]],
                                     [[0.0, 100.0, 40.0], [100.0, 100.0, 40.0], [200.0, 100.0, 40.0]]]))])
def test_horizon_points_ref_k0(basic_regular_grid, ref_k0, expected_horizon_points):
    # Act
    horizon_points = pf.horizon_points(basic_regular_grid, ref_k0 = ref_k0)

    # Assert
    np.testing.assert_array_almost_equal(horizon_points, expected_horizon_points)


def test_horizon_points_k_gaps(s_bend_k_gap_grid):
    # Act
    horizon_points = pf.horizon_points(s_bend_k_gap_grid, ref_k0 = 1)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(horizon_points[3, 38], np.array([-4.486836, 24., 170.449719]))
    np.testing.assert_array_almost_equal(horizon_points[0, 26], np.array([14.3, 0., 146.248]))
    np.testing.assert_array_almost_equal(horizon_points[12, 19], np.array([70.641522, 96., 145.428053]))
    np.testing.assert_array_almost_equal(horizon_points[4, 50], np.array([57.84, 32., 181.503781]))


def test_horizon_points_k_gaps_kp(s_bend_k_gap_grid):
    # Act
    horizon_points = pf.horizon_points(s_bend_k_gap_grid, ref_k0 = 3, kp = 1)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(horizon_points[3, 38], np.array([-11.252623, 24., 172.912264]))
    np.testing.assert_array_almost_equal(horizon_points[0, 26], np.array([14.3, 0., 139.048]))
    np.testing.assert_array_almost_equal(horizon_points[12, 19], np.array([66.013451, 96., 139.912533]))
    np.testing.assert_array_almost_equal(horizon_points[4, 50], np.array([57.84, 32., 188.703781]))


def test_horizon_points_split_coordinate_lines(faulted_grid):
    # Arrange
    expected_horizon_points = np.array([[[1000., 2000., 3020.], [1100., 2000., 3020.], [1200., 2000., 3020.],
                                         [1300., 2000., 3020.], [1400., 2000., 3020.], [1500., 2000., 3020.],
                                         [1600., 2000., 3020.], [1700., 2000., 3020.], [1800., 2000., 3020.]],
                                        [[1000., 2100., 3020.], [1100., 2100., 3021.], [1200., 2100., 3021.],
                                         [1300., 2100., 3020.], [1400., 2100., 3016.5], [1500., 2100., 3021.],
                                         [1600., 2100., 3021.], [1700., 2100., 3021.], [1800., 2100., 3020.]],
                                        [[1000., 2200., 3020.], [1100., 2200., 3023.5], [1200., 2200., 3023.5],
                                         [1300., 2200., 3023.5], [1400., 2200., 3020.], [1500., 2200., 3023.5],
                                         [1600., 2200., 3023.5], [1700., 2200., 3023.5], [1800., 2200., 3020.]],
                                        [[1000., 2300., 3020.], [1100., 2300., 3019.], [1200., 2300., 3019.],
                                         [1300., 2300., 3019.], [1400., 2300., 3016.5], [1500., 2300., 3020.],
                                         [1600., 2300., 3019.], [1700., 2300., 3019.], [1800., 2300., 3020.]],
                                        [[1000., 2400., 3020.], [1100., 2400., 3020.], [1200., 2400., 3020.],
                                         [1300., 2400., 3019.], [1400., 2400., 3016.5], [1500., 2400., 3021.],
                                         [1600., 2400., 3020.], [1700., 2400., 3020.], [1800., 2400., 3020.]],
                                        [[1000., 2500., 3020.], [1100., 2500., 3020.], [1200., 2500., 3020.],
                                         [1300., 2500., 3020.], [1400., 2500., 3020.], [1500., 2500., 3020.],
                                         [1600., 2500., 3020.], [1700., 2500., 3020.], [1800., 2500., 3020.]]])

    # Act
    horizon_points = pf.horizon_points(faulted_grid, ref_k0 = 1)

    # Assert
    np.testing.assert_array_almost_equal(horizon_points, expected_horizon_points)


def test_x_section_corner_points_axis_I(basic_regular_grid):
    # Arrange
    expected_x_section_corner_points = np.array([[[[[0.0, 0.0, 0.0], [0.0, 50.0, 0.0]],
                                                   [[0.0, 0.0, 20.0], [0.0, 50.0, 20.0]]],
                                                  [[[0.0, 50.0, 0.0], [0.0, 100.0, 0.0]],
                                                   [[0.0, 50.0, 20.0], [0.0, 100.0, 20.0]]]],
                                                 [[[[0.0, 0.0, 20.0], [0.0, 50.0, 20.0]],
                                                   [[0.0, 0.0, 40.0], [0.0, 50.0, 40.0]]],
                                                  [[[0.0, 50.0, 20.0], [0.0, 100.0, 20.0]],
                                                   [[0.0, 50.0, 40.0], [0.0, 100.0, 40.0]]]]])

    # Act
    x_section_corner_points = pf.x_section_corner_points(basic_regular_grid, axis = 'I')

    # Assert
    np.testing.assert_array_almost_equal(x_section_corner_points, expected_x_section_corner_points)


def test_x_section_corner_points_axis_J(basic_regular_grid):
    # Arrange
    expected_x_section_corner_points = np.array([[[[[0., 0., 0.], [100., 0., 0.]], [[0., 0., 20.], [100., 0., 20.]]],
                                                  [[[100., 0., 0.], [200., 0., 0.]], [[100., 0., 20.], [200., 0.,
                                                                                                        20.]]]],
                                                 [[[[0., 0., 20.], [100., 0., 20.]], [[0., 0., 40.], [100., 0., 40.]]],
                                                  [[[100., 0., 20.], [200., 0., 20.]], [[100., 0., 40.],
                                                                                        [200., 0., 40.]]]]])

    # Act
    x_section_corner_points = pf.x_section_corner_points(basic_regular_grid, axis = 'J')

    # Assert
    np.testing.assert_array_almost_equal(x_section_corner_points, expected_x_section_corner_points)


def test_x_section_corner_points_k_gaps(s_bend_k_gap_grid):
    # Act
    x_section_corner_points = pf.x_section_corner_points(s_bend_k_gap_grid, axis = 'I')

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(x_section_corner_points[2, 7, 1, 1], np.array([11., 64., 128.08]))
    np.testing.assert_array_almost_equal(x_section_corner_points[3, 1, 0, 1], np.array([0., 16., 122.8]))
    np.testing.assert_array_almost_equal(x_section_corner_points[0, 11, 0, 1], np.array([14., 96., 119.92]))
    np.testing.assert_array_almost_equal(x_section_corner_points[3, 8, 1, 0], np.array([14., 64., 131.92]))


def test_x_section_corner_points_split_coordinate_lines_axis_I(faulted_grid):
    # Arrange
    expected_x_section_corner_points = np.array([[[[[1000., 2000., 3000.], [1000., 2100., 3000.]],
                                                   [[1000., 2000., 3020.], [1000., 2100., 3020.]]],
                                                  [[[1000., 2100., 3000.], [1000., 2200., 3000.]],
                                                   [[1000., 2100., 3020.], [1000., 2200., 3020.]]],
                                                  [[[1000., 2200., 3000.], [1000., 2300., 3000.]],
                                                   [[1000., 2200., 3020.], [1000., 2300., 3020.]]],
                                                  [[[1000., 2300., 3000.], [1000., 2400., 3000.]],
                                                   [[1000., 2300., 3020.], [1000., 2400., 3020.]]],
                                                  [[[1000., 2400., 3000.], [1000., 2500., 3000.]],
                                                   [[1000., 2400., 3020.], [1000., 2500., 3020.]]]],
                                                 [[[[1000., 2000., 3020.], [1000., 2100., 3020.]],
                                                   [[1000., 2000., 3040.], [1000., 2100., 3040.]]],
                                                  [[[1000., 2100., 3020.], [1000., 2200., 3020.]],
                                                   [[1000., 2100., 3040.], [1000., 2200., 3040.]]],
                                                  [[[1000., 2200., 3020.], [1000., 2300., 3020.]],
                                                   [[1000., 2200., 3040.], [1000., 2300., 3040.]]],
                                                  [[[1000., 2300., 3020.], [1000., 2400., 3020.]],
                                                   [[1000., 2300., 3040.], [1000., 2400., 3040.]]],
                                                  [[[1000., 2400., 3020.], [1000., 2500., 3020.]],
                                                   [[1000., 2400., 3040.], [1000., 2500., 3040.]]]],
                                                 [[[[1000., 2000., 3040.], [1000., 2100., 3040.]],
                                                   [[1000., 2000., 3050.], [1000., 2100., 3050.]]],
                                                  [[[1000., 2100., 3040.], [1000., 2200., 3040.]],
                                                   [[1000., 2100., 3050.], [1000., 2200., 3050.]]],
                                                  [[[1000., 2200., 3040.], [1000., 2300., 3040.]],
                                                   [[1000., 2200., 3050.], [1000., 2300., 3050.]]],
                                                  [[[1000., 2300., 3040.], [1000., 2400., 3040.]],
                                                   [[1000., 2300., 3050.], [1000., 2400., 3050.]]],
                                                  [[[1000., 2400., 3040.], [1000., 2500., 3040.]],
                                                   [[1000., 2400., 3050.], [1000., 2500., 3050.]]]]])

    # Act
    x_section_corner_points = pf.x_section_corner_points(faulted_grid, axis = 'I')

    # Assert
    np.testing.assert_array_almost_equal(x_section_corner_points, expected_x_section_corner_points)


def test_x_section_corner_points_split_coordinate_lines_axis_J(faulted_grid):
    # Arrange
    expected_x_section_corner_points = np.array([[[[[1000., 2000., 3000.], [1100., 2000., 3000.]],
                                                   [[1000., 2000., 3020.], [1100., 2000., 3020.]]],
                                                  [[[1100., 2000., 3000.], [1200., 2000., 3000.]],
                                                   [[1100., 2000., 3020.], [1200., 2000., 3020.]]],
                                                  [[[1200., 2000., 3000.], [1300., 2000., 3000.]],
                                                   [[1200., 2000., 3020.], [1300., 2000., 3020.]]],
                                                  [[[1300., 2000., 3000.], [1400., 2000., 3000.]],
                                                   [[1300., 2000., 3020.], [1400., 2000., 3020.]]],
                                                  [[[1400., 2000., 3000.], [1500., 2000., 3000.]],
                                                   [[1400., 2000., 3020.], [1500., 2000., 3020.]]],
                                                  [[[1500., 2000., 3000.], [1600., 2000., 3000.]],
                                                   [[1500., 2000., 3020.], [1600., 2000., 3020.]]],
                                                  [[[1600., 2000., 3000.], [1700., 2000., 3000.]],
                                                   [[1600., 2000., 3020.], [1700., 2000., 3020.]]],
                                                  [[[1700., 2000., 3000.], [1800., 2000., 3000.]],
                                                   [[1700., 2000., 3020.], [1800., 2000., 3020.]]]],
                                                 [[[[1000., 2000., 3020.], [1100., 2000., 3020.]],
                                                   [[1000., 2000., 3040.], [1100., 2000., 3040.]]],
                                                  [[[1100., 2000., 3020.], [1200., 2000., 3020.]],
                                                   [[1100., 2000., 3040.], [1200., 2000., 3040.]]],
                                                  [[[1200., 2000., 3020.], [1300., 2000., 3020.]],
                                                   [[1200., 2000., 3040.], [1300., 2000., 3040.]]],
                                                  [[[1300., 2000., 3020.], [1400., 2000., 3020.]],
                                                   [[1300., 2000., 3040.], [1400., 2000., 3040.]]],
                                                  [[[1400., 2000., 3020.], [1500., 2000., 3020.]],
                                                   [[1400., 2000., 3040.], [1500., 2000., 3040.]]],
                                                  [[[1500., 2000., 3020.], [1600., 2000., 3020.]],
                                                   [[1500., 2000., 3040.], [1600., 2000., 3040.]]],
                                                  [[[1600., 2000., 3020.], [1700., 2000., 3020.]],
                                                   [[1600., 2000., 3040.], [1700., 2000., 3040.]]],
                                                  [[[1700., 2000., 3020.], [1800., 2000., 3020.]],
                                                   [[1700., 2000., 3040.], [1800., 2000., 3040.]]]],
                                                 [[[[1000., 2000., 3040.], [1100., 2000., 3040.]],
                                                   [[1000., 2000., 3050.], [1100., 2000., 3050.]]],
                                                  [[[1100., 2000., 3040.], [1200., 2000., 3040.]],
                                                   [[1100., 2000., 3050.], [1200., 2000., 3050.]]],
                                                  [[[1200., 2000., 3040.], [1300., 2000., 3040.]],
                                                   [[1200., 2000., 3050.], [1300., 2000., 3040.]]],
                                                  [[[1300., 2000., 3040.], [1400., 2000., 3040.]],
                                                   [[1300., 2000., 3040.], [1400., 2000., 3040.]]],
                                                  [[[1400., 2000., 3040.], [1500., 2000., 3040.]],
                                                   [[1400., 2000., 3040.], [1500., 2000., 3040.]]],
                                                  [[[1500., 2000., 3040.], [1600., 2000., 3040.]],
                                                   [[1500., 2000., 3040.], [1600., 2000., 3050.]]],
                                                  [[[1600., 2000., 3040.], [1700., 2000., 3040.]],
                                                   [[1600., 2000., 3050.], [1700., 2000., 3050.]]],
                                                  [[[1700., 2000., 3040.], [1800., 2000., 3040.]],
                                                   [[1700., 2000., 3050.], [1800., 2000., 3050.]]]]])

    # Act
    x_section_corner_points = pf.x_section_corner_points(faulted_grid, axis = 'J')

    # Assert
    np.testing.assert_array_almost_equal(x_section_corner_points, expected_x_section_corner_points)


def test_x_section_corner_points_rotate_azimuth(basic_regular_grid):
    # Arrange
    expected_x_section_corner_points = np.array([[[[[0.0, 0.0, 0.0], [35.35533906, 35.35533906, 0.0]],
                                                   [[0.0, 0.0, 20.0], [35.35533906, 35.35533906, 20.0]]],
                                                  [[[35.35533906, 35.35533906, 0.0], [70.71067812, 70.71067812, 0.0]],
                                                   [[35.35533906, 35.35533906, 20.0], [70.71067812, 70.71067812,
                                                                                       20.0]]]],
                                                 [[[[0.0, 0.0, 20.0], [35.35533906, 35.35533906, 20.0]],
                                                   [[0.0, 0.0, 40.0], [35.35533906, 35.35533906, 40.0]]],
                                                  [[[35.35533906, 35.35533906, 20.0], [70.71067812, 70.71067812, 20.0]],
                                                   [[35.35533906, 35.35533906, 40.0], [70.71067812, 70.71067812,
                                                                                       40.0]]]]])

    # Act
    x_section_corner_points = pf.x_section_corner_points(basic_regular_grid, axis = 'I', rotate = True, azimuth = 45)

    # Assert
    np.testing.assert_array_almost_equal(x_section_corner_points, expected_x_section_corner_points)


def test_coordinate_line_end_points(basic_regular_grid):
    # Arrange
    expected_coordinate_line_end_points = np.array([[[[0.0, 0.0, 0.0], [0.0, 0.0, 40.0]],
                                                     [[100.0, 0.0, 0.0], [100.0, 0.0, 40.0]],
                                                     [[200.0, 0.0, 0.0], [200.0, 0.0, 40.0]]],
                                                    [[[0.0, 50.0, 0.0], [0.0, 50.0, 40.0]],
                                                     [[100.0, 50.0, 0.0], [100.0, 50.0, 40.0]],
                                                     [[200.0, 50.0, 0.0], [200.0, 50.0, 40.0]]],
                                                    [[[0.0, 100.0, 0.0], [0.0, 100.0, 40.0]],
                                                     [[100.0, 100.0, 0.0], [100.0, 100.0, 40.0]],
                                                     [[200.0, 100.0, 0.0], [200.0, 100.0, 40.0]]]])

    # Act
    coordinate_line_end_points = pf.coordinate_line_end_points(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(coordinate_line_end_points, expected_coordinate_line_end_points)


def test_coordinate_line_end_points_s_bend_faulted_grid(s_bend_faulted_grid):
    # Act
    coordinate_line_end_points = pf.coordinate_line_end_points(s_bend_faulted_grid)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(coordinate_line_end_points[2, 7, 1], np.array([54.104242, 16., 112.723689]))
    np.testing.assert_array_almost_equal(coordinate_line_end_points[11, 28, 1], np.array([14.132444, 88., 146.132614]))
    np.testing.assert_array_almost_equal(coordinate_line_end_points[8, 18, 0], np.array([76.685067, 64., 146.146903]))
    np.testing.assert_array_almost_equal(coordinate_line_end_points[5, 15, 1], np.array([67.117693, 40., 131.363778]))


def test_coordinate_line_end_points_s_bend_k_gep_grid(s_bend_k_gap_grid):
    # Act
    coordinate_line_end_points = pf.coordinate_line_end_points(s_bend_k_gap_grid)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(coordinate_line_end_points[2, 7, 1], np.array([54.104242, 16., 112.723689]))
    np.testing.assert_array_almost_equal(coordinate_line_end_points[11, 28, 1], np.array([14.132444, 88., 146.132614]))
    np.testing.assert_array_almost_equal(coordinate_line_end_points[8, 18, 0], np.array([76.685067, 64., 146.146903]))
    np.testing.assert_array_almost_equal(coordinate_line_end_points[5, 15, 1], np.array([67.117693, 40., 131.363778]))


def test_z_corner_point_depths(basic_regular_grid):
    # Arrange
    expected_z_corner_point_depths = np.array([[[[[[0.0, 0.0], [0.0, 0.0]], [[20.0, 20.0], [20.0, 20.0]]],
                                                 [[[0.0, 0.0], [0.0, 0.0]], [[20.0, 20.0], [20.0, 20.0]]]],
                                                [[[[0.0, 0.0], [0.0, 0.0]], [[20.0, 20.0], [20.0, 20.0]]],
                                                 [[[0.0, 0.0], [0.0, 0.0]], [[20.0, 20.0], [20.0, 20.0]]]]],
                                               [[[[[20.0, 20.0], [20.0, 20.0]], [[40.0, 40.0], [40.0, 40.0]]],
                                                 [[[20.0, 20.0], [20.0, 20.0]], [[40.0, 40.0], [40.0, 40.0]]]],
                                                [[[[20.0, 20.0], [20.0, 20.0]], [[40.0, 40.0], [40.0, 40.0]]],
                                                 [[[20.0, 20.0], [20.0, 20.0]], [[40.0, 40.0], [40.0, 40.0]]]]]])

    # Act
    z_corner_point_depths = pf.z_corner_point_depths(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(z_corner_point_depths, expected_z_corner_point_depths)


def test_z_corner_point_depths_linear(basic_regular_grid):
    # Arrange
    expected_z_corner_point_depths = np.array([[[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                                                 [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]],
                                                [[[[20.0, 20.0], [20.0, 20.0]], [[20.0, 20.0], [20.0, 20.0]]],
                                                 [[[20.0, 20.0], [20.0, 20.0]], [[20.0, 20.0], [20.0, 20.0]]]]],
                                               [[[[[20.0, 20.0], [20.0, 20.0]], [[20.0, 20.0], [20.0, 20.0]]],
                                                 [[[20.0, 20.0], [20.0, 20.0]], [[20.0, 20.0], [20.0, 20.0]]]],
                                                [[[[40.0, 40.0], [40.0, 40.0]], [[40.0, 40.0], [40.0, 40.0]]],
                                                 [[[40.0, 40.0], [40.0, 40.0]], [[40.0, 40.0], [40.0, 40.0]]]]]])

    # Act
    z_corner_point_depths = pf.z_corner_point_depths(basic_regular_grid, order = 'linear')

    # Assert
    np.testing.assert_array_almost_equal(z_corner_point_depths, expected_z_corner_point_depths)


def test_z_corner_point_depths_s_bend_faulted_grid(s_bend_faulted_grid):
    # Act
    z_corner_point_depths = pf.z_corner_point_depths(s_bend_faulted_grid)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(z_corner_point_depths[2, 7, 34, 0],
                                         np.array([[162.582062, 165.410711], [162.582062, 165.410711]]))
    np.testing.assert_array_almost_equal(z_corner_point_depths[4, 11, 22, 1],
                                         np.array([[142.537693, 142.72], [142.537693, 142.72]]))
    np.testing.assert_array_almost_equal(z_corner_point_depths[1, 4, 47, 0],
                                         np.array([[179.103781, 179.103781], [179.103781, 179.103781]]))


def test_z_corner_point_depths_s_bend_K_gap_grid(s_bend_k_gap_grid):
    # Act
    z_corner_point_depths = pf.z_corner_point_depths(s_bend_k_gap_grid)

    # Assert
    # Large array so only checking a subset of points.
    # np.testing.assert_array_almost_equal(z_corner_point_depths[2, 7, 34, 0],
    #                                      np.array([[162.582062, 165.410711], [162.582062, 165.410711]]))
    # np.testing.assert_array_almost_equal(z_corner_point_depths[4, 11, 22, 1],
    #                                      np.array([[142.537693, 142.72], [142.537693, 142.72]]))
    # np.testing.assert_array_almost_equal(z_corner_point_depths[1, 4, 47, 0],
    #                                      np.array([[179.103781, 179.103781], [179.103781, 179.103781]]))


def test_corner_points_cell_kji0(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)
    expected_corner_points = np.array([[[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [[0.0, 50.0, 0.0], [100.0, 50.0, 0.0]]],
                                       [[[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]], [[0.0, 50.0, 20.0], [100.0, 50.0,
                                                                                                     20.0]]]])

    # Act
    corner_points = pf.corner_points(basic_regular_grid, cell_kji0 = cell)

    # Assert
    np.testing.assert_array_almost_equal(corner_points, expected_corner_points)


def test_corner_points_cell_k_gaps(s_bend_k_gap_grid):
    # Arrange
    cell = (0, 0, 0)
    expected_corner_points = np.array([[[[0., 0., 113.2], [10., 0., 113.2]], [[0., 8., 113.2], [10., 8., 113.2]]],
                                       [[[0., 0., 115.6], [10., 0., 115.6]], [[0., 8., 115.6], [10., 8., 115.6]]]])
    # Act
    corner_points = pf.corner_points(s_bend_k_gap_grid, cell_kji0 = cell)

    # Assert
    np.testing.assert_array_almost_equal(corner_points, expected_corner_points)


def test_corner_points_cell_split_coordinate_lines(faulted_grid):
    # Arrange
    cell = (0, 0, 0)
    expected_corner_points = np.array([[[[1000., 2000., 3000.], [1100., 2000., 3000.]],
                                        [[1000., 2100., 3000.], [1100., 2100., 3001.]]],
                                       [[[1000., 2000., 3020.], [1100., 2000., 3020.]],
                                        [[1000., 2100., 3020.], [1100., 2100., 3021.]]]])
    # Act
    corner_points = pf.corner_points(faulted_grid, cell_kji0 = cell)

    # Assert
    np.testing.assert_array_almost_equal(corner_points, expected_corner_points)


def test_centre_point_default(basic_regular_grid):
    # Arrange
    expected_centre_points = np.array([[[[50.0, 25.0, 10.0], [150.0, 25.0, 10.0]],
                                        [[50.0, 75.0, 10.0], [150.0, 75.0, 10.0]]],
                                       [[[50.0, 25.0, 30.0], [150.0, 25.0, 30.0]],
                                        [[50.0, 75.0, 30.0], [150.0, 75.0, 30.0]]]])
    # Act
    centre_points = pf.centre_point(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(centre_points, expected_centre_points)


def test_centre_point_k_gaps(s_bend_k_gap_grid):
    # Act
    centre_points = pf.centre_point(s_bend_k_gap_grid)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(centre_points[2, 7, 1], np.array([26., 60., 123.28]))
    np.testing.assert_array_almost_equal(centre_points[3, 9, 34], np.array([-0.175204, 76., 164.817584]))
    np.testing.assert_array_almost_equal(centre_points[1, 5, 21], np.array([60.559817, 44., 148.909904]))
    np.testing.assert_array_almost_equal(centre_points[2, 8, 29], np.array([11.227031, 68., 154.599969]))


def test_centre_point_split_coordinate_lines(faulted_grid):
    # Act
    centre_points = pf.centre_point(faulted_grid)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(centre_points[2, 4, 1], np.array([1150., 2450., 3045.]))
    np.testing.assert_array_almost_equal(centre_points[0, 2, 7], np.array([1750., 2250., 3008.875]))
    np.testing.assert_array_almost_equal(centre_points[1, 1, 4], np.array([1450., 2150., 3033.75]))
    np.testing.assert_array_almost_equal(centre_points[2, 3, 5], np.array([1550., 2350., 3042.5]))


@pytest.mark.parametrize("interpolation_fraction, expected_interpolated_point",
                         [(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
                          (np.array([0.75, 0.75, 0.75]), np.array([75.0, 37.5, 15.0])),
                          (np.array([0.75, 0.2, 0.1]), np.array([10.0, 10.0, 15.0]))])
def test_interpolated_point(basic_regular_grid, interpolation_fraction, expected_interpolated_point):
    # Arrange
    cell = (0, 0, 0)

    # Act
    interpolated_point = pf.interpolated_point(basic_regular_grid,
                                               cell_kji0 = cell,
                                               interpolation_fraction = interpolation_fraction)

    # Assert
    np.testing.assert_array_almost_equal(interpolated_point, expected_interpolated_point)


@pytest.mark.parametrize("interpolation_fraction, expected_interpolated_point",
                         [(np.array([0.0, 0.0, 0.0]), np.array([47., 32., 112.72])),
                          (np.array([0.75, 0.75, 0.75]), np.array([54.5, 38., 114.52])),
                          (np.array([0.75, 0.2, 0.1]), np.array([48., 33.6, 114.52]))])
def test_interpolated_point_s_bend_faulted_grid(s_bend_faulted_grid, interpolation_fraction,
                                                expected_interpolated_point):
    # Arrange
    cell = (4, 4, 4)

    # Act
    interpolated_point = pf.interpolated_point(s_bend_faulted_grid,
                                               cell_kji0 = cell,
                                               interpolation_fraction = interpolation_fraction)

    # Assert
    np.testing.assert_array_almost_equal(interpolated_point, expected_interpolated_point)


def test_interpolated_points(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)
    interpolation_fractions = np.array([[0.0, 0.0, 0.0], [0.75, 0.75, 0.75], [0.75, 0.2, 0.1]])
    expected_interpolated_points = np.array([[[[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]],
                                              [[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]],
                                              [[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]]],
                                             [[[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]],
                                              [[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]],
                                              [[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]]],
                                             [[[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]],
                                              [[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]],
                                              [[75.0, 37.5, 0.0], [20.0, 37.5, 0.0], [10.0, 37.5, 0.0]]]])

    # Act
    interpolated_points = pf.interpolated_points(basic_regular_grid,
                                                 cell_kji0 = cell,
                                                 interpolation_fractions = interpolation_fractions)

    # Assert
    np.testing.assert_array_almost_equal(interpolated_points, expected_interpolated_points)


def test_split_horizons_points(basic_regular_grid):
    # Arrange
    expected_split_horizons_points = np.array([[[[[[0., 0., 0.], [100., 0., 0.]], [[0., 50., 0.], [100., 50., 0.]]],
                                                 [[[100., 0., 0.], [200., 0., 0.]], [[100., 50., 0.], [200., 50.,
                                                                                                       0.]]]],
                                                [[[[0., 50., 0.], [100., 50., 0.]], [[0., 100., 0.], [100., 100., 0.]]],
                                                 [[[100., 50., 0.], [200., 50., 0.]],
                                                  [[100., 100., 0.], [200., 100., 0.]]]]],
                                               [[[[[0., 0., 20.], [100., 0., 20.]], [[0., 50., 20.], [100., 50., 20.]]],
                                                 [[[100., 0., 20.], [200., 0., 20.]],
                                                  [[100., 50., 20.], [200., 50., 20.]]]],
                                                [[[[0., 50., 20.], [100., 50., 20.]],
                                                  [[0., 100., 20.], [100., 100., 20.]]],
                                                 [[[100., 50., 20.], [200., 50., 20.]],
                                                  [[100., 100., 20.], [200., 100., 20.]]]]],
                                               [[[[[0., 0., 40.], [100., 0., 40.]], [[0., 50., 40.], [100., 50., 40.]]],
                                                 [[[100., 0., 40.], [200., 0., 40.]],
                                                  [[100., 50., 40.], [200., 50., 40.]]]],
                                                [[[[0., 50., 40.], [100., 50., 40.]],
                                                  [[0., 100., 40.], [100., 100., 40.]]],
                                                 [[[100., 50., 40.], [200., 50., 40.]],
                                                  [[100., 100., 40.], [200., 100., 40.]]]]]])

    # Act
    split_horizons_points = pf.split_horizons_points(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(split_horizons_points, expected_split_horizons_points)


def test_split_horizons_points_s_bend_faulted_grid(s_bend_faulted_grid):
    # Act
    split_horizons_points = pf.split_horizons_points(s_bend_faulted_grid)

    # Assert
    # Large array so only checking a subset of points.
    np.testing.assert_array_almost_equal(split_horizons_points[2, 4, 17, 1, 0], np.array([67.927688, 40., 136.72]))
    np.testing.assert_array_almost_equal(split_horizons_points[4, 9, 30, 1, 1], np.array([4.415788, 80., 153.22144]))
    np.testing.assert_array_almost_equal(split_horizons_points[5, 0, 11, 0, 1], np.array([61.276311, 0., 119.895758]))
    np.testing.assert_array_almost_equal(split_horizons_points[3, 6, 44, 0, 0], np.array([12.382711, 48., 183.408551]))


def test_find_cell_for_point_xy(faulted_grid):
    j, i = pf.find_cell_for_point_xy(faulted_grid, 1150.0, 2450.0, k0 = 2, vertical_ref = 'top', local_coords = True)
    assert j == 4 and i == 1
    j, i = pf.find_cell_for_point_xy(faulted_grid, 1750.0, 2250.0, k0 = 0, vertical_ref = 'base', local_coords = True)
    assert j == 2 and i == 7
    j, i = pf.find_cell_for_point_xy(faulted_grid, 1460.0, 2140.0, k0 = 1, vertical_ref = 'top', local_coords = True)
    assert j == 1 and i == 4
    j, i = pf.find_cell_for_point_xy(faulted_grid, 1530.0, 2370.0, k0 = 2, vertical_ref = 'base', local_coords = True)
    assert j == 3 and i == 5
