import numpy as np
import pytest

import resqpy.grid.points_functions as pf


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


# Need to complete test
def test_horizon_points_k_gaps(basic_regular_grid):
    # Arrange

    # Act
    # horizon_points = pf.horizon_points(basic_regular_grid, ref_k0=1)

    # Assert
    pass


# Need to complete test
def test_horizon_points_split_coordinate_lines(basic_regular_grid):
    # Arrange

    # Act
    # horizon_points = pf.horizon_points(basic_regular_grid, ref_k0=1)

    # Assert
    pass


def test_x_section_corner_points_default(basic_regular_grid):
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


# Need to complete test
def test_x_section_corner_points_k_gaps(basic_regular_grid):
    # Arrange

    # Act
    # x_section_corner_points = pf.x_section_corner_points(basic_regular_grid, axis='I')

    # Assert
    pass


# Need to complete test
def test_x_section_corner_points_split_coordinate_lines(basic_regular_grid):
    # Arrange

    # Act
    # x_section_corner_points = pf.x_section_corner_points(basic_regular_grid, axis='I')

    # Assert
    pass


# Need to complete test
def test_x_section_corner_points_rotate_true(basic_regular_grid):
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
    # x_section_corner_points = pf.x_section_corner_points(basic_regular_grid, axis='I', rotate=True)

    # Assert
    # np.testing.assert_array_almost_equal(x_section_corner_points, expected_x_section_corner_points)


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


# Need to complete test
def test_corner_points_cell_k_gaps(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)
    expected_corner_points = np.array([[[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [[0.0, 50.0, 0.0], [100.0, 50.0, 0.0]]],
                                       [[[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]], [[0.0, 50.0, 20.0], [100.0, 50.0,
                                                                                                     20.0]]]])
    # Act
    # corner_points = pf.corner_points(basic_regular_grid, cell_kji0=cell)

    # Assert
    # np.testing.assert_array_almost_equal(corner_points, expected_corner_points)


# Need to complete test
def test_corner_points_cell_split_coordinate_lines(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)
    expected_corner_points = np.array([[[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [[0.0, 50.0, 0.0], [100.0, 50.0, 0.0]]],
                                       [[[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]], [[0.0, 50.0, 20.0], [100.0, 50.0,
                                                                                                     20.0]]]])
    # Act
    # corner_points = pf.corner_points(basic_regular_grid, cell_kji0=cell)

    # Assert
    # np.testing.assert_array_almost_equal(corner_points, expected_corner_points)


def test_centre_point_deafult(basic_regular_grid):
    # Arrange
    expected_centre_points = np.array([[[[50.0, 25.0, 10.0], [150.0, 25.0, 10.0]],
                                        [[50.0, 75.0, 10.0], [150.0, 75.0, 10.0]]],
                                       [[[50.0, 25.0, 30.0], [150.0, 25.0, 30.0]],
                                        [[50.0, 75.0, 30.0], [150.0, 75.0, 30.0]]]])
    # Act
    centre_points = pf.centre_point(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(centre_points, expected_centre_points)


# Need to complete test
def test_centre_point_k_gaps(basic_regular_grid):
    # Arrange
    expected_centre_points = np.array([[[[50.0, 25.0, 10.0], [150.0, 25.0, 10.0]],
                                        [[50.0, 75.0, 10.0], [150.0, 75.0, 10.0]]],
                                       [[[50.0, 25.0, 30.0], [150.0, 25.0, 30.0]],
                                        [[50.0, 75.0, 30.0], [150.0, 75.0, 30.0]]]])
    # Act
    # centre_points = pf.centre_point(basic_regular_grid)

    # Assert
    # np.testing.assert_array_almost_equal(centre_points, expected_centre_points)


# Need to complete test
def test_centre_point_split_coordinate_lines(basic_regular_grid):
    # Arrange
    expected_centre_points = np.array([[[[50.0, 25.0, 10.0], [150.0, 25.0, 10.0]],
                                        [[50.0, 75.0, 10.0], [150.0, 75.0, 10.0]]],
                                       [[[50.0, 25.0, 30.0], [150.0, 25.0, 30.0]],
                                        [[50.0, 75.0, 30.0], [150.0, 75.0, 30.0]]]])
    # Act
    # centre_points = pf.centre_point(basic_regular_grid)

    # Assert
    # np.testing.assert_array_almost_equal(centre_points, expected_centre_points)


# Need to check third test case
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
