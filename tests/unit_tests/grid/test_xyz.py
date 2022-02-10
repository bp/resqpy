import pytest

import resqpy.grid.xyz as xyz
import numpy as np
import resqpy.crs as rqc
from resqpy.olio.exceptions import InvalidUnitError

METRES_TO_FEET = 3.280839895013123


def test_box_centre(basic_regular_grid):
    # Act
    box_centre = xyz.xyz_box_centre(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(box_centre, np.array([100.0, 50.0, 20.0]))


def test_box_centre_faulted_grid(faulted_grid):
    # Act
    box_centre = xyz.xyz_box_centre(faulted_grid)

    # Assert
    np.testing.assert_array_almost_equal(box_centre, np.array([1400., 2250., 3023.25]))


@pytest.mark.parametrize("cell, expected_bounding",
                         [((0, 0, 0), np.array([[0.0, 0.0, 0.0], [100.0, 50.0, 20.0]])),
                          ((1, 1, 1), np.array([[100.0, 50.0, 20.0], [200.0, 100.0, 40.0]]))])
def test_bounding_box(basic_regular_grid, cell, expected_bounding):
    # Act
    bounding_box = xyz.bounding_box(basic_regular_grid, cell_kji0 = cell)

    # Assert
    np.testing.assert_array_almost_equal(bounding_box, expected_bounding)


@pytest.mark.parametrize("cell, expected_bounding",
                         [((0, 0, 0), np.array([[1000., 2000., 3000.], [1100., 2100., 3021.]])),
                          ((1, 1, 1), np.array([[1100., 2100., 3021.], [1200., 2200., 3043.5]]))])
def test_bounding_box_faulted_grid(faulted_grid, cell, expected_bounding):
    # Act
    bounding_box = xyz.bounding_box(faulted_grid, cell_kji0 = cell)

    # Assert
    np.testing.assert_array_almost_equal(bounding_box, expected_bounding)


def test_local_to_global_crs_metres_to_feet(basic_regular_grid):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'

    a = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    a_expected = a * METRES_TO_FEET

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize("a, a_expected",
                         [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                           np.array([[2.0, 3.0, -2.0], [2.0, 3.0, -2.0], [2.0, 3.0, -2.0]]) * METRES_TO_FEET),
                          (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]]),
                           np.array([[3.0, 3.0, 6.0], [9.0, 3.0, 3.0], [8.0, 11.0, 1.0]]) * METRES_TO_FEET),
                          (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]]),
                           np.array([[7.0, 9.0, 3.0], [2.0, 4.0, 6.0], [10.0, 7.0, 5.0]]) * METRES_TO_FEET)])
def test_local_to_global_crs_xyz_offset(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 1.0
    y_offset = 2.0
    z_offset = -3.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


def test_local_to_global_crs_invalid_units(basic_regular_grid):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ly'
    global_z_units = 'ly'

    a = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    # Act & Asert
    with pytest.raises(InvalidUnitError):
        basic_regular_grid.local_to_global_crs(a,
                                               crs_uuid = crs.uuid,
                                               global_xy_units = global_xy_units,
                                               global_z_units = global_z_units)


@pytest.mark.parametrize("a, a_expected",
                         [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                           np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]) * METRES_TO_FEET),
                          (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]]),
                           np.array([[2.0, 1.0, -9.0], [8.0, 1.0, -6.0], [7.0, 9.0, -4.0]]) * METRES_TO_FEET),
                          (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]]),
                           np.array([[6.0, 7.0, -6.0], [1.0, 2.0, -9.0], [9.0, 5.0, -8.0]]) * METRES_TO_FEET)])
def test_local_to_global_crs_global_z_increasing_downward(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'
    global_z_increasing_downward = False

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units,
                                                           global_z_increasing_downward = global_z_increasing_downward)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize(
    "a, a_expected",
    [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0],
                [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0]])),
     (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]
               ]), np.array([[1.866025, -1.232051, 9.], [4.866025, -6.428203, 6.], [11.294229, -1.562178, 4.]])),
     (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]
               ]), np.array([[9.062178, -1.696152, 6.], [2.232051, 0.133975, 9.], [8.830127, -5.294229, 8.]]))])
def test_local_to_global_crs_degree_rotation(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 60
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'm'
    global_z_units = 'm'

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize(
    "a, a_expected",
    [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0],
                [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0]])),
     (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]
               ]), np.array([[1.866025, -1.232052, 9.], [4.866023, -6.428205, 6.], [11.294228, -1.562183, 4.]])),
     (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]
               ]), np.array([[9.062177, -1.696156, 6.], [2.232051, 0.133974, 9.], [8.830125, -5.294233, 8.]]))])
def test_local_to_global_crs_radian_rotation(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = np.pi / 3  # 60 degrees
    rotation_units = 'rad'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'm'
    global_z_units = 'm'

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize(
    "a, a_expected",
    [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, -1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, -1.0],
                [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, -1.0]])),
     (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]
               ]), np.array([[1.866025, -1.232051, -9.], [4.866025, -6.428203, -6.], [11.294229, -1.562178, -4.]])),
     (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]
               ]), np.array([[9.062178, -1.696152, -6.], [2.232051, 0.133975, -9.], [8.830127, -5.294229, -8.]]))])
def test_local_to_global_crs_rotation_and_global_z_increasing_downward(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 60
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'm'
    global_z_units = 'm'
    global_z_increasing_downward = False

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units,
                                                           global_z_increasing_downward = global_z_increasing_downward)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize("a, a_expected", [
    (np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
              ]), np.array([[1.358968, 9.84252, 3.28084], [1.358968, 9.84252, 3.28084], [1.358968, 9.84252, 3.28084]])),
    (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]]),
     np.array([[3.678873, 7.522616, -22.965879], [17.598297, -6.396809, -13.12336], [33.837626, 14.482328, -6.56168]])),
    (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]]),
     np.array([[26.877914, 12.162424, -13.12336], [3.678873, 12.162424, -22.965879], [29.197818, 0.562903, -19.685039]
              ]))
])
def test_local_to_global_crs_full_transformation(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = -1.0
    y_offset = 3.0
    z_offset = 2.0
    rotation = 45.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'
    global_z_increasing_downward = False

    # Act
    a_transformed = basic_regular_grid.local_to_global_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units,
                                                           global_z_increasing_downward = global_z_increasing_downward)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


def test_global_to_local_crs_metres_to_feet(basic_regular_grid):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'

    a = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    a_expected = a / METRES_TO_FEET

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize("a_expected, a",
                         [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                           np.array([[2.0, 3.0, -2.0], [2.0, 3.0, -2.0], [2.0, 3.0, -2.0]]) * METRES_TO_FEET),
                          (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]]),
                           np.array([[3.0, 3.0, 6.0], [9.0, 3.0, 3.0], [8.0, 11.0, 1.0]]) * METRES_TO_FEET),
                          (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]]),
                           np.array([[7.0, 9.0, 3.0], [2.0, 4.0, 6.0], [10.0, 7.0, 5.0]]) * METRES_TO_FEET)])
def test_global_to_local_crs_xyz_offset(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 1.0
    y_offset = 2.0
    z_offset = -3.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


def test_global_to_local_crs_invalid_units(basic_regular_grid):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ly'
    global_z_units = 'ly'

    a = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    # Act & Asert
    with pytest.raises(InvalidUnitError):
        basic_regular_grid.global_to_local_crs(a,
                                               crs_uuid = crs.uuid,
                                               global_xy_units = global_xy_units,
                                               global_z_units = global_z_units)


@pytest.mark.parametrize("a_expected, a",
                         [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                           np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]) * METRES_TO_FEET),
                          (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]]),
                           np.array([[2.0, 1.0, -9.0], [8.0, 1.0, -6.0], [7.0, 9.0, -4.0]]) * METRES_TO_FEET),
                          (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]]),
                           np.array([[6.0, 7.0, -6.0], [1.0, 2.0, -9.0], [9.0, 5.0, -8.0]]) * METRES_TO_FEET)])
def test_global_to_local_crs_global_z_increasing_downward(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 0.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'
    global_z_increasing_downward = False

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units,
                                                           global_z_increasing_downward = global_z_increasing_downward)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize(
    "a_expected, a",
    [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0],
                [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0]])),
     (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]
               ]), np.array([[1.866025, -1.232051, 9.], [4.866025, -6.428203, 6.], [11.294229, -1.562178, 4.]])),
     (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]
               ]), np.array([[9.062178, -1.696152, 6.], [2.232051, 0.133975, 9.], [8.830127, -5.294229, 8.]]))])
def test_global_to_local_crs_degree_rotation(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 60
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'm'
    global_z_units = 'm'

    a = np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0],
                  [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0]])
    a_expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize(
    "a_expected, a",
    [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0],
                [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0]])),
     (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]
               ]), np.array([[1.866025, -1.232052, 9.], [4.866023, -6.428205, 6.], [11.294228, -1.562183, 4.]])),
     (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]
               ]), np.array([[9.062177, -1.696156, 6.], [2.232051, 0.133974, 9.], [8.830125, -5.294233, 8.]]))])
def test_global_to_local_crs_radian_rotation(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = np.pi / 3  # 60 degrees
    rotation_units = 'rad'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'm'
    global_z_units = 'm'

    a = np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0],
                  [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, 1.0]])
    a_expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize(
    "a_expected, a",
    [(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      np.array([[0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, -1.0], [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, -1.0],
                [0.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2, -1.0]])),
     (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]
               ]), np.array([[1.866025, -1.232051, -9.], [4.866025, -6.428203, -6.], [11.294229, -1.562178, -4.]])),
     (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]
               ]), np.array([[9.062178, -1.696152, -6.], [2.232051, 0.133975, -9.], [8.830127, -5.294229, -8.]]))])
def test_global_to_local_crs_rotation_and_global_z_increasing_downward(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0
    rotation = 60
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'm'
    global_z_units = 'm'
    global_z_increasing_downward = False

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units,
                                                           global_z_increasing_downward = global_z_increasing_downward)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)


@pytest.mark.parametrize("a_expected, a", [
    (np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
              ]), np.array([[1.358968, 9.84252, 3.28084], [1.358968, 9.84252, 3.28084], [1.358968, 9.84252, 3.28084]])),
    (np.array([[2.0, 1.0, 9.0], [8.0, 1.0, 6.0], [7.0, 9.0, 4.0]]),
     np.array([[3.678873, 7.522616, -22.965879], [17.598297, -6.396809, -13.12336], [33.837626, 14.482328, -6.56168]])),
    (np.array([[6.0, 7.0, 6.0], [1.0, 2.0, 9.0], [9.0, 5.0, 8.0]]),
     np.array([[26.877914, 12.162424, -13.12336], [3.678873, 12.162424, -22.965879], [29.197818, 0.562903, -19.685039]
              ]))
])
def test_global_to_local_crs_full_transformation(basic_regular_grid, a, a_expected):
    # Arrange
    x_offset = -1.0
    y_offset = 3.0
    z_offset = 2.0
    rotation = 45.0
    rotation_units = 'dega'

    crs = rqc.Crs(basic_regular_grid.model,
                  x_offset = x_offset,
                  y_offset = y_offset,
                  z_offset = z_offset,
                  z_inc_down = True,
                  rotation = rotation,
                  rotation_units = rotation_units,
                  xy_units = 'm',
                  z_units = 'm',
                  axis_order = 'easting northing',
                  epsg_code = '32630')
    crs.create_xml()

    global_xy_units = 'ft'
    global_z_units = 'ft'
    global_z_increasing_downward = False

    # Act
    a_transformed = basic_regular_grid.global_to_local_crs(a,
                                                           crs_uuid = crs.uuid,
                                                           global_xy_units = global_xy_units,
                                                           global_z_units = global_z_units,
                                                           global_z_increasing_downward = global_z_increasing_downward)

    # Assert
    np.testing.assert_array_almost_equal(a_transformed, a_expected)
