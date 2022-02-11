import numpy as np
import pytest

from resqpy.grid import Grid
from resqpy.model import Model

import resqpy.grid._face_functions as ff


def test_face_centre_axis_0(basic_regular_grid: Grid):
    # Arrange
    cell = None
    axis = 0
    zero_or_one = 0
    # yapf: disable
    expected_face_centres = np.array([[[[50.0, 25.0, 0.0], [150.0, 25.0, 0.0]],
                                       [[50.0, 75.0, 0.0], [150.0, 75.0, 0.0]]],
                                      [[[50.0, 25.0, 20.0], [150.0, 25.0, 20.0]],
                                       [[50.0, 75.0, 20.0], [150.0, 75.0, 20.0]]]])  # yapf: enable

    # Act & Assert
    face_centre = ff.face_centre(basic_regular_grid, cell, axis, zero_or_one)
    np.testing.assert_array_almost_equal(face_centre, expected_face_centres)


def test_face_centre_axis_1(basic_regular_grid: Grid):
    # Arrange
    cell = None
    axis = 1
    zero_or_one = 1
    # yapf: disable
    expected_face_centres = np.array([[[[50.0, 50.0, 10.0], [150.0, 50.0, 10.0]],
                                       [[50.0, 100.0, 10.0], [150.0, 100.0, 10.0]]],
                                      [[[50.0, 50.0, 30.0], [150.0, 50.0, 30.0]],
                                       [[50.0, 100.0, 30.0], [150.0, 100.0, 30.0]]]])  # yapf: enable

    # Act & Assert
    face_centre = ff.face_centre(basic_regular_grid, cell, axis, zero_or_one)
    np.testing.assert_array_almost_equal(face_centre, expected_face_centres)


def test_face_centre_cell_kji0(basic_regular_grid: Grid):
    # Arrange
    cell = (0, 0, 0)
    axis = 0

    # Act & Assert
    zero_or_one = 0
    face_centre = ff.face_centre(basic_regular_grid, cell, axis, zero_or_one)
    np.testing.assert_array_equal(face_centre, np.array([50.0, 25.0, 0.0]))

    zero_or_one = 1
    face_centre = ff.face_centre(basic_regular_grid, cell, axis, zero_or_one)
    np.testing.assert_array_almost_equal(face_centre, np.array([50.0, 25.0, 20.0]))


def test_face_centre_invalid_axis(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    cell = (1, 1, 1)
    axis = 4
    zero_or_one = 0

    # Act & Assert
    with pytest.raises(ValueError):
        ff.face_centre(grid, cell, axis, zero_or_one)


@pytest.mark.parametrize(
    "cell, expected_face_centres",
    # yapf: disable
    [((0, 0, 0), np.array([[[50.0, 25.0, 0.0], [50.0, 25.0, 20.0]],
                           [[50.0, 0.0, 10.0], [50.0, 50.0, 10.0]],
                           [[0.0, 25.0, 10.0], [100.0, 25.0, 10.0]]])),
     ((1, 1, 1), np.array([[[150.0, 75.0, 20.0], [150.0, 75.0, 40.0]],
                           [[150.0, 50.0, 30.0], [150.0, 100.0, 30.0]],
                           [[100.0, 75.0, 30.0], [200.0, 75.0, 30.0]]]))])  # yapf: enable
def test_face_centres_kji_01(basic_regular_grid: Grid, cell, expected_face_centres):
    # Act & Assert
    face_centres = ff.face_centres_kji_01(basic_regular_grid, cell)
    np.testing.assert_array_almost_equal(face_centres, expected_face_centres)
