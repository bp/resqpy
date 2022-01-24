import numpy as np
import pytest

from resqpy.grid import Grid
from resqpy.model import Model

import resqpy.grid.face_functions as ff


def test_face_centre(basic_regular_grid: Grid):
    # Arrange
    cell = None
    axis = 0
    zero_or_one = 0
    expected_face_centres = np.array([[[[50.0, 25.0, 0.0], [150.0, 25.0, 0.0]], [[50.0, 75.0, 0.0], [150.0, 75.0,
                                                                                                     0.0]]],
                                      [[[50.0, 25.0, 20.0], [150.0, 25.0, 20.0]],
                                       [[50.0, 75.0, 20.0], [150.0, 75.0, 20.0]]]])

    # Act & Assert
    face_centre = ff.face_centre(basic_regular_grid, cell, axis, zero_or_one)
    np.testing.assert_array_equal(face_centre, expected_face_centres)


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
    np.testing.assert_array_equal(face_centre, np.array([50.0, 25.0, 20.0]))


def test_face_centre_invalid_axis(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    cell = (1, 1, 1)
    axis = 4
    zero_or_one = 0

    # Act & Assert
    with pytest.raises(ValueError):
        ff.face_centre(grid, cell, axis, zero_or_one)


def test_face_centres_kji_01(basic_regular_grid: Grid):
    # Arrange
    cell = (0, 0, 0)
    expected_face_centres = np.array([[[50.0, 25.0, 0.0], [50.0, 25.0, 20.0]], [[50.0, 0.0, 10.0], [50.0, 50.0, 10.0]],
                                      [[0.0, 25.0, 10.0], [100.0, 25.0, 10.0]]])

    # Act & Assert
    face_centres = ff.face_centres_kji_01(basic_regular_grid, cell)
    np.testing.assert_array_equal(face_centres, expected_face_centres)
