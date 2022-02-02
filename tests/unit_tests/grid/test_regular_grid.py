import numpy as np


def test_half_cell_transmissibility_already_set(basic_regular_grid):
    # Arrange
    array_half_cell_t = np.random.random(basic_regular_grid.extent_kji)
    basic_regular_grid.array_half_cell_t = array_half_cell_t

    # Act
    half_cell_transmissibility = basic_regular_grid.half_cell_transmissibility()

    # Assert
    np.testing.assert_array_almost_equal(half_cell_transmissibility, array_half_cell_t)


def test_half_cell_transmissibility_default(example_model_with_prop_ts_rels):
    # Arrange
    grid = example_model_with_prop_ts_rels.grid()

    # Act
    half_cell_transmissibility = grid.half_cell_transmissibility()

    # Assert
    assert half_cell_transmissibility.shape == (grid.nk, grid.nj, grid.ni, 3, 2)


def test_half_cell_transmissibility_use_property_false(example_model_with_prop_ts_rels):
    # Arrange
    grid = example_model_with_prop_ts_rels.grid()

    # Act
    half_cell_transmissibility = grid.half_cell_transmissibility(use_property = False)

    # Assert
    assert half_cell_transmissibility.shape == (grid.nk, grid.nj, grid.ni, 3, 2)


def test_volume(basic_regular_grid):
    # Act
    volume = basic_regular_grid.volume()

    # Assert
    np.testing.assert_array_almost_equal(volume, 100000.0)


def test_volume_cell_kji0(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)

    # Act
    volume = basic_regular_grid.volume(cell_kji0 = cell)

    # Assert
    assert volume == 100000.0


def test_thickness(basic_regular_grid):
    # Act
    thickness = basic_regular_grid.thickness()

    # Assert
    np.testing.assert_array_almost_equal(thickness, 20.0)


def test_thickness_cell_kji0(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)

    # Act
    thickness = basic_regular_grid.thickness(cell_kji0 = cell)

    # Assert
    assert thickness == 20.0
