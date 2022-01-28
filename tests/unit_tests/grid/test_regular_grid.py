import numpy as np


def test_half_cell_transmissibility_already_set(basic_regular_grid):
    # Arrange
    array_half_cell_t = np.random.random(basic_regular_grid.extent_kji)
    basic_regular_grid.array_half_cell_t = array_half_cell_t

    # Act
    half_cell_transmissibility = basic_regular_grid.half_cell_transmissibility()

    # Assert
    np.testing.assert_array_almost_equal(half_cell_transmissibility, array_half_cell_t)


# Need to complete test
def test_half_cell_transmissibility_default(example_model_with_properties):
    # Arrange
    grid = example_model_with_properties.grid()

    # Act
    # half_cell_transmissibility = grid.half_cell_transmissibility()

    # Assert
    pass


def test_volume(basic_regular_grid):
    # Act
    volume = basic_regular_grid.volume()

    # Assert
    np.testing.assert_array_almost_equal(volume, 100000.0)


def test_volume_faulted_grid(faulted_grid):
    # Arrange
    expected_volume = np.array([[[200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.]],
                                [[200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.]],
                                [[100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.]]])

    # Act
    volume = faulted_grid.volume()

    # Assert
    np.testing.assert_array_almost_equal(volume, expected_volume)


def test_volume_cell_kji0(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)

    # Act
    volume = basic_regular_grid.volume(cell_kji0=cell)

    # Assert
    assert volume == 100000.0


def test_thickness(basic_regular_grid):
    # Act
    thickness = basic_regular_grid.thickness()

    # Assert
    np.testing.assert_array_almost_equal(thickness, 20.0)


def test_thickness_faulted_grid(faulted_grid):
    # Arrange
    expected_thickness = np.array([[[20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.]],
                                   [[20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.]],
                                   [[10., 10., 5., 0., 0., 5., 10., 10.],
                                    [10., 10., 5., 0., 0., 5., 10., 10.],
                                    [10., 10., 5., 0., 0., 5., 10., 10.],
                                    [10., 10., 5., 0., 0., 5., 10., 10.],
                                    [10., 10., 5., 0., 0., 5., 10., 10.]]])

    # Act
    thickness = faulted_grid.thickness()

    # Assert
    np.testing.assert_array_almost_equal(thickness, expected_thickness)


def test_thickness_cell_kji0(basic_regular_grid):
    # Arrange
    cell = (0, 0, 0)

    # Act
    thickness = basic_regular_grid.thickness(cell_kji0=cell)

    # Assert
    assert thickness == 20.0
