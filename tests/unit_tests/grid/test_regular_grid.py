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


def test_centres(basic_regular_grid):
    grid = basic_regular_grid
    grid.is_aligned = False
    if hasattr(grid, 'array_centre_point'):
        delattr(grid, 'array_centre_point')
    generic_centres = grid.centre_point().copy()
    grid.is_aligned = True
    if hasattr(grid, 'array_centre_point'):
        delattr(grid, 'array_centre_point')
    aligned_centres = grid.centre_point().copy()
    np.testing.assert_array_almost_equal(aligned_centres, generic_centres)


def test_aligned_column_centres(basic_regular_grid):
    grid = basic_regular_grid
    grid.is_aligned = False
    if hasattr(grid, 'array_centre_point'):
        delattr(grid, 'array_centre_point')
    generic_centres = grid.centre_point().copy()
    grid.is_aligned = True
    if hasattr(grid, 'array_centre_point'):
        delattr(grid, 'array_centre_point')
    col_centres = grid.aligned_column_centres()
    assert col_centres.shape == (grid.nj, grid.ni, 2)
    np.testing.assert_array_almost_equal(col_centres, generic_centres[0, :, :, :2])


def test_slice_points_k(aligned_regular_grid):
    grid = aligned_regular_grid
    assert grid.is_aligned
    # test K slicing
    e = np.array([[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0), (300.0, 0.0, 0.0), (400.0, 0.0, 0.0)],
                  [(0.0, 50.0, 0.0), (100.0, 50.0, 0.0), (200.0, 50.0, 0.0), (300.0, 50.0, 0.0), (400.0, 50.0, 0.0)],
                  [(0.0, 100.0, 0.0), (100.0, 100.0, 0.0), (200.0, 100.0, 0.0), (300.0, 100.0, 0.0),
                   (400.0, 100.0, 0.0)],
                  [(0.0, 150.0, 0.0), (100.0, 150.0, 0.0), (200.0, 150.0, 0.0), (300.0, 150.0, 0.0),
                   (400.0, 150.0, 0.0)]])
    p = grid.slice_points(local = True)
    np.testing.assert_array_almost_equal(p, e)
    e[..., 2] = 40.0
    for local in [True, False]:
        p = grid.slice_points(axis = 0, ref_slice = 2, local = local)
        np.testing.assert_array_almost_equal(p, e)
    # test J slicing
    e = np.array([[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0), (300.0, 0.0, 0.0), (400.0, 0.0, 0.0)],
                  [(0.0, 0.0, 20.0), (100.0, 0.0, 20.0), (200.0, 0.0, 20.0), (300.0, 0.0, 20.0), (400.0, 0.0, 20.0)],
                  [(0.0, 0.0, 40.0), (100.0, 0.0, 40.0), (200.0, 0.0, 40.0), (300.0, 0.0, 40.0), (400.0, 0.0, 40.0)]])
    p = grid.slice_points(axis = 1, local = True)
    np.testing.assert_array_almost_equal(p, e)
    # test I slicing
    e = np.array([[(0.0, 0.0, 0.0), (0.0, 50.0, 0.0), (0.0, 100.0, 0.0), (0.0, 150.0, 0.0)],
                  [(0.0, 0.0, 20.0), (0.0, 50.0, 20.0), (0.0, 100.0, 20.0), (0.0, 150.0, 20.0)],
                  [(0.0, 0.0, 40.0), (0.0, 50.0, 40.0), (0.0, 100.0, 40.0), (0.0, 150.0, 40.0)]])
    p = grid.slice_points(axis = 2, local = True)
    np.testing.assert_array_almost_equal(p, e)
