import resqpy.grid_surface as rqgs
import numpy as np
import pytest


def test_find_faces_to_represent_surface_regular_optimised(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    # Assert
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)


def test_find_faces_to_represent_surface_regular_optimised_with_return_properties(small_grid_and_surface,):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    return_properties = ["offset"]

    # Act
    gcs_normal, properties_dict = rqgs.find_faces_to_represent_surface_regular(grid,
                                                                               surface,
                                                                               name,
                                                                               return_properties = return_properties)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs
    offsets_normal = properties_dict["offset"]

    return_properties.append("depth")
    return_properties.append("triangle")
    (
        gcs_optimised,
        properties_optimised,
    ) = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                               surface,
                                                               name,
                                                               return_properties = return_properties)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs
    triangles_optimised = properties_optimised["triangle"]
    depths_optimised = properties_optimised["depth"]
    offsets_optimised = properties_optimised["offset"]

    # Assert
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_almost_equal(offsets_normal, offsets_optimised)
    assert depths_optimised.shape == offsets_optimised.shape
    assert np.all(depths_optimised > 0.0)
    assert triangles_optimised.shape == offsets_optimised.shape
    assert np.all(triangles_optimised >= 0)


def test_bisector_from_faces_flat_surface_k():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array([[[True, True, True], [True, True, True], [True, True, True]],
                        [[False, False, False], [False, False, False], [False, False, False]]],
                       dtype = bool)
    j_faces = np.array([[[False, False, False], [False, False, False]], [[False, False, False], [False, False, False]],
                        [[False, False, False], [False, False, False]]],
                       dtype = bool)
    i_faces = np.array(
        [[[False, False], [False, False], [False, False]], [[False, False], [False, False], [False, False]],
         [[False, False], [False, False], [False, False]]],
        dtype = bool)

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)
    bounds = rqgs.get_boundary(k_faces, j_faces, i_faces, grid_extent_kji)

    # Assert
    np.all(a == np.array([[[True, True, True], [True, True, True], [True, True, True]],
                          [[False, False, False], [False, False, False], [False, False, False]],
                          [[False, False, False], [False, False, False], [False, False, False]]],
                         dtype = bool))
    assert is_curtain is False
    assert all([(bounds[f'{axis}_min'] == 0) for axis in 'kji'])
    assert bounds['k_max'] == 1
    assert bounds['j_max'] == 2
    assert bounds['i_max'] == 2


def test_bisector_from_faces_flat_surface_j():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array([[[False, False, False], [False, False, False], [False, False, False]],
                        [[False, False, False], [False, False, False], [False, False, False]]],
                       dtype = bool)
    j_faces = np.array([[[True, True, True], [False, False, False]], [[True, True, True], [False, False, False]],
                        [[True, True, True], [False, False, False]]],
                       dtype = bool)
    i_faces = np.array(
        [[[False, False], [False, False], [False, False]], [[False, False], [False, False], [False, False]],
         [[False, False], [False, False], [False, False]]],
        dtype = bool)

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)

    # Assert
    np.all(a == np.array([[[True, True, True], [False, False, False], [False, False, False]],
                          [[True, True, True], [False, False, False], [False, False, False]],
                          [[True, True, True], [False, False, False], [False, False, False]]],
                         dtype = bool))
    assert is_curtain is True


def test_shadow_from_faces_curtain():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array([[[False, False, False], [False, False, False], [False, False, False]],
                        [[False, False, False], [False, False, False], [False, False, False]]],
                       dtype = bool)

    # Act
    a = rqgs.shadow_from_faces(grid_extent_kji, k_faces)

    # Assert
    assert a.shape == grid_extent_kji
    assert np.all(a == 0)


def test_bisector_from_faces_flat_surface_i():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array([[[False, False, False], [False, False, False], [False, False, False]],
                        [[False, False, False], [False, False, False], [False, False, False]]],
                       dtype = bool)
    j_faces = np.array([[[False, False, False], [False, False, False]], [[False, False, False], [False, False, False]],
                        [[False, False, False], [False, False, False]]],
                       dtype = bool)
    i_faces = np.array([[[True, False], [True, False], [True, False]], [[True, False], [True, False], [True, False]],
                        [[True, False], [True, False], [True, False]]],
                       dtype = bool)

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)

    # Assert
    np.all(a == np.array([[[True, False, False], [True, False, False], [True, False, False]],
                          [[True, False, False], [True, False, False], [True, False, False]],
                          [[True, False, False], [True, False, False], [True, False, False]]],
                         dtype = bool))
    assert is_curtain is True


def test_bisector_from_faces_flat_surface_k_hole():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array([[[True, True, True], [True, False, True], [True, True, True]],
                        [[False, False, False], [False, False, False], [False, False, False]]],
                       dtype = bool)
    j_faces = np.array([[[False, False, False], [False, False, False]], [[False, False, False], [False, False, False]],
                        [[False, False, False], [False, False, False]]],
                       dtype = bool)
    i_faces = np.array(
        [[[False, False], [False, False], [False, False]], [[False, False], [False, False], [False, False]],
         [[False, False], [False, False], [False, False]]],
        dtype = bool)

    # Act & Assert
    with pytest.raises(AssertionError):
        rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)


def test_shadow_from_faces_flat_surface_k_hole():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array([[[True, True, True], [True, False, True], [True, True, True]],
                        [[False, False, False], [False, False, False], [True, False, False]]],
                       dtype = bool)

    # Act
    a = rqgs.shadow_from_faces(grid_extent_kji, k_faces)

    # Assert
    assert a.shape == grid_extent_kji
    assert np.all(a == np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[2, 2, 2], [2, 0, 2], [3, 2, 2]],
                                 [[2, 2, 2], [2, 0, 2], [2, 2, 2]]]))
