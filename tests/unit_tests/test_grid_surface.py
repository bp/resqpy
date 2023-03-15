import numpy as np
import pytest

import resqpy.grid_surface as rqgs
import resqpy.property as rqp


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
    surf_flange = rqp.Property.from_array(surface.model,
                                          cached_array = None,
                                          source_info = 'constant False',
                                          keyword = 'flange bool',
                                          support_uuid = surface.uuid,
                                          property_kind = 'flange bool',
                                          indexable_element = 'faces',
                                          discrete = True,
                                          const_value = 0,
                                          expand_const_arrays = False,
                                          dtype = bool)
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
    return_properties.append("flange bool")
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
    flange_optimised = properties_optimised["flange bool"]

    # Assert
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_almost_equal(offsets_normal, offsets_optimised)
    assert depths_optimised.shape == offsets_optimised.shape
    assert np.all(depths_optimised > 0.0)
    assert triangles_optimised.shape == offsets_optimised.shape
    assert np.all(triangles_optimised >= 0)
    assert flange_optimised.shape == offsets_optimised.shape
    assert not np.any(flange_optimised)


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


def test_where_true_and_get_boundary():
    grid_extent_kji = (7, 8, 9)
    nk, nj, ni = grid_extent_kji
    k_faces = np.zeros((nk - 1, nj, ni), dtype = bool)
    j_faces = np.zeros((nk, nj - 1, ni), dtype = bool)
    i_faces = np.zeros((nk, nj, ni - 1), dtype = bool)
    k_faces[3, 3:7, 4:6] = True
    j_faces[4:6, 2, 4] = True
    i_faces[2:5, 3:6, 5] = True

    w_k, w_j, w_i = rqgs.where_true(k_faces)
    assert len(w_k) == 8
    assert np.all(np.unique(w_k) == (3,))
    assert np.all(np.unique(w_j) == (3, 4, 5, 6))
    assert np.all(np.unique(w_i) == (4, 5))
    w_k, w_j, w_i = rqgs.where_true(j_faces)
    assert len(w_k) == 2
    assert np.all(np.unique(w_k) == (4, 5))
    assert np.all(np.unique(w_j) == (2,))
    assert np.all(np.unique(w_i) == (4,))
    w_k, w_j, w_i = rqgs.where_true(i_faces)
    assert len(w_k) == 9
    assert np.all(np.unique(w_k) == (2, 3, 4))
    assert np.all(np.unique(w_j) == (3, 4, 5))
    assert np.all(np.unique(w_i) == (5,))

    bounds = rqgs.get_boundary(k_faces, j_faces, i_faces, grid_extent_kji)

    # note: get_boundary() includes a buffer slice where faces do not reach edge of grid
    assert bounds['k_min'] == 1
    assert bounds['k_max'] == 6
    assert bounds['j_min'] == 2
    assert bounds['j_max'] == 7
    assert bounds['i_min'] == 3
    assert bounds['i_max'] == 6


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
    ca = rqgs.column_bisector_from_faces(grid_extent_kji[1:], j_faces[0], i_faces[0])

    # Assert
    np.all(a == np.array([[[True, True, True], [False, False, False], [False, False, False]],
                          [[True, True, True], [False, False, False], [False, False, False]],
                          [[True, True, True], [False, False, False], [False, False, False]]],
                         dtype = bool))
    assert is_curtain is True
    assert ca.shape == tuple(grid_extent_kji[1:])
    assert np.all(ca == a[0]) or np.all(ca == np.logical_not(a[0]))


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
