import resqpy.grid_surface as rqgs
import numpy as np


def test_find_faces_to_represent_surface_regular_optimised(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    # Assert
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)


def test_find_faces_to_represent_surface_regular_optimised_with_consistent_side(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"

    def sort_array(array):
        return np.sort(array)[np.lexsort((np.sort(array)[:, 1], np.sort(array)[:, 0]))]

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name, consistent_side = True)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name, consistent_side = True)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    # Assert
    np.testing.assert_array_equal(sort_array(cip_normal), sort_array(cip_optimised))
    np.testing.assert_array_equal(sort_array(fip_normal), sort_array(fip_optimised))


def test_find_faces_to_represent_surface_regular_optimised_with_return_properties(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    return_properties = ['offset', 'normal vector']

    # Act
    gcs_normal, properties_dict = rqgs.find_faces_to_represent_surface_regular(grid,
                                                                               surface,
                                                                               name,
                                                                               return_properties = return_properties)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs
    offsets_normal = properties_dict["offset"]
    normal_vectors_normal = properties_dict["normal vector"]

    return_properties.append('depth')
    return_properties.append('triangle')
    gcs_optimised, properties_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(
        grid, surface, name, return_properties = return_properties)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs
    triangles_optimised = properties_optimised['triangle']
    depths_optimised = properties_optimised["depth"]
    offsets_optimised = properties_optimised["offset"]
    normal_vectors_optimised = properties_optimised["normal vector"]

    # Assert
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_almost_equal(offsets_normal, offsets_optimised)
    np.testing.assert_array_almost_equal(normal_vectors_normal, normal_vectors_optimised)
    assert depths_optimised.shape == offsets_optimised.shape
    assert np.all(depths_optimised > 0.0)
    assert triangles_optimised.shape == offsets_optimised.shape
    assert np.all(triangles_optimised >= 0)
