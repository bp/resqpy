import pytest
from resqpy.multiprocessing.wrappers.grid_surface import find_faces_to_represent_surface_regular_wrapper
from resqpy.model import Model


@pytest.mark.skip("Incomplete")
def test_find_faces_to_represent_surface_regular_wrapper(small_grid_and_surface, tmp_path):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index, tmp_path, use_index_as_realisation, grid, surface, name)
    model = Model(epc_file = epc_file)

    # Assert
    assert success is True
    assert len(uuid_list) == 3
    assert index == input_index
    assert len(model.uuids()) == 6


@pytest.mark.skip("Incomplete")
def test_find_faces_to_represent_surface_regular_wrapper_properties(small_grid_and_surface, tmp_path):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    input_index = 0
    use_index_as_realisation = False
    return_properties = ["normal vector", "offset", "triangle"]

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index, tmp_path, use_index_as_realisation, grid, surface, name, return_properties = return_properties)
    model = Model(epc_file = epc_file)

    # Assert
    assert success is True
    assert len(uuid_list) == 6
    assert index == input_index
    assert len(model.uuids()) == 6
