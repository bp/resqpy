from resqpy.multiprocessing.wrappers.grid_surface import find_faces_to_represent_surface_regular_wrapper
from resqpy.model import Model


def test_find_faces_to_represent_surface_regular_wrapper(small_grid_and_surface, tmp_path):
    # Arrange
    grid = small_grid_and_surface[0]
    grid_epc = grid.model.epc_file
    grid_uuid = grid.uuid

    surface = small_grid_and_surface[1]
    surface_epc = surface.model.epc_file
    surface_uuid = surface.uuid

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index, tmp_path, use_index_as_realisation, grid_epc, grid_uuid, surface_epc, surface_uuid, name)
    model = Model(epc_file = epc_file)

    # Assert
    assert success is True
    assert len(uuid_list) == 3
    assert index == input_index
    assert len(model.uuids()) == 6


def test_find_faces_to_represent_surface_regular_wrapper_properties(small_grid_and_surface, tmp_path):
    # Arrange
    grid = small_grid_and_surface[0]
    grid_epc = grid.model.epc_file
    grid_uuid = grid.uuid

    surface = small_grid_and_surface[1]
    surface_epc = surface.model.epc_file
    surface_uuid = surface.uuid
    return_properties = ["normal vector", "triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        tmp_path,
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties)
    model = Model(epc_file = epc_file)

    # Assert
    assert success is True
    assert len(uuid_list) == 6
    assert index == input_index
    assert len(model.uuids()) == 9
