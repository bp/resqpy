from resqpy.multiprocessing.wrappers.grid_surface_mp import find_faces_to_represent_surface_regular_wrapper
from resqpy.model import Model
from typing import Tuple
from resqpy.grid import RegularGrid
from resqpy.surface import Surface
from resqpy.multiprocessing.multiprocessing import rm_tree


def test_find_faces_to_represent_surface_regular_wrapper(small_grid_and_surface: Tuple[RegularGrid, Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(input_index,
                                                                                          use_index_as_realisation,
                                                                                          grid_epc,
                                                                                          grid_uuid,
                                                                                          surface_epc,
                                                                                          surface_uuid,
                                                                                          name,
                                                                                          trimmed = True)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids()) == 9
    assert len(uuid_list) == 7


def test_find_faces_to_represent_surface_regular_wrapper_properties(small_grid_and_surface: Tuple[RegularGrid,
                                                                                                  Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["normal vector", "triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 1
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids(obj_type = 'PointsProperty')) == 1
    assert len(model.uuids()) == 13
    assert len(uuid_list) == 10


def test_find_faces_to_represent_surface_extended_bisector(small_grid_and_extended_surface: Tuple[RegularGrid,
                                                                                                  Surface]):
    # Arrange
    grid, surface = small_grid_and_extended_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["normal vector", "triangle", "offset", "grid bisector"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 2
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids(obj_type = 'PointsProperty')) == 1
    assert len(model.uuids()) == 14
    assert len(uuid_list) == 11


def test_find_faces_to_represent_surface_regular_wrapper_properties_flange(small_grid_and_surface: Tuple[RegularGrid,
                                                                                                         Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["normal vector", "triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        extend_fault_representation = True)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 2
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids(obj_type = 'PointsProperty')) == 1
    assert len(model.uuids()) == 14
    assert len(uuid_list) == 11
