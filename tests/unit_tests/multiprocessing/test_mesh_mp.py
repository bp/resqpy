from pathlib import Path

from resqpy.model import Model
from resqpy.surface import Mesh
from resqpy.multi_processing.wrappers import mesh_mp


def test_mesh_from_regular_grid_column_property_wrapper(small_grid_with_properties, tmp_path: Path):
    # Arrange
    grid, prop_uuids = small_grid_with_properties
    grid_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    input_index = 0

    # Act
    (
        index,
        success,
        epc_file,
        uuid_list,
    ) = mesh_mp.mesh_from_regular_grid_column_property_wrapper(
        input_index,
        tmp_path,
        grid_epc,
        grid_uuid,
        prop_uuids,
    )
    model = Model(epc_file = epc_file)

    # Assert
    assert success is True
    assert index == input_index
    for prop_uuid in prop_uuids:
        mesh_uuid = model.uuid(obj_type = "Grid2dRepresentation", related_uuid = prop_uuid)
        assert mesh_uuid is not None
        mesh = Mesh(model, uuid = mesh_uuid)
        assert mesh is not None
        assert mesh.ni == grid.ni
        assert mesh.nj == grid.nj
    assert len(uuid_list) == 21
