import pytest

from resqpy.rq_import._import_vdb_ensemble import import_vdb_ensemble


def test_default_args(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir)

    # Assert
    pass


def test_existing_epc_true(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, existing_epc=True)

    # Assert
    pass


def test_keyword_list(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'
    keyword_list = ['PVR', 'MDEP', 'KH', 'PRESSURE', 'SATW', 'SATO']

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, keyword_list=keyword_list)

    # Assert
    pass


def test_property_kind_list(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'
    property_kind_list = ['pore volume', 'permeability thickness', 'cell depth', 'pressure', 'saturation']

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, property_kind_list=property_kind_list)

    # Assert
    pass


@pytest.mark.parametrize("vdb_static_properties, vdb_recurrent_properties",
                         [(True, True), (True, False), (False, True)])
def test_vdb_properties(tmp_path, vdb_static_properties, vdb_recurrent_properties):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, vdb_static_properties=vdb_static_properties,
                        vdb_recurrent_properties=vdb_recurrent_properties)

    # Assert
    pass


@pytest.mark.parametrize("timestep_selection", ['first', 'last', 'first and last', 'all'])
def test_timestep_collection(tmp_path, timestep_selection):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, timestep_selection=timestep_selection)

    # Assert
    pass


def test_create_property_set_per_realization_false(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, create_property_set_per_realization=False)

    # Assert
    pass


def test_create_property_set_per_timestep_false(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, create_property_set_per_timestep=False)

    # Assert
    pass


@pytest.mark.parametrize("resqml_xy_units, resqml_z_units",
                         [('metres', 'metres'), ('feet', 'feet'), ('metres', 'feet')])
def test_resqml_units(tmp_path, resqml_xy_units, resqml_z_units):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, resqml_xy_units=resqml_xy_units, resqml_z_units=resqml_z_units)

    # Assert
    pass


def test_split_pillars_false(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, split_pillars=False)

    # Assert
    pass
