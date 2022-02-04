import pytest
import numpy as np
import resqpy.model as rq
import resqpy.property as rqp
import math as maths
from resqpy.rq_import._import_vdb_ensemble import import_vdb_ensemble
from resqpy.rq_import._import_nexus import import_nexus


@pytest.mark.skip('Blocked by issue #416')
def test_default_args(tmp_path):
    # Arrange
    ensemble_dir = '../../test_data/wren'
    epc_file = f'{tmp_path}/test.epc'
    parts_expected = [('ContinuousProperty', 123), ('DiscreteProperty', 15), ('EpcExternalPartReference', 1),
                      ('IjkGridRepresentation', 1), ('LocalDepth3dCrs', 1), ('PropertyKind', 1), ('PropertySet', 7),
                      ('TimeSeries', 1)]
    pk_list_expected = [
        'cell length', 'code', 'continuous', 'depth', 'fluid volume', 'index', 'permeability thickness', 'pore volume',
        'pressure', 'region initialization', 'rock volume', 'saturation', 'thickness', 'transmissibility'
    ]
    pc_titles_expected = {
        'MDEP', 'DEADCELL', 'SG', 'IGRID', 'OIP', 'SO', 'WIP', 'KID', 'GIP', 'DAD', 'DZN', 'TX', 'KH', 'SW', 'UID',
        'DXC', 'DZC', 'BV', 'TZ', 'P', 'PVR', 'UNPACK', 'DYC', 'TNSC', 'TY'
    }

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir)
    model = rq.Model(epc_file)
    parts = model.parts_count_by_type()
    grid = model.grid()
    pc = grid.property_collection
    pc_titles = {pc.title_for_part(part) for part in pc.parts()}
    pc_realization_list = pc.realization_list()
    pk_list = pc.property_kind_list()
    mean_pv = np.nanmean(pc.single_array_ref(realization = 1, property_kind = 'pore volume'))
    ts_uuid_list = pc.time_series_uuid_list()
    time_index_list = pc.time_index_list()
    sat_pc = rqp.selective_version_of_collection(pc, property_kind = 'saturation', realization = 2)
    part = sat_pc.parts()[-1]
    sat_ft = sat_pc.facet_type_for_part(part)
    sat_facet = sat_pc.facet_for_part(part)
    sat_r = sat_pc.realization_for_part(part)
    sat_ti = sat_pc.time_index_for_part(part)
    sat_ts_uuid = sat_pc.time_series_uuid_for_part(part)
    sat_title = sat_pc.title_for_part(part)
    ftl = sat_pc.facet_type_list()
    fl = sat_pc.facet_list()
    no = sat_pc.number_of_parts()

    # Assert
    assert parts == parts_expected
    assert grid is not None
    assert pc is not None
    assert pc.has_multiple_realizations()
    assert pc_titles == pc_titles_expected
    assert pk_list == pk_list_expected
    assert pc_realization_list == [0, 1, 2]
    assert maths.isclose(mean_pv, 172682.75305989583)
    assert len(ts_uuid_list) == 1
    assert time_index_list == [0, 1, 2, 3]
    assert sat_ft == 'what'
    assert sat_facet == 'water'
    assert sat_r == 2
    assert sat_ti == 3
    assert sat_ts_uuid is not None
    assert sat_title == 'SW'
    assert ftl == ['what']
    assert fl == ['gas', 'oil', 'water']


def test_existing_epc_true(tmp_path):
    # Arrange
    ensemble_dir = '../../test_data/wren'
    case_dir = f'{ensemble_dir}/wren2.vdb'
    epc_file = f'{tmp_path}/test.epc'
    import_nexus(epc_file[:-4], vdb_file = case_dir, vdb_static_properties = False, vdb_recurrent_properties = False)

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, existing_epc = True)
    model = rq.Model(epc_file)

    # Assert
    assert model.number_of_parts() == 150


def test_keyword_list(tmp_path):
    # Arrange
    ensemble_dir = '../../test_data/wren'
    epc_file = f'{tmp_path}/test.epc'
    keyword_set = {'PVR', 'MDEP', 'KH', 'SW', 'SO', 'P'}

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, keyword_list = keyword_set)
    model = rq.Model(epc_file)
    pc = model.grid().property_collection
    pc_keys = {pc.title_for_part(part) for part in pc.parts()}

    # Assert
    assert set(model.titles(parts_list = pc.parts())) == keyword_set


def test_property_kind_list(tmp_path):
    # Arrange
    ensemble_dir = '../../test_data/wren'
    epc_file = f'{tmp_path}/test.epc'
    property_kind_set = {'pore volume', 'permeability thickness', 'depth', 'pressure', 'saturation'}

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, property_kind_list = property_kind_set)
    model = rq.Model(epc_file)
    pc = model.grid().property_collection

    # Assert
    assert set(pc.property_kind_list()) == property_kind_set


@pytest.mark.parametrize("vdb_static_properties, vdb_recurrent_properties, no_parts_expected", [(True, True, 138),
                                                                                                (True, False, 39),
                                                                                                (False, True, 99)])
def test_vdb_properties(tmp_path, vdb_static_properties, vdb_recurrent_properties, no_parts_expected):
    # Arrange
    ensemble_dir = '../../test_data/wren'
    epc_file = f'{tmp_path}/test.epc'

    # Act
    import_vdb_ensemble(epc_file,
                        ensemble_dir,
                        vdb_static_properties = vdb_static_properties,
                        vdb_recurrent_properties = vdb_recurrent_properties)
    model = rq.Model(epc_file)
    pc = model.grid().property_collection

    # Assert
    assert pc.number_of_parts() == no_parts_expected


@pytest.mark.parametrize("timestep_selection", ['first', 'last', 'first and last', 'all'])
def test_timestep_collection(tmp_path, timestep_selection):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, timestep_selection = timestep_selection)

    # Assert
    pass


def test_create_property_set_per_realization_false(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, create_property_set_per_realization = False)

    # Assert
    pass


def test_create_property_set_per_timestep_false(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, create_property_set_per_timestep = False)

    # Assert
    pass


@pytest.mark.parametrize("resqml_xy_units, resqml_z_units", [('m', 'm'), ('ft', 'ft'), ('m', 'ft')])
def test_resqml_units(tmp_path, resqml_xy_units, resqml_z_units):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, resqml_xy_units = resqml_xy_units, resqml_z_units = resqml_z_units)

    # Assert
    pass


def test_split_pillars_false(tmp_path):
    # Arrange
    ensemble_dir = tmp_path
    epc_file = f'{ensemble_dir}/test.epc'

    # Act
    import_vdb_ensemble(epc_file, ensemble_dir, split_pillars = False)

    # Assert
    pass
