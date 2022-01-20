import os
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
import pytest

import resqpy.olio.uuid as bu
from resqpy.grid import RegularGrid
from resqpy.model import Model
import resqpy.property as rqp
from resqpy.well.well_utils import _derive_from_wellspec_check_grid_name
import resqpy.well


def test_wellspec_properties(example_model_and_crs):
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9],
                              [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9]],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF'])
    with open(wellspec_file, 'w') as fp:
        fp.write(F'WELLSPEC {well_name}\n')
        for col in source_df.columns:
            fp.write(f' {col:>6s}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col in ['IW', 'JW', 'L']:
                    fp.write(f' {int(row[col]):6d}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = resqpy.well.BlockedWell(
        model,
        wellspec_file = wellspec_file,
        well_name = well_name,
        use_face_centres = True,
        add_wellspec_properties = True,
    )
    assert bw is not None
    bw_uuid = bw.uuid
    skin_uuid = model.uuid(title = 'SKIN', related_uuid = bw.uuid)
    assert skin_uuid is not None
    skin_prop = rqp.Property(model, uuid = skin_uuid)
    assert skin_prop is not None
    assert_array_almost_equal(skin_prop.array_ref(), [0.0, 2.5, 1.0, -0.5])
    model.store_epc()
    print(model.grid().property_collection.titles())
    # re-open model from persistent storage
    model = Model(model.epc_file)
    bw2_uuid = model.uuid(obj_type = 'BlockedWellboreRepresentation', title = 'DOGLEG')
    assert bw2_uuid is not None
    bw2 = resqpy.well.BlockedWell(model, uuid = bw2_uuid)
    assert bu.matching_uuids(bw_uuid, bw2_uuid)
    df2 = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'LENGTH', 'SKIN', 'RADW', 'PPERF'],
                       use_properties = True)
    assert df2 is not None
    assert len(df2.columns) == 9
    for col in ['ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF']:
        assert_array_almost_equal(np.array(source_df[col]), np.array(df2[col]))
    df3 = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'LENGTH', 'SKIN', 'RADW', 'PPERF'],
                       use_properties = ['SKIN', 'RADW'],
                       perforation_list = [(125, 175)])
    for col in ['SKIN', 'RADW']:
        assert_array_almost_equal(np.array(source_df[col]), np.array(df3[col]))


@pytest.mark.parametrize('check_grid_name,name_for_check,col_list', [(True, 'BATTLESHIP', ['IW', 'JW', 'L', 'GRID']),
                                                                     (False, None, ['IW', 'JW', 'L'])])
def test_derive_from_wellspec_check_grid_name(example_model_and_crs, check_grid_name, name_for_check, col_list):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 3, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       title = 'Battleship',
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    col_list_orig = ['IW', 'JW', 'L']

    # --------- Act ----------
    result = _derive_from_wellspec_check_grid_name(check_grid_name = check_grid_name,
                                                   grid = grid,
                                                   col_list = col_list_orig)

    # --------- Assert ----------
    assert result[0] == name_for_check
    assert result[1] == col_list


def test_set_for_column(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 3, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)

    # --------- Act ----------
    # populate empty blocked well object for a 'vertical' well in the given column
    bw.set_for_column(well_name = well_name, grid = grid, col_ji0 = (1, 1))

    # --------- Assert ----------
    assert bw.cell_count == 5
    assert bw.node_count == len(
        bw.node_mds) == len(bw.trajectory.measured_depths) - 1  # tail added to trajectory measured depths
    assert (len(bw.grid_list) == 1) & (bw.grid_list[0] == grid)


def test_derive_from_cell_list(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 3, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    cell_kji0_list = np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3), (3, 3, 4)])
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)

    # --------- Act ----------
    # populate empty blocked well object for a 'vertical' well in the given column
    bw.derive_from_cell_list(cell_kji0_list = cell_kji0_list, well_name = well_name, grid = grid)

    # --------- Assert ----------
    # no tail added to trajectory measured depths as well terminates in 4th layer
    assert bw.node_count == len(bw.node_mds) == len(bw.trajectory.measured_depths)
    assert bw.cell_count == len(cell_kji0_list)


def test_grid_uuid_list(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 3, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    grid_uuid = grid.uuid
    well_name = 'DOGLEG'
    cell_kji0_list = np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3), (3, 3, 4)])
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)

    # --------- Act ----------
    # populate empty blocked well object for a 'vertical' well in the given column
    bw.derive_from_cell_list(cell_kji0_list = cell_kji0_list, well_name = well_name, grid = grid)
    # get list of grid uuids associated with this blocked well
    grid_uuid_list = bw.grid_uuid_list()

    # --------- Assert ----------
    assert len(grid_uuid_list) == 1
    assert bu.matching_uuids(grid_uuid_list[0], grid_uuid)


def test_verify_grid_name(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9, 'grid_1'], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9, 'grid_1'],
         [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9, 'grid_2'], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9, 'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    row = source_df.iloc[2]
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True)

    # --------- Act ----------
    skipped_warning_grid, skip_row = bw._BlockedWell__verify_grid_name(grid_name_to_check = 'grid_1',
                                                                       row = row,
                                                                       skipped_warning_grid = None,
                                                                       well_name = well_name)

    # --------- Assert ----------
    assert skipped_warning_grid == 'grid_2'
    assert skip_row


def test_calculate_exit_and_entry(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    cp = grid.corner_points(cell_kji0 = (2, 2, 1))
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9, 'grid_1'], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9, 'grid_1'],
         [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9, 'grid_1'], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9, 'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    row = source_df.iloc[0]
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True)

    # --------- Act ----------
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) = bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name)
    print((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))

    # --------- Assert ----------
    # TODO: less trivial assertion
    for value in (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz):
        assert value is not None


def test_calculate_cell_cp_center_and_vectors(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    cp_expected = grid.corner_points(cell_kji0 = (2, 2, 1))
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9, 'grid_1'], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9, 'grid_1'],
         [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9, 'grid_1'], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9, 'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True)

    # --------- Act ----------
    cp, cell_centre, entry_vector, exit_vector = bw._BlockedWell__calculate_cell_cp_center_and_vectors(
        grid = grid,
        cell_kji0 = (2, 2, 1),
        entry_xyz = np.array([75, -125, 200]),
        exit_xyz = np.array([100, -100, 250]),
        well_name = well_name)

    # --------- Assert ----------
    np.testing.assert_equal(cp, cp_expected)
    np.testing.assert_equal(cell_centre, np.array([75., -125., 225.]))
    np.testing.assert_equal(entry_vector, np.array([0., 0., -2500.]))
    np.testing.assert_equal(exit_vector, np.array([2500., 2500., 2500.]))


def test_import_from_cellio_file(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)

    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    grid_uuid = grid.uuid
    cellio_file = os.path.join(model.epc_directory, 'cellio.dat')
    well_name = 'Banoffee'
    source_df = pd.DataFrame(
        [[2, 2, 1, 25, -25, 125, 26, -26, 126], [2, 2, 2, 26, -26, 126, 27, -27, 127],
         [2, 2, 3, 27, -27, 127, 28, -28, 128]],
        columns = ['i_index', 'j_index', 'k_index', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'])

    with open(cellio_file, 'w') as fp:
        fp.write('1.0\n')
        fp.write('Undefined\n')
        fp.write(f'{well_name}\n')
        fp.write('9\n')
        for col in source_df.columns:
            fp.write(f' {col}\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                fp.write(f' {int(row[col])}')
            fp.write('\n')


# --------- Act ----------
    bw = resqpy.well.BlockedWell(model, use_face_centres = True)
    # assert that certain attributes have not been populated
    assert bw.grid_list == []
    assert bw.trajectory is None
    assert bw.cell_count is None
    assert bw.node_count is None
    assert bw.node_mds is None

    bw.import_from_rms_cellio(cellio_file = cellio_file, well_name = well_name, grid = grid)

    # --------- Assert ----------
    assert bw.grid_list[0].uuid == grid_uuid
    assert bw.trajectory is not None
    assert bw.cell_count == 3  # 3 lines in the cellio file
    assert bw.node_count == len(bw.node_mds) == 4  # added tail to trajectory


@pytest.mark.parametrize('ntg_multiplier,length_mode,status', [(0.5, 'straight', 'OFF'), (1, 'MD', 'ON')])
def test_dataframe(example_model_and_crs, ntg_multiplier, length_mode, status):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    perm_i_array = np.random.random(grid.extent_kji)
    ntg_array = np.ones(grid.extent_kji) * ntg_multiplier
    perm_i_prop = rqp.Property.from_array(model,
                                          perm_i_array,
                                          source_info = 'random',
                                          keyword = 'PERMI',
                                          support_uuid = grid.uuid,
                                          property_kind = 'permeability rock',
                                          indexable_element = 'cells',
                                          uom = 'Euc')
    ntg_prop = rqp.Property.from_array(model,
                                       ntg_array,
                                       keyword = 'NTG',
                                       source_info = 'random',
                                       support_uuid = grid.uuid,
                                       property_kind = 'net to gross ratio',
                                       indexable_element = 'cells',
                                       uom = 'Euc')
    perm_i_prop.write_hdf5()
    perm_i_prop.create_xml()
    ntg_prop.write_hdf5()
    ntg_prop.create_xml()
    perm_i_uuid = perm_i_prop.uuid
    ntg_uuid = ntg_prop.uuid
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25], [2, 2, 2, 45, -90.0, 2.5, 0.25], [2, 3, 2, 45, -90.0, 1.0, 0.20],
         [2, 3, 3, 45, -90.0, -0.5, 0.20], [2, 3, 4, 45, -90.0, 1.1, 0.20], [2, 3, 5, 0.0, 0.0, 1.0, 0.20]],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW'])
    with open(wellspec_file, 'w') as fp:
        fp.write(F'WELLSPEC {well_name}\n')
        for col in source_df.columns:
            fp.write(f' {col:>6s}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col in ['IW', 'JW', 'L']:
                    fp.write(f' {int(row[col]):6d}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = resqpy.well.BlockedWell(model,
                                 wellspec_file = wellspec_file,
                                 well_name = well_name,
                                 use_face_centres = True,
                                 add_wellspec_properties = True)

    # --------- Act ----------
    df = bw.dataframe(extra_columns_list = ['X', 'Y', 'DEPTH', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'KH', 'WI', 'WBC'],
                      add_as_properties = True,
                      perforation_list = [(125, 320)],
                      max_depth = 500,
                      perm_i_uuid = perm_i_uuid,
                      perm_j_uuid = perm_i_uuid,
                      ntg_uuid = ntg_uuid,
                      preferential_perforation = True,
                      stat = status,
                      min_k0 = 1,
                      max_k0 = 5,
                      use_face_centres = True,
                      length_uom = 'm',
                      length_mode = length_mode)

    # --------- Assert ----------
    # print(df)
    assert all(df['KH'])  # successfully added a KH column as an i-direction permeability array was specified
    assert set(df['STAT']) == {status}
    assert all(df['WI'])
    assert all(df['WBC'])


def test_dataframe_from_trajectory(example_model_and_crs):
    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 4, 4),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)

    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([100, 210, 230, 240, 250])
    zs = mds - elevation
    well_name = 'CoconutDrop'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [25, 50, 75, 100, 100],
        'Y': [25, -50, -75, -100, -100],
        'Z': zs,
        'WELL': ['CoconutDrop', 'CoconutDrop', 'CoconutDrop', 'CoconutDrop', 'CoconutDrop']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')
    trajectory.write_hdf5()
    trajectory.create_xml()
    bw = resqpy.well.BlockedWell(model,
                                 well_name = well_name,
                                 grid = grid,
                                 trajectory = trajectory,
                                 use_face_centres = True)

    # --------- Act ----------
    df = bw.dataframe(extra_columns_list = ['X', 'Y', 'DEPTH', 'RADW'], stat = 'ON', length_uom = 'ft')

    # --------- Assert ----------
    assert bw.trajectory is not None
    assert set(df['RADW']) == {0.25}
    assert set(df['STAT']) == {'ON'}
    assert all(df['X']) == all(df['Y']) == all(df['DEPTH'])  # successfully got xyz points from trajectory and
    # converted them to ft


def test_write_wellspec(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    perm_array = np.random.random(grid.extent_kji)
    perm_prop = rqp.Property.from_array(model,
                                        perm_array,
                                        source_info = 'random',
                                        keyword = 'PERMI',
                                        support_uuid = grid.uuid,
                                        property_kind = 'permeability',
                                        indexable_element = 'cells',
                                        uom = 'Euc')
    perm_prop.write_hdf5()
    perm_prop.create_xml()
    perm_uuid = perm_prop.uuid
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25], [2, 2, 2, 45, -90.0, 2.5, 0.25],
                              [2, 3, 2, 45, -90.0, 1.0, 0.20], [2, 3, 3, 0.0, 0.0, -0.5, 0.20]],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW'])
    with open(wellspec_file, 'w') as fp:
        fp.write(F'WELLSPEC {well_name}\n')
        for col in source_df.columns:
            fp.write(f' {col:>6s}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col in ['IW', 'JW', 'L']:
                    fp.write(f' {int(row[col]):6d}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = resqpy.well.BlockedWell(model,
                                 wellspec_file = wellspec_file,
                                 well_name = well_name,
                                 use_face_centres = True,
                                 add_wellspec_properties = True)

    # --------- Act ----------
    df = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'SKIN', 'RADW', 'KH'],
                      add_as_properties = True,
                      perm_i_uuid = perm_uuid)

    wellspec_file2 = os.path.join(model.epc_directory, 'wellspec2.dat')
    df2 = bw.write_wellspec(wellspec_file = wellspec_file2,
                            well_name = well_name,
                            extra_columns_list = ['ANGLV', 'ANGLA', 'SKIN', 'RADW'],
                            length_uom = 'm',
                            length_uom_comment = '?')

    # --------- Assert ----------
    pd.testing.assert_frame_equal(df[['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW']], df2, check_dtype = False)
    # TODO find out why initially when ANGLV was 0.45, the Blocked Well dataframe method changed the values to 45
    # TODO find out why AngleA values of 0 transformed to nan values?


def test_convenience_methods_xyz_and_kji0_marker(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame([[1, 2, 2, 0.0, 0.0, 0.0, 0.25], [2, 2, 2, 45, -90.0, 2.5, 0.25],
                              [2, 3, 2, 45, -90.0, 1.0, 0.20], [2, 3, 3, 0.0, 0.0, -0.5, 0.20]],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW'])
    with open(wellspec_file, 'w') as fp:
        fp.write(F'WELLSPEC {well_name}\n')
        for col in source_df.columns:
            fp.write(f' {col:>6s}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col in ['IW', 'JW', 'L']:
                    fp.write(f' {int(row[col]):6d}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = resqpy.well.BlockedWell(model,
                                 wellspec_file = wellspec_file,
                                 well_name = well_name,
                                 use_face_centres = True,
                                 add_wellspec_properties = True)

    # --------- Act ----------
    cell_0, grid_0_uuid = bw.kji0_marker()  # cell_0 returned in the format (k0, j0, i0)
    _, crs_uuid = bw.xyz_marker()

    # --------- Assert ----------
    np.testing.assert_equal(cell_0, np.array([1, 1, 0]))  # start blocked well at (iw, jw, l) == (1, 2, 2)
    assert grid_0_uuid == grid.uuid
    assert crs_uuid == crs.uuid  # TODO: less trivial assertion?
