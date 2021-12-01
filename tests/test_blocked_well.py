import os

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

import resqpy.olio.uuid as bu
from resqpy.grid import RegularGrid
from resqpy.model import Model
from resqpy.property import Property
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
    bw = resqpy.well.BlockedWell(model,
                                 wellspec_file = wellspec_file,
                                 well_name = well_name,
                                 use_face_centres = True,
                                 add_wellspec_properties = True)
    assert bw is not None
    bw_uuid = bw.uuid
    skin_uuid = model.uuid(title = 'SKIN', related_uuid = bw.uuid)
    assert skin_uuid is not None
    skin_prop = Property(model, uuid = skin_uuid)
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
                       use_properties = ['SKIN', 'RADW'])
    for col in ['SKIN', 'RADW']:
        assert_array_almost_equal(np.array(source_df[col]), np.array(df3[col]))


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
    assert grid_uuid_list[0] == grid_uuid


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


#
# --------- Act ----------
    bw = resqpy.well.BlockedWell(model, use_face_centres = True)
    # assert that certain attributes have not bee populated
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
