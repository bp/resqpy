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
    grid_uuid = grid.uuid
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25], [2, 2, 2, 0.45, -90.0, 2.5, 0.25],
                              [2, 3, 2, 0.45, -90.0, 1.0, 0.20], [2, 3, 3, 0.0, 0.0, -0.5, 0.20]],
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
    assert bw is not None
    bw_uuid = bw.uuid
    skin_uuid = model.uuid(title = 'SKIN', related_uuid = bw.uuid)
    assert skin_uuid is not None
    skin_prop = Property(model, uuid = skin_uuid)
    assert skin_prop is not None
    assert_array_almost_equal(skin_prop.array_ref(), [0.0, 2.5, 1.0, -0.5])
    model.store_epc()
    # re-open model from persistent storage
    model = Model(model.epc_file)
    bw2_uuid = model.uuid(obj_type = 'BlockedWellboreRepresentation', title = 'DOGLEG')
    assert bw2_uuid is not None
    bw2 = resqpy.well.BlockedWell(model, uuid = bw2_uuid)
    assert bu.matching_uuids(bw_uuid, bw2_uuid)
    df2 = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'LENGTH', 'SKIN', 'RADW'], use_properties = True)
    assert df2 is not None
    assert len(df2.columns) == 8
    for col in ['ANGLV', 'ANGLA', 'SKIN', 'RADW']:
        assert_array_almost_equal(np.array(source_df[col]), np.array(df2[col]))
    df3 = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'LENGTH', 'SKIN', 'RADW'],
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
