import logging

log = logging.getLogger(__name__)

import os
import math as maths
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
import pytest

import resqpy
import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.time_series as rqts
import resqpy.well as rqw
import resqpy.olio.uuid as bu
from resqpy.well.well_utils import _derive_from_wellspec_check_grid_name
from resqpy.well import BlockedWell


def test_wellspec_properties(example_model_and_crs):
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    # yapf: disable
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9],
                              [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9],
                              [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9],
                              [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9]],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF'])
    # yapf: enable
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
                elif col == 'STAT':
                    fp.write(f' {"ON" if row[col] else "OFF":4}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = rqw.BlockedWell(model,
                         wellspec_file = wellspec_file,
                         well_name = well_name,
                         use_face_centres = True,
                         add_wellspec_properties = True)
    assert bw is not None
    bw_uuid = bw.uuid
    skin_uuid = model.uuid(title = 'SKIN', related_uuid = bw.uuid)
    assert skin_uuid is not None
    skin_prop = rqp.Property(model, uuid = skin_uuid)
    assert skin_prop is not None
    assert_array_almost_equal(skin_prop.array_ref(), [0.0, 2.5, 1.0, -0.5])
    model.store_epc()
    log.debug(model.grid().property_collection.titles())
    # re-open model from persistent storage
    model = rq.Model(model.epc_file)
    bw2_uuid = model.uuid(obj_type = 'BlockedWellboreRepresentation', title = 'DOGLEG')
    assert bw2_uuid is not None
    bw2 = rqw.BlockedWell(model, uuid = bw2_uuid)
    assert bu.matching_uuids(bw_uuid, bw2_uuid)
    pc = bw2.extract_property_collection()
    assert pc is not None and pc.number_of_parts() == 6
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
    length_df3 = np.array(df3['LENGTH'])
    df4 = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'LENGTH', 'SKIN', 'RADW', 'PPERF'],
                       use_properties = ['LENGTH', 'RADW'],
                       perforation_list = [(125, 175)],
                       length_uom = 'ft')
    assert_array_almost_equal(3.2808 * np.array(source_df['RADW']), np.array(df4['RADW']), decimal = 3)
    assert_array_almost_equal(3.2808 * length_df3, np.array(df4['LENGTH']), decimal = 2)


@pytest.mark.parametrize('check_grid_name,name_for_check,col_list', [(True, 'BATTLESHIP', ['IW', 'JW', 'L', 'GRID']),
                                                                     (False, None, ['IW', 'JW', 'L'])])
def test_derive_from_wellspec_check_grid_name(example_model_and_crs, check_grid_name, name_for_check, col_list):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 3, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           title = 'Battleship',
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'DOGLEG'
    col_list_orig = ['IW', 'JW', 'L']

    # --------- Act ----------
    result = _derive_from_wellspec_check_grid_name(check_grid_name = check_grid_name,
                                                   grid = grid,
                                                   col_list = col_list_orig)

    # --------- Assert ----------
    assert result[0] == name_for_check
    assert result[1] == col_list


def test_set_for_column_and_other_things(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 3, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'VERTICAL'
    bw = rqw.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)

    # --------- Act ----------
    # populate empty blocked well object for a 'vertical' well in the given column
    bw.set_for_column(well_name = well_name, grid = grid, col_ji0 = (1, 1))
    bw.write_hdf5()
    bw.create_xml()

    wellbore_frame_mds = np.array([90.0, 110.0, 220.0, 270.0, 290.0, 360.0], dtype = float)
    wellbore_frame = rqw.WellboreFrame(parent_model = model,
                                       trajectory = bw.trajectory,
                                       mds = wellbore_frame_mds,
                                       title = 'w.b.f.')
    assert wellbore_frame is not None and wellbore_frame.node_count == len(wellbore_frame_mds)
    assert wellbore_frame.node_mds is not None and len(wellbore_frame.node_mds) == wellbore_frame.node_count
    wellbore_frame.write_hdf5()
    wellbore_frame.create_xml()
    # add a time series for dynamic properties to refer to
    ts = rqts.TimeSeries(model, first_timestamp = '1997-03-03', yearly = 2, title = 'well connection times')
    ts.create_xml()
    # add some properties on the wellbore frame
    wbf_pc = wellbore_frame.extract_property_collection()
    a = np.array((True, False, True, True, False), dtype = bool)
    wbf_pc.add_cached_array_to_imported_list(a,
                                             'unit test',
                                             'perforated',
                                             discrete = True,
                                             property_kind = 'active',
                                             indexable_element = 'intervals')
    r = np.array((0.1, 0.1, 0.075, 0.075, 0.075), dtype = float)
    wbf_pc.add_cached_array_to_imported_list(r,
                                             'unit test',
                                             'hole size',
                                             discrete = False,
                                             property_kind = 'wellbore radius',
                                             uom = 'm',
                                             indexable_element = 'intervals')
    o = np.array([(False, False, True, True, False), (True, False, True, True, False),
                  (True, False, True, False, False)],
                 dtype = bool)
    for ti in range(3):
        wbf_pc.add_cached_array_to_imported_list(o[ti],
                                                 'unit test',
                                                 'open flag',
                                                 discrete = True,
                                                 property_kind = 'well connection open',
                                                 time_index = ti,
                                                 time_series_uuid = ts.uuid,
                                                 indexable_element = 'intervals')
    kh = np.array([(2000.0, 0.0, 2500.0, 3000.0, 0.0), (2000.0, 0.0, 2000.0, 3000.0, 0.0),
                   (2000.0, 0.0, 1500.0, 3000.0, 0.0)],
                  dtype = float)
    for ti in range(3):
        wbf_pc.add_cached_array_to_imported_list(kh[ti],
                                                 'unit test',
                                                 'KH',
                                                 discrete = False,
                                                 property_kind = 'permeability length',
                                                 uom = 'mD.m',
                                                 time_index = ti,
                                                 time_series_uuid = ts.uuid,
                                                 indexable_element = 'intervals')
    skin = np.array([(3.0, np.NaN, -1.0, 2.0, np.NaN), (3.5, np.NaN, -1.0, 2.0, np.NaN),
                     (4.0, np.NaN, -1.0, -2.0, np.NaN)],
                    dtype = float)
    for ti in range(3):
        wbf_pc.add_cached_array_to_imported_list(skin[ti],
                                                 'unit test',
                                                 'SKIN',
                                                 discrete = False,
                                                 property_kind = 'skin',
                                                 uom = 'Euc',
                                                 time_index = ti,
                                                 time_series_uuid = ts.uuid,
                                                 indexable_element = 'intervals')

    wbf_pc.write_hdf5_for_imported_list()
    uuids = wbf_pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 11

    model.store_epc()

    # --------- Assert ----------
    model = rq.Model(model.epc_file)
    bw = rqw.BlockedWell(model, uuid = bw.uuid)
    wellbore_frame = rqw.WellboreFrame(model, uuid = wellbore_frame.uuid)

    assert bw.cell_count == 5
    # note: tail added to trajectory measured depths
    assert bw.node_count == len(bw.node_mds) == len(bw.trajectory.measured_depths) - 1
    assert (len(bw.grid_list) == 1) and bu.matching_uuids(bw.grid_list[0].uuid, grid.uuid)

    # test wb_cell_for_md() method
    ci, f = bw.wb_cell_for_md(88.0)
    assert ci == -1 and maths.isclose(f, 0.88)
    ci, f = bw.wb_cell_for_md(110.0)
    assert ci == 0 and maths.isclose(f, 0.2)
    ci, f = bw.wb_cell_for_md(225.0)
    assert ci == 2 and maths.isclose(f, 0.5)
    ci, f = bw.wb_cell_for_md(351.0)
    assert ci == -1

    # test wellbore frame contributions per cell functionality
    contribs = bw.frame_contributions_list(wellbore_frame)
    assert len(contribs) == bw.cell_count
    c0 = contribs[0]  # contributions for cell 0
    assert len(c0) == 2
    c00 = c0[0]
    assert len(c00) == 3  # entries in inner lists are triplets
    assert c00[0] == 0 and maths.isclose(c00[1], 0.5) and maths.isclose(c00[2], 0.2)
    c01 = c0[1]
    assert c01[0] == 1 and maths.isclose(c01[1], 40.0 / 110.0) and maths.isclose(c01[2], 0.8)
    c1 = contribs[1]
    assert len(c1) == 1
    c10 = c1[0]
    assert c10[0] == 1 and maths.isclose(c10[1], 50.0 / 110.0) and maths.isclose(c10[2], 1.0)
    c2 = contribs[2]
    assert len(c2) == 2
    c20 = c2[0]
    assert c20[0] == 1 and maths.isclose(c20[1], 20.0 / 110.0) and maths.isclose(c20[2], 0.4)
    c21 = c2[1]
    assert c21[0] == 2 and maths.isclose(c21[1], 0.6) and maths.isclose(c21[2], 0.6)
    c3 = contribs[3]
    assert len(c3) == 3
    c30 = c3[0]
    assert c30[0] == 2 and maths.isclose(c30[1], 0.4) and maths.isclose(c30[2], 0.4)
    c31 = c3[1]
    assert c31[0] == 3 and maths.isclose(c31[1], 1.0) and maths.isclose(c31[2], 0.4)
    c32 = c3[2]
    assert c32[0] == 4 and maths.isclose(c32[1], 10.0 / 70.0) and maths.isclose(c32[2], 0.2)
    c4 = contribs[4]
    assert len(c4) == 1
    c40 = c4[0]
    assert c40[0] == 4 and maths.isclose(c40[1], 50.0 / 70.0) and maths.isclose(c40[2], 1.0)

    # check derivation of blocked well properties from wellbore frame properties
    pks = ['active', 'wellbore radius', 'well connection open', 'permeability length', 'skin']
    uuids = bw.add_properties_from_wellbore_frame(frame_uuid = wellbore_frame.uuid,
                                                  property_kinds_list = pks,
                                                  set_length = True,
                                                  set_perforation_fraction = True,
                                                  set_frame_interval = True)
    assert len(uuids) == 14
    model.store_epc()
    model = rq.Model(model.epc_file)
    bw = BlockedWell(model, uuid = bw.uuid)
    bw_pc = bw.extract_property_collection()
    # check active flag has been sampled correctly (cell active if any part of any frame interval active)
    bw_active = bw_pc.single_array_ref(property_kind = 'active')
    assert bw_active is not None and bw_active.shape == (bw.cell_count,)
    assert np.all(bw_active == (True, False, True, True, False))
    # check active length
    bw_length_part = bw_pc.singleton(property_kind = 'length')
    assert bw_length_part is not None
    assert bw_pc.uom_for_part(bw_length_part) == bw.trajectory.md_uom
    bw_length = bw_pc.cached_part_array_ref(bw_length_part)
    assert bw_length is not None and bw_length.shape == (bw.cell_count,)
    assert_array_almost_equal(bw_length, (10.0, 0.0, 30.0, 40.0, 0.0))
    # check perforation fraction
    bw_pperf = bw_pc.single_array_ref(property_kind = 'perforation fraction')
    assert bw_pperf is not None and bw_pperf.shape == (bw.cell_count,)
    assert_array_almost_equal(bw_pperf, (0.2, 0.0, 0.6, 0.8, 0.0))
    # check wellbore radius
    radw_part = bw_pc.singleton(property_kind = 'wellbore radius')
    assert radw_part is not None
    assert bw_pc.uom_for_part(radw_part) == 'm'
    radw = bw_pc.cached_part_array_ref(radw_part)
    assert radw is not None and radw.shape == (bw.cell_count,)
    assert_array_almost_equal(radw, (0.1, 0.1, 0.1, 0.075, 0.075))
    # check dynamic open connection flag
    open_flag_pc = rqp.selective_version_of_collection(bw_pc, property_kind = 'well connection open')
    assert open_flag_pc.number_of_parts() == 3
    open_flag = open_flag_pc.time_series_array_ref(fill_missing = False, indexable_element = 'cells')
    assert open_flag.shape == (3, bw.cell_count)
    # yapf: disable
    assert np.all(open_flag.astype(bool) == np.array([(False, False, True, True, False),
                                                      (True, False, True, True, False),
                                                      (True, False, True, True, False)], dtype = bool))
    # yapf: enable
    # check dynamic KH
    kh_pc = rqp.selective_version_of_collection(bw_pc, citation_title = 'KH')
    assert kh_pc.number_of_parts() == 3
    bw_kh = kh_pc.time_series_array_ref(fill_missing = False, indexable_element = 'cells')
    assert bw_kh.shape == (3, bw.cell_count)
    # yapf: disable
    expect_kh = np.array([(1000.0, 0.0, 1500.0, 4000.0, 0.0),
                          (1000.0, 0.0, 1200.0, 3800.0, 0.0),
                          (1000.0, 0.0,  900.0, 3600.0, 0.0)], dtype = float)
    # yapf: enable
    assert_array_almost_equal(bw_kh, expect_kh)
    # check dynamic skin
    skin_pc = rqp.selective_version_of_collection(bw_pc, property_kind = 'skin')
    assert skin_pc.number_of_parts() == 3
    bw_skin = skin_pc.time_series_array_ref(fill_missing = False, indexable_element = 'cells')
    assert bw_skin.shape == (3, bw.cell_count)
    # yapf: disable
    expect_skin = np.array([(3.0, np.NaN, -1, 0.5, np.NaN),
                            (3.5, np.NaN, -1, 0.5, np.NaN),
                            (4.0, np.NaN, -1, -1.5, np.NaN)], dtype = float)
    # yapf: enable
    assert_array_almost_equal(bw_skin, expect_skin)
    # check wellbore frame interval blocked well property
    wbf_ii_part = bw_pc.singleton(property_kind = 'wellbore frame interval')
    assert wbf_ii_part is not None
    wbf_ii_uuid = model.uuid_for_part(wbf_ii_part)
    rel_frame_uuid = model.uuid(obj_type = 'WellboreFrameRepresentation', related_uuid = wbf_ii_uuid, related_mode = 2)
    assert rel_frame_uuid is not None and bu.matching_uuids(rel_frame_uuid, wellbore_frame.uuid)
    wbf_ii = bw_pc.cached_part_array_ref(wbf_ii_part)
    assert wbf_ii is not None and wbf_ii.shape == (bw.cell_count,)
    assert np.all(wbf_ii == (0, -1, 2, 2, -1)) or np.all(wbf_ii == (0, -1, 2, 3, -1))


def test_derive_from_cell_list(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 3, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'DOGLEG'
    cell_kji0_list = np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3), (3, 3, 4)])
    bw = rqw.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)

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
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 3, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    grid_uuid = grid.uuid
    well_name = 'DOGLEG'
    cell_kji0_list = np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3), (3, 3, 4)])
    bw = rqw.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)

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
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'DOGLEG'
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9, 'grid_1'], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9, 'grid_1'],
         [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9, 'grid_2'], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9, 'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    row = source_df.iloc[2]
    bw = rqw.BlockedWell(model, well_name = well_name, use_face_centres = True)

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
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'DOGLEG'
    # todo: check sign of ANGLA relative to I axis
    # yapf: disable
    source_df = pd.DataFrame(
                  [[2,    2,    1,    0.0,     0.0,     0.0,   0.25,   0.9,    'grid_1'],
                   [2,    2,    2,   90.0,    90.0,     2.5,   0.25,   0.9,    'grid_1'],
                   [2,    3,    2,    0.0,     0.0,     1.0,   0.20,   0.9,    'grid_1'],
                   [2,    3,    3,   60.0,     0.0,    -0.5,   0.20,   0.9,    'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    # yapf: enable
    bw = rqw.BlockedWell(model, well_name = well_name)

    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (0, 1, 1))
    row = source_df.iloc[0]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (0, 0, 0, 1)
    assert_array_almost_equal(entry_xyz, (75.0, -75.0, 100.0))
    assert_array_almost_equal(exit_xyz, (75.0, -75.0, 150.0))
    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (1, 1, 1))
    row = source_df.iloc[1]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (1, 1, 1, 0)
    assert_array_almost_equal(entry_xyz, (75.0, -100.0, 175.0))
    assert_array_almost_equal(exit_xyz, (75.0, -50.0, 175.0))
    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (1, 2, 1))
    row = source_df.iloc[2]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (0, 0, 0, 1)
    assert_array_almost_equal(entry_xyz, (75.0, -125.0, 150.0))
    assert_array_almost_equal(exit_xyz, (75.0, -125.0, 200.0))
    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (1, 2, 2))
    row = source_df.iloc[3]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (2, 0, 2, 1)
    delta_z = 25.0 / maths.tan(maths.pi / 3.0)
    assert_array_almost_equal(entry_xyz, (100.0, -125.0, 175.0 - delta_z))
    assert_array_almost_equal(exit_xyz, (150.0, -125.0, 175.0 + delta_z))


def test_calculate_exit_and_entry_mixed_units(example_model_and_mixed_units_crs):
    # has xy units of 'm'; z units 'ft'

    # --------- Arrange ----------
    model, crs = example_model_and_mixed_units_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    assert grid.crs.xy_units == 'm' and grid.crs.z_units == 'ft'
    well_name = 'DOGLEG'
    # todo: check sign of ANGLA relative to I axis
    # yapf: disable
    source_df = pd.DataFrame(
                  [[2,    2,    1,    0.0,     0.0,     0.0,   0.25,   0.9,    'grid_1'],
                   [2,    2,    2,   90.0,    90.0,     2.5,   0.25,   0.9,    'grid_1'],
                   [2,    3,    2,    0.0,     0.0,     1.0,   0.20,   0.9,    'grid_1'],
                   [2,    3,    3,   60.0,     0.0,    -0.5,   0.20,   0.9,    'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    # yapf: enable
    bw = rqw.BlockedWell(model, well_name = well_name)

    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (0, 1, 1))
    row = source_df.iloc[0]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (0, 0, 0, 1)
    assert_array_almost_equal(entry_xyz, (75.0, -75.0, 100.0))
    assert_array_almost_equal(exit_xyz, (75.0, -75.0, 150.0))
    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (1, 1, 1))
    row = source_df.iloc[1]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (1, 1, 1, 0)
    assert_array_almost_equal(entry_xyz, (75.0, -100.0, 175.0))
    assert_array_almost_equal(exit_xyz, (75.0, -50.0, 175.0))
    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (1, 2, 1))
    row = source_df.iloc[2]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (0, 0, 0, 1)
    assert_array_almost_equal(entry_xyz, (75.0, -125.0, 150.0))
    assert_array_almost_equal(exit_xyz, (75.0, -125.0, 200.0))
    # --------- Act ----------
    cp = grid.corner_points(cell_kji0 = (1, 2, 2))
    row = source_df.iloc[3]
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =  \
        bw._BlockedWell__calculate_entry_and_exit_axes_polarities_and_points_using_angles\
        (row = row, cp = cp, well_name = well_name, xy_units = grid.crs.xy_units, z_units = grid.crs.z_units)
    log.debug((entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz))
    # --------- Assert ----------
    # assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (2, 0, 2, 1)
    assert (entry_axis, entry_polarity, exit_axis, exit_polarity) == (0, 0, 0, 1)
    delta_x = 25.0 * 0.3048 * maths.tan(maths.pi / 3.0)
    assert_array_almost_equal(entry_xyz, (125.0 - delta_x, -125.0, 150.0))
    assert_array_almost_equal(exit_xyz, (125.0 + delta_x, -125.0, 200.0))


def test_calculate_cell_cp_center_and_vectors(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    cp_expected = grid.corner_points(cell_kji0 = (2, 2, 1))
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'DOGLEG'
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9, 'grid_1'], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9, 'grid_1'],
         [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9, 'grid_1'], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9, 'grid_1']],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF', 'GRID'])
    bw = rqw.BlockedWell(model, well_name = well_name, use_face_centres = True)

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
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)

    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
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
    bw = rqw.BlockedWell(model, use_face_centres = True)
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
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
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
    # yapf: disable
    source_df = pd.DataFrame(
        [[2, 2, 1, 0.0, 0.0, 0.0, 0.25],
         [2, 2, 2, 45, -90.0, 2.5, 0.25],
         [2, 3, 2, 45, -90.0, 1.0, 0.20],
         [2, 3, 3, 45, -90.0, -0.5, 0.20],
         [2, 3, 4, 45, -90.0, 1.1, 0.20],
         [2, 3, 5, 0.0, 0.0, 1.0, 0.20]],
        columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW'])
    # yapf: enable
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
                elif col == 'STAT':
                    fp.write(f' {row[col]:4}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = rqw.BlockedWell(model,
                         wellspec_file = wellspec_file,
                         well_name = well_name,
                         use_face_centres = True,
                         add_wellspec_properties = True)

    # --------- Act ----------
    df = bw.dataframe(extra_columns_list = ['X', 'Y', 'DEPTH', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'KH', 'WI', 'WBC'],
                      add_as_properties = True,
                      perm_i_uuid = perm_i_uuid,
                      perm_j_uuid = perm_i_uuid,
                      ntg_uuid = ntg_uuid,
                      stat = status,
                      use_face_centres = True,
                      length_uom = 'm',
                      length_mode = length_mode)
    # switch off adding properties when creating filtered dataframe
    df = bw.dataframe(extra_columns_list = ['X', 'Y', 'DEPTH', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'KH', 'WI', 'WBC'],
                      add_as_properties = False,
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
    # log.debug(df)
    assert all(df['KH'])  # successfully added a KH column as an i-direction permeability array was specified
    assert set(df['STAT']) == {status}
    assert all(df['WI'])
    assert all(df['WBC'])


def test_dataframe_from_trajectory(example_model_and_crs):
    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 4, 4),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           as_irregular_grid = True)

    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    elevation = 100.0

    # Create a measured depth datum
    location = (0.0, 0.0, -elevation)
    datum = rqw.MdDatum(parent_model = model, crs_uuid = crs.uuid, location = location, md_reference = 'kelly bushing')
    mds = np.array([0.0, 100, 210, 230, 240, 250])
    zs = mds * 1.03 - elevation
    well_name = 'CoconutDrop'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [location[0], 25.0, 50, 75, 100, 100],
        'Y': [location[1], 25.0, -50, -75, -100, -100],
        'Z': zs,
        'WELL': [well_name, well_name, well_name, well_name, well_name, well_name]
    })

    # Create a trajectory from dataframe
    trajectory = rqw.Trajectory(parent_model = model,
                                data_frame = source_dataframe,
                                well_name = well_name,
                                md_datum = datum,
                                length_uom = 'm')
    trajectory.write_hdf5()
    trajectory.create_xml()
    bw = rqw.BlockedWell(model, well_name = well_name, grid = grid, trajectory = trajectory, use_face_centres = True)

    # --------- Act ----------
    df = bw.dataframe(extra_columns_list = ['X', 'Y', 'DEPTH', 'RADW'], stat = 'ON', length_uom = 'ft')

    # --------- Assert ----------
    assert bw.trajectory is not None
    assert set(df['RADW']) == {0.33}
    assert set(df['STAT']) == {'ON'}
    assert all(df['X']) == all(df['Y']) == all(df['DEPTH'])  # successfully got xyz points from trajectory and
    # converted them to ft


def test_write_wellspec(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
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
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 'ON'], [2, 2, 2, 45, -90.0, 2.5, 0.25, 'OFF'],
                              [2, 3, 2, 45, -90.0, 1.0, 0.20, 'OFF'], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 'ON']],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'STAT'])
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
                elif col == 'STAT':
                    fp.write(f' {row[col]:4}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = rqw.BlockedWell(model,
                         wellspec_file = wellspec_file,
                         well_name = well_name,
                         use_face_centres = True,
                         add_wellspec_properties = True)

    # --------- Act ----------
    df = bw.dataframe(extra_columns_list = ['ANGLV', 'ANGLA', 'SKIN', 'STAT', 'RADW', 'KH'],
                      add_as_properties = True,
                      perm_i_uuid = perm_uuid)

    wellspec_file2 = os.path.join(model.epc_directory, 'wellspec2.dat')
    df2 = bw.write_wellspec(wellspec_file = wellspec_file2,
                            well_name = well_name,
                            extra_columns_list = ['ANGLV', 'ANGLA', 'SKIN', 'STAT', 'RADW'],
                            length_uom = 'm',
                            length_uom_comment = '?')

    # --------- Assert ----------
    pd.testing.assert_frame_equal(df[['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'STAT', 'RADW']],
                                  df2,
                                  check_dtype = False)
    # TODO find out why initially when ANGLV was 0.45, the Blocked Well dataframe method changed the values to 45
    # TODO find out why AngleA values of 0 transformed to nan values?


def test_convenience_methods_xyz_and_kji0_marker(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           as_irregular_grid = True)
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
                elif col == 'STAT':
                    fp.write(f' {row[col]:4}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')
    bw = rqw.BlockedWell(model,
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


def test_add_df_properties(tmp_path, tmp_model):
    # Arrange
    df = pd.DataFrame({
        "IW": [18, 18, 18, 18],
        "JW": [28, 28, 28, 28],
        "L": [2, 3, 4, 5],
        "KH": [np.nan, np.nan, np.nan, np.nan],
        "RADW": [0.32, 0.32, 0.32, 0.32],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "RADB": [np.nan, np.nan, np.nan, np.nan],
        "WI": [np.nan, np.nan, np.nan, np.nan],
        "STAT": [True, True, True, True],
        "LENGTH": [5.0, 5.0, 5.0, 5.0],
        "ANGLV": [88.08, 88.08, 88.08, 88.08],
        "ANGLA": [86.8, 86.8, 86.8, 86.8],
        "DEPTH": [9165.28, 9165.28, 9165.28, 9165.28],
    })
    columns = df[3:]
    length_uom = 'm'

    model = tmp_model
    time_series = rqts.TimeSeries(model)
    time_series.timestamps = ["2004-04-11"]
    time_series.create_xml()
    time_index = 0
    time_series_uuid = time_series.uuid

    wellspec_file = f"{tmp_path}/test.dat"
    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
            """)

    grid = grr.RegularGrid(model, extent_kji = (50, 50, 50))
    grid.create_xml()

    bw = BlockedWell(
        model,
        wellspec_file = wellspec_file,
        well_name = "TEST_WELL",
        use_face_centres = True,
        add_wellspec_properties = False,
    )

    # Act
    bw.add_df_properties(df, columns, length_uom, time_index, time_series_uuid)

    # Assert
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 1
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 15


def test_add_grid_properties(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 3, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)

    elevation = 100.0
    location = (0.0, 0.0, -elevation)
    datum = rqw.MdDatum(parent_model = model, crs_uuid = crs.uuid, location = location, md_reference = 'kelly bushing')
    mds = np.array([0.0, 100, 210, 230, 240, 250])
    zs = mds * 1.03 - elevation
    well_name = 'TestWell'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [location[0], 25.0, 50, 75, 100, 100],
        'Y': [location[1], 25.0, -50, -75, -100, -100],
        'Z': zs,
        'WELL': [well_name, well_name, well_name, well_name, well_name, well_name]
    })
    # Create a trajectory from dataframe
    trajectory = rqw.Trajectory(parent_model = model,
                                data_frame = source_dataframe,
                                well_name = well_name,
                                md_datum = datum,
                                length_uom = 'm')
    trajectory.write_hdf5()
    trajectory.create_xml()

    ts = rqts.TimeSeries(model, first_timestamp = '1997-03-03', yearly = 2, title = 'an example time series')
    ts.create_xml()
    slup1 = rqp.StringLookup(model, title = 'example lookup 1', int_to_str_dict = {1: 'one', 2: 'two', 3: 'three'})
    slup1.create_xml()
    slup2 = rqp.StringLookup(model, title = 'example lookup 2', int_to_str_dict = {4: 'four', 5: 'five', 6: 'six'})
    slup2.create_xml()

    grid_pc = grid.property_collection
    for ti in range(3):
        grid_pc.add_cached_array_to_imported_list(np.full(shape = grid.extent_kji, fill_value = ti),
                                                  'unit test',
                                                  'time step data',
                                                  discrete = True,
                                                  property_kind = 'example data',
                                                  time_index = ti)
    grid_pc.write_hdf5_for_imported_list()
    uuids_nosl = grid_pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts.uuid)

    array = np.random.randint(1, 3, size = (3, 5, 3, 3))

    for ti in range(3):
        grid_pc.add_cached_array_to_imported_list(array[ti],
                                                  'unit test',
                                                  'time step and sl data',
                                                  discrete = True,
                                                  property_kind = 'example data',
                                                  time_index = ti)
    grid_pc.write_hdf5_for_imported_list()
    uuids_sl = grid_pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts.uuid,
                                                                           string_lookup_uuid = slup1.uuid)
    array2 = np.random.randint(4, 6, size = (5, 3, 3))
    grid_pc.add_cached_array_to_imported_list(array2,
                                              'unit test',
                                              'sl data',
                                              discrete = True,
                                              property_kind = 'example data static')
    grid_pc.write_hdf5_for_imported_list()
    uuids_static = grid_pc.create_xml_for_imported_list_and_add_parts_to_model(string_lookup_uuid = slup2.uuid)

    array3 = np.random.rand(5, 3, 3)
    grid_pc.add_cached_array_to_imported_list(array3,
                                              'unit test',
                                              'continuous array',
                                              discrete = False,
                                              property_kind = 'example data continuous')
    grid_pc.write_hdf5_for_imported_list()
    uuids_static_continuous = grid_pc.create_xml_for_imported_list_and_add_parts_to_model()

    well_name = 'VERTICAL'
    bw = rqw.BlockedWell(model, well_name = well_name, trajectory = trajectory)
    bw.write_hdf5()
    bw.create_xml()
    model.store_epc()

    uuids_to_add = uuids_nosl + uuids_sl + uuids_static + uuids_static_continuous
    orig_counts_dict = model.parts_count_dict()

    # ---------- Act ------------
    bw.add_grid_property_to_blocked_well(uuids_to_add)
    model.store_epc()

    # --------- Assert ----------
    reload = rq.Model(model.epc_file)
    new_counts_dict = reload.parts_count_dict()

    assert new_counts_dict['CategoricalProperty'] - orig_counts_dict['CategoricalProperty'] == 4
    assert new_counts_dict['DiscreteProperty'] - orig_counts_dict['DiscreteProperty'] == 3
    assert new_counts_dict['ContinuousProperty'] - orig_counts_dict['ContinuousProperty'] == 1

    bw = rqw.BlockedWell(reload, uuid = bw.uuid)
    bw_pc = bw.extract_property_collection()
    assert len(bw_pc.time_series_uuid_list()) == 1
    assert len(bw_pc.string_lookup_uuid_list()) == 2

    assert bw_pc.single_array_ref(property_kind = 'example data continuous').dtype == float
    assert bw_pc.single_array_ref(property_kind = 'example data static').dtype == int


def test_temporary_handling_of_badly_formed_grid_indices(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (5, 3, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    well_name = 'VERTICAL'
    bw = rqw.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)
    bw.set_for_column(well_name = well_name, grid = grid, col_ji0 = (1, 1))

    # --------- Act ----------
    # corrupt the nodes to mimic bad data (non-resqpy) and write
    assert bw.node_count == bw.cell_count + 2  # one extra top node in good test data
    assert bw.grid_indices.size == bw.node_count - 1
    node_mds = np.zeros(2 * bw.cell_count, dtype = float)
    node_mds[::2] = bw.node_mds[1:-1]  # drop the first node
    node_mds[1::2] = bw.node_mds[2:]  # duplicate all the internal nodes
    bw.grid_indices = bw.grid_indices[1:]  # drop the first interval from the grid indices
    bw.node_mds = node_mds
    bw.node_count = 2 * bw.cell_count
    bw.write_hdf5()
    bw.create_xml()
    model.store_epc()

    # --------- Assert ----------
    reload = rq.Model(model.epc_file)
    bw2 = rqw.BlockedWell(reload, uuid = bw.uuid)
    assert bw2 is not None
    assert bw2.node_count == 2 * bw.cell_count
    assert bw2.grid_indices.size == bw2.node_count - 1
    assert bw2.cell_count == bw.cell_count
    assert np.all(bw2.grid_indices[1::2] == -1)
