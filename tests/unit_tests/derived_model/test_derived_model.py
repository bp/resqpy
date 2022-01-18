import os

import math as maths
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.crs as rqc
import resqpy.derived_model as rqdm
import resqpy.fault as rqf
import resqpy.grid as grr
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.olio.box_utilities as bx
import resqpy.olio.fine_coarse as rqfc
import resqpy.olio.uuid as bu
import resqpy.property as rqp
import resqpy.surface as rqs

import resqpy.well as rqw

from resqpy.olio.random_seed import seed

seed(2349857)


def test_add_single_cell_grid(tmp_path):

    epc = os.path.join(tmp_path, 'amoeba.epc')

    points = np.array([(100.0, 250.0, -3500.0), (140.0, 200.0, -3700.0), (300.0, 400.0, -3600.0),
                       (180.0, 300.0, -3800.0), (220.0, 350.0, -3750.0)])
    expected_xyz_box = np.array([(100.0, 200.0, -3800.0), (300.0, 400.0, -3500.0)])

    # create a single cell grid containing points
    rqdm.add_single_cell_grid(points, new_grid_title = 'Amoeba', new_epc_file = epc)

    # re-open model and have a quick look at the grid
    model = rq.Model(epc)
    assert model is not None
    grid = grr.Grid(model, uuid = model.uuid(title = 'Amoeba'))
    assert grid is not None
    assert tuple(grid.extent_kji) == (1, 1, 1)
    assert_array_almost_equal(grid.xyz_box(lazy = False), expected_xyz_box)


def test_add_zone_by_layer_property(tmp_path):

    def check_zone_prop(z_prop):
        assert z_prop is not None
        assert not z_prop.is_continuous()
        assert not z_prop.is_points()
        assert z_prop.indexable_element() == 'layers'
        lpk_uuid = z_prop.local_property_kind_uuid()
        assert lpk_uuid is not None
        lpk = rqp.PropertyKind(z_prop.model, uuid = lpk_uuid)
        assert lpk.title == 'zone'

    epc = os.path.join(tmp_path, 'in the zone.epc')

    model = rq.new_model(epc)

    # create a basic block grid
    grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'In The Zone', set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    grid_uuid = grid.uuid

    model.store_epc()

    # add zone property based on an explicit vector (one value per layer)
    zone_vector = (2, 7, 5, 7)
    v, z_uuid = rqdm.add_zone_by_layer_property(epc_file = epc,
                                                zone_by_layer_vector = (2, 7, 5, 7),
                                                title = 'from vector')
    assert tuple(v) == zone_vector

    # check that zone property looks okay
    model = rq.Model(epc)
    z_prop = rqp.Property(model, uuid = z_uuid)
    check_zone_prop(z_prop)

    # add a neatly set up grid cells property
    za = np.array((1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5),
                  dtype = int).reshape(grid.extent_kji)
    za_uuid = rqdm.add_one_grid_property_array(epc,
                                               za,
                                               discrete = True,
                                               property_kind = 'code',
                                               title = 'clean zone',
                                               grid_uuid = grid_uuid,
                                               null_value = -1)
    assert za_uuid is not None

    # add a zone by layer property based on the neat cells property
    v, z_uuid = rqdm.add_zone_by_layer_property(epc_file = epc,
                                                zone_by_cell_property_uuid = za_uuid,
                                                title = 'from cells array')
    assert tuple(v) == (1, 2, 3, 5)

    # check that zone property looks okay
    model = rq.Model(epc)
    z_prop = rqp.Property(model, uuid = z_uuid)
    check_zone_prop(z_prop)

    # make the cells array less tidy and add another copy
    za[1, 2, :] = 3
    za_uuid = rqdm.add_one_grid_property_array(epc,
                                               za,
                                               discrete = True,
                                               property_kind = 'code',
                                               title = 'messy zone',
                                               grid_uuid = grid_uuid,
                                               null_value = -1)
    assert za_uuid is not None

    # fail to add a zone by layer property based on the messy cells property
    with pytest.raises(Exception):
        v, z2_uuid = rqdm.add_zone_by_layer_property(epc_file = epc,
                                                     zone_by_cell_property_uuid = za_uuid,
                                                     use_dominant_zone = False,
                                                     title = 'should fail')

    # add a zone by layer property based on the neat cells property
    v, z3_uuid = rqdm.add_zone_by_layer_property(epc_file = epc,
                                                 zone_by_cell_property_uuid = za_uuid,
                                                 use_dominant_zone = True,
                                                 title = 'from messy cells array')
    assert tuple(v) == (1, 2, 3, 5)

    # check that zone property looks okay
    model = rq.Model(epc)
    grid = model.grid()
    z_prop = rqp.Property(model, uuid = z3_uuid)
    check_zone_prop(z_prop)

    # create a zonal grid based on a (neat) zone property array
    z_grid = rqdm.zonal_grid(epc,
                             source_grid = grid,
                             zone_uuid = z_uuid,
                             use_dominant_zone = False,
                             inactive_laissez_faire = True,
                             new_grid_title = 'zonal grid')
    assert z_grid is not None
    assert z_grid.nk == 4

    # and another zonal grid based on the dominant zone
    z3_grid = rqdm.zonal_grid(epc,
                              source_grid = grid,
                              zone_uuid = za_uuid,
                              use_dominant_zone = True,
                              new_grid_title = 'dominant zone grid')
    assert z3_grid is not None
    assert z3_grid.nk == 4


def test_single_layer_grid(tmp_path):

    epc = os.path.join(tmp_path, 'squash.epc')

    model = rq.new_model(epc)

    # create a basic block grid with geometry
    grid = grr.RegularGrid(model,
                           extent_kji = (4, 3, 2),
                           origin = (1000.0, 2000.0, 3000.0),
                           dxyz = (100.0, 130.0, 25.0),
                           title = 'to be squashed',
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    grid_uuid = grid.uuid
    model.store_epc()

    # create a single layer version of the grid
    simplified = rqdm.single_layer_grid(epc, source_grid = grid, new_grid_title = 'squashed')
    assert simplified is not None
    simplified_uuid = simplified.uuid

    # re-open the model and load the new grid
    model = rq.Model(epc)
    s_uuid = model.uuid(obj_type = 'IjkGridRepresentation', title = 'squashed')
    assert bu.matching_uuids(s_uuid, simplified_uuid)
    simplified = grr.any_grid(model, uuid = s_uuid)
    assert simplified.nk == 1
    simplified.cache_all_geometry_arrays()
    assert not simplified.has_split_coordinate_lines
    assert simplified.points_cached.shape == (2, 4, 3, 3)
    assert_array_almost_equal(simplified.points_cached[0, ..., 2], np.full((4, 3), 3000.0))
    assert_array_almost_equal(simplified.points_cached[1, ..., 2], np.full((4, 3), 3100.0))


def test_extract_box_for_well(tmp_path):

    epc = os.path.join(tmp_path, 'tube.epc')

    model = rq.new_model(epc)

    # create a basic block grid with geometry
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 5, 7),
                           origin = (0.0, 0.0, 1000.0),
                           dxyz = (100.0, 100.0, 20.0),
                           title = 'main grid',
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    grid_uuid = grid.uuid

    # create a couple of well trajectories
    cells_visited = [(0, 1, 2), (1, 1, 2), (1, 1, 3), (1, 2, 3), (1, 2, 4), (2, 2, 4)]
    traj_1 = rqw.Trajectory(model,
                            grid = grid,
                            cell_kji0_list = cells_visited,
                            length_uom = 'm',
                            spline_mode = 'linear',
                            well_name = 'well 1')
    traj_2 = rqw.Trajectory(model,
                            grid = grid,
                            cell_kji0_list = cells_visited,
                            length_uom = 'm',
                            spline_mode = 'cube',
                            well_name = 'well 2')
    for traj in (traj_1, traj_2):
        traj.write_hdf5()
        traj.create_xml()
    traj_1_uuid = traj_1.uuid
    traj_2_uuid = traj_2.uuid

    # create a blocked well for one of the trajectories
    assert traj_2.root is not None
    bw = rqw.BlockedWell(model, grid = grid, trajectory = traj_2)
    bw.write_hdf5()
    bw.create_xml()
    bw_uuid = bw.uuid

    # store source model
    model.store_epc()

    # extract box for linear trajectory
    grid_1, box_1 = rqdm.extract_box_for_well(epc_file = epc,
                                              source_grid = grid,
                                              trajectory_uuid = traj_1_uuid,
                                              radius = 120.0,
                                              active_cells_shape = 'tube',
                                              new_grid_title = 'grid 1')

    # check basics of resulting grid
    assert grid_1 is not None
    assert box_1 is not None
    assert tuple(grid_1.extent_kji) == tuple(bx.extent_of_box(box_1))
    expected_box = np.array([(0, 0, 1), (2, 3, 5)], dtype = int)
    assert np.all(box_1 == expected_box)
    #   expected_inactive_1 = np.array(
    #      [[[1, 0, 1, 1, 1], [0, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1]],
    #       [[1, 0, 0, 1, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 0, 0, 1]],
    #       [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 1]]], dtype = bool)   expected_inactive_1 = np.array(
    expected_inactive_1 = np.array([[[1, 0, 1, 1, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 0, 0, 1]],
                                    [[1, 0, 0, 1, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 0, 0, 1]],
                                    [[1, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 0, 1]]],
                                   dtype = bool)
    assert np.all(grid_1.inactive == expected_inactive_1)

    # extract box for blocked well made from splined trajectory
    grid_2, box_2 = rqdm.extract_box_for_well(epc_file = epc,
                                              source_grid = grid,
                                              blocked_well_uuid = bw_uuid,
                                              radius = 120.0,
                                              active_cells_shape = 'prism',
                                              new_grid_title = 'grid 2')
    assert grid_2 is not None
    assert box_2 is not None
    assert tuple(grid_2.extent_kji) == tuple(bx.extent_of_box(box_2))
    assert np.all(box_2 == expected_box)
    # active cells should be superset of those for linear trajectory tube box
    assert np.count_nonzero(grid_2.inactive) <= np.count_nonzero(grid_1.inactive)
    assert np.all(np.logical_not(grid_2.inactive[np.logical_not(expected_inactive_1)]))
    # check prism shape to inactive cells
    assert np.all(grid_2.inactive == grid_2.inactive[0])


def test_extract_box(tmp_path):
    # create an empty model and add a crs
    epc = os.path.join(tmp_path, 'box_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    model.store_epc()
    # create a grid
    grid = grr.RegularGrid(model,
                           crs_uuid = crs.uuid,
                           extent_kji = (5, 10, 12),
                           origin = (2000.0, 3000.0, 1000.0),
                           dxyz = (100.0, 100.0, 20.0),
                           title = 'original grid',
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml()
    # introduce a couple of faults
    fault_lines = []
    fl1 = np.array([(2600.0, 2900.0, 0.0), (2600.0, 4100.0, 0.0)])
    fault_lines.append(rql.Polyline(model, set_bool = False, set_coord = fl1, set_crs = crs.uuid, title = 'fault 1'))
    fl2 = np.array([(1900.0, 3500.0, 0.0), (3300.0, 3500.0, 0.0)])
    fault_lines.append(rql.Polyline(model, set_bool = False, set_coord = fl2, set_crs = crs.uuid, title = 'fault 2'))
    for fl in fault_lines:
        fl.write_hdf5()
        fl.create_xml()
    model.store_epc()
    fs_grid = rqdm.add_faults(epc, grid, polylines = fault_lines, new_grid_title = 'small throws')
    # scale the throw on the faults
    fb_grid = rqdm.global_fault_throw_scaling(epc,
                                              source_grid = fs_grid,
                                              scaling_factor = 30.0,
                                              cell_range = 2,
                                              new_grid_title = 'big throws')
    # extract a box
    box = np.array([(1, 2, 3), (3, 8, 9)], dtype = int)
    e_grid = rqdm.extract_box(epc, source_grid = fb_grid, box = box, new_grid_title = 'extracted grid')
    assert e_grid is not None
    # re-open model and check extent of extracted grid
    model = rq.Model(epc)
    grid = grr.Grid(model, uuid = e_grid.uuid)
    assert np.all(grid.extent_kji == (3, 7, 7))


def test_add_grid_points_property(tmp_path):

    epc = os.path.join(tmp_path, 'bland.epc')
    new_epc = os.path.join(tmp_path, 'pointy.epc')

    model = rq.new_model(epc)

    # create a basic block grid with geometry
    extent_kji = (3, 5, 2)
    grid = grr.RegularGrid(model,
                           extent_kji = extent_kji,
                           origin = (2000.0, 3000.0, 1000.0),
                           dxyz = (10.0, 10.0, 20.0),
                           title = 'the grid',
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False)
    grid_uuid = grid.uuid

    # store grid
    model.store_epc()

    # create a points property array
    diagonal = grid.axial_lengths_kji()
    diagonals_extent = tuple(list(extent_kji) + [3])
    diagonal_array = np.empty(diagonals_extent)
    diagonal_array[:] = np.array(diagonal).reshape(1, 1, 1, 3)

    # add to model using derived model function but save as new dataset
    rqdm.add_one_grid_property_array(epc_file = epc,
                                     a = diagonal_array,
                                     property_kind = 'length',
                                     grid_uuid = grid_uuid,
                                     source_info = 'test',
                                     title = 'diagonal vectors',
                                     discrete = False,
                                     uom = grid.xy_units(),
                                     points = True,
                                     extra_metadata = {'test': 'true'},
                                     new_epc_file = new_epc)

    # re-open the original model and check that the points property is not there
    model = rq.Model(epc)
    grid = model.grid()
    pc = grid.property_collection
    assert pc is not None
    assert len(pc.selective_parts_list(points = True)) == 0

    # re-open the new model and load the points property
    model = rq.Model(new_epc)
    grid = model.grid()
    pc = grid.property_collection
    assert pc is not None
    assert len(pc.selective_parts_list(points = True)) == 1
    diag = pc.single_array_ref(points = True)
    assert_array_almost_equal(diag, diagonal_array)


def test_add_edges_per_column_property_array(tmp_path):

    # create a new model with a grid
    epc = os.path.join(tmp_path, 'edges_per_column.epc')
    model = rq.new_model(epc)
    grid = grr.RegularGrid(model, extent_kji = (2, 3, 4))
    grid.write_hdf5()
    grid.create_xml()
    model.store_epc()

    # fabricate an edges per column property
    edge_prop = np.zeros((grid.nj, grid.ni, 2, 2))
    edge_prop[:] = np.linspace(0.1, 0.9, num = edge_prop.size).reshape(edge_prop.shape)

    # add the edges per column property
    prop_uuid = rqdm.add_edges_per_column_property_array(epc,
                                                         edge_prop,
                                                         property_kind = 'multiplier',
                                                         grid_uuid = grid.uuid,
                                                         source_info = 'unit testing',
                                                         title = 'test property on column edges',
                                                         discrete = False,
                                                         uom = 'm3/m3')
    assert prop_uuid is not None

    # re-open the model and inspect the property
    model = rq.Model(epc)
    assert len(model.parts(obj_type = 'ContinuousProperty')) > 0
    edge_property = rqp.Property(model, uuid = prop_uuid)
    assert edge_property is not None
    ep_array = edge_property.array_ref()
    # RESQML holds array with last two dimensions flattened and reordered
    assert ep_array.shape == (grid.nj, grid.ni, 4)
    # restore logical resqpy order and shape
    ep_restored = rqp.reformat_column_edges_from_resqml_format(ep_array)
    assert_array_almost_equal(ep_restored, edge_prop)
    assert edge_property.is_continuous()
    assert not edge_property.is_categorical()
    assert edge_property.indexable_element() == 'edges per column'
    assert edge_property.uom() == 'm3/m3'
    assert edge_property.property_kind() == 'multiplier'
    assert edge_property.facet() is None


def test_add_one_blocked_well_property(example_model_with_well):
    model, well_interp, datum, traj = example_model_with_well
    epc = model.epc_file
    # make a grid positioned in the area of the well
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           origin = (-150.0, -150.0, 1500.0),
                           extent_kji = (5, 3, 3),
                           dxyz = (100.0, 100.0, 50.0),
                           set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False)
    # create a blocked well
    bw = rqw.BlockedWell(model, grid = grid, trajectory = traj)
    bw.write_hdf5()
    bw.create_xml()
    model.store_epc()
    assert bw is not None
    assert bw.cell_count == grid.nk
    # fabricate a blocked well property holding the depths of the centres of penetrated cells
    wb_prop = np.zeros(grid.nk,)
    cells_kji = bw.cell_indices_kji0()
    for i, cell_kji in enumerate(cells_kji):
        wb_prop[i] = grid.centre_point(cell_kji)[2]
    # add the property
    p_uuid = rqdm.add_one_blocked_well_property(epc,
                                                wb_prop,
                                                'depth',
                                                bw.uuid,
                                                source_info = 'unit test',
                                                title = 'DEPTH',
                                                discrete = False,
                                                uom = 'm',
                                                indexable_element = 'cells')
    assert p_uuid is not None
    # re-open the model and check the wellbore property
    model = rq.Model(epc)
    prop = rqp.Property(model, uuid = p_uuid)
    assert prop is not None
    assert bu.matching_uuids(model.supporting_representation_for_part(model.part(uuid = prop.uuid)), bw.uuid)
    assert prop.title == 'DEPTH'
    assert prop.property_kind() == 'depth'
    assert prop.uom() == 'm'
    assert prop.is_continuous()
    assert not prop.is_categorical()
    assert prop.minimum_value() is not None and prop.maximum_value() is not None
    assert prop.minimum_value() > grid.xyz_box(lazy = True)[0, 2]  # minimum z
    assert prop.maximum_value() < grid.xyz_box(lazy = True)[1, 2]  # maximum z


def test_add_wells_from_ascii_file(tmp_path):
    # create an empty model and add a crs
    epc = os.path.join(tmp_path, 'well_model.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    model.store_epc()
    # fabricate some test data as an ascii table
    well_file = os.path.join(tmp_path, 'well_table.txt')
    df = pd.DataFrame(columns = ['WELL', 'MD', 'X', 'Y', 'Z'])
    well_count = 3
    for wi in range(well_count):
        well_name = 'Hole_' + str(wi + 1)
        wdf = pd.DataFrame(columns = ['WELL', 'MD', 'X', 'Y', 'Z'])
        row_count = 3 + wi
        wdf['MD'] = np.linspace(0.0, 1000.0, num = row_count)
        wdf['X'] = np.linspace(100.0 * wi, 100.0 * wi + 10.0 * row_count, num = row_count)
        wdf['Y'] = np.linspace(500.0 * wi, 500.0 * wi - 15.0 * row_count, num = row_count)
        wdf['Z'] = np.linspace(0.0, 1000.0 + 5.0 * row_count, num = row_count)
        wdf['WELL'] = well_name
        df = df.append(wdf)
    df.to_csv(well_file, sep = ' ', index = False)
    # call the derived model function to add the wells
    added = rqdm.add_wells_from_ascii_file(epc,
                                           crs.uuid,
                                           well_file,
                                           space_separated_instead_of_csv = True,
                                           length_uom = 'ft',
                                           md_domain = ['driller', 'logger'][wi % 2],
                                           drilled = True,
                                           z_inc_down = True)
    assert added == well_count
    # re-open the model and check that all expected objects have appeared
    model = rq.Model(epc)
    assert len(model.parts(obj_type = 'WellboreTrajectoryRepresentation')) == well_count
    assert len(model.parts(obj_type = 'MdDatum')) == well_count
    assert len(model.parts(obj_type = 'WellboreInterpretation')) == well_count
    assert len(model.parts(obj_type = 'WellboreFeature')) == well_count
    for wi in range(well_count):
        well_name = 'Hole_' + str(wi + 1)
        traj = rqw.Trajectory(model,
                              uuid = model.uuid(obj_type = 'WellboreTrajectoryRepresentation', title = well_name))
        assert traj is not None
        assert traj.knot_count == 3 + wi


def test_interpolated_grid(tmp_path):
    # create an empty model and add a crs
    epc = os.path.join(tmp_path, 'interpolation.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    # create a pair of grids to act as boundary case geometries
    grid0 = grr.RegularGrid(model,
                            crs_uuid = model.crs_uuid,
                            origin = (0.0, 0.0, 1000.0),
                            extent_kji = (5, 4, 3),
                            dxyz = (100.0, 150.0, 50.0),
                            set_points_cached = True)
    grid0.grid_representation = 'IjkGrid'  # overwrite block grid setting
    grid0.write_hdf5()
    grid0.create_xml(write_geometry = True, add_cell_length_properties = False)
    grid1 = grr.RegularGrid(model,
                            crs_uuid = model.crs_uuid,
                            origin = (15.0, 35.0, 1030.0),
                            extent_kji = (5, 4, 3),
                            dxyz = (97.0, 145.0, 47.0),
                            set_points_cached = True)
    grid1.grid_representation = 'IjkGrid'  # overwrite block grid setting
    grid1.write_hdf5()
    grid1.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()
    # interpolate between the two grids
    between_grid_uuids = []
    for f in [0.0, 0.23, 0.5, 1.0]:
        grid = rqdm.interpolated_grid(epc,
                                      grid0,
                                      grid1,
                                      a_to_b_0_to_1 = f,
                                      split_tolerance = 0.01,
                                      inherit_properties = False,
                                      inherit_realization = None,
                                      inherit_all_realizations = False,
                                      new_grid_title = 'between_' + str(f))
        assert grid is not None
        between_grid_uuids.append(grid.uuid)
    # re-open model and check interpolated grid geometries
    model = rq.Model(epc)
    for i, g_uuid in enumerate(between_grid_uuids):
        grid = grr.Grid(model, uuid = g_uuid)
        assert grid is not None
        grid.cache_all_geometry_arrays()
        assert hasattr(grid, 'points_cached') and grid.points_cached is not None
        assert np.all(grid.points_cached >= grid0.points_cached)
        assert np.all(grid.points_cached <= grid1.points_cached)
        if i == 0:
            assert_array_almost_equal(grid.points_cached, grid0.points_cached)
        elif i == len(between_grid_uuids) - 1:
            assert_array_almost_equal(grid.points_cached, grid1.points_cached)
        else:
            assert not np.any(np.isclose(grid.points_cached, grid0.points_cached))
            assert not np.any(np.isclose(grid.points_cached, grid1.points_cached))


def test_interpolated_faulted_grid(tmp_path):
    # create an empty model and add a crs
    epc = os.path.join(tmp_path, 'faulted_interpolation.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    # create a regular grid
    grid0 = grr.RegularGrid(model,
                            crs_uuid = model.crs_uuid,
                            origin = (0.0, 0.0, 1000.0),
                            extent_kji = (5, 4, 3),
                            dxyz = (100.0, 150.0, 50.0),
                            as_irregular_grid = True)
    grid0.grid_representation = 'IjkGrid'  # overwrite block grid setting
    grid0.write_hdf5()
    grid0.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()
    # prepare fault data and add faults to grid with two different throw sets
    pl_dict = {}
    pl_dict['f1'] = [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3)]
    pl_dict['f2'] = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    lr_dict = {}
    lr_dict['f1'] = (23.0, -23.0)
    lr_dict['f2'] = (-37.3, 14.8)
    grid_a = rqdm.add_faults(epc,
                             source_grid = grid0,
                             full_pillar_list_dict = pl_dict,
                             left_right_throw_dict = lr_dict,
                             new_grid_title = 'grid a')
    lr_dict = {}
    lr_dict['f1'] = (-23.0, 23.0)
    lr_dict['f2'] = (17.2, -22.9)
    grid_b = rqdm.add_faults(epc,
                             source_grid = grid0,
                             full_pillar_list_dict = pl_dict,
                             left_right_throw_dict = lr_dict,
                             new_grid_title = 'grid b')
    # interpolate between the two grids
    between_grid_uuids = []
    for f in [0.0, 0.23, 0.5, 1.0]:
        grid = rqdm.interpolated_grid(epc,
                                      grid_a,
                                      grid_b,
                                      a_to_b_0_to_1 = f,
                                      split_tolerance = 0.01,
                                      inherit_properties = False,
                                      inherit_realization = None,
                                      inherit_all_realizations = False,
                                      new_grid_title = 'between_' + str(f))
        assert grid is not None
        between_grid_uuids.append(grid.uuid)
    # re-open model and check end point interpolated grid geometries
    model = rq.Model(epc)
    for i, g_uuid in enumerate(between_grid_uuids):
        grid = grr.Grid(model, uuid = g_uuid)
        assert grid is not None
        grid.cache_all_geometry_arrays()
        assert hasattr(grid, 'points_cached') and grid.points_cached is not None
        if i == 0:
            assert_array_almost_equal(grid.points_cached, grid_a.points_cached)
        elif i == len(between_grid_uuids) - 1:
            assert_array_almost_equal(grid.points_cached, grid_b.points_cached)


def test_interpolated_grid_using_cp(tmp_path):
    # create an empty model and add a crs
    epc = os.path.join(tmp_path, 'cp_interpolation.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    # create a regular grid
    grid0 = grr.RegularGrid(model,
                            crs_uuid = model.crs_uuid,
                            origin = (0.0, 0.0, 1000.0),
                            extent_kji = (5, 4, 3),
                            dxyz = (100.0, 150.0, 50.0),
                            as_irregular_grid = True)
    grid0.grid_representation = 'IjkGrid'  # overwrite block grid setting
    grid0.write_hdf5()
    grid0.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()
    # prepare fault data and add faults to a copy of the grid
    pl_dict = {}
    pl_dict['f1'] = [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3)]
    pl_dict['f2'] = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    lr_dict = {}
    lr_dict['f1'] = (23.0, -23.0)
    lr_dict['f2'] = (-37.3, 14.8)
    grid_a = rqdm.add_faults(epc,
                             source_grid = grid0,
                             full_pillar_list_dict = pl_dict,
                             left_right_throw_dict = lr_dict,
                             new_grid_title = 'grid a')
    # interpolate between the two grids
    between_grid_uuids = []
    for f in [0.0, 0.23, 0.5, 1.0]:
        grid = rqdm.interpolated_grid(epc,
                                      grid0,
                                      grid_a,
                                      a_to_b_0_to_1 = f,
                                      split_tolerance = 0.01,
                                      inherit_properties = False,
                                      inherit_realization = None,
                                      inherit_all_realizations = False,
                                      new_grid_title = 'between_' + str(f))
        assert grid is not None
        between_grid_uuids.append(grid.uuid)
    # re-open model and check end point interpolated grid geometries
    model = rq.Model(epc)
    for i, g_uuid in enumerate(between_grid_uuids):
        grid = grr.Grid(model, uuid = g_uuid)
        assert grid is not None
        grid.cache_all_geometry_arrays()
        assert hasattr(grid, 'points_cached') and grid.points_cached is not None
        if i == 0:
            assert_array_almost_equal(grid.corner_points(), grid0.corner_points(), decimal = 3)
        elif i == len(between_grid_uuids) - 1:
            assert_array_almost_equal(grid.corner_points(), grid_a.corner_points(), decimal = 3)


def test_refined_and_coarsened_grid(tmp_path):

    # create a model and a coarse grid
    epc = os.path.join(tmp_path, 'refinement.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()

    # create a coarse grid
    c_dxyz = (100.0, 150.0, 50.0)
    c_grid = grr.RegularGrid(model,
                             crs_uuid = model.crs_uuid,
                             extent_kji = (5, 3, 4),
                             dxyz = c_dxyz,
                             as_irregular_grid = True)
    c_grid.write_hdf5()
    c_grid.create_xml(write_geometry = True, add_cell_length_properties = True, expand_const_arrays = True)
    model.store_epc()

    # set up a coarse to fine mapping
    fine_extent = np.array(c_grid.extent_kji)
    fine_extent[0] *= 2
    fine_extent[1] *= 4
    fine_extent[2] *= 3
    fc = rqfc.FineCoarse(fine_extent, c_grid.extent_kji)
    fc.set_all_ratios_constant()

    # make the refinement
    f_grid = rqdm.refined_grid(epc,
                               source_grid = None,
                               fine_coarse = fc,
                               inherit_properties = True,
                               inherit_realization = None,
                               inherit_all_realizations = False,
                               source_grid_uuid = c_grid.uuid,
                               set_parent_window = None,
                               infill_missing_geometry = True,
                               new_grid_title = 'fine grid')
    assert f_grid is not None

    # re-open model and check some things about the refined grid
    model = rq.Model(epc)
    f_grid = grr.Grid(model, uuid = f_grid.uuid)
    assert tuple(f_grid.extent_kji) == tuple(fine_extent)
    f_grid.cache_all_geometry_arrays()
    assert_array_almost_equal(c_grid.xyz_box(lazy = False), f_grid.xyz_box(lazy = False))
    assert not f_grid.has_split_coordinate_lines
    assert f_grid.points_cached is not None
    assert f_grid.points_cached.shape == (c_grid.nk * 2 + 1, c_grid.nj * 4 + 1, c_grid.ni * 3 + 1, 3)
    # check that refined grid geometry is monotonic in each of the three axes
    p = f_grid.points_ref(masked = False)
    assert np.all(p[:-1, :, :, 2] < p[1:, :, :, 2])
    assert np.all(p[:, :-1, :, 1] < p[:, 1:, :, 1])
    assert np.all(p[:, :, :-1, 0] < p[:, :, 1:, 0])

    # check property inheritance of cell lengths
    pc = f_grid.extract_property_collection()
    assert pc is not None and pc.number_of_parts() >= 3
    lpc = rqp.selective_version_of_collection(pc, property_kind = 'cell length')
    assert lpc.number_of_parts() == 3
    for axis in range(3):
        length_array = lpc.single_array_ref(facet_type = 'direction', facet = 'KJI'[axis])
        assert length_array is not None
        assert np.allclose(length_array, c_dxyz[2 - axis] / (2, 4, 3)[axis])

    # make a (re-)coarsened version of the fine grid
    cfc_grid = rqdm.coarsened_grid(epc,
                                   f_grid,
                                   fc,
                                   inherit_properties = True,
                                   set_parent_window = None,
                                   infill_missing_geometry = True,
                                   new_grid_title = 're-coarsened grid')
    assert cfc_grid is not None

    # re-open the model and re-load the new grid
    model = rq.Model(epc)
    assert len(model.parts(obj_type = 'IjkGridRepresentation')) == 3
    cfc_grid = grr.Grid(model, uuid = cfc_grid.uuid)

    # compare the re-coarsened grid with the original coarse grid
    assert tuple(cfc_grid.extent_kji) == tuple(c_grid.extent_kji)
    assert not cfc_grid.has_split_coordinate_lines
    cfc_grid.cache_all_geometry_arrays()
    assert_array_almost_equal(c_grid.points_cached, cfc_grid.points_cached)

    # see how the cell length properties have fared
    cfc_pc = cfc_grid.extract_property_collection()
    assert cfc_grid is not None
    assert cfc_pc.number_of_parts() >= 3
    cfc_lpc = rqp.selective_version_of_collection(cfc_pc, property_kind = 'cell length')
    assert cfc_lpc.number_of_parts() == 3
    for axis in range(3):
        length_array = cfc_lpc.single_array_ref(facet_type = 'direction', facet = 'KJI'[axis])
        assert length_array is not None
        assert np.allclose(length_array, c_dxyz[2 - axis])


def test_add_faults(tmp_path):

    def write_poly(filename, a, mode = 'w'):
        nines = 999.0
        with open(filename, mode = mode) as fp:
            for row in range(len(a)):
                fp.write(f'{a[row, 0]:8.3f} {a[row, 1]:8.3f} {a[row, 2]:8.3f}\n')
            fp.write(f'{nines:8.3f} {nines:8.3f} {nines:8.3f}\n')

    def make_poly(model, a, title, crs):
        return [rql.Polyline(model, set_bool = False, set_coord = a, set_crs = crs.uuid, title = title)]

    epc = os.path.join(tmp_path, 'tic_tac_toe.epc')

    for test_mode in ['file', 'polyline']:

        model = rq.new_model(epc)
        grid = grr.RegularGrid(model, extent_kji = (1, 3, 3), set_points_cached = True)
        grid.write_hdf5()
        grid.create_xml(write_geometry = True)
        crs = rqc.Crs(model, uuid = grid.crs_uuid)
        model.store_epc()

        # single straight fault
        a = np.array([[-0.2, 2.0, -0.1], [3.2, 2.0, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f1.dat')
        if test_mode == 'file':
            write_poly(f, a)
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f1', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = False,
                            new_grid_title = 'ttt_f1 straight')

        # single zig-zag fault
        a = np.array([[-0.2, 1.0, -0.1], [1.0, 1.0, -0.1], [1.0, 2.0, -0.1], [3.2, 2.0, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f2.dat')
        if test_mode == 'file':
            write_poly(f, a)
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f2', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f2 zig_zag')

        # single zig-zag-zig fault
        a = np.array([[-0.2, 1.0, -0.1], [1.0, 1.0, -0.1], [1.0, 2.0, -0.1], [2.0, 2.0, -0.1], [2.0, 1.0, -0.1],
                      [3.2, 1.0, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f3.dat')
        if test_mode == 'file':
            write_poly(f, a)
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f3', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f3 zig_zag_zig')

        # horst block
        a = np.array([[-0.2, 1.0, -0.1], [3.2, 1.0, -0.1]])
        b = np.array([[3.2, 2.0, -0.1], [-0.2, 2.0, -0.1]])
        fa = os.path.join(tmp_path, 'ttt_f4a.dat')
        fb = os.path.join(tmp_path, 'ttt_f4b.dat')
        if test_mode == 'file':
            write_poly(fa, a)
            write_poly(fb, b)
            lines_file_list = [fa, fb]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f4a', crs) + make_poly(model, b, 'ttt_f4b', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f4 horst')

        # asymmetrical horst block
        lr_throw_dict = {'ttt_f4a': (0.0, -0.3), 'ttt_f4b': (0.0, -0.6)}
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            left_right_throw_dict = lr_throw_dict,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f5 horst')
        assert g is not None

        # scaled version of asymmetrical horst block
        model = rq.Model(epc)
        grid = model.grid(title = 'ttt_f5 horst')
        assert grid is not None
        gcs_uuids = model.uuids(obj_type = 'GridConnectionSetRepresentation', related_uuid = grid.uuid)
        assert gcs_uuids
        scaling_dict = {'ttt_f4a': 3.0, 'ttt_f4b': 1.7}
        for i, gcs_uuid in enumerate(gcs_uuids):
            gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
            assert gcs is not None
            assert bu.matching_uuids(gcs.uuid, gcs_uuid)
            rqdm.fault_throw_scaling(epc,
                                     source_grid = grid,
                                     scaling_factor = None,
                                     connection_set = gcs,
                                     scaling_dict = scaling_dict,
                                     ref_k0 = 0,
                                     ref_k_faces = 'top',
                                     cell_range = 0,
                                     offset_decay = 0.5,
                                     store_displacement = False,
                                     inherit_properties = True,
                                     inherit_realization = None,
                                     inherit_all_realizations = False,
                                     new_grid_title = f'ttt_f6 scaled {i+1}',
                                     new_epc_file = None)
            model = rq.Model(epc)
            grid = model.grid(title = f'ttt_f6 scaled {i+1}')
            assert grid is not None

        # two intersecting straight faults
        a = np.array([[-0.2, 2.0, -0.1], [3.2, 2.0, -0.1]])
        b = np.array([[1.0, -0.2, -0.1], [1.0, 3.2, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f7.dat')
        write_poly(f, a)
        write_poly(f, b, mode = 'a')
        if test_mode == 'file':
            write_poly(f, a)
            write_poly(f, b, mode = 'a')
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f7_1', crs) + make_poly(model, b, 'ttt_f7_2', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f7')

        # re-open and check a few things
        model = rq.Model(epc)
        assert len(model.titles(obj_type = 'IjkGridRepresentation')) == 8
        g1 = model.grid(title = 'ttt_f7')
        assert g1.split_pillars_count == 5
        cpm = g1.create_column_pillar_mapping()
        assert cpm.shape == (3, 3, 2, 2)
        extras = (cpm >= 16)
        assert np.count_nonzero(extras) == 7
        assert np.all(np.sort(np.unique(cpm)) == np.arange(21))


def test_add_faults_and_scaling(tmp_path):

    # create a model with crs
    epc = os.path.join(tmp_path, 'fault_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()

    # create a grid
    dxyz = (100.0, 150.0, 20.0)
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           extent_kji = (3, 5, 5),
                           dxyz = dxyz,
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = True, expand_const_arrays = True)
    model.store_epc()

    # prepare some polylines to define faults
    pla = rql.Polyline(model,
                       set_bool = False,
                       set_crs = crs.uuid,
                       title = 'line_a',
                       set_coord = np.array([(50.0, -10.0, 0.0), (450.0, 760.0, 0.0)]))
    plb = rql.Polyline(model,
                       set_bool = False,
                       set_crs = crs.uuid,
                       title = 'line_b',
                       set_coord = np.array([(-10.0, 530.0, 0.0), (510.0, 330.0, 0.0)]))
    for pl in (pla, plb):
        pl.write_hdf5()
        pl.create_xml()
    model.store_epc()

    # add faults with default throws
    f1a_grid = rqdm.add_faults(epc, source_grid = grid, polylines = [pla, plb], new_grid_title = 'faulted by polylines')
    f1a_grid_uuid = f1a_grid.uuid

    # prepare dictionaries to define more faults and their throws
    pillar_list_dict = {}
    pillar_list_dict['step_fault'] = [(1, 0), (1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (1, 4), (1, 5)]
    pillar_list_dict['north_south_fault'] = [(0, 3), (4, 3)]
    throw_dict = {}
    throw_dict['step_fault'] = (7.0, -7.0)
    throw_dict['north_south_fault'] = (-3.3, 4.8)

    # add faults with explicit throws
    f2a_grid = rqdm.add_faults(epc,
                               source_grid = grid,
                               full_pillar_list_dict = pillar_list_dict,
                               left_right_throw_dict = throw_dict,
                               new_grid_title = 'faulted by dictionaries')
    f2a_grid_uuid = f2a_grid.uuid

    # scale faults globally
    f1b_grid = rqdm.global_fault_throw_scaling(epc,
                                               source_grid = f1a_grid,
                                               scaling_factor = 10.0,
                                               cell_range = 2,
                                               new_grid_title = 'globally scaled faults')
    f1b_grid_uuid = f1b_grid.uuid

    # re-open the model to identify a grid connection set
    model = rq.Model(epc)
    gcs_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation', related_uuid = f2a_grid_uuid)
    assert gcs_uuid is not None
    gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
    assert gcs is not None

    # create a scaling dictionary
    scaling_dict = {'step_fault': 1.5, 'north_south_fault': 3.0}

    # scale faults
    f2b_grid = rqdm.fault_throw_scaling(epc,
                                        source_grid = f2a_grid,
                                        connection_set = gcs,
                                        scaling_dict = scaling_dict,
                                        cell_range = 1,
                                        new_grid_title = 'dictionary scaled faults')
    f2b_grid_uuid = f2b_grid.uuid

    # re-open model and check grids
    model = rq.Model(epc)
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 5

    g = grr.Grid(model, uuid = f1a_grid_uuid)
    gbox = g.xyz_box(lazy = False)
    assert_array_almost_equal(gbox, np.array([(0.0, 0.0, -1.0), (500.0, 750.0, 61.0)]))

    g = grr.Grid(model, uuid = f2a_grid_uuid)
    gbox = g.xyz_box(lazy = False)
    # assert_array_almost_equal(gbox, np.array([(0.0, 0.0, -10.3), (500.0, 750.0, 71.8)]))
    assert_array_almost_equal(gbox, np.array([(0.0, 0.0, -7.0), (500.0, 750.0, 67.0)]))

    g = grr.Grid(model, uuid = f1b_grid_uuid)
    gbox = g.xyz_box(lazy = False)
    # assert_array_almost_equal(gbox, np.array([(0.0, 0.0, -5.0), (500.0, 750.0, 65.0)]))
    assert_array_almost_equal(gbox, np.array([(0.0, 0.0, -7.25), (500.0, 750.0, 67.75)]))

    g = grr.Grid(model, uuid = f2b_grid_uuid)
    gbox = g.xyz_box(lazy = False)
    assert_array_almost_equal(gbox, np.array([(0.0, 0.0, -10.5), (500.0, 750.0, 70.5)]))


def test_drape_to_surface(tmp_path):

    # create a model and a regular grid
    epc = os.path.join(tmp_path, 'drape_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    dxyz = (180.0, -250.0, 25.0)
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           extent_kji = (2, 5, 6),
                           dxyz = dxyz,
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False, expand_const_arrays = True)
    model.store_epc()

    # make a wavey surface covering the area of the grid with a boundary buffer
    model = rq.Model(epc)
    source_grid = model.grid()
    xyz_box = source_grid.xyz_box(lazy = False, local = True)
    surf = rqs.Surface(model)
    nx = ny = 100
    mesh_xyz = np.empty((100, 100, 3))
    xy_range = np.max(xyz_box[1, 0:2] - xyz_box[0, 0:2]) * 1.2
    xy_centre = np.mean(xyz_box[:, 0:2], axis = 0)
    origin_xy = xy_centre - xy_range * 0.5
    dx = xy_range / nx
    dy = xy_range / ny
    points_z = source_grid.points_ref()[..., 2]
    thickness = np.nanmean(points_z[-1, ...] - points_z[0, ...])
    x_wavelength = xy_range / 20.0
    x_half_amplitude = thickness / 10.0
    y_wavelength = xy_range / 10.0
    y_half_amplitude = thickness / 4.0
    mesh_xyz[:, :, 2] = xyz_box[0, 2]  # set z initially to flat plane at depth of shallowest point of grid
    for i in range(nx):
        x = dx * i
        mesh_xyz[i, :, 0] = origin_xy[0] + dx * i  # x
        mesh_xyz[i, :, 2] += x_half_amplitude * maths.sin(
            x * 2.0 * maths.pi / x_wavelength)  # add a depth wave in x direction
    for j in range(ny):
        y = dy * j
        mesh_xyz[:, j, 1] = origin_xy[1] + dy * j  # y
        mesh_xyz[:, j, 2] += y_half_amplitude * maths.sin(
            y * 2.0 * maths.pi / y_wavelength)  # add a depth wave in y direction
    surf.set_from_irregular_mesh(mesh_xyz)

    # drape the grid to the surface
    rqdm.drape_to_surface(epc,
                          source_grid,
                          surf,
                          ref_k0 = 0,
                          ref_k_faces = 'top',
                          store_displacement = True,
                          new_grid_title = 'draped')

    # reopen the model
    model = rq.Model(epc)
    draped = model.grid(title = 'draped')
    assert draped is not None


def test_zonal_grid(tmp_path):

    # create a model and a regular grid
    epc = os.path.join(tmp_path, 'zonal_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    dxyz = (200.0, -270.0, 12.0)
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           extent_kji = (9, 2, 3),
                           dxyz = dxyz,
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()

    # create a zonal version of the grid
    zone_ranges = [(0, 1, 0), (2, 4, 1), (5, 8, 2), (9, 9, 3)]
    rqdm.zonal_grid(epc, zone_layer_range_list = zone_ranges, new_grid_title = 'four zone grid')

    # create a single layer version of the grid
    rqdm.single_layer_grid(epc, source_grid = grid, k0_min = 1, k0_max = 7, new_grid_title = 'single layer grid')

    # re-open the model and take a look at the zonal grids
    model = rq.Model(epc)
    z_grid = model.grid(title = 'four zone grid')
    assert z_grid.nk == 4
    o_grid = model.grid(title = 'single layer grid')
    assert o_grid.nk == 1
    # check z range of single layer grid
    o_box = o_grid.xyz_box()
    assert maths.isclose(o_box[1, 2] - o_box[0, 2], 7.0 * 12.0)


def test_unsplit_grid(tmp_path):

    # create a model and a regular grid
    epc = os.path.join(tmp_path, 'unsplit_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    dxyz = (100.0, 150.0, 30.0)
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           extent_kji = (2, 3, 5),
                           dxyz = dxyz,
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()

    # add a fault
    # prepare dictionaries to define more faults and their throws
    pillar_list_dict = {}
    pillar_list_dict['fault'] = [(0, 2), (1, 2), (1, 3), (2, 3)]
    throw_dict = {}
    throw_dict['fault'] = (7.0, -7.0)
    f2a_grid = rqdm.add_faults(epc,
                               source_grid = grid,
                               full_pillar_list_dict = pillar_list_dict,
                               left_right_throw_dict = throw_dict,
                               new_grid_title = 'faulted by dictionaries')
    f2a_grid_uuid = f2a_grid.uuid

    # heal faults
    healed_grid = rqdm.unsplit_grid(epc, source_grid = f2a_grid, new_grid_title = 'healed grid')
    assert healed_grid is not None
    assert not healed_grid.has_split_coordinate_lines


def test_tilted_grid(tmp_path):

    # create a model and a regular grid
    epc = os.path.join(tmp_path, 'tilted_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    dxyz = (50.0, 50.0, 10.0)
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           extent_kji = (2, 2, 2),
                           origin = (200.0, 200.0, 1000.0),
                           dxyz = dxyz,
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()

    # tilt by 45 degrees
    t_grid = rqdm.tilted_grid(epc,
                              pivot_xyz = (250.0, 250.0, 1000.0),
                              azimuth = 90.0,
                              dip = 45.0,
                              new_grid_title = 'tilted')
    assert t_grid is not None

    # check xyz box of tilted grid
    root_two = maths.sqrt(2.0)
    expected_box = np.array([(250.0 - 70.0 / root_two, 200.0, 1000.0 - 50.0 / root_two),
                             (250.0 + 50.0 / root_two, 300.0, 1000.0 + 70.0 / root_two)])
    assert_array_almost_equal(t_grid.xyz_box(lazy = False), expected_box)


def test_local_depth_adjustment(tmp_path):

    # create a model and a regular grid
    epc = os.path.join(tmp_path, 'depth_adjust_test.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    dxyz = (50.0, 50.0, 10.0)
    grid = grr.RegularGrid(model,
                           crs_uuid = model.crs_uuid,
                           extent_kji = (2, 20, 20),
                           origin = (500.0, 700.0, 2000.0),
                           dxyz = dxyz,
                           as_irregular_grid = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True, add_cell_length_properties = False)
    model.store_epc()

    a_grid = rqdm.local_depth_adjustment(epc, grid, 700.0, 1000.0, 170.0, -7.0, False)
    assert a_grid is not None
    p = a_grid.points_ref()
    assert maths.isclose(np.min(p[..., 2]), 2000.0 - 7.0)


def test_gather_ensemble(tmp_path):
    # create three models, each with a regular grid and a grid property
    epc_list = []
    for m in range(3):
        epc = os.path.join(tmp_path, f'grid_{m}.epc')
        model = rq.new_model(epc)
        crs = rqc.Crs(model)
        crs.create_xml()
        dxyz = (50.0, 50.0, 10.0)
        extent_kji = (3, 20, 20)
        grid = grr.RegularGrid(model,
                               crs_uuid = model.crs_uuid,
                               extent_kji = extent_kji,
                               origin = (500.0, 700.0, 1000.0),
                               dxyz = dxyz,
                               as_irregular_grid = True)
        grid.write_hdf5()
        grid.create_xml(write_geometry = True, add_cell_length_properties = False)
        model.store_epc()
        ntg = np.random.random(extent_kji)
        rqdm.add_one_grid_property_array(epc,
                                         ntg,
                                         property_kind = 'net to gross ratio',
                                         grid_uuid = grid.uuid,
                                         title = 'NETGRS',
                                         uom = 'm3/m3',
                                         indexable_element = 'cells')
        epc_list.append(epc)
    # gather ensemble
    combined_epc = os.path.join(tmp_path, 'combo.epc')
    rqdm.gather_ensemble(epc_list, combined_epc)
    # open combined model and check realisations
    model = rq.Model(combined_epc)
    grid = model.grid()
    assert grid is not None
    pc = grid.property_collection
    assert pc.has_multiple_realizations()
    assert pc.realization_list(sort_list = True) == [0, 1, 2]
    ntg_pc = rqp.selective_version_of_collection(pc, property_kind = 'net to gross ratio')
    assert ntg_pc.number_of_parts() == 3
    ntg3 = ntg_pc.realizations_array_ref()
    assert ntg3.shape == (3, 3, 20, 20)
    assert np.all(ntg3 >= 0.0) and np.all(ntg3 <= 1.0)
