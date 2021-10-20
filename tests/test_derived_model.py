import pytest
import os
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.well as rqw
import resqpy.derived_model as rqdm
import resqpy.olio.uuid as bu
import resqpy.olio.box_utilities as bx


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
      traj = rqw.Trajectory(model, uuid = model.uuid(obj_type = 'WellboreTrajectoryRepresentation', title = well_name))
      assert traj is not None
      assert traj.knot_count == 3 + wi
