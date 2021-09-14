import pytest
import os
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.derived_model as rqdm
import resqpy.olio.uuid as bu


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
   grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'In The Zone')
   grid.create_xml()
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
                                              property_kind = 'code',
                                              title = 'messy zone',
                                              grid_uuid = grid_uuid,
                                              null_value = -1)
   assert za_uuid is not None

   # fail to add a zone by layer property based on the messy cells property
   with pytest.raises(Exception):
      v, z_uuid = rqdm.add_zone_by_layer_property(epc_file = epc,
                                                  zone_by_cell_property_uuid = za_uuid,
                                                  use_dominant_zone = False,
                                                  title = 'should fail')

   # add a zone by layer property based on the neat cells property
   v, z_uuid = rqdm.add_zone_by_layer_property(epc_file = epc,
                                               zone_by_cell_property_uuid = za_uuid,
                                               use_dominant_zone = True,
                                               title = 'from messy cells array')
   assert tuple(v) == (1, 2, 3, 5)

   # check that zone property looks okay
   model = rq.Model(epc)
   z_prop = rqp.Property(model, uuid = z_uuid)
   check_zone_prop(z_prop)


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
