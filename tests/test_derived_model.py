import os
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.derived_model as rqdm


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

   # check that property looks okay
   model = rq.Model(epc)
   z_prop = rqp.Property(model, uuid = z_uuid)
   assert z_prop is not None
   assert not z_prop.is_continuous()
   assert not z_prop.is_points()
   assert z_prop.indexable_element() == 'layers'
   lpk_uuid = z_prop.local_property_kind_uuid()
   assert lpk_uuid is not None
   lpk = rqp.PropertyKind(model, uuid = lpk_uuid)
   assert lpk.title == 'zone'
