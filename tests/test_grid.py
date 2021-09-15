import pytest
import os
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.grid as grr


def test_regular_grid_no_geometry(tmp_path):
   # issue #222

   epc = os.path.join(tmp_path, 'abstract.epc')

   model = rq.new_model(epc)

   # create a basic block grid
   grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'spaced out')
   grid.create_xml(add_cell_length_properties = False)
   grid_uuid = grid.uuid

   model.store_epc()

   # check that the grid can be read
   model = rq.Model(epc)

   grid = grr.any_grid(model, uuid = grid_uuid)
