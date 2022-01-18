import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.grid as grr
import resqpy.model as rq


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


def test_regular_grid_with_geometry(tmp_path):

    epc = os.path.join(tmp_path, 'concrete.epc')

    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (55.0, 65.0, 27.0)
    grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'concrete', origin = (0.0, 0.0, 1000.0), dxyz = dxyz)
    grid.create_xml(add_cell_length_properties = True)
    grid_uuid = grid.uuid

    # store with constant arrays (no hdf5 data)
    model.store_epc()

    # check that the grid can be read
    model = rq.Model(epc)
    grid = grr.any_grid(model, uuid = grid_uuid)

    # check that the cell size has been preserved
    expected_dxyz_dkji = np.zeros((3, 3))
    for i in range(3):
        expected_dxyz_dkji[2 - i, i] = dxyz[i]
    assert_array_almost_equal(expected_dxyz_dkji, grid.block_dxyz_dkji)
