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


@pytest.mark.parametrize("extent, expected_cell_count", [((2, 2, 2), 8), ((1, 1, 1), 1), ((4, 2, 3), 24)])
def test_cell_count_default(basic_regular_grid, extent, expected_cell_count):
    # Arrange
    basic_regular_grid.extent_kji = extent

    # Act
    cell_count = basic_regular_grid.cell_count()

    # Assert
    assert cell_count == expected_cell_count


def test_cell_count_default_faulted_grid(faulted_grid):
    # Act
    cell_count = faulted_grid.cell_count()

    # Assert
    assert cell_count == 120


def test_cell_count_default_s_bend_grid(s_bend_grid):
    # Act
    cell_count = s_bend_grid.cell_count()

    # Assert
    assert cell_count == 3000


def test_cell_count_non_pinched_out_only(basic_regular_grid):
    # Arrange
    basic_regular_grid.extent_kji = (2, 2, 2)
    pinchout = np.random.choice([True, False], (2, 2, 2))
    basic_regular_grid.pinchout = pinchout

    # Act
    cell_count = basic_regular_grid.cell_count(non_pinched_out_only = True)

    # Assert
    assert cell_count == pinchout.size - pinchout.sum()


def test_cell_count_non_pinched_out_only_faulted_grid(faulted_grid):
    # Act
    cell_count = faulted_grid.cell_count(non_pinched_out_only = True)

    # Assert
    assert cell_count == 110


def test_cell_count_non_pinched_out_only_s_bend_grid(s_bend_grid):
    # Act
    cell_count = s_bend_grid.cell_count(non_pinched_out_only = True)

    # Assert
    assert cell_count == 2400


def test_cell_count_active_only(basic_regular_grid):
    # Arrange
    basic_regular_grid.extent_kji = (2, 2, 2)
    inactive = np.random.choice([True, False], (2, 2, 2))
    basic_regular_grid.inactive = inactive

    # Act
    cell_count = basic_regular_grid.cell_count(active_only = True)

    # Assert
    assert cell_count == inactive.size - inactive.sum()


def test_cell_count_active_only_faulted_grid(faulted_grid):
    # Act
    cell_count = faulted_grid.cell_count(active_only = True)

    # Assert
    assert cell_count == 103


def test_cell_count_active_only_s_bend_grid(s_bend_grid):
    # Act
    cell_count = s_bend_grid.cell_count(active_only = True)

    # Assert
    assert cell_count == 3000


def test_cell_count_geometry_defined_only(basic_regular_grid):
    # Arrange
    basic_regular_grid.geometry_defined_for_all_cells_cached = False
    geometry_defined = np.random.choice([True, False], (2, 2, 2))
    basic_regular_grid.array_cell_geometry_is_defined = geometry_defined

    # Act
    cell_count = basic_regular_grid.cell_count(geometry_defined_only = True)

    # Assert
    assert cell_count == geometry_defined.sum()


def test_cell_count_geometry_defined_only_faulted_grid(faulted_grid):
    # Act
    cell_count = faulted_grid.cell_count(geometry_defined_only = True)

    # Assert
    assert cell_count == 120


def test_cell_count_geometry_defined_only_s_bend_grid(s_bend_grid):
    # Act
    cell_count = s_bend_grid.cell_count(geometry_defined_only = True)

    # Assert
    assert cell_count == 3000


def test_actual_pillar_shape(basic_regular_grid):
    # Act
    pillar_shape = basic_regular_grid.actual_pillar_shape()

    # Assert
    assert pillar_shape == 'vertical'


def test_actual_pillar_shape_s_bend_grid(s_bend_grid):
    # Act
    pillar_shape = s_bend_grid.actual_pillar_shape()

    # Assert
    assert pillar_shape == 'straight'


def test_is_big(tmp_path):
    epc = os.path.join(tmp_path, 'monster.epc')

    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (155.0, 165.0, 127.0)
    grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'tiny', origin = (0.0, 0.0, 1000.0), dxyz = dxyz)
    assert not grid.is_big()
    grid = grr.RegularGrid(model,
                           extent_kji = (12345, 23456, 10),
                           title = 'huge',
                           origin = (0.0, 0.0, 1000.0),
                           dxyz = dxyz)
    assert grid.is_big()
