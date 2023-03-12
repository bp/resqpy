import numpy as np
import pytest

import resqpy.model as rq
import resqpy.grid as grr


def add_grid_with_missing_points(model, crs):
    """Create an irregular grid with some points missing."""
    grid = grr.Grid(model, title = 'partial')
    grid.crs = crs
    grid.crs_uuid = crs.uuid
    grid.extent_kji = (2, 2, 2)
    grid.nk, grid.nj, grid.ni = grid.extent_kji
    grid.grid_is_right_handed = True
    grid.k_direction_is_down = True
    grid.pillar_shape = 'straight'
    grid.has_split_coordinate_lines = False
    grid.points_cached = np.array(
        [[[(100.0, 1000.0, 5000.0), (200.0, 1000.0, 5000.0),
           (np.NaN, np.NaN, np.NaN)], [(100.0, 1200.0, 5000.0), (200.0, 1200.0, 5000.0), (300.0, 1200.0, 5000.0)],
          [(100.0, 1500.0, 5000.0), (200.0, 1500.0, 5000.0), (300.0, 1500.0, 5000.0)]],
         [[(110.0, 1005.0, 5020.0), (210.0, 1005.0, 5020.0),
           (310.0, 1005.0, 5020.0)], [(110.0, 1205.0, 5020.0), (210.0, 1205.0, 5020.0), (310.0, 1205.0, 5020.0)],
          [(110.0, 1505.0, 5020.0), (210.0, 1505.0, 5020.0), (310.0, 1505.0, 5020.0)]],
         [[(120.0, 1010.0, 5040.0), (220.0, 1010.0, 5040.0),
           (320.0, 1010.0, 5040.0)], [(120.0, 1210.0, 5040.0), (220.0, 1210.0, 5040.0), (320.0, 1210.0, 5040.0)],
          [(np.NaN, np.NaN, np.NaN), (220.0, 1510.0, 5040.0), (320.0, 1510.0, 5040.0)]]],
        dtype = float)
    assert grid.points_cached.shape == (3, 3, 3, 3)
    grid.array_cell_geometry_is_defined = np.ones((2, 2, 2), dtype = bool)
    grid.array_cell_geometry_is_defined[0, 0, 1] = False
    grid.array_cell_geometry_is_defined[1, 1, 0] = False
    grid.write_hdf5_from_caches()
    grid.create_xml()
    return grid


def test_cell_geometry_is_defined_regular_grid(basic_regular_grid):
    grid = basic_regular_grid
    assert grid.cell_geometry_is_defined(cell_kji0 = (0, 0, 0), cache_array = False)
    assert grid.cell_geometry_is_defined(cell_kji0 = (1, 1, 1), cache_array = True)
    assert hasattr(grid, 'array_cell_geometry_is_defined')
    assert np.all(grid.cell_geometry_is_defined)
    assert grid.geometry_defined_for_all_cells_cached


def test_cell_geometry_is_defined_cell_kji0(example_model_and_crs):
    model, crs = example_model_and_crs
    add_grid_with_missing_points(model, crs)
    model.store_epc()
    model = rq.Model(model.epc_file)
    grid = model.grid()
    grid.cache_all_geometry_arrays()
    assert grid is not None
    assert tuple(grid.extent_kji) == (2, 2, 2)
    assert grid.cell_geometry_is_defined(cell_kji0 = (0, 0, 0), cache_array = False)
    assert not grid.cell_geometry_is_defined(cell_kji0 = (0, 0, 1), cache_array = False)
    assert not grid.cell_geometry_is_defined(cell_kji0 = (1, 1, 0), cache_array = False)
    assert not grid.geometry_defined_for_all_cells_cached
