import pytest
import numpy as np

import resqpy.model as rq
import resqpy.grid as grr


def test_parametric_line_geometry(basic_regular_grid):
    # finish converting the regular grid to irregular form
    old_model = basic_regular_grid.model
    basic_regular_grid.write_hdf5()
    basic_regular_grid.create_xml()
    basic_grid = grr.Grid(old_model, uuid = basic_regular_grid.uuid)
    new_file = f'{old_model.epc_file[:-4]}_pl_basic.epc'
    new_model = rq.new_model(new_file)
    # copy crs
    new_model.copy_uuid_from_other_model(old_model, basic_grid.crs_uuid)
    # hijack grid
    basic_grid.cache_all_geometry_arrays()
    basic_grid.model = new_model
    # write hdf5 data for grid using paramtric lines option
    basic_grid.write_hdf5(use_parametric_lines = True)
    # create xml for grid using parametric lines option
    basic_grid.create_xml(use_parametric_lines = True)
    # store new version of model and reload
    new_model.store_epc()
    reload_model = rq.Model(new_model.epc_file)
    # check reloaded grid
    reload_grid = reload_model.grid()
    reload_grid.cache_all_geometry_arrays()
    # assertions
    assert tuple(reload_grid.extent_kji) == tuple(basic_grid.extent_kji)
    assert reload_grid.points_cached is not None
    assert reload_grid.points_cached.shape == basic_grid.points_cached.shape
    np.testing.assert_array_almost_equal(reload_grid.points_cached, basic_grid.points_cached)


def test_parametric_line_geometry_faulted(faulted_grid):
    # Act
    old_model = faulted_grid.model
    new_file = f'{old_model.epc_file[:-4]}_pl_faulted.epc'
    new_model = rq.new_model(new_file)
    # copy crs
    new_model.copy_uuid_from_other_model(old_model, faulted_grid.crs_uuid)
    # hijack grid
    faulted_grid.cache_all_geometry_arrays()
    faulted_grid.model = new_model
    # write hdf5 data for grid using paramtric lines option
    faulted_grid.write_hdf5(use_parametric_lines = True)
    # create xml for grid using parametric lines option
    faulted_grid.create_xml(use_parametric_lines = True)
    # store new version of model and reload
    new_model.store_epc()
    reload_model = rq.Model(new_model.epc_file)
    # check reloaded grid
    reload_grid = reload_model.grid()
    reload_grid.cache_all_geometry_arrays()
    # assertions
    assert tuple(reload_grid.extent_kji) == tuple(faulted_grid.extent_kji)
    assert reload_grid.points_cached is not None
    assert reload_grid.points_cached.shape == faulted_grid.points_cached.shape
    np.testing.assert_array_almost_equal(reload_grid.points_cached, faulted_grid.points_cached)
