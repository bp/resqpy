from resqpy.multiprocessing.multiprocessing import function_multiprocessing
from resqpy.multiprocessing.wrappers.grid_surface import find_faces_to_represent_surface_regular_wrapper
from resqpy.model import Model, new_model
from resqpy.crs import Crs
import resqpy.grid as grr
import resqpy.surface as rqs
import numpy as np
import resqpy.olio.triangulation as tri
import pytest


def small_grid_and_surface(epc_file):
    """Creates a small RegularGrid and a random triangular surface."""
    model = new_model(epc_file)

    crs = Crs(model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    n_points = 100
    points = np.random.rand(n_points, 3) * extent
    triangles = tri.dt(points)
    surface = rqs.Surface(model, crs_uuid = crs_uuid, title = "small_surface")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    model.store_epc()


@pytest.mark.skip("Incomplete")
def test_function_multiprocessing_find_faces_same_grid_and_surface(tmp_path):
    # Arrange
    recombined_epc = f"{tmp_path}/test_recombined.epc"

    epc_file = f"{tmp_path}/test.epc"
    small_grid_and_surface(epc_file)

    func = find_faces_to_represent_surface_regular_wrapper

    kwargs_1 = {'grid_epc': epc_file, 'surface_epc': epc_file, 'use_index_as_realisation': False, 'name': 'first'}
    kwargs_2 = {'grid_epc': epc_file, 'surface_epc': epc_file, 'use_index_as_realisation': False, 'name': 'second'}
    kwargs_3 = {'grid_epc': epc_file, 'surface_epc': epc_file, 'use_index_as_realisation': False, 'name': 'third'}

    kwargs_list = [kwargs_1, kwargs_2, kwargs_3]
    success_list_expected = [True] * 3

    # Act
    success_list = function_multiprocessing(func, kwargs_list, recombined_epc, processes = 3)
    model = Model(recombined_epc)

    # Assert
    assert success_list == success_list_expected
    assert len(model.uuids) == 18


@pytest.mark.skip("Incomplete")
def test_function_multiprocessing_find_faces_different_grid_and_surface(tmp_path, tmp_model):
    # Arrange
    recombined_epc = f"{tmp_path}/test.epc"

    func = find_faces_to_represent_surface_regular_wrapper

    grid1, surface1 = small_grid_and_surface(tmp_model)
    kwargs_1 = {'grid': grid1, 'surface': surface1, 'name': 'first'}

    grid2, surface2 = small_grid_and_surface(tmp_model)
    kwargs_2 = {'grid': grid2, 'surface': surface2, 'name': 'second'}

    grid3, surface3 = small_grid_and_surface(tmp_model)
    kwargs_3 = {'grid': grid3, 'surface': surface3, 'name': 'third'}

    kwargs_list = [kwargs_1, kwargs_2, kwargs_3]
    success_list_expected = [True] * 3

    # Act
    success_list = function_multiprocessing(func, kwargs_list, recombined_epc, processes = 3)
    model = Model(recombined_epc)

    # Assert
    assert success_list == success_list_expected
    assert len(model.uuids) == 18