from resqpy.multiprocessing.multiprocessing import function_multiprocessing
from resqpy.multiprocessing.wrappers.grid_surface import find_faces_to_represent_surface_regular_wrapper
from resqpy.model import Model, new_model
from resqpy.crs import Crs
import resqpy.grid as grr
import resqpy.surface as rqs
import numpy as np
import resqpy.olio.triangulation as tri
import os
import resqpy.grid_surface as rqgs
import resqpy.fault as rqf


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
    return grid, surface


def test_function_multiprocessing_find_faces_same_grid_and_surface(tmp_path):
    # Arrange
    recombined_epc = f"{tmp_path}/test_recombined.epc"

    epc_file = f"{tmp_path}/test.epc"
    grid, surface = small_grid_and_surface(epc_file)
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid

    func = find_faces_to_represent_surface_regular_wrapper

    kwargs_1 = {
        'grid_epc': epc_file,
        'grid_uuid': grid_uuid,
        'surface_epc': epc_file,
        'surface_uuid': surface_uuid,
        'use_index_as_realisation': False,
        'name': 'first',
        'title': 'first'
    }
    kwargs_2 = {
        'grid_epc': epc_file,
        'grid_uuid': grid_uuid,
        'surface_epc': epc_file,
        'surface_uuid': surface_uuid,
        'use_index_as_realisation': False,
        'name': 'second'
    }
    kwargs_3 = {
        'grid_epc': epc_file,
        'grid_uuid': grid_uuid,
        'surface_epc': epc_file,
        'surface_uuid': surface_uuid,
        'use_index_as_realisation': False,
        'name': 'third'
    }
    kwargs_list = [kwargs_1, kwargs_2, kwargs_3]
    success_list_expected = [True] * 3

    # Act
    success_list = function_multiprocessing(func, kwargs_list, recombined_epc, processes = 3)
    model = Model(recombined_epc)
    gcs_recombined_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation', title = 'first')
    gcs_recombined = rqf.GridConnectionSet(model, uuid = gcs_recombined_uuid)
    gcs_recombined.cache_arrays()
    gcs = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, kwargs_1["name"])

    # Assert
    assert success_list == success_list_expected
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 3
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 3
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 3
    assert len(model.uuids()) == 12

    np.testing.assert_array_almost_equal(gcs.cell_index_pairs, gcs_recombined.cell_index_pairs)
    np.testing.assert_array_almost_equal(gcs.face_index_pairs, gcs_recombined.face_index_pairs)


def test_function_multiprocessing_find_faces_same_grid_and_surface_with_properties(tmp_path):
    # Arrange
    recombined_epc = f"{tmp_path}/test_recombined.epc"

    epc_file = f"{tmp_path}/test.epc"
    grid, surface = small_grid_and_surface(epc_file)
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid

    func = find_faces_to_represent_surface_regular_wrapper

    kwargs_1 = {
        'grid_epc': epc_file,
        'grid_uuid': grid_uuid,
        'surface_epc': epc_file,
        'surface_uuid': surface_uuid,
        'use_index_as_realisation': False,
        'name': 'first',
        'title': 'first',
        'return_properties': ["normal vector", "offset", "triangle"]
    }
    kwargs_2 = {
        'grid_epc': epc_file,
        'grid_uuid': grid_uuid,
        'surface_epc': epc_file,
        'surface_uuid': surface_uuid,
        'use_index_as_realisation': False,
        'name': 'second'
    }
    kwargs_3 = {
        'grid_epc': epc_file,
        'grid_uuid': grid_uuid,
        'surface_epc': epc_file,
        'surface_uuid': surface_uuid,
        'use_index_as_realisation': False,
        'name': 'third'
    }
    kwargs_list = [kwargs_1, kwargs_2, kwargs_3]
    success_list_expected = [True] * 3

    # Act
    success_list = function_multiprocessing(func, kwargs_list, recombined_epc, processes = 3)
    model = Model(recombined_epc)
    gcs_recombined_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation', title = 'first')
    gcs_recombined = rqf.GridConnectionSet(model, uuid = gcs_recombined_uuid)
    gcs_recombined.cache_arrays()
    print(model.parts())
    gcs = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, kwargs_1["name"])

    # Assert
    assert success_list == success_list_expected
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 3
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 3
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 3
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 1
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 1
    assert len(model.uuids(obj_type = 'PointsProperty')) == 1
    assert len(model.uuids()) == 15

    np.testing.assert_array_almost_equal(gcs.cell_index_pairs, gcs_recombined.cell_index_pairs)
    np.testing.assert_array_almost_equal(gcs.face_index_pairs, gcs_recombined.face_index_pairs)


def test_function_multiprocessing_find_faces_different_grid_and_surface(tmp_path):
    # Arrange
    recombined_epc = f"{tmp_path}/test.epc"

    func = find_faces_to_represent_surface_regular_wrapper

    epc_file1 = f"{tmp_path}/test1.epc"
    grid1, surface1 = small_grid_and_surface(epc_file1)
    kwargs_1 = {
        'grid_epc': epc_file1,
        'grid_uuid': grid1.uuid,
        'surface_epc': epc_file1,
        'surface_uuid': surface1.uuid,
        'use_index_as_realisation': False,
        'name': 'first'
    }
    epc_file2 = f"{tmp_path}/test2.epc"
    grid2, surface2 = small_grid_and_surface(epc_file2)
    kwargs_2 = {
        'grid_epc': epc_file2,
        'grid_uuid': grid2.uuid,
        'surface_epc': epc_file2,
        'surface_uuid': surface2.uuid,
        'use_index_as_realisation': False,
        'name': 'second'
    }
    epc_file3 = f"{tmp_path}/test3.epc"
    grid3, surface3 = small_grid_and_surface(epc_file3)
    kwargs_3 = {
        'grid_epc': epc_file3,
        'grid_uuid': grid3.uuid,
        'surface_epc': epc_file3,
        'surface_uuid': surface3.uuid,
        'use_index_as_realisation': False,
        'name': 'third'
    }

    kwargs_list = [kwargs_1, kwargs_2, kwargs_3]
    success_list_expected = [True] * 3

    # Act
    success_list = function_multiprocessing(func, kwargs_list, recombined_epc, processes = 3)
    model = Model(recombined_epc)
    gcs_recombined_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation', title = 'first')
    gcs_recombined = rqf.GridConnectionSet(model, uuid = gcs_recombined_uuid)
    gcs_recombined.cache_arrays()
    gcs = rqgs.find_faces_to_represent_surface_regular_optimised(grid1, surface1, kwargs_1["name"])

    # Assert
    assert success_list == success_list_expected
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 3
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 3
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 3
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 3
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 3
    print(model.parts())
    assert len(model.uuids()) == 16

    np.testing.assert_array_almost_equal(gcs.cell_index_pairs, gcs_recombined.cell_index_pairs)
    np.testing.assert_array_almost_equal(gcs.face_index_pairs, gcs_recombined.face_index_pairs)
